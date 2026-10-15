"""Compare two engines on identical photons.

    uv run tissuemc-cross-engine --b-engine cuda

Both engines give photon i the same random stream and split photons into the same batches, so a
correct implementation should agree far more closely than statistical noise allows. This is a much
sharper test than comparing with published values.

The limits below were fixed before any GPU number existed. They come from the one comparison that
could be made then: this engine's float32 and float64 transport on the same photons, which agreed
to about 0.01 standard errors on totals, 5e-4 per batch, and 2e-5 on event counts (V3, 1e6
photons). The limits are 25 to 100 times looser than that because the GPU's math library and
fused multiply-add contraction round differently from the CPU's. If a GPU run fails them, the
cause is to be found, not the limit loosened.

The absorbed-weight limits were added after R1, whose cross-check had not covered the one tally
that rung changed. They use the same limits as R_d and T_t.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from . import stats, validate

# Largest accepted gap, in standard errors of the total.
MAX_TOTAL_SE = 0.25
# Largest accepted relative gap in any single batch's sum.
MAX_BATCH_REL = 5e-3
# Largest accepted relative gap in the total number of absorption/scattering events.
MAX_EVENTS_REL = 1e-3


@dataclass(frozen=True)
class Gap:
    name: str
    value: float
    limit: float

    @property
    def passed(self) -> bool:
        return self.value <= self.limit

    def line(self) -> str:
        flag = "PASS" if self.passed else "FAIL"
        return f"  [{flag}] {self.name:<34} {self.value:.3e}  (limit {self.limit:.1e})"


def _sums(run: dict, key: str) -> list[float]:
    if key == "absorbed":
        return [math.fsum(b["absorbed"]) for b in run["batches"]]
    return [b[key] for b in run["batches"]]


def compare_runs(a: dict, b: dict) -> list[Gap]:
    """Gaps between two result dicts that simulated the same photons in the same batches."""
    if [x["photons"] for x in a["batches"]] != [x["photons"] for x in b["batches"]]:
        raise ValueError("the two runs do not use the same batches")
    gaps: list[Gap] = []
    photons = [x["photons"] for x in a["batches"]]
    # R_d and T_t are added once per photon; absorbed weight is the tally a rung may change.
    for key in ("rd", "tt", "absorbed"):
        sa, sb = _sums(a, key), _sums(b, key)
        ea = stats.batch_estimate(sa, photons)
        eb = stats.batch_estimate(sb, photons)
        se = max(ea.se, eb.se)
        gaps.append(
            Gap(f"{key}: total gap / standard error", abs(ea.mean - eb.mean) / se, MAX_TOTAL_SE)
        )
        worst = max(abs(x - y) / abs(x) for x, y in zip(sa, sb, strict=True) if x != 0)
        gaps.append(Gap(f"{key}: worst per-batch relative gap", worst, MAX_BATCH_REL))
    ev_a, ev_b = a["events"], b["events"]
    gaps.append(Gap("events: relative gap", abs(ev_a - ev_b) / ev_a, MAX_EVENTS_REL))
    gaps.append(Gap("anomalies (both runs)", float(a["anomalies"] + b["anomalies"]), 0.0))
    return gaps


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tissuemc-cross-engine", description=__doc__.split("\n\n")[0])
    ap.add_argument("--case", default="v3_three_layer", help="validation input (default: V3)")
    ap.add_argument("--photons", type=int, default=2_000_000)
    ap.add_argument("--batches", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--binary", type=Path, help="engine binary (default: build one)")
    ap.add_argument("--a-engine", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--a-precision", default="float32", choices=["float64", "float32"])
    ap.add_argument("--b-engine", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--b-precision", default="float32", choices=["float64", "float32"])
    ap.add_argument("--b-rung", type=int, default=0, help="GPU rung for engine B")
    ap.add_argument("--report", type=Path, help="write the gaps to this JSON file")
    args = ap.parse_args(argv)

    root = validate.find_root()
    binary = args.binary or validate.ensure_built(root)
    config = root / "configs" / "validation" / f"{args.case}.mci"
    if not config.exists():
        raise SystemExit(f"no such validation input: {config}")

    def run(engine: str, precision: str, rung: int = 0) -> dict:
        return validate.run_engine(
            binary,
            config,
            args.photons,
            args.batches,
            args.seed,
            precision,
            engine=engine,
            rung=rung,
        )

    a = run(args.a_engine, args.a_precision)
    b = run(args.b_engine, args.b_precision, args.b_rung)
    print(f"{args.case}: {args.photons} photons, {args.batches} batches, seed {args.seed}")
    print(f"  A = {args.a_engine} {args.a_precision}    B = {args.b_engine} {args.b_precision}")
    if b.get("device"):
        print(f"  device: {b['device']}")
    gaps = compare_runs(a, b)
    for g in gaps:
        print(g.line())
    ok = all(g.passed for g in gaps)
    print("\nengines agree on identical photons" if ok else "\nENGINES DISAGREE: investigate")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "case": args.case,
                    "photons": args.photons,
                    "batches": args.batches,
                    "seed": args.seed,
                    "a": {"engine": args.a_engine, "precision": args.a_precision},
                    "b": {
                        "engine": args.b_engine,
                        "precision": args.b_precision,
                        "device": b.get("device", ""),
                        "rung": args.b_rung,
                    },
                    "gaps": [
                        {"name": g.name, "value": g.value, "limit": g.limit, "passed": g.passed}
                        for g in gaps
                    ],
                    "agree": ok,
                },
                indent=1,
            )
            + "\n"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
