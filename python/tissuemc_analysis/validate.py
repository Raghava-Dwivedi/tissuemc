"""Run the validation cases and compare the engine with published values and the oracle.

    uv run tissuemc-validate

Builds the engine if needed, runs each case in ``configs/validation``, and prints one line per
check. Exits non-zero if any check fails. Reference values and their sources are in
``configs/validation/references.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import oracle, stats


def find_root() -> Path:
    """The repository root: the nearest parent directory holding CMakeLists.txt."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "CMakeLists.txt").exists():
            return parent
    raise SystemExit("tissuemc-validate: cannot find the repository root (CMakeLists.txt)")


def ensure_built(root: Path) -> Path:
    """Configure and build the engine with the CMake on PATH (an incremental no-op when current)."""
    build = root / "build"
    subprocess.run(
        ["cmake", "-S", str(root), "-B", str(build), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(["cmake", "--build", str(build)], check=True, stdout=subprocess.DEVNULL)
    return build / "tissuemc"


@dataclass(frozen=True)
class CheckResult:
    case: str
    seed: int
    quantity: str
    source: str
    comparison: stats.Comparison

    def line(self) -> str:
        c = self.comparison
        flag = "PASS" if c.passed else "FAIL"
        return (
            f"  [{flag}] {self.quantity:<17} ours {c.ours:.6f} +/- {c.se_ours:.6f}   "
            f"ref {c.reference:.6f}   z = {c.z:+.2f}   {self.source}"
        )


def run_engine(binary: Path, config: Path, photons: int, batches: int, seed: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "result.json"
        cmd = [
            str(binary), "run", str(config),
            "--photons", str(photons), "--batches", str(batches),
            "--seed", str(seed), "--output", str(out),
        ]  # fmt: skip
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"engine failed ({proc.returncode}): {proc.stderr.strip()}")
        return json.loads(out.read_text())["runs"][0]


def estimates(run: dict) -> dict[str, stats.Estimate]:
    """Recompute the estimates from the raw batch sums, independent of the engine's own summary."""
    batches = run["batches"]
    photons = [b["photons"] for b in batches]
    rd = stats.batch_estimate([b["rd"] for b in batches], photons)
    tt = stats.batch_estimate([b["tt"] for b in batches], photons)
    # The specular reflectance is a constant, so it shifts the mean and leaves the error alone.
    plus = stats.Estimate(rd.mean + run["specular"], rd.se)
    return {"rd": rd, "tt": tt, "rd_plus_specular": plus}


def oracle_value(run: dict, quantity: str) -> float:
    layer = run["layers"][0]
    if len(run["layers"]) != 1 or layer["g"] != 0:
        raise SystemExit("the semi-infinite oracle needs one isotropic layer")
    res = oracle.semi_infinite_isotropic(layer["mua"], layer["mus"], layer["n"], run["n_above"])
    return {"rd": res.rd, "rd_plus_specular": res.total}[quantity]


def evaluate_case(name: str, case: dict, run: dict, seed: int) -> list[CheckResult]:
    est = estimates(run)
    photons = run["photons"]
    results: list[CheckResult] = []
    for chk in case["checks"]:
        q = chk["quantity"]
        ours = est[q]
        if "oracle" in chk:
            ref, se_ref, rounding = oracle_value(run, q), 0.0, 0.0
        else:
            ref = chk["value"]
            rounding = 0.5 * 10 ** (-chk["decimals"]) if "decimals" in chk else 0.0
            if "se" in chk:
                se_ref = chk["se"]
            elif "n_photons" in chk:
                se_ref = stats.estimated_reference_se(ours.se, photons, chk["n_photons"])
            else:
                se_ref = 0.0
        cmp = stats.compare(ours.mean, ours.se, ref, se_ref, rounding)
        results.append(CheckResult(name, seed, q, chk["source"], cmp))
    return results


def invariant_results(name: str, run: dict, seed: int) -> list[CheckResult]:
    """Checks that must hold for every run: unit directions, finite weights, energy conservation."""
    out: list[CheckResult] = []
    # Conservation is exact in expectation (roulette is unbiased): the residual must be within
    # a few standard errors of zero. The tiny floor covers runs with no roulette at all.
    batches = run["batches"]
    photons = [b["photons"] for b in batches]
    per_batch = [b["rd"] + b["tt"] + math.fsum(b["absorbed"]) for b in batches]
    total = stats.batch_estimate(per_batch, photons)
    residual = total.mean + run["specular"] - 1.0
    c = stats.compare(residual, total.se, 0.0, 0.0, rounding=1e-12)
    out.append(CheckResult(name, seed, "conservation", "rd + tt + absorbed + rsp = 1", c))
    anomalies = float(run["anomalies"])
    c = stats.compare(anomalies, 0.0, 0.0)
    out.append(CheckResult(name, seed, "anomalies", "non-finite weights or non-unit directions", c))
    return out


def summary_matches_engine(run: dict) -> bool:
    """The engine's own summary and our recomputation of it must agree."""
    est = estimates(run)
    for key in ("rd", "tt"):
        theirs = run["summary"][key]
        if not math.isclose(theirs["mean"], est[key].mean, rel_tol=1e-12):
            return False
        if not math.isclose(theirs["se"], est[key].se, rel_tol=1e-9):
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tissuemc-validate", description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", help="comma-separated case names (default: all)")
    ap.add_argument("--photons", type=int, help="photons per case (default: per-case setting)")
    ap.add_argument("--batches", type=int, default=50, help="batches for the standard error")
    ap.add_argument("--seeds", default="1", help="comma-separated seeds (default: 1)")
    ap.add_argument("--binary", type=Path, help="use this engine binary instead of building one")
    ap.add_argument("--report", type=Path, help="write all comparisons to this JSON file")
    args = ap.parse_args(argv)

    root = find_root()
    binary = args.binary or ensure_built(root)
    refs = json.loads((root / "configs" / "validation" / "references.json").read_text())
    names = args.cases.split(",") if args.cases else list(refs["cases"])
    unknown = [n for n in names if n not in refs["cases"]]
    if unknown:
        raise SystemExit(
            f"unknown case(s): {', '.join(unknown)}; known: {', '.join(refs['cases'])}"
        )
    seeds = [int(s) for s in args.seeds.split(",")]

    all_results: list[CheckResult] = []
    engine_agrees = True
    for name in names:
        case = refs["cases"][name]
        print(f"{case['title']}")
        if "note" in case:
            print(f"  note: {case['note']}")
        for seed in seeds:
            run = run_engine(
                binary,
                root / "configs" / case["config"],
                args.photons or case["photons"],
                args.batches,
                seed,
            )
            if not summary_matches_engine(run):
                engine_agrees = False
                print(f"  [FAIL] seed {seed}: engine summary disagrees with Python recomputation")
            results = evaluate_case(name, case, run, seed) + invariant_results(name, run, seed)
            if len(seeds) > 1:
                print(f"  seed {seed}:")
            for r in results:
                print(r.line())
            all_results.extend(results)

    failed = [r for r in all_results if not r.comparison.passed]
    ok = not failed and engine_agrees
    print(
        f"\n{len(all_results) - len(failed)}/{len(all_results)} checks passed"
        f"{'' if engine_agrees else '; engine summary mismatch'}"
    )
    if args.report:
        args.report.write_text(
            json.dumps(
                [
                    {
                        "case": r.case,
                        "seed": r.seed,
                        "quantity": r.quantity,
                        "source": r.source,
                        **{k: getattr(r.comparison, k) for k in r.comparison.__dataclass_fields__},
                    }
                    for r in all_results
                ],
                indent=1,
            )
            + "\n"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
