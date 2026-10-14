"""Cross-check the engine against the original MCML program.

    uv run tissuemc-compare-mcml

The original program (L. Wang and S. L. Jacques, 1992-93, University of Texas M. D. Anderson Cancer
Center) is not distributed with this repository: its page states no license. Download the five
source files yourself from https://omlc.org/software/mc/mcml/mcml-src/ into ``third_party/mcml/``.
They are checked against the SHA-256 values below, so the comparison always says which files it
used. The checksums were recorded from the copy first compared on 2026-10-14; they guard against
accidental change, not against a tampered first download.

Each case is run through the original program as many independent runs inside one launch (they
continue one random stream), and through this engine with the same photon counts. The two
estimates are compared with a z-score using both standard errors. The pass rule, fixed before the
first comparison was made, is |z| < 3 for every quantity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from . import stats, validate

SOURCE_FILES = ["mcml.h", "mcmlio.c", "mcmlnr.c", "mcmlgo.c", "mcmlmain.c"]
PINNED_SHA256 = {
    "mcml.h": "574a94d97759d4fd968a5a6c72ab1f329b307720bec61411055accc1133d610c",
    "mcmlio.c": "b83e005ca980d08b42316f2e32647450cade496e2a0d758d203fbbb1f896b09c",
    "mcmlnr.c": "7e470165711ad7a8f7d1327b8d4da9f2bf2dd9dbb7121f6f5e439f47859e385f",
    "mcmlgo.c": "2135c419e00cf0d7be9f709b5884b43e2a7b88b0bc540c8747fe2996bd3a5e6e",
    "mcmlmain.c": "1a9b6b7c23bd472ec5cd917f050c827598cfa2abeb985a6ca3bddbaa018e5fa3",
}

# The 1993 source relies on implicit function declarations (C89), which current compilers reject
# by default. The flags change how it is compiled, not the source.
COMPILE_FLAGS = ["-std=gnu89", "-O2", "-Wno-implicit-function-declaration", "-Wno-error"]

DEFAULT_CASES = ["v1_slab", "v2_semi_infinite", "o1_semi_infinite_matched", "v3_three_layer"]


# --------------------------------------------------------------------------------------------
# Source handling
# --------------------------------------------------------------------------------------------


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(src: Path) -> None:
    """Refuse to continue unless all five files are present and match the pinned checksums."""
    missing = [n for n in SOURCE_FILES if not (src / n).is_file()]
    if missing:
        raise SystemExit(
            f"missing in {src}: {', '.join(missing)}\n"
            "Download the five MCML source files from "
            "https://omlc.org/software/mc/mcml/mcml-src/ into that directory."
        )
    for name in SOURCE_FILES:
        if sha256(src / name) != PINNED_SHA256[name]:
            raise SystemExit(f"{src / name}: SHA-256 does not match the pinned value; not using it")


def build_mcml(src: Path, build_dir: Path, name: str = "mcml") -> Path:
    """Compile the program in ``src`` into ``build_dir`` (kept outside the source directory)."""
    build_dir.mkdir(parents=True, exist_ok=True)
    binary = build_dir / name
    sources = [str(src / n) for n in SOURCE_FILES if n.endswith(".c")]
    proc = subprocess.run(
        ["cc", *COMPILE_FLAGS, "-o", str(binary), *sources, "-lm"], capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise SystemExit(f"could not compile the original MCML:\n{proc.stderr}")
    return binary


XOSHIRO_RANDOM_NUM = r"""/* Replacement for the original RandomNum(): xoshiro256++,
 * splitmix64-seeded, 53 bits, strictly inside (0, 1).
 * The rest of the program is unchanged. */
static unsigned long long xo_s[4];
static unsigned long long xo_rotl(unsigned long long x, int k)
{
  return (x << k) | (x >> (64 - k));
}
static unsigned long long xo_next(void)
{
  unsigned long long result = xo_rotl(xo_s[0] + xo_s[3], 23) + xo_s[0];
  unsigned long long t = xo_s[1] << 17;
  xo_s[2] ^= xo_s[0]; xo_s[3] ^= xo_s[1]; xo_s[1] ^= xo_s[2]; xo_s[0] ^= xo_s[3];
  xo_s[2] ^= t; xo_s[3] = xo_rotl(xo_s[3], 45);
  return result;
}
double RandomNum(void)
{
  static int first_time = 1;
  if (first_time) {
    unsigned long long z = 0x9E3779B97F4A7C15ULL ^ 20261014ULL; int i;
    for (i = 0; i < 4; i++) {
      unsigned long long x;
      z += 0x9E3779B97F4A7C15ULL;
      x = z;
      x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
      x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
      xo_s[i] = x ^ (x >> 31);
    }
    first_time = 0;
  }
  return ((double)(xo_next() >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}
"""


def make_xoshiro_sources(src: Path, dest: Path) -> None:
    """Copy the sources to ``dest``, replacing only ``RandomNum()`` in ``mcmlgo.c``.

    The original uses a single-precision subtractive generator seeded from the clock in whole
    seconds. Swapping only the generator isolates its effect: on the three-layer case the
    original's transmittance differs from this engine's by 2e-4 (5 standard errors), and with a
    modern generator in its place the difference disappears. The edit is made to a copy; the
    sources in ``src`` are never touched.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_FILES:
        text = (src / name).read_text()
        if name == "mcmlgo.c":
            start = text.index("double RandomNum(void)")
            end = text.index("\n}\n", start) + 3
            text = text[:start] + XOSHIRO_RANDOM_NUM + text[end:]
        (dest / name).write_text(text)


# --------------------------------------------------------------------------------------------
# Input and output files
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Spec:
    photons: int
    dz: float
    dr: float
    nz: int
    nr: int
    na: int
    n_above: float
    layers: tuple[tuple[float, float, float, float, float], ...]  # n, mua, mus, g, d
    n_below: float


def _data_lines(text: str) -> list[list[str]]:
    lines = []
    for raw in text.splitlines():
        body = raw.split("#", 1)[0].split()
        if body:
            lines.append(body)
    return lines


def read_spec(mci: Path) -> Spec:
    """First run of an .mci file."""
    rows = _data_lines(mci.read_text())
    # rows: version, runs, then per run: name+format, photons, dz dr, nz nr na, layers, n above,
    # layer lines, n below.
    r = rows[2:]
    n_layers = int(r[4][0])
    layers = tuple(tuple(float(x) for x in row) for row in r[6 : 6 + n_layers])
    return Spec(
        photons=int(r[1][0]),
        dz=float(r[2][0]),
        dr=float(r[2][1]),
        nz=int(r[3][0]),
        nr=int(r[3][1]),
        na=int(r[3][2]),
        n_above=float(r[5][0]),
        layers=layers,  # type: ignore[arg-type]
        n_below=float(r[6 + n_layers][0]),
    )


def write_multirun_mci(spec: Spec, runs: int, photons_per_run: int, path: Path) -> None:
    out = ["1.0", str(runs)]
    for i in range(runs):
        out += [
            f"run{i:03d}.mco A",
            str(photons_per_run),
            f"{spec.dz!r} {spec.dr!r}",
            f"{spec.nz} {spec.nr} {spec.na}",
            str(len(spec.layers)),
            repr(spec.n_above),
        ]
        out += [" ".join(repr(x) for x in layer) for layer in spec.layers]
        out.append(repr(spec.n_below))
    path.write_text("\n".join(out) + "\n")


@dataclass(frozen=True)
class McmlRun:
    rsp: float
    rd: float
    absorbed: float
    tt: float
    echo: dict


def parse_mco(text: str) -> McmlRun:
    """Read the reflectance/absorption/transmission block and the echoed input parameters."""
    lines = text.splitlines()
    try:
        rat = next(i for i, ln in enumerate(lines) if ln.startswith("RAT"))
        inparm = next(i for i, ln in enumerate(lines) if ln.startswith("InParm"))
    except StopIteration:
        raise ValueError("not an MCML output file: no InParm/RAT section") from None
    values = [float(lines[rat + 1 + k].split()[0]) for k in range(4)]

    # The echoed input is the data rows between "InParm" and "RAT" (comments and blanks dropped):
    # name+format, photons, dz dr, nz nr na, layer count, n above, layer rows, n below.
    r = _data_lines("\n".join(lines[inparm + 1 : rat]))
    n_layers = int(r[4][0])
    echo = {
        "photons": int(r[1][0]),
        "dz": float(r[2][0]),
        "dr": float(r[2][1]),
        "grid": [int(x) for x in r[3]],
        "n_layers": n_layers,
        "n_above": float(r[5][0]),
        "layers": [[float(x) for x in r[6 + i]] for i in range(n_layers)],
        "n_below": float(r[6 + n_layers][0]),
    }
    return McmlRun(rsp=values[0], rd=values[1], absorbed=values[2], tt=values[3], echo=echo)


def check_echo(spec: Spec, photons_per_run: int, run: McmlRun) -> list[str]:
    """Differences between what was sent and what the program says it read.

    The 1993 input reader returns the address of a local buffer, which happens to work but is
    undefined behaviour, so every output is checked rather than trusting the parse.
    """
    e, problems = run.echo, []

    def close(a: float, b: float) -> bool:
        return math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-12)

    if e["photons"] != photons_per_run:
        problems.append(f"photons {e['photons']} != {photons_per_run}")
    if not (close(e["dz"], spec.dz) and close(e["dr"], spec.dr)):
        problems.append("dz/dr differ")
    if e["grid"] != [spec.nz, spec.nr, spec.na]:
        problems.append(f"grid {e['grid']} != {[spec.nz, spec.nr, spec.na]}")
    if e["n_layers"] != len(spec.layers):
        problems.append("layer count differs")
    elif not all(
        all(close(a, b) for a, b in zip(got, want, strict=True))
        for got, want in zip(e["layers"], spec.layers, strict=True)
    ):
        problems.append("layer parameters differ")
    if not (close(e["n_above"], spec.n_above) and close(e["n_below"], spec.n_below)):
        problems.append("ambient indices differ")
    return problems


def run_original(binary: Path, spec: Spec, runs: int, photons_per_run: int) -> list[McmlRun]:
    """One launch of the original program with ``runs`` independent runs of ``photons_per_run``.

    The program seeds from the wall clock in whole seconds (16 bits), once per launch, and the
    runs in one launch continue that single stream. Launches within the same second would share a
    seed, so callers wait between launches.
    """
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        write_multirun_mci(spec, runs, photons_per_run, work / "input.mci")
        proc = subprocess.run(
            [str(binary), "input.mci"], cwd=work, capture_output=True, text=True, timeout=3600
        )
        if proc.returncode != 0:
            raise SystemExit(f"original MCML failed ({proc.returncode}): {proc.stderr.strip()}")
        results = []
        for i in range(runs):
            run = parse_mco((work / f"run{i:03d}.mco").read_text())
            problems = check_echo(spec, photons_per_run, run)
            if problems:
                raise SystemExit(f"run {i}: the program did not read the input as sent: {problems}")
            results.append(run)
        return results


# --------------------------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseComparison:
    case: str
    quantity: str
    ours: stats.Estimate
    mcml: stats.Estimate
    comparison: stats.Comparison

    def line(self) -> str:
        c = self.comparison
        flag = "PASS" if c.passed else "FAIL"
        return (
            f"  [{flag}] {self.quantity:<9} ours {self.ours.mean:.6f} +/- {self.ours.se:.6f}   "
            f"MCML {self.mcml.mean:.6f} +/- {self.mcml.se:.6f}   z = {c.z:+.2f}"
        )


def _estimate(values: list[float]) -> stats.Estimate:
    # Equal photons per run, so the per-run fractions are averaged directly.
    return stats.batch_estimate([v for v in values], [1] * len(values))


def compare_case(case: str, ours_run: dict, theirs: list[McmlRun]) -> list[CaseComparison]:
    batches = ours_run["batches"]
    photons = [b["photons"] for b in batches]
    ours = {
        "rd": stats.batch_estimate([b["rd"] for b in batches], photons),
        "tt": stats.batch_estimate([b["tt"] for b in batches], photons),
        "absorbed": stats.batch_estimate([math.fsum(b["absorbed"]) for b in batches], photons),
    }
    mcml = {
        "rd": _estimate([r.rd for r in theirs]),
        "tt": _estimate([r.tt for r in theirs]),
        "absorbed": _estimate([r.absorbed for r in theirs]),
    }
    out = []
    for q in ("rd", "tt", "absorbed"):
        # A quantity that is identically zero in both (T_t for a semi-infinite medium) says nothing.
        if ours[q].mean == 0.0 and mcml[q].mean == 0.0:
            continue
        cmp = stats.compare(ours[q].mean, ours[q].se, mcml[q].mean, mcml[q].se)
        out.append(CaseComparison(case, q, ours[q], mcml[q], cmp))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tissuemc-compare-mcml", description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", help="comma-separated case names (default: all four)")
    ap.add_argument("--runs", type=int, default=40, help="independent runs per case")
    ap.add_argument("--photons-per-run", type=int, default=250_000)
    ap.add_argument("--source", type=Path, help="directory with the MCML sources")
    ap.add_argument("--binary", type=Path, help="engine binary (default: build one)")
    ap.add_argument("--seed", type=int, default=1, help="seed for this engine")
    ap.add_argument(
        "--original-rng",
        choices=["original", "xoshiro"],
        default="original",
        help="'xoshiro' replaces only the original's random number generator (see "
        "make_xoshiro_sources); everything else is the original code",
    )
    ap.add_argument("--report", type=Path, help="write all comparisons to this JSON file")
    args = ap.parse_args(argv)

    root = validate.find_root()
    src = args.source or root / "third_party" / "mcml"
    verify_sources(src)
    if args.original_rng == "xoshiro":
        variant = root / "build-mcml" / "xoshiro-src"
        make_xoshiro_sources(src, variant)
        mcml_bin = build_mcml(variant, root / "build-mcml", "mcml-xoshiro")
    else:
        mcml_bin = build_mcml(src, root / "build-mcml")
    engine = args.binary or validate.ensure_built(root)
    refs = json.loads((root / "configs" / "validation" / "references.json").read_text())
    names = args.cases.split(",") if args.cases else DEFAULT_CASES
    unknown = [n for n in names if n not in refs["cases"]]
    if unknown:
        raise SystemExit(f"unknown case(s): {', '.join(unknown)}")

    total_photons = args.runs * args.photons_per_run
    what = "original MCML" if args.original_rng == "original" else "original MCML with xoshiro256++"
    print(f"{what}, {args.runs} runs x {args.photons_per_run} photons = {total_photons}")
    print(f"sources: {', '.join(f'{n} {PINNED_SHA256[n][:8]}' for n in SOURCE_FILES)}\n")

    everything: list[CaseComparison] = []
    for name in names:
        config = root / "configs" / refs["cases"][name]["config"]
        spec = read_spec(config)
        print(f"{refs['cases'][name]['title']}")
        time.sleep(1.1)  # the original seeds from the clock in whole seconds
        started = time.time()
        theirs = run_original(mcml_bin, spec, args.runs, args.photons_per_run)
        mcml_seconds = time.time() - started
        ours_run = validate.run_engine(engine, config, total_photons, args.runs, args.seed)
        results = compare_case(name, ours_run, theirs)
        for r in results:
            print(r.line())
        print(f"  (original: {mcml_seconds:.0f} s; engine: {ours_run['wall_seconds']:.1f} s)")
        everything.extend(results)

    failed = [r for r in everything if not r.comparison.passed]
    print(
        f"\n{len(everything) - len(failed)}/{len(everything)} comparisons within 3 standard errors"
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "runs": args.runs,
                    "photons_per_run": args.photons_per_run,
                    "engine_seed": args.seed,
                    "original_rng": args.original_rng,
                    "source_sha256": PINNED_SHA256,
                    "compile_flags": COMPILE_FLAGS,
                    "comparisons": [
                        {
                            "case": r.case,
                            "quantity": r.quantity,
                            "ours": r.ours.mean,
                            "se_ours": r.ours.se,
                            "mcml": r.mcml.mean,
                            "se_mcml": r.mcml.se,
                            "z": r.comparison.z,
                            "passed": r.comparison.passed,
                        }
                        for r in everything
                    ],
                },
                indent=1,
            )
            + "\n"
        )
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
