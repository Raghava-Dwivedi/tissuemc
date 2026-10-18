"""Hardware-counter profile of the GPU engine with Nsight Compute.

    uv run tissuemc-profile --rung 0 --binary build/tissuemc

Runs ``ncu`` on one rung with a modest photon count (profiling replays the kernel many times) and
keeps the numbers that say what limits it: how many threads of a warp are active per instruction
(divergence), how busy the compute and memory systems are, and occupancy.

    uv run tissuemc-profile --rung 3 --full --photons 1000000 --output results/gpu-r3-full.json
    uv run tissuemc-profile --digest results/gpu-r1-full.json results/gpu-r3-full.json

``--full`` collects every counter (about forty replays of the kernel) and ``--digest`` prints two
such files side by side for the questions the key counters cannot answer: why warps wait, how many
instructions each rung executes, and how much local-memory traffic there is.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import validate

SECTIONS = [
    "SpeedOfLight",
    "WarpStateStats",
    "SourceCounters",
    "Occupancy",
    "InstructionStats",
    "MemoryWorkloadAnalysis",
    "LaunchStats",
]

# (section, metric name as Nsight Compute prints it, what it tells us)
KEY_METRICS = [
    ("GPU Speed Of Light Throughput", "Duration", "kernel time"),
    (
        "GPU Speed Of Light Throughput",
        "Compute (SM) Throughput",
        "how busy the arithmetic units are (% of peak)",
    ),
    (
        "GPU Speed Of Light Throughput",
        "Memory Throughput",
        "how busy the memory system is (% of peak)",
    ),
    (
        "GPU Speed Of Light Throughput",
        "L2 Cache Throughput",
        "L2 busy, where global atomics are resolved (% of peak)",
    ),
    (
        "Warp State Statistics",
        "Avg. Active Threads Per Warp",
        "threads active per warp instruction, of 32 (divergence)",
    ),
    (
        "Source Counters",
        "Avg. Not Predicated Off Threads Per Warp",
        "same, excluding predicated-off lanes",
    ),
    (
        "Warp State Statistics",
        "Warp Cycles Per Issued Instruction",
        "average stall between issued instructions",
    ),
    ("Occupancy", "Achieved Occupancy", "warps resident per SM, % of the maximum"),
    ("Occupancy", "Theoretical Occupancy", "the limit set by registers and block size"),
    ("Instruction Statistics", "Executed Ipc Active", "instructions per cycle while active"),
    ("Launch Statistics", "Threads", "threads launched (the lanes, for the persistent kernel)"),
    ("Launch Statistics", "Registers Per Thread", "registers per thread; sets the occupancy limit"),
]


def parse_ncu_csv(text: str) -> dict[str, dict[str, dict[str, str]]]:
    """Section -> metric name -> {value, unit}, from ``ncu --csv --page details`` output.

    Only the first profiled kernel is kept. Lines before the CSV header (program output) are
    ignored.
    """
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith('"ID"')), None)
    if start is None:
        raise ValueError("no Nsight Compute CSV header found")
    out: dict[str, dict[str, dict[str, str]]] = {}
    first_id = None
    for row in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
        if first_id is None:
            first_id = row["ID"]
        if row["ID"] != first_id:
            break
        section = out.setdefault(row["Section Name"], {})
        section[row["Metric Name"]] = {"value": row["Metric Value"], "unit": row["Metric Unit"]}
    return out


def find_ncu() -> str:
    path = shutil.which("ncu") or "/usr/local/cuda/bin/ncu"
    if not Path(path).exists():
        raise SystemExit("Nsight Compute (ncu) not found")
    return path


def flatten(sections: dict[str, dict[str, dict[str, str]]]) -> dict[str, dict[str, str]]:
    """Metric name -> {value, unit}, dropping the section level (raw pages have no sections)."""
    return {name: v for section in sections.values() for name, v in section.items()}


def _number(text: str) -> float | None:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


# (heading, regular expression on the metric name, most rows to show). The stall-reason names are
# matched loosely because Nsight Compute has two families of them; every row found is printed.
DIGEST_GROUPS = [
    ("Warp stall reasons", r"issue_stalled_.*\.ratio$", 14),
    (
        "Instructions executed",
        r"^smsp__(thread_)?inst_executed(_op_(local|global|shared)_(ld|st))?\.sum$"
        r"|^smsp__thread_inst_executed_per_inst_executed\.ratio$",
        12,
    ),
    (
        "Local memory and L1",
        r"^l1tex__.*local.*\.(sum|pct)$|^l1tex__t_sector_hit_rate\.pct$|spill",
        14,
    ),
    (
        "Launch and occupancy",
        r"^launch__(registers_per_thread|thread_count|grid_size|block_size|occupancy_limit_.*)$"
        r"|^sm__warps_active\.avg\.pct_of_peak_sustained_active$",
        12,
    ),
]
FALLBACK_STALL = r"stall"


def digest(a: dict[str, dict[str, str]], b: dict[str, dict[str, str]]) -> list[str]:
    """Side-by-side lines for the counters that explain why one kernel waits longer than another."""
    lines: list[str] = []
    for title, pattern, limit in DIGEST_GROUPS:
        names = sorted(set(a) | set(b))
        hits = [m for m in names if re.search(pattern, m)]
        if not hits and title == "Warp stall reasons":
            hits = [m for m in names if re.search(FALLBACK_STALL, m)]

        def size(m: str) -> float:
            return max(abs(_number(x[m]["value"]) or 0.0) for x in (a, b) if m in x)

        lines.append(f"\n{title}  (A, B, B/A)")
        if not hits:
            lines.append("  (no matching counters)")
        for m in sorted(hits, key=size, reverse=True)[:limit]:
            va = a.get(m, {}).get("value", "-")
            vb = b.get(m, {}).get("value", "-")
            na, nb = _number(va), _number(vb)
            ratio = f"{nb / na:.2f}" if na and nb is not None else "-"
            lines.append(f"  {m:<78} {va:>16} {vb:>16} {ratio:>8}")
    return lines


def profile(
    binary: Path, config: Path, rung: int, photons: int, batches: int, full: bool = False
) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "ncu.csv"
        cmd = [
            find_ncu(),
            "--csv",
            "--page",
            "raw" if full else "details",
            "--log-file",
            str(log),
            "--launch-count",
            "1",
        ]
        if full:
            cmd += ["--set", "full"]
        else:
            for section in SECTIONS:
                cmd += ["--section", section]
        cmd += [
            str(binary),
            "run",
            str(config),
            "--photons",
            str(photons),
            "--batches",
            str(batches),
        ]
        cmd += ["--engine", "cuda", "--rung", str(rung)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not log.exists():
            raise SystemExit(
                f"ncu failed ({proc.returncode}): {(proc.stderr or proc.stdout)[-800:]}"
            )
        return parse_ncu_csv(log.read_text())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tissuemc-profile", description=__doc__.split("\n\n")[0])
    ap.add_argument("--rung", type=int, default=0)
    ap.add_argument("--binary", type=Path, help="engine binary built with CUDA")
    ap.add_argument(
        "--config", type=Path, help="input file (default: configs/bench/tissue_g09.mci)"
    )
    ap.add_argument("--photons", type=int, default=200_000)
    ap.add_argument("--batches", type=int, default=50)
    ap.add_argument("--output", type=Path, help="write all counters to this JSON file")
    ap.add_argument(
        "--full", action="store_true", help="collect every counter (slow); use with --output"
    )
    ap.add_argument(
        "--digest",
        nargs=2,
        type=Path,
        metavar=("A.json", "B.json"),
        help="print two --full files side by side and exit",
    )
    args = ap.parse_args(argv)

    if args.digest:
        a, b = (json.loads(p.read_text())["metrics"] for p in args.digest)
        print(f"A = {args.digest[0].name}    B = {args.digest[1].name}")
        print("\n".join(digest(a, b)))
        return 0

    root = validate.find_root()
    binary = args.binary or root / "build" / "tissuemc"
    config = args.config or root / "configs" / "bench" / "tissue_g09.mci"
    if args.full:
        metrics = flatten(profile(binary, config, args.rung, args.photons, args.batches, full=True))
        print(f"rung {args.rung}, {args.photons} photons: {len(metrics)} counters collected")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {"rung": args.rung, "photons": args.photons, "full": True, "metrics": metrics},
                    indent=1,
                )
                + "\n"
            )
        return 0
    sections = profile(binary, config, args.rung, args.photons, args.batches)

    print(f"rung {args.rung}, {args.photons} photons ({config.name}); key counters:")
    shown = {}
    for section, metric, meaning in KEY_METRICS:
        entry = sections.get(section, {}).get(metric)
        if entry:
            unit = f" {entry['unit']}" if entry["unit"] else ""
            print(f"  {metric:<42} {entry['value']:>12}{unit}   {meaning}")
            shown[metric] = entry
    if not shown:
        print("  (none of the expected counters were found; all sections are in the output file)")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"rung": args.rung, "photons": args.photons, "sections": sections}, indent=1)
            + "\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
