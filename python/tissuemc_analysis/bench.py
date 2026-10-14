"""Benchmark the CPU engine across thread counts.

    uv run tissuemc-bench --output results/cpu-scaling.json

Throughput is only recorded for a build that passes the validation suite, and only if every
thread count produces identical results (the engine promises this). Each result file carries the
hardware, compiler, flags and git commit it was measured with.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import validate

NOTES = [
    "Throughput is the median of the repeats; iqr is the interquartile range across repeats.",
    "macOS cannot pin threads to cores. Threads beyond the performance-core count also run on "
    "efficiency cores, which are slower and add noise; efficiency is speedup / threads.",
    "Photons per second counts launched photon packets; events per second counts "
    "absorption/scattering events, which is the better measure of work done.",
]


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def hardware_info() -> dict:
    """CPU model and core counts. Fields that cannot be determined are left out, not guessed."""
    info: dict = {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
    }
    if platform.system() == "Darwin":
        for key, name in [
            ("cpu", "machdep.cpu.brand_string"),
            ("performance_cores", "hw.perflevel0.physicalcpu"),
            ("efficiency_cores", "hw.perflevel1.physicalcpu"),
            ("memory_bytes", "hw.memsize"),
        ]:
            value = _run(["sysctl", "-n", name])
            if value:
                info[key] = int(value) if value.isdigit() else value
    else:
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            match = re.search(r"model name\s*:\s*(.+)", cpuinfo.read_text())
            if match:
                info["cpu"] = match.group(1).strip()
    return info


def build_info(root: Path) -> dict:
    """Compiler, flags and source revision the binary was built from."""
    info: dict = {}
    cache = root / "build" / "CMakeCache.txt"
    if cache.exists():
        text = cache.read_text()

        def entry(name: str) -> str:
            m = re.search(rf"^{name}:[A-Z]+=(.*)$", text, re.M)
            return m.group(1) if m else ""

        compiler = entry("CMAKE_CXX_COMPILER")
        info["compiler"] = compiler
        version = _run([compiler, "--version"]).splitlines()
        if version:
            info["compiler_version"] = version[0]
        info["build_type"] = entry("CMAKE_BUILD_TYPE")
        info["flags"] = (entry("CMAKE_CXX_FLAGS") + " " + entry("CMAKE_CXX_FLAGS_RELEASE")).strip()
    commit = _run(["git", "-C", str(root), "rev-parse", "HEAD"])
    if commit:
        info["git_commit"] = commit
        info["git_dirty"] = bool(_run(["git", "-C", str(root), "status", "--porcelain"]))
    return info


def summarize(values: list[float]) -> dict:
    """Median and interquartile range of the repeats."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    if len(ordered) >= 4:
        q1, _, q3 = statistics.quantiles(ordered, n=4, method="inclusive")
    else:
        q1, q3 = ordered[0], ordered[-1]
    return {"median": statistics.median(ordered), "q1": q1, "q3": q3, "iqr": q3 - q1}


def parse_threads(text: str) -> list[int]:
    threads = [int(t) for t in text.split(",") if t.strip()]
    if not threads or any(t < 1 for t in threads) or len(set(threads)) != len(threads):
        raise ValueError("threads must be distinct positive integers, e.g. 1,2,4,8,12")
    return sorted(threads)


def run_benchmark(
    binary: Path,
    config: Path,
    photons: int,
    batches: int,
    threads: list[int],
    repeats: int,
    warmup: int,
    seed: int,
) -> dict:
    results = []
    fingerprints = []
    for t in threads:
        walls, events, summary = [], 0, None
        for i in range(warmup + repeats):
            cmd = [str(binary), "run", str(config), "--photons", str(photons)]
            cmd += ["--batches", str(batches), "--seed", str(seed), "--threads", str(t)]
            run = _run_engine_json(cmd)
            if i >= warmup:
                walls.append(run["wall_seconds"])
            events = run["events"]
            summary = run["summary"]
            fingerprints.append((t, run["events"], summary["rd"]["mean"], summary["tt"]["mean"]))
        results.append({"threads": t, "wall_seconds": walls, "events": events, "summary": summary})
        print(f"  threads {t:>2}: median {statistics.median(walls):7.2f} s over {repeats} runs")

    # Same seed and photon count must give the same answer at every thread count.
    first = fingerprints[0][1:]
    for fp in fingerprints:
        if fp[1:] != first:
            raise SystemExit(
                f"benchmark aborted: results differ between thread counts (threads {fp[0]} gave "
                f"{fp[1:]}, expected {first}). Throughput is not meaningful for a wrong answer."
            )

    base_wall = statistics.median(results[0]["wall_seconds"])
    base_threads = results[0]["threads"]
    for r in results:
        wall = summarize(r["wall_seconds"])
        r["wall"] = wall
        r["photons_per_second"] = photons / wall["median"]
        r["events_per_second"] = r["events"] / wall["median"]
        r["speedup"] = base_wall / wall["median"]
        r["efficiency"] = r["speedup"] * base_threads / r["threads"]
    return {"results": results, "speedup_baseline_threads": base_threads}


def _run_engine_json(cmd: list[str]) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "result.json"
        proc = subprocess.run([*cmd, "--output", str(out)], capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"engine failed ({proc.returncode}): {proc.stderr.strip()}")
        return json.loads(out.read_text())["runs"][0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tissuemc-bench", description=__doc__.split("\n\n")[0])
    ap.add_argument("--binary", type=Path, help="engine binary (default: build one)")
    ap.add_argument(
        "--config", type=Path, help="input file (default: configs/bench/tissue_g09.mci)"
    )
    ap.add_argument("--photons", type=int, default=300_000)
    ap.add_argument("--batches", type=int, default=240, help="batches; use well over 10x threads")
    ap.add_argument("--threads", default="1,2,4,8,12", help="thread counts to measure")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--output", type=Path, default=Path("results/cpu-scaling.json"))
    ap.add_argument("--skip-validation", action="store_true", help="NOT for published numbers")
    args = ap.parse_args(argv)

    threads = parse_threads(args.threads)
    if args.repeats < 1 or args.warmup < 0:
        raise SystemExit("--repeats must be >= 1 and --warmup >= 0")
    root = validate.find_root()
    binary = args.binary or validate.ensure_built(root)
    config = args.config or root / "configs" / "bench" / "tissue_g09.mci"

    if not args.skip_validation:
        print("Validating the build before measuring anything ...")
        v_args = ["--binary", str(binary), "--photons", "1000000"]
        if validate.main(v_args) != 0:
            print("benchmark aborted: the build fails validation", file=sys.stderr)
            return 1
        validated = True
    else:
        validated = False

    load_before = os.getloadavg()[0]
    print(f"\nBenchmarking {config.name}: {args.photons} photons, {args.batches} batches")
    started = time.time()
    body = run_benchmark(
        binary, config, args.photons, args.batches, threads, args.repeats, args.warmup, args.seed
    )
    data = {
        "benchmark": "cpu-scaling",
        "workload": config.name,
        "photons": args.photons,
        "batches": args.batches,
        "seed": args.seed,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "validated_before_measuring": validated,
        "load_average_1min_before": load_before,
        "hardware": hardware_info(),
        "build": build_info(root),
        "engine_version": _run([str(binary), "--version"]),
        "started_unix": started,
        "notes": NOTES,
        **body,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=1) + "\n")
    print(f"\nwrote {args.output}")
    print(f"{'threads':>8} {'Mevents/s':>10} {'speedup':>8} {'efficiency':>11}")
    for r in data["results"]:
        print(
            f"{r['threads']:>8} {r['events_per_second'] / 1e6:>10.1f} "
            f"{r['speedup']:>8.2f} {r['efficiency']:>10.0%}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
