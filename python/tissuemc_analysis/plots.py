"""Plot benchmark results.

    uv run tissuemc-plot results/cpu-scaling.json --output docs/img/cpu-scaling.png

Needs matplotlib (``uv sync --extra plots``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def plot_scaling(data: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = data["results"]
    threads = [r["threads"] for r in results]
    speedup = [r["speedup"] for r in results]
    base = data.get("speedup_baseline_threads", threads[0])
    hw = data.get("hardware", {})
    perf = hw.get("performance_cores")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    ax1.plot(threads, [t / base for t in threads], "--", color="0.6", label="ideal")
    ax1.errorbar(
        threads,
        speedup,
        yerr=[
            [s - base_wall_ratio(results, r, "q3") for s, r in zip(speedup, results, strict=True)],
            [base_wall_ratio(results, r, "q1") - s for s, r in zip(speedup, results, strict=True)],
        ],
        marker="o",
        capsize=3,
        label="measured (median, IQR)",
    )
    ax1.set_xlabel("threads")
    ax1.set_ylabel(f"speedup over {base} thread{'s' if base != 1 else ''}")
    ax1.set_title("CPU scaling")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.bar([str(t) for t in threads], [r["events_per_second"] / 1e6 for r in results], color="C0")
    ax2.set_xlabel("threads")
    ax2.set_ylabel("million events / s")
    ax2.set_title("Throughput")
    ax2.grid(axis="y", alpha=0.3)

    if perf:
        for ax, x in ((ax1, perf), (ax2, None)):
            if x is not None:
                ax.axvline(x, color="C3", ls=":", lw=1)
                ax.annotate(
                    f"{perf} performance cores",
                    (x, ax.get_ylim()[0]),
                    xytext=(4, 6),
                    textcoords="offset points",
                    color="C3",
                    fontsize=8,
                )

    cpu = hw.get("cpu", "unknown CPU")
    fig.suptitle(
        f"{data.get('workload', '')}: {data.get('photons', '?')} photons, {cpu}", fontsize=10
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def base_wall_ratio(results: list[dict], r: dict, which: str) -> float:
    """Speedup implied by the q1 (fast end) or q3 (slow end) wall time of this thread count."""
    base_wall = results[0]["wall"]["median"]
    return base_wall / r["wall"][which]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tissuemc-plot", description=__doc__.split("\n\n")[0])
    ap.add_argument("input", type=Path, help="results JSON from tissuemc-bench")
    ap.add_argument("--output", type=Path, required=True, help="PNG to write")
    args = ap.parse_args(argv)
    data = json.loads(args.input.read_text())
    if data.get("benchmark") != "cpu-scaling":
        raise SystemExit(f"{args.input}: not a cpu-scaling result file")
    plot_scaling(data, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
