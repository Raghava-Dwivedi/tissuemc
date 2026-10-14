from pathlib import Path

import pytest
from tissuemc_analysis import bench, plots

ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / "build" / "tissuemc"
needs_binary = pytest.mark.skipif(not BINARY.exists(), reason="engine not built (build/tissuemc)")


def test_summarize_median_and_quartiles():
    s = bench.summarize([5.0, 1.0, 3.0, 2.0, 4.0])
    assert s["median"] == 3.0
    assert (s["q1"], s["q3"]) == (2.0, 4.0)
    assert s["iqr"] == 2.0


def test_summarize_few_values_falls_back_to_range():
    assert bench.summarize([7.0]) == {"median": 7.0, "q1": 7.0, "q3": 7.0, "iqr": 0.0}
    two = bench.summarize([1.0, 3.0])
    assert (two["median"], two["q1"], two["q3"]) == (2.0, 1.0, 3.0)
    with pytest.raises(ValueError):
        bench.summarize([])


def test_parse_threads():
    assert bench.parse_threads("4,1,2") == [1, 2, 4]
    for bad in ("", "0", "1,1", "-2,4", "a"):
        with pytest.raises(ValueError):
            bench.parse_threads(bad)


def test_background_cpu_percent_is_a_nonnegative_number():
    value = bench.background_cpu_percent()
    assert value is None or value >= 0.0


def test_hardware_and_build_info_have_the_basics():
    hw = bench.hardware_info()
    assert {"os", "machine", "logical_cpus"} <= hw.keys()
    assert hw["logical_cpus"] >= 1
    info = bench.build_info(ROOT)
    if (ROOT / ".git").exists():
        assert len(info["git_commit"]) == 40
        assert isinstance(info["git_dirty"], bool)


@needs_binary
def test_run_benchmark_structure_and_baseline():
    config = ROOT / "configs" / "validation" / "v3_three_layer.mci"
    body = bench.run_benchmark(
        BINARY, config, photons=4000, batches=8, threads=[1, 2], repeats=2, warmup=0, seed=1
    )
    first, second = body["results"]
    assert (first["threads"], second["threads"]) == (1, 2)
    assert first["speedup"] == pytest.approx(1.0)
    assert first["efficiency"] == pytest.approx(1.0)
    assert first["events"] == second["events"]  # same photons, same answer at any thread count
    assert len(first["wall_seconds"]) == 2
    assert second["speedup"] == pytest.approx(second["efficiency"] * 2)


def test_run_benchmark_refuses_inconsistent_results(monkeypatch):
    """Throughput of a wrong answer is meaningless, so differing results must abort the run."""

    def fake(cmd):
        threads = int(cmd[cmd.index("--threads") + 1])
        events = 1000 if threads == 1 else 1001
        return {
            "wall_seconds": 1.0,
            "events": events,
            "summary": {"rd": {"mean": 0.5}, "tt": {"mean": 0.1}},
        }

    monkeypatch.setattr(bench, "_run_engine_json", fake)
    with pytest.raises(SystemExit, match="differ between thread counts"):
        bench.run_benchmark(
            Path("x"), Path("y"), photons=10, batches=4, threads=[1, 2], repeats=1, warmup=0, seed=1
        )


def test_scaling_plot_writes_a_png(tmp_path):
    def row(t, wall):
        return {
            "threads": t,
            "speedup": 10.0 / wall,
            "efficiency": 10.0 / wall / t,
            "events_per_second": 3e7 * 10.0 / wall,
            "wall": {"median": wall, "q1": wall * 0.98, "q3": wall * 1.02},
        }

    data = {
        "benchmark": "cpu-scaling",
        "workload": "demo.mci",
        "photons": 1000,
        "speedup_baseline_threads": 1,
        "hardware": {"cpu": "Test CPU", "performance_cores": 2},
        "results": [row(1, 10.0), row(2, 5.2), row(4, 3.0)],
    }
    out = tmp_path / "plot.png"
    plots.plot_scaling(data, out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert out.stat().st_size > 5000


def _fake_gpu_run(events=1000, rd=0.24, kernel=2.0):
    return {
        "kernel_seconds": kernel,
        "wall_seconds": kernel + 0.5,
        "events": events,
        "device": "Test GPU (sm_75)",
        "tally_precision": "float64",
        "summary": {"rd": {"mean": rd}, "tt": {"mean": 0.1}},
    }


def test_gpu_benchmark_summary(monkeypatch):
    kernels = iter([9.0, 2.0, 4.0, 3.0])  # first is the warm-up
    monkeypatch.setattr(bench, "_run_engine_json", lambda cmd: _fake_gpu_run(kernel=next(kernels)))
    out = bench.run_gpu_benchmark(Path("x"), Path("y"), 1000, 10, repeats=3, warmup=1, seed=1)
    assert out["kernel"]["median"] == 3.0
    assert out["events_per_second"] == pytest.approx(1000 / 3.0)
    assert out["photons_per_second"] == pytest.approx(1000 / 3.0)
    assert out["device"] == "Test GPU (sm_75)"


def test_gpu_benchmark_tolerates_atomic_rounding_but_not_real_differences(monkeypatch):
    runs = iter([_fake_gpu_run(rd=0.24), _fake_gpu_run(rd=0.24 * (1 + 1e-12))])
    monkeypatch.setattr(bench, "_run_engine_json", lambda cmd: next(runs))
    bench.run_gpu_benchmark(Path("x"), Path("y"), 10, 2, repeats=2, warmup=0, seed=1)

    runs = iter([_fake_gpu_run(events=1000), _fake_gpu_run(events=1001)])
    monkeypatch.setattr(bench, "_run_engine_json", lambda cmd: next(runs))
    with pytest.raises(SystemExit, match="event counts differ"):
        bench.run_gpu_benchmark(Path("x"), Path("y"), 10, 2, repeats=2, warmup=0, seed=1)

    runs = iter([_fake_gpu_run(rd=0.24), _fake_gpu_run(rd=0.2401)])
    monkeypatch.setattr(bench, "_run_engine_json", lambda cmd: next(runs))
    with pytest.raises(SystemExit, match="rd differs"):
        bench.run_gpu_benchmark(Path("x"), Path("y"), 10, 2, repeats=2, warmup=0, seed=1)


def test_gpu_info_is_empty_without_a_gpu_or_has_a_name():
    info = bench.gpu_info()
    assert info == {} or "gpu" in info or "nvcc" in info
