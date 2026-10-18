"""The --no-profiles switch: a GPU diagnostic that must never be mistaken for a result."""

import json
import subprocess
from pathlib import Path

import pytest
from tissuemc_analysis import bench, cross_engine, validate

ROOT = Path(__file__).resolve().parents[2]


def _engine_result(profiles: bool = True) -> dict:
    return {
        "events": 1000,
        "anomalies": 0,
        "profiles": profiles,
        "device": "Tesla T4 (sm_75)",
        "kernel_seconds": 1.0,
        "wall_seconds": 1.1,
        "summary": {"rd": {"mean": 0.24}, "tt": {"mean": 0.096}},
        "batches": [
            {"photons": 100, "rd": 24.0, "tt": 9.6, "absorbed": [30.0, 20.0, 14.0]},
            {"photons": 100, "rd": 24.4, "tt": 9.5, "absorbed": [30.2, 19.8, 14.1]},
            {"photons": 100, "rd": 23.7, "tt": 9.7, "absorbed": [29.9, 20.1, 13.9]},
        ],
        "specular": 0.02,
        "a_z": [0.0, 0.0] if not profiles else [1.0, 2.0],
        "rd_r": [0.0, 0.0] if not profiles else [3.0, 4.0],
    }


def test_run_engine_passes_the_flag_only_to_the_gpu_and_only_when_asked(monkeypatch):
    seen = []

    def fake_run(cmd, **kwargs):
        seen.append(cmd)
        out = Path(cmd[cmd.index("--output") + 1])
        out.write_text(json.dumps({"runs": [_engine_result()]}))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validate.subprocess, "run", fake_run)
    args = (Path("b"), Path("c"), 1000, 10, 1)
    validate.run_engine(*args, engine="cuda", rung=6)
    validate.run_engine(*args, engine="cuda", rung=6, profiles=False)
    validate.run_engine(*args, engine="cpu")
    assert "--no-profiles" not in seen[0]
    assert "--no-profiles" in seen[1] and seen[1][seen[1].index("--rung") + 1] == "6"
    assert "--no-profiles" not in seen[2]  # the CPU engine never gets it


def test_the_benchmark_passes_the_flag_and_records_it(monkeypatch):
    cmds = []

    def fake(cmd):
        cmds.append(cmd)
        return _engine_result(profiles="--no-profiles" not in cmd)

    monkeypatch.setattr(bench, "_run_engine_json", fake)
    kwargs = dict(
        binary=Path("b"), config=Path("c"), photons=100, batches=3, repeats=2, warmup=0, seed=1
    )
    ablated = bench.run_gpu_benchmark(**kwargs, rung=6, profiles=False)
    normal = bench.run_gpu_benchmark(**kwargs, rung=6)
    assert ablated["profiles"] is False and normal["profiles"] is True
    assert all("--no-profiles" in c for c in cmds[:2]) and all(
        "--no-profiles" not in c for c in cmds[2:]
    )


def test_the_benchmark_refuses_a_run_that_did_not_honour_the_flag(monkeypatch):
    monkeypatch.setattr(bench, "_run_engine_json", lambda cmd: _engine_result(profiles=True))
    with pytest.raises(SystemExit, match="did not report the requested profiles"):
        bench.run_gpu_benchmark(Path("b"), Path("c"), 100, 3, 1, 0, 1, rung=6, profiles=False)


def test_a_diagnostic_benchmark_is_labelled_not_validated(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(bench, "_run_engine_json", lambda cmd: _engine_result(profiles=False))
    monkeypatch.setattr(bench.validate, "find_root", lambda: ROOT)
    monkeypatch.setattr(bench, "gpu_info", lambda: {})
    monkeypatch.setattr(bench, "hardware_info", lambda: {})
    monkeypatch.setattr(bench, "build_info", lambda root: {})
    monkeypatch.setattr(bench, "_run", lambda cmd: "tissuemc test")
    monkeypatch.setattr(bench, "cpu_reference", lambda root: None)
    out = tmp_path / "ablated.json"
    argv = ["--engine", "cuda", "--rung", "6", "--no-profiles", "--binary", "b"]
    argv += ["--photons", "100", "--batches", "3", "--repeats", "2", "--warmup", "0"]
    argv += ["--output", str(out)]
    assert bench.main(argv) == 0
    data = json.loads(out.read_text())
    assert data["validated_before_measuring"] is False
    assert data["benchmark"] == "gpu-r6-noprofiles"
    assert "DIAGNOSTIC" in capsys.readouterr().out


def test_the_flag_is_refused_for_the_cpu_engine():
    with pytest.raises(SystemExit):
        bench.main(["--no-profiles"])


def test_cross_engine_checks_the_diagnostic_really_dropped_the_profiles(monkeypatch, tmp_path):
    def run_ok(*a, **kw):
        return _engine_result(profiles=kw.get("profiles", True))

    monkeypatch.setattr(validate, "find_root", lambda: ROOT)
    monkeypatch.setattr(validate, "ensure_built", lambda root: Path("b"))
    monkeypatch.setattr(validate, "run_engine", run_ok)
    report = tmp_path / "r.json"
    argv = ["--b-engine", "cuda", "--b-rung", "6", "--b-no-profiles", "--report", str(report)]
    assert cross_engine.main(argv) == 0
    assert json.loads(report.read_text())["b"]["profiles"] is False

    # An engine that claims profiles are off but still reports a depth profile is refused.
    def run_bad(*a, **kw):
        r = _engine_result(profiles=kw.get("profiles", True))
        r["profiles"] = False
        r["a_z"] = [1.0, 2.0]
        return r

    monkeypatch.setattr(validate, "run_engine", run_bad)
    with pytest.raises(SystemExit, match="still tallied"):
        cross_engine.main(argv)
