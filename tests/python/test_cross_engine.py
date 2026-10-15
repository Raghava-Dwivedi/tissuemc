from pathlib import Path

import pytest
from tissuemc_analysis import cross_engine, validate

ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / "build" / "tissuemc"
CONFIG = ROOT / "configs" / "validation" / "v3_three_layer.mci"
needs_binary = pytest.mark.skipif(not BINARY.exists(), reason="engine not built (build/tissuemc)")


def fake_run(scale: float = 1.0, events: int = 1000) -> dict:
    batches = [
        {"photons": 100, "rd": 24.0 * scale, "tt": 9.6 * scale, "absorbed": [30.0, 20.0, 14.0]},
        {"photons": 100, "rd": 24.4 * scale, "tt": 9.5 * scale, "absorbed": [30.2, 19.8, 14.1]},
        {"photons": 100, "rd": 23.7 * scale, "tt": 9.7 * scale, "absorbed": [29.9, 20.1, 13.9]},
    ]
    return {"batches": batches, "events": events, "anomalies": 0}


def test_identical_runs_have_zero_gaps():
    gaps = cross_engine.compare_runs(fake_run(), fake_run())
    assert all(g.passed for g in gaps)
    assert all(g.value == 0 for g in gaps)


def test_a_small_systematic_shift_is_caught():
    gaps = cross_engine.compare_runs(fake_run(), fake_run(scale=1.01))
    assert not all(g.passed for g in gaps)
    assert any(not g.passed and "per-batch" in g.name for g in gaps)


def test_event_count_and_anomalies_are_checked():
    assert not all(g.passed for g in cross_engine.compare_runs(fake_run(), fake_run(events=1100)))
    bad = fake_run()
    bad["anomalies"] = 1
    assert not all(g.passed for g in cross_engine.compare_runs(fake_run(), bad))


def test_different_batching_is_refused():
    other = fake_run()
    other["batches"][0]["photons"] = 99
    with pytest.raises(ValueError, match="same batches"):
        cross_engine.compare_runs(fake_run(), other)


@needs_binary
def test_float32_and_float64_cpu_agree_within_the_limits():
    """The comparison the limits were derived from: same photons, two transport precisions."""
    a = validate.run_engine(BINARY, CONFIG, 400_000, 20, 9, "float64")
    b = validate.run_engine(BINARY, CONFIG, 400_000, 20, 9, "float32")
    gaps = cross_engine.compare_runs(a, b)
    assert all(g.passed for g in gaps), [g.line() for g in gaps if not g.passed]


@needs_binary
def test_independent_runs_do_not_pass():
    """Different seeds are different photons; the check must tell them apart."""
    a = validate.run_engine(BINARY, CONFIG, 400_000, 20, 9, "float32")
    b = validate.run_engine(BINARY, CONFIG, 400_000, 20, 10, "float32")
    assert not all(g.passed for g in cross_engine.compare_runs(a, b))


def test_a_shift_in_absorbed_weight_alone_is_caught():
    """R_d and T_t unchanged but absorbed weight off: the check must still fail."""
    shifted = fake_run()
    for b in shifted["batches"]:
        b["absorbed"] = [x * 1.02 for x in b["absorbed"]]
    gaps = cross_engine.compare_runs(fake_run(), shifted)
    failed = [g.name for g in gaps if not g.passed]
    assert failed and all("absorbed" in name for name in failed)
