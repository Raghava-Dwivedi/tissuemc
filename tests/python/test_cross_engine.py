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


def _with_profiles(run: dict, az: list[float], rr: list[float]) -> dict:
    return {**run, "a_z": az, "rd_r": rr}


AZ = [40.0, 25.0, 15.0, 10.0, 6.0, 3.0, 1.0]
RR = [5.0, 4.0, 3.0, 2.0, 1.0]


def test_identical_profiles_pass_and_are_reported():
    run = _with_profiles(fake_run(), AZ, RR)
    gaps = cross_engine.compare_runs(run, run)
    names = [g.name for g in gaps]
    assert "a_z: total relative gap" in names and "rd_r: worst cumulative gap / total" in names
    assert all(g.passed for g in gaps)


def test_a_lost_copy_of_the_depth_profile_is_caught():
    """The error a per-warp privatization invites: one warp's copy never reaches the total."""
    a = _with_profiles(fake_run(), AZ, RR)
    lost = [v * 0.75 for v in AZ]  # a quarter of the profile missing, as with one of four warps
    b = _with_profiles(fake_run(), lost, RR)
    failed = {g.name for g in cross_engine.compare_runs(a, b) if not g.passed}
    assert failed == {"a_z: total relative gap", "a_z: worst cumulative gap / total"}


def test_a_double_flush_of_the_radius_profile_is_caught():
    a = _with_profiles(fake_run(), AZ, RR)
    b = _with_profiles(fake_run(), AZ, [v * 2 for v in RR])
    failed = {g.name for g in cross_engine.compare_runs(a, b) if not g.passed}
    assert "rd_r: total relative gap" in failed and not any(n.startswith("a_z") for n in failed)


def test_a_photon_moved_to_a_neighbouring_bin_is_tolerated():
    """Bin-edge photons differ between engines that bin in float and in double."""
    a = _with_profiles(fake_run(), AZ, RR)
    moved = list(AZ)
    moved[0] -= 1e-5
    moved[1] += 1e-5
    assert all(
        g.passed for g in cross_engine.compare_runs(a, _with_profiles(fake_run(), moved, RR))
    )


def test_profiles_are_skipped_when_absent_or_when_the_diagnostic_zeroes_them():
    names = [g.name for g in cross_engine.compare_runs(fake_run(), fake_run())]
    assert not any(n.startswith(("a_z", "rd_r")) for n in names)  # runs without profiles
    a = _with_profiles(fake_run(), AZ, RR)
    zeros = _with_profiles(fake_run(), [0.0] * len(AZ), [0.0] * len(RR))
    assert not all(g.passed for g in cross_engine.compare_runs(a, zeros))
    assert all(g.passed for g in cross_engine.compare_runs(a, zeros, profiles=False))
