import math

import pytest
from tissuemc_analysis import stats


def test_batch_estimate_known_values():
    # Per-batch means 0.1, 0.2, 0.3 -> mean 0.2, sample sd 0.1, standard error 0.1 / sqrt(3).
    e = stats.batch_estimate([1.0, 2.0, 3.0], [10, 10, 10])
    assert e.mean == pytest.approx(0.2)
    assert e.se == pytest.approx(0.1 / math.sqrt(3))


def test_batch_estimate_single_batch_has_no_error_estimate():
    e = stats.batch_estimate([5.0], [100])
    assert e.mean == pytest.approx(0.05)
    assert math.isnan(e.se)


def test_batch_estimate_rejects_bad_shapes():
    with pytest.raises(ValueError):
        stats.batch_estimate([1.0, 2.0], [10])
    with pytest.raises(ValueError):
        stats.batch_estimate([], [])


def test_compare_pass_rule_is_inclusive_at_three_standard_errors():
    assert stats.compare(1.0 + 3.0e-3, 1e-3, 1.0).passed
    assert not stats.compare(1.0 + 3.01e-3, 1e-3, 1.0).passed
    assert not stats.compare(1.0 - 3.01e-3, 1e-3, 1.0).passed


def test_compare_combines_both_standard_errors():
    c = stats.compare(1.0, 3e-4, 1.0 + 1e-3, 4e-4)  # combined SE = 5e-4, so z = -2
    assert c.z == pytest.approx(-2.0)
    assert c.passed


def test_compare_rounding_allowance_widens_the_window():
    # An exact reference printed to 2 decimals is only known to +/- 0.005.
    assert not stats.compare(0.2049, 0.0, 0.20).passed
    assert stats.compare(0.2049, 0.0, 0.20, rounding=0.005).passed
    assert not stats.compare(0.2051, 0.0, 0.20, rounding=0.005).passed


def test_compare_with_zero_error_and_exact_match():
    c = stats.compare(0.0, 0.0, 0.0)
    assert c.passed
    assert c.z == 0.0
    assert not stats.compare(1.0, 0.0, 0.0).passed


def test_estimated_reference_se_scales_with_photons():
    # Ten times fewer photons behind the reference -> sqrt(10) times the error.
    assert stats.estimated_reference_se(1e-4, 4_000_000, 400_000) == pytest.approx(
        1e-4 * math.sqrt(10)
    )
