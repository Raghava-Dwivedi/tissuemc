import numpy as np
import pytest
from tissuemc_analysis import oracle


def test_h_function_converges_with_the_number_of_nodes():
    omega = 0.9
    values = []
    for nodes in (60, 120, 240):
        mu, wt, h = oracle.h_function(omega, nodes)
        values.append(oracle.h_at(1.0, omega, mu, wt, h))
    assert abs(values[1] - values[2]) < 1e-10
    assert abs(values[0] - values[2]) < 1e-6


@pytest.mark.parametrize("mu0", [1.0, 0.5, 0.1])
def test_absorbed_fraction_identity(mu0):
    """Integrating the exit-angle density must give 1 - sqrt(1 - omega) H(mu0).

    The identity follows from the integral equation but is not used to build the kernel, so it
    checks both the H-function and the kernel.
    """
    omega = 0.9
    mu, wt, h = oracle.h_function(omega, 240)
    h0 = oracle.h_at(mu0, omega, mu, wt, h)
    returned = np.sum(wt * omega * mu * h * h0 / (2.0 * (mu + mu0)))
    assert returned == pytest.approx(1.0 - np.sqrt(1.0 - omega) * h0, abs=1e-9)


def test_matched_boundary_has_no_specular_and_matches_h_function():
    omega = 0.9
    res = oracle.semi_infinite_isotropic(mua=10, mus=90, n_medium=1.0, n_ambient=1.0)
    mu, wt, h = oracle.h_function(omega, 200)
    expected = 1.0 - np.sqrt(1.0 - omega) * oracle.h_at(1.0, omega, mu, wt, h)
    assert res.rsp == 0.0
    assert res.rd == pytest.approx(expected, abs=1e-9)
    assert res.total == res.rd


def test_mismatch_traps_light_and_lowers_diffuse_reflectance():
    matched = oracle.semi_infinite_isotropic(10, 90, 1.0)
    mismatched = oracle.semi_infinite_isotropic(10, 90, 1.5)
    assert mismatched.rsp == pytest.approx(0.04)
    assert mismatched.rd < matched.rd


def test_reflectance_rises_with_albedo():
    rd = [oracle.semi_infinite_isotropic(1 - w, w, 1.0).rd for w in (0.5, 0.9, 0.99)]
    assert rd[0] < rd[1] < rd[2] < 1.0


def test_agrees_with_the_published_giovanelli_total_reflectance():
    """An external anchor: the manual quotes 0.2600 for this medium (n = 1.5, albedo 0.9).

    The two differ by about 3e-4; the published value's own precision is not stated, so the
    tolerance here is deliberately loose and the number is a consistency check, not a proof.
    """
    res = oracle.semi_infinite_isotropic(10, 90, 1.5)
    assert abs(res.total - 0.2600) < 5e-4


def test_fresnel_reflectance_limits():
    r = oracle.fresnel_reflectance(1.5, 1.0, np.array([1.0, 0.7, 0.3]))
    assert r[0] == pytest.approx(0.04)
    assert r[1] == 1.0  # beyond the critical angle (cos 41.8 deg = 0.745)
    assert r[2] == 1.0


def test_rejects_non_absorbing_or_invalid_media():
    with pytest.raises(ValueError):
        oracle.h_function(1.0)
    with pytest.raises(ValueError):
        oracle.semi_infinite_isotropic(mua=0.0, mus=10, n_medium=1.0)
