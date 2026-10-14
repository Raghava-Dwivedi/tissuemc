"""First-principles reference for a semi-infinite, isotropically scattering medium.

This module shares no code, and no modelling choices, with the C++ engine, so agreement between
the two is evidence about the engine and not about a shared mistake.

Method (S. Chandrasekhar, *Radiative Transfer*, 1960): for isotropic scattering with single-
scattering albedo ``omega`` the exit-angle distribution of light returning from a matched,
semi-infinite medium is expressed through the H-function, the solution of

    H(mu) = 1 + (omega / 2) * mu * H(mu) * integral_0^1 H(mu') / (mu + mu') dmu'.

A beam entering at direction cosine ``mu0`` is absorbed with probability ``sqrt(1-omega) H(mu0)``,
and leaves upward at cosine ``mu`` with density

    K(mu, mu0) = omega * mu * H(mu) * H(mu0) / (2 * (mu + mu0)).

A refractive-index mismatch is handled by a boundary recursion: light reaching the boundary from
inside leaves with probability ``1 - r(mu)`` (Fresnel, unpolarized, with total internal
reflection) and otherwise returns as a new downward beam at the same cosine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SemiInfiniteResult:
    rsp: float  # specular reflectance of the incident beam
    rd: float  # diffuse reflectance (specular excluded)

    @property
    def total(self) -> float:
        """Total reflectance: diffuse plus specular."""
        return self.rd + self.rsp


def _quadrature(nodes: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(nodes)
    return 0.5 * (x + 1.0), 0.5 * w  # Gauss-Legendre on (0, 1)


def h_function(omega: float, nodes: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return quadrature nodes ``mu``, weights, and ``H(mu)`` at the nodes.

    Solved by fixed-point iteration, which converges for omega < 1.
    """
    if not 0.0 <= omega < 1.0:
        raise ValueError("omega must be in [0, 1): the medium must absorb")
    mu, wt = _quadrature(nodes)
    h = np.ones(nodes)
    for _ in range(100_000):
        integral = (wt[None, :] * h[None, :] / (mu[:, None] + mu[None, :])).sum(axis=1)
        h_next = 1.0 / (1.0 - 0.5 * omega * mu * integral)
        done = np.max(np.abs(h_next - h)) < 1e-15
        h = h_next
        if done:
            return mu, wt, h
    raise RuntimeError("H-function iteration did not converge")


def h_at(m: float, omega: float, mu: np.ndarray, wt: np.ndarray, h: np.ndarray) -> float:
    """Evaluate H at an arbitrary cosine through its defining integral equation."""
    integral = float(np.sum(wt * h / (m + mu)))
    return 1.0 / (1.0 - 0.5 * omega * m * integral)


def fresnel_reflectance(n1: float, n2: float, cos_i: np.ndarray) -> np.ndarray:
    """Unpolarized Fresnel reflectance going from index n1 to n2 (1 on total reflection)."""
    sin_i = np.sqrt(np.maximum(0.0, 1.0 - cos_i**2))
    sin_t = n1 * sin_i / n2
    out = np.ones_like(cos_i)
    ok = sin_t < 1.0
    cos_t = np.sqrt(1.0 - sin_t[ok] ** 2)
    ci = cos_i[ok]
    rs = (n1 * ci - n2 * cos_t) / (n1 * ci + n2 * cos_t)
    rp = (n1 * cos_t - n2 * ci) / (n1 * cos_t + n2 * ci)
    out[ok] = 0.5 * (rs**2 + rp**2)
    return out


def semi_infinite_isotropic(
    mua: float, mus: float, n_medium: float, n_ambient: float = 1.0, nodes: int = 200
) -> SemiInfiniteResult:
    """Reflectance of a normally incident pencil beam on a semi-infinite isotropic medium."""
    if mua <= 0 or mus < 0:
        raise ValueError("need mua > 0 and mus >= 0")
    omega = mus / (mua + mus)
    mu, wt, h = h_function(omega, nodes)
    h1 = h_at(1.0, omega, mu, wt, h)

    # F[i] = probability that a downward beam at cosine mu[i] eventually leaves the medium.
    # F = A (1 - r) + A diag(r) F with A[i, j] = K(mu_i, mu_j) * wt_j.
    r = fresnel_reflectance(n_medium, n_ambient, mu)
    kernel = omega * mu[None, :] * h[None, :] * h[:, None] / (2.0 * (mu[None, :] + mu[:, None]))
    a = kernel * wt[None, :]
    f = np.linalg.solve(np.eye(nodes) - a * r[None, :], a @ (1.0 - r))

    # Normal incidence from outside refracts to cosine 1 inside the medium.
    k1 = omega * mu * h * h1 / (2.0 * (mu + 1.0))
    escape = float(np.sum(wt * k1 * ((1.0 - r) + r * f)))
    rsp = ((n_medium - n_ambient) / (n_medium + n_ambient)) ** 2
    return SemiInfiniteResult(rsp=rsp, rd=(1.0 - rsp) * escape)
