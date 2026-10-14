"""Batch statistics and the pass/fail rule used to compare against published values."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

# A check passes if |ours - reference| <= Z_LIMIT * combined standard error + rounding allowance.
Z_LIMIT = 3.0


@dataclass(frozen=True)
class Estimate:
    mean: float
    se: float  # nan with fewer than two batches


def batch_estimate(sums: Sequence[float], photons: Sequence[int]) -> Estimate:
    """Mean per-photon value and its standard error from per-batch weight sums.

    Packet weights are not Bernoulli trials, so the standard error comes from the spread of the
    per-batch means. This matches the C++ engine's summary exactly; a test compares the two.
    """
    if len(sums) != len(photons) or not sums:
        raise ValueError("need one photon count per batch and at least one batch")
    mean = math.fsum(sums) / sum(photons)
    n = len(sums)
    if n < 2:
        return Estimate(mean, math.nan)
    per_batch = [s / p for s, p in zip(sums, photons, strict=True)]
    center = math.fsum(per_batch) / n
    ss = math.fsum((x - center) ** 2 for x in per_batch)
    return Estimate(mean, math.sqrt(ss / (n - 1) / n))


@dataclass(frozen=True)
class Comparison:
    ours: float
    se_ours: float
    reference: float
    se_reference: float
    rounding: float  # half the last printed digit of the reference
    z: float
    passed: bool


def compare(
    ours: float,
    se_ours: float,
    reference: float,
    se_reference: float = 0.0,
    rounding: float = 0.0,
) -> Comparison:
    """Compare an estimate with a reference value.

    ``z`` uses the combined standard error. The pass rule adds the rounding allowance, because a
    published value printed to k decimals is only known to within half a unit in the last place.
    """
    se = math.hypot(se_ours, se_reference)
    diff = ours - reference
    z = diff / se if se > 0 else (0.0 if diff == 0 else math.copysign(math.inf, diff))
    passed = abs(diff) <= Z_LIMIT * se + rounding
    return Comparison(ours, se_ours, reference, se_reference, rounding, z, passed)


def estimated_reference_se(se_ours: float, photons_ours: int, photons_reference: int) -> float:
    """Standard error of a published value that came without one.

    Assumes the same per-photon variance as ours, scaled by the number of photons behind each.
    """
    return se_ours * math.sqrt(photons_ours / photons_reference)
