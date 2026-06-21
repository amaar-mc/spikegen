"""NumPy-backed fast path for homogeneous Poisson spike generation.

This module is an OPTIONAL fast path. It is only importable when NumPy is
installed. The public API surface is ``homogeneous_poisson_numpy``, which is
re-exported from the package ``__init__`` when numpy is available via the
``spikegen[fast]`` extra.

The pure-Python ``homogeneous_poisson`` remains the reference implementation
and the default when NumPy is not installed.

Equivalence note: the fast path draws exponential inter-spike intervals from a
NumPy ``Generator`` (PCG64), which is a different random stream from the pure
path's ``random.Random`` (Mersenne Twister). The two are therefore NOT
bit-identical for a given seed. They are statistically equivalent: both produce
a homogeneous Poisson process with the same rate, so spike counts, mean rate,
and the inter-spike-interval distribution agree in distribution. The fast path
is itself fully reproducible for a fixed seed.
"""

from __future__ import annotations

import numpy as np

from ._validate import check_duration, check_rate, check_seed


def homogeneous_poisson_numpy(*, rate: float, duration: float, seed: int) -> list[float]:
    """Homogeneous Poisson process on [0, duration) with the given rate, using NumPy.

    Equivalent in distribution to ``homogeneous_poisson`` but vectorized: it draws
    exponential inter-spike intervals in batches with NumPy and takes their cumulative
    sum, rather than accumulating one interval at a time in a Python loop. This is much
    faster for long, high-rate trains where the pure path spends most of its time in the
    interpreter loop.

    The result is the sorted list of spike times strictly less than ``duration``. The
    fast path uses a NumPy ``Generator`` seeded by ``seed`` and is fully reproducible,
    but it is NOT bit-identical to the pure-Python ``homogeneous_poisson`` for the same
    seed because NumPy's RNG stream differs from ``random.Random``. The two agree
    statistically (same rate, count distribution, and inter-spike-interval law).

    Requires NumPy (install with ``pip install spikegen[fast]``).

    Args:
        rate: positive spike rate; mean inter-spike interval is 1 / rate.
        duration: non-negative length of the interval [0, duration).
        seed: integer seed for the NumPy generator; fixes the output.

    Returns:
        A sorted list of spike times, each in [0, duration).

    Raises:
        ValueError: if rate is not positive, duration is negative, or seed is not an int.
    """
    check_rate(rate)
    check_duration(duration)
    check_seed(seed)
    if duration == 0.0:
        return []
    rng = np.random.default_rng(seed)
    times: list[float] = []
    carry = 0.0
    while True:
        # Draw enough intervals to very likely cross the remaining duration. The
        # remaining expected count is rate * (duration - carry); a 1.5x margin plus a
        # fixed pad of 64 makes a refill rare, and the loop handles the rare shortfall.
        remaining = duration - carry
        batch = int(rate * remaining * 1.5) + 64
        intervals = rng.exponential(scale=1.0 / rate, size=batch)
        cumulative = carry + np.cumsum(intervals)
        within = cumulative[cumulative < duration]
        times.extend(within.tolist())
        if cumulative.size > within.size:
            # At least one drawn time landed at or beyond duration, so we are done.
            break
        # Every drawn interval stayed inside duration; continue from the last time.
        carry = float(cumulative[-1])
    return times
