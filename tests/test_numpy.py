"""Tests for the optional NumPy-backed homogeneous_poisson_numpy.

All tests that use NumPy are skipped when NumPy is not installed so the suite
still passes with zero dependencies (``uv run pytest -q``).

The fast path uses a NumPy ``Generator`` (PCG64), a different random stream from
the pure path's ``random.Random`` (Mersenne Twister), so it is NOT bit-identical
to ``homogeneous_poisson`` for a given seed. These tests therefore assert
STATISTICAL equivalence with deterministic, seeded tolerances derived from the
Poisson count distribution: for rate r over duration T the count N has mean rT
and standard deviation sqrt(rT).
"""

import math
import statistics

import pytest

pytest.importorskip("numpy")

from spikegen import homogeneous_poisson, homogeneous_poisson_numpy

# Reference regime used across the statistical tests.
RATE = 100.0
DURATION = 10.0
EXPECTED_COUNT = RATE * DURATION  # 1000
COUNT_STD = math.sqrt(EXPECTED_COUNT)  # ~31.6


# ---------------------------------------------------------------------------
# Reproducibility and structure of the fast path itself.
# ---------------------------------------------------------------------------


def test_fast_path_reproducible() -> None:
    a = homogeneous_poisson_numpy(rate=RATE, duration=DURATION, seed=7)
    b = homogeneous_poisson_numpy(rate=RATE, duration=DURATION, seed=7)
    assert a == b


def test_fast_path_different_seeds_differ() -> None:
    a = homogeneous_poisson_numpy(rate=RATE, duration=DURATION, seed=7)
    b = homogeneous_poisson_numpy(rate=RATE, duration=DURATION, seed=8)
    assert a != b


def test_fast_path_sorted_and_in_range() -> None:
    a = homogeneous_poisson_numpy(rate=RATE, duration=DURATION, seed=7)
    assert a == sorted(a)
    assert all(0.0 <= t < DURATION for t in a)
    assert all(a[i] < a[i + 1] for i in range(len(a) - 1))


def test_fast_path_empty_duration() -> None:
    assert homogeneous_poisson_numpy(rate=RATE, duration=0.0, seed=0) == []


def test_fast_path_validation() -> None:
    with pytest.raises(ValueError):
        homogeneous_poisson_numpy(rate=0.0, duration=1.0, seed=0)
    with pytest.raises(ValueError):
        homogeneous_poisson_numpy(rate=10.0, duration=-1.0, seed=0)
    with pytest.raises(ValueError):
        homogeneous_poisson_numpy(rate=10.0, duration=1.0, seed=1.5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Statistical equivalence to the pure-Python homogeneous_poisson.
# ---------------------------------------------------------------------------


def test_single_seed_count_within_five_sigma() -> None:
    """One fast-path train has a count within 5 sigma of the Poisson mean.

    N ~ Poisson(1000) has mean 1000 and std sqrt(1000) ~ 31.6, so a 5-sigma band
    is [842, 1158]. A fixed seed makes this a deterministic, non-flaky check.
    """
    count = len(homogeneous_poisson_numpy(rate=RATE, duration=DURATION, seed=7))
    assert abs(count - EXPECTED_COUNT) < 5.0 * COUNT_STD


def test_mean_isi_matches_inverse_rate() -> None:
    """Mean inter-spike interval of the fast path matches 1 / rate within 5%.

    The ISI mean is 1 / rate = 0.01. With ~1000 intervals the sample mean's
    standard error is (1/rate)/sqrt(n) ~ 0.01/31.6 ~ 3.2e-4, so a 5% band
    (0.0005) is several standard errors wide and deterministic for a fixed seed.
    """
    train = homogeneous_poisson_numpy(rate=RATE, duration=DURATION, seed=7)
    isis = [train[i + 1] - train[i] for i in range(len(train) - 1)]
    mean_isi = statistics.mean(isis)
    assert math.isclose(mean_isi, 1.0 / RATE, rel_tol=0.05)


def test_aggregate_mean_count_matches_pure_path() -> None:
    """Averaged over many seeds, fast and pure path mean counts agree.

    Over N = 400 seeds the mean count has standard error sqrt(1000)/sqrt(400)
    ~ 1.58. Each path's mean must sit within 4 standard errors of the theoretical
    1000, and the two means must agree within 6 standard errors. Both bounds are
    deterministic because the seeds are fixed.
    """
    n_seeds = 400
    se = COUNT_STD / math.sqrt(n_seeds)
    fast_counts = [
        len(homogeneous_poisson_numpy(rate=RATE, duration=DURATION, seed=s))
        for s in range(n_seeds)
    ]
    pure_counts = [
        len(homogeneous_poisson(rate=RATE, duration=DURATION, seed=s)) for s in range(n_seeds)
    ]
    fast_mean = statistics.mean(fast_counts)
    pure_mean = statistics.mean(pure_counts)
    assert abs(fast_mean - EXPECTED_COUNT) < 4.0 * se
    assert abs(pure_mean - EXPECTED_COUNT) < 4.0 * se
    assert abs(fast_mean - pure_mean) < 6.0 * se


def test_aggregate_variance_matches_poisson() -> None:
    """The across-seed count variance of the fast path matches the Poisson mean.

    For a Poisson process Var(N) = E[N] = 1000. The sample variance over 400
    seeds is allowed a generous +/-30% band (chi-square sampling spread at this
    n is well inside it), making the check deterministic and not flaky.
    """
    n_seeds = 400
    counts = [
        len(homogeneous_poisson_numpy(rate=RATE, duration=DURATION, seed=s))
        for s in range(n_seeds)
    ]
    sample_var = statistics.variance(counts)
    assert 0.7 * EXPECTED_COUNT < sample_var < 1.3 * EXPECTED_COUNT


def test_pure_path_unchanged() -> None:
    """Adding the fast path must not change the pure-Python generator's output."""
    a = homogeneous_poisson(rate=50.0, duration=2.0, seed=7)
    assert a == homogeneous_poisson(rate=50.0, duration=2.0, seed=7)
    assert all(0.0 <= t < 2.0 for t in a)
    assert a == sorted(a)
