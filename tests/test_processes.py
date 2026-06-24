import math

import pytest

from spikegen import (
    gamma_renewal,
    homogeneous_poisson,
    inhomogeneous_poisson,
    inverse_gaussian_renewal,
    lognormal_renewal,
    regular,
    with_refractory,
)


def _intervals(times: list[float]) -> list[float]:
    """Inter-spike intervals of a train, including the first interval from 0."""
    out: list[float] = []
    prev = 0.0
    for t in times:
        out.append(t - prev)
        prev = t
    return out


def test_regular_exact() -> None:
    assert regular(rate=2.0, duration=1.0) == [0.0, 0.5]
    assert regular(rate=1.0, duration=3.0) == [0.0, 1.0, 2.0]
    assert regular(rate=10.0, duration=0.0) == []


def test_with_refractory_exact() -> None:
    assert with_refractory([0.0, 0.1, 0.15, 0.3], refractory=0.12) == [0.0, 0.15, 0.3]
    assert with_refractory([0.3, 0.0, 0.1], refractory=0.05) == [0.0, 0.1, 0.3]


def test_homogeneous_reproducible_and_structural() -> None:
    a = homogeneous_poisson(rate=50.0, duration=2.0, seed=7)
    assert a == homogeneous_poisson(rate=50.0, duration=2.0, seed=7)
    assert homogeneous_poisson(rate=50.0, duration=2.0, seed=8) != a
    assert all(0.0 <= t < 2.0 for t in a)
    assert a == sorted(a)
    assert all(a[i] < a[i + 1] for i in range(len(a) - 1))


def test_gamma_reproducible_and_in_range() -> None:
    g = gamma_renewal(rate=20.0, shape=2.0, duration=5.0, seed=3)
    assert g == gamma_renewal(rate=20.0, shape=2.0, duration=5.0, seed=3)
    assert all(0.0 <= t < 5.0 for t in g)
    assert g == sorted(g)


def test_inverse_gaussian_reproducible_and_in_range() -> None:
    a = inverse_gaussian_renewal(mu=0.05, lam=0.2, duration=5.0, seed=4)
    assert a == inverse_gaussian_renewal(mu=0.05, lam=0.2, duration=5.0, seed=4)
    assert inverse_gaussian_renewal(mu=0.05, lam=0.2, duration=5.0, seed=5) != a
    assert all(0.0 <= t < 5.0 for t in a)
    assert a == sorted(a)
    assert all(a[i] < a[i + 1] for i in range(len(a) - 1))
    assert all(math.isfinite(t) for t in a)


def test_inverse_gaussian_isi_moments_converge() -> None:
    # Long run so the ISI sample mean approaches mu and variance approaches mu**3 / lam.
    # mu is the mean interval; over a duration D the expected count is about D / mu, so a
    # very long duration gives a large sample. Tolerances are set from the standard error of
    # the mean (sd / sqrt(n)) with a generous multiplier, not bit-exact.
    mu = 0.02
    lam = 0.05
    duration = 4000.0  # about 200000 intervals at mu = 0.02
    spikes = inverse_gaussian_renewal(mu=mu, lam=lam, duration=duration, seed=11)
    isis = _intervals(spikes)[:-1]  # drop the final interval that overran the duration
    n = len(isis)
    assert n > 50000
    mean = sum(isis) / n
    var = sum((x - mean) ** 2 for x in isis) / n
    expected_var = mu**3 / lam
    sd = math.sqrt(expected_var)
    # Standard error of the mean; 6 sigma is a very safe band for a clean estimator.
    sem = sd / math.sqrt(n)
    assert abs(mean - mu) < 6.0 * sem
    # Variance of an IG estimator converges more slowly; allow 8 percent relative error.
    assert abs(var - expected_var) / expected_var < 0.08


def test_inverse_gaussian_cv_squared_matches_mu_over_lam() -> None:
    # CV**2 = mu / lam. Large lam -> low CV (regular); small lam -> high CV. Assert the
    # relationship empirically within tolerance for two regimes.
    mu = 0.02
    duration = 4000.0

    def cv_squared(lam: float, seed: int) -> float:
        spikes = inverse_gaussian_renewal(mu=mu, lam=lam, duration=duration, seed=seed)
        isis = _intervals(spikes)[:-1]
        n = len(isis)
        mean = sum(isis) / n
        var = sum((x - mean) ** 2 for x in isis) / n
        return var / (mean * mean)

    lam_large = 2.0  # CV**2 = 0.01, CV = 0.1 (regular)
    cv2_large = cv_squared(lam_large, seed=21)
    assert cv2_large < 0.02
    assert abs(cv2_large - mu / lam_large) / (mu / lam_large) < 0.1

    lam_small = 0.01  # CV**2 = 2.0 (very irregular)
    cv2_small = cv_squared(lam_small, seed=22)
    assert cv2_small > 1.0
    assert abs(cv2_small - mu / lam_small) / (mu / lam_small) < 0.1


def test_lognormal_reproducible_and_in_range() -> None:
    a = lognormal_renewal(mean=0.05, cv=0.5, duration=5.0, seed=4)
    assert a == lognormal_renewal(mean=0.05, cv=0.5, duration=5.0, seed=4)
    assert lognormal_renewal(mean=0.05, cv=0.5, duration=5.0, seed=5) != a
    assert all(0.0 <= t < 5.0 for t in a)
    assert a == sorted(a)
    assert all(a[i] < a[i + 1] for i in range(len(a) - 1))
    assert all(math.isfinite(t) for t in a)


def test_lognormal_isi_mean_and_cv_converge() -> None:
    # Lognormal ISIs are parameterized by mean and cv directly, so the sample mean approaches
    # mean and the sample CV approaches cv over a long run. Tolerances are set from the
    # standard error of the mean (sd / sqrt(n)) with a generous multiplier, not bit-exact.
    mean = 0.02
    cv = 0.6
    duration = 4000.0  # about 200000 intervals at mean = 0.02
    spikes = lognormal_renewal(mean=mean, cv=cv, duration=duration, seed=11)
    isis = _intervals(spikes)[:-1]  # drop the final interval that overran the duration
    n = len(isis)
    assert n > 50000
    sample_mean = sum(isis) / n
    var = sum((x - sample_mean) ** 2 for x in isis) / n
    sd = cv * mean  # true ISI standard deviation
    # Standard error of the mean; 6 sigma is a very safe band for a clean estimator.
    sem = sd / math.sqrt(n)
    assert abs(sample_mean - mean) < 6.0 * sem
    sample_cv = math.sqrt(var) / sample_mean
    # CV converges more slowly than the mean; allow 5 percent relative error.
    assert abs(sample_cv - cv) / cv < 0.05


def test_lognormal_cv_tracks_target_across_regimes() -> None:
    # Small cv gives nearly regular spiking (low empirical CV); large cv gives bursty,
    # irregular spiking (high empirical CV). Assert the monotone relationship and that the
    # empirical CV tracks the target across two regimes.
    mean = 0.02
    duration = 4000.0

    def empirical_cv(cv: float, seed: int) -> float:
        spikes = lognormal_renewal(mean=mean, cv=cv, duration=duration, seed=seed)
        isis = _intervals(spikes)[:-1]
        n = len(isis)
        sample_mean = sum(isis) / n
        var = sum((x - sample_mean) ** 2 for x in isis) / n
        return math.sqrt(var) / sample_mean

    cv_small = empirical_cv(0.1, seed=21)  # nearly regular
    assert cv_small < 0.2
    assert abs(cv_small - 0.1) / 0.1 < 0.05

    cv_large = empirical_cv(1.5, seed=22)  # bursty, irregular
    assert cv_large > 1.0
    assert abs(cv_large - 1.5) / 1.5 < 0.1

    assert cv_large > cv_small


def test_inhomogeneous_zero_rate_gives_no_spikes() -> None:
    assert inhomogeneous_poisson(rate_fn=lambda t: 0.0, max_rate=100.0, duration=1.0, seed=1) == []


def test_inhomogeneous_reproducible_and_in_range() -> None:
    def rate_fn(t: float) -> float:
        return 50.0 if t < 0.5 else 5.0

    a = inhomogeneous_poisson(rate_fn=rate_fn, max_rate=50.0, duration=1.0, seed=2)
    assert a == inhomogeneous_poisson(rate_fn=rate_fn, max_rate=50.0, duration=1.0, seed=2)
    assert all(0.0 <= t < 1.0 for t in a)


def test_inhomogeneous_rejects_out_of_range_rate() -> None:
    with pytest.raises(ValueError):
        inhomogeneous_poisson(rate_fn=lambda t: 200.0, max_rate=50.0, duration=1.0, seed=1)


def test_validation() -> None:
    with pytest.raises(ValueError):
        regular(rate=0.0, duration=1.0)
    with pytest.raises(ValueError):
        homogeneous_poisson(rate=10.0, duration=-1.0, seed=1)
    with pytest.raises(ValueError):
        gamma_renewal(rate=10.0, shape=0.0, duration=1.0, seed=1)
    with pytest.raises(ValueError):
        inverse_gaussian_renewal(mu=0.0, lam=1.0, duration=1.0, seed=1)
    with pytest.raises(ValueError):
        inverse_gaussian_renewal(mu=-1.0, lam=1.0, duration=1.0, seed=1)
    with pytest.raises(ValueError):
        inverse_gaussian_renewal(mu=1.0, lam=0.0, duration=1.0, seed=1)
    with pytest.raises(ValueError):
        inverse_gaussian_renewal(mu=1.0, lam=1.0, duration=-1.0, seed=1)
    with pytest.raises(ValueError):
        inverse_gaussian_renewal(mu=1.0, lam=1.0, duration=1.0, seed=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        lognormal_renewal(mean=0.0, cv=0.5, duration=1.0, seed=1)
    with pytest.raises(ValueError):
        lognormal_renewal(mean=-1.0, cv=0.5, duration=1.0, seed=1)
    with pytest.raises(ValueError):
        lognormal_renewal(mean=1.0, cv=0.0, duration=1.0, seed=1)
    with pytest.raises(ValueError):
        lognormal_renewal(mean=1.0, cv=0.5, duration=-1.0, seed=1)
    with pytest.raises(ValueError):
        lognormal_renewal(mean=1.0, cv=0.5, duration=1.0, seed=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        with_refractory([0.0], refractory=-0.1)
