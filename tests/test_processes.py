import pytest

from spikegen import (
    gamma_renewal,
    homogeneous_poisson,
    inhomogeneous_poisson,
    regular,
    with_refractory,
)


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
        with_refractory([0.0], refractory=-0.1)
