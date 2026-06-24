from hypothesis import given
from hypothesis import strategies as st

from spikegen import gamma_renewal, homogeneous_poisson, inverse_gaussian_renewal

rates = st.floats(min_value=1.0, max_value=200.0, allow_nan=False, allow_infinity=False)
durations = st.floats(min_value=0.0, max_value=3.0, allow_nan=False, allow_infinity=False)
shapes = st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False)
mus = st.floats(min_value=0.005, max_value=1.0, allow_nan=False, allow_infinity=False)
lams = st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False)
seeds = st.integers(min_value=0, max_value=100000)


@given(rates, durations, seeds)
def test_poisson_is_sorted_and_in_range(rate: float, duration: float, seed: int) -> None:
    spikes = homogeneous_poisson(rate=rate, duration=duration, seed=seed)
    assert spikes == sorted(spikes)
    assert all(0.0 <= t < duration for t in spikes)


@given(rates, shapes, durations, seeds)
def test_gamma_is_sorted_and_in_range(
    rate: float, shape: float, duration: float, seed: int
) -> None:
    spikes = gamma_renewal(rate=rate, shape=shape, duration=duration, seed=seed)
    assert spikes == sorted(spikes)
    assert all(0.0 <= t < duration for t in spikes)


@given(mus, lams, durations, seeds)
def test_inverse_gaussian_is_sorted_and_in_range(
    mu: float, lam: float, duration: float, seed: int
) -> None:
    spikes = inverse_gaussian_renewal(mu=mu, lam=lam, duration=duration, seed=seed)
    assert spikes == sorted(spikes)
    assert all(0.0 <= t < duration for t in spikes)
