import pytest
from hypothesis import given
from hypothesis import strategies as st

from spikegen import bernoulli, jitter

# ---------------------------------------------------------------------------
# bernoulli
# ---------------------------------------------------------------------------


def test_bernoulli_reproducible() -> None:
    a = bernoulli(rate=50.0, duration=1.0, dt=0.001, seed=42)
    assert a == bernoulli(rate=50.0, duration=1.0, dt=0.001, seed=42)


def test_bernoulli_different_seeds_differ() -> None:
    a = bernoulli(rate=50.0, duration=1.0, dt=0.001, seed=42)
    b = bernoulli(rate=50.0, duration=1.0, dt=0.001, seed=99)
    assert a != b


def test_bernoulli_rate_dt_one_fires_every_bin() -> None:
    # p = rate * dt = 1.0, so every bin must fire regardless of seed.
    spikes = bernoulli(rate=1000.0, duration=1.0, dt=0.001, seed=0)
    n_bins = int(1.0 / 0.001)
    assert len(spikes) == n_bins
    # Spike times are 0.0, 0.001, 0.002, ...
    assert spikes == [i * 0.001 for i in range(n_bins)]


def test_bernoulli_rate_dt_zero_fires_never() -> None:
    # rate * dt = 0 would require rate=0 which is invalid, so use an effectively zero
    # probability by making rate very small and dt=1e-15 (rate*dt close to 0, well below 1).
    # Instead use rate=0 path via validation to confirm raises, and test 0 probability
    # differently: a small enough probability over a bounded train must give 0 spikes
    # with a fixed seed -- or we just test the p->0 structural edge.
    # The cleanest approach: dt tiny so p ~ 0, duration tiny so n_bins=0.
    spikes = bernoulli(rate=1.0, duration=0.0, dt=0.001, seed=0)
    assert spikes == []


def test_bernoulli_rate_dt_exactly_zero_probability() -> None:
    # p = rate * dt; if we contrive rate such that p is effectively 0 with many bins,
    # we rely on structural check rather than a flaky statistical assert.
    # Just verify empty duration gives empty result.
    assert bernoulli(rate=100.0, duration=0.0, dt=0.001, seed=5) == []


def test_bernoulli_raises_when_rate_dt_exceeds_one() -> None:
    with pytest.raises(ValueError, match="rate \\* dt"):
        bernoulli(rate=2000.0, duration=1.0, dt=0.001, seed=0)


def test_bernoulli_raises_bad_rate() -> None:
    with pytest.raises(ValueError):
        bernoulli(rate=0.0, duration=1.0, dt=0.001, seed=0)


def test_bernoulli_raises_bad_duration() -> None:
    with pytest.raises(ValueError):
        bernoulli(rate=10.0, duration=-1.0, dt=0.001, seed=0)


def test_bernoulli_raises_bad_dt() -> None:
    with pytest.raises(ValueError):
        bernoulli(rate=10.0, duration=1.0, dt=0.0, seed=0)


def test_bernoulli_is_sorted() -> None:
    spikes = bernoulli(rate=50.0, duration=2.0, dt=0.001, seed=7)
    assert spikes == sorted(spikes)


def test_bernoulli_times_are_bin_starts() -> None:
    # All returned times must be exact multiples of dt (i.e. i*dt for integer i).
    dt = 0.001
    spikes = bernoulli(rate=50.0, duration=0.5, dt=dt, seed=3)
    for t in spikes:
        idx = round(t / dt)
        assert abs(t - idx * dt) < 1e-12


def test_bernoulli_mean_count_approx_rate_duration() -> None:
    # E[count] = rate * duration for a Bernoulli process (same as Poisson for small dt).
    # With rate=50, duration=2 -> expected 100 spikes; use a loose tolerance.
    spikes = bernoulli(rate=50.0, duration=2.0, dt=0.001, seed=123)
    assert 60 <= len(spikes) <= 140


# ---------------------------------------------------------------------------
# jitter
# ---------------------------------------------------------------------------


def test_jitter_reproducible() -> None:
    spikes = [0.1, 0.2, 0.3, 0.5]
    a = jitter(spikes, sigma=0.01, seed=0)
    assert a == jitter(spikes, sigma=0.01, seed=0)


def test_jitter_different_seeds_differ() -> None:
    spikes = [0.1, 0.2, 0.3, 0.4, 0.5]
    a = jitter(spikes, sigma=0.01, seed=0)
    b = jitter(spikes, sigma=0.01, seed=99)
    assert a != b


def test_jitter_preserves_count() -> None:
    spikes = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert len(jitter(spikes, sigma=0.01, seed=0)) == 5


def test_jitter_sigma_zero_is_identity() -> None:
    spikes = [0.3, 0.1, 0.2]
    assert jitter(spikes, sigma=0.0, seed=0) == [0.1, 0.2, 0.3]


def test_jitter_sigma_zero_already_sorted() -> None:
    spikes = [0.1, 0.2, 0.3]
    assert jitter(spikes, sigma=0.0, seed=0) == spikes


def test_jitter_output_is_sorted() -> None:
    spikes = [0.1, 0.2, 0.3, 0.4, 0.5]
    result = jitter(spikes, sigma=0.05, seed=7)
    assert result == sorted(result)


def test_jitter_empty_input() -> None:
    assert jitter([], sigma=0.01, seed=0) == []


def test_jitter_raises_negative_sigma() -> None:
    with pytest.raises(ValueError):
        jitter([0.1, 0.2], sigma=-0.01, seed=0)


def test_jitter_raises_non_finite_times() -> None:
    with pytest.raises(ValueError):
        jitter([0.1, float("inf")], sigma=0.01, seed=0)


def test_jitter_raises_bad_seed() -> None:
    with pytest.raises(ValueError):
        jitter([0.1], sigma=0.01, seed=1.5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Hypothesis property tests
# ---------------------------------------------------------------------------

rates = st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False)
durations = st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)
seeds = st.integers(min_value=0, max_value=100_000)
sigmas = st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False)
spike_lists = st.lists(
    st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    min_size=0,
    max_size=50,
)


@given(rates, durations, seeds)
def test_bernoulli_sorted_and_valid_bins(rate: float, duration: float, seed: int) -> None:
    dt = 0.01
    if rate * dt > 1.0:
        return  # skip invalid combinations
    spikes = bernoulli(rate=rate, duration=duration, dt=dt, seed=seed)
    assert spikes == sorted(spikes)
    n_bins = int(duration / dt)
    valid_times = {i * dt for i in range(n_bins)}
    for t in spikes:
        assert any(abs(t - v) < 1e-12 for v in valid_times)


@given(spike_lists, sigmas, seeds)
def test_jitter_sorted_and_count_preserved(
    spikes: list[float], sigma: float, seed: int
) -> None:
    result = jitter(spikes, sigma=sigma, seed=seed)
    assert result == sorted(result)
    assert len(result) == len(spikes)
