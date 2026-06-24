import math
import random
from collections.abc import Callable, Sequence

from ._validate import (
    check_duration,
    check_non_negative,
    check_positive,
    check_rate,
    check_seed,
    check_times_finite,
)


def regular(*, rate: float, duration: float) -> list[float]:
    """Evenly spaced spikes at 0, 1/rate, 2/rate, ... within [0, duration). Deterministic.
    Indices are multiplied by the step rather than accumulated to avoid floating drift."""
    check_rate(rate)
    check_duration(duration)
    step = 1.0 / rate
    times: list[float] = []
    i = 0
    t = 0.0
    while t < duration:
        times.append(t)
        i += 1
        t = i * step
    return times


def homogeneous_poisson(*, rate: float, duration: float, seed: int) -> list[float]:
    """Homogeneous Poisson process on [0, duration) with the given rate, drawing
    exponential inter-spike intervals. Seeded for reproducibility."""
    check_rate(rate)
    check_duration(duration)
    check_seed(seed)
    rng = random.Random(seed)
    times: list[float] = []
    t = 0.0
    while True:
        # 1 - U avoids log(0) when U happens to be 0.
        t += -math.log(1.0 - rng.random()) / rate
        if t >= duration:
            break
        times.append(t)
    return times


def inhomogeneous_poisson(
    *, rate_fn: Callable[[float], float], max_rate: float, duration: float, seed: int
) -> list[float]:
    """Inhomogeneous Poisson process by thinning: propose spikes at max_rate and keep each
    at time t with probability rate_fn(t) / max_rate. rate_fn must return a value in
    [0, max_rate] for every t in [0, duration). Seeded for reproducibility."""
    check_positive("max_rate", max_rate)
    check_duration(duration)
    check_seed(seed)
    rng = random.Random(seed)
    times: list[float] = []
    t = 0.0
    while True:
        t += -math.log(1.0 - rng.random()) / max_rate
        if t >= duration:
            break
        rate = rate_fn(t)
        if rate < 0.0 or rate > max_rate:
            raise ValueError(f"rate_fn returned {rate!r} at t={t!r}, outside [0, max_rate]")
        if rng.random() < rate / max_rate:
            times.append(t)
    return times


def gamma_renewal(*, rate: float, shape: float, duration: float, seed: int) -> list[float]:
    """Gamma renewal process: inter-spike intervals follow a gamma distribution with the
    given shape and a mean of 1/rate. shape = 1 reduces to a Poisson process; larger shape
    gives more regular spiking. Seeded for reproducibility."""
    check_rate(rate)
    check_positive("shape", shape)
    check_duration(duration)
    check_seed(seed)
    rng = random.Random(seed)
    scale = 1.0 / (rate * shape)  # mean interval = shape * scale = 1 / rate
    times: list[float] = []
    t = 0.0
    while True:
        t += rng.gammavariate(shape, scale)
        if t >= duration:
            break
        times.append(t)
    return times


def inverse_gaussian_renewal(*, mu: float, lam: float, duration: float, seed: int) -> list[float]:
    """Inverse-Gaussian (Wald) renewal process: inter-spike intervals are i.i.d.
    inverse-Gaussian IG(mu, lam) with mean mu and shape lam, so the interval mean is mu and
    its variance is mu**3 / lam. This is the first-passage-time law of a drift-diffusion
    (perfect integrate-and-fire) neuron, making it a principled companion to gamma_renewal.
    As lam -> infinity the intervals concentrate at mu (regular spiking, low CV); the squared
    coefficient of variation is CV**2 = mu / lam. Seeded for reproducibility.

    Intervals are sampled with the Michael-Schucany-Haas algorithm (Michael, Schucany, Haas,
    "Generating Random Variates Using Transformations with Multiple Roots", The American
    Statistician 30(2):88-90, 1976): draw y = N(0, 1)**2, form
    x = mu + (mu**2 * y) / (2 * lam) - (mu / (2 * lam)) * sqrt(4 * mu * lam * y + mu**2 * y**2),
    then return x with probability mu / (mu + x), else mu**2 / x."""
    check_positive("mu", mu)
    check_positive("lam", lam)
    check_duration(duration)
    check_seed(seed)
    rng = random.Random(seed)
    times: list[float] = []
    t = 0.0
    while True:
        n = rng.gauss(0.0, 1.0)
        y = n * n
        x = mu + (mu * mu * y) / (2.0 * lam) - (mu / (2.0 * lam)) * math.sqrt(
            4.0 * mu * lam * y + mu * mu * y * y
        )
        # Selection step: pick the smaller root x with probability mu / (mu + x).
        interval = x if rng.random() <= mu / (mu + x) else mu * mu / x
        t += interval
        if t >= duration:
            break
        times.append(t)
    return times


def lognormal_renewal(*, mean: float, cv: float, duration: float, seed: int) -> list[float]:
    """Lognormal renewal process: inter-spike intervals are i.i.d. lognormal, the common
    empirical fit for cortical ISI distributions and a natural companion to gamma_renewal and
    inverse_gaussian_renewal.

    Parameterized by the ISI mean and coefficient of variation directly (the
    neuroscience-friendly form), not by the underlying normal's parameters. Given a target ISI
    mean and cv, the underlying normal N(mu, sigma**2) is recovered from
    sigma**2 = ln(1 + cv**2) and mu = ln(mean) - sigma**2 / 2, and each interval is
    exp(mu + sigma * Z) with Z ~ N(0, 1) (via random.gauss). This yields E[ISI] = mean and
    CV[ISI] = cv exactly in expectation. The lognormal CV depends only on sigma:
    CV = sqrt(exp(sigma**2) - 1), so small cv gives nearly regular spiking and large cv gives
    bursty, irregular spiking. Seeded for reproducibility."""
    check_positive("mean", mean)
    check_positive("cv", cv)
    check_duration(duration)
    check_seed(seed)
    rng = random.Random(seed)
    sigma2 = math.log(1.0 + cv * cv)
    sigma = math.sqrt(sigma2)
    mu = math.log(mean) - sigma2 / 2.0
    times: list[float] = []
    t = 0.0
    while True:
        t += math.exp(mu + sigma * rng.gauss(0.0, 1.0))
        if t >= duration:
            break
        times.append(t)
    return times


def with_refractory(times: Sequence[float], *, refractory: float) -> list[float]:
    """Enforce a minimum inter-spike interval by dropping each spike that falls within
    refractory of the previously kept spike. Inputs are sorted first."""
    check_non_negative("refractory", refractory)
    kept: list[float] = []
    last: float | None = None
    for t in sorted(times):
        if last is None or t - last >= refractory:
            kept.append(t)
            last = t
    return kept


def bernoulli(*, rate: float, duration: float, dt: float, seed: int) -> list[float]:
    """Discrete-time Bernoulli spiking process on [0, duration).

    Time is divided into bins of width dt. Each bin starting at time t fires a spike
    with probability p = rate * dt. The spike time is the bin's start (left edge).

    rate * dt must be in [0, 1]; raising ValueError if rate * dt > 1 since the
    per-bin probability would be invalid."""
    check_rate(rate)
    check_duration(duration)
    check_positive("dt", dt)
    check_seed(seed)
    p = rate * dt
    if p > 1.0:
        raise ValueError(
            f"rate * dt = {p!r} exceeds 1.0; reduce rate or dt so each bin probability"
            f" is valid (rate={rate!r}, dt={dt!r})"
        )
    rng = random.Random(seed)
    times: list[float] = []
    n_bins = int(duration / dt)
    for i in range(n_bins):
        if rng.random() < p:
            times.append(i * dt)
    return times


def jitter(times: Sequence[float], *, sigma: float, seed: int) -> list[float]:
    """Add independent Gaussian jitter to each spike time and return the sorted result.

    Each spike time t becomes t + N(0, sigma^2). sigma = 0 returns the input sorted
    unchanged. Useful for constructing surrogate or null datasets by destroying precise
    spike timing while preserving the overall count."""
    check_non_negative("sigma", sigma)
    check_seed(seed)
    check_times_finite(times)
    if sigma == 0.0:
        return sorted(times)
    rng = random.Random(seed)
    return sorted(t + rng.gauss(0.0, sigma) for t in times)
