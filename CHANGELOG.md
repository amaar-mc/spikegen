# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-06-23

### Added
- `inverse_gaussian_renewal(*, mu, lam, duration, seed)`: inverse-Gaussian (Wald) renewal
  process. Inter-spike intervals are i.i.d. inverse-Gaussian `IG(mu, lam)` with mean `mu` and
  variance `mu**3 / lam`, so the squared coefficient of variation is `CV**2 = mu / lam`. Large
  `lam` concentrates the intervals at `mu` (regular spiking), small `lam` gives irregular
  spiking. This is the first-passage-time law of a drift-diffusion (perfect
  integrate-and-fire) neuron, a principled companion to `gamma_renewal`. Intervals are sampled
  with the Michael-Schucany-Haas algorithm (Michael, Schucany, Haas, "Generating Random
  Variates Using Transformations with Multiple Roots", The American Statistician 30(2):88-90,
  1976): draw `y = N(0, 1)**2`, form the smaller root `x`, and return `x` with probability
  `mu / (mu + x)`, else `mu**2 / x`. Implemented in pure Python over the standard library
  `random` (zero runtime dependencies); seeded and reproducible. Tested for reproducibility,
  ISI moment convergence (sample mean to `mu`, variance to `mu**3 / lam`) with tolerances
  derived from the standard error, the empirical `CV**2 = mu / lam` relationship across
  regular and irregular regimes, ordering and range invariants, and validation.

## [0.4.0] - 2026-06-20

### Added
- Optional NumPy fast path behind the `[fast]` extra (`pip install spikegen[fast]`):
  `homogeneous_poisson_numpy(*, rate, duration, seed)`. It draws exponential inter-spike
  intervals in batches and takes their cumulative sum with NumPy, rather than accumulating
  one interval at a time in a Python loop, which is much faster for long, high-rate trains.
  The pure-Python `homogeneous_poisson` remains the zero-dependency default; NumPy is
  imported lazily only inside the fast path. The fast path uses a NumPy `Generator` (PCG64),
  a different random stream from the pure path's `random.Random` (Mersenne Twister), so it is
  not bit-identical to `homogeneous_poisson` for a given seed. It is statistically equivalent:
  both produce a homogeneous Poisson process with the same rate, and their spike counts, mean
  rate, and inter-spike-interval distribution agree. The fast path is itself fully
  reproducible for a fixed seed. Equivalence is enforced by seeded tests with tolerances
  derived from the Poisson count distribution (mean rT, standard deviation sqrt(rT)).

## [0.3.0]

### Added
- `bernoulli(*, rate, duration, dt, seed)`: discrete-time Bernoulli spiking process. Time is
  tiled into bins of width dt; each bin fires a spike at its start time with probability
  `rate * dt`. Raises `ValueError` when `rate * dt > 1`. Seeded and reproducible.
- `jitter(times, *, sigma, seed)`: add independent Gaussian jitter (mean 0, standard
  deviation sigma) to each spike time and return the sorted result. Useful for constructing
  surrogate or null datasets. `sigma = 0` returns the sorted input unchanged.

## [0.2.0]

### Added
- `population(make, *, units, seed)`: generate a population of spike trains by calling `make`
  once per unit with an independent, reproducible seed derived from the base seed. The whole
  population depends only on `(seed, units)`.

## [0.1.0]

### Added
- `regular`, `homogeneous_poisson`, `inhomogeneous_poisson` (thinning), and `gamma_renewal`
  spike-train generators returning plain sorted lists of times, with explicit seeds.
- `with_refractory` to enforce a minimum inter-spike interval.
- Validation with clear errors, and a test suite with exact values, seeded reproducibility,
  structural invariants, and Hypothesis property tests.
