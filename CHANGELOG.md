# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
