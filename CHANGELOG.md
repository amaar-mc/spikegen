# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Multi-unit population generation.
- An optional NumPy fast path.

## [0.1.0]

### Added
- `regular`, `homogeneous_poisson`, `inhomogeneous_poisson` (thinning), and `gamma_renewal`
  spike-train generators returning plain sorted lists of times, with explicit seeds.
- `with_refractory` to enforce a minimum inter-spike interval.
- Validation with clear errors, and a test suite with exact values, seeded reproducibility,
  structural invariants, and Hypothesis property tests.
