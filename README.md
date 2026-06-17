# spikegen

<p align="center">
  <img src="assets/logo.png" alt="spikegen logo" width="160">
</p>

[![PyPI](https://img.shields.io/pypi/v/spikegen)](https://pypi.org/project/spikegen/)
[![CI](https://github.com/amaar-mc/spikegen/actions/workflows/ci.yml/badge.svg)](https://github.com/amaar-mc/spikegen/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

Generate spike trains in pure Python with zero dependencies. Poisson, gamma renewal, regular, and inhomogeneous processes, returned as plain sorted lists of spike times, with explicit seeds for reproducibility.

## Install

```sh
pip install spikegen
```

## 30-second example

```python
from spikegen import homogeneous_poisson, regular, gamma_renewal, with_refractory

homogeneous_poisson(rate=50.0, duration=2.0, seed=0)   # Poisson spikes in [0, 2)
regular(rate=10.0, duration=1.0)                        # [0.0, 0.1, 0.2, ...]
gamma_renewal(rate=20.0, shape=2.0, duration=1.0, seed=0)  # more regular than Poisson

spikes = homogeneous_poisson(rate=80.0, duration=1.0, seed=0)
with_refractory(spikes, refractory=0.002)              # enforce a 2 ms refractory period
```

Times are in the same units as `1 / rate` (seconds if rate is in Hz). Seeded processes are
reproducible: the same seed gives the same train.

## Why this exists

Generating synthetic spike trains is a daily need, but the generators live inside heavy
frameworks: `elephant` requires neo and quantities, `pyspike` is NumPy-based, and other
options are old or partial. `spikegen` is a small, dependency-free generator that returns
plain lists of floats, so reproducible spike trains are one import away. It pairs with
[spikedist](https://pypi.org/project/spikedist/): generate trains, then measure the
distance between them.

## Processes

- `regular(rate, duration)`: evenly spaced spikes. Deterministic.
- `homogeneous_poisson(rate, duration, seed)`: constant-rate Poisson process.
- `inhomogeneous_poisson(rate_fn, max_rate, duration, seed)`: time-varying rate by thinning.
- `gamma_renewal(rate, shape, duration, seed)`: gamma inter-spike intervals; shape 1 is
  Poisson, larger shape is more regular.
- `with_refractory(times, refractory)`: drop spikes within a minimum interval.

All parameters are keyword-only and explicit.

## Testing

```sh
pip install -e ".[dev]"
pytest
```

Tests cover exact values for the deterministic generators, seeded reproducibility, the
rate-bound and ordering invariants, and the validation paths, with property tests via
Hypothesis.

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](./CONTRIBUTING.md).

## License

MIT. See [LICENSE](./LICENSE).
