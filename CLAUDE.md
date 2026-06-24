# spikegen

Pure-Python spike-train generators. Zero runtime dependencies. Complements spikedist.

## Commands

- Create env and install: `uv venv && uv pip install -e ".[dev]"`
- Test: `uv run pytest -q`
- Lint: `uv run ruff check .` (format with `uv run ruff format .`)
- Types: `uv run mypy src`
- Build: `uv build` (then `uv run --with twine twine check dist/*` before publishing)

## Architecture

`src/spikegen/`:
- `_validate.py` shared validation (rate, duration, seed, positivity)
- `processes.py` the generators (regular, homogeneous and inhomogeneous Poisson, gamma
  renewal, inverse-Gaussian renewal) plus with_refractory
- `__init__.py` public surface

See `docs/architecture.md` for the algorithms.

## Conventions

- Generators return plain sorted `list[float]` of spike times.
- Stochastic generators take an explicit integer `seed` and are reproducible via
  `random.Random(seed)`.
- Parameters are keyword-only; no default values.
- Use `1 - random()` inside the exponential-interval log to avoid log(0).
- No runtime dependencies (standard library `random` and `math` only).

## Testing rules

- Exact values for the deterministic generators (regular, with_refractory).
- Seeded reproducibility for the stochastic generators.
- Structural invariants (sorted, within [0, duration)).
- Hypothesis property tests; bug fixes start with a failing test.

## Release

- Semantic versioning; update `CHANGELOG.md` and `__version__`.
- Gates: `uv run pytest && uv run ruff check . && uv run mypy src && uv build && uv run twine check dist/*`.
- Publish to PyPI, tag `vX.Y.Z`, GitHub release.

## Style

- No em dash characters in docs, comments, or commit messages.
- Comments explain non-obvious reasoning only.
