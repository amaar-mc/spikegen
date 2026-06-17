# Contributing to spikegen

Thanks for your interest. This project values correctness, reproducibility, and zero
runtime dependencies.

## Development

```sh
uv venv
uv pip install -e ".[dev]"
uv run pytest -q
uv run ruff check .
uv run mypy src
```

A standard virtual environment with `pip install -e ".[dev]"` works the same way.

## Guidelines

- No runtime dependencies. The standard library `random` and `math` are enough.
- Seeded generators must be reproducible: the same seed gives the same train. Tests assert
  this.
- Every generator needs structural tests (sorted, within range) and, where it is
  deterministic, exact-value tests.
- Parameters are keyword-only with no default values.
- A bug fix starts with a failing test.
- Run `uv run ruff format .` before committing.
- Commit messages follow `type(scope): description`.

## Reporting issues

Open an issue with the call and its parameters, the seed, and what you expected versus what
you observed.
