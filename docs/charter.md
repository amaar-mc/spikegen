# Charter

## Purpose

Provide correct, reproducible, dependency-free spike-train generators that return plain
lists of spike times, so synthetic spike data is one import away and drops straight into
other lightweight tools.

## Scope

- The common point processes: regular, homogeneous Poisson, inhomogeneous Poisson (by
  thinning), and gamma renewal.
- A refractory-period filter.
- Explicit seeding for reproducibility.

## Non-goals

- A simulation framework or neuron models. spikegen produces spike times, not dynamics.
- A required NumPy dependency. NumPy may appear later only as an optional accelerator.
- Analysis of spike trains. spikedist covers distances and similarities.

## Principles

- Correctness and reproducibility first. Seeded generators are deterministic and tested.
- Zero runtime dependencies.
- Small, stable, explicit API. Keyword-only parameters.
- Plain data out (lists of floats), so it composes with anything.

## Audience

Computational neuroscience researchers and students, and neuromorphic engineers who need
reproducible synthetic spike trains without a framework.
