import random
from collections.abc import Callable

from ._validate import check_count, check_seed


def population(
    make: Callable[[int], list[float]], *, units: int, seed: int
) -> list[list[float]]:
    """Generate a population of spike trains, one per unit.

    `make` takes an integer seed and returns one spike train, for example
    `lambda s: homogeneous_poisson(rate=50, duration=2, seed=s)`. Each unit is called with an
    independent seed drawn from `random.Random(seed)`, so the whole population depends only on
    `(seed, units)` and is reproducible. A deterministic generator that ignores its seed (such
    as `regular`) yields identical units, which is the expected behavior.
    """
    check_count("units", units)
    check_seed(seed)
    rng = random.Random(seed)
    # Draw all child seeds up front so each unit is independent and the population depends only
    # on (seed, units), not on how much randomness make consumes internally.
    child_seeds = [rng.randrange(2**63) for _ in range(units)]
    return [make(child_seed) for child_seed in child_seeds]
