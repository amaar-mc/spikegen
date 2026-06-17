import pytest

from spikegen import homogeneous_poisson, population, regular


def poisson_unit(rate: float, duration: float):
    return lambda s: homogeneous_poisson(rate=rate, duration=duration, seed=s)


class TestPopulation:
    def test_unit_count(self) -> None:
        pop = population(poisson_unit(50.0, 1.0), units=7, seed=0)
        assert len(pop) == 7

    def test_reproducible(self) -> None:
        a = population(poisson_unit(50.0, 1.0), units=5, seed=3)
        b = population(poisson_unit(50.0, 1.0), units=5, seed=3)
        assert a == b

    def test_units_are_independent(self) -> None:
        # Distinct child seeds give distinct stochastic trains, so no two units coincide.
        pop = population(poisson_unit(50.0, 2.0), units=6, seed=1)
        as_tuples = {tuple(train) for train in pop}
        assert len(as_tuples) == len(pop)

    def test_base_seed_changes_population(self) -> None:
        a = population(poisson_unit(50.0, 1.0), units=4, seed=0)
        b = population(poisson_unit(50.0, 1.0), units=4, seed=1)
        assert a != b

    def test_deterministic_generator_repeats(self) -> None:
        # regular ignores the seed, so every unit is identical, which is expected.
        pop = population(lambda _s: regular(rate=10.0, duration=1.0), units=3, seed=0)
        assert pop[0] == regular(rate=10.0, duration=1.0)
        assert all(train == pop[0] for train in pop)

    def test_units_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="units must be a positive integer"):
            population(poisson_unit(50.0, 1.0), units=0, seed=0)

    def test_units_rejects_bool(self) -> None:
        with pytest.raises(ValueError, match="units must be a positive integer"):
            population(poisson_unit(50.0, 1.0), units=True, seed=0)
