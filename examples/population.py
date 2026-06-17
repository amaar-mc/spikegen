"""Generate a population of independent Poisson spike trains.

Run with: python examples/population.py

The population is reproducible: the base seed and unit count fully determine every train.
"""

from spikegen import homogeneous_poisson, population


def main() -> None:
    units = 5
    rate = 50.0
    duration = 1.0
    pop = population(
        lambda s: homogeneous_poisson(rate=rate, duration=duration, seed=s),
        units=units,
        seed=0,
    )
    print(f"{units} units, Poisson {rate} Hz over {duration} s, base seed 0")
    for index, train in enumerate(pop):
        print(f"  unit {index}: {len(train)} spikes")


if __name__ == "__main__":
    main()
