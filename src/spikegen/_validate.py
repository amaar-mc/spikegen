from math import isfinite


def check_rate(rate: float) -> None:
    if not isfinite(rate) or rate <= 0:
        raise ValueError(f"rate must be a positive number, received {rate!r}")


def check_duration(duration: float) -> None:
    if not isfinite(duration) or duration < 0:
        raise ValueError(f"duration must be a non-negative number, received {duration!r}")


def check_positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number, received {value!r}")


def check_non_negative(name: str, value: float) -> None:
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative number, received {value!r}")


def check_seed(seed: int) -> None:
    if not isinstance(seed, int):
        raise ValueError(f"seed must be an integer, received {seed!r}")
