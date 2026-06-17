"""Generate spike trains in pure Python with zero dependencies."""

from .population import population
from .processes import (
    bernoulli,
    gamma_renewal,
    homogeneous_poisson,
    inhomogeneous_poisson,
    jitter,
    regular,
    with_refractory,
)

__all__ = [
    "bernoulli",
    "gamma_renewal",
    "homogeneous_poisson",
    "inhomogeneous_poisson",
    "jitter",
    "population",
    "regular",
    "with_refractory",
]
__version__ = "0.3.0"
