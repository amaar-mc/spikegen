"""Generate spike trains in pure Python with zero dependencies."""

from .population import population
from .processes import (
    gamma_renewal,
    homogeneous_poisson,
    inhomogeneous_poisson,
    regular,
    with_refractory,
)

__all__ = [
    "gamma_renewal",
    "homogeneous_poisson",
    "inhomogeneous_poisson",
    "population",
    "regular",
    "with_refractory",
]
__version__ = "0.2.0"
