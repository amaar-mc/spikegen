"""Generate spike trains in pure Python with zero dependencies."""

import contextlib

from .population import population
from .processes import (
    bernoulli,
    gamma_renewal,
    homogeneous_poisson,
    inhomogeneous_poisson,
    inverse_gaussian_renewal,
    jitter,
    lognormal_renewal,
    regular,
    with_refractory,
)

__all__ = [
    "bernoulli",
    "gamma_renewal",
    "homogeneous_poisson",
    "homogeneous_poisson_numpy",
    "inhomogeneous_poisson",
    "inverse_gaussian_renewal",
    "jitter",
    "lognormal_renewal",
    "population",
    "regular",
    "with_refractory",
]
__version__ = "0.6.0"

# ``homogeneous_poisson_numpy`` is available only when NumPy is installed
# (``pip install spikegen[fast]``). Import it lazily so the package still
# loads with zero dependencies.
with contextlib.suppress(ImportError):
    from ._numpy import homogeneous_poisson_numpy
