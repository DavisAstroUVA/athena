"""
Support for manipulating and plotting Monte Carlo outputs.

The package replaces the single file athena_mc.py and keeps its interface: every public
name of the submodules is a name of the package, so ``import athena_mc as athenamc`` and
``athenamc.make_spectrum(...)`` work as before.  The submodules are lists (photon lists),
binning (bins and directions), spectra, images and plotting; plotting is the only one
that imports matplotlib, and it is loaded on first use, so the tools that bin lists
start without it.
"""

import importlib

from .lists import *
from .binning import *
from .spectra import *
from .images import *
from . import lists, binning, spectra, images

_PLOTTING = (
    'compute_nulnu_error',
    'compute_lnu_error',
    'compute_counts_error',
    'compute_pol_frac_error',
    'compute_pol_angle_error',
    'compute_q_error',
    'compute_u_error',
    'compute_v_error',
    'compute_flux_frac_error',
    'polarization_requested',
    'circular_requested',
    'check_polarization',
    'plot_frequency',
    'plot_theta',
    'plot_phi',
    'make_plot',
    'subsample_polarization',
    'plot_image',
)

__all__ = lists.__all__ + binning.__all__ + spectra.__all__ + images.__all__ + list(_PLOTTING)


def __getattr__(name):
    # the plotting names, imported on first use so that matplotlib loads only when asked
    # for; import_module rather than "from . import", which would ask this function for
    # the submodule again before it is bound
    if name in _PLOTTING or name == 'plotting':
        plotting = importlib.import_module('.plotting', __name__)
        return plotting if name == 'plotting' else getattr(plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
