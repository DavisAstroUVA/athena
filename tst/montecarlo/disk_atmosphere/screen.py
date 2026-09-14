"""
Screen functions for make_spectrum.py --screen, for the disk atmosphere runs.

make_spectrum.py imports this module by name and calls the chosen function with a Photons
object; it returns a boolean array, True for photons to leave out of the spectrum.
"""

import numpy as np

# The top of the atmosphere in athinput.mcdisk and disk_atmosphere.py.  Kept in step by hand.
ZMAX = 1.0e11


def above_zmax(phots):
    """Drop photons that left the domain at z <= zmax.

    On the annular spherical grid a photon can leave through a radial wall while still
    inside the atmosphere; it has not emerged and would otherwise be binned as if it had.
    Photons leaving above zmax through either wall have emerged and drifted outward, and are
    kept.  The tolerance keeps those that exit the upper cone exactly at z = zmax.
    """
    z = phots.x1 * np.cos(phots.x2)
    return z <= ZMAX * (1.0 - 1.0e-6)
