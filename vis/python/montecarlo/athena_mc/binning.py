"""
Bins and photon directions, shared by the spectra and the images.
"""

import numpy as np

__all__ = [
    'sum_into_bins',
    'build_bins',
    'get_bins',
    'get_bins_binary_search',
    'effective_kvec',
    'unit_direction_cartesian',
    'unit_direction_radial',
    'get_angle_bins_cartesian',
    'get_angle_bins_spherical',
    'get_angle_bins_hybrid',
]

def sum_into_bins(shape, indices, weights):
    """
    Sum weights into a zero array of the given shape at the bin indices, a tuple of one
    integer array per axis, all in range.  Does what np.add.at(out, indices, weights)
    does, several times faster: the multi-index is flattened and summed with
    np.bincount, which is a single pass in C rather than an unbuffered per-element loop.
    """
    flat = np.ravel_multi_index(indices, shape)
    counts = np.bincount(flat, weights=weights, minlength=int(np.prod(shape)))
    return counts.reshape(shape)


def build_bins(xmin, xmax, nx, logx):
    """
    Builds a x-axis grid for binning the photons
    """
    if logx:
        return np.logspace(np.log10(xmin),np.log10(xmax),nx+1)
    return np.linspace(xmin,xmax,nx+1)

def get_bins(xphots, xfaces, nx, uniform=True, log=True):
    """
    Returns the bin numbers corresponding photon samples
    """
    xbins = np.zeros(len(xphots), dtype=int)-1
    if uniform:
        if log:
            xlfaces = np.log10(xfaces)
            xwidth = xlfaces[nx] - xlfaces[0]

            # get integer bin number
            log_xphots = np.log10(xphots)
            xbins = np.floor((log_xphots - xlfaces[0]) / xwidth * nx).astype(int)

            # catch out-of-bounds values
            xbins = np.where((xbins < 0) | (xbins >= nx), -1, xbins)
        else:
            xwidth = xfaces[nx] - xfaces[0]

            # get integer bin number
            xbins = np.floor((xphots - xfaces[0]) / xwidth * nx).astype(int)

            # catch out-of-bounds values
            xbins = np.where((xbins < 0) | (xbins >= nx), -1, xbins)

        return xbins
    else:
        # use binary search for non uniform data (generally much slower)
        return get_bins_binary_search(xphots, xfaces, nx)

def get_bins_binary_search(xphots, xfaces, nx):
    """
    Returns x bin numbers corresponding to xphots for non-uniformly
    binned data via binary search.
    """
    # Exclude values outside of search range
    indsp = (xphots > xfaces[nx]).nonzero()
    indsm = (xphots < xfaces[0]).nonzero()
    xbins = np.searchsorted(xfaces,xphots)-1
    xbins[indsp] = -1
    xbins[indsm] = -1

    return xbins

def effective_kvec(photons):
    """
    Whether the stored wavevector is an orthonormal three-vector or contravariant
    coordinate components.

    _COORD_TAGS records what a coord tag meant when the tag was introduced.  A file that
    carries a frame field says it directly and wins: both 'normal' and 'lab' store
    components in an orthonormal frame, so the scale factors and metric projections the
    coordinate form needs must not be applied on top.
    """
    if getattr(photons, 'frame', None) in ('normal', 'lab'):
        return 'unit'
    return photons.props['kvec']


def unit_direction_cartesian(photons):
    """
    Cartesian components of the photon propagation direction, as a unit vector.

    Handles every wavevector convention the format has carried.  A file whose header
    declares basis=cartesian already holds the global cartesian components and needs
    only normalizing.  Otherwise the columns are on the local orthonormal legs: under the
    legacy pushers directly, under GeneralPusher as contravariant components whose
    spherical k^theta and k^phi carry factors of 1/r and 1/(r sin(theta)), and whose
    cylindrical k^phi carries 1/R, all removed before rotating.  The result is normalized
    either way: for a null vector the orthonormal spatial components have norm equal to
    the time component, not unity.
    """
    props = photons.props
    k1, k2, k3 = photons.k1, photons.k2, photons.k3

    if getattr(photons, 'basis', None) == 'cartesian':
        norm = np.sqrt(k1*k1 + k2*k2 + k3*k3)
        return k1/norm, k2/norm, k3/norm

    if props['geometry'] == 'spherical':
        cth = np.cos(photons.x2)
        sth = np.sin(photons.x2)
        cph = np.cos(photons.x3)
        sph = np.sin(photons.x3)
        if effective_kvec(photons) == 'coord':
            k2 = k2 * photons.x1
            k3 = k3 * photons.x1 * sth
        kx = k1*sth*cph + k2*cth*cph - k3*sph
        ky = k1*sth*sph + k2*cth*sph + k3*cph
        kz = k1*cth - k2*sth
    elif props['geometry'] == 'cylindrical':
        cph = np.cos(photons.x2)
        sph = np.sin(photons.x2)
        if effective_kvec(photons) == 'coord':
            k2 = k2 * photons.x1
        kx = k1*cph - k2*sph
        ky = k1*sph + k2*cph
        kz = k3
    else:
        kx, ky, kz = k1, k2, k3

    norm = np.sqrt(kx*kx + ky*ky + kz*kz)
    return kx/norm, ky/norm, kz/norm


def unit_direction_radial(photons):
    """
    Component of the photon propagation direction along the local radial unit vector.
    """
    props = photons.props

    # Go through the cartesian direction, which already resolves every basis the file
    # can carry, and dot it with the radial unit vector built from the position.  The
    # spherical shortcut of reading k1 directly is gone: it was only right while the
    # columns were on the local legs.
    kx, ky, kz = unit_direction_cartesian(photons)
    if props['geometry'] == 'spherical':
        sth = np.sin(photons.x2)
        x = sth*np.cos(photons.x3)
        y = sth*np.sin(photons.x3)
        z = np.cos(photons.x2)
        return kx*x + ky*y + kz*z
    if props['geometry'] == 'cylindrical':
        x = photons.x1*np.cos(photons.x2)
        y = photons.x1*np.sin(photons.x2)
        z = photons.x3
    else:
        x, y, z = photons.x1, photons.x2, photons.x3
    rnorm = np.sqrt(x*x + y*y + z*z)
    return (kx*x + ky*y + kz*z)/rnorm


def get_angle_bins_cartesian(photons,nmu, mufaces, nphi, phifaces):
    """
    Bin angles in theta, phi defined relative to x,y,z axes
    """

    if ((nmu == 1) and (mufaces[0] <= 0.) and (mufaces[1] >= 1.0)):
        skipmu = True
    else:
        skipmu = False
    if ((nphi == 1) and (phifaces[0] <= 0.) and (phifaces[1] >= 2.*np.pi)):
        skipphi = True
    else:
        skipphi = False

    if (skipmu and skipphi):
        return np.zeros(photons.nphot,dtype=int),np.zeros(photons.nphot,dtype=int)

    kx, ky, kz = unit_direction_cartesian(photons)

    if skipmu:
        # return 0
        mubins = np.zeros(photons.nphot,dtype=int)
    else:
        # Bin based on k . z
        mu = abs(kz)
        mubins = get_bins(mu,mufaces,nmu,log=False)
    if skipphi:
        # return 0
        phibins = np.zeros(photons.nphot,dtype=int)
    else:
        phi = np.arctan2(ky, kx)
        phi[(phi<0.).nonzero()] += 2.*np.pi
        phibins = get_bins(phi, phifaces, nphi, log=False)

    return mubins, phibins


def get_angle_bins_spherical(photons,nmu,mufaces):
    """
    Bin angles relative to local radial direction.  Here we only bin
    in polar angle.
    """
    if ((nmu == 1) and (mufaces[0] <= 0.) and (mufaces[1] >= 1.0)):
        skipmu = True
    else:
        skipmu = False

    if skipmu:
        return np.zeros(photons.nphot,dtype=int),np.zeros(photons.nphot,dtype=int)

    # Bin based on k_r
    mu = unit_direction_radial(photons)
    mubins = get_bins(mu, mufaces ,nmu, log=False)

    # return 0 for phi
    phibins = np.zeros(photons.nphot, dtype=int)

    return mubins, phibins

def get_angle_bins_hybrid(photons, nmu, mufaces, nphi, phifaces):
    """
    Bin angles in theta, phi defined relative to x,y,z axes, but with phi
    defined by local azimuthal angle
    """
    if nmu == 1:
        print("Error: this function requires nmu > 1")
        return  np.zeros(photons.nphot,dtype=int),np.zeros(photons.nphot,dtype=int)
    if nphi == 1:
        print("Error: this function requires nphi > 1")
        return np.zeros(photons.nphot,dtype=int),np.zeros(photons.nphot,dtype=int)

    if photons.props['geometry'] != 'spherical':
        print("Error: this function only works with spherical geometry")
        return np.zeros(photons.nphot,dtype=int),np.zeros(photons.nphot,dtype=int)

    # The z and phi-hat components of the direction, from the cartesian form so that the
    # basis the file was written in is resolved in one place.
    kx, ky, kz = unit_direction_cartesian(photons)
    cph = np.cos(photons.x3)
    sph = np.sin(photons.x3)
    kph = -kx*sph + ky*cph
    # radial component, used below to place the azimuth in the right quadrant; it was
    # never defined here, so the hybrid binning raised NameError on every call
    sth = np.sin(photons.x2)
    cth = np.cos(photons.x2)
    kr = (kx*cph + ky*sph)*sth + kz*cth

    # Bin based on k . z
    mu = abs(kz)
    mubins = get_bins(mu,mufaces,nmu,log=False)

    phi = np.arcsin(kph)
    # inds should be empty for outgoing photons
    inds = (kr < 0.).nonzero()
    phi[inds] = np.pi - phi[inds]
    phi[(phi<0.).nonzero()] += 2.*np.pi
    phibins = get_bins(phi, phifaces, nphi, log=False)

    return mubins,phibins
