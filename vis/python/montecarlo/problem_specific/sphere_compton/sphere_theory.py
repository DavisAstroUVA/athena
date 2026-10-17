"""
Theory for the uniform isothermal Compton sphere (src/pgen/mc_sphere_isoth.cpp): the
constants and composition the code uses, the diffusion escape-path distributions for a
point source at the centre, the free-free absorption parameter of the Kompaneets Green's
function, and the escaping spectrum predicted by convolving the Green's function with the
escape-path distribution.

Conventions.  tau is the Thomson optical depth of the sphere, tau_path the Thomson optical
path a photon has travelled (its geometric path times n_e sigma_T), x = h nu / k T,
theta = k T / m_e c^2 and y = theta tau_path.  The dimensionless diffusion time is
t = pi^2 tau_path / (3 tau^2), so that the slowest mode of the absorbing sphere decays as
exp(-t).

Two escape-path distributions are given.  The absorbing boundary (Fleck & Canfield 1984;
Min et al. 2009 eq. 7) is the first-passage distribution the modified random walk uses
inside a cell, where the sphere surface is where ordinary transport resumes.  The mixed
boundary, tan(lambda) = lambda / (1 - 1.5 tau), is the Marshak-type condition for a sphere
radiating into vacuum and is what the sphere test's escapers follow; it tends to the
absorbing one as tau grows.
"""

import numpy as np
from scipy import optimize

import kompaneets_greens as kg

# Constants as the code has them (montecarloblock.cpp, opacity.cpp, mc_sphere_isoth.cpp)
H_PLANCK = 6.62607015e-27
K_BOLTZ = 1.380649e-16
M_ELECTRON = 9.1093897e-28
C_LIGHT = 2.99792458e10
SIGMA_T = 6.65248e-25
M_PROTON = 1.6726e-24
EV = 1.6021772e-12


def kappa_es(heabund=0.09):
    """electron-scattering opacity per gram of a fully ionized H/He mixture"""
    return SIGMA_T*(1.+2.*heabund)/(M_PROTON*(1.+4.*heabund))


def density(tau, radius, heabund=0.09):
    """density that gives Thomson depth tau across radius"""
    return tau/(kappa_es(heabund)*radius)


def theta(temp):
    return K_BOLTZ*temp/(M_ELECTRON*C_LIGHT**2)


def lam_freefree(tau, radius, temp, heabund=0.09):
    """Kompaneets absorption parameter kappa_ff(x=1)/(kappa_es theta) of the sphere"""
    return kg.lam_freefree(density(tau, radius, heabund), temp, heabund)


def diffusion_time(tau_path, tau):
    """dimensionless diffusion time t of a path tau_path in a sphere of depth tau"""
    return np.pi**2*tau_path/(3.*tau*tau)


def escape_cdf_absorbing(tau_path, tau, nmax=200):
    """
    Probability that a photon born at the centre has reached the sphere by Thomson path
    tau_path, absorbing boundary: 1 - 2 sum (-1)^(n+1) Y^(n^2), Y = exp(-t).
    """
    t = np.atleast_1d(diffusion_time(tau_path, tau)).astype(float)
    cdf = np.zeros_like(t)
    small = t < 1.e-3
    # the series in Y converges slowly as t -> 0; there the answer is 0 anyway
    tt = np.where(small, 1.e-3, t)
    for n in range(1, nmax+1):
        term = 2.*(-1.)**(n+1)*np.exp(-n*n*tt)
        cdf += term
        if np.all(np.abs(term) < 1.e-15):
            break
    cdf = 1.-cdf
    cdf[small] = 0.
    return cdf


class MixedBoundaryEscape:
    """
    Escape-path distribution for a point source at the centre of a sphere of depth tau
    with the mixed boundary condition tan(lambda) = lambda/(1 - 1.5 tau): the
    eigenvalues are found once and the CDF and PDF evaluated by summing the series.
    """

    def __init__(self, tau, tol=1.e-17, nmax=2000):
        self.tau = tau
        a = 1.-1.5*tau
        roots, norms = [], []
        lo, hi = 0.51*np.pi, 1.49*np.pi
        sol = optimize.root_scalar(lambda x: np.tan(x)-x/a, bracket=[lo, hi])
        lamn = sol.root
        dl = lamn
        for n in range(1, nmax+1):
            In = 0.5*tau*(1.+(1.5*tau-1.)/((1.5*tau-1.)**2+lamn*lamn))
            cn = np.cos(lamn)*1.5*tau*tau/((1.-1.5*tau)*In)
            roots.append(lamn)
            norms.append(cn)
            br = [lamn+(1.-0.1/(n+1))*dl, lamn+(1.+0.1/(n+1))*dl]
            sol = optimize.root_scalar(lambda x: np.tan(x)-x/a, bracket=br)
            dl = sol.root-lamn
            lamn = sol.root
        self.k2 = np.array(roots)**2/np.pi**2
        self.cn = np.array(norms)
        self.tol = tol

    def survival(self, tau_path):
        """probability of still being inside after path tau_path"""
        t = np.atleast_1d(diffusion_time(tau_path, self.tau)).astype(float)
        s = np.zeros_like(t)
        for k2, cn in zip(self.k2, self.cn):
            term = cn*np.exp(-k2*t)
            s += term
            if np.all(np.abs(term) < self.tol):
                break
        return np.clip(s, 0., 1.)

    def cdf(self, tau_path):
        return 1.-self.survival(tau_path)

    def pdf(self, tau_path):
        """per unit tau_path"""
        t = np.atleast_1d(diffusion_time(tau_path, self.tau)).astype(float)
        p = np.zeros_like(t)
        for k2, cn in zip(self.k2, self.cn):
            term = cn*k2*np.exp(-k2*t)
            p += term
            if np.all(np.abs(term) < self.tol):
                break
        return np.maximum(p, 0.)*np.pi**2/(3.*self.tau**2)


def greens_bins(xi, lam, y, sedges, ngrid=400):
    """
    For each photon's y, the Green's function P_lam(ln xf, y | xi) integrated over the
    ln xf bins sedges (unnormalized, summing to the survival S(y)), evaluated on a log
    grid of ngrid points spanning the y values; returns the (ngrid, nbin) bin masses,
    the survival at each grid point and each photon's grid index.  Same as predict()
    in mc_sphere_check.py.
    """
    ylo, yhi = y.min()*0.999, y.max()*1.001
    yg = np.logspace(np.log10(ylo), np.log10(yhi), ngrid)
    snk = kg.freefree_shape if lam > 0. else None
    prop = kg.Propagator(sink=snk, lam=lam)
    res = kg.greens(xi, yg, prop=prop, sink=snk, lam=lam)
    m = np.zeros((ngrid, len(sedges)-1))
    surv = np.zeros(ngrid)
    for i, (s, p) in enumerate(res):
        c = np.concatenate([[0.], np.cumsum(0.5*(p[1:]+p[:-1])*np.diff(s))])
        m[i] = np.diff(np.interp(sedges, s, c))
        surv[i] = c[-1]
    idx = np.clip(np.rint(np.log(y/ylo)/np.log(yg[1]/yg[0])).astype(int), 0, ngrid-1)
    return m, surv, idx


def escape_spectrum(xi, tau, thet, lam, sedges, ny=300, tmax=30.):
    """
    Escaping spectrum per injected photon in the ln xf bins sedges, the Green's
    function convolved with the mixed-boundary escape-path pdf, and the escaping
    fraction.  Paths run to diffusion time tmax.
    """
    esc = MixedBoundaryEscape(tau)
    tp_max = tmax*3.*tau*tau/np.pi**2
    tp = np.logspace(np.log10(tp_max)-6., np.log10(tp_max), ny)
    pdf = esc.pdf(tp)
    y = np.maximum(thet*tp, 1.e-12)
    snk = kg.freefree_shape if lam > 0. else None
    prop = kg.Propagator(sink=snk, lam=lam)
    res = kg.greens(xi, y, prop=prop, sink=snk, lam=lam)
    spec = np.zeros(len(sedges)-1)
    frac = 0.
    # trapezoid in tau_path
    masses = []
    survs = []
    for s, p in res:
        c = np.concatenate([[0.], np.cumsum(0.5*(p[1:]+p[:-1])*np.diff(s))])
        masses.append(np.diff(np.interp(sedges, s, c)))
        survs.append(c[-1])
    masses = np.array(masses)
    survs = np.array(survs)
    w = pdf[:, None]*masses
    spec = np.sum(0.5*(w[1:]+w[:-1])*np.diff(tp)[:, None], axis=0)
    frac = np.sum(0.5*(pdf[1:]*survs[1:]+pdf[:-1]*survs[:-1])*np.diff(tp))
    return spec, frac
