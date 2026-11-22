"""
The Lyman alpha scattering kernel of ScatterResonanceLine in numpy, for the frequency-only
toys of the frequency lattice: the parallel atom velocity by the Zheng & Miralda-Escude
scheme (the Laursen u0) below x = 1 + sqrt 2 and the ratio-of-uniforms scheme of
Seon & Kim (2020) with exact envelope extrema above (the hybrid of SampleVelocityParallel
with upar_sampler = seon), the perpendicular velocity Gaussian, the dipole phase function.
"""
import numpy as np
from scipy.special import wofz


def voigt(a, x):
    """H(a, x), the Voigt function normalized to 1 at line center as a -> 0"""
    return wofz(np.asarray(x, float) + 1j*a).real


def laursen_u0(a, x):
    xcw_l = 1.59 - 0.60*np.log10(a) - 0.03*np.log10(a)**2
    u0 = np.where(x < 0.2, 0., x - 0.01*a**(1./6.)*np.exp(1.2*x))
    return np.where(x >= xcw_l, 4.5, u0)


XC = 1. + np.sqrt(2.)


XC = 1. + np.sqrt(2.)


def sample_upar(rng, a, x):
    """the parallel atom velocity for photons at x (any sign), vectorized: the Zheng &
    Miralda-Escude scheme with the Laursen u0 below x = 1 + sqrt(2), the
    ratio-of-uniforms scheme of Seon & Kim (2020) with exact envelope extrema above"""
    ax = np.abs(x)
    u = np.empty_like(ax)
    near = ax < XC
    idx = np.nonzero(near)[0]
    if idx.size:
        xx = ax[idx]
        u0 = laursen_u0(a, xx)
        th0 = np.arctan((u0 - xx)/a)
        p = (th0 + 0.5*np.pi)/((th0 + 0.5*np.pi) + np.exp(-u0*u0)*(0.5*np.pi - th0))
        todo = np.ones(idx.size, bool)
        out = np.empty(idx.size)
        while todo.any():
            t = np.nonzero(todo)[0]
            first = rng.uniform(size=t.size) < p[t]
            th = np.where(first, rng.uniform(-0.5*np.pi, th0[t]),
                          rng.uniform(th0[t], 0.5*np.pi))
            uu = xx[t] + a*np.tan(th)
            acc = np.where(first, np.exp(-uu*uu), np.exp(-(uu*uu - u0[t]*u0[t])))
            ok = rng.uniform(size=t.size) < acc
            out[t[ok]] = uu[ok]
            todo[t[ok]] = False
        u[idx] = out
    idx = np.nonzero(~near)[0]
    if idx.size:
        xx = ax[idx]
        X = xx*xx + a*a
        r = np.sqrt(X*X - 6.*X + 1.)
        slo = 0.5*((X - 1.) - r)
        shi = 0.5*((X - 1.) + r)
        b0 = np.exp(-0.5*xx*xx)
        bmin = np.exp(-0.5*shi)
        h2 = (1. + 1e-8)*(2./np.pi)*np.exp(-0.5*slo)*np.sqrt(slo)/(X - slo)
        h0 = bmin/(2.*a)
        S0 = b0*b0/(2.*a)
        S1 = h0*(bmin - b0)
        stot = S0 + S1 + h2*(1. - bmin)
        todo = np.ones(idx.size, bool)
        out = np.empty(idx.size)
        while todo.any():
            t = np.nonzero(todo)[0]
            xi = rng.uniform(size=t.size)*stot[t]
            eta = rng.uniform(size=t.size)
            beta = np.where(xi < S0[t], b0[t]*np.sqrt(eta),
                            np.where(xi < S0[t] + S1[t], b0[t] + eta*(bmin[t] - b0[t]),
                                     bmin[t] + eta*(1. - bmin[t])))
            comp = np.where(xi < S0[t], beta/a, np.where(xi < S0[t] + S1[t], h0[t], h2[t]))
            pp = np.sqrt(-2.*np.log(np.maximum(beta, 1e-300)))
            thlo = np.arctan((-pp - xx[t])/a)
            dth = np.where(pp < xx[t], np.arctan(2.*pp*a/(X[t] - pp*pp)),
                           np.arctan((pp - xx[t])/a) - thlo)
            q = beta/(a*np.pi)*dth
            ok = rng.uniform(size=t.size)*comp < q
            uu = xx[t] + a*np.tan(thlo + rng.uniform(size=t.size)*dth)
            out[t[ok]] = uu[ok]
            todo[t[ok]] = False
        u[idx] = out
    return np.where(x < 0., -u, u)


def kernel_scatter(rng, a, x):
    n = x.size
    cgam = np.empty(n)
    todo = np.ones(n, bool)
    while todo.any():
        t = np.nonzero(todo)[0]
        c = rng.uniform(-1., 1., t.size)
        ok = 2.*rng.uniform(size=t.size) <= 1. + c*c
        cgam[t[ok]] = c[ok]
        todo[t[ok]] = False
    sgam = np.sqrt(1. - cgam*cgam)
    uperp = np.sqrt(-np.log(np.maximum(rng.uniform(size=n), 1e-300))) \
        * np.cos(2.*np.pi*rng.uniform(size=n))
    upar = sample_upar(rng, a, x)
    return x + (cgam - 1.)*upar + sgam*uperp
