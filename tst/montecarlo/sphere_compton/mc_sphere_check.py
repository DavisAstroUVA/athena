
"""
Direct Monte Carlo check of kompaneets_greens with absorption.

Photons start at the centre of a uniform sphere (radius tau in Thomson
optical depth), undergo exact Compton scattering (Klein-Nishina, in the
electron rest frame) on relativistic Maxwellian electrons, and are absorbed
by free-free opacity kappa_ff/kappa_es = theta*lam*a(x), applied as a
continuous weight along each path.  For each escaping photon we record xf,
the weight (survival) and y = theta*(path length in tau_es).

If energy and space separate (uniform sphere, kappa_abs << kappa_es), the
escaping photons with a given y must have energies distributed as the
absorbed Kompaneets Green's function P_lam(xf, y | x0), with survival
S(y) = int P_lam.  We compare, in bins of y:
  - the survival (MC mean weight vs S at each photon's y)
  - the weighted distribution of ln xf (MC histogram vs the average of
    P_lam over the photons' y)
and, for information, the MC escape-path distribution against the diffusion
escape-path distribution (sphere_theory.MixedBoundaryEscape).

    python mc_sphere_check.py                  # default cases
    python mc_sphere_check.py --x0 0.03 --lam 2e-3 --theta 2e-3 --tau 15
"""

import argparse
import numpy as np
import os
import sys
_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      '..', '..', '..', 'vis', 'python', 'montecarlo'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'problem_specific', 'sphere_compton'))
import kompaneets_greens as kg  # noqa: E402


def kn_sigma(e):
    """total Klein-Nishina cross-section / sigma_T at rest-frame energy e"""
    small = e < 1.e-3
    es = np.where(small, 1., e)
    lg = np.log1p(2.*es)
    big = 0.75*((1.+es)/es**3*(2.*es*(1.+es)/(1.+2.*es)-lg) + lg/(2.*es)
                - (1.+3.*es)/(1.+2.*es)**2)
    return np.where(small, 1.-2.*e+5.2*e*e, big)


class MaxwellJuttner:
    """electron momentum (units m_e c) sampler by tabulated inverse CDF"""

    def __init__(self, theta, ncut=60., n=40000):
        gmax = 1.+ncut*theta
        p = np.linspace(0., np.sqrt(gmax**2-1.), n)
        g = np.sqrt(1.+p*p)
        f = p*p*np.exp(-(g-1.)/theta)
        c = np.concatenate([[0.], np.cumsum(0.5*(f[1:]+f[:-1])*np.diff(p))])
        self.cdf = c/c[-1]
        self.p = p
        self.betamax = p[-1]/g[-1]

    def sample(self, rng, n):
        return np.interp(rng.random(n), self.cdf, self.p)


def isotropic(rng, n):
    mu = 2.*rng.random(n)-1.
    ph = 2.*np.pi*rng.random(n)
    st = np.sqrt(1.-mu*mu)
    return np.stack([st*np.cos(ph), st*np.sin(ph), mu], axis=1)


def rotate(om, mu, ph):
    """direction at polar cosine mu, azimuth ph about the unit vectors om"""
    a = np.zeros_like(om)
    usex = np.abs(om[:, 0]) < 0.9
    a[usex, 0] = 1.
    a[~usex, 1] = 1.
    u = np.cross(a, om)
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(om, u)
    st = np.sqrt(np.maximum(1.-mu*mu, 0.))
    return (mu[:, None]*om + (st*np.cos(ph))[:, None]*u
            + (st*np.sin(ph))[:, None]*v)


def compton_event(rng, x, om, theta, mj, kmaj):
    """
    Null-collision Compton event for photons (x = E/kT, directions om).
    Returns new x, om (unchanged where the event is a null collision).
    """
    n = len(x)
    eps = x*theta
    p = mj.sample(rng, n)
    gam = np.sqrt(1.+p*p)
    beta = p/gam
    ne = isotropic(rng, n)
    mue = np.einsum("ij,ij->i", ne, om)
    epsr = gam*eps*(1.-beta*mue)
    acc = rng.random(n) < (1.-beta*mue)*kn_sigma(epsr)/kmaj
    idx = np.nonzero(acc)[0]
    if len(idx) == 0:
        return x, om
    g, b, nn, e, er = gam[idx], beta[idx], ne[idx], eps[idx], epsr[idx]
    k = e[:, None]*om[idx]
    kn = np.einsum("ij,ij->i", k, nn)
    kr = k + ((g-1.)*kn - g*b*e)[:, None]*nn
    omr = kr/er[:, None]
    # Klein-Nishina polar angle by rejection in the rest frame
    mus = np.empty(len(idx))
    todo = np.arange(len(idx))
    while len(todo):
        mu = 2.*rng.random(len(todo))-1.
        rho = 1./(1.+er[todo]*(1.-mu))
        ok = rng.random(len(todo)) < 0.5*rho*rho*(rho+1./rho-(1.-mu*mu))
        mus[todo[ok]] = mu[ok]
        todo = todo[~ok]
    er1 = er/(1.+er*(1.-mus))
    om1r = rotate(omr, mus, 2.*np.pi*rng.random(len(idx)))
    k1r = er1[:, None]*om1r
    k1n = np.einsum("ij,ij->i", k1r, nn)
    e1 = g*(er1 + b*k1n)
    k1 = k1r + ((g-1.)*k1n + g*b*er1)[:, None]*nn
    x = x.copy()
    om = om.copy()
    x[idx] = e1/theta
    om[idx] = k1/e1[:, None]
    return x, om


def run_mc(x0, tau, theta, lam, nphot=200000, seed=1, sink=kg.freefree_shape):
    rng = np.random.default_rng(seed)
    mj = MaxwellJuttner(theta)
    kmaj = 1.+mj.betamax
    r = np.zeros((nphot, 3))
    om = isotropic(rng, nphot)
    x = np.full(nphot, float(x0))
    w = np.ones(nphot)
    path = np.zeros(nphot)
    ids = np.arange(nphot)
    xf = np.zeros(nphot)
    wf = np.zeros(nphot)
    pf = np.zeros(nphot)
    while len(ids):
        ell = -np.log(rng.random(len(ids)))/kmaj
        b = np.einsum("ij,ij->i", r, om)
        c = np.einsum("ij,ij->i", r, r)-tau*tau
        tb = -b+np.sqrt(np.maximum(b*b-c, 0.))
        esc = ell >= tb
        step = np.where(esc, tb, ell)
        path += step
        w *= np.exp(-theta*lam*sink(x)*step)
        r += step[:, None]*om
        # record and drop escapers
        e = np.nonzero(esc)[0]
        xf[ids[e]], wf[ids[e]], pf[ids[e]] = x[e], w[e], path[e]
        keep = ~esc
        ids, r, om, x, w, path = (ids[keep], r[keep], om[keep], x[keep],
                                  w[keep], path[keep])
        if len(ids):
            x, om = compton_event(rng, x, om, theta, mj, kmaj)
    return xf, wf, theta*pf


def predict(x0, lam, yph, sedges, sink=kg.freefree_shape, ngrid=400):
    """
    For each photon's y, the solver's integral of P_lam over each ln xf bin
    (unnormalised: sums to the survival S(y)).  Photons are mapped to a fine
    log grid in y.
    """
    ylo, yhi = yph.min()*0.999, yph.max()*1.001
    yg = np.logspace(np.log10(ylo), np.log10(yhi), ngrid)
    snk = sink if lam > 0. else None
    prop = kg.Propagator(sink=snk, lam=lam)
    res = kg.greens(x0, yg, prop=prop, sink=snk, lam=lam)
    m = np.zeros((ngrid, len(sedges)-1))
    surv = np.zeros(ngrid)
    for i, (s, p) in enumerate(res):
        c = np.concatenate([[0.], np.cumsum(0.5*(p[1:]+p[:-1])*np.diff(s))])
        m[i] = np.diff(np.interp(sedges, s, c))
        surv[i] = c[-1]
    idx = np.clip(np.rint(np.log(yph/ylo)/np.log(yg[1]/yg[0])).astype(int),
                  0, ngrid-1)
    return m, surv, idx


def check(x0, tau, theta, lam, nphot, nybin=6, nsbin=40, seed=1,
          plot=None):
    print("\n=== x0={:g} tau={:g} theta={:g} lam={:g} nphot={:d} ==="
          .format(x0, tau, theta, lam, nphot))
    print("    max kappa_abs/kappa_es at x0: {:.2e}".format(
        theta*lam*kg.freefree_shape(x0)))
    xf, wf, y = run_mc(x0, tau, theta, lam, nphot, seed)
    s = np.log(xf)
    lo, hi = np.percentile(s, [0.05, 99.95])
    sedges = np.linspace(lo, hi, nsbin+1)
    m, surv, idx = predict(x0, lam, y, sedges)
    ybins = np.quantile(y, np.linspace(0., 1., nybin+1))
    ybins[-1] *= 1.0001
    print("    {:>17s} {:>7s} {:>11s} {:>11s} {:>9s} {:>9s} {:>9s}".format(
        "y bin", "N", "S_mc", "S_kg", "dS/err", "<x>_mc/kg", "maxdCDF"))
    rows = []
    for b in range(nybin):
        sel = (y >= ybins[b]) & (y < ybins[b+1])
        nb = sel.sum()
        hmc, _ = np.histogram(s[sel], sedges, weights=wf[sel])
        hkg = m[idx[sel]].sum(axis=0)
        smc = wf[sel].sum()/nb
        skg = surv[idx[sel]].mean()
        err = wf[sel].std()/np.sqrt(nb)
        dse = (smc-skg)/err if err > 0. else np.nan
        cmc = np.cumsum(hmc)/max(hmc.sum(), 1.e-300)
        ckg = np.cumsum(hkg)/max(hkg.sum(), 1.e-300)
        sc = 0.5*(sedges[1:]+sedges[:-1])
        xmc = np.sum(hmc*np.exp(sc))/hmc.sum()
        xkg = np.sum(hkg*np.exp(sc))/hkg.sum()
        print("    {:8.3g}-{:<8.3g} {:7d} {:11.4e} {:11.4e} {:+9.2f} {:9.4f} "
              "{:9.4f}".format(ybins[b], ybins[b+1], nb, smc, skg,
                               dse, xmc/xkg,
                               np.max(np.abs(cmc-ckg))))
        rows.append((ybins[b], ybins[b+1], sc, hmc/nb, hkg/nb))
    # whole escaping spectrum
    hmc, _ = np.histogram(s, sedges, weights=wf)
    hkg = m[idx].sum(axis=0)
    print("    all y: S_mc={:.4e} S_kg={:.4e}  max dCDF={:.4f}".format(
        wf.mean(), surv[idx].mean(),
        np.max(np.abs(np.cumsum(hmc)/hmc.sum()-np.cumsum(hkg)/hkg.sum()))))
    # MC escape paths vs the diffusion escape-time distribution (mixed boundary, the
    # one a sphere radiating into vacuum follows)
    import sphere_theory as st
    tp = y/theta
    tq = np.quantile(tp, [0.1, 0.25, 0.5, 0.75, 0.9])
    cdf = st.MixedBoundaryEscape(tau).cdf(tq)
    print("    escape-path CDF at MC quantiles 0.1..0.9 (mixed-boundary diffusion): "
          + " ".join("{:.3f}".format(v) for v in cdf))
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for k, (y0, y1, sc, a, b) in enumerate(rows):
            col = "C{:d}".format(k)
            ax.step(np.exp(sc), a/np.diff(sedges), where="mid", color=col,
                    label="y {:.2g}-{:.2g}".format(y0, y1))
            ax.plot(np.exp(sc), b/np.diff(sedges), "--", color=col)
        ax.set_xscale("log")
        ax.set_yscale("log")
        top = max(np.max(r[3]/np.diff(sedges)) for r in rows)
        ax.set_ylim(1.e-5*top, 2.*top)
        ax.set_xlabel("x_f")
        ax.set_ylabel("dN/dln x_f per photon (solid MC, dashed solver)")
        ax.set_title("x0={:g} tau={:g} theta={:g} lam={:g}".format(
            x0, tau, theta, lam))
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plot)
        print("    plot:", plot)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--x0", type=float)
    ap.add_argument("--tau", type=float, default=15.)
    ap.add_argument("--theta", type=float, default=2.e-3)
    ap.add_argument("--lam", type=float, default=0.)
    ap.add_argument("--nphot", type=int, default=200000)
    ap.add_argument("--plot", default=None)
    a = ap.parse_args()
    if a.x0 is not None:
        check(a.x0, a.tau, a.theta, a.lam, a.nphot, plot=a.plot)
    else:
        for x0, lam in [(1., 0.), (0.03, 2.e-3), (0.3, 2.e-2), (3., 1.)]:
            check(x0, a.tau, a.theta, lam, a.nphot,
                  plot="mc_check_x{:g}_lam{:g}.pdf".format(x0, lam))
