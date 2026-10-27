
"""
Green's function of the linear Kompaneets equation (Doppler + recoil, no induced
scattering), with an optional energy-dependent absorption sink, for sampling
photon energies after a Compton y-parameter y in the Monte Carlo code.

Formulation: with s = ln(x) and P(s,y) = x^3 n (photon number per unit ln x),
the Kompaneets equation dn/dy = x^-2 d/dx[x^4 (dn/dx + n)] becomes the
Fokker-Planck equation

    dP/dy = d/ds [ dP/ds + V'(s) P ] - Lam*a(x) P,    V(s) = e^s - 3 s

i.e. unit diffusion in ln x with drift 3 - x.  The equilibrium is the Wien
spectrum per ln x, w(s) = exp(-V) = x^3 e^-x.  Substituting P = e^{-V/2} psi
gives dpsi/dy = psi_ss - U psi with the Morse potential
U = e^{2s}/4 - 2 e^s + 9/4, whose spectrum (discrete 0 and 2, continuum from
9/4) is the one in Becker (2003, MNRAS 343, 215), eq. 33.

Three evaluators:
  short_time_kernel   analytic heat-kernel approximation, used for y < y_s(xi)
  Propagator          Scharfetter-Gummel / Chang-Cooper discretisation in s,
                      TR-BDF2 in y; seeded with the short-time kernel at y_s
  becker_greens       Becker's exact integral via mpmath (slow, validation only)

Normalisation throughout: integral of P ds = 1 at y = 0, i.e. the returned
densities are per photon.  In terms of the old code's n(xf) (normalised to
int xf^2 n dxf = xi^2):  n = xi^2 P / xf^3.

Absorption: the sink lam*a(x) (default a = freefree_shape, lam from
lam_freefree) is exact within the Kompaneets description.  For a uniform
sphere with kappa_abs << kappa_es, energy and space separate, so a photon
escaping after Thomson path tau_path has energy drawn from P_lam(xf, y) with
y = theta*tau_path and survives with probability S = int P_lam ds.
mc_sphere_check.py tests this against direct Compton Monte Carlo.

Tables: gen_quantile_table / `python kompaneets_greens.py [--absorption]`
write quantiles of ln(xf/xi)/sqrt(2y) on a (lam, y, xi, level) grid plus
abar = -ln S/(lam y); Sampler reads them (sample() returns xf and S).
check_table.py spot-checks a table against direct solutions.
"""

import numpy as np
from scipy.linalg import solve_banded

# Default switch between the short-time kernel and the discrete propagator:
# y_s(x) = YS_COEF / max(1, x).  Error of the kernel is ~ y^2 U''/6 ~ y^2 x^2/6.
YS_COEF = 0.02
# Beyond this y the Green's function is the Wien spectrum to ~1e-13 (e^{-2y})
Y_WIEN = 15.
# Survival fractions below this are round-off; treated as fully absorbed
S_FLOOR = 1.e-12


def lnw(s):
    """log of the Wien equilibrium per ln x: ln(x^3 e^-x) = -V(s)"""
    return 3.*s - np.exp(s)


def morse_u(s):
    """Morse potential U = V'^2/4 - V''/2 of the symmetrised problem"""
    x = np.exp(s)
    return 0.25*x*x - 2.*x + 2.25


def _line_mean_exp(s0, ds, k):
    """mean of exp(k s) along the straight line from s0 to s0+ds"""
    kd = k*ds
    small = np.abs(kd) < 1.e-8
    kd_safe = np.where(small, 1., kd)
    return np.exp(k*s0)*np.where(small, 1.+0.5*kd, np.expm1(kd_safe)/kd_safe)


def freefree_shape(x):
    """
    Free-free absorption vs x = h nu/kT (Gaunt factor 1, stimulated emission
    included), normalised to 1 at x = 1.  Same as compton_greens.freefree.
    """
    return -np.expm1(-x)/x**3/(1.-np.exp(-1.))


def lam_freefree(rho, temp, heabund=0.09):
    """
    Absorption parameter Lam = kappa_ff(x=1)/(kappa_es theta) for a fully
    ionised H/He plasma, so that the sink in the Kompaneets equation per
    unit y is Lam*freefree_shape(x).  Uses the constants of
    compton_greens.freefreecgs.  rho in g/cm^3, temp in K.
    """
    ffnrm = 3.692146e8
    mp = 1.6726e-24
    h = 6.6262e-27
    kb = 1.3807e-16
    me = 9.1093897e-28
    c = 2.9979246e10
    sigmat = 6.65248e-25
    nh = rho/(mp*(1.+4.*heabund))
    nhe = nh*heabund
    ne = nh+2.*nhe
    nu1 = kb*temp/h
    alpha1 = ne*(nh+4.*nhe)*ffnrm/np.sqrt(temp)/nu1**3*(1.-np.exp(-1.))
    theta = kb*temp/(me*c*c)
    return alpha1/(ne*sigmat*theta)


def ys_switch(x0, coef=YS_COEF, sink=None, lam=0., kerr=5.e-5, lcoef=0.1,
              ysmin=1.e-4):
    """
    y below which the short-time kernel is used for initial energy x0.
    With absorption the kernel's error in the survival grows as
    (2/3) lam a y^2 for a ~ x^-2, so y_s is also limited to
    sqrt(1.5 kerr/(lam a(x0))), and to lcoef/(lam a(x0)) (absorption depth
    across the kernel step), but not below ysmin: photons that are strongly
    absorbed are dead before y_s anyway.
    """
    ys = coef/np.maximum(1., x0)
    if sink is not None and lam > 0.:
        la = lam*sink(x0)
        ys = np.minimum(ys, np.maximum(np.minimum(np.sqrt(1.5*kerr/la),
                                                  lcoef/la), ysmin))
    return ys


def _sink_line_mean(s0, ds, sink):
    """8-point Gauss-Legendre mean of sink(x) along the line s0 -> s0+ds"""
    t, wt = np.polynomial.legendre.leggauss(8)
    t = 0.5*(t+1.)
    wt = 0.5*wt
    return sum(wi*sink(np.exp(s0+ti*ds)) for ti, wi in zip(t, wt))


def _local_grid(xi, y, nloc=2001):
    """grid in s covering the short-time kernel from ln xi at y"""
    s0 = np.log(xi)
    wid = np.sqrt(2.*y)
    mu = s0 + (3.-xi)*y
    return np.linspace(min(s0, mu)-14.*wid, max(s0, mu)+14.*wid, nloc)


def _kernel_lnsurv(xi, y, sink, lam, nloc=2001):
    """
    ln of the survival fraction of the short-time kernel.  From the absorbed
    fraction int P0 (1 - exp(-y lam <a>_line)) / int P0 while most photons
    survive (no cancellation when little is absorbed); otherwise in log
    space, so it stays finite however strong the absorption.
    """
    if sink is None or lam <= 0.:
        return 0.
    s0 = np.log(xi)
    sl = _local_grid(xi, y, nloc)
    p0 = short_time_kernel(sl, y, s0)
    tau = y*lam*_sink_line_mean(s0, sl-s0, sink)
    n0 = np.trapezoid(p0, sl)
    fab = np.trapezoid(p0*(-np.expm1(-tau)), sl)/n0
    if fab < 0.5:
        return np.log1p(-fab)
    # log-space trapezoid of p0*exp(-tau)
    lg = np.log(np.maximum(p0, 1.e-300)) - tau
    m = lg.max()
    return m + np.log(np.trapezoid(np.exp(lg-m), sl)/n0)


def short_time_kernel(s, y, s0, sink=None, lam=0.):
    """
    Heat-kernel (Wigner-Kirkwood) short-time approximation to P(s,y|s0):

        P = (4 pi y)^-1/2 exp[-(s-s0)^2/4y - (V(s)-V(s0))/2 - y <U>_line]

    with <U> the mean of U along the line from s0 to s.  Exact for x << 1
    (lognormal of Zel'dovich & Sunyaev 1969); relative error ~ y^2 x^2/6.
    If sink is given, lam*sink(x) is added to U (absorption).
    """
    s = np.asarray(s, dtype=float)
    ds = s - s0
    ubar = (0.25*_line_mean_exp(s0, ds, 2.) - 2.*_line_mean_exp(s0, ds, 1.)
            + 2.25)
    if sink is not None and lam > 0.:
        ubar = ubar + lam*_sink_line_mean(s0, ds, sink)
    lnp = (-ds*ds/(4.*y) + 0.5*(lnw(s)-lnw(s0)) - y*ubar
           - 0.5*np.log(4.*np.pi*y))
    return np.exp(lnp)


def make_grid(xmin=1.e-9, xmax=150., hmax=0.01, hdrift=0.05, coef=YS_COEF,
              nres=8., sink=None, lam=0., xres_min=1.e-4):
    """
    Non-uniform grid in s = ln x.  Spacing is hmax at small x; at large x it
    is limited by hdrift/x, since the discretisation error is set by
    h*|V'| ~ h*x (second order), and by the requirement that the short-time
    kernel used as the seed at y_s(x) (width sqrt(2 y_s)) spans ~nres points.
    With a sink, y_s shrinks where absorption is strong; that resolution is
    only enforced for x > xres_min, since no photons are seeded below it.
    """
    smin, smax = np.log(xmin), np.log(xmax)
    s = [smin]
    while s[-1] < smax:
        x = np.exp(s[-1])
        ys = ys_switch(x, coef, sink, lam) if x > xres_min else \
            ys_switch(x, coef)
        h = min(hmax, hdrift/max(1., x), np.sqrt(2.*ys)/nres)
        s.append(s[-1]+h)
    s = np.array(s)
    # stretch so the last node lands on smax
    return smin + (s-smin)*(smax-smin)/(s[-1]-smin)


class Propagator:
    """
    Discrete Kompaneets operator on a grid in s, with zero-flux boundaries.

    Flux (Scharfetter-Gummel, equivalent to Chang & Cooper 1970 for this
    problem): F_{j+1/2} = -(wbar/h)(P_{j+1}/w_{j+1} - P_j/w_j), wbar the
    logarithmic mean of w.  This conserves photon number to round-off, keeps
    the scheme positive and makes the Wien spectrum an exact discrete
    equilibrium.  Time integration is TR-BDF2 (second order, L-stable) on P
    itself, so all quantities stay O(1); an eigendecomposition of the
    symmetrised operator would need variables scaled by w^-1/2, which span
    ~60 decades and amplify round-off.
    """

    def __init__(self, s=None, sink=None, lam=0., dymax=0.02, dyfrac=0.02):
        if s is None:
            s = make_grid(sink=sink, lam=lam)
        self.s = s
        self.x = np.exp(s)
        self.dymax = dymax
        self.dyfrac = dyfrac
        n = len(s)
        h = np.diff(s)
        cw = np.empty(n)
        cw[1:-1] = 0.5*(s[2:]-s[:-2])
        cw[0] = 0.5*h[0]
        cw[-1] = 0.5*h[-1]
        self.cw = cw
        d = np.diff(lnw(s))
        small = np.abs(d) < 1.e-10
        dsafe = np.where(small, 1., d)
        # face coefficients wbar/(h w_j) and wbar/(h w_{j+1})
        cl = np.where(small, 1.+0.5*d, np.expm1(dsafe)/dsafe)/h
        cr = np.where(small, 1.-0.5*d, -np.expm1(-dsafe)/dsafe)/h
        # banded operator A (dP/dy = A P): rows 0 upper, 1 diag, 2 lower
        ab = np.zeros((3, n))
        ab[0, 1:] = cr/cw[:-1]
        ab[1, :-1] -= cl/cw[:-1]
        ab[1, 1:] -= cr/cw[1:]
        ab[2, :-1] = cl/cw[1:]
        self.absvec = np.zeros(n)
        if sink is not None and lam > 0.:
            self.absvec = lam*sink(self.x)
            ab[1] -= self.absvec
        self.ab = ab

    def _apply(self, p):
        ab = self.ab
        out = ab[1]*p
        out[:-1] += ab[0, 1:]*p[1:]
        out[1:] += ab[2, :-1]*p[:-1]
        return out

    def _solve(self, c, rhs):
        """solve (I - c A) u = rhs"""
        m = -c*self.ab
        m[1] += 1.
        return solve_banded((1, 1), m, rhs)

    def _step(self, p, dy):
        g = 2.-np.sqrt(2.)
        ps = self._solve(0.5*g*dy, p+0.5*g*dy*self._apply(p))
        rhs = (ps - (1.-g)**2*p)/(g*(2.-g))
        return self._solve((1.-g)/(2.-g)*dy, rhs)

    def propagate(self, p0, y, absorbed=False):
        """
        Evolve the nodal density p0 (per unit s) by each y in the (sorted)
        array y.  Steps grow geometrically (dyfrac of the elapsed y, capped
        at dymax) and land exactly on each output y.
        Returns array [len(y), len(s)]; with absorbed=True also the number
        absorbed by each y, integrated directly (trapezoid in y) so that it
        stays accurate when it is tiny.
        """
        y = np.atleast_1d(y)
        out = np.zeros((len(y), len(p0)))
        absn = np.zeros(len(y))
        wabs = self.absvec*self.cw
        p = p0.copy()
        r = wabs @ p
        acc = 0.
        t = 0.
        dy0 = max(1.e-3*y[y > 0].min(), 1.e-9) if np.any(y > 0) else 0.
        for k, yk in enumerate(y):
            while t < yk*(1.-1.e-13):
                dy = min(max(self.dyfrac*t, dy0), self.dymax, yk-t)
                p = self._step(p, dy)
                rn = wabs @ p
                acc += 0.5*(r+rn)*dy
                r = rn
                t += dy
            out[k] = p
            absn[k] = acc
        out = np.maximum(out, 0.)
        return (out, absn) if absorbed else out

    def number(self, p):
        """photon number (integral of P ds) of nodal densities p[..., s]"""
        return p @ self.cw


def greens(xi, y, prop=None, sink=None, lam=0., coef=YS_COEF, nloc=2001,
           lnsurv=False):
    """
    Green's function P(s, y | ln xi) per unit ln xf for each y.

    Returns a list of (s, P) pairs.  For y < y_s(xi) the short-time kernel is
    evaluated on a local grid; otherwise the discrete propagator is seeded
    with the kernel at y_s and advanced by y - y_s.  With a sink, the
    integral of P ds is the survival fraction; lnsurv=True also returns
    ln(survival) computed from the absorbed fraction, which stays accurate
    when almost nothing is absorbed.
    """
    if prop is None:
        prop = Propagator(sink=sink, lam=lam)
    y = np.atleast_1d(y)
    s0 = np.log(xi)
    ys = ys_switch(xi, coef, sink, lam)
    out = [None]*len(y)
    lns = np.zeros(len(y))
    small = y < ys
    for i in np.nonzero(small)[0]:
        sl = _local_grid(xi, y[i], nloc)
        out[i] = (sl, short_time_kernel(sl, y[i], s0, sink, lam))
        if lnsurv:
            lns[i] = _kernel_lnsurv(xi, y[i], sink, lam, nloc)
    if np.any(~small):
        seed = short_time_kernel(prop.s, ys, s0, sink, lam)
        if sink is None or lam <= 0.:
            seed /= prop.number(seed)
        big = np.nonzero(~small)[0]
        p, absn = prop.propagate(seed, y[big]-ys, absorbed=True)
        for k, i in enumerate(big):
            out[i] = (prop.s, p[k])
        if lnsurv:
            # fraction of the seed surviving: from the absorbed number while
            # most survive, directly from the remaining number otherwise
            nseed = max(prop.number(seed), 1.e-300)
            rem = prop.number(p)/nseed
            fab = absn/nseed
            with np.errstate(divide="ignore"):
                lnf = np.where(rem > 0.5, np.log1p(-np.minimum(fab, 0.5)),
                               np.log(np.maximum(rem, 1.e-300)))
            lns[big] = _kernel_lnsurv(xi, ys, sink, lam, nloc) + lnf
    return (out, lns) if lnsurv else out


# ---------------------------------------------------------------------------
# Quantile tables for Monte Carlo sampling
# ---------------------------------------------------------------------------

def quantile_levels(nq=257, pmin=1.e-7):
    """Quantile levels, dense in both tails (uniform in logit p)"""
    lo = np.log(pmin/(1.-pmin))
    z = np.linspace(lo, -lo, nq)
    p = 1./(1.+np.exp(-z))
    p[0], p[-1] = 0., 1.
    return p


def _quantiles(s, p, levels):
    """Inverse CDF of density p(s) at the given levels (trapezoid CDF)"""
    c = np.concatenate([[0.], np.cumsum(0.5*(p[1:]+p[:-1])*np.diff(s))])
    total = c[-1]
    c /= total
    # tails beyond the support: use the first/last grid point with p > 0
    nz = np.nonzero(p > 0.)[0]
    i0, i1 = max(nz[0]-1, 0), min(nz[-1]+1, len(s)-1)
    q = np.interp(levels, c[i0:i1+1], s[i0:i1+1])
    return q, total


def _table_slice(args):
    """quantile/absorption tables for one value of lam (worker for Pool)"""
    y, xi, levels, sink, lam = args
    if lam <= 0.:
        sink = None
    prop = Propagator(sink=sink, lam=lam)
    uq = np.zeros((len(y), len(xi), len(levels)), dtype=np.float32)
    abar = np.zeros((len(y), len(xi)))
    dead = np.zeros((len(y), len(xi)), dtype=bool)
    for j, x0 in enumerate(xi):
        res, lns = greens(x0, y, prop=prop, sink=sink, lam=lam, lnsurv=True)
        for i, (s, p) in enumerate(res):
            # below ~1e-12 the propagated density is round-off noise:
            # treat the photons as absorbed
            dead[i, j] = not lns[i] > np.log(S_FLOOR)
            if not dead[i, j]:
                q, _ = _quantiles(s, p, levels)
                uq[i, j] = (q-np.log(x0))/np.sqrt(2.*y[i])
            else:
                # keep the shape from the previous y
                uq[i, j] = uq[i-1, j]*np.sqrt(y[i-1]/y[i]) if i > 0 else 0.
            if lam > 0.:
                abar[i, j] = -max(lns[i], -700.)/(lam*y[i])
    return uq, abar, dead


def gen_quantile_table(filename="kgreens_table.npz", ny=73, nxi=61,
                       ylim=(1.e-6, Y_WIEN), xilim=(1.e-3, 60.), nq=257,
                       lams=None, sink=freefree_shape, nproc=1,
                       verbose=True):
    """
    Tabulate quantiles of u = ln(xf/xi)/sqrt(2y) on a (lam, y, xi, level)
    grid, with lam the absorption parameter of sink (default free-free,
    lam = kappa_ff(x=1)/(kappa_es theta); see lam_freefree).  lams=None
    gives a single lam = 0 slice (no absorption).  Otherwise lam = 0 is
    prepended to lams.  Scaling by sqrt(2y) keeps the table smooth as
    y -> 0.  The survival fraction S is stored as the mean absorption
    abar = -ln S/(lam y), which varies slowly with lam.
    Slices are computed in parallel with nproc processes (the calling
    script needs an `if __name__ == "__main__":` guard when nproc > 1).
    """
    y = np.logspace(np.log10(ylim[0]), np.log10(ylim[1]), ny)
    xi = np.logspace(np.log10(xilim[0]), np.log10(xilim[1]), nxi)
    levels = quantile_levels(nq)
    lam = np.zeros(1) if lams is None else \
        np.concatenate([[0.], np.sort(np.asarray(lams, dtype=float))])
    args = [(y, xi, levels, sink, l) for l in lam]
    if nproc > 1:
        from multiprocessing import Pool
        with Pool(nproc) as pool:
            res = []
            for k, r in enumerate(pool.imap(_table_slice, args)):
                res.append(r)
                if verbose:
                    print("lam = {:.3e} done".format(lam[k]), flush=True)
    else:
        res = []
        for k, a in enumerate(args):
            res.append(_table_slice(a))
            if verbose:
                print("lam = {:.3e} done".format(lam[k]), flush=True)
    uq = np.array([r[0] for r in res])
    abar = np.array([r[1] for r in res])
    dead = np.array([r[2] for r in res])
    np.savez(filename, lam=lam, y=y, xi=xi, levels=levels, uq=uq,
             abar=abar, dead=dead, ys_coef=YS_COEF,
             sink=getattr(sink, "__name__", "unknown"))
    return filename


def _axis_weights(grid, v):
    """lower index and linear weight of v on a sorted grid, clamped"""
    v = np.clip(v, grid[0], grid[-1])
    i = int(np.clip(np.searchsorted(grid, v)-1, 0, len(grid)-2))
    return i, (v-grid[i])/(grid[i+1]-grid[i])


def _cubic_weights(grid, v):
    """
    indices and 4-point Lagrange weights of v on a sorted grid (linear in
    the first and last interval), clamped to the grid
    """
    i, t = _axis_weights(grid, v)
    if i == 0 or i >= len(grid)-2:
        return [i, i+1], [1.-t, t]
    x = grid[i-1:i+3]
    v = min(max(v, grid[0]), grid[-1])
    w = []
    for k in range(4):
        wk = 1.
        for m in range(4):
            if m != k:
                wk *= (v-x[m])/(x[k]-x[m])
        w.append(wk)
    return list(range(i-1, i+3)), w


class Sampler:
    """
    Samples xf given xi, y (and the absorption parameter lam) from a quantile
    table: linear interpolation in (ln lam, ln y, ln xi) of the scaled
    quantiles, then in the level.  Between lam = 0 and the smallest positive
    table lam, interpolation is linear in lam.

    y < table min: small-y Gaussian limit in ln xf, S = exp(-lam a(xi) y).
    y > table max: the energy distribution has reached its y-independent
    form (Wien without absorption, the slowest-decaying absorbed mode with
    it), so the ln xf quantiles at y_max are used and ln S is extrapolated
    linearly in y.
    """

    def __init__(self, filename="kgreens_table.npz", sink=freefree_shape,
                 rng=None):
        t = np.load(filename)
        self.lam = t["lam"]
        self.y = t["y"]
        self.lny = np.log(self.y)
        self.lnxi = np.log(t["xi"])
        self.levels = t["levels"]
        self.uq = t["uq"]
        self.abar = t["abar"]
        self.dead = t["dead"]
        self._lnabar = np.log(np.maximum(self.abar, 1.e-300))
        self.lnlam = np.log(self.lam[1:]) if len(self.lam) > 1 else None
        self.sink = sink
        self.rng = np.random.default_rng() if rng is None else rng

    def _lam_weights(self, lam):
        """list of (slice index, weight) for lam"""
        if lam <= 0. or self.lnlam is None:
            return [(0, 1.)]
        if lam <= self.lam[1]:
            b = lam/self.lam[1]
            return [(0, 1.-b), (1, b)]
        k, b = _axis_weights(self.lnlam, np.log(lam))
        return [(k+1, 1.-b), (k+2, b)]

    def _interp(self, table, lamw, iy, a, ix, b):
        out = 0.
        for k, wk in lamw:
            t = table[k]
            out = out + wk*((1-a)*(1-b)*t[iy, ix] + a*(1-b)*t[iy+1, ix]
                            + (1-a)*b*t[iy, ix+1] + a*b*t[iy+1, ix+1])
        return out

    def _abar(self, xi, y, lam):
        """
        mean absorption abar(lam, y, xi): interpolation of ln abar, cubic in
        ln lam and ln xi (the survival error scales with lam*y), linear in
        ln y.  Below the smallest positive table lam, abar is held fixed.
        """
        ll = np.log(max(lam, self.lam[1]))
        il, wl = _cubic_weights(self.lnlam, ll)
        ix, wx = _cubic_weights(self.lnxi, np.log(xi))
        iy, a = _axis_weights(self.lny, np.log(y))
        # near fully absorbed cells ln abar is not smooth: go linear
        if self.dead[np.ix_(np.array(il)+1, [iy, iy+1], ix)].any():
            k, b = _axis_weights(self.lnlam, ll)
            il, wl = [k, k+1], [1.-b, b]
            j, c = _axis_weights(self.lnxi, np.log(xi))
            ix, wx = [j, j+1], [1.-c, c]
        out = 0.
        for k, wk in zip(il, wl):
            t = self._lnabar[k+1]
            for j, wj in zip(ix, wx):
                out += wk*wj*((1.-a)*t[iy, j]+a*t[iy+1, j])
        return np.exp(out)

    def survival(self, xi, y, lam):
        """fraction of photons not absorbed after y"""
        if lam <= 0. or self.lnlam is None:
            return 1.
        if y < self.y[0]:
            return np.exp(-lam*self.sink(xi)*y)
        if y > self.y[-1]:
            # ln S linear in y beyond the table
            y1, y2 = self.y[-2], self.y[-1]
            l1 = -lam*y1*self._abar(xi, y1, lam)
            l2 = -lam*y2*self._abar(xi, y2, lam)
            return np.exp(l2+(l2-l1)/(y2-y1)*(y-y2))
        return np.exp(-lam*y*self._abar(xi, y, lam))

    def quantile(self, xi, y, p, lam=0.):
        """xf at cumulative probability p of the surviving photons"""
        if y < self.y[0]:
            from scipy.special import erfinv
            z = np.sqrt(2.)*erfinv(2.*np.asarray(p)-1.)
            return xi*np.exp((3.-xi)*y + np.sqrt(2.*y)*z)
        ye = min(y, self.y[-1])
        iy, a = _axis_weights(self.lny, np.log(ye))
        ix, b = _axis_weights(self.lnxi, np.log(xi))
        u = self._interp(self.uq, self._lam_weights(lam), iy, a, ix, b)
        return xi*np.exp(np.interp(p, self.levels, u)*np.sqrt(2.*ye))

    def sample(self, xi, y, lam=0., size=None):
        """
        Returns (xf, S): sampled final energies of surviving photons and
        the survival fraction, to be used as a weight (or for Russian
        roulette) by the Monte Carlo.
        """
        return (self.quantile(xi, y, self.rng.random(size), lam),
                self.survival(xi, y, lam))


# ---------------------------------------------------------------------------
# Becker (2003) exact solution, for validation
# ---------------------------------------------------------------------------

def becker_greens(x, x0, y, dps=30, tol=1.e-16):
    """
    Exact Green's function (Becker 2003, eq. 33), normalised so that
    int x^2 f dx = 1, returned per unit ln x: P = x^3 f.

    Uses the closed-form weight u sinh(pi u)/((1+4u^2)(9+4u^2)) and the
    scaled Whittaker function e^{pi u/2} W_{2,iu} evaluated by mpmath at
    elevated precision; the u-integral is cut where exp(-u^2 y) < tol.
    Slow; intended for y >~ 0.05.
    """
    import mpmath as mp
    mp.mp.dps = dps
    x, x0, y = mp.mpf(x), mp.mpf(x0), mp.mpf(y)
    umax = mp.sqrt(-mp.log(tol)/y)

    def integrand(u):
        if u == 0:
            return mp.mpf(0)
        wx = mp.whitw(2, 1j*u, x)
        wx0 = mp.whitw(2, 1j*u, x0)
        return (mp.exp(-u*u*y)*u*mp.sinh(mp.pi*u)
                / ((1+4*u*u)*(9+4*u*u))*mp.re(wx*wx0))

    # split into panels so the oscillation in u*ln x is resolved
    npan = int(max(8, float(umax)*max(1., abs(float(mp.log(x))),
                                     abs(float(mp.log(x0))))/2.))
    pts = mp.linspace(0, umax, npan+1)
    integ = mp.quad(integrand, pts)
    cont = (32/mp.pi*mp.exp(-9*y/4)/(x0*x0*x*x)*mp.exp((x0-x)/2)*integ)
    disc = (mp.exp(-x)/2
            + mp.exp(-x-2*y)*(2-x)*(2-x0)/(2*x0*x))
    return float(x**3*(cont+disc))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate Kompaneets Green's "
                                 "function quantile tables")
    ap.add_argument("--out", default="kgreens_table.npz")
    ap.add_argument("--absorption", action="store_true",
                    help="include free-free absorption (lam dimension)")
    ap.add_argument("--lamlim", type=float, nargs=2, default=[1.e-10, 1.e3])
    ap.add_argument("--nlam", type=int, default=27)
    ap.add_argument("--ny", type=int, default=73)
    ap.add_argument("--nxi", type=int, default=61)
    ap.add_argument("--nq", type=int, default=257)
    ap.add_argument("--nproc", type=int, default=1)
    arg = ap.parse_args()
    lams = np.logspace(np.log10(arg.lamlim[0]), np.log10(arg.lamlim[1]),
                       arg.nlam) if arg.absorption else None
    gen_quantile_table(arg.out, ny=arg.ny, nxi=arg.nxi, nq=arg.nq, lams=lams,
                       nproc=arg.nproc)
