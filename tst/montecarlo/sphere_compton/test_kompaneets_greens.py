
"""
Tests for kompaneets_greens.py.  Run with `python test_kompaneets_greens.py`
(or pytest).  The Becker comparison uses mpmath and takes ~1 minute; skip it
with `python test_kompaneets_greens.py fast`.
"""

import sys
import numpy as np
import os
import sys
_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      '..', '..', '..', 'vis', 'python', 'montecarlo'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'problem_specific', 'sphere_compton'))
import kompaneets_greens as kg  # noqa: E402

PROP = None


def prop():
    global PROP
    if PROP is None:
        PROP = kg.Propagator()
    return PROP


XI = [1.e-3, 0.1, 1., 5., 20., 60.]
Y = np.array([1.e-5, 1.e-3, 0.01, 0.1, 0.5, 1., 3., 10.])


def test_photon_number():
    """integral of P ds = 1 (exact for the propagator, ~y^2 x^2 for kernel)"""
    for xi in XI:
        for y, (s, p) in zip(Y, kg.greens(xi, Y, prop=prop())):
            assert abs(np.trapezoid(p, s)-1.) < 1.e-4, (xi, y)


def test_inverse_energy_moment():
    """exact: <1/x> = 1/2 + (1/xi - 1/2) exp(-2y)  (the s = -2 mode)"""
    for xi in XI:
        for y, (s, p) in zip(Y, kg.greens(xi, Y, prop=prop())):
            m = np.trapezoid(p*np.exp(-s), s)/np.trapezoid(p, s)
            exact = 0.5+(1./xi-0.5)*np.exp(-2.*y)
            assert abs(m/exact-1.) < 2.e-3, (xi, y, m/exact-1.)


def test_wien_limit():
    """large y: P -> x^3 e^-x / 2 per unit ln x"""
    for xi in [0.01, 1., 30.]:
        (s, p), = kg.greens(xi, [kg.Y_WIEN], prop=prop())
        wien = 0.5*np.exp(kg.lnw(s))
        assert np.max(np.abs(p-wien)) < 1.e-4*np.max(wien), xi


def test_lognormal_limit():
    """x << 1: exact lognormal, drift 3 and variance 2y in ln x"""
    xi, y = 1.e-3, 0.05
    (s, p), = kg.greens(xi, [y], prop=prop())
    ds = s-np.log(xi)
    ln = np.exp(-(ds-3.*y)**2/(4.*y))/np.sqrt(4.*np.pi*y)
    assert np.max(np.abs(p-ln)) < 1.e-3*np.max(ln)


def test_kernel_matches_propagator():
    """short-time kernel and propagator agree near the switch"""
    for xi in [0.1, 1., 10., 50.]:
        ys = kg.ys_switch(xi)
        y = 3.*ys
        s = prop().s
        seed = kg.short_time_kernel(s, ys, np.log(xi))
        seed /= prop().number(seed)
        p = prop().propagate(seed, [y-ys])[0]
        k = kg.short_time_kernel(s, y, np.log(xi))
        assert np.max(np.abs(p-k)) < 2.e-3*np.max(k), xi


def test_absorption_survival():
    """a constant sink lam gives survival exp(-lam y) exactly"""
    lam = 0.7
    pa = kg.Propagator(sink=lambda x: np.ones_like(x), lam=lam)
    y = np.array([0.1, 1., 3.])
    for xi in [0.01, 1., 10.]:
        for yy, (s, p) in zip(y, kg.greens(xi, y, prop=pa,
                                           sink=lambda x: np.ones_like(x),
                                           lam=lam)):
            surv = np.trapezoid(p, s)
            assert abs(surv/np.exp(-lam*yy)-1.) < 1.e-3, (xi, yy, surv)


def test_sampler(tmp="kgreens_test_table.npz"):
    """table quantiles reproduce the exact <1/x>, including off-grid xi, y"""
    import os
    kg.gen_quantile_table(tmp, ny=41, nxi=25, ylim=(1.e-3, 10.),
                          xilim=(0.1, 10.), verbose=False)
    smp = kg.Sampler(tmp, rng=np.random.default_rng(1))
    lev = (np.arange(400000)+0.5)/400000
    for xi, y in [(1., 0.1), (0.3, 1.), (10., 0.5), (3., 0.03), (0.15, 2.)]:
        xf = smp.quantile(xi, y, lev)
        exact = 0.5+(1./xi-0.5)*np.exp(-2.*y)
        assert abs(np.mean(1./xf)/exact-1.) < 3.e-3, (xi, y)
    xf, _ = smp.sample(1., 0.1, size=100000)
    assert abs(np.mean(1./xf)/(0.5+0.5*np.exp(-0.2))-1.) < 1.e-2
    os.remove(tmp)


def test_freefree_switch_insensitive():
    """
    with the free-free sink, results at 3 y_s do not depend on whether the
    propagator is seeded at y_s or at y_s/4
    """
    ff = kg.freefree_shape
    for lam, xi in [(1.e-3, 0.03), (1., 0.3), (100., 3.)]:
        pa = kg.Propagator(sink=ff, lam=lam)
        ys = kg.ys_switch(xi, sink=ff, lam=lam)
        out = []
        for y0 in [ys, 0.25*ys]:
            seed = kg.short_time_kernel(pa.s, y0, np.log(xi), ff, lam)
            out.append(pa.propagate(seed, [3.*ys-y0])[0])
        assert np.max(np.abs(out[0]-out[1])) < 1.e-3*np.max(out[1]), \
            (lam, xi)
        assert abs(pa.number(out[0])/pa.number(out[1])-1.) < 1.e-4, \
            (lam, xi)


def test_sampler_absorption(tmp="kgreens_test_table_ff.npz"):
    """survival and energy quantiles at off-grid (lam, y, xi)"""
    import os
    ff = kg.freefree_shape
    kg.gen_quantile_table(tmp, ny=31, nxi=13, ylim=(1.e-3, 3.),
                          xilim=(0.03, 3.), lams=np.logspace(-4, 1, 11),
                          verbose=False)
    smp = kg.Sampler(tmp)
    lev = (np.arange(20000)+0.5)/20000
    for lam, xi, y in [(3.e-3, 0.05, 0.4), (0.2, 0.5, 0.07), (2., 2., 1.3),
                       (5.e-5, 0.1, 1.)]:
        (s, p), = kg.greens(xi, [y], sink=ff, lam=lam)
        surv = np.trapezoid(p, s)
        assert abs(smp.survival(xi, y, lam)/surv-1.) < 2.e-2, (lam, xi, y)
        mean_s = np.trapezoid(p*s, s)/surv
        mean_mc = np.mean(np.log(smp.quantile(xi, y, lev, lam)))
        assert abs(mean_mc-mean_s) < 1.e-2, (lam, xi, y, mean_mc, mean_s)
    os.remove(tmp)


def test_becker_exact():
    """agreement with Becker (2003) eq. 33 to 1e-4 of the peak"""
    for x0, y in [(1., 0.1), (5., 0.5), (0.1, 2.)]:
        (s, p), = kg.greens(x0, [y], prop=prop())
        for xf in [0.5*x0, x0, 2.*x0]:
            b = kg.becker_greens(xf, x0, y, dps=25)
            n = np.interp(np.log(xf), s, p)
            assert abs(n-b) < 1.e-4*np.max(p), (x0, y, xf, n, b)


if __name__ == "__main__":
    fast = len(sys.argv) > 1 and sys.argv[1] == "fast"
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        if fast and t is test_becker_exact:
            continue
        t()
        print("ok  ", t.__name__)
