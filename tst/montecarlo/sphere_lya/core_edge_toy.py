#! /usr/bin/env python
"""
The core edge of the frequency lattice placed inside the Doppler core, frequency only.

A kinetic core visit to |x| < xc: begins when a photon at |x| uniform in [xc, xc + 1]
scatters into the core, ends at the first scattering that leaves it.  Measured per xc
and a: the visit length N (scatterings), the exit overshoot d = |x'| - xc of the leaving
scattering (its mean, quantiles, and the fit of an exponential), and the frequency the
visit entered at.

Then the hybrid walk: outside the core the kinetic kernel; inside, a dwell of
exponential length with mean N followed by an emergence at xc + d, with d drawn from
(a) the measured exit distribution, (b) an exponential of the measured mean, (c) the
present rule, a uniform fraction of a Rayleigh step of unit variance.  The observable is
the mean number of scatterings and of core visits to first reach |x| >= X from a start
in the core, against the fully kinetic walk.

    python core_edge_toy.py [--a 4.7e-4 1.5e-3] [--xc 1.5 2 2.5 3 3.3] [--nvisit 2000]
"""
import argparse
import time

import numpy as np

from lya_kernel import kernel_scatter  # noqa: E402


def core_visits(rng, a, xc, nvisit):
    """nvisit kinetic core visits in parallel: lengths, exit overshoots, entry |x|"""
    # entries: wing photons uniform in [xc, xc + 1] scattered until they land inside
    x = np.empty(nvisit)
    todo = np.ones(nvisit, bool)
    while todo.any():
        t = np.nonzero(todo)[0]
        xn = kernel_scatter(rng, a, xc + rng.uniform(size=t.size))
        ok = np.abs(xn) < xc
        x[t[ok]] = xn[ok]
        todo[t[ok]] = False
    xin = np.abs(x).copy()
    n = np.zeros(nvisit, np.int64)
    over = np.empty(nvisit)
    alive = np.ones(nvisit, bool)
    while alive.any():
        idx = np.nonzero(alive)[0]
        xn = kernel_scatter(rng, a, x[idx])
        n[idx] += 1
        out = np.abs(xn) >= xc
        over[idx[out]] = np.abs(xn[out]) - xc
        alive[idx[out]] = False
        x[idx] = xn
    return n, over, xin


def hybrid_reach(rng, a, xc, X, nvisit_mean, exit_draw, nwalk, nmax=10**9):
    """scatterings and core visits to first reach |x| >= X, starting with a kinetic
    photon just emerged from the core (one visit counted); exit_draw(size) gives d"""
    nsc = np.zeros(nwalk)
    nvis = np.ones(nwalk)
    x = xc + exit_draw(nwalk)
    x *= np.where(rng.uniform(size=nwalk) < 0.5, -1., 1.)
    # an exit that lands beyond X has reached it
    alive = np.abs(x) < X
    it = 0
    while alive.any() and it < nmax:
        it += 1
        idx = np.nonzero(alive)[0]
        xn = kernel_scatter(rng, a, x[idx])
        nsc[idx] += 1
        done = np.abs(xn) >= X
        alive[idx[done]] = False
        back = (np.abs(xn) < xc) & ~done
        b = idx[back]
        if b.size:
            nsc[b] += rng.exponential(nvisit_mean, b.size)
            nvis[b] += 1
            d = exit_draw(b.size)
            xn[back] = (xc + d)*np.where(rng.uniform(size=b.size) < 0.5, -1., 1.)
            alive[b[d >= X - xc]] = False
        x[idx] = xn
    return nsc.mean(), nsc.std()/np.sqrt(nwalk), nvis.mean(), nvis.std()/np.sqrt(nwalk)


def kinetic_reach(rng, a, xc, X, nwalk, start):
    """the same, fully kinetic: start at the given entry frequencies in the core"""
    nsc = np.zeros(nwalk)
    nvis = np.ones(nwalk)
    x = start.copy()
    inside = np.ones(nwalk, bool)
    alive = np.ones(nwalk, bool)
    while alive.any():
        idx = np.nonzero(alive)[0]
        xn = kernel_scatter(rng, a, x[idx])
        nsc[idx] += 1
        done = np.abs(xn) >= X
        alive[idx[done]] = False
        now_in = np.abs(xn) < xc
        entered = now_in & ~inside[idx] & ~done
        nvis[idx[entered]] += 1
        inside[idx] = now_in
        x[idx] = xn
    return nsc.mean(), nsc.std()/np.sqrt(nwalk), nvis.mean(), nvis.std()/np.sqrt(nwalk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', type=float, nargs='+', default=[4.7e-4, 1.5e-3])
    ap.add_argument('--xc', type=float, nargs='+', default=[1.5, 2., 2.5, 3., 3.3])
    ap.add_argument('--nvisit', type=int, default=2000)
    ap.add_argument('--nwalk', type=int, default=4000)
    ap.add_argument('--dX', type=float, nargs='+', default=[1., 2.])
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    for a in args.a:
        print(f"\n=== a = {a:.2e}")
        print("   xc     N_visit    +-   1/erfc(xc)   <d>   d50    d90   d99   "
              "max  <entry |x|>   kinetic ms/scat")
        exits = {}
        for xc in args.xc:
            t0 = time.time()
            n, over, xin = core_visits(rng, a, xc, args.nvisit)
            dt = time.time() - t0
            q = np.quantile(over, [0.5, 0.9, 0.99])
            from scipy.special import erfc
            print(f"  {xc:4.2f} {n.mean():10.1f} {n.std()/np.sqrt(n.size):6.1f} "
                  f"{1./erfc(xc):10.1f} {over.mean():6.3f} {q[0]:6.3f} {q[1]:6.3f} "
                  f"{q[2]:6.3f} {over.max():6.2f} {xin.mean():8.3f}  "
                  f"{1e3*dt/max(n.sum(), 1):.4f}")
            exits[xc] = (n.mean(), over)
        print("\n  hybrid against kinetic: scatterings and core visits to first reach "
              "|x| >= X")
        print("    xc    X   kinetic nsc (visits)  |  exit=sample  |  exit=exp(<d>)  |  "
              "exit=Rayleigh*U")
        for xc in args.xc:
            nmean, over = exits[xc]
            for dX in args.dX:
                X = xc + dX
                # kinetic: start from entry frequencies, by the visit machinery
                _, _, xin = core_visits(rng, a, xc, args.nwalk)
                start = xin*np.where(rng.uniform(size=args.nwalk) < 0.5, -1., 1.)
                k = kinetic_reach(rng, a, xc, X, args.nwalk, start)
                draws = {
                    'sample': lambda m: rng.choice(over, m),
                    'exp': lambda m: rng.exponential(over.mean(), m),
                    'rayleigh': lambda m: rng.uniform(size=m)
                    * np.sqrt(-2.*np.log(rng.uniform(size=m))),
                }
                row = f"  {xc:4.2f} {X:4.1f} {k[0]:10.0f} ({k[2]:6.2f})"
                for name in ('sample', 'exp', 'rayleigh'):
                    h = hybrid_reach(rng, a, xc, X, nmean, draws[name], args.nwalk)
                    row += f" | {h[0]/k[0]:5.3f} ({h[2]/k[2]:5.3f})"
                print(row)
    print()


if __name__ == '__main__':
    main()
