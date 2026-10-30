#! /usr/bin/env python
"""
The frequency interface of the DDMC line lattice in one dimension, without transport:
a diffusion in x on [0, L] with coefficient D = 1/2 (the Lyman alpha wing kernel, unit
variance per scattering, drift and gradient of the coefficient cancelling), born in
[0, dx], reflecting at 0 and absorbed at L.  Below xe the walk is the lattice (bins of
width dx, the finite-difference rate 1 / (2 dx^2) into each neighbour per unit time),
above xe it is kinetic (Gaussian unit steps at unit rate).  The mean time to absorption
is compared with the exact (L^2 - x0^2) / (2 D) for the entry rules at xe:

  P        the probability that a kinetic step across xe converts to a lattice walker
           in the edge bin (the rest reflect about xe), 1 or sqrt(2 pi) / (2 dx)
  place    where a walker leaking out of the edge bin is put: uniform in the kinetic
           bin [xe, xe + dx], or at xe plus the outward overshoot of a step (a uniform
           fraction of a Rayleigh step), the landing point of a kinetic crossing from a
           uniform population

Only P = sqrt(2 pi) / (2 dx) with the overshoot placement converges to the exact value
as dx shrinks; converting every crossing (P = 1) holds the walk on the lattice too long,
and a uniform placement puts the emerged walker half a bin too far out.  This is what
DDMCLineEntry and DDMCLineEmerge do (R1 of compton_ddmc_plan.md).

    python freq_interface_toy.py [--dx 10 5 2.5] [--nwalk 40000]
"""
import argparse
import numpy as np


def hybrid(rng, L, xe, dx, pf, place, n):
    nb = int(round(xe/dx))
    r = 1./(2.*dx*dx)
    on_lat = np.ones(n, bool)
    b = np.zeros(n, int)
    x = np.zeros(n)
    t = np.zeros(n)
    alive = np.ones(n, bool)
    while alive.any():
        m = alive & on_lat
        if m.any():
            idx = np.nonzero(m)[0]
            bb = b[idx]
            rate = np.where(bb > 0, 2.*r, r)
            t[idx] += rng.exponential(1./rate)
            down = (bb > 0) & (rng.uniform(size=idx.size) < 0.5)
            bb_new = np.where(down, bb - 1, bb + 1)
            em = bb_new >= nb
            b[idx] = np.where(em, bb, bb_new)
            e = idx[em]
            if e.size:
                on_lat[e] = False
                if place == 'uniform':
                    x[e] = xe + rng.uniform(0., dx, e.size)
                else:
                    x[e] = xe + rng.uniform(size=e.size)*np.sqrt(
                        -2.*np.log(rng.uniform(size=e.size)))
        m = alive & ~on_lat
        if m.any():
            idx = np.nonzero(m)[0]
            t[idx] += rng.exponential(1., idx.size)
            xn = x[idx] + rng.normal(size=idx.size)
            absorbed = xn >= L
            alive[idx[absorbed]] = False
            cross = (xn < xe) & ~absorbed
            conv = cross & (rng.uniform(size=idx.size) < pf)
            on_lat[idx[conv]] = True
            b[idx[conv]] = nb - 1
            xn = np.where(cross & ~conv, 2.*xe - xn, xn)
            x[idx] = xn
    return t.mean(), t.std()/np.sqrt(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dx', type=float, nargs='+', default=[10., 5., 2.5])
    ap.add_argument('--nwalk', type=int, default=40000)
    ap.add_argument('--L', type=float, default=120.)
    ap.add_argument('--xe', type=float, default=60.)
    args = ap.parse_args()
    rng = np.random.default_rng(2)
    D = 0.5
    x0 = rng.uniform(0., 1., 100000)
    for dx in args.dx:
        exact = np.mean((args.L**2 - (x0*dx)**2)/(2.*D))
        print(f"dx {dx:5.2f} ({args.xe/dx:.0f} lattice bins), exact time {exact:.0f}")
        for name, pf in (('P = 1', 1.), ('P = sqrt(2 pi)/(2 dx)', np.sqrt(2.*np.pi)/(2.*dx))):
            for place in ('uniform', 'overshoot'):
                m, s = hybrid(rng, args.L, args.xe, dx, pf, place, args.nwalk)
                print(f"   {name:22s} placed {place:9s}: {m/exact:.3f} +- {s/exact:.3f}"
                      " of exact")


if __name__ == '__main__':
    main()
