#! /usr/bin/env python
"""
The core visit defined on the stationary kinetic chain: nchain photons scatter forever in
frequency (the kernel of ScatterResonanceLine); every crossing into |x| < xc opens a
visit and the first scattering that leaves it closes it.  The visits' lengths and exit
overshoots are then those of the stationary process (the entrance distribution is the
chain's own), and by Kac's lemma the mean length is 1 / p_out.  Against the table's
definition (a wing photon uniform in [xc, xc + 1] scattering in), and the hybrid walk
with each exit law against the kinetic walk, as in core_edge_toy.py.

    python chain_toy.py [--a 4.7e-4] [--xc 1.5 2 2.5 3] [--nvisit 2000] [--nwalk 1000]
"""
import argparse
import time

import numpy as np

from core_edge_toy import core_visits, hybrid_reach, kinetic_reach
from lya_kernel import kernel_scatter


def core_chain(rng, a, xc, nvisit, nchain=256):
    """visits of the stationary chain: lengths, exit overshoots, entry |x|"""
    x = rng.normal(size=nchain)*0.7
    inside = np.abs(x) < xc
    nsc = np.zeros(nchain, np.int64)
    lengths, overs, entries = [], [], []
    while len(lengths) < nvisit:
        xn = kernel_scatter(rng, a, x)
        now_in = np.abs(xn) < xc
        # closes: inside before, outside now
        closes = inside & ~now_in
        nsc[inside] += 1
        for i in np.nonzero(closes)[0]:
            lengths.append(nsc[i])
            overs.append(np.abs(xn[i]) - xc)
        opens = ~inside & now_in
        for i in np.nonzero(opens)[0]:
            entries.append(np.abs(xn[i]))
        nsc[~now_in] = 0
        inside = now_in
        x = xn
    return np.array(lengths), np.array(overs), np.array(entries)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', type=float, default=4.7e-4)
    ap.add_argument('--xc', type=float, nargs='+', default=[1.5, 2., 2.5, 3.])
    ap.add_argument('--nvisit', type=int, default=2000)
    ap.add_argument('--nwalk', type=int, default=1000)
    ap.add_argument('--dX', type=float, nargs='+', default=[1., 2.])
    args = ap.parse_args()
    rng = np.random.default_rng(11)
    a = args.a
    print(f"a = {a:.2e}, {args.nvisit} visits per definition")
    print("   xc | table: N  <d>  d50  d90  d99 <entry> | chain: N  <d>  d50  d90  d99 <entry>")
    tabs = {}
    for xc in args.xc:
        t0 = time.time()
        n1, o1, e1 = core_visits(rng, a, xc, args.nvisit)
        n2, o2, e2 = core_chain(rng, a, xc, args.nvisit)
        q1 = np.quantile(o1, [.5, .9, .99]); q2 = np.quantile(o2, [.5, .9, .99])
        print(f"  {xc:4.2f} | {n1.mean():8.1f} {o1.mean():.3f} {q1[0]:.3f} {q1[1]:.3f} {q1[2]:.3f} {e1.mean():.3f}"
              f" | {n2.mean():8.1f} {o2.mean():.3f} {q2[0]:.3f} {q2[1]:.3f} {q2[2]:.3f} {e2.mean():.3f}"
              f"   [{time.time()-t0:.0f} s]", flush=True)
        tabs[xc] = (n1.mean(), o1, n2.mean(), o2, e2)
    print("\n  hybrid against kinetic: scatterings and core visits to first reach |x| >= X")
    print("    xc    X   kinetic nsc (visits)  | table N+exits | chain N+exits | chain N, exp(<d>)")
    for xc in args.xc:
        n1, o1, n2, o2, e2 = tabs[xc]
        for dX in args.dX:
            X = xc + dX
            start = rng.choice(e2, args.nwalk)*np.where(rng.uniform(size=args.nwalk) < .5, -1., 1.)
            k = kinetic_reach(rng, a, xc, X, args.nwalk, start)
            row = f"  {xc:4.2f} {X:4.1f} {k[0]:10.0f} ({k[2]:7.2f})"
            for nmean, draw in ((n1, lambda m: rng.choice(o1, m)), (n2, lambda m: rng.choice(o2, m)),
                                (n2, lambda m: rng.exponential(o2.mean(), m))):
                h = hybrid_reach(rng, a, xc, X, nmean, draw, args.nwalk)
                row += f" | {h[0]/k[0]:5.3f} ({h[2]/k[2]:5.3f})"
            print(row, flush=True)


if __name__ == '__main__':
    main()
