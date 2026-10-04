
"""
Spot-check a kompaneets_greens quantile table against direct solutions at
random off-grid (lam, y, xi): survival and the 1%, 50%, 99% quantiles of
ln xf.  Points with survival below smin are skipped (dead photons).

    python check_table.py kgreens_table_ff.npz --npts 60
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table", nargs="?", default="kgreens_table_ff.npz")
    ap.add_argument("--npts", type=int, default=60)
    ap.add_argument("--smin", type=float, default=1.e-6)
    ap.add_argument("--seed", type=int, default=2)
    a = ap.parse_args()
    smp = kg.Sampler(a.table)
    rng = np.random.default_rng(a.seed)
    ff = kg.freefree_shape
    lamlim = (smp.lam[1], smp.lam[-1]) if len(smp.lam) > 1 else None
    rows = []
    lev = np.array([0.01, 0.5, 0.99])
    while len(rows) < a.npts:
        y = np.exp(rng.uniform(np.log(smp.y[0]), np.log(smp.y[-1])))
        xi = np.exp(rng.uniform(smp.lnxi[0], smp.lnxi[-1]))
        lam = 0. if lamlim is None else \
            np.exp(rng.uniform(np.log(lamlim[0]), np.log(lamlim[1])))
        snk = ff if lam > 0. else None
        res, lns = kg.greens(xi, [y], sink=snk, lam=lam, lnsurv=True)
        s, p = res[0]
        c = np.concatenate([[0.], np.cumsum(0.5*(p[1:]+p[:-1])*np.diff(s))])
        surv = np.exp(lns[0])
        if surv < a.smin:
            continue
        qd = np.interp(lev, c/c[-1], s)
        qt = np.log(smp.quantile(xi, y, lev, lam))
        st = smp.survival(xi, y, lam)
        # quantile errors relative to the width of the distribution
        wid = max(qd[2]-qd[0], 1.e-30)
        rows.append((lam, y, xi, surv, st/surv-1., *((qt-qd)/wid)))
    r = np.array(rows)
    print("{:>9s} {:>9s} {:>9s} {:>10s} {:>9s} {:>8s} {:>8s} {:>8s}".format(
        "lam", "y", "xi", "S", "dS/S", "dq01/w", "dq50/w", "dq99/w"))
    for row in r[np.argsort(-np.max(np.abs(r[:, 4:]), axis=1))][:15]:
        print("{:9.2e} {:9.2e} {:9.2e} {:10.3e} {:+9.1e} {:+8.1e} {:+8.1e} "
              "{:+8.1e}".format(*row))
    print("... (15 worst of {:d})".format(len(r)))
    for k, name in zip(range(4, 8), ["dS/S", "dq01/w", "dq50/w", "dq99/w"]):
        v = np.abs(r[:, k])
        print("{:>7s}: median {:.1e}  90% {:.1e}  max {:.1e}".format(
            name, np.median(v), np.quantile(v, 0.9), v.max()))


if __name__ == "__main__":
    main()
