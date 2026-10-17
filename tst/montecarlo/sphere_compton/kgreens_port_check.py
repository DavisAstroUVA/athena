#! /usr/bin/env python

"""
Check the C++ port of the Kompaneets table sampler (src/monte_carlo/kgreens.cpp) against
the Python reference (kompaneets_greens.Sampler) on the cases of
doc/monte_carlo/kompaneets/KGREENS_TABLE_FORMAT.md Section 7 plus a sweep of random
points, including the neighbourhood of dead cells and 0 < lam < lam[1].

    python kgreens_port_check.py kgreens_table_ff.npz [--bin kgreens_table_ff.bin]
                                 [--npts 2000] [--workdir DIR]

Exports the binary if --bin is not given, builds the harness with g++, and compares.
PASS when every quantile agrees to 1e-6 relative in ln(xf/xi)/sqrt(2y) terms (float32
table entries) and every survival to 1e-10 relative.
"""

import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..', '..'))
VIS = os.path.join(ROOT, 'vis', 'python', 'montecarlo')
sys.path.insert(0, VIS)
sys.path.insert(0, os.path.join(VIS, 'problem_specific', 'sphere_compton'))

import kompaneets_greens as kg  # noqa: E402
import kgreens_export  # noqa: E402


def cases(smp, npts, seed=3):
    rows = [(0.03, 0.2, 2e-3, 0.37), (1.0, 3.0, 0.0, 0.9), (5.0, 1e-7, 1.0, 0.5),
            (0.3, 40.0, 0.1, 0.01), (0.01, 2.0, 100., 0.5), (0.01, 0.5, 300., 0.2),
            (2.0, 0.3, 0.5e-10, 0.6), (60., 1.0, 0., 0.999), (1e-3, 1e-5, 1e-6, 1e-4)]
    rng = np.random.default_rng(seed)
    lam_choices = np.concatenate([[0.], np.logspace(-11, 3.3, 40)])
    for _ in range(npts):
        xi = 10**rng.uniform(-3.1, 1.85)
        y = 10**rng.uniform(-7, 1.5)
        lam = lam_choices[rng.integers(len(lam_choices))]*10**rng.uniform(-0.3, 0.3)
        rows.append((xi, y, lam, rng.uniform(1e-6, 1 - 1e-6)))
    return rows


def main(args):
    npz = args.npz
    workdir = args.workdir or os.getcwd()
    os.makedirs(workdir, exist_ok=True)
    binfile = args.bin
    if binfile is None:
        binfile = os.path.join(workdir, os.path.basename(npz).replace('.npz', '.bin'))
        kgreens_export.export(npz, binfile)
    exe = os.path.join(workdir, 'kgreens_port_check')
    cmd = ['g++', '-O2', '-std=c++11', '-I' + ROOT,
           os.path.join(HERE, 'kgreens_port_check.cpp'),
           os.path.join(ROOT, 'src', 'monte_carlo', 'kgreens.cpp'), '-o', exe]
    print(' '.join(cmd))
    subprocess.check_call(cmd)
    smp = kg.Sampler(npz)
    rows = cases(smp, args.npts)
    inp = ''.join(f"{xi:.17g} {y:.17g} {lam:.17g} {r:.17g}\n" for xi, y, lam, r in rows)
    out = subprocess.run([exe, binfile], input=inp, capture_output=True, text=True,
                         check=True).stdout.split('\n')
    worst_q, worst_s = 0., 0.
    bad = []
    for (xi, y, lam, r), line in zip(rows, out):
        xf_c, s_c = map(float, line.split())
        xf_p = float(smp.quantile(xi, y, np.array([r]), lam)[0])
        s_p = float(smp.survival(xi, y, lam))
        # compare in the table's own variable, u = ln(xf/xi)/sqrt(2 y)
        du = abs(np.log(xf_c/xi) - np.log(xf_p/xi))/np.sqrt(2.*y)
        ds = abs(s_c - s_p)/max(s_p, 1e-300)
        worst_q = max(worst_q, du)
        worst_s = max(worst_s, ds)
        if du > 1e-6 or ds > 1e-10:
            bad.append((xi, y, lam, r, xf_c, xf_p, s_c, s_p))
    print(f"{len(rows)} cases: max |du| {worst_q:.3e}, max dS/S {worst_s:.3e}")
    for b in bad[:10]:
        print("  mismatch xi=%g y=%g lam=%g r=%g: xf %.10g vs %.10g, S %.12g vs %.12g" % b)
    print("RESULT:", "PASS" if not bad else "FAIL")
    sys.exit(0 if not bad else 1)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('npz')
    ap.add_argument('--bin', default=None)
    ap.add_argument('--npts', type=int, default=2000)
    ap.add_argument('--workdir', default=None)
    return ap.parse_args(argv)


if __name__ == '__main__':
    main(parse_args())
