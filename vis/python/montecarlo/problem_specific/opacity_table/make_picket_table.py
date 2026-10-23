#! /usr/bin/env python
"""
A synthetic picket-fence opacity table for testing the two-band split: every group of
every (T, rho) cell holds the same two bands, a fraction w at kappa_h and 1-w at kappa_l
(cm^2/g, absorption), with Thomson scattering on the fully ionized hydrogen-helium gas
of the Monte Carlo module.  The file carries what the module reads: the Planck group
mean w kappa_h + (1-w) kappa_l, the Rosseland group mean with the scattering,
[w/(kappa_h + s) + (1-w)/(kappa_l + s)]^-1, the Rosseland absorption block (the same
less s), and the free electrons per atom beside it.  Reading it with two_band = true,
table_ne = true and two_band_w = w, OpacityTable::Split recovers kappa_h and kappa_l
exactly, so a slab of it has a known Rosseland flux in every group and the single-band
transport a known deficit, the Planck to Rosseland ratio.

  python make_picket_table.py OUT.txt [--kh 300 --kl 0.3 --w 0.01]
                                      [--ngroups 32 --emin 1e-3 --emax 7.4989]
                                      [--nt 5 --tmin 1e3 --tmax 1e9 --nrho 5
                                      --rhomin 1e-18 --rhomax 1e-6 --heabund 0.09]
"""
import argparse

import numpy as np

import make_ff_table as ff


def write_picket(out, edges, tgrid, rgrid, kh, kl, w, heabund=0.09, digits=8):
    ng, nt, nr = len(edges) - 1, len(tgrid), len(rgrid)
    kp = w*kh + (1. - w)*kl
    plan = np.full((ng, nt, nr), kp)
    ross = np.zeros((ng, nt, nr))
    rossabs = np.zeros((ng, nt, nr))
    for i, rho in enumerate(rgrid):
        s = ff.kappa_es(rho, heabund)
        kr = 1./(w/(kh + s) + (1. - w)/(kl + s))
        ross[:, :, i] = kr
        rossabs[:, :, i] = kr - s
    fmt = '{:.%dE}' % digits
    with open(out, 'w') as f:
        f.write('{}\n{}\n{}\n'.format(ng, nt, nr))
        f.writelines(fmt.format(e/ff.KEV) + '\n' for e in edges[:-1])
        f.writelines(fmt.format(t/ff.KEV_K) + '\n' for t in tgrid)
        f.writelines(fmt.format(r) + '\n' for r in rgrid)
        blocks = [ross[0], plan[0]] + [ross[g] for g in range(ng)] \
            + [plan[g] for g in range(ng)] + [rossabs[g] for g in range(ng)]
        for block in blocks:
            for j in range(nt):
                f.write(' '.join(fmt.format(v) for v in block[j]) + '\n')
    nfree = (1. + 2.*heabund)/(1. + heabund)
    abar = (1. + 4.*heabund)/(1. + heabund)*ff.MP/ff.MU
    stem = out[:-4] if out.endswith('.txt') else out
    with open(stem + '_nfree.txt', 'w') as f:
        f.write('# N_free, free electrons per atom, fully ionized H + He with {:g} He per H '
                '(make_picket_table.py)\n'.format(heabund))
        f.write('# Abar = {:.5f}  Z_full = {:.5f}  n_e = rho * N_free / (Abar * m_u)\n'.format(
            abar, nfree))
        f.write('# layout: nT, nrho, T [keV] (nT lines), rho [g/cm^3] (nrho lines), '
                'then nT rows of nrho values\n')
        f.write('{}\n{}\n'.format(nt, nr))
        f.writelines(fmt.format(t/ff.KEV_K) + '\n' for t in tgrid)
        f.writelines(fmt.format(r) + '\n' for r in rgrid)
        for j in range(nt):
            f.write(' '.join(fmt.format(nfree) for _ in range(nr)) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out')
    ap.add_argument('--kh', type=float, default=300., help='high-band absorption [cm^2/g]')
    ap.add_argument('--kl', type=float, default=0.3, help='low-band absorption [cm^2/g]')
    ap.add_argument('--w', type=float, default=0.01, help='high-band frequency fraction')
    ap.add_argument('--ngroups', type=int, default=32)
    ap.add_argument('--emin', type=float, default=1.e-3, help='lowest lower bound [keV]')
    ap.add_argument('--emax', type=float, default=7.4989, help='highest lower bound [keV]')
    ap.add_argument('--nt', type=int, default=5)
    ap.add_argument('--tmin', type=float, default=1.e3)
    ap.add_argument('--tmax', type=float, default=1.e9)
    ap.add_argument('--nrho', type=int, default=5)
    ap.add_argument('--rhomin', type=float, default=1.e-18)
    ap.add_argument('--rhomax', type=float, default=1.e-6)
    ap.add_argument('--heabund', type=float, default=0.09)
    args = ap.parse_args()
    edges = ff.group_edges(args.ngroups, args.emin, args.emax)
    write_picket(args.out, edges, np.geomspace(args.tmin, args.tmax, args.nt),
                 np.geomspace(args.rhomin, args.rhomax, args.nrho), args.kh, args.kl, args.w,
                 args.heabund)
    print('{}: {} groups, kappa_h = {:g}, kappa_l = {:g}, w = {:g}, Planck mean {:g}'.format(
        args.out, args.ngroups, args.kh, args.kl, args.w, args.w*args.kh + (1-args.w)*args.kl))


if __name__ == '__main__':
    main()
