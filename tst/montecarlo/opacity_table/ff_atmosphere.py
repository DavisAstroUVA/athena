#! /usr/bin/env python
"""
The emergent spectrum of a scattering atmosphere with table absorption against the same
atmosphere with free-free absorption, on a table that is free-free (make_ff_table.py).

mc_isoth_opac's isothermal atmosphere (density falling exponentially between the Thomson
depths taumin and taumax, Thomson scattering, escape through the top) is run with
emission = table, absorption = table and with emission = freefree, absorption = freefree
at each temperature asked for.  In a scattering atmosphere the emergent flux in a group
depends on the absorption through the thermalization depth, so a piecewise-constant
opacity does not give the free-free spectrum within a group: where the opacity changes
across a group so does sqrt(epsilon), epsilon = kappa_a/(kappa_a + kappa_es), and the
two-stream estimate F ~ B 2 sqrt(eps)/(1 + sqrt(eps)) says by how much.  The test
therefore compares the group luminosities of the two runs against that estimate's ratio,
int_g B 2 sqrt(eps_g)/(1+sqrt(eps_g)) / int_g B_nu 2 sqrt(eps_nu)/(1+sqrt(eps_nu)), at
the density of the thermalization layer, within the sampling noise plus 3%, and the
total luminosities within the same.  The noise comes from the energy weights of the
escapers of both runs (equal-weight sampling of the cells, so that the dense base does
not ride on a few heavy samples).  PASS/FAIL.

  python ff_atmosphere.py --path /path/to/athena --workdir DIR [--temps 1e5 1e6 1e7]
         [--dens 1e-9] [--taumax 100] [--nphot 50000] [--table TABLE.txt]

The bottom Thomson depth is 100 by default: the free-free run's cost grows as its
square, since the photons born at the base scatter that many times squared before they
are absorbed or escape, and the groups the comparison is about thermalize well above it.

The binary must be built with --prob=mc_isoth_opac -mc -hdf5.  Without --table the table
is written into the workdir.
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VIS = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'vis', 'python'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(VIS, 'montecarlo'))
sys.path.insert(0, os.path.join(VIS, 'montecarlo', 'problem_specific', 'opacity_table'))
import opacity_test_util as u  # noqa: E402
import band_luminosity  # noqa: E402
import make_ff_table as ff  # noqa: E402
from ff_table import read_table_grid  # noqa: E402

H, KEV = ff.H, ff.KEV


def two_stream_ratio(elo, ehi, temp, rho, heabund, n=201):
    """
    ratio of the two-stream emergent flux in a group with the group's Planck mean opacity
    to that with the frequency-dependent opacity
    """
    nu, w = ff._lognu(elo, ehi, n)
    b = ff.planck_nu(nu, temp)*nu*w
    kes = ff.kappa_es(rho, heabund)
    knu = ff.kappa_ff(nu, temp, rho, heabund)
    kg = ff.planck_group(elo, ehi, temp, rho, heabund)
    eps_nu = knu/(knu + kes)
    eps_g = kg/(kg + kes)
    f_nu = (b*2.*np.sqrt(eps_nu)/(1. + np.sqrt(eps_nu))).sum()
    f_g = (b*2.*np.sqrt(eps_g)/(1. + np.sqrt(eps_g))).sum()
    if f_nu <= 0.:
        return 1.
    return f_g/f_nu


def run_luminosities(binary, workdir, table, scheme, temp, dens, taumin, taumax, nphot,
                     iseed, edges, mcranks):
    emission = absorption = 'table' if scheme == 'table' else 'freefree'
    # cells drawn in proportion to their emission, so the base of the atmosphere does
    # not put its whole emission on a few heavy samples
    extra = {'weights': 'equal'}
    d = os.path.join(workdir, 'T{:.3g}_{}'.format(temp, scheme))
    u.link_inputs(d, table)
    ath = os.path.join(d, 'athinput.mciso')
    u.write_isoth_opac_athinput(ath, os.path.basename(table), nphot, iseed, 'thomson',
                                emission, absorption, extra, temp=temp, dens=dens,
                                taumin=taumin, taumax=taumax)
    if scheme != 'table':
        # the free-free sampler's window is the table's
        with open(ath) as f:
            text = f.read()
        text = text.replace('tnorm    = true', 'tnorm    = false')
        text = text.replace('emin     = 1.0e-02', 'emin     = {:e}'.format(edges[0]/1.602176634e-12))
        # the free-free window stops at 60 kT, see ff_table.ff_emax_ev
        emax_ev = min(edges[-1]/1.602176634e-12, 60.*ff.KB*temp/1.602176634e-12)
        text = text.replace('emax     = 3.0e+01', 'emax     = {:e}'.format(emax_ev))
        with open(ath, 'w') as f:
            f.write(text)
    u.run(binary, ath, d, mcranks)
    files = sorted(glob.glob(os.path.join(d, '*.list')))
    total, lum, _ = band_luminosity.band_luminosities(files, edges/KEV)
    # photon counts and the sum of the squared energy weights per group, for the noise
    nph = np.zeros(len(edges)-1)
    sq = np.zeros(len(edges)-1)
    sq_tot = 0.
    for fn in files:
        gen = band_luminosity.athena_mc.read_list_generator(fn)
        header = next(gen)['header']
        dt = header['dt']
        for c in gen:
            if c['chunk'] is None:
                continue
            p = band_luminosity.athena_mc.Photons(dict(header, list=c['chunk'],
                                                       length=c['length']))
            we = p.weight*p.energy/dt
            idx = np.searchsorted(edges, p.energy) - 1
            ok = (idx >= 0) & (idx < len(nph))
            np.add.at(nph, idx[ok], 1)
            np.add.at(sq, idx[ok], we[ok]**2)
            sq_tot += (we**2).sum()
            if c['done']:
                break
    with np.errstate(divide='ignore', invalid='ignore'):
        sig = np.where(lum > 0., np.sqrt(sq)/lum, 1.)
    return total, lum, nph, sig, np.sqrt(sq_tot)/total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--path', default=os.path.join(HERE, '..', '..', '..'))
    ap.add_argument('--workdir', default='.')
    ap.add_argument('--table', default=None)
    ap.add_argument('--temps', type=float, nargs='*', default=[1.e5, 1.e6, 1.e7])
    ap.add_argument('--dens', type=float, default=1.e-9, help='density scale')
    ap.add_argument('--taumin', type=float, default=1.e-3)
    ap.add_argument('--taumax', type=float, default=100.)
    ap.add_argument('--nphot', type=int, default=50000)
    ap.add_argument('--iseed', type=int, default=146137)
    ap.add_argument('--heabund', type=float, default=0.09)
    ap.add_argument('--mcranks', type=int, default=1)
    args = ap.parse_args()

    binary = args.path if os.path.isfile(args.path) else os.path.join(args.path, 'bin', 'athena')
    workdir = os.path.abspath(args.workdir)
    os.makedirs(workdir, exist_ok=True)
    table = args.table and os.path.abspath(args.table)
    if table is None:
        table = os.path.join(workdir, 'ff_uniform.txt')
        if not os.path.exists(table):
            ff.write_tables(table, ff.group_edges(32, 1.e-3, 7.4989),
                            np.geomspace(5.8e3, 1.e9, 60), np.geomspace(1.e-18, 1.e-6, 31),
                            args.heabund)
    edges, tg, rg = read_table_grid(table)
    ng = len(edges) - 1

    allok = True
    for temp in args.temps:
        res = {}
        for scheme in ('table', 'freefree'):
            res[scheme] = run_luminosities(binary, workdir, table, scheme, temp, args.dens,
                                           args.taumin, args.taumax, args.nphot,
                                           args.iseed, edges, args.mcranks)
        lt, lg_t, n_t, s_t, st_t = res['table']
        lf, lg_f, n_f, s_f, st_f = res['freefree']
        # The thermalization layer: the density where the effective depth
        # sqrt(3 tau_a (tau_a + tau_s)) reaches one is where the emergent spectrum forms.
        # Use the density at Thomson depth of a few, which is where it sits for the groups
        # near the Planck peak; the ratio is insensitive to this choice.
        rho_form = args.dens
        ratio_est = np.array([two_stream_ratio(edges[g], edges[g+1], temp, rho_form,
                                               args.heabund) for g in range(ng)])
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_mc = lg_t/lg_f
        # the sampling error of the ratio from the weights of both runs
        sig = np.sqrt(s_t**2 + s_f**2)
        sig_tot = np.sqrt(st_t**2 + st_f**2)
        good = (n_t >= 20) & (n_f >= 20)
        dev = np.abs(ratio_mc - ratio_est)
        within = int(((dev < 3.*sig + 0.03) | ~good).sum())
        tot_ratio = lt/lf
        ok = within >= ng - 1 and abs(tot_ratio - 1.) < 0.03 + 3.*sig_tot
        allok = allok and ok
        print('T = {:.3g} K: L_table/L_ff = {:.4f} +- {:.4f}; groups within estimate: {}/{} '
              '({} with counts); {}'.format(temp, tot_ratio, sig_tot, within, ng,
                                            int(good.sum()), 'ok' if ok else 'FAIL'))
        for g in range(ng):
            if good[g]:
                print('   g{:2d} {:8.4f} keV  MC {:.3f}  two-stream {:.3f}  sigma {:.3f}'.format(
                    g, edges[g]/KEV, ratio_mc[g], ratio_est[g], sig[g]))
    print('PASS' if allok else 'FAIL')
    sys.exit(0 if allok else 1)


if __name__ == '__main__':
    main()
