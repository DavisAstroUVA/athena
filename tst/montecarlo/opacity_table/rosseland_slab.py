#! /usr/bin/env python
"""
Diffusion through a slab with a temperature gradient: the Rosseland test of the two-band
split.  By default on a synthetic picket-fence table (make_picket_table.py: every group
of every cell a fraction w at kappa_h and 1-w at kappa_l), on which the slab is equally
thick in every group and the expected flux exact; with --table on any table with its
free-electron file, as a demonstration, where groups the slab is thin in free-stream and
the comparison is loose.

mc_isoth_opac fills a slab of constant density whose temperature rises linearly across
x3 (tgrad), periodic in x1 and x2, absorbing at both x3 faces, thick in every band of
every group that carries the gradient.  In the middle of the slab the radiative flux is
the diffusion flux,

    F = -(4 pi / 3 rho) dT/dz sum_g (dB_g/dT) / kappa_R,g,

with kappa_R,g the group's Rosseland mean as the transport sees it, the Rosseland
absorption plus the cell's Thomson scattering.  A single-band run (two_band off) absorbs
with the Planck mean in every group and carries the flux of kappa_P + s instead, smaller
by the Planck to Rosseland ratio (a factor 3.2 on the default picket table, one to four
orders of magnitude in line-dominated gas); the two-band run must recover the Rosseland
flux, and the thermalized high band (two_band_thick) must give the same flux with far
fewer samples dying where they were born.  The measured flux is the cell-averaged Frmc3
over the middle half of the slab against the expectation at each cell's temperature,
within the noise plus 5%: the two-band runs against the Rosseland flux, and on the picket
table the single-band run against the Planck-only one.  PASS/FAIL.

  python rosseland_slab.py --path /path/to/athena --workdir DIR [--table TABLE.txt]
         [--temp 1e5 --dens 1e-9 --tgrad 0.5] [--tau 10] [--nphot 1000000] [--w0 0.01]
         [--kh 300 --kl 1.0]

The flux is a small anisotropy of a nearly isotropic field, of order the fractional
change of a T^4 across a mean free path, so the gradient is steep, the slab ten depths
across and the sample a million; the ratio is still known only to a few percent.

The binary must be built with --prob=mc_isoth_opac -mc -hdf5.
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VIS = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'vis', 'python'))
sys.path.insert(0, HERE)
sys.path.insert(0, VIS)
import opacity_test_util as u  # noqa: E402
import athena_read  # noqa: E402
import table_tools as tt  # noqa: E402

MODES = {'single': {'two_band': 'false'},
         'twoband': {'two_band': 'true'},
         'thick': {'two_band': 'true', 'two_band_thick': '10.'}}


def slab_athinput(path, table, nfree, nphot, iseed, temp, rho, tgrad, length, nz, extra):
    keys = {'opacity_file': table, 'nfree_file': nfree, 'table_ne': 'true',
            'abs_method': 'prob'}
    keys.update(extra)
    dx = length/nz
    text = """<comment>
problem   = slab with a temperature gradient, two-band diffusion
configure = --prob=mc_isoth_opac -mc -mpi -hdf5

<job>
problem_id = slab

<output1>
file_type = hdf5
variable  = mclab
id        = lab
dt = 1.e-11

<mesh>
nx1        = 4
x1min      = {xlo:e}
x1max      = {xhi:e}
ix1_mc_bc  = periodic
ox1_mc_bc  = periodic
nx2        = 4
x2min      = {xlo:e}
x2max      = {xhi:e}
ix2_mc_bc  = periodic
ox2_mc_bc  = periodic
nx3        = {nz}
x3min      = {zlo:e}
x3max      = {zhi:e}
ix3_mc_bc  = absorb
ox3_mc_bc  = absorb

<meshblock>
nx1 = 4
nx2 = 4
nx3 = {nz}

<hydro>
gamma = 1.666666666666667

""".format(xlo=-2.*dx, xhi=2.*dx, zlo=-0.5*length, zhi=0.5*length, nz=nz) + u.mc_block(
        nphot, iseed, 'thomson', 'table', 'table', keys) + """
<problem>
constdens = true
dens      = {rho:e}
temp      = {temp:e}
tgrad     = {tgrad:e}
""".format(rho=rho, temp=temp, tgrad=tgrad)
    with open(path, 'w') as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--path', default=os.path.join(HERE, '..', '..', '..'))
    ap.add_argument('--workdir', default='.')
    ap.add_argument('--table', default=None, help='default: a picket-fence table is written')
    ap.add_argument('--nfree', default=None)
    ap.add_argument('--kh', type=float, default=300., help='picket table: high band [cm^2/g]')
    ap.add_argument('--kl', type=float, default=1.0, help='picket table: low band [cm^2/g]')
    ap.add_argument('--temp', type=float, default=1.e5)
    ap.add_argument('--dens', type=float, default=1.e-9)
    ap.add_argument('--tgrad', type=float, default=0.5, help='fractional rise of T across')
    ap.add_argument('--tau', type=float, default=10., help='depth across the slab in the '
                    'thinnest band carrying the gradient')
    ap.add_argument('--nz', type=int, default=32)
    ap.add_argument('--modes', nargs='*', default=['single', 'twoband', 'thick'])
    ap.add_argument('--nphot', type=int, default=1000000)
    ap.add_argument('--w0', type=float, default=0.01)
    ap.add_argument('--iseed', type=int, default=146137)
    ap.add_argument('--mcranks', type=int, default=1)
    args = ap.parse_args()

    binary = args.path if os.path.isfile(args.path) else os.path.join(args.path, 'bin', 'athena')
    workdir = os.path.abspath(args.workdir)
    os.makedirs(workdir, exist_ok=True)
    gated_modes = ('twoband', 'thick')
    if args.table is None:
        sys.path.insert(0, os.path.join(VIS, 'montecarlo', 'problem_specific', 'opacity_table'))
        import make_ff_table as ff
        import make_picket_table as pk
        table = os.path.join(workdir, 'picket.txt')
        pk.write_picket(table, ff.group_edges(32, 1.e-3, 7.4989), np.geomspace(1.e3, 1.e9, 5),
                        np.geomspace(1.e-18, 1.e-6, 5), args.kh, args.kl, args.w0)
        gated_modes = ('single', 'twoband', 'thick')
    else:
        table = os.path.abspath(args.table)
    nfree = os.path.abspath(args.nfree) if args.nfree else table[:-4] + '_nfree.txt'
    tab = tt.Table(table, nfree)
    edges, ng = tab.edges, tab.ng
    temp, rho = args.temp, args.dens

    # the slab's thickness from the split at the mean temperature: the thinnest band
    # (absorption plus scattering) among the groups holding 99% of dB/dT
    s = tt.SIGMA_T*tab.ne(temp, rho)/rho
    w, kh, kl = tab.split(temp, rho, s, args.w0)
    dbdt = np.array([tt.dplanck_dt_band(edges[g], edges[g+1], temp) for g in range(ng)])
    order = np.argsort(dbdt)[::-1]
    keep = order[np.cumsum(dbdt[order]) <= 0.99*dbdt.sum()]
    kmin = min((kl[keep] + s).min(), (tab.planck(temp, rho)[keep] + s).min())
    length = args.tau/(rho*kmin)
    print('slab: T = {:.3g} K (+-{:.0f}%), rho = {:.3g}, {} of {} groups two-band, '
          'thinnest band kappa = {:.3g} cm^2/g, L = {:.3g} cm'.format(
              temp, 50.*args.tgrad, rho, int((w < 1.).sum()), ng, kmin, length))

    def expected_flux(tz):
        """diffusion flux at temperature tz, erg/cm^2/s, positive toward lower z"""
        kp = tab.planck(tz, rho)
        sz = tt.SIGMA_T*tab.ne(tz, rho)/rho
        wz, khz, klz = tab.split(tz, rho, sz, args.w0)
        kra = tab.rosseland_abs(tz, rho)
        if kra is None:
            kra = tab.rosseland(tz, rho) - sz
        kr = np.maximum(kra, 0.003*sz) + sz          # as the split reproduces it
        kr_single = kp + sz                           # the Planck-only transport
        db = np.array([tt.dplanck_dt_band(edges[g], edges[g+1], tz) for g in range(ng)])
        dtdz = temp*args.tgrad/length
        f_ross = -4.*np.pi/(3.*rho)*dtdz*(db/kr).sum()
        f_single = -4.*np.pi/(3.*rho)*dtdz*(db/kr_single).sum()
        return f_ross, f_single

    allok = True
    for mode in args.modes:
        extra = dict(MODES[mode], two_band_w='{:g}'.format(args.w0))
        d = os.path.join(workdir, mode)
        u.link_inputs(d, table, nfree)
        ath = os.path.join(d, 'athinput.slab')
        slab_athinput(ath, os.path.basename(table), os.path.basename(nfree), args.nphot,
                      args.iseed, temp, rho, args.tgrad, length, args.nz, extra)
        u.run(binary, ath, d, args.mcranks)
        lab = athena_read.athdf(glob.glob(os.path.join(d, '*.lab.*athdf'))[0],
                                quantities=['Frmc3', 'tgas'])
        fz = lab['Frmc3']
        tg = lab['tgas']
        # the middle half of the slab, averaged over x1 and x2
        nz = fz.shape[0]
        sl = slice(nz//4, 3*nz//4)
        fmid = fz[sl].mean(axis=(1, 2))
        tmid = tg[sl].mean(axis=(1, 2))
        exp_r = np.array([expected_flux(t)[0] for t in tmid])
        exp_s = np.array([expected_flux(t)[1] for t in tmid])
        ratio_r = fmid/exp_r
        ratio_s = fmid/exp_s
        # the scatter between the middle cells as the noise of their mean
        mean_r = ratio_r.mean()
        err_r = ratio_r.std()/np.sqrt(len(ratio_r))
        first = u.grep(os.path.join(d, 'run.log'), 'died in their first step')
        first = first[-1].split('(')[1].split(')')[0] if first else '-'
        # The two-band runs must give the Rosseland flux; on the picket table the
        # single-band run must give the Planck-only flux, the deficit the split removes.
        if mode == 'single':
            target, err = ratio_s.mean(), ratio_s.std()/np.sqrt(len(ratio_s))
        else:
            target, err = mean_r, err_r
        gated = mode in gated_modes
        ok = (abs(target - 1.) < 0.05 + 3.*err) if gated else True
        allok = allok and ok
        print('{:>8s}: F/F_Rosseland = {:.4f} +- {:.4f}, F/F_Planck = {:.4f}; first-step '
              'deaths {}; {}'.format(mode, mean_r, err_r, ratio_s.mean(), first,
                                     'ok' if ok else ('FAIL' if gated else 'not gated')))
    print('PASS' if allok else 'FAIL')
    sys.exit(0 if allok else 1)


if __name__ == '__main__':
    main()
