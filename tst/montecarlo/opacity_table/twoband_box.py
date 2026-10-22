#! /usr/bin/env python
"""
The two-band split in uniform boxes, on any table with its free-electron file.

At each (T, rho), mc_isoth_opac runs with two_band off, on, and on with the high band
thermalized (two_band_thick), Thomson scattering and table_ne:

  thin box    escape boundaries, optically thin in every band: the escaping photon rate
              is the emission rate, which the split leaves at 4 pi rho sum_g kappa_P,g N_g
              (thermalized: the dropped high-band emission less the scattering leakage
              the low band receives instead), to 1e-3; and the escaping energy per group
              within the sampling noise of 4 pi rho kappa_P,g B_g.
  closed box  periodic: the field relaxes to the Planck function in every band
              (Kirchhoff holds band by band), so the energy density is
              (4 pi/c) sum_g B_g and the J-weighted absorption kapJ/Er the window Planck
              mean, within 10% at a million photons (the Wien-tail variance of analog
              Planck sampling, see ff_table.py).  With the high band thermalized the
              transported field holds the low band's share, 1-w, of those groups and its
              J-weighted absorption is the low band's, which is what is compared; the
              plain two-band run's closed box is reported only, since its field rests on
              the one sample in a million that is not absorbed where it was born.

The run report's "died in their first step" count shows the cost the thermalized high
band removes.  The split itself is checked at every build against both of its moments
(BuildOpacityTables).  PASS/FAIL.

  python twoband_box.py --path /path/to/athena --workdir DIR --table TABLE.txt
         [--points T,rho ...] [--nphot 20000] [--nphot-closed 1000000] [--w0 0.01]

The binary must be built with --prob=mc_isoth_opac -mc -hdf5.  The default points are
line-dominated cells of the solar TOPS table.
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
sys.path.insert(0, os.path.join(VIS, 'montecarlo'))
import opacity_test_util as u  # noqa: E402
import athena_read  # noqa: E402
import band_luminosity  # noqa: E402
import table_tools as tt  # noqa: E402

MODES = {'single': {'two_band': 'false'},
         'twoband': {'two_band': 'true'},
         'thick': {'two_band': 'true', 'two_band_thick': '10.'}}


def box_athinput(path, table, nfree, nphot, iseed, temp, rho, length, closed, extra):
    keys = {'opacity_file': table, 'nfree_file': nfree, 'table_ne': 'true',
            'abs_method': 'prob'}
    keys.update(extra)
    bc = 'periodic' if closed else 'escape'
    text = """<comment>
problem   = uniform box, two-band split
configure = --prob=mc_isoth_opac -mc -mpi -hdf5

<job>
problem_id = tbbox

<output1>
file_type = phlist

<output2>
file_type = hdf5
variable  = mclab
id        = lab
dt = 1.e-11

<output3>
file_type = hdf5
variable  = uom
id        = uom
dt = 1.e-11

<mesh>
nx1        = 8
x1min      = {lo:e}
x1max      = {hi:e}
ix1_mc_bc  = {bc}
ox1_mc_bc  = {bc}
nx2        = 8
x2min      = {lo:e}
x2max      = {hi:e}
ix2_mc_bc  = {bc}
ox2_mc_bc  = {bc}
nx3        = 8
x3min      = {lo:e}
x3max      = {hi:e}
ix3_mc_bc  = {bc}
ox3_mc_bc  = {bc}

<meshblock>
nx1 = 8
nx2 = 8
nx3 = 8

<hydro>
gamma = 1.666666666666667

""".format(lo=-0.5*length, hi=0.5*length, bc=bc) + u.mc_block(
        nphot, iseed, 'thomson', 'table', 'table', keys) + """
<problem>
constdens = true
dens      = {rho:e}
temp      = {temp:e}
""".format(rho=rho, temp=temp)
    with open(path, 'w') as f:
        f.write(text)


def escaping(workdir, edges):
    """photon rate and its relative error, energy per group, photons per group"""
    files = sorted(glob.glob(os.path.join(workdir, '*.list')))
    rate = wsq = 0.
    nph = np.zeros(len(edges)-1)
    for f in files:
        gen = band_luminosity.athena_mc.read_list_generator(f)
        header = next(gen)['header']
        dt = header['dt']
        for c in gen:
            if c['chunk'] is None:
                continue
            p = band_luminosity.athena_mc.Photons(dict(header, list=c['chunk'],
                                                       length=c['length']))
            rate += p.weight.sum()/dt
            wsq += (p.weight**2).sum()/dt**2
            idx = np.searchsorted(edges, p.energy) - 1
            ok = (idx >= 0) & (idx < len(nph))
            np.add.at(nph, idx[ok], 1)
            if c['done']:
                break
    total, lum, _ = band_luminosity.band_luminosities(files, edges/tt.KEV)
    return rate, (np.sqrt(wsq)/rate if rate > 0 else 1.), total, lum, nph


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--path', default=os.path.join(HERE, '..', '..', '..'))
    ap.add_argument('--workdir', default='.')
    ap.add_argument('--table', required=True)
    ap.add_argument('--nfree', default=None)
    ap.add_argument('--points', nargs='*', default=['2e4,1e-15', '1e4,1e-10', '3e4,1e-12',
                                                    '1e5,1e-11'])
    ap.add_argument('--modes', nargs='*', default=['single', 'twoband', 'thick'])
    ap.add_argument('--nphot', type=int, default=20000)
    ap.add_argument('--nphot-closed', type=int, default=1000000)
    ap.add_argument('--w0', type=float, default=0.01)
    ap.add_argument('--iseed', type=int, default=146137)
    ap.add_argument('--mcranks', type=int, default=1)
    args = ap.parse_args()

    binary = args.path if os.path.isfile(args.path) else os.path.join(args.path, 'bin', 'athena')
    table = os.path.abspath(args.table)
    nfree = os.path.abspath(args.nfree) if args.nfree else table[:-4] + '_nfree.txt'
    workdir = os.path.abspath(args.workdir)
    tab = tt.Table(table, nfree)
    edges = tab.edges
    ng = tab.ng
    allok = True
    print('{:>8s} {:>8s} {:>8s} {:>7s} {:>11s} {:>11s} {:>9s} {:>9s} {:>9s} {:>6s}  {}'.format(
        'T [K]', 'rho', 'mode', 'groups', 'Ndot/exact', 'L/L_P', 'grp<3sig', 'Er/exact',
        'kapJ/ex', 'first', 'ok'))
    print('  (first: absorbed samples that died in their first step; prob absorption)')
    for pt in args.points:
        temp, rho = (float(v) for v in pt.split(','))
        kp = tab.planck(temp, rho)
        s = tt.SIGMA_T*tab.ne(temp, rho)/rho
        w, kh, kl = tab.split(temp, rho, s, args.w0)
        two = int((w < 1.).sum())
        n_g = np.array([tt.photon_band(edges[g], edges[g+1], temp) for g in range(ng)])
        b_g = np.array([tt.planck_band(edges[g], edges[g+1], temp) for g in range(ng)])
        lum_exact = 4.*np.pi*rho*kp*b_g
        # thin: tau 1e-4 across the box at the thickest band; closed: tau 10 at the
        # thinnest band of any group that emits, high band included
        emitting = b_g > 1.e-8*b_g.max()
        kmax = max(kp.max(), kh.max())
        kmin = min(kp[emitting].min(), kl[emitting][kl[emitting] > 0].min()
                   if (kl[emitting] > 0).any() else kp[emitting].min())
        length = 1.e-4/(rho*kmax)
        length_closed = 10./(rho*kmin)
        for mode in args.modes:
            extra = dict(MODES[mode], two_band_w='{:g}'.format(args.w0))
            # Thermalized: the high band of a two-band group is thick when
            # rho kappa_h W > 10 with W the cell size, which differs between the two
            # boxes.  Its emission goes to the low band less what the high band keeps;
            # the transported field then holds the low band's share of those groups.
            def effective(width):
                thick = (w < 1.) & (kh*rho*width > 10.) if mode == 'thick' \
                    else np.zeros(ng, bool)
                ps = s/(kh + s)
                kp_e = np.where(thick, (1.-w)*(kl + w*kh*ps/(1.-w*ps)), kp)
                share = np.where(thick, 1.-w, 1.)      # of B_g in the transported field
                # kappa J over the transported field is the absorbed power, which in
                # equilibrium is the emitted one: the low band's own and the leakage
                # it receives, absorbed again on scattering back into the high band
                return kp_e, share, kp_e
            kp_eff, _, _ = effective(length/8.)
            _, share_c, kj_c = effective(length_closed/8.)
            er_exact = 4.*np.pi/tt.C*(share_c*b_g).sum()
            kapj_exact = (kj_c*b_g).sum()/(share_c*b_g).sum()
            ndot_exact = 4.*np.pi*rho*(kp_eff*n_g).sum()
            tag = 'T{:.3g}_r{:.3g}_{}'.format(temp, rho, mode)
            dthin = os.path.join(workdir, 'thin', tag)
            dclosed = os.path.join(workdir, 'closed', tag)
            for d, closed in ((dthin, False), (dclosed, True)):
                u.link_inputs(d, table, nfree)
                ath = os.path.join(d, 'athinput.tbbox')
                box_athinput(ath, os.path.basename(table), os.path.basename(nfree),
                             args.nphot_closed if closed else args.nphot, args.iseed,
                             temp, rho, length_closed if closed else length, closed, extra)
                u.run(binary, ath, d, args.mcranks)
            rate, sig, total, lum, nph = escaping(dthin, edges)
            vol = length**3
            ndot_ratio = rate/(ndot_exact*vol)
            lum_ratio = total/(lum_exact.sum()*vol)
            ref = lum_exact*vol
            if mode == 'thick':
                ref = 4.*np.pi*rho*kp_eff*b_g*vol
            sg = np.where(nph > 0, 1./np.sqrt(np.maximum(nph, 1)), 1.)
            dev = np.abs(lum/np.maximum(ref, 1.e-300) - 1.)
            within = int(((dev < 3.*sg + 1.e-3) | (nph < 5)).sum())
            lab = athena_read.athdf(glob.glob(os.path.join(dclosed, '*.lab.*athdf'))[0],
                                    quantities=['Ermc'])
            usr = athena_read.athdf(glob.glob(os.path.join(dclosed, '*.uom.*athdf'))[0],
                                    quantities=['kapJ'])
            er = float(np.mean(lab['Ermc']))
            kapj = float(np.mean(usr['kapJ']))/er/rho
            first = u.grep(os.path.join(dclosed, 'run.log'), 'died in their first step')
            first = first[-1].split('(')[1].split(')')[0] if first else '-'
            # The closed box is gated for the single-band and thermalized runs.  The
            # plain two-band run's field is carried by the few samples born in the low
            # band or scattered into it, 1e-6 of them in a line-dominated cell, so its
            # energy density is only reported; so is the single-band run's where the
            # Planck means of the emitting groups span more than four decades, since the
            # samples born in the most transparent groups then dominate its variance
            # (see ff_table.py) beyond what a million photons settle.
            ok = abs(ndot_ratio - 1.) < 1.e-3 + 3.*sig and within >= ng - 1
            span = kp[emitting].max()/max(kp[emitting].min(), 1.e-300)
            gate_closed = (mode == 'thick') or (mode == 'single' and span < 1.e4)
            if gate_closed:
                ok = ok and abs(er/er_exact - 1.) < 0.10 and abs(kapj/kapj_exact - 1.) < 0.10
            allok = allok and ok
            print('{:8.3g} {:8.3g} {:>8s} {:3d}/{:<3d} {:11.6f} {:11.5f} {:5d}/{:<3d} {:9.4f} '
                  '{:9.4f} {:>6s}  {}'.format(temp, rho, mode, two, ng, ndot_ratio, lum_ratio,
                                               within, ng, er/er_exact, kapj/kapj_exact,
                                               first, ('ok' if ok else 'FAIL')
                                               + ('' if gate_closed else ' (box not gated)')))
    print('PASS' if allok else 'FAIL')
    sys.exit(0 if allok else 1)


if __name__ == '__main__':
    main()
