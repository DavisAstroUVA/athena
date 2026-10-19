#! /usr/bin/env python
"""
The table opacities against free-free, on a table that is free-free.

make_ff_table.py writes a multigroup table whose Planck group means are those of the
module's own free-free opacity, so a run with emission = table, absorption = table on it
and a run with emission = freefree, absorption = freefree describe the same gas.  Over a
grid of (T, rho) that sits on the table's nodes, mc_isoth_opac runs two uniform boxes in
each of three schemes (table read as group means, table read at the group edges, and
free-free):

  thin box      escape boundaries, optically thin at every group.  The escaping photon
                rate (sum of the sample weights) is the cell emission to the absorbed
                fraction, so it tests the emission integral deterministically: the mean
                reading must give 4 pi rho sum_g kappa_g N_g with N_g the photon Planck
                integral over the group; the free-free scheme its own integral.  The
                escaping energy per group tests the group spectrum against
                4 pi rho kappa_P,g B_g, which is the same number for the table and for
                free-free by the definition of the Planck mean, within the sampling noise.
  closed box    periodic, scattering = none.  The radiation field relaxes to the Planck
                function on the table's window, so the energy density must be
                (4 pi/c) sum_g B_g and the J-weighted absorption kapJ/Er the Planck mean of
                free-free over the window: the same for the mean reading and free-free,
                and about 30% low for the edge reading, which this test records.  The
                energy density is a noisy estimate under the table schemes, which draw
                photon energies from the Planck spectrum: a photon's path, and its
                contribution, grow as E^4 (opacity falling as E^-3) while the counts fall
                as e^-x, so the Wien-tail photons dominate the variance and a million
                photons give a few percent.  The free-free sampler draws energies flat in
                ln E with e^-x in the weight and has no such tail.

  python ff_table.py --path /path/to/athena --workdir DIR [--temps 1e4 1e5 1e6 1e7 1e8]
         [--dens 1e-12 1e-9 1e-6] [--nphot 20000] [--table TABLE.txt] [--emax-ext 100]

With --emax-ext the window of every scheme is extended to that energy in keV: the table
schemes through table_emax, whose groups above the table carry the free-free Planck
means (table_above = freefree), the free-free scheme through its emax; the analytic
values follow, so this exercises the extension against the same integrals.

The binary must be built with --prob=mc_isoth_opac -mc -hdf5.  Without --table the
script writes ff_uniform.txt (60 temperatures, 31 densities) into the workdir with
make_ff_table.py, which takes a minute; the temperatures and densities asked for are
moved to the nearest interior nodes.  PASS/FAIL on the deterministic quantities at 1e-3 (the thin
box's photon rate, the closed box's mean-reading and free-free energy density and kapJ
against the analytic values; the closed box at 10% for the table and 3% for free-free)
and on the per-group energies at three sigma.
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
sys.path.insert(0, os.path.join(VIS, 'montecarlo', 'problem_specific', 'opacity_table'))
import opacity_test_util as u  # noqa: E402
import athena_read  # noqa: E402
import band_luminosity  # noqa: E402
import make_ff_table as ff  # noqa: E402

H, C, KB, KEV = ff.H, ff.C, ff.KB, ff.KEV
ETA0 = 1.032521e-11  # the module's free-free photon emissivity prefactor


def read_table_grid(path, emax_ext=0.):
    """edges [erg] (ngroups+1, the top one continued, and beyond it to emax_ext keV with
    the same spacing), T grid [K], rho grid"""
    with open(path) as f:
        ng, nt, nr = (int(f.readline()) for _ in range(3))
        lo = np.array([float(f.readline()) for _ in range(ng)])*KEV
        tg = np.array([float(f.readline()) for _ in range(nt)])*ff.KEV_K
        rg = np.array([float(f.readline()) for _ in range(nr)])
    dle = (np.log10(lo[-1]) - np.log10(lo[0]))/(ng - 1)
    next_ = 0
    top = 10.**(np.log10(lo[0]) + ng*dle)
    if emax_ext*KEV > top:
        next_ = int(np.ceil(np.log10(emax_ext*KEV/top)/dle))
    edges = 10.**(np.log10(lo[0]) + dle*np.arange(ng + next_ + 1))
    edges[:ng] = lo
    return edges, tg, rg


def photon_planck(elo, ehi, temp, n=2001):
    """int B_nu/(h nu) dnu over [elo, ehi], photons/cm^2/s/sr"""
    nu, w = ff._lognu(elo, ehi, n)
    return float((ff.planck_nu(nu, temp)/(H*nu)*nu*w).sum())


def energy_planck(elo, ehi, temp, n=2001):
    nu, w = ff._lognu(elo, ehi, n)
    return float((ff.planck_nu(nu, temp)*nu*w).sum())


def ff_group_rates(elo, ehi, temp, rho, heabund, n=2001):
    """free-free photon rate and energy rate per volume over [elo, ehi], 4 pi included"""
    nu, w = ff._lognu(elo, ehi, n)
    j = ff.kappa_ff(nu, temp, rho, heabund)*rho*ff.planck_nu(nu, temp)  # chi B
    return 4.*np.pi*float((j/(H*nu)*nu*w).sum()), 4.*np.pi*float((j*nu*w).sum())


def box_athinput(path, table, nphot, iseed, scheme, temp, rho, length, closed, extra):
    """mc_isoth_opac uniform box, 8^3 cells in one block"""
    if scheme == 'freefree':
        emission = absorption = 'freefree'
    else:
        emission = absorption = 'table'
    keys = dict(extra)
    if scheme != 'freefree':
        keys['opacity_file'] = table
        keys['table_groups'] = scheme
    bc = 'periodic' if closed else 'escape'
    text = """<comment>
problem   = uniform box, table opacities against free-free
configure = --prob=mc_isoth_opac -mc -mpi -hdf5

<job>
problem_id = ffbox

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
        nphot, iseed, 'none', emission, absorption, keys) + """
<problem>
constdens = true
dens      = {rho:e}
temp      = {temp:e}
emin      = {emin:e}
emax      = {emax:e}
""".format(rho=rho, temp=temp, emin=extra_emin(table),
           emax=min(extra_emax(table), ff_emax_ev(temp)))
    with open(path, 'w') as f:
        f.write(text)


_grid_cache = {}
EMAX_EXT = 0.


def grid(table):
    if table not in _grid_cache:
        _grid_cache[table] = read_table_grid(table, EMAX_EXT)
    return _grid_cache[table]


def ff_emax_ev(temp):
    """
    Top of the free-free sampler's window: 60 kT in eV.  The sampler draws energies flat
    in ln E with e^-x in the weight, so a window reaching the table's top edge at a low
    temperature would spend most of its samples on photons of no weight whose free paths,
    at an opacity falling as E^-3, are astronomically long in a closed box.  The emission
    beyond 60 kT is e^-60 of the total.
    """
    return 60.*KB*temp/1.602176634e-12


def extra_emin(table):
    return grid(table)[0][0]/1.602176634e-12  # eV


def extra_emax(table):
    return grid(table)[0][-1]/1.602176634e-12


def escaping(workdir, edges):
    """total photon rate and its relative sampling error, total energy rate, energy rate
    per group, photons per group"""
    files = sorted(glob.glob(os.path.join(workdir, '*.list')))
    rate = 0.
    wsq = 0.
    nphot_g = np.zeros(len(edges)-1)
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
            ok = (idx >= 0) & (idx < len(nphot_g))
            np.add.at(nphot_g, idx[ok], 1)
            if c['done']:
                break
    total, lum, _ = band_luminosity.band_luminosities(files, edges/KEV)
    return rate, np.sqrt(wsq)/rate, total, lum, nphot_g


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--path', default=os.path.join(HERE, '..', '..', '..'),
                    help='root of the tree (bin/athena), or a binary')
    ap.add_argument('--workdir', default='.', help='where to run')
    ap.add_argument('--table', default=None, help='a make_ff_table.py table; else written')
    ap.add_argument('--temps', type=float, nargs='*', default=[1.e4, 1.e5, 1.e6, 1.e7, 1.e8])
    ap.add_argument('--dens', type=float, nargs='*', default=[1.e-12, 1.e-9, 1.e-6])
    ap.add_argument('--schemes', nargs='*', default=['mean', 'edge', 'freefree'])
    ap.add_argument('--nphot', type=int, default=20000, help='thin box')
    ap.add_argument('--nphot-closed', type=int, default=1000000, help='closed box')
    ap.add_argument('--iseed', type=int, default=146137)
    ap.add_argument('--heabund', type=float, default=0.09)
    ap.add_argument('--mcranks', type=int, default=1)
    ap.add_argument('--emax-ext', type=float, default=0., help='extend the window [keV]')
    args = ap.parse_args()

    global EMAX_EXT
    EMAX_EXT = args.emax_ext
    binary = args.path if os.path.isfile(args.path) else os.path.join(args.path, 'bin', 'athena')
    workdir = os.path.abspath(args.workdir)
    os.makedirs(workdir, exist_ok=True)
    table = args.table and os.path.abspath(args.table)
    if table is None:
        table = os.path.join(workdir, 'ff_uniform.txt')
        if not os.path.exists(table):
            edges = ff.group_edges(32, 1.e-3, 7.4989)
            ff.write_tables(table, edges, np.geomspace(5.8e3, 1.e9, 60),
                            np.geomspace(1.e-18, 1.e-6, 31), args.heabund)
    edges, tg, rg = grid(table)
    lo_edges = read_table_grid(table)[0][:-1]
    ng = len(edges) - 1
    extra = {'table_emax': '{:g}'.format(args.emax_ext)} if args.emax_ext > 0. else {}
    # interior nodes: a cell on the last node of an axis is off the grid to Locate
    temps = [tg[1:-1][np.argmin(np.abs(np.log(tg[1:-1]/t)))] for t in args.temps]
    dens = [rg[1:-1][np.argmin(np.abs(np.log(rg[1:-1]/r)))] for r in args.dens]

    allok = True
    print('{:>9s} {:>9s} {:>8s}  {:>12s} {:>12s} {:>12s} {:>12s} {:>12s} {:>7s}'.format(
        'T [K]', 'rho', 'scheme', 'Ndot/exact', 'L/L_ff', 'groups<3sig', 'Er/exact',
        'kapJ/exact', 'ok'))
    for temp in temps:
        for rho in dens:
            # analytic: per-group photon and energy rates of the three schemes
            kap = np.array([ff.planck_group(edges[g], edges[g+1], temp, rho, args.heabund)
                            for g in range(ng)])
            n_g = np.array([photon_planck(edges[g], edges[g+1], temp) for g in range(ng)])
            b_g = np.array([energy_planck(edges[g], edges[g+1], temp) for g in range(ng)])
            # the free-free scheme's window stops at 60 kT (ff_emax_ev); the groups
            # beyond it get what lies below the cap, which is nothing to e^-60
            cap = ff_emax_ev(temp)*1.602176634e-12
            ff_n, ff_e = zip(*[ff_group_rates(edges[g], min(edges[g+1], cap), temp, rho,
                                              args.heabund)
                               if edges[g] < cap else (0., 0.) for g in range(ng)])
            ff_n, ff_e = np.array(ff_n), np.array(ff_e)
            exact = {
                'mean': (4.*np.pi*rho*(kap*n_g).sum(), 4.*np.pi*rho*kap*b_g),
                'freefree': (ff_n.sum(), ff_e),
            }
            # the edge reading's own integral: nodes at the table's lower bounds (no
            # extension), trapezoid in ln nu
            nt_ = len(lo_edges)
            nu = edges[:nt_]/H
            eta = kap[:nt_]*rho*ff.planck_nu(nu, temp)/H
            dln = np.diff(np.log(nu))
            edge_n = 4.*np.pi*(0.5*(eta[1:] + eta[:-1])*dln).sum()
            exact['edge'] = (edge_n, None)
            # the thin box: tau = 1e-4 across it at the thickest group.  The closed box:
            # tau = 10 across it at the thinnest group that emits at all, so that every
            # photon is absorbed within a crossing and no rare photon in a transparent
            # group carries a path, and a contribution to the energy density, out of
            # proportion
            length = 1.e-4/(rho*kap.max())
            emitting = b_g > 1.e-8*b_g.max()
            length_closed = 10./(rho*kap[emitting].min())
            # the closed box's targets: Planck on the window, and the window Planck mean
            er_exact = 4.*np.pi/C*b_g.sum()
            kapj_exact = (kap*b_g).sum()/b_g.sum()
            for scheme in args.schemes:
                tag = 'T{:.3g}_r{:.3g}_{}'.format(temp, rho, scheme)
                dthin = os.path.join(workdir, 'thin', tag)
                dclosed = os.path.join(workdir, 'closed', tag)
                for d, closed in ((dthin, False), (dclosed, True)):
                    u.link_inputs(d, table)
                    ath = os.path.join(d, 'athinput.ffbox')
                    box_athinput(ath, os.path.basename(table),
                                 args.nphot_closed if closed else args.nphot, args.iseed,
                                 scheme, temp, rho, length_closed if closed else length,
                                 closed, extra)
                    u.run(binary, ath, d, args.mcranks)
                rate, sig_rate, total, lum, nph = escaping(dthin, edges)
                vol = length**3
                ndot_ratio = rate/(exact[scheme][0]*vol)
                lum_ratio = total/(ff_e.sum()*vol)
                # per-group energies against the free-free integral, noise from the count
                ref = exact['freefree'][1]*vol
                sig = np.where(nph > 0, 1./np.sqrt(np.maximum(nph, 1)), 1.)
                dev = np.abs(lum/np.maximum(ref, 1.e-300) - 1.)
                within = int(((dev < 3.*sig + 1.e-3) | (nph < 5)).sum())
                lab = athena_read.athdf(glob.glob(os.path.join(dclosed, '*.lab.*athdf'))[0],
                                        quantities=['Ermc'])
                usr = athena_read.athdf(glob.glob(os.path.join(dclosed, '*.uom.*athdf'))[0],
                                        quantities=['kapJ'])
                er = float(np.mean(lab['Ermc']))
                kapj = float(np.mean(usr['kapJ']))/er/rho
                er_ratio = er/er_exact
                kapj_ratio = kapj/kapj_exact
                # Gates.  The photon rate to 1e-3 plus three times its sampling error
                # (zero for the table schemes, whose weights are equal; the free-free
                # sampler's weights carry e^-x).  The group energies and the closed box
                # for the mean reading and free-free only: the edge reading's groups
                # are registered differently and its field is not Planck.
                ok = abs(ndot_ratio - 1.) < 1.e-3 + 3.*sig_rate
                if scheme != 'edge':
                    ok = ok and within >= ng - 1
                    tol = 0.03 if scheme == 'freefree' else 0.10
                    ok = ok and abs(er_ratio - 1.) < tol and abs(kapj_ratio - 1.) < tol
                allok = allok and ok
                print('{:9.3g} {:9.3g} {:>8s}  {:12.6f} {:12.5f} {:9d}/{:<2d} {:12.5f} '
                      '{:12.5f} {:>7s}'.format(temp, rho, scheme, ndot_ratio, lum_ratio,
                                               within, ng, er_ratio, kapj_ratio,
                                               'ok' if ok else 'FAIL'))
    print('PASS' if allok else 'FAIL')
    sys.exit(0 if allok else 1)


if __name__ == '__main__':
    main()
