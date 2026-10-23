#! /usr/bin/env python
"""
The electron density from the free-electron table (table_ne, nfree_file).

mc_isoth_opac runs a closed uniform box at each (T, rho) with emission = table,
absorption = table and table_ne = true (no scattering: the moment below needs only
paths, and photons that scatter thousands of times before a weak absorption would make
the run long for nothing), and its nelJ user moment,
the n_e-weighted mean intensity, divided by the energy density gives the cell's electron
density exactly, whatever the sampling noise.  That is compared with
rho N_free(T, rho) / (Abar m_u) from the nfree file, interpolated bilinearly in (log T,
log rho) as the code does, at points on and between the grid's nodes, to 1e-5 (the
opacity table's temperature grid, which sets the bracket, is written to four digits).  The
same box without table_ne must give the fully ionized hydrogen-helium value, and a run
on a free-free table's nfree file (make_ff_table.py, full ionization) with table_ne must
reproduce it to 1e-4.  PASS/FAIL.

  python ne_table.py --path /path/to/athena --workdir DIR --table TABLE.txt
         [--nfree TABLE_nfree.txt] [--points T,rho ...] [--nphot 20000]

The binary must be built with --prob=mc_isoth_opac -mc -hdf5.  The default points are
chosen for a solar-abundance TOPS table (tops_regrid.py), where N_free falls well below
full ionization at 1e4 K.
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

KB = 1.380649e-16
MP = 1.67262192369e-24
MU = 1.66053906660e-24
KEV_K = 1.602176634e-9/KB
SIGMA_T = 6.65248e-25


def read_nfree(path):
    """T [K], rho, N_free (nT, nrho), Abar"""
    abar = None
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                if 'Abar =' in line:
                    abar = float(line.split('Abar =')[1].split()[0])
                continue
            rows.append(line)
    nt, nr = int(rows[0]), int(rows[1])
    tg = np.array(rows[2:2+nt], float)*KEV_K
    rg = np.array(rows[2+nt:2+nt+nr], float)
    nf = np.array([r.split() for r in rows[2+nt+nr:2+nt+nr+nt]], float)
    return tg, rg, nf, abar


def nfree_at(tg, rg, nf, temp, rho):
    """bilinear in (log T, log rho), the bracket and weights as OpacityTable::Locate"""
    lt, lr = np.log10(tg), np.log10(rg)
    dlt = (lt[-1] - lt[0])/(len(lt) - 1)
    dlr = (lr[-1] - lr[0])/(len(lr) - 1)
    xi = (np.log10(rho) - lr[0])/dlr
    ii = int(np.floor(xi))
    ii = min(max(ii, 0), len(rg) - 2)
    xi = min(max(xi - ii, 0.), 1.)
    jj = int(np.floor((np.log10(temp) - lt[0])/dlt))
    jj = min(max(jj, 0), len(tg) - 2)
    while jj < len(tg) - 2 and tg[jj+1] < temp:
        jj += 1
    while jj > 0 and tg[jj] > temp:
        jj -= 1
    xj = np.log(temp/tg[jj])/np.log(tg[jj+1]/tg[jj])
    xj = min(max(xj, 0.), 1.)
    return ((1.-xi)*((1.-xj)*nf[jj, ii] + xj*nf[jj+1, ii])
            + xi*((1.-xj)*nf[jj, ii+1] + xj*nf[jj+1, ii+1]))


def box_athinput(path, table, nfree, nphot, iseed, temp, rho, length, table_ne):
    extra = {'opacity_file': table}
    if table_ne:
        extra['nfree_file'] = nfree
        extra['table_ne'] = 'true'
    text = """<comment>
problem   = closed uniform box, electron density from the table
configure = --prob=mc_isoth_opac -mc -mpi -hdf5

<job>
problem_id = nebox

<output1>
file_type = hdf5
variable  = mclab
id        = lab
dt = 1.e-11

<output2>
file_type = hdf5
variable  = uom
id        = uom
dt = 1.e-11

<mesh>
nx1        = 4
x1min      = {lo:e}
x1max      = {hi:e}
ix1_mc_bc  = periodic
ox1_mc_bc  = periodic
nx2        = 4
x2min      = {lo:e}
x2max      = {hi:e}
ix2_mc_bc  = periodic
ox2_mc_bc  = periodic
nx3        = 4
x3min      = {lo:e}
x3max      = {hi:e}
ix3_mc_bc  = periodic
ox3_mc_bc  = periodic

<meshblock>
nx1 = 4
nx2 = 4
nx3 = 4

<hydro>
gamma = 1.666666666666667

""".format(lo=-0.5*length, hi=0.5*length) + u.mc_block(
        nphot, iseed, 'none', 'table', 'table', extra) + """
<problem>
constdens = true
dens      = {rho:e}
temp      = {temp:e}
""".format(rho=rho, temp=temp)
    with open(path, 'w') as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--path', default=os.path.join(HERE, '..', '..', '..'))
    ap.add_argument('--workdir', default='.')
    ap.add_argument('--table', required=True, help='opacity table')
    ap.add_argument('--nfree', default=None, help='its nfree file; default <table>_nfree.txt')
    ap.add_argument('--points', nargs='*', default=['1e4,1e-10', '1.2e4,3e-14', '2e4,1e-15',
                                                    '3.3e5,2.2e-12', '1e6,1e-9'],
                    help='T,rho pairs [K, g/cm^3]')
    ap.add_argument('--nphot', type=int, default=5000)
    ap.add_argument('--iseed', type=int, default=146137)
    ap.add_argument('--heabund', type=float, default=0.09)
    ap.add_argument('--mcranks', type=int, default=1)
    args = ap.parse_args()

    binary = args.path if os.path.isfile(args.path) else os.path.join(args.path, 'bin', 'athena')
    table = os.path.abspath(args.table)
    nfree = os.path.abspath(args.nfree) if args.nfree else table[:-4] + '_nfree.txt'
    workdir = os.path.abspath(args.workdir)
    tg, rg, nf, abar = read_nfree(nfree)
    print('{}: Abar = {}, N_free {:.4f} to {:.4f}'.format(os.path.basename(nfree), abar,
                                                        nf.min(), nf.max()))
    allok = True
    print('{:>9s} {:>9s} {:>8s} {:>13s} {:>13s} {:>10s} {:>13s} {:>10s}  {}'.format(
        'T [K]', 'rho', 'N_free', 'n_e table', 'n_e run', 'ratio', 'n_e H/He run', 'ratio',
        'ok'))
    for pt in args.points:
        temp, rho = (float(v) for v in pt.split(','))
        nfr = nfree_at(tg, rg, nf, temp, rho)
        ne_tab = rho*nfr/(abar*MU)
        nh = rho/(MP*(1. + 4.*args.heabund))
        ne_hhe = nh*(1. + 2.*args.heabund)
        # the box is ten Thomson depths across at the table's n_e; its size does not
        # enter the measurement
        length = 10./(SIGMA_T*max(ne_tab, 1.e-3*ne_hhe))
        got = {}
        for table_ne in (True, False):
            d = os.path.join(workdir, 'T{:.3g}_r{:.3g}_{}'.format(temp, rho,
                                                               'table' if table_ne else 'hhe'))
            u.link_inputs(d, table, nfree)
            ath = os.path.join(d, 'athinput.nebox')
            box_athinput(ath, os.path.basename(table), os.path.basename(nfree), args.nphot,
                         args.iseed, temp, rho, length, table_ne)
            u.run(binary, ath, d, args.mcranks)
            lab = athena_read.athdf(glob.glob(os.path.join(d, '*.lab.*athdf'))[0],
                                    quantities=['Ermc'])
            usr = athena_read.athdf(glob.glob(os.path.join(d, '*.uom.*athdf'))[0],
                                    quantities=['nelJ'])
            er = lab['Ermc'].ravel()
            ne = usr['nelJ'].ravel()
            ok = er > 0.
            got[table_ne] = (ne[ok]/er[ok])
        r_tab = got[True]/ne_tab
        r_hhe = got[False]/ne_hhe
        # 1e-5: the bracket weights come from the opacity table's temperature grid, which
        # tops_regrid.py writes to four digits, the nfree file's to five
        ok = np.all(np.abs(r_tab - 1.) < 1.e-5) and np.all(np.abs(r_hhe - 1.) < 1.e-6)
        allok = allok and ok
        print('{:9.3g} {:9.3g} {:8.4f} {:13.5e} {:13.5e} {:10.7f} {:13.5e} {:10.7f}  {}'.format(
            temp, rho, nfr, ne_tab, got[True].mean(), r_tab.mean(), got[False].mean(),
            r_hhe.mean(), 'ok' if ok else 'FAIL'))
    print('PASS' if allok else 'FAIL')
    sys.exit(0 if allok else 1)


if __name__ == '__main__':
    main()
