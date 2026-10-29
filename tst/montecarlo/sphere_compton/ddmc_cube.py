#! /usr/bin/env python

"""
The discrete diffusion lattice against its own exact solution, and against transport.

A uniform Thomson-scattering cube fills the whole mesh, every mesh boundary is an escape
boundary, and a point source sits at the centre cell.  With the lattice on, every cell is
a diffusion cell and every mesh face is a boundary face of the lattice, so the run is a
pure random walk on the lattice: no interface with transport cells is ever crossed, and
a walker leaving through a mesh face emerges and escapes at once.  The mean escape
path of that walk, and the mean number of dwells, follow from the lattice's rates alone
by solving the mean first-passage system once, which this script does.  Two comparisons
come out of one run:

  lattice against itself   the run's mean path (the scatterings its dwells stood for,
                           from the report line, or the escapers' path lengths) and its
                           mean number of dwells against the exact solution.  This is
                           the check of the code: the leakage rates, the face draw, the
                           dwell draw and the bookkeeping, with no modelling error in
                           the way.  It must agree to the statistics.

  lattice against analog   the same cube run with the lattice off.  The difference is the
                           lattice's discretization and its boundary condition, which
                           fall with the cell size; at a cell depth of 5 it is about a
                           percent in the mean path.

The cube's half-width optical depth is --tau; the cell depth is 2 tau / nx.

    python ddmc_cube.py --path /path/to/athena --workdir DIR [--nx 8 --tau 40 --nphot 20000]
                        [--ddmc-tau 3 --emergence asymptotic --mcranks 4 --mb 8]

needs: python configure.py --prob=mc_sphere_isoth -mc -mpi && make
"""

import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'vis', 'python', 'montecarlo'))
import athena_mc  # noqa: E402

LAMBDA = 0.7104   # extrapolation distance of the asymptotic boundary condition
R0 = 1.e10        # the cube's half-width is 1.25 R0, as the gridded sphere decks


def write_athinput(filename, name, nx, mb, tau, nphot, iseed, ddmc, ddmc_tau, emergence):
    """the gridded-sphere deck with the sphere pushed out to a hundred times the mesh, so
    the medium fills the mesh and the escape surface is never met; escape at the mesh
    boundaries.  mc_sphere_isoth takes the density from tau over radius, so the deck's
    tau is scaled to keep the half-width depth at tau."""
    h = 2.5*R0/nx
    lo, hi = -1.25*R0 - 0.5*h, 1.25*R0 - 0.5*h
    radius = 100.*R0
    mesh = "\n".join(f"nx{d} = {nx}\nx{d}min = {lo:.7e}\nx{d}max = {hi:.7e}\n"
                     f"ix{d}_mc_bc = escape\nox{d}_mc_bc = escape" for d in '123')
    s = f"""<job>
problem_id = {name}

<output1>
file_type = phlist
nuser     = 3

<mesh>
{mesh}

<meshblock>
nx1 = {mb}
nx2 = {mb}
nx3 = {mb}

<hydro>
gamma = 1.666666666666667

<montecarlo>
nphot      = {nphot}
iseed      = {iseed}
scattering = thomson
absorption = none
abs_method = weight
polarized  = none
verbose    = true

<ddmc>
enable  = {'true' if ddmc else 'false'}
tau     = {ddmc_tau}
emergence = {emergence}
report  = true

<problem>
temp     = 1.1865e7
tau      = {tau*radius/(1.25*R0)}
x0       = 1.
radius   = {radius:.6e}
srcdist  = false
heabund  = 0.09
"""
    open(filename, 'w').write(s)


def exact_lattice(nx, chid):
    """mean escape path (mean free paths) and mean number of dwells of a walker starting
    at the centre cell of an nx^3 lattice of cells of depth chid, with the code's rates"""
    k_int = 2./(3.*(chid + chid))          # per cell width; Delta = 1
    k_bdy = 2./(3.*(chid + 2.*LAMBDA))
    n = nx**3

    def idx(i, j, k):
        return (k*nx + j)*nx + i
    rows, cols, vals = [], [], []
    for k in range(nx):
        for j in range(nx):
            for i in range(nx):
                c = idx(i, j, k)
                ktot = 0.
                for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                                   (0, 0, 1), (0, 0, -1)):
                    ii, jj, kk = i+di, j+dj, k+dk
                    inside = 0 <= ii < nx and 0 <= jj < nx and 0 <= kk < nx
                    kf = k_int if inside else k_bdy
                    ktot += kf
                    if inside:
                        rows.append(c)
                        cols.append(idx(ii, jj, kk))
                        vals.append(-kf)
                rows.append(c)
                cols.append(c)
                vals.append(ktot)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    t = spl.spsolve(A, np.ones(n))          # mean path in cell widths
    ndwell = spl.spsolve(A, A.diagonal())   # mean number of dwells
    c0 = idx(nx//2, nx//2, nx//2)           # the origin is the centre of cell nx/2
    return t[c0]*chid, ndwell[c0]


def load_paths(files):
    """escapers' path lengths from the lists"""
    path = []
    for f in files:
        gen = athena_mc.read_list_generator(f)
        header = next(gen)['header']
        for c in gen:
            if c['chunk'] is None:
                continue
            p = athena_mc.Photons.from_chunk(header, c)
            path.append(p.x0)
            if c['done']:
                break
    return np.concatenate(path)


def run(args, name, ddmc):
    deck = os.path.join(args.workdir, f'athinput.{name}')
    write_athinput(deck, name, args.nx, args.mb or args.nx, args.tau, args.nphot,
                   args.iseed, ddmc, args.ddmc_tau, args.emergence)
    exe = os.path.join(args.path, 'bin', 'athena')
    cmd = [exe, '-i', deck, '-d', args.workdir]
    if args.mcranks > 1:
        cmd = ['mpirun', '-n', str(args.mcranks)] + cmd
    log = os.path.join(args.workdir, f'{name}.log')
    with open(log, 'w') as f:
        ret = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    if ret != 0:
        sys.exit(f"athena exited with {ret}; see {log}")
    report = {}
    for line in open(log):
        if line.startswith('discrete diffusion:'):
            w = line.split()
            report['cells'] = int(w[2])
            report['dwells'] = int(w[4])
            report['scat'] = float(w[8])
        if line.startswith('ntot:'):
            w = line.split()
            report['ntot'] = int(w[1])
        if 'cpu time used' in line:
            report['cpu'] = float(line.split('=')[1].split()[0])
    return load_paths(sorted(glob.glob(os.path.join(args.workdir, f'{name}.out1.*.list')))), \
        report


def main(args):
    os.makedirs(args.workdir, exist_ok=True)
    chi = args.tau/(1.25*R0)
    chid = 2.*args.tau/args.nx
    print(f"cube of half-width depth {args.tau:g} over {args.nx}^3 cells: cell depth {chid:.3g}, "
          f"threshold {args.ddmc_tau:g}")
    t_exact, n_exact = exact_lattice(args.nx, chid)
    print(f"  exact lattice: mean escape path {t_exact:.1f} mean free paths, "
          f"{n_exact:.2f} dwells per photon")
    ok = True
    pd, rd = run(args, 'ddmc', True)
    tp = pd*chi
    err = tp.std()/np.sqrt(len(tp))
    pull = (tp.mean() - t_exact)/err
    dw = rd['dwells']/rd['ntot']
    print(f"  lattice run:   {rd['cells']} cells; mean escape path {tp.mean():.1f} +- {err:.1f} "
          f"(pull {pull:+.1f}), scatterings stood for per photon {rd['scat']/rd['ntot']:.1f}, "
          f"{dw:.2f} dwells per photon ({dw/n_exact-1.:+.2%}); cpu {rd.get('cpu', 0):.2f} s")
    if abs(pull) > 4. or abs(dw/n_exact - 1.) > 0.03:
        ok = False
    if not args.no_analog:
        pa, ra = run(args, 'analog', False)
        ta = pa*chi
        erra = ta.std()/np.sqrt(len(ta))
        print(f"  analog run:    mean escape path {ta.mean():.1f} +- {erra:.1f}; cpu "
              f"{ra.get('cpu', 0):.2f} s; lattice/analog {tp.mean()/ta.mean():.4f}, "
              f"exact lattice/analog {t_exact/ta.mean():.4f}")
        if rd.get('cpu', 0) > 0:
            print(f"  cost: {1e6*rd['cpu']/rd['dwells']:.2f} us per dwell, "
                  f"{1e6*ra['cpu']/(ta.mean()*len(ta)):.2f} us per scattering, "
                  f"speedup {ra['cpu']/rd['cpu']:.1f}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--path', default=os.path.join(HERE, '..', '..', '..'))
    ap.add_argument('--workdir', default='.')
    ap.add_argument('--nx', type=int, default=8)
    ap.add_argument('--mb', type=int, default=0, help='meshblock cells (0: one block)')
    ap.add_argument('--tau', type=float, default=40., help='half-width optical depth')
    ap.add_argument('--nphot', type=int, default=20000)
    ap.add_argument('--iseed', type=int, default=1215)
    ap.add_argument('--ddmc-tau', type=float, default=3.)
    ap.add_argument('--emergence', default='asymptotic', choices=['asymptotic', 'isotropic'])
    ap.add_argument('--mcranks', type=int, default=1)
    ap.add_argument('--no-analog', action='store_true', help='skip the analog run')
    return ap.parse_args(argv)


if __name__ == '__main__':
    sys.exit(main(parse_args()))
