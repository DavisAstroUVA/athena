#! /usr/bin/env python
"""
The frequency lattice of discrete diffusion (<ddmc> line = lya, phase R1 of
compton_ddmc_plan.md) on the uniform Lyman alpha medium of mc_sphere_lya: a slab (the
mesh one cell wide in two periodic directions, escape in the third, the escape sphere
pushed past the mesh) and a sphere (the cells of a cube whose centres lie inside the
escape surface, the exact solution taken at the radius of their volume),
a point source at the centre at line centre.  Four checks:

  slab      the lattice against Neufeld (1990), J(x) ~ x^2 sech(sqrt(pi^3/54) |x|^3 / a tau0),
            at --atau (default 1e5 1e6 1e7): the median of |x| of the escapers against
            the exact 1.0516 (a tau0)^(1/3) within --gate-median bin widths, and the
            Kolmogorov distance of the |x| distribution from the exact one below --gate-k
  sphere    the same against Dijkstra et al. (2006), J(x) ~ x^2 / (1 + cosh(sqrt(2 pi^3/27)
            |x|^3 / a tau0)), median 0.8980 (a tau0)^(1/3)
  analog    the lattice at each --nz-analog (default 8 16 32, which moves the node band's
            edge through the spectrum) against one analog run of the slab at --tau-analog
            (default 1e6) and --temp-analog (default 10 K, a tau0 = 1.5e4, where the kernel is
            affordable and the medium diffusive): the two-sample Kolmogorov distance of
            the escaping x and of the escape path within --gate-analog times the scale
  scan      the slab at --atau-scan (default 1e6) over --nz-scan with the bin width on
            Smith et al. (2018)'s matching line dx = (10 a tau0 / nz^2)^(1/3), the
            Kolmogorov distance from Neufeld and the cpu time per photon at each

The bins: --dxf and --xmaxf give the wing bin width and the grid's extent in units of
(a tau0)^(1/3) (defaults 0.21, the matching line at nz = 32, and 2.5); the core bin
half-width is --core (3.3).  The gas temperature --temp (1e4 K) sets a; tau0 follows
from a tau0.

    python sphere_lya.py --path /path/to/athena --workdir DIR [--checks slab sphere analog scan]
                         [--nz 32 --nx 32 --nphot 4000 --mcranks 8] [--plot]

needs: python configure.py --prob=mc_sphere_lya -mc -mpi && make
"""
import argparse
import glob
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'vis', 'python', 'montecarlo'))
import athena_mc  # noqa: E402

H_PLANCK = 6.62607015e-27
K_BOLTZ = 1.380649e-16
M_H = 1.007825*1.660538782e-24
C_LIGHT = 2.99792458e10
NU0 = 1.e8*C_LIGHT/1215.6701
LORWIDTH = 6.265e8/(4.*np.pi)
R0 = 1.e11   # the slab's half-thickness and the sphere's radius, cm


def doppler_width(temp):
    return NU0*np.sqrt(2.*K_BOLTZ*temp/M_H)/C_LIGHT


def voigt_a(temp):
    return LORWIDTH/doppler_width(temp)


def cdf_neufeld(ax, atau):
    """CDF of |x| of the escaping spectrum of a slab, Neufeld 1990 eq 33"""
    b = np.sqrt(np.pi**3/54.)/atau
    return 2.*np.arctan(np.tanh(0.5*b*ax**3))/(0.5*np.pi)


def cdf_dijkstra(ax, atau):
    """CDF of |x| of the escaping spectrum of a sphere, Dijkstra et al. 2006 eq 31"""
    c = np.sqrt(2.*np.pi**3/27.)/atau
    return np.tanh(0.5*c*ax**3)


MEDIAN = {'slab': 1.0516, 'sphere': 0.8980}
CDF = {'slab': cdf_neufeld, 'sphere': cdf_dijkstra}


def write_deck(filename, name, geometry, temp, tau0, nphot, iseed, ddmc, dx, xmax, core,
               nz=32, nx=32, mcranks=1, ddmc_tau=3.):
    """the slab: nz cells along x1 over +-R0, one periodic cell of 100 R0 in x2 and x3,
    escape at the x1 faces, the escape sphere at 100 R0; the sphere: nx^3 cells over
    +-1.25 R0 offset half a cell so the origin is a cell centre, escape at R0.  The
    deck's tau is the line-centre depth over the escape radius, so the slab scales it"""
    if geometry == 'slab':
        radius = 100.*R0
        tau = tau0*radius/R0
        mb1 = max(nz//mcranks, 1)
        mesh = (f"nx1 = {nz}\nx1min = {-R0:.6e}\nx1max = {R0:.6e}\n"
                "ix1_mc_bc = escape\nox1_mc_bc = escape\n")
        for d in '23':
            mesh += (f"nx{d} = 1\nx{d}min = {-50.*R0:.6e}\nx{d}max = {50.*R0:.6e}\n"
                     f"ix{d}_bc = periodic\nox{d}_bc = periodic\n"
                     f"ix{d}_mc_bc = periodic\nox{d}_mc_bc = periodic\n")
        block = f"nx1 = {mb1}\nnx2 = 1\nnx3 = 1\n"
    else:
        radius = R0
        tau = tau0
        h = 2.5*R0/nx
        lo, hi = -1.25*R0 - 0.5*h, 1.25*R0 - 0.5*h
        mesh = "".join(f"nx{d} = {nx}\nx{d}min = {lo:.7e}\nx{d}max = {hi:.7e}\n"
                       f"ix{d}_mc_bc = escape\nox{d}_mc_bc = escape\n" for d in '123')
        mb = max(nx//2, 1) if mcranks > 1 else nx
        block = f"nx1 = {mb}\nnx2 = {mb}\nnx3 = {mb}\n"
    s = f"""<job>
problem_id = {name}

<output1>
file_type = phlist
nuser     = 3

<mesh>
{mesh}
<meshblock>
{block}
<hydro>
gamma = 1.666666666666667
dfloor = 1.e-40
pfloor = 1.e-40

<montecarlo>
nphot      = {nphot}
iseed      = {iseed}
scattering = resonance
absorption = none
abs_method = weight
polarized  = none
verbose    = true

<ddmc>
enable     = {'true' if ddmc else 'false'}
tau        = {ddmc_tau}
line       = lya
line_temp  = {temp:.6e}
dx         = {dx:.6g}
xmax       = {xmax:.6g}
core_width = {core}
report     = true

<problem>
temp     = {temp:.6e}
tau      = {tau:.8e}
radius   = {radius:.6e}
x0       = 0.
heabund  = 0.09
"""
    open(filename, 'w').write(s)


def run(args, workdir, deck, name, mcranks):
    exe = os.path.join(args.path, 'bin', 'athena')
    cmd = [exe, '-i', deck, '-d', workdir]
    if mcranks > 1:
        cmd = ['mpirun', '-n', str(mcranks)] + cmd
    for f in glob.glob(os.path.join(workdir, f'{name}.*')):
        os.remove(f)
    log = os.path.join(workdir, f'{name}.log')
    with open(log, 'w') as f:
        ret = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    files = sorted(glob.glob(os.path.join(workdir, f'{name}.out1.*.list')))
    if ret != 0 or not files:
        sys.exit(f"athena exited with {ret} and wrote {len(files)} list files; see {log}")
    report = {}
    for line in open(log):
        if line.startswith('discrete diffusion:'):
            report['ddmc'] = line.strip()
            m = re.search(r'(\d+) cells \((\d+) at level jumps(?:, (\d+) nodes of (\d+) bins)?\)',
                          line)
            report['cells'] = int(m.group(1))
            report['nodes'] = int(m.group(3)) if m.group(3) else 0
            m = re.search(r'(\d+) dwells', line)
            report['dwells'] = int(m.group(1))
        if line.startswith('ntot:'):
            for key, val in re.findall(r'(?<![\w/])(ntot|nesc|nabs|ndes|nrem): (\d+)', line):
                report[key] = int(val)
            m = re.search(r'nscat/ntot: ([0-9.eE+-]+)', line)
            report['nscat'] = float(m.group(1))
        if line.startswith('cpu_seconds/photon'):
            report['cpu'] = float(line.split('=')[1].split()[0])
        if line.startswith('wall time used'):
            report['wall'] = float(line.split('=')[1].split()[0])
    return files, report


def load(files, temp):
    """x of the escapers in Doppler units of temp, their path lengths and scattering
    counts"""
    e, path, nsc = [], [], []
    for f in files:
        gen = athena_mc.read_list_generator(f)
        header = next(gen)['header']
        for c in gen:
            if c['chunk'] is None:
                continue
            p = athena_mc.Photons.from_chunk(header, c)
            e.append(p.energy)
            path.append(p.x0)
            nsc.append(p.user[:, 1] if p.nuser > 1 else np.zeros(p.energy.size))
            if c['done']:
                break
    e = np.concatenate(e)
    x = (e/H_PLANCK - NU0)/doppler_width(temp)
    return x, np.concatenate(path), np.concatenate(nsc)


def ks_exact(ax, cdf):
    """Kolmogorov distance of a sample of |x| from an exact CDF"""
    ax = np.sort(ax)
    n = ax.size
    f = cdf(ax)
    return max(np.max(np.arange(1, n+1)/n - f), np.max(f - np.arange(0, n)/n))


def ks_two(a, b):
    """two-sample Kolmogorov distance and its scale sqrt((n + m) / n m)"""
    a = np.sort(a)
    b = np.sort(b)
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side='right')/a.size
    fb = np.searchsorted(b, grid, side='right')/b.size
    return np.max(np.abs(fa - fb)), np.sqrt((a.size + b.size)/(a.size*b.size))


def peak(x, dx):
    """the two peaks of the escaping spectrum from a histogram of bin width dx"""
    edges = np.arange(-np.max(np.abs(x)) - dx, np.max(np.abs(x)) + 2*dx, dx)
    hist, edges = np.histogram(x, bins=edges)
    mid = 0.5*(edges[1:] + edges[:-1])
    neg = mid < 0
    return mid[neg][np.argmax(hist[neg])], mid[~neg][np.argmax(hist[~neg])]


def plot_spectrum(x, atau, geometry, dx, filename, label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = x/atau**(1./3.)
    bins = np.arange(-3., 3. + 1e-9, dx/atau**(1./3.))
    hist, edges = np.histogram(xs, bins=bins, density=True)
    ax.step(edges[:-1], hist, where='post', label=label)
    grid = np.linspace(-3., 3., 601)
    if geometry == 'slab':
        b = np.sqrt(np.pi**3/54.)
        j = grid**2/np.cosh(b*np.abs(grid)**3)
    else:
        c = np.sqrt(2.*np.pi**3/27.)
        j = grid**2/(1. + np.cosh(c*np.abs(grid)**3))
    j /= np.trapezoid(j, grid)
    ax.plot(grid, j, 'k--', label='Neufeld 1990' if geometry == 'slab' else 'Dijkstra et al. 2006')
    ax.set_xlabel(r'$x / (a\tau_0)^{1/3}$')
    ax.set_ylabel('escaping spectrum')
    ax.set_title(f'{geometry}, a tau0 = {atau:g}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)


def band_edge(rep, dx, core):
    """the outer edge |x| of the node band from the report's nodes per cell (a uniform
    medium: every cell has the same band), in the grid's Doppler units"""
    per_cell = rep.get('nodes', 0)/max(rep.get('cells', 1), 1)
    return core + round((per_cell - 1.)/2.)*dx


def check_exact(args, geometry, atau, nz, dxf, label, iseed, nphot=None):
    """one lattice run against the asymptotic solution; returns (ok, record)"""
    temp = args.temp
    a = voigt_a(temp)
    tau0 = atau/a
    dx = dxf*atau**(1./3.)
    xmax = args.xmaxf*atau**(1./3.)
    name = f'{geometry}_{label}'
    wd = os.path.join(args.workdir, name)
    os.makedirs(wd, exist_ok=True)
    deck = os.path.join(wd, f'athinput.{name}')
    write_deck(deck, name, geometry, temp, tau0, nphot or args.nphot, iseed, True, dx, xmax,
               args.core, nz=nz, nx=args.nx, mcranks=args.mcranks, ddmc_tau=args.ddmc_tau)
    files, rep = run(args, wd, deck, name, args.mcranks)
    x, path, nsc = load(files, temp)
    # the sphere's medium is the cells whose centres lie inside R0, so the exact solution
    # is taken at the radius of a sphere of their volume
    atau_ref = atau
    if geometry == 'sphere':
        h = 2.5*R0/args.nx
        cx = -1.25*R0 + np.arange(args.nx)*h
        r2 = cx[:, None, None]**2 + cx[None, :, None]**2 + cx[None, None, :]**2
        reff = (3.*np.count_nonzero(r2 < R0**2)*h**3/(4.*np.pi))**(1./3.)
        atau_ref = atau*reff/R0
    ax = np.abs(x)
    med = np.median(ax)
    med_exact = MEDIAN[geometry]*atau_ref**(1./3.)
    dks = ks_exact(ax, lambda y: CDF[geometry](y, atau_ref))
    pk = peak(x, dx)
    ok_med = abs(med - med_exact) <= args.gate_median*dx
    ok_k = dks <= args.gate_k
    print(f"  a tau0 {atau:8.1e} tau0 {tau0:8.2e} nz {nz:3d} dx {dx:6.2f} xmax {xmax:6.1f}: "
          f"{rep.get('nodes', 0)} nodes in {rep.get('cells', 0)} cells, "
          f"{rep.get('dwells', 0)/x.size:.0f} dwells/photon, {rep.get('nscat', 0):.0f} "
          f"scatterings/photon, cpu {rep.get('cpu', 0)*1e3:.1f} ms/photon, node band to "
          f"|x| = {band_edge(rep, dx, args.core):.1f} "
          f"({band_edge(rep, dx, args.core)/((1.0664 if geometry == 'slab' else 0.92)*atau_ref**(1./3.)):.2f} of the peak)"
          + (f"; exact solution at a tau0 {atau_ref:.3e} (R_eff/R0 = {atau_ref/atau:.4f})"
             if geometry == 'sphere' else ''))
    print(f"      escapers {x.size}: median |x| {med:.2f} (exact {med_exact:.2f}, "
          f"{(med - med_exact)/dx:+.2f} bins) {'ok' if ok_med else 'OFF'}; peaks {pk[0]:.1f} "
          f"{pk[1]:.1f} (exact +-{(1.0664 if geometry == 'slab' else 0.92)*atau_ref**(1./3.):.1f}); "
          f"K distance {dks:.3f} (sampling scale {1./np.sqrt(x.size):.3f}) "
          f"{'ok' if ok_k else 'OFF'}")
    if args.plot:
        plot_spectrum(x, atau_ref, geometry, dx, os.path.join(wd, f'{name}.png'),
                      f'lattice nz {nz} dx {dx:.1f}')
    return ok_med and ok_k, dict(atau=atau, nz=nz, dx=dx, dks=dks, med=med,
                                 cpu=rep.get('cpu', 0.), dwells=rep.get('dwells', 0))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--path', default=os.path.join(HERE, '..', '..', '..'))
    ap.add_argument('--workdir', required=True)
    ap.add_argument('--checks', nargs='+', default=['slab', 'sphere', 'analog', 'scan'],
                    choices=['slab', 'sphere', 'analog', 'scan'])
    ap.add_argument('--atau', type=float, nargs='+', default=[1.e5, 1.e6, 1.e7])
    ap.add_argument('--temp', type=float, default=1.e4, help='gas temperature, K')
    ap.add_argument('--nz', type=int, default=32, help='cells across the slab')
    ap.add_argument('--nx', type=int, default=32, help='cells per side of the sphere cube')
    ap.add_argument('--dxf', type=float, default=0.21,
                    help='wing bin width in units of (a tau0)^(1/3)')
    ap.add_argument('--xmaxf', type=float, default=2.5,
                    help='grid extent in units of (a tau0)^(1/3)')
    ap.add_argument('--core', type=float, default=3.3, help='core bin half-width')
    ap.add_argument('--ddmc-tau', type=float, default=3.)
    ap.add_argument('--nphot', type=int, default=4000)
    ap.add_argument('--iseed', type=int, default=121500)
    ap.add_argument('--mcranks', type=int, default=1)
    ap.add_argument('--tau-analog', type=float, default=1.e6)
    ap.add_argument('--temp-analog', type=float, default=10.)
    ap.add_argument('--nphot-analog', type=int, default=1000)
    ap.add_argument('--nz-analog', type=int, nargs='+', default=[8, 16, 32],
                    help='cells across the slab of the lattice runs against the analog run')
    ap.add_argument('--atau-scan', type=float, default=1.e6)
    ap.add_argument('--nz-scan', type=int, nargs='+', default=[16, 32, 64])
    ap.add_argument('--gate-median', type=float, default=1.,
                    help='median |x| within this many bin widths of the exact value')
    ap.add_argument('--gate-k', type=float, default=0.1,
                    help='Kolmogorov distance from the exact distribution')
    ap.add_argument('--gate-analog', type=float, default=3.,
                    help='two-sample Kolmogorov distance, in units of its scale')
    ap.add_argument('--plot', action='store_true', help='write a png of each spectrum')
    return ap.parse_args(argv)


def main(args):
    os.makedirs(args.workdir, exist_ok=True)
    results = {}
    a = voigt_a(args.temp)
    print(f"Lyman alpha at {args.temp:g} K: a = {a:.3e}, Doppler width "
          f"{doppler_width(args.temp):.4e} Hz")

    for geometry in ('slab', 'sphere'):
        if geometry not in args.checks:
            continue
        print(f"{geometry}: the lattice against "
              f"{'Neufeld 1990' if geometry == 'slab' else 'Dijkstra et al. 2006'}")
        ok_all = True
        for i, atau in enumerate(args.atau):
            ok, _ = check_exact(args, geometry, atau, args.nz, args.dxf, f'at{atau:.0e}',
                                args.iseed + i)
            ok_all &= ok
        results[geometry] = ok_all

    if 'analog' in args.checks:
        temp = args.temp_analog
        atau = voigt_a(temp)*args.tau_analog
        dx = args.dxf*atau**(1./3.)
        xmax = args.xmaxf*atau**(1./3.)
        print(f"analog: the slab at tau0 {args.tau_analog:g}, {temp:g} K (a tau0 {atau:.3g}), "
              f"lattice against the analog kernel, {args.nphot_analog} photons each")
        # one analog run (exact at any resolution) against the lattice at each
        # --nz-analog, which moves the node band's edge through the spectrum
        runs = []
        for nz, ddmc in [(args.nz, False)] + [(nz, True) for nz in args.nz_analog]:
            name = f'analog_lattice{nz}' if ddmc else 'analog_analog'
            wd = os.path.join(args.workdir, name)
            os.makedirs(wd, exist_ok=True)
            deck = os.path.join(wd, f'athinput.{name}')
            write_deck(deck, name, 'slab', temp, args.tau_analog, args.nphot_analog,
                       args.iseed, ddmc, dx, xmax, args.core, nz=nz,
                       mcranks=args.mcranks, ddmc_tau=args.ddmc_tau)
            files, rep = run(args, wd, deck, name, args.mcranks)
            runs.append((name, nz, load(files, temp), rep))
            print(f"  {name:16s}: {rep.get('nscat', 0):.3g} scatterings/photon, cpu "
                  f"{rep.get('cpu', 0):.3g} s/photon, wall {rep.get('wall', 0):.0f} s"
                  + (f"; node band to |x| = {band_edge(rep, dx, args.core):.1f} "
                     f"({band_edge(rep, dx, args.core)/(1.0664*atau**(1./3.)):.2f} of the peak)"
                     if ddmc else ''))
        _, _, (xa, pa, _), _ = runs[0]
        ok = True
        for name, nz, (xl, pl, _), _ in runs[1:]:
            dkx, s = ks_two(xl, xa)
            dkp, _ = ks_two(pl, pa)
            ok_nz = (dkx <= args.gate_analog*s) and (dkp <= args.gate_analog*s)
            ok &= ok_nz
            print(f"  nz {nz:3d}: escapers {xl.size} vs {xa.size}: x dCDF {dkx:.4f} "
                  f"({dkx/s:.1f} x scale), path dCDF {dkp:.4f} ({dkp/s:.1f} x); median |x| "
                  f"{np.median(np.abs(xl)):.2f} vs {np.median(np.abs(xa)):.2f} (Neufeld "
                  f"{1.0516*atau**(1./3.):.2f}); mean path ratio {pl.mean()/pa.mean():.4f}  "
                  f"{'ok' if ok_nz else 'OFF'}")
        if args.plot:
            for name, nz, (x, _, _), _ in runs:
                plot_spectrum(x, atau, 'slab', dx, os.path.join(args.workdir, name,
                              f'{name}.png'), name)
        results['analog'] = ok

    if 'scan' in args.checks:
        print(f"scan: the slab at a tau0 {args.atau_scan:g} along the matching line "
              f"dx = (10 a tau0 / nz^2)^(1/3)")
        recs = []
        for i, nz in enumerate(args.nz_scan):
            dxf = (10./nz**2)**(1./3.)
            ok, rec = check_exact(args, 'slab', args.atau_scan, nz, dxf, f'scan{nz}',
                                  args.iseed + 10 + i)
            recs.append(rec)
        dks = [r['dks'] for r in recs]
        ok = all(dks[i+1] <= dks[i] + 2./np.sqrt(args.nphot) for i in range(len(dks)-1))
        print("  K distance by nz: " + ", ".join(f"{r['nz']}: {r['dks']:.3f} "
              f"({r['cpu']*1e3:.0f} ms/photon)" for r in recs)
              + f"  {'converging' if ok else 'NOT converging'}")
        results['scan'] = ok

    print()
    for name, ok in results.items():
        print(f"  {name:8s} {'PASS' if ok else 'FAIL'}")
    print("RESULT:", "PASS" if all(results.values()) else "FAIL")
    return 0 if all(results.values()) else 1


if __name__ == '__main__':
    sys.exit(main(parse_args()))
