#! /usr/bin/env python

"""
Compton scattering in a uniform isothermal sphere against the Kompaneets Green's function.

A point source at the centre of the sphere emits photons of energy x0 = E/kT, which
Compton scatter on the sphere's thermal electrons until they cross the sphere's surface
(src/pgen/mc_sphere_isoth.cpp, inputs/mc/athinput.sphere_compton).  Three comparisons are
made on the escaping photons, each independent of the others:

  separation   In bins of y = theta tau_path the energy distribution of the escapers must
               be the Green's function P_lam(xf, y | x0) of the Kompaneets equation and
               their mean weight its survival S(y).  This uses no escape-time theory; it
               tests the scattering kernel (and, with absorption = freefree, the free-free
               opacity) against the Fokker-Planck limit.  Photons that escape after only a
               few sphere radii have fewer scatterings than their path implies, so the
               lowest y bin is expected to be off by a few percent (Section 5 of the
               kompaneets handoff) and is reported but not gated.
  escape path  The CDF of tau_path at its own quantiles against the diffusion
               first-passage distributions, absorbing and mixed boundary.
  spectrum     The whole escaping spectrum and escaping fraction against the Green's
               function convolved with the mixed-boundary escape-path pdf.

Run on an existing run's lists, or let the script write the athinput file and run it:

  python sphere_compton.py --lists run/sphcomp.out1.*.list --athinput run/athinput
  python sphere_compton.py --run --path /path/to/athena --workdir DIR [--tau 15 --x0 1
         --theta 2e-3 --absorption freefree --nphot 100000]

The binary must be built with --prob=mc_sphere_isoth -mc.  Needs kompaneets_greens on
PYTHONPATH (vis/python/montecarlo) and the sphere_compton theory module next to it.
"""

import argparse
import glob
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VIS = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'vis', 'python', 'montecarlo'))
sys.path.insert(0, VIS)
sys.path.insert(0, os.path.join(VIS, 'problem_specific', 'sphere_compton'))

import athena_mc  # noqa: E402
import kompaneets_greens as kg  # noqa: E402
import sphere_theory as st  # noqa: E402


def read_athinput(filename):
    """flat dict of the <problem> and <montecarlo> keys of an athinput file"""
    out = {}
    block = None
    for line in open(filename):
        line = line.split('#')[0].strip()
        if not line:
            continue
        if line.startswith('<'):
            block = line.strip('<>')
            continue
        if block in ('problem', 'montecarlo', 'mesh') and '=' in line:
            k, v = [s.strip() for s in line.split('=', 1)]
            out[k] = v
    return out


def load(files):
    """weight, escape energy, birth energy, path, scattering count of all escapers"""
    w, e, e0, path, nsc, upath = [], [], [], [], [], []
    for f in files:
        gen = athena_mc.read_list_generator(f)
        header = next(gen)['header']
        for c in gen:
            if c['chunk'] is None:
                continue
            p = athena_mc.Photons(dict(header, list=c['chunk'], length=c['length']))
            if p.nuser < 3:
                sys.exit(f"{f}: the list needs the three user columns (nuser = 3 in the "
                         "phlist output block)")
            w.append(p.weight)
            e.append(p.energy)
            e0.append(p.user[:, 0])
            nsc.append(p.user[:, 1])
            # x0 is the path length (c = 1 in the pusher); user[2] is the hook's own sum
            path.append(p.x0)
            upath.append(p.user[:, 2])
            if c['done']:
                break
    path = np.concatenate(path)
    upath = np.concatenate(upath)
    if np.max(np.abs(upath - path)) > 1e-6*np.max(path):
        print(f"    note: the hook's path sum differs from x0 by up to "
              f"{np.max(np.abs(upath-path))/np.max(path):.2e} of the longest path")
    return (np.concatenate(w), np.concatenate(e), np.concatenate(e0), path,
            np.concatenate(nsc))


def compare(files, temp, tau, radius, x0, lam, nybin=6, nsbin=40, gate_cdf=0.02,
            gate_sigma=4., plot=None, verbose=True):
    thet = st.theta(temp)
    kT = st.K_BOLTZ*temp
    w, e, e0, path, nsc = load(files)
    n = len(w)
    xf = e/kT
    xi = e0/kT
    tau_path = path*tau/radius
    y = thet*tau_path
    ok = True
    print(f"\n=== x0={x0:g} tau={tau:g} theta={thet:.3g} lam={lam:.3g} "
          f"escapers={n:d} ===")
    print(f"    birth x: {xi.min():.4g} .. {xi.max():.4g}; "
          f"kappa_abs/kappa_es at x0: {thet*lam*kg.freefree_shape(x0):.2e}; "
          f"<nscat> {nsc.mean():.1f}, <tau_path> {tau_path.mean():.1f}")
    if abs(xi.mean()/x0-1.) > 1.e-6:
        print("    WARNING: birth energies do not match x0")
        ok = False

    # --- separation test, as mc_sphere_check.py
    s = np.log(xf)
    lo, hi = np.percentile(s, [0.05, 99.95])
    sedges = np.linspace(lo, hi, nsbin+1)
    m, surv, idx = st.greens_bins(x0, lam, y, sedges)
    ybins = np.quantile(y, np.linspace(0., 1., nybin+1))
    ybins[-1] *= 1.0001
    print("    separation test (per y bin: MC against the Green's function; the CDF gate "
          f"is max({gate_cdf:g}, 2.5/sqrt(N)))")
    print("    {:>17s} {:>7s} {:>11s} {:>11s} {:>8s} {:>9s} {:>8s} {:>6s}".format(
        "y bin", "N", "S_mc", "S_kg", "dS/err", "<x>mc/kg", "maxdCDF", "gate"))
    rows = []
    for b in range(nybin):
        sel = (y >= ybins[b]) & (y < ybins[b+1])
        nb = sel.sum()
        hmc, _ = np.histogram(s[sel], sedges, weights=w[sel])
        hkg = m[idx[sel]].sum(axis=0)
        smc = w[sel].sum()/nb
        skg = surv[idx[sel]].mean()
        err = w[sel].std()/np.sqrt(nb)
        dse = (smc-skg)/err if err > 0. else 0.
        cmc = np.cumsum(hmc)/max(hmc.sum(), 1.e-300)
        ckg = np.cumsum(hkg)/max(hkg.sum(), 1.e-300)
        sc = 0.5*(sedges[1:]+sedges[:-1])
        xmc = np.sum(hmc*np.exp(sc))/hmc.sum()
        xkg = np.sum(hkg*np.exp(sc))/hkg.sum()
        dcdf = np.max(np.abs(cmc-ckg))
        gated = b > 0
        # the CDF gate is the larger of the fixed tolerance and the one-sample
        # Kolmogorov scale of the bin, so small runs are judged by their statistics
        gate_here = max(gate_cdf, 2.5/np.sqrt(nb))
        passed = (dcdf < gate_here) and (abs(dse) < gate_sigma)
        if gated and not passed:
            ok = False
        print("    {:8.3g}-{:<8.3g} {:7d} {:11.4e} {:11.4e} {:+8.2f} {:9.4f} {:8.4f} "
              "{:>6s}".format(ybins[b], ybins[b+1], nb, smc, skg, dse, xmc/xkg, dcdf,
                              ("PASS" if passed else "FAIL") if gated else "early"))
        rows.append((ybins[b], ybins[b+1], sc, hmc/n, hkg/n))
    hmc, _ = np.histogram(s, sedges, weights=w)
    hkg = m[idx].sum(axis=0)
    dcdf_all = np.max(np.abs(np.cumsum(hmc)/hmc.sum()-np.cumsum(hkg)/hkg.sum()))
    print(f"    all y: S_mc={w.mean():.4e} S_kg={surv[idx].mean():.4e}  "
          f"max dCDF={dcdf_all:.4f}")

    # --- escape path
    q = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    tq = np.quantile(tau_path, q)
    esc = st.MixedBoundaryEscape(tau)
    cdf_mixed = esc.cdf(tq)
    cdf_abs = st.escape_cdf_absorbing(tq, tau)
    print("    escape path: CDF at the MC quantiles " + " ".join(f"{v:.2f}" for v in q))
    print("      mixed boundary    " + " ".join(f"{v:.3f}" for v in cdf_mixed))
    print("      absorbing boundary" + " ".join(f"{v:.3f}" for v in cdf_abs))
    print(f"      tau_path quantiles {' '.join(f'{v:.1f}' for v in tq)}; "
          f"escaped fraction by weight {w.sum()/n:.4e} (per injected photon, "
          "counting only escapers in the list)")

    # --- spectrum
    spec, frac = st.escape_spectrum(x0, tau, thet, lam, sedges)
    cmc = np.cumsum(hmc)/hmc.sum()
    cth = np.cumsum(spec)/spec.sum()
    dcdf_spec = np.max(np.abs(cmc-cth))
    xm = np.sum(hmc*np.exp(sc))/hmc.sum()
    xt = np.sum(spec*np.exp(sc))/spec.sum()
    print(f"    spectrum: max dCDF against the convolution {dcdf_spec:.4f}, "
          f"<x> MC/theory {xm/xt:.4f}, escaping weight per escaper MC {w.mean():.4e} "
          f"theory {frac:.4e}")

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        ax = axes[0]
        for k, (y0, y1, sc, a, b) in enumerate(rows):
            col = f"C{k:d}"
            ax.step(np.exp(sc), a/np.diff(sedges), where="mid", color=col,
                    label=f"y {y0:.2g}-{y1:.2g}")
            ax.plot(np.exp(sc), b/np.diff(sedges), "--", color=col)
        ax.set_xscale("log")
        ax.set_yscale("log")
        top = max(np.max(r[3]/np.diff(sedges)) for r in rows)
        ax.set_ylim(1.e-5*top, 2.*top)
        ax.set_xlabel("x_f")
        ax.set_ylabel("dN/dln x_f per escaper (solid MC, dashed Green's function)")
        ax.set_title(f"x0={x0:g} tau={tau:g} theta={thet:.2g} lam={lam:.2g}")
        ax.legend(fontsize=7)
        ax = axes[1]
        ax.step(np.exp(sc), hmc/n/np.diff(sedges), where="mid", color="k", label="MC")
        ax.plot(np.exp(sc), spec/np.diff(sedges)*(w.sum()/n)/frac, "--", color="C3",
                label="convolution (scaled to MC escape)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("x_f")
        ax.set_ylabel("escaping dN/dln x_f per photon")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plot)
        print("    plot:", plot)
    print("    RESULT:", "PASS" if ok else "FAIL")
    return ok


def compare_runs(files_a, files_b, temp, tau, radius, label_a='this run',
                 label_b='reference', nbin=40):
    """
    The escapers of two runs of the same problem against each other: max CDF difference
    of the escaping spectrum in ln x, of the escape path, mean energy ratio and escaping
    weight ratio, with the statistical error of each from the two sample sizes.
    """
    kT = st.K_BOLTZ*temp
    wa, ea, _, pa, _ = load(files_a)
    wb, eb, _, pb, _ = load(files_b)
    print(f"\n    {label_a} against {label_b}: {len(wa)} and {len(wb)} escapers")
    for name, xa, xb in (("spectrum (ln x)", np.log(ea/kT), np.log(eb/kT)),
                         ("escape path", pa*tau/radius, pb*tau/radius)):
        lo = min(xa.min(), xb.min())
        hi = max(xa.max(), xb.max())
        edges = np.linspace(lo, hi, 400)
        ca = np.cumsum(np.histogram(xa, edges, weights=wa)[0])
        cb = np.cumsum(np.histogram(xb, edges, weights=wb)[0])
        ca /= ca[-1]
        cb /= cb[-1]
        d = np.max(np.abs(ca - cb))
        # Kolmogorov scale for two samples of these sizes (weights near equal)
        na = wa.sum()**2/np.sum(wa**2)
        nb = wb.sum()**2/np.sum(wb**2)
        sig = np.sqrt((na + nb)/(na*nb))
        qa = np.quantile(xa, [0.1, 0.5, 0.9])
        qb = np.quantile(xb, [0.1, 0.5, 0.9])
        print(f"      {name:16s} max dCDF {d:.4f} ({d/sig:.1f} x the two-sample "
              f"Kolmogorov scale {sig:.4f}); quantiles 0.1/0.5/0.9 "
              f"{qa[0]:.4g} {qa[1]:.4g} {qa[2]:.4g} vs {qb[0]:.4g} {qb[1]:.4g} {qb[2]:.4g}")
    ma = np.sum(wa*ea)/wa.sum()
    mb = np.sum(wb*eb)/wb.sum()
    print(f"      mean escaping energy ratio {ma/mb:.4f}; escaping weight per photon "
          f"ratio {wa.sum()/len(wa)/(wb.sum()/len(wb)):.4f} (counting escapers only)")


def write_athinput(args, filename):
    template = args.template or os.path.join(HERE, '..', '..', '..', 'inputs', 'mc',
                                             'athinput.sphere_compton')
    temp = args.theta*st.M_ELECTRON*st.C_LIGHT**2/st.K_BOLTZ
    subs = {'nphot': str(args.nphot), 'iseed': str(args.iseed),
            'absorption': args.absorption, 'temp': f"{temp:.6e}", 'tau': str(args.tau),
            'x0': str(args.x0), 'radius': str(args.radius), 'scattering': args.scattering}
    # the mesh extents follow the radius; with --nx the grid is nx^3 over 2.5 radius,
    # offset half a cell so the origin is a cell centre, in blocks of nx/2
    scale = args.radius/float(read_athinput(template)['radius'])
    if args.nx:
        h = 2.5*args.radius/args.nx
        lo, hi = -1.25*args.radius - 0.5*h, 1.25*args.radius - 0.5*h
        for d in '123':
            subs[f'nx{d}'] = str(args.nx)
            subs[f'x{d}min'] = f"{lo:.7e}"
            subs[f'x{d}max'] = f"{hi:.7e}"
    out = []
    block = None
    for line in open(template):
        key = line.split('=')[0].strip() if '=' in line else None
        if line.startswith('<'):
            block = line.strip().strip('<>')
        if key in ('x1min', 'x1max', 'x2min', 'x2max', 'x3min', 'x3max') and scale != 1. \
                and key not in subs:
            subs[key] = f"{float(line.split('=')[1].split('#')[0])*scale:.7e}"
        if block == 'meshblock' and args.nx and key in ('nx1', 'nx2', 'nx3'):
            line = f"{key} = {max(args.nx//2, 1)}\n"
        elif key in subs and not (block == 'meshblock'):
            comment = line.split('#', 1)[1].rstrip() if '#' in line else None
            line = f"{key:<10s} = {subs[key]}"
            line += f"    # {comment.strip()}\n" if comment else "\n"
        if block == 'montecarlo' and line.startswith('polarized') and args.accel:
            line += (f"acceleration = true\naccel_tau = {args.accel_tau}\n"
                     f"accel_pmax = {args.accel_pmax}\n")
            if args.kgreens:
                line += f"kgreens_file = {args.kgreens}\n"
        if block == 'montecarlo' and line.startswith('polarized') and args.beta:
            line += "boosts = true\n"
        if block == 'montecarlo' and line.startswith('polarized') and args.general_pusher:
            # the integrator's step is a fraction of the cell crossing; the fixed default
            # affine step is far too small for a cell of 1e10 cm
            line += "general_pusher = true\nvarystep = true\nstepsize = 0.02\n"
        if block == 'montecarlo' and line.startswith('polarized') and args.accel_report:
            line += "accel_report = true\n"
        if block == 'montecarlo' and line.startswith('polarized') and args.accel \
                and args.accel_domain != 'cell':
            line += f"accel_domain = {args.accel_domain}\n"
        if block == 'montecarlo' and line.startswith('polarized') and args.accel:
            line += f"accel_face_tau = {args.accel_face_tau}\n"
        if block == 'problem' and line.startswith('heabund') and args.beta:
            line += f"velocity = {args.beta}    # fluid velocity along z, in units of c\n"
        out.append(line)
    with open(filename, 'w') as f:
        f.writelines(out)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lists', nargs='*', help='photon list files of an existing run')
    ap.add_argument('--athinput', help='athinput file of that run (parameters)')
    ap.add_argument('--run', action='store_true', help='write the athinput file and run')
    ap.add_argument('--path', default=os.path.join(HERE, '..', '..', '..'),
                    help='athena distribution with bin/athena')
    ap.add_argument('--workdir', default='.', help='where to run')
    ap.add_argument('--template', default=None,
                    help='athinput file to take the deck from (default '
                         'inputs/mc/athinput.sphere_compton; the _grid one spreads the '
                         'sphere over 32^3 cells)')
    ap.add_argument('--nx', type=int, default=0,
                    help='cells per side over 2.5 radius (0: as the template)')
    ap.add_argument('--accel', action='store_true', help='turn the random walk on')
    ap.add_argument('--accel-tau', type=float, default=20.)
    ap.add_argument('--accel-pmax', type=float, default=0.)
    ap.add_argument('--accel-face-tau', type=float, default=5.)
    ap.add_argument('--accel-domain', default='cell', choices=['cell', 'sphere'])
    ap.add_argument('--accel-report', action='store_true',
                    help='print the scatterings-by-cell-depth histogram')
    ap.add_argument('--kgreens', default=None, help='Kompaneets table binary')
    ap.add_argument('--general-pusher', action='store_true',
                    help='integrate with the general pusher (flat spacetime here)')
    ap.add_argument('--beta', type=float, default=0.,
                    help='uniform fluid velocity along z in units of c, with boosts on')
    ap.add_argument('--compare-to', default=None,
                    help='another run directory: compare the escapers of the two runs '
                         'directly (spectrum, escape path, weight), for a moving medium '
                         'or MRW against analog')
    ap.add_argument('--mcranks', type=int, default=1)
    ap.add_argument('--nphot', type=int, default=100000)
    ap.add_argument('--iseed', type=int, default=121500)
    ap.add_argument('--tau', type=float, default=15.)
    ap.add_argument('--x0', type=float, default=1.)
    ap.add_argument('--theta', type=float, default=2.e-3)
    ap.add_argument('--radius', type=float, default=1.e10)
    ap.add_argument('--absorption', default='none', choices=['none', 'freefree'])
    ap.add_argument('--scattering', default='compton', choices=['compton', 'thomson'],
                    help='thomson has no Green\'s function comparison; use --compare-to')
    ap.add_argument('--nybin', type=int, default=6)
    ap.add_argument('--nsbin', type=int, default=40)
    ap.add_argument('--gate-cdf', type=float, default=0.02)
    ap.add_argument('--gate-sigma', type=float, default=4.)
    ap.add_argument('--plot', default=None)
    return ap.parse_args(argv)


def main(args):
    if args.run:
        os.makedirs(args.workdir, exist_ok=True)
        deck = os.path.join(args.workdir, 'athinput.sphere_compton')
        write_athinput(args, deck)
        exe = os.path.join(args.path, 'bin', 'athena')
        cmd = [exe, '-i', deck, '-d', args.workdir]
        if args.mcranks > 1:
            cmd = ['mpirun', '-n', str(args.mcranks)] + cmd
        print("running:", " ".join(cmd))
        with open(os.path.join(args.workdir, 'athena.log'), 'w') as log:
            ret = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        if ret != 0:
            sys.exit(f"athena exited with {ret}; see {args.workdir}/athena.log")
        files = sorted(glob.glob(os.path.join(args.workdir, 'sphcomp.out1.*.list')))
        params = read_athinput(deck)
        for line in open(os.path.join(args.workdir, 'athena.log')):
            if line.startswith(('random walk', 'photons transported', 'wall time used',
                                'ntot:', 'scatterings by cell')):
                print('   ', line.rstrip())
    else:
        if not args.lists or not args.athinput:
            sys.exit("give --lists and --athinput, or --run")
        files = args.lists
        params = read_athinput(args.athinput)
    temp = float(params['temp'])
    tau = float(params['tau'])
    radius = float(params['radius'])
    x0 = float(params['x0'])
    heabund = float(params.get('heabund', 0.09))
    lam = 0.
    if params.get('absorption', 'none') == 'freefree':
        lam = st.lam_freefree(tau, radius, temp, heabund)
    no_greens = (float(params.get('velocity', 0.)) != 0.
                 or params.get('scattering', 'compton') != 'compton')
    if no_greens and not args.compare_to:
        sys.exit("a moving medium or Thomson scattering has no Green's function "
                 "comparison; give --compare-to")
    ok = True
    if not no_greens:
        ok = compare(files, temp, tau, radius, x0, lam, nybin=args.nybin,
                     nsbin=args.nsbin, gate_cdf=args.gate_cdf, gate_sigma=args.gate_sigma,
                     plot=args.plot)
    if args.compare_to:
        ref = sorted(glob.glob(os.path.join(args.compare_to, 'sphcomp.out1.*.list')))
        compare_runs(files, ref, temp, tau, radius, label_b=args.compare_to)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main(parse_args())
