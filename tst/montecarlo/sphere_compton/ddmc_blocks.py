#! /usr/bin/env python
"""
Blocks, ranks and refinement for the DDMC lattice (phase D3 of compton_ddmc_plan.md),
on the Thomson sphere of sphere_compton.py with the lattice on.  Three checks, each
PASS/FAIL:

  blocks      the sphere over 8 and 64 blocks on 1, 4 and 16 ranks: photon conservation
              (ntot = nesc + nabs + ndes + nrem) on every run, and the escape paths and
              energies of every run against the 8-block one-rank run within --gate times
              the two-sample Kolmogorov scale (statistical, since each block seeds its
              own generator and several ranks order the draws by arrival)
  repack      on one rank, plain against <loadbalancing> test_repack = local and mesh:
              byte-identical photon lists, which is the check that a walker survives the
              pack and unpack and that the lattice is rebuilt from the fluid on arrival
  refinement  a static refinement block inside the sphere: the lattice against the analog
              run on the same mesh within --gate-refine times the scale (a face to a
              block at another level is a boundary face of the lattice), with the
              report's level-jump counts printed

  python ddmc_blocks.py --path /path/to/athena --workdir DIR [--tau 40 --nphot 50000]
         [--mcranks 1 4 16] [--checks blocks repack refinement]

The binary is --prob=mc_sphere_isoth -mc -mpi.
"""
import argparse
import filecmp
import glob
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sphere_compton as sc  # noqa: E402

TEMPLATE = os.path.join(HERE, '..', '..', '..', 'inputs', 'mc', 'athinput.sphere_compton_grid')


def write_deck(args, workdir, mb, ddmc, repack=None, refine=False, mcranks=1):
    """the sphere_compton deck (Thomson, point source) with the lattice on or off, the
    repack knob, and a static refinement block inside the sphere"""
    # the mesh stays 32^3 (sphere_compton's --nx also sets the meshblock to half of it, so
    # the meshblock size is rewritten below)
    argv = ['--run', '--path', args.path, '--workdir', workdir, '--template', TEMPLATE,
            '--nx', '32', '--tau', str(args.tau), '--x0', '1', '--theta', '2e-3',
            '--nphot', str(args.nphot), '--iseed', str(args.iseed),
            '--scattering', 'thomson', '--mcranks', str(mcranks)]
    if ddmc:
        argv += ['--ddmc', '--ddmc-tau', str(args.ddmc_tau)]
    a = sc.parse_args(argv)
    os.makedirs(workdir, exist_ok=True)
    deck = os.path.join(workdir, 'athinput.sphere_compton')
    sc.write_athinput(a, deck)
    text = open(deck).read()
    text = re.sub(r'(<meshblock>\s*\n)(?:nx[123]\s*=\s*\d+\s*\n){3}',
                  lambda m: m.group(1) + f"nx1 = {mb}\nnx2 = {mb}\nnx3 = {mb}\n", text, count=1)
    if repack:
        text = text.replace('<problem>', f"<loadbalancing>\ntest_repack = {repack}\n\n<problem>", 1)
    if refine:
        # a cube of side 0.5 R in the octant x, y, z < 0, inside the sphere and off the
        # source at the centre, refined one level
        radius = float(re.search(r'^radius\s*=\s*([0-9.eE+-]+)', text, re.M).group(1))
        lo, hi = -0.6*radius, -0.1*radius
        text = text.replace('<mesh>', '<mesh>\nrefinement = static', 1)
        text = text.replace('<meshblock>', f"<refinement1>\nx1min = {lo:.6e}\nx1max = {hi:.6e}\n"
                            f"x2min = {lo:.6e}\nx2max = {hi:.6e}\nx3min = {lo:.6e}\n"
                            f"x3max = {hi:.6e}\nlevel = 1\n\n<meshblock>", 1)
    open(deck, 'w').write(text)
    return deck


def run(args, workdir, deck, mcranks):
    exe = os.path.join(args.path, 'bin', 'athena')
    cmd = [exe, '-i', deck, '-d', workdir]
    if mcranks > 1:
        cmd = ['mpirun', '-n', str(mcranks)] + cmd
    for f in glob.glob(os.path.join(workdir, 'sphcomp.*')):
        os.remove(f)
    with open(os.path.join(workdir, 'athena.log'), 'w') as log:
        ret = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
    files = sorted(glob.glob(os.path.join(workdir, 'sphcomp.out1.*.list')))
    if ret != 0 or not files:
        sys.exit(f"athena exited with {ret} and wrote {len(files)} list files; "
                 f"see {workdir}/athena.log")
    report = {}
    for line in open(os.path.join(workdir, 'athena.log')):
        if line.startswith('ntot:'):
            # the line ends in nscat/ntot, which must not be read as ntot
            for key, val in re.findall(r'(?<![\w/])(ntot|nesc|nabs|ndes|nrem): (\d+)', line):
                report[key] = int(val)
        if line.startswith('discrete diffusion:'):
            report['ddmc'] = line.strip()
        if line.startswith('wall time used'):
            report['wall'] = line.strip()
    return files, report


def conserved(report):
    n = report.get('nesc', 0) + report.get('nabs', 0) + report.get('ndes', 0) \
        + report.get('nrem', 0)
    return n == report.get('ntot', -1), n


def ks(a, b):
    """two-sample Kolmogorov distance and its scale sqrt((n + m) / n m)"""
    a = np.sort(a)
    b = np.sort(b)
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side='right')/a.size
    fb = np.searchsorted(b, grid, side='right')/b.size
    return np.max(np.abs(fa - fb)), np.sqrt((a.size + b.size)/(a.size*b.size))


def compare(files_a, files_b, label, gate):
    wa, ea, _, pa, _ = sc.load(files_a)
    wb, eb, _, pb, _ = sc.load(files_b)
    dp, s = ks(pa, pb)
    de, _ = ks(np.log(ea), np.log(eb))
    ok = (dp <= gate*s) and (de <= gate*s)
    print(f"    {label:38s} escapers {pa.size:7d} vs {pb.size:7d}; escape path dCDF "
          f"{dp:.4f} ({dp/s:.1f} x scale), ln energy dCDF {de:.4f} ({de/s:.1f} x); "
          f"mean path ratio {pa.mean()/pb.mean():.4f}  {'ok' if ok else 'OFF'}")
    return ok


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--path', default=os.path.join(HERE, '..', '..', '..'))
    ap.add_argument('--workdir', required=True)
    ap.add_argument('--tau', type=float, default=40.)
    ap.add_argument('--ddmc-tau', type=float, default=3.)
    ap.add_argument('--nphot', type=int, default=50000)
    ap.add_argument('--iseed', type=int, default=121500)
    ap.add_argument('--mcranks', type=int, nargs='+', default=[1, 4, 16])
    ap.add_argument('--blocks', type=int, nargs='+', default=[16, 8],
                    help='meshblock size on the 32^3 mesh: 16 is 8 blocks, 8 is 64')
    ap.add_argument('--checks', nargs='+', default=['blocks', 'repack', 'refinement'],
                    choices=['blocks', 'repack', 'refinement'])
    ap.add_argument('--gate', type=float, default=3.,
                    help='Kolmogorov distance gate, in units of the two-sample scale')
    ap.add_argument('--gate-refine', type=float, default=4.)
    return ap.parse_args(argv)


def main(args):
    os.makedirs(args.workdir, exist_ok=True)
    results = {}
    ref_files = None

    if 'blocks' in args.checks:
        print(f"blocks: the lattice over {' and '.join(str((32//m)**3) for m in args.blocks)} "
              f"blocks on {' '.join(map(str, args.mcranks))} ranks")
        ok_all = True
        for mb in args.blocks:
            for nr in args.mcranks:
                if nr > (32//mb)**3:
                    print(f"  {(32//mb)**3:3d} blocks, {nr:2d} ranks: skipped, fewer blocks than ranks")
                    continue
                wd = os.path.join(args.workdir, f'blocks_mb{mb}_r{nr}')
                deck = write_deck(args, wd, mb, ddmc=True, mcranks=nr)
                files, rep = run(args, wd, deck, nr)
                cons, n = conserved(rep)
                ok_all &= cons
                print(f"  {(32//mb)**3:3d} blocks, {nr:2d} ranks: ntot {rep.get('ntot')} = "
                      f"{n} {'conserved' if cons else 'NOT CONSERVED'}; "
                      f"{rep.get('ddmc', '')[:110]}")
                if ref_files is None:
                    ref_files = files
                else:
                    ok_all &= compare(files, ref_files,
                                      f"against {(32//args.blocks[0])**3} blocks, "
                                      f"{args.mcranks[0]} rank", args.gate)
        results['blocks'] = ok_all

    if 'repack' in args.checks:
        print("repack: one rank, plain against test_repack = local and mesh, byte-identical lists")
        mb = args.blocks[0]
        wd0 = os.path.join(args.workdir, 'repack_plain')
        deck = write_deck(args, wd0, mb, ddmc=True)
        files0, rep0 = run(args, wd0, deck, 1)
        ok_all = conserved(rep0)[0]
        for mode in ('local', 'mesh'):
            wd = os.path.join(args.workdir, f'repack_{mode}')
            deck = write_deck(args, wd, mb, ddmc=True, repack=mode)
            files, rep = run(args, wd, deck, 1)
            same = len(files) == len(files0) and all(
                filecmp.cmp(a, b, shallow=False) for a, b in zip(files, files0))
            ok_all &= same and conserved(rep)[0]
            print(f"  {mode:5s}: {len(files)} list file(s) "
                  f"{'identical' if same else 'DIFFER'} to the plain run; "
                  f"nesc {rep.get('nesc')} vs {rep0.get('nesc')}")
        results['repack'] = ok_all

    if 'refinement' in args.checks:
        print("refinement: a static level-1 block inside the sphere, lattice against analog")
        mb = args.blocks[0]
        # the refined mesh has 7 coarse blocks and 8 fine ones, so at most 15 ranks
        nr = max(r for r in args.mcranks if r <= 15)
        runs = {}
        for kind, ddmc in (('analog', False), ('lattice', True)):
            wd = os.path.join(args.workdir, f'refine_{kind}')
            deck = write_deck(args, wd, mb, ddmc=ddmc, refine=True, mcranks=nr)
            files, rep = run(args, wd, deck, nr)
            cons, n = conserved(rep)
            runs[kind] = (files, rep, cons)
            print(f"  {kind:7s}: ntot {rep.get('ntot')} = {n} "
                  f"{'conserved' if cons else 'NOT CONSERVED'}; {rep.get('wall', '')}")
            if rep.get('ddmc'):
                print(f"    {rep['ddmc']}")
        ok = runs['analog'][2] and runs['lattice'][2]
        ok &= compare(runs['lattice'][0], runs['analog'][0], "lattice against analog, refined mesh",
                      args.gate_refine)
        results['refinement'] = ok

    print()
    for name, ok in results.items():
        print(f"  {name:11s} {'PASS' if ok else 'FAIL'}")
    print("RESULT:", "PASS" if all(results.values()) else "FAIL")
    return 0 if all(results.values()) else 1


if __name__ == '__main__':
    sys.exit(main(parse_args()))
