#! /usr/bin/env python

"""
Band luminosities of a run's escaping photons, output by output, and the comparison of
two runs by the scatter between their outputs.

    python band_luminosity.py RUNDIR [RUNDIR2] [--bands 0.1 0.3 1 3 10] [--base xrb.out1]
                              [--screen NAME] [--nproc N]

Each output of a run is the set of list files <base>.proc*.<nnnnn>.list; the luminosity
in a band is sum(w E)/dt over the escapers in it, summed over ranks.  With one run it
prints the per-output table, the mean and the standard error of the mean from the
scatter between outputs.  With two it also prints the ratio run1/run2 per band with the
error from both scatters, which is the standard the biased-sampling guide used: two runs
of the same problem agree when their bands agree within their own output-to-output
scatter.

Leaving photons out with a screen function, as make_spectrum.py does: screen.py in the
working directory (or on PYTHONPATH) defines functions that take a Photons chunk and
return True for the photons to drop; see tst/montecarlo/disk_atmosphere/screen.py for
an example.  On a mesh whose inner radial boundary is an escape boundary, a screen on
the exit radius separates the light that reaches the observer from the light that
falls in:

    python band_luminosity.py twoband_1e9 oneband_1e9 --base mctde.out1 --screen inner
"""

import argparse
import functools
import glob
import multiprocessing
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import athena_mc  # noqa: E402
import mc_cli  # noqa: E402

KEV = 1.602176634e-9  # erg


def outputs(rundir, base):
    """{output number: [list files]}"""
    files = sorted(glob.glob(os.path.join(rundir, f"{base}.proc*.*.list")))
    out = {}
    for f in files:
        m = re.search(r"\.proc\d+\.(\d+)\.list$", f)
        if m:
            out.setdefault(int(m.group(1)), []).append(f)
    return out


def band_luminosities_file(f, edges, screen_function=None):
    """luminosity in each band and in total, erg/s, of one list file; edges in erg"""
    lum = np.zeros(len(edges)-1)
    total = 0.
    nphot = 0
    gen = athena_mc.read_list_generator(f)
    header = next(gen)['header']
    dt = header['dt']
    for c in gen:
        if c['chunk'] is None:
            continue
        p = athena_mc.Photons.from_chunk(header, c)
        keep = (np.ones(p.nphot, dtype=bool) if screen_function is None
                else ~np.asarray(screen_function(p), dtype=bool))
        we = (p.weight*p.energy)[keep]
        energy = p.energy[keep]
        total += we.sum()/dt
        nphot += len(we)
        idx = np.searchsorted(edges, energy) - 1
        ok = (idx >= 0) & (idx < len(lum))
        np.add.at(lum, idx[ok], we[ok]/dt)
        if c['done']:
            break
    return total, lum, nphot


def band_luminosities(files, edges_kev, screen_function=None, nproc=1):
    """luminosity in each band and in total, erg/s, summed over the files of one output;
    screen_function(phots) marks the photons to leave out; nproc > 1 reads the files in
    that many processes (the screen function comes from a module, so it can be sent to
    them; the pool is forked so that it need not be)"""
    edges = np.asarray(edges_kev)*KEV
    task = functools.partial(band_luminosities_file, edges=edges,
                             screen_function=screen_function)
    if nproc > 1 and len(files) > 1:
        with multiprocessing.get_context('fork').Pool(min(nproc, len(files))) as pool:
            parts = pool.map(task, files)
    else:
        parts = [task(f) for f in files]
    total = sum(part[0] for part in parts)
    lum = sum(part[1] for part in parts)
    nphot = sum(part[2] for part in parts)
    return total, lum, nphot


def run_table(rundir, base, edges, screen_function=None, nproc=1):
    outs = outputs(rundir, base)
    if not outs:
        sys.exit(f"no {base}.proc*.*.list in {rundir}")
    rows = []
    for n in sorted(outs):
        total, lum, nphot = band_luminosities(outs[n], edges, screen_function, nproc)
        rows.append((n, nphot, total, lum))
    return rows


def summarize(rows):
    tot = np.array([r[2] for r in rows])
    bands = np.array([r[3] for r in rows])
    m = len(rows)
    mean_t, mean_b = tot.mean(), bands.mean(axis=0)
    if m > 1:
        err_t = tot.std(ddof=1)/np.sqrt(m)
        err_b = bands.std(axis=0, ddof=1)/np.sqrt(m)
    else:
        err_t, err_b = np.nan, np.full(bands.shape[1], np.nan)
    return mean_t, err_t, mean_b, err_b


def print_run(name, rows, edges):
    labels = [f"{a:g}-{b:g}" for a, b in zip(edges[:-1], edges[1:])]
    print(f"\n{name}: luminosities in erg/s, bands in keV")
    print("  out  escapers   total      " + "  ".join(f"{l:>10s}" for l in labels))
    for n, nphot, total, lum in rows:
        print(f"  {n:3d} {nphot:9d} {total:10.4e}  " + "  ".join(f"{v:10.4e}" for v in lum))
    mt, et, mb, eb = summarize(rows)
    print(f"  mean           {mt:10.4e}  " + "  ".join(f"{v:10.4e}" for v in mb))
    if len(rows) > 1:
        print(f"  sem/mean       {et/mt:10.4f}  " + "  ".join(f"{e/v:10.4f}" for v, e in zip(mb, eb)))
    return mt, et, mb, eb


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', help='one or two run directories')
    ap.add_argument('--bands', type=float, nargs='+', default=[0.1, 0.3, 1., 3., 10.],
                    help='band edges in keV')
    ap.add_argument('--base', default='xrb.out1', help='list file base name')
    ap.add_argument('--nproc', type=int, default=1,
                    help='read the files of each output in this many processes')
    ap.add_argument('--screen',
                    help='name of a function in a user-supplied screen.py that takes '
                         'a Photons chunk and returns True for photons to leave out')
    return ap.parse_args(argv)


def main(args, screen_function=None):
    """the tables from the options of parse_args(); a caller such as a notebook can pass
    its own screen function instead of naming one in screen.py"""
    if screen_function is None:
        screen_function = mc_cli.load_screen(args.screen, 'band_luminosity.py')
    edges = args.bands
    results = []
    for r in args.runs[:2]:
        rows = run_table(r, args.base, edges, screen_function, args.nproc)
        results.append(print_run(r, rows, edges))
    if len(results) == 2:
        (mt1, et1, mb1, eb1), (mt2, et2, mb2, eb2) = results
        labels = ["total"] + [f"{a:g}-{b:g}" for a, b in zip(edges[:-1], edges[1:])]
        r = np.concatenate([[mt1/mt2], mb1/mb2])
        e = r*np.sqrt(np.concatenate([[(et1/mt1)**2 + (et2/mt2)**2],
                                      (eb1/mb1)**2 + (eb2/mb2)**2]))
        print(f"\nratio {args.runs[0]} / {args.runs[1]}, error from both runs' output scatter")
        for l, rr, ee in zip(labels, r, e):
            sig = (rr - 1.)/ee if ee > 0 else np.nan
            print(f"  {l:>10s}: {rr:.4f} +- {ee:.4f}  ({sig:+.1f} sigma from 1)")


if __name__ == '__main__':
    main(parse_args())
