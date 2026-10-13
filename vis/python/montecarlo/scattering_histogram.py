#!/usr/bin/env python3
"""
Distribution of the per-photon scattering count over a set of photon lists.

In a resonant-line problem the mean number of scatterings per photon says almost nothing
about where the time goes: the distribution is heavy-tailed, and a handful of photons that
happen to be emitted near line centre can carry essentially all of the work while the rest
escape without scattering at all.  This prints the distribution, how concentrated the work
is in the worst photons, and how much of the elapsed time those photons account for.

The scattering count is not a photon-list column of its own; a problem generator that wants
it copies Photon::nscp into one of its user variables in FinalizePhoton, and raises the
output's `nuser` to match.  mc_hotjupiter uses user variable 0 for the count and 1 for the
wall time between emission and termination in ms; mc_readhdf and mc_readhdf_gr use user
variable 2 for the count.  Point --nscat-col (and --time-col) at whichever slots a run
used.

Only escaped photons reach a list, so an absorbed or destroyed photon is missing here.  For
a run that reports nesc == ntot, which is common for a scattering-dominated resonance
line problem, that is the whole sample.

usage:
  scattering_histogram.py <list files...> [--nscat-col N] [--time-col N] [--png out.png]
"""
import argparse
import sys

import numpy as np

import athena_mc as am

# Lyman alpha, for the escape-offset table; matches MCConstants in montecarlo.hpp
C_CGS = 2.99792458e10
H_CGS = 6.62607015e-27
E_LYA = H_CGS * 1.0e8 * C_CGS / 1215.6701   # erg


def load(files, nscat_col, time_col):
    """Concatenate the scattering count, wall time and energy over every list file."""
    nscat, elapsed, energy = [], [], []
    for fn in files:
        phl = am.read_list(fn)
        if phl is None:
            raise SystemExit(f"could not read {fn}")
        ph = am.Photons(phl)
        if ph.nphot == 0:
            continue
        need = max(nscat_col, time_col if time_col is not None else 0) + 1
        if ph.nuser < need:
            raise SystemExit(
                f"{fn} carries {ph.nuser} user variable(s); column {need-1} was asked for."
                "\nCheck <output>/nuser and what the problem generator stores in"
                " FinalizePhoton.")
        nscat.append(ph.user[:, nscat_col])
        if time_col is not None:
            elapsed.append(ph.user[:, time_col])
        energy.append(ph.energy)
    if not nscat:
        raise SystemExit("every list was empty")
    return (np.concatenate(nscat),
            np.concatenate(elapsed) if elapsed else None,
            np.concatenate(energy))


def report(ns, el):
    n = ns.size
    tot = ns.sum()
    print(f"photons            : {n}")
    print(f"total scatterings  : {tot:.6g}")
    print(f"mean per photon    : {tot/n:.6g}")
    print(f"median             : {np.median(ns):.6g}")
    print("percentiles        : " + "  ".join(
        f"{q}%={np.percentile(ns, q):.4g}" for q in (0, 25, 50, 75, 90, 95, 99, 100)))
    print(f"photons that never scattered: {(ns == 0).sum()} "
          f"({100.0*(ns == 0).sum()/n:.1f}%)")

    if tot <= 0:
        return
    order = np.argsort(ns)[::-1]
    csum = np.cumsum(ns[order])
    print()
    print("concentration of the work:")
    for k in (1, 2, 5, 10, 20, 50, 100):
        if k <= n:
            print(f"  worst {k:4d} photons: {100*csum[k-1]/tot:6.2f}% of all scatterings")
    half = int(np.searchsorted(csum, 0.5*tot)) + 1
    print(f"  half the scatterings sit in the worst {half} photon(s), "
          f"{100.0*half/n:.3f}% of the sample")

    if el is not None and el.sum() > 0:
        elt = el.sum()
        ec = np.cumsum(el[np.argsort(el)[::-1]])
        print()
        print(f"per-photon wall time [ms]: median {np.median(el):.4g}, "
              f"max {el.max():.6g}")
        for k in (1, 2, 5, 10, 20, 50, 100):
            if k <= n:
                print(f"  worst {k:4d} photons: {100*ec[k-1]/elt:6.2f}% of the summed "
                      f"photon lifetimes")
        # Photon lifetimes overlap, so their sum is not the run's wall time.  The largest
        # single lifetime is a lower bound on how long the run had to last.
        print(f"  the longest-lived photon alone spans {el.max()/1000.0:.4g} s")


def report_line(ns, energy):
    """Escape offset from Lyman alpha line centre against how often a photon scattered."""
    dv = np.abs(C_CGS*(energy - E_LYA)/E_LYA/1.0e5)   # km/s
    print()
    print("escape offset from Lya line centre:")
    edges = (0, 10, 30, 60, 120, 400, np.inf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dv >= lo) & (dv < hi)
        if m.sum():
            print(f"  |dv| in [{lo:5g},{hi:6g}) km/s: {m.sum():7d} photons, "
                  f"mean scatterings {ns[m].mean():12.5g}, max {ns[m].max():12.5g}")


def plot(ns, el, png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pos = ns[ns > 0]
    if pos.size == 0:
        print("nothing scattered; no plot written")
        return
    bins = np.logspace(0, np.log10(ns.max())*1.0001 + 1e-6, 41)

    ncol = 2 if el is not None else 1
    fig, axes = plt.subplots(1, ncol, figsize=(5.6*ncol, 4.3))
    axes = np.atleast_1d(axes)

    axes[0].hist(np.clip(ns, 1.0, None), bins=bins, color='#4878a8',
                 edgecolor='k', lw=0.4)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('scatterings per photon  (0 shown in the first bin)')
    axes[0].set_ylabel('photons per bin')
    axes[0].set_title(f'N = {ns.size},  mean = {ns.mean():.3g},  max = {ns.max():.3g}')
    axes[0].axvline(max(ns.mean(), 1.0), color='r', ls='--', lw=1, label='mean')
    axes[0].legend(fontsize=8)

    if el is not None:
        axes[1].plot(np.clip(ns, 1.0, None), np.clip(el, 1e-1, None), '.', ms=3,
                     alpha=0.4)
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        axes[1].set_xlabel('scatterings per photon')
        axes[1].set_ylabel('emission to escape [ms]')
        axes[1].set_title('cost per photon')

    fig.tight_layout()
    fig.savefig(png, dpi=130)
    print(f"\nwrote {png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+', help='photon list files')
    ap.add_argument('--nscat-col', type=int, default=0,
                    help='user variable holding the scattering count (default 0)')
    ap.add_argument('--time-col', type=int, default=None,
                    help='user variable holding the per-photon wall time in ms')
    ap.add_argument('--no-line', action='store_true',
                    help='skip the Lyman alpha escape-offset table')
    ap.add_argument('--png', default=None, help='write the histogram here')
    args = ap.parse_args()

    ns, el, en = load(args.files, args.nscat_col, args.time_col)
    report(ns, el)
    if not args.no_line:
        report_line(ns, en)
    if args.png:
        plot(ns, el, args.png)


if __name__ == '__main__':
    sys.exit(main())
