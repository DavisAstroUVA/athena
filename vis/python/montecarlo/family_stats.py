#!/usr/bin/env python3
"""Effective sample counts of a photon list at the photon and at the family level.

    family_stats.py LIST [LIST ...] [--tcol 0] [--ecol 1] [--emin E --emax E]

A weight window splits samples into copies that share their history up to the split, so
the copies are correlated and (sum w)^2 / sum w^2 over photons overstates the effective
count.  Copies of one birth photon share its emission temperature and emitted energy, the
user columns the mc_readhdf* generators write (--tcol, --ecol), so grouping the escaping
photons by those two values recovers the families.  The family-level count is the one to
compare between runs; the scatter between the outputs of a nout > 1 run is the error that
needs no such bookkeeping at all.

Prints, for all photons and optionally for an escape-energy band in eV, the escaped
weight, the two effective counts and the share of the weight in the largest families.
"""
import argparse
import sys

import numpy as np

import athena_mc


def load(files):
    w, e, u = [], [], []
    for f in files:
        gen = athena_mc.read_list_generator(f)
        header = next(gen)['header']
        for c in gen:
            if c['chunk'] is None:
                continue
            d = dict(header)
            d['list'] = c['chunk']
            d['length'] = c['length']
            p = athena_mc.Photons(d)
            if p.nuser == 0:
                sys.exit(f"{f} carries no user columns; the generator must write the "
                         "emission temperature and energy")
            w.append(p.weight)
            e.append(p.energy)
            u.append(p.user.copy())
    return np.concatenate(w), np.concatenate(e), np.concatenate(u)


def family_index(u, tcol, ecol):
    return np.unique(athena_mc.family_key(u, tcol, ecol), return_inverse=True)[1]


def report(label, w, inv):
    fw = np.bincount(inv, weights=w)
    fw = fw[fw > 0]
    order = np.argsort(fw)[::-1]
    print(f"{label}: escaped weight {w.sum():.4e}, photons {w.size}, families {fw.size}")
    print(f"  effective samples: photon level {w.sum()**2/(w**2).sum():.0f}, "
          f"family level {fw.sum()**2/(fw**2).sum():.0f}")
    print(f"  share of weight in largest 1 / 10 / 100 families: "
          f"{fw[order[0]]/fw.sum():.4f} / {fw[order[:10]].sum()/fw.sum():.4f} / "
          f"{fw[order[:100]].sum()/fw.sum():.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('files', nargs='+')
    p.add_argument('--tcol', type=int, default=0, help='user column of the emission temperature')
    p.add_argument('--ecol', type=int, default=1, help='user column of the emitted energy')
    p.add_argument('--emin', type=float, help='escape-energy band lower edge [eV]')
    p.add_argument('--emax', type=float, help='escape-energy band upper edge [eV]')
    a = p.parse_args()
    w, e, u = load(a.files)
    inv = family_index(u, a.tcol, a.ecol)
    report('all photons', w, inv)
    if a.emin is not None and a.emax is not None:
        everg = 1.602176634e-12
        m = (e >= a.emin*everg) & (e < a.emax*everg)
        report(f'{a.emin:g}-{a.emax:g} eV', w[m], inv[m])


if __name__ == '__main__':
    main()
