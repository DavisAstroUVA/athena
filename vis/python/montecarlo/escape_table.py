#! /usr/bin/env python
"""
Per-cell escape probabilities for weights = biased, integrated through the whole mesh.

    escape_table.py SNAPSHOT OUTFILE [options]

Reads the athdf snapshot a mc_readhdf_gr run starts from and the run's input deck,
builds the effective extinction sqrt(3 alpha_a (alpha_a + alpha_s)) of every cell in a
few energy bands, integrates it from each cell to the domain boundary along the six axis
directions through every block in the way, and writes exp(-tau) of the smallest column,
floored, as one variable per band in an athdf file with the snapshot's block structure.
The problem generator reads that file through <problem>/escape_file and uses it in
place of the block-local column it would otherwise integrate.

The block-local column stops at the cell's own block face, so a cell in the face layer
of a block buried deep in the disk looks like a surface cell.  This tool integrates past
the face, which is why it exists.  It reports the emission-weighted mean importance both
ways so the difference can be seen before a run is spent.

Physics follows the problem generator: composition heabund, free-free absorption with
Gaunt factor 1, Thomson scattering, temperature tgas_cgs p/rho clipped to the floor and
ceiling, cells above tcut or below dcut or inside the horizon treated as empty.  The
emitting cell's own depth enters as the skin fraction (1 - e^-tau)/tau.

Bands are log-spaced between --emin and --emax (eV).  The problem generator maps each of
its escape-table groups to the band containing the group's centre, and clamps outside.

Run it on the file the deck names in <montecarlo>/grid_from_file, not on a dump the
Monte Carlo run wrote: the output keeps the input's block order, and the problem
generator addresses it with the grid file's block index map.  An AthenaK dump (dens,
eint) is converted with the deck's <hydro>/gamma, as the problem generator does.

Memory: only the through-columns of every block face are held across blocks (a few
hundred kilobytes per block); each block's cells are finished and written in turn.
"""
import argparse
import sys

import h5py
import numpy as np

C = 2.99792458e10
H = 6.62607015e-27
KB = 1.380649e-16
MP = 1.67262192369e-24
SIGMA_T = 6.6524587e-25
FFNRM = 3.692146e8
EVERG = 1.602176634e-12
DIRS = ['-x', '+x', '-y', '+y', '-z', '+z']


def read_deck(path):
    """<section> key = value pairs, values as strings."""
    deck = {}
    section = None
    with open(path) as f:
        for line in f:
            line = line.split('#')[0].strip()
            if not line:
                continue
            if line.startswith('<'):
                section = line.strip('<>').strip()
                deck.setdefault(section, {})
            elif '=' in line and section is not None:
                key, val = line.split('=', 1)
                deck[section][key.strip()] = val.strip()
    return deck


def get(deck, section, key, default=None, cast=float):
    val = deck.get(section, {}).get(key)
    if val is None:
        if default is None:
            sys.exit('deck lacks <%s>/%s' % (section, key))
        return default
    if cast is bool:
        return val.lower() == 'true'
    return cast(val)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('snapshot')
    p.add_argument('deck')
    p.add_argument('outfile')
    p.add_argument('--nbands', type=int, default=8, help='energy bands (default 8)')
    p.add_argument('--emin', type=float, default=10., help='lowest band edge, eV')
    p.add_argument('--emax', type=float, default=1.e5, help='highest band edge, eV')
    p.add_argument('--pesc-min', type=float, default=1.e-10,
                   help='floor on the escape probability')
    return p.parse_args(argv)


def horizon_mask(xc, yc, zc, spin, mass):
    """True where the cell centre lies inside the Kerr-Schild horizon."""
    z, y, x = np.meshgrid(zc, yc, xc, indexing='ij')
    rr2 = x * x + y * y + z * z
    a2 = spin * spin
    r2 = 0.5 * (rr2 - a2 + np.sqrt((rr2 - a2)**2 + 4. * a2 * z * z))
    return np.sqrt(r2) < mass + np.sqrt(mass * mass - a2)


class Snapshot:
    """Block-by-block access to what the tables need."""

    def __init__(self, path, prm, ecen):
        self.f = h5py.File(path, 'r')
        f = self.f
        self.prm = prm
        self.ecen = ecen
        names = [n.decode() for n in f.attrs['VariableNames']]
        dsets = [n.decode() for n in f.attrs['DatasetNames']]
        where = {}
        k = 0
        for d, n in zip(dsets, f.attrs['NumVariables']):
            for i in range(n):
                where[names[k]] = (d, i)
                k += 1
        self.irho = where.get('rho', where.get('dens'))
        self.ipress = where.get('press')
        self.ieint = where.get('eint')
        if self.irho is None or (self.ipress is None and self.ieint is None):
            sys.exit('snapshot must hold rho or dens, and press or eint')
        self.x1f = f['x1f'][:].astype(np.float64)
        self.x2f = f['x2f'][:].astype(np.float64)
        self.x3f = f['x3f'][:].astype(np.float64)
        self.nblk = self.x1f.shape[0]
        self.nx = (self.x3f.shape[1] - 1, self.x2f.shape[1] - 1, self.x1f.shape[1] - 1)

    def block(self, b):
        """Effective extinction per code length (nb, nz, ny, nx), photon number emission
        rate per cell, and the band shares of that emission."""
        prm = self.prm
        rho_code = self.f[self.irho[0]][self.irho[1], b].astype(np.float64)
        if self.ipress is not None:
            press = self.f[self.ipress[0]][self.ipress[1], b].astype(np.float64)
        else:
            press = (prm['gamma'] - 1.) * self.f[self.ieint[0]][self.ieint[1], b].astype(np.float64)
        rho = rho_code * prm['rho_cgs']
        with np.errstate(divide='ignore', invalid='ignore'):
            temp = np.where(rho_code > 0., prm['tgas_cgs'] * press / rho_code, prm['tfloor'])
        temp = np.clip(temp, prm['tfloor'], prm['tceiling'])
        xc = 0.5 * (self.x1f[b, 1:] + self.x1f[b, :-1])
        yc = 0.5 * (self.x2f[b, 1:] + self.x2f[b, :-1])
        zc = 0.5 * (self.x3f[b, 1:] + self.x3f[b, :-1])
        empty = ((rho_code < prm['dcut']) | (temp > prm['tcut'])
                 | horizon_mask(xc, yc, zc, prm['spin'], prm['mass']))
        rho = np.where(empty, 1.e-30, rho)
        y = prm['heabund']
        nh = rho / (MP * (1. + 4. * y))
        ne = nh * (1. + 2. * y)
        nz2 = nh * (1. + 4. * y)
        ecen = self.ecen
        nu = ecen[:, None, None, None] / H
        x = np.minimum(ecen[:, None, None, None] / (KB * temp[None]), 700.)
        alpha_a = (ne * nz2 / np.sqrt(temp))[None] * FFNRM / nu**3 * (-np.expm1(-x))
        alpha_s = SIGMA_T * ne
        kap = (np.sqrt(3. * alpha_a * (alpha_a + alpha_s[None])) * prm['l_cgs']).astype(np.float32)

        dx = np.diff(self.x1f[b]); dy = np.diff(self.x2f[b]); dz = np.diff(self.x3f[b])
        vol = (dz[:, None, None] * dy[None, :, None] * dx[None, None, :]) * prm['l_cgs']**3
        emis = np.where(empty, 0., 1.032521e-11 / np.sqrt(temp) * ne * nz2 * vol)
        lo = np.log(prm['emin'] * (KB * temp if prm['tnorm'] else EVERG))
        hi = np.log(prm['emax'] * (KB * temp if prm['tnorm'] else EVERG))
        wband = np.empty((len(ecen),) + temp.shape, dtype=np.float32)
        for ib in range(len(ecen)):
            d = np.clip(np.minimum(np.log(self.edges[ib + 1]), hi)
                        - np.maximum(np.log(self.edges[ib]), lo), 0., None)
            wband[ib] = d * np.exp(-np.minimum(ecen[ib] / (KB * temp), 700.))
        return kap, (dz, dy, dx), emis, wband


def block_sums(kap, widths):
    """Depth from each cell to each of the six block faces, the cell's own depth
    entering as the skin term, and the total through-column on each face."""
    dz, dy, dx = widths
    out = {}
    tot = {}
    for name, w, axis in (('z', dz[None, :, None, None], 1), ('y', dy[None, None, :, None], 2),
                          ('x', dx[None, None, None, :], 3)):
        own = kap * w
        with np.errstate(divide='ignore', invalid='ignore'):
            skin = np.where(own > 1.e-6, -np.log(-np.expm1(-own) / own), 0.)
        cum = np.cumsum(own, axis=axis)
        out['-' + name] = (cum - own) + skin
        total = np.take(cum, -1, axis=axis)
        out['+' + name] = (np.expand_dims(total, axis) - cum) + skin
        tot[name] = total
    return out, tot


def face_columns(snap, through, b, nb):
    """Column beyond each face of block b, from every block past that face whose
    footprint covers the line.  Blocks tile space, so the sum is the full column."""
    fc = [snap.x1f, snap.x2f, snap.x3f]
    lo_b = snap.lo_b
    hi_b = snap.hi_b
    tol = 1.e-6
    result = {}
    for ax, name in ((0, 'x'), (1, 'y'), (2, 'z')):
        other = [o for o in range(3) if o != ax]
        cc = [0.5 * (fc[o][b, 1:] + fc[o][b, :-1]) for o in other]
        for sign, dname in ((-1, '-' + name), (+1, '+' + name)):
            if sign < 0:
                cand = np.where(hi_b[:, ax] <= lo_b[b, ax] + tol)[0]
            else:
                cand = np.where(lo_b[:, ax] >= hi_b[b, ax] - tol)[0]
            cand = cand[(hi_b[cand][:, other[0]] > lo_b[b, other[0]] + tol)
                        & (lo_b[cand][:, other[0]] < hi_b[b, other[0]] - tol)
                        & (hi_b[cand][:, other[1]] > lo_b[b, other[1]] + tol)
                        & (lo_b[cand][:, other[1]] < hi_b[b, other[1]] - tol)]
            col = np.zeros((nb, len(cc[0]), len(cc[1])))
            for a in cand:
                idx = []
                ok = np.ones((len(cc[0]), len(cc[1])), dtype=bool)
                for m, o in enumerate(other):
                    fa = fc[o][a]
                    ia = np.searchsorted(fa, cc[m]) - 1
                    inside = (ia >= 0) & (ia < len(fa) - 1)
                    idx.append(np.clip(ia, 0, len(fa) - 2))
                    ok &= inside[:, None] if m == 0 else inside[None, :]
                # through[name][a] is stored (transverse slow, transverse fast) in
                # (z, y, x) order, which for every axis is (other[1], other[0])
                samp = through[name][a][:, idx[1][None, :], idx[0][:, None]]
                col += np.where(ok[None], samp, 0.)
            # broadcast the face column to every cell along ax
            if name == 'x':      # col (nb, y, z) -> (nb, z, y, 1)
                result[dname] = np.transpose(col, (0, 2, 1))[:, :, :, None]
            elif name == 'y':    # col (nb, x, z) -> (nb, z, 1, x)
                result[dname] = np.transpose(col, (0, 2, 1))[:, :, None, :]
            else:                # col (nb, x, y) -> (nb, 1, y, x)
                result[dname] = np.transpose(col, (0, 2, 1))[:, None, :, :]
    return result


def main(argv=None):
    args = parse_args(argv)
    deck = read_deck(args.deck)
    prm = {
        'rho_cgs': get(deck, 'problem', 'rho_cgs'),
        'tgas_cgs': get(deck, 'problem', 'tgas_cgs'),
        'l_cgs': get(deck, 'problem', 'l_cgs'),
        'heabund': get(deck, 'problem', 'heabund', 0.09),
        'tfloor': max(get(deck, 'problem', 'tfloor_cgs', 0.), 1.),
        'tceiling': get(deck, 'problem', 'tceiling_cgs', 1.e300),
        'dcut': get(deck, 'problem', 'dcut', 1.e-20),
        'tcut': get(deck, 'problem', 'tcut', 1.e20),
        'tnorm': get(deck, 'problem', 'tnorm', False, bool),
        'emin': get(deck, 'problem', 'emin'),
        'emax': get(deck, 'problem', 'emax'),
        'spin': get(deck, 'coord', 'a', 0.),
        'mass': get(deck, 'coord', 'm', 1.),
        'gamma': get(deck, 'hydro', 'gamma', 5. / 3.),
    }
    nb = args.nbands
    edges = np.logspace(np.log10(args.emin), np.log10(args.emax), nb + 1) * EVERG
    ecen = np.sqrt(edges[1:] * edges[:-1])
    snap = Snapshot(args.snapshot, prm, ecen)
    snap.edges = edges
    snap.lo_b = np.stack([snap.x1f[:, 0], snap.x2f[:, 0], snap.x3f[:, 0]], axis=1)
    snap.hi_b = np.stack([snap.x1f[:, -1], snap.x2f[:, -1], snap.x3f[:, -1]], axis=1)
    nblk = snap.nblk
    nx3, nx2, nx1 = snap.nx

    # Pass 1: the through-column of every block on each of its faces.  This is all that
    # is kept across blocks.
    print('pass 1: through-columns of %d blocks, %d bands' % (nblk, nb))
    through = {'z': np.zeros((nblk, nb, nx2, nx1), dtype=np.float32),
               'y': np.zeros((nblk, nb, nx3, nx1), dtype=np.float32),
               'x': np.zeros((nblk, nb, nx3, nx2), dtype=np.float32)}
    for b in range(nblk):
        kap, widths, _, _ = snap.block(b)
        _, tot = block_sums(kap, widths)
        for a in 'xyz':
            through[a][b] = tot[a]
        if b % 500 == 0:
            print('  block %d' % b)

    # Pass 2: each block's cells, finished and written in turn
    print('pass 2: cells, written block by block')
    out = h5py.File(args.outfile, 'w')
    for key, val in snap.f.attrs.items():
        if key in ('DatasetNames', 'NumVariables', 'VariableNames'):
            continue
        out.attrs[key] = val
    out.attrs['DatasetNames'] = np.array([b'pesc'], dtype='S16')
    out.attrs['NumVariables'] = np.array([nb], dtype=np.int32)
    out.attrs['VariableNames'] = np.array([('pesc%d' % i).encode() for i in range(nb)],
                                          dtype='S16')
    for name in ('Levels', 'LogicalLocations', 'x1f', 'x1v', 'x2f', 'x2v', 'x3f', 'x3v'):
        out.create_dataset(name, data=snap.f[name][:])
    dset = out.create_dataset('pesc', shape=(nb, nblk, nx3, nx2, nx1), dtype=np.float32,
                              chunks=(nb, 1, nx3, nx2, nx1))
    out.create_dataset('pesc_edges', data=edges)

    etot = 0.
    s_local = s_global = 0.
    share = {'local': np.zeros(3), 'global': np.zeros(3)}
    for b in range(nblk):
        kap, widths, emis, wband = snap.block(b)
        within, _ = block_sums(kap, widths)
        beyond = face_columns(snap, through, b, nb)
        tau_local = np.minimum.reduce([within[d] for d in DIRS])
        tau_global = np.minimum.reduce([within[d] + beyond[d] for d in DIRS])
        p_local = np.maximum(np.exp(-tau_local), args.pesc_min)
        p_global = np.maximum(np.exp(-tau_global), args.pesc_min).astype(np.float32)
        dset[:, b] = p_global
        wsum = wband.sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            imp_l = np.where(wsum > 0., (wband * p_local).sum(axis=0) / wsum, 1.)
            imp_g = np.where(wsum > 0., (wband * p_global).sum(axis=0) / wsum, 1.)
        etot += emis.sum()
        s_local += (emis * imp_l).sum()
        s_global += (emis * imp_g).sum()
        for label, imp in (('local', imp_l), ('global', imp_g)):
            s = emis * imp
            share[label] += [s[imp > 0.1].sum(), s[imp > 0.01].sum(), s[imp < 1.e-6].sum()]
        if b % 500 == 0:
            print('  block %d' % b)
    out.close()

    print('emission-weighted mean importance: block-local %.3e, global columns %.3e'
          % (s_local / etot, s_global / etot))
    for label, stot in (('local', s_local), ('global', s_global)):
        sh = share[label] / stot
        print('  %s: share of samples from cells with importance > 0.1: %.3f, > 0.01: %.3f,'
              ' < 1e-6: %.3f' % (label, sh[0], sh[1], sh[2]))
    print('wrote %s: %d bands, edges %s eV'
          % (args.outfile, nb, np.array2string(edges / EVERG, precision=3)))


if __name__ == '__main__':
    main()
