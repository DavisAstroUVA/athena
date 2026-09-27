#! /usr/bin/env python
"""
Per-cell escape probabilities for weights = biased, integrated through the whole mesh.

    escape_table.py SNAPSHOT DECK OUTFILE [options]

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


def cell_opacities(rho_code, press_code, prm, ecen):
    """alpha_a per band and alpha_s in 1/cm for one block; empty cells give zero."""
    rho = rho_code * prm['rho_cgs']
    with np.errstate(divide='ignore', invalid='ignore'):
        temp = np.where(rho_code > 0., prm['tgas_cgs'] * press_code / rho_code,
                        prm['tfloor'])
    temp = np.clip(temp, prm['tfloor'], prm['tceiling'])
    empty = (rho_code < prm['dcut']) | (temp > prm['tcut'])
    rho = np.where(empty, 1.e-30, rho)
    y = prm['heabund']
    nh = rho / (MP * (1. + 4. * y))
    ne = nh * (1. + 2. * y)
    nz2 = nh * (1. + 4. * y)
    nu = ecen[:, None, None, None] / H
    x = np.minimum(ecen[:, None, None, None] / (KB * temp[None]), 700.)
    alpha_a = (ne * nz2 / np.sqrt(temp))[None] * FFNRM / nu**3 * (-np.expm1(-x))
    return alpha_a, SIGMA_T * ne, temp, ne, nz2


def horizon_mask(xc, yc, zc, spin, mass):
    """True where the cell centre lies inside the Kerr-Schild horizon."""
    z, y, x = np.meshgrid(zc, yc, xc, indexing='ij')
    rr2 = x * x + y * y + z * z
    a2 = spin * spin
    r2 = 0.5 * (rr2 - a2 + np.sqrt((rr2 - a2)**2 + 4. * a2 * z * z))
    return np.sqrt(r2) < mass + np.sqrt(mass * mass - a2)


def block_sums(kap, dz, dy, dx):
    """Cumulative depth from each cell outward to each of the six block faces, the cell's
    own depth entering as the skin term; and the total through-column on each face.
    kap has shape (nb, nz, ny, nx); dz, dy, dx are 1-D cell widths."""
    nb = kap.shape[0]
    own_z = kap * dz[None, :, None, None]
    own_y = kap * dy[None, None, :, None]
    own_x = kap * dx[None, None, None, :]
    out = {}
    tot = {}
    for name, own, axis in (('z', own_z, 1), ('y', own_y, 2), ('x', own_x, 3)):
        with np.errstate(divide='ignore', invalid='ignore'):
            skin = np.where(own > 1.e-6, -np.log(-np.expm1(-own) / own), 0.)
        cum = np.cumsum(own, axis=axis)              # includes the cell itself
        # toward the low face: cells below this one, i.e. cum - own
        out['-' + name] = (cum - own) + skin
        # toward the high face: cells above, i.e. total - cum
        total = np.take(cum, -1, axis=axis)
        out['+' + name] = (np.expand_dims(total, axis) - cum) + skin
        tot[name] = total                            # (nb, face grid)
    return out, tot


def main(argv=None):
    args = parse_args(argv)
    deck = read_deck(args.deck)
    prm = {
        'rho_cgs': get(deck, 'problem', 'rho_cgs'),
        'tgas_cgs': get(deck, 'problem', 'tgas_cgs'),
        'l_cgs': get(deck, 'problem', 'l_cgs'),
        'heabund': get(deck, 'problem', 'heabund', 0.09),
        'tfloor': get(deck, 'problem', 'tfloor_cgs', 0.),
        'tceiling': get(deck, 'problem', 'tceiling_cgs', 1.e300),
        'dcut': get(deck, 'problem', 'dcut', 1.e-20),
        'tcut': get(deck, 'problem', 'tcut', 1.e20),
        'tnorm': get(deck, 'problem', 'tnorm', False, bool),
        'emin': get(deck, 'problem', 'emin'),
        'emax': get(deck, 'problem', 'emax'),
        'spin': get(deck, 'coord', 'a', 0.),
        'mass': get(deck, 'coord', 'm', 1.),
    }
    prm['tfloor'] = max(prm['tfloor'], 1.)

    f = h5py.File(args.snapshot, 'r')
    names = [n.decode() for n in f.attrs['VariableNames']]
    dsets = [n.decode() for n in f.attrs['DatasetNames']]
    nvar = list(f.attrs['NumVariables'])
    # locate rho and press by name across datasets
    where = {}
    k = 0
    for d, n in zip(dsets, nvar):
        for i in range(n):
            where[names[k]] = (d, i)
            k += 1
    irho = where.get('rho', where.get('dens'))
    ipress = where.get('press')
    ieint = where.get('eint')
    if irho is None or (ipress is None and ieint is None):
        sys.exit('snapshot must hold rho or dens, and press or eint')
    # an AthenaK dump holds the internal energy density; the problem generator converts
    # it with the deck's gamma, and so does this
    gamma = get(deck, 'hydro', 'gamma', 5. / 3.)
    x1f = f['x1f'][:].astype(np.float64)
    x2f = f['x2f'][:].astype(np.float64)
    x3f = f['x3f'][:].astype(np.float64)
    nblk = x1f.shape[0]
    nx1, nx2, nx3 = x1f.shape[1] - 1, x2f.shape[1] - 1, x3f.shape[1] - 1

    edges = np.logspace(np.log10(args.emin), np.log10(args.emax), args.nbands + 1) * EVERG
    ecen = np.sqrt(edges[1:] * edges[:-1])
    nb = args.nbands

    # Pass 1: per block, the extinction, the six cumulative depths, and the through
    # columns.  The cumulative depths are kept (nb x cells x 6 floats) to finish pass 2.
    print('pass 1: %d blocks, %d bands' % (nblk, nb))
    within = np.zeros((6, nb, nblk, nx3, nx2, nx1), dtype=np.float32)
    through = {a: np.zeros((nb, nblk) + shape, dtype=np.float64)
               for a, shape in (('z', (nx2, nx1)), ('y', (nx3, nx1)), ('x', (nx3, nx2)))}
    emis = np.zeros((nblk, nx3, nx2, nx1), dtype=np.float64)   # photon number rate
    wband = np.zeros((nb, nblk, nx3, nx2, nx1), dtype=np.float32)  # band share
    dirs = ['-x', '+x', '-y', '+y', '-z', '+z']
    for b in range(nblk):
        rho = f[irho[0]][irho[1], b].astype(np.float64)
        if ipress is not None:
            press = f[ipress[0]][ipress[1], b].astype(np.float64)
        else:
            press = (gamma - 1.) * f[ieint[0]][ieint[1], b].astype(np.float64)
        alpha_a, alpha_s, temp, ne, nz2 = cell_opacities(rho, press, prm, ecen)
        xc = 0.5 * (x1f[b, 1:] + x1f[b, :-1])
        yc = 0.5 * (x2f[b, 1:] + x2f[b, :-1])
        zc = 0.5 * (x3f[b, 1:] + x3f[b, :-1])
        hole = horizon_mask(xc, yc, zc, prm['spin'], prm['mass'])
        alpha_a[:, hole] = 0.
        alpha_s[hole] = 0.
        kap = np.sqrt(3. * alpha_a * (alpha_a + alpha_s[None])) * prm['l_cgs']
        dx = np.diff(x1f[b]); dy = np.diff(x2f[b]); dz = np.diff(x3f[b])
        out, tot = block_sums(kap, dz, dy, dx)
        for n, d in enumerate(dirs):
            within[n, :, b] = out[d]
        for a in 'xyz':
            through[a][:, b] = tot[a]
        vol = np.outer(np.outer(dz, dy).ravel(), dx).reshape(nx3, nx2, nx1) * prm['l_cgs']**3
        emis[b] = np.where(hole, 0., 1.032521e-11 / np.sqrt(temp) * ne * nz2 * vol)
        # share of the cell's free-free number emission per band, over the sampler's range
        lo = np.log(prm['emin'] * (KB * temp if prm['tnorm'] else EVERG))
        hi = np.log(prm['emax'] * (KB * temp if prm['tnorm'] else EVERG))
        for ib in range(nb):
            d = np.clip(np.minimum(np.log(edges[ib + 1]), hi) - np.maximum(np.log(edges[ib]), lo), 0., None)
            wband[ib, b] = d * np.exp(-np.minimum(ecen[ib] / (KB * temp), 700.))
        if b % 50 == 0:
            print('  block %d' % b)

    # Pass 2: the column beyond each face from every block lying past it.  Blocks tile
    # space, so summing the through-columns of all blocks past the face whose footprint
    # covers a cell's line gives the whole column to the domain boundary.
    print('pass 2: columns past the block faces')
    lo_b = np.stack([x1f[:, 0], x2f[:, 0], x3f[:, 0]], axis=1)
    hi_b = np.stack([x1f[:, -1], x2f[:, -1], x3f[:, -1]], axis=1)
    faces = {1: (x1f, x2f, x3f), 2: (x2f, x1f, x3f), 3: (x3f, x1f, x2f)}
    outside = np.zeros((6, nb, nblk, nx3, nx2, nx1), dtype=np.float32)
    tol = 1.e-6
    for b in range(nblk):
        for ax, name in ((0, 'x'), (1, 'y'), (2, 'z')):
            other = [o for o in range(3) if o != ax]
            # cell-centre coordinates of this block in the two transverse directions
            fc = [x1f, x2f, x3f]
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
                # transverse grid of this block's face: (len(cc[0]), len(cc[1]))
                col = np.zeros((nb, len(cc[0]), len(cc[1])))
                for a in cand:
                    idx = []
                    ok = np.ones((len(cc[0]), len(cc[1])), dtype=bool)
                    for m, o in enumerate(other):
                        fa = fc[o][a]
                        ia = np.searchsorted(fa, cc[m]) - 1
                        inside = (ia >= 0) & (ia < len(fa) - 1)
                        ia = np.clip(ia, 0, len(fa) - 2)
                        idx.append(ia)
                        ok &= inside[:, None] if m == 0 else inside[None, :]
                    # through[name][:, a] is indexed (transverse slow, transverse fast)
                    # in the order the other axes appear in (z, y, x) storage
                    t = through[name][:, a]
                    if name == 'x':      # stored (z, y); other = [1 (y), 2 (z)]
                        samp = t[:, idx[1][None, :], idx[0][:, None]]
                    elif name == 'y':    # stored (z, x); other = [0 (x), 2 (z)]
                        samp = t[:, idx[1][None, :], idx[0][:, None]]
                    else:                # stored (y, x); other = [0 (x), 1 (y)]
                        samp = t[:, idx[1][None, :], idx[0][:, None]]
                    col += np.where(ok[None], samp, 0.)
                # broadcast the face column to every cell of the block along ax
                n = dirs.index(dname)
                if name == 'x':      # col is (nb, y, z) -> (nb, z, y, 1)
                    outside[n, :, b] = np.transpose(col, (0, 2, 1))[:, :, :, None]
                elif name == 'y':    # col is (nb, x, z) -> (nb, z, 1, x)
                    outside[n, :, b] = np.transpose(col, (0, 2, 1))[:, :, None, :]
                else:                # col is (nb, x, y) -> (nb, 1, y, x)
                    outside[n, :, b] = np.transpose(col, (0, 2, 1))[:, None, :, :]
        if b % 50 == 0:
            print('  block %d' % b)

    # Escape probability, block-local and global, and the emission-weighted importance
    tau_local = within.min(axis=0)
    tau_global = (within + outside).min(axis=0)
    del within, outside
    p_local = np.maximum(np.exp(-tau_local), args.pesc_min).astype(np.float32)
    p_global = np.maximum(np.exp(-tau_global), args.pesc_min).astype(np.float32)
    wsum = wband.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        imp_local = np.where(wsum > 0., (wband * p_local).sum(axis=0) / wsum, 1.)
        imp_global = np.where(wsum > 0., (wband * p_global).sum(axis=0) / wsum, 1.)
    etot = emis.sum()
    print('emission-weighted mean importance: block-local %.3e, global columns %.3e'
          % ((emis * imp_local).sum() / etot, (emis * imp_global).sum() / etot))
    for label, imp in (('block-local', imp_local), ('global', imp_global)):
        s = emis * imp
        stot = s.sum()
        print('  %s: share of samples from cells with importance > 0.1: %.3f, > 0.01: %.3f,'
              ' < 1e-6: %.3f' % (label, s[imp > 0.1].sum() / stot, s[imp > 0.01].sum() / stot,
                                 s[imp < 1.e-6].sum() / stot))

    # Write the escape file with the snapshot's block structure
    with h5py.File(args.outfile, 'w') as g:
        for key, val in f.attrs.items():
            if key in ('DatasetNames', 'NumVariables', 'VariableNames'):
                continue
            g.attrs[key] = val
        g.attrs['DatasetNames'] = np.array([b'pesc'], dtype='S16')
        g.attrs['NumVariables'] = np.array([nb], dtype=np.int32)
        g.attrs['VariableNames'] = np.array([('pesc%d' % i).encode() for i in range(nb)],
                                            dtype='S16')
        for name in ('Levels', 'LogicalLocations', 'x1f', 'x1v', 'x2f', 'x2v', 'x3f', 'x3v'):
            g.create_dataset(name, data=f[name][:])
        g.create_dataset('pesc', data=p_global)
        g.create_dataset('pesc_edges', data=edges)
    print('wrote %s: %d bands, edges %s eV' % (args.outfile, nb, np.array2string(edges / EVERG, precision=3)))


if __name__ == '__main__':
    main()
