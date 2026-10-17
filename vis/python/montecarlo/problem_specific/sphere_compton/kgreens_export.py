#! /usr/bin/env python

"""
Convert a Kompaneets Green's function quantile table (kgreens_table*.npz, written by
kompaneets_greens.py) to the flat binary the Monte Carlo code reads through
<montecarlo> kgreens_file, and read one back to check it.

    python kgreens_export.py kgreens_table_ff.npz kgreens_table_ff.bin
    python kgreens_export.py --check kgreens_table_ff.bin kgreens_table_ff.npz

Layout, all little-endian, no padding:

    magic    8 bytes   b"KGREENS1"
    dims     4 x int32 NL, NY, NX, NQ
    lam      NL float64
    y        NY float64
    xi       NX float64
    levels   NQ float64
    uq       NL*NY*NX*NQ float32, C order (level fastest)
    abar     NL*NY*NX float64, C order
    dead     NL*NY*NX uint8

The meaning of each array and the sampling algorithm are in
doc/monte_carlo/kompaneets/KGREENS_TABLE_FORMAT.md.  No HDF5, so that a sphere build
without -hdf5 can read it.
"""

import argparse
import struct
import sys

import numpy as np

MAGIC = b"KGREENS1"


def export(npz, out):
    t = np.load(npz)
    lam, y, xi, levels = (np.asarray(t[k], dtype='<f8') for k in ('lam', 'y', 'xi', 'levels'))
    uq = np.ascontiguousarray(t['uq'], dtype='<f4')
    abar = np.ascontiguousarray(t['abar'], dtype='<f8')
    dead = np.ascontiguousarray(t['dead'], dtype=np.uint8)
    nl, ny, nx, nq = uq.shape
    assert lam.shape == (nl,) and y.shape == (ny,) and xi.shape == (nx,)
    assert levels.shape == (nq,) and abar.shape == (nl, ny, nx) and dead.shape == (nl, ny, nx)
    with open(out, 'wb') as f:
        f.write(MAGIC)
        f.write(struct.pack('<4i', nl, ny, nx, nq))
        for a in (lam, y, xi, levels, uq, abar, dead):
            f.write(a.tobytes(order='C'))
    print(f"wrote {out}: NL={nl} NY={ny} NX={nx} NQ={nq}, "
          f"lam {lam[0]:g}..{lam[-1]:g}, y {y[0]:g}..{y[-1]:g}, xi {xi[0]:g}..{xi[-1]:g}")


def read(binfile):
    with open(binfile, 'rb') as f:
        if f.read(8) != MAGIC:
            sys.exit(f"{binfile}: not a KGREENS1 table")
        nl, ny, nx, nq = struct.unpack('<4i', f.read(16))
        lam = np.frombuffer(f.read(8*nl), '<f8')
        y = np.frombuffer(f.read(8*ny), '<f8')
        xi = np.frombuffer(f.read(8*nx), '<f8')
        levels = np.frombuffer(f.read(8*nq), '<f8')
        uq = np.frombuffer(f.read(4*nl*ny*nx*nq), '<f4').reshape(nl, ny, nx, nq)
        abar = np.frombuffer(f.read(8*nl*ny*nx), '<f8').reshape(nl, ny, nx)
        dead = np.frombuffer(f.read(nl*ny*nx), np.uint8).reshape(nl, ny, nx).astype(bool)
        if f.read(1):
            sys.exit(f"{binfile}: trailing bytes")
    return dict(lam=lam, y=y, xi=xi, levels=levels, uq=uq, abar=abar, dead=dead)


def check(binfile, npz):
    b = read(binfile)
    t = np.load(npz)
    ok = True
    for k in ('lam', 'y', 'xi', 'levels', 'uq', 'abar', 'dead'):
        same = np.array_equal(np.asarray(t[k], dtype=b[k].dtype), b[k])
        print(f"  {k:7s} {'same' if same else 'DIFFERENT'}")
        ok &= same
    print("binary matches the npz" if ok else "MISMATCH")
    return ok


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src', help='npz to export, or with --check the binary to read')
    ap.add_argument('dst', help='binary to write, or with --check the npz to compare to')
    ap.add_argument('--check', action='store_true')
    return ap.parse_args(argv)


def main(args):
    if args.check:
        sys.exit(0 if check(args.src, args.dst) else 1)
    export(args.src, args.dst)


if __name__ == '__main__':
    main(parse_args())
