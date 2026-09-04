#! /usr/bin/env python
"""
Generate athdf fixtures for the MCGridFile reader tests.

Writes one valid file per grid flavor plus a set of deliberately malformed files, each
exercising one of the checks in MCGridFile::ReadHeader/ValidateStructure/ValidateFaces.
The faces are computed with the same mesh generator Athena++ uses, so a correct reader
reproduces them exactly; the "usergen" fixture perturbs them to stand in for a source run
that used an enrolled mesh generator.

Usage: python make_fixtures.py <outdir> [--coord kerr-schild]

Requires h5py.
"""

import argparse
import os

import h5py
import numpy as np


def mesh_position(index, nrange, xmin, xmax, rat, nx):
    """Face position for global logical face `index` of `nrange`.

    Mirrors ComputeMeshGeneratorX() composed with Uniform/DefaultMeshGeneratorX* in
    src/mesh/mesh.hpp.
    """
    index = np.asarray(index, dtype=np.float64)
    if rat == 1.0:
        noffset = index - nrange // 2
        noffset_ceil = index - (nrange + 1) // 2
        x = (noffset + noffset_ceil) / (2.0 * nrange)
        return 0.5 * (xmin + xmax) + (x * xmax - x * xmin)
    x = index / float(nrange)
    ratn = rat ** nx
    rnx = rat ** (x * nx)
    lw = (rnx - ratn) / (1.0 - ratn)
    return xmin * lw + xmax * (1.0 - lw)


def root_level_of(nrbx):
    nbmax = max(nrbx)
    r = 0
    while (1 << r) < nbmax:
        r += 1
    return r


def build_tree(nrbx, refine=None):
    """Return (levels, locs) for a root grid, optionally refining one root block.

    `refine` is a root-block (lx1, lx2, lx3) to split into its 2^ndim children.
    """
    levels, locs = [], []
    for k in range(nrbx[2]):
        for j in range(nrbx[1]):
            for i in range(nrbx[0]):
                if refine is not None and (i, j, k) == tuple(refine):
                    for ck in range(2):
                        for cj in range(2):
                            for ci in range(2):
                                levels.append(1)
                                locs.append((2 * i + ci, 2 * j + cj, 2 * k + ck))
                else:
                    levels.append(0)
                    locs.append((i, j, k))
    return np.array(levels, dtype=np.int32), np.array(locs, dtype=np.int64)


def write(path, rgs, mbs, extent, levels, locs, coord="kerr-schild",
          mesh_data=True, face_perturb=0.0, int_data=False, drop_block=False):
    nrbx = [rgs[d] // mbs[d] for d in range(3)]
    rl = root_level_of(nrbx)

    if drop_block:
        levels, locs = levels[:-1], locs[:-1]

    nb = len(levels)
    with h5py.File(path, "w") as f:
        f.attrs["NumCycles"] = np.int32(0)
        f.attrs["Time"] = np.float32(0.0)
        f.attrs["MaxLevel"] = np.int32(levels.max() if nb else 0)
        f.attrs["NumMeshBlocks"] = np.int32(nb)
        f.attrs["DatasetNames"] = np.array([b"prim"], dtype="S16")
        f.attrs["NumVariables"] = np.array([1], dtype=np.int32)
        f.attrs["VariableNames"] = np.array([b"rho"], dtype="S16")
        if mesh_data:
            f.attrs["Coordinates"] = np.bytes_(coord)
            f.attrs["RootGridSize"] = np.array(rgs, dtype=np.int32)
            f.attrs["MeshBlockSize"] = np.array(mbs, dtype=np.int32)
            for d in range(3):
                f.attrs["RootGridX%d" % (d + 1)] = np.array(extent[d],
                                                            dtype=np.float32)

        f.create_dataset("Levels", data=levels.astype(">i4"))
        f.create_dataset("LogicalLocations", data=locs.astype(">i8"))

        if mesh_data:
            for d in range(3):
                xmin, xmax, rat = extent[d]
                nf = mbs[d] + 1
                arr = np.empty((nb, nf), dtype=np.float32)
                for b in range(nb):
                    nrbx_ll = nrbx[d] << int(levels[b])
                    idx = locs[b][d] * mbs[d] + np.arange(nf)
                    arr[b] = mesh_position(idx, nrbx_ll * mbs[d], xmin, xmax,
                                           rat, rgs[d])
                if face_perturb:
                    arr[:, 1:-1] *= (1.0 + face_perturb)
                f.create_dataset("x%df" % (d + 1), data=arr)
                cen = 0.5 * (arr[:, :-1] + arr[:, 1:])
                f.create_dataset("x%dv" % (d + 1), data=cen)

        shape = (1, nb, mbs[2], mbs[1], mbs[0])
        if int_data:
            f.create_dataset("prim", data=np.ones(shape, dtype=np.uint16))
        else:
            f.create_dataset("prim", data=np.ones(shape, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--coord", default="kerr-schild")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    def p(name):
        return os.path.join(args.outdir, name)

    rgs = [32, 32, 32]
    mbs = [8, 8, 8]
    nrbx = [4, 4, 4]
    uni = [(1.0, 5.0, 1.0), (0.0, 2.0, 1.0), (0.0, 4.0, 1.0)]
    rat = [(1.0, 100.0, 1.05), (0.0, 2.0, 1.0), (0.0, 4.0, 1.0)]

    flat_l, flat_o = build_tree(nrbx)
    amr_l, amr_o = build_tree(nrbx, refine=(1, 1, 1))

    # valid
    write(p("uniform_ok.athdf"), rgs, mbs, uni, flat_l, flat_o, args.coord)
    write(p("ratio_ok.athdf"), rgs, mbs, rat, flat_l, flat_o, args.coord)
    write(p("amr_ok.athdf"), rgs, mbs, uni, amr_l, amr_o, args.coord)

    # malformed, one broken invariant each
    write(p("bad_ghost.athdf"), rgs, [12, 12, 12], uni, flat_l, flat_o, args.coord)
    write(p("bad_slice.athdf"), rgs, [8, 8, 1], uni, flat_l, flat_o, args.coord)
    write(p("bad_nomesh.athdf"), rgs, mbs, uni, flat_l, flat_o, args.coord,
          mesh_data=False)
    # a genuinely different grid topology must be rejected ...
    write(p("bad_coord.athdf"), rgs, mbs, uni, flat_l, flat_o, "spherical_polar")
    # ... while a different name for the same topology must not be.  This is the
    # AthenaK case: such a snapshot is labelled "cartesian" even when the run was
    # general-relativistic, because the name records the grid, not the spacetime.
    write(p("coord_compatible.athdf"), rgs, mbs, uni, flat_l, flat_o, "minkowski")
    write(p("bad_tiling.athdf"), rgs, mbs, uni, flat_l, flat_o, args.coord,
          drop_block=True)
    write(p("bad_usergen.athdf"), rgs, mbs, uni, flat_l, flat_o, args.coord,
          face_perturb=1.0e-3)
    write(p("bad_intdata.athdf"), rgs, mbs, uni, flat_l, flat_o, args.coord,
          int_data=True)

    print("fixtures written to", args.outdir)


if __name__ == "__main__":
    main()
