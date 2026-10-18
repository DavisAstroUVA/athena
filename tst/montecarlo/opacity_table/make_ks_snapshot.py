#! /usr/bin/env python
"""
A small synthetic snapshot for mc_readhdf_gr: a Gaussian torus around a Kerr black hole in
Cartesian Kerr-Schild coordinates, written in the athdf layout MCReadSnapshotBlock reads
(attributes DatasetNames, VariableNames, NumVariables; one dataset 'prim' of shape
(5, nblocks, nx3, nx2, nx1) holding rho, press, vel1, vel2, vel3 in code units).  It lets
the GR reader's table path be exercised on a laptop; the production snapshot lives on the
cluster.

The mesh is nb blocks per direction of nx cells each on [-half, half]^3, and the blocks are
written in the order Athena++ assigns gids to a uniform root grid with two blocks per
direction (x fastest, then y, then z), which is why nb must be 1 or 2.  The matching
athinput is written by opacity_test_util.write_readhdf_gr_athinput.

Density: rho = rho0 exp(-(R - R0)^2 / (2 w^2) - z^2 / (2 h^2)) with a floor, zero inside
2 M; temperature T = T0 (rho/rho0)^0.3 clamped to [T_min, T_max] so that the (T, rho)
plane of the opacity table is swept; pressure from T through tgas_cgs; velocity a slow
azimuthal flow, v_phi = v0 sqrt(R0/R) in the normal frame.

  python make_ks_snapshot.py OUT.athdf [--nx 16 --nb 2 --half 24]
"""
import argparse

import h5py
import numpy as np

# code-to-cgs conversions the athinput must repeat
RHO_CGS = 1.e-3      # g/cm^3 per code density unit
TGAS_CGS = 1.e8      # K per unit of p/rho
L_CGS = 1.4766e6     # cm, GM/c^2 of a 10 solar mass black hole


def fields(x, y, z, args):
    R = np.sqrt(x*x + y*y)
    r = np.sqrt(x*x + y*y + z*z)
    rho = args.rho0 * np.exp(-0.5*((R - args.r0)/args.width)**2 - 0.5*(z/args.height)**2)
    rho = np.maximum(rho, args.rho0*args.floor)
    rho = np.where(r < 2.0, args.rho0*args.floor, rho)
    temp = args.t0 * (rho/args.rho0)**0.3
    temp = np.clip(temp, args.tmin, args.tmax)
    press = rho * temp / TGAS_CGS
    vphi = args.v0 * np.sqrt(args.r0/np.maximum(R, 1.0))
    vphi = np.where(R < 2.0, 0., vphi)
    vel1 = -vphi * y / np.maximum(R, 1.e-10)
    vel2 = vphi * x / np.maximum(R, 1.e-10)
    vel3 = np.zeros_like(rho)
    return rho, press, vel1, vel2, vel3


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out')
    ap.add_argument('--nx', type=int, default=16, help='cells per block per direction')
    ap.add_argument('--nb', type=int, default=2, choices=[1, 2],
                    help='blocks per direction')
    ap.add_argument('--half', type=float, default=24., help='half-width of the box [M]')
    ap.add_argument('--rho0', type=float, default=1.e-3, help='peak density [code]')
    ap.add_argument('--floor', type=float, default=1.e-6, help='floor, fraction of rho0')
    ap.add_argument('--r0', type=float, default=10., help='torus radius [M]')
    ap.add_argument('--width', type=float, default=4., help='radial width [M]')
    ap.add_argument('--height', type=float, default=3., help='scale height [M]')
    ap.add_argument('--t0', type=float, default=1.e7, help='peak temperature [K]')
    ap.add_argument('--tmin', type=float, default=1.e4)
    ap.add_argument('--tmax', type=float, default=3.e7)
    ap.add_argument('--v0', type=float, default=0.2, help='azimuthal speed at r0 [c]')
    args = ap.parse_args()

    nx, nb = args.nx, args.nb
    ntot = nx*nb
    edges = np.linspace(-args.half, args.half, ntot+1)
    centers = 0.5*(edges[1:] + edges[:-1])
    nblocks = nb**3
    prim = np.zeros((5, nblocks, nx, nx, nx))
    x1f = np.zeros((nblocks, nx+1))
    x2f = np.zeros((nblocks, nx+1))
    x3f = np.zeros((nblocks, nx+1))
    for kb in range(nb):
        for jb in range(nb):
            for ib in range(nb):
                gid = ib + nb*(jb + nb*kb)
                xs = centers[ib*nx:(ib+1)*nx]
                ys = centers[jb*nx:(jb+1)*nx]
                zs = centers[kb*nx:(kb+1)*nx]
                z, y, x = np.meshgrid(zs, ys, xs, indexing='ij')
                for n, f in enumerate(fields(x, y, z, args)):
                    prim[n, gid] = f
                x1f[gid] = edges[ib*nx:(ib+1)*nx+1]
                x2f[gid] = edges[jb*nx:(jb+1)*nx+1]
                x3f[gid] = edges[kb*nx:(kb+1)*nx+1]

    with h5py.File(args.out, 'w') as f:
        f.attrs['DatasetNames'] = np.array([b'prim'], dtype='S21')
        f.attrs['VariableNames'] = np.array([b'rho', b'press', b'vel1', b'vel2', b'vel3'],
                                            dtype='S21')
        f.attrs['NumVariables'] = np.array([5], dtype=np.int32)
        f.attrs['NumMeshBlocks'] = np.int32(nblocks)
        f.attrs['MeshBlockSize'] = np.array([nx, nx, nx], dtype=np.int32)
        f.attrs['RootGridSize'] = np.array([ntot, ntot, ntot], dtype=np.int32)
        f.attrs['RootGridX1'] = np.array([-args.half, args.half, 1.0])
        f.attrs['RootGridX2'] = np.array([-args.half, args.half, 1.0])
        f.attrs['RootGridX3'] = np.array([-args.half, args.half, 1.0])
        f.attrs['MaxLevel'] = np.int32(0)
        f.attrs['Coordinates'] = np.bytes_(b'gr_user')
        f.create_dataset('prim', data=prim)
        f.create_dataset('x1f', data=x1f)
        f.create_dataset('x2f', data=x2f)
        f.create_dataset('x3f', data=x3f)
        locs = np.array([[ib, jb, kb] for kb in range(nb) for jb in range(nb)
                         for ib in range(nb)], dtype=np.int64)
        f.create_dataset('LogicalLocations', data=locs)
        f.create_dataset('Levels', data=np.zeros(nblocks, dtype=np.int32))
    rho = prim[0]*RHO_CGS
    temp = prim[1]/prim[0]*TGAS_CGS
    print('{}: {} blocks of {}^3; rho {:.2e} to {:.2e} g/cm^3, T {:.2e} to {:.2e} K'.format(
        args.out, nblocks, nx, rho.min(), rho.max(), temp.min(), temp.max()))


if __name__ == '__main__':
    main()
