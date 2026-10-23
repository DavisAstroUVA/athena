#! /usr/bin/env python
"""
Emitted luminosity of a snapshot's cells per table group, no transport, under the two
readings of an opacity table (`table_groups = mean` and `edge`), summed over the mesh
with each cell's volume.  It says how much of a difference between two transport runs
is already in the emission and in which groups.

   python snapshot_emission.py SNAPSHOT TABLE --rho-cgs 1e-8 --tgas-cgs 1.633e6
       --l-cgs 8.86e11 [--tfloor-cgs 8e3] [--bands 0.001 0.01 0.1 1 10]

The temperature is tgas_cgs times pressure over density as the readers take it, floored
at tfloor_cgs; cells off the table's grid are clamped to its edge (the readers put
free-free there, so a mesh with many of them is not for this script).  Under the mean
reading group g emits 4 pi kappa_g rho times the Planck integral over the group; under
the edge reading value k sits at the group's lower bound, the emission is the trapezoid
in ln nu of B_nu kappa / (h nu) between nodes, and an energy is log-uniform in the node
interval.  Needs tst/montecarlo/opacity_table on the path for table_tools.
"""
import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '../../../../../tst/montecarlo/opacity_table'))
from table_tools import Table, H, C, KB, KEV  # noqa: E402


def planck_nu(nu, temp):
    """B_nu on (nu, cell)"""
    x = H*nu[:, None]/(KB*temp[None, :])
    return 2.*H*nu[:, None]**3/C**2/np.expm1(np.minimum(x, 700.))


def band(fun, elo, ehi, temp, n=65):
    """int fun(nu, T) dnu over [elo, ehi] erg per cell, trapezoid in ln nu"""
    lnu = np.linspace(np.log(elo/H), np.log(ehi/H), n)
    nu = np.exp(lnu)
    wts = np.full(n, lnu[1] - lnu[0])
    wts[0] *= 0.5
    wts[-1] *= 0.5
    return (fun(nu, temp)*(nu*wts)[:, None]).sum(axis=0)


def cell_volumes(f, coord, l_cgs):
    x1f, x2f, x3f = (f[k][...].astype(np.float64) for k in ('x1f', 'x2f', 'x3f'))
    if coord == 'spherical_polar':
        vol = ((x1f[:, 1:]**3 - x1f[:, :-1]**3)/3.)[:, None, None, :] \
            * (np.cos(x2f[:, :-1]) - np.cos(x2f[:, 1:]))[:, None, :, None] \
            * (x3f[:, 1:] - x3f[:, :-1])[:, :, None, None]
    elif coord == 'cartesian':
        vol = (x1f[:, 1:] - x1f[:, :-1])[:, None, None, :] \
            * (x2f[:, 1:] - x2f[:, :-1])[:, None, :, None] \
            * (x3f[:, 1:] - x3f[:, :-1])[:, :, None, None]
    else:
        sys.exit('coordinates ' + coord + ' not handled')
    return vol*l_cgs**3


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('snapshot')
    ap.add_argument('table')
    ap.add_argument('--rho-cgs', type=float, required=True)
    ap.add_argument('--tgas-cgs', type=float, required=True)
    ap.add_argument('--l-cgs', type=float, required=True)
    ap.add_argument('--tfloor-cgs', type=float, default=0.)
    ap.add_argument('--bands', type=float, nargs='+', default=[0.001, 0.01, 0.1, 1., 10.],
                    help='band edges in keV for the summary (default decades)')
    ap.add_argument('--chunk', type=int, default=200000)
    args = ap.parse_args()

    tab = Table(args.table)
    ng, nt, nr = len(tab.lo), tab.nt, tab.nr
    lr = np.log10(tab.rg)
    dlr = (lr[-1] - lr[0])/(nr - 1)
    lplan = np.log(np.where(tab.plan > 0, tab.plan, 1e-300))
    edges = np.concatenate([tab.lo, [tab.lo[-1]*(tab.lo[1]/tab.lo[0])]])

    with h5py.File(args.snapshot, 'r') as f:
        names = [n.decode() for n in f.attrs['VariableNames']]
        prim = f['prim']
        dens = prim[names.index('rho')].astype(np.float64)
        press = prim[names.index('press')].astype(np.float64)
        vol = cell_volumes(f, f.attrs['Coordinates'].decode(), args.l_cgs)
    rho = args.rho_cgs*dens
    temp = np.where(dens > 0., args.tgas_cgs*press/np.where(dens > 0., dens, 1.),
                    args.tfloor_cgs)
    temp = np.maximum(temp, args.tfloor_cgs)
    rho, temp = rho.ravel(), temp.ravel()
    vol = np.broadcast_to(vol, dens.shape).ravel()
    off = ((temp < tab.tg[0]) | (temp > tab.tg[-1]) | (rho < tab.rg[0])
           | (rho > tab.rg[-1])).sum()
    print('{} cells, {} off the table grid (clamped)'.format(rho.size, off))

    # the bracket and weights of OpacityTable::Locate, clamped
    xi = (np.log10(rho) - lr[0])/dlr
    ii = np.clip(np.floor(xi).astype(int), 0, nr - 2)
    xi = np.clip(xi - ii, 0., 1.)
    jj = np.clip(np.searchsorted(tab.tg, temp) - 1, 0, nt - 2)
    xj = np.clip(np.log(temp/tab.tg[jj])/np.log(tab.tg[jj+1]/tab.tg[jj]), 0., 1.)

    lum_mean = np.zeros(ng)
    lum_edge = np.zeros(ng)
    nu_nodes = tab.lo/H
    for start in range(0, rho.size, args.chunk):
        sel = slice(start, min(start + args.chunk, rho.size))
        kap = np.exp((1 - xi[sel])*((1 - xj[sel])*lplan[:, jj[sel], ii[sel]]
                                    + xj[sel]*lplan[:, jj[sel]+1, ii[sel]])
                     + xi[sel]*((1 - xj[sel])*lplan[:, jj[sel], ii[sel]+1]
                                + xj[sel]*lplan[:, jj[sel]+1, ii[sel]+1]))
        tc = temp[sel]
        rv = rho[sel]*vol[sel]
        bnode = planck_nu(nu_nodes, tc)
        for g in range(ng):
            lum_mean[g] += (4*np.pi*band(planck_nu, edges[g], edges[g+1], tc)*kap[g]*rv).sum()
            if g < ng - 1:
                dln = np.log(nu_nodes[g+1]/nu_nodes[g])
                emean = (tab.lo[g+1] - tab.lo[g])/np.log(tab.lo[g+1]/tab.lo[g])
                rate = 4*np.pi/H*0.5*(bnode[g]*kap[g] + bnode[g+1]*kap[g+1])*dln
                lum_edge[g] += (rate*emean*rv).sum()

    print('emitted luminosity per group [erg/s]')
    print('{:>3s} {:>10s} {:>11s} {:>11s} {:>9s}'.format('g', 'E_lo keV', 'mean', 'edge',
                                                        'mean/edge'))
    for g in range(ng):
        ratio = lum_mean[g]/lum_edge[g] if lum_edge[g] > 0. else float('inf')
        print('{:3d} {:10.5f} {:11.4e} {:11.4e} {:9.4f}'.format(g, tab.lo[g]/KEV, lum_mean[g],
                                                               lum_edge[g], ratio))
    print('per band [keV]')
    bands = np.array(args.bands)*KEV
    glo = edges[:-1]
    for lo, hi in zip(bands[:-1], bands[1:]):
        inb = (glo >= lo*(1 - 1e-6)) & (glo < hi*(1 - 1e-6))
        lm, le = lum_mean[inb].sum(), lum_edge[inb].sum()
        print('{:9.4g}-{:<9.4g} {:11.4e} {:11.4e} {:9.4f}'.format(
            lo/KEV, hi/KEV, lm, le, lm/le if le > 0. else float('inf')))
    print('{:>19s} {:11.4e} {:11.4e} {:9.4f}'.format('total', lum_mean.sum(), lum_edge.sum(),
                                                   lum_mean.sum()/lum_edge.sum()))


if __name__ == '__main__':
    main()
