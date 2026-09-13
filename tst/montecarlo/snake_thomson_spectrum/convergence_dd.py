#! /usr/bin/env python

"""
Polarized Thomson atmosphere in snake coordinates, checked against Feautrier.

The same plane-parallel isothermal scattering atmosphere that
tst/montecarlo/thomson_polarized_spectrum runs in cartesian coordinates with the legacy
pusher, run instead in the sheared chart of White, Stone & Gammie (2016) with the general
pusher.  Snake coordinates are flat spacetime in a curved chart, and the atmosphere is
stratified along x3 where z = z_M exactly, so the physical problem is unchanged and the
emergent intensity and polarization must converge to the same Feautrier solution.

That makes this a joint test of everything the general pusher does that the legacy pusher
does not: integrating geodesics through a metric with g_xy != 0, parallel transporting the
coherency tensor between scatterings, and projecting into and out of the meridian basis at
every scattering.  All of it has to cancel exactly, because the shear is a change of
variables and nothing else.

Run --beta 0 for the control: the metric degenerates to Minkowski while the run still goes
through the general pusher and the GR machinery, which separates "the shear is handled
correctly" from "the general pusher path works at all".

The norm is a mean fractional error over every populated bin, so it is dominated by shot
noise at these photon numbers and falls slowly; the cartesian version of this test drops at
a rate near 0.3, not the 0.5 of pure noise.  Compare against that, not against 0.5.

--stepsize refines the geodesic step, which is the one error source the legacy pusher does
not have.  Refining it 4x at 20000 photons was measured to leave both norms unchanged to
within the run-to-run scatter, so at these settings the residual is shot noise and not the
transport; keep that in mind before reading a difference against the cartesian run as a
systematic.

Requires a build configured for this problem generator:

    python configure.py --prob=mc_snake_atm --coord=gr_user -g -mc && make
"""

# python standard modules
import argparse
import math
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from os import system

# Athena++ modules
import athena_mc as mcspec

# feautrier.py lives with the cartesian version of this test; there is no reason to have
# two copies of the reference solver.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "thomson_polarized_spectrum"))
import feautrier as feaut

# Box.  The face the spectrum is collected on is x1 by x2, and snake has unit spatial
# determinant, so the proper area of that face equals the coordinate area and the
# normalization is the same one the cartesian test uses.
LX = 1.0e11
AREA = LX * LX


def interp_feaut(mu0, mu, varin):
    """Interpolate the Feautrier solution in angle."""
    nnu = len(varin[:, 0])
    varout = np.zeros(nnu)
    for i in range(nnu):
        varout[i] = np.interp(mu0, mu, varin[i, :])
    return varout


def write_athinput(iseed, nphot, nen, emin, emax, beta, stepsize,
                   file='athinput.mcsnakeatm'):
    """Write the athinput file for one photon number."""
    snake_k = 2.0 * math.pi / LX
    snake_a = beta / snake_k if snake_k != 0.0 else 0.0

    o = ["<comment>",
         "problem   = Polarized Thomson atmosphere in snake coordinates",
         "reference = White, Stone & Gammie (2016), ApJS 225, 22",
         "configure = --prob=mc_snake_atm --coord=gr_user -g -mc", "",
         "<job>", "problem_id = mcsnakeatm", "",
         "<output1>", "file_type = spec", "face      = outer_x3",
         "ne        = {:d}".format(nen),
         "emin      = {:e}".format(emin),
         "emax      = {:e}".format(emax),
         "ncth      = 8", "nphi      = 8", "",
         "<time>", "cfl_number = 0.1", "nlim = 1", "tlim = 1.0", "",
         # GRUser reads m and a unconditionally and neither means anything here.
         "<coord>", "m = 0.0", "a = 0.0",
         "snake_a = {0!r}".format(snake_a), "snake_k = {0!r}".format(snake_k), "",
         "<mesh>"]

    # x1 and x2 are periodic and the shear closes over x1 by construction, so the box is a
    # slab: photons only ever leave through x3.
    for d in (1, 2):
        o += ["nx{0} = 32".format(d),
              "x{0}min = {1!r}".format(d, -0.5 * LX),
              "x{0}max = {1!r}".format(d, 0.5 * LX),
              "ix{0}_bc = periodic".format(d), "ox{0}_bc = periodic".format(d),
              "ix{0}_mc_bc = periodic".format(d), "ox{0}_mc_bc = periodic".format(d), ""]
    o += ["nx3 = 128", "x3min = 0.0", "x3max = {0!r}".format(LX),
          "ix3_bc = outflow", "ox3_bc = outflow",
          "ix3_mc_bc = absorb", "ox3_mc_bc = escape", "",
          "<meshblock>", "nx1 = 16", "nx2 = 16", "nx3 = 32", "",
          "<hydro>", "gamma = 1.666666666666667", "",
          "<montecarlo>",
          "mc_coord       = snake",
          "general_pusher = true",
          "nphot     = {:d}".format(nphot),
          "iseed     = {:d}".format(iseed),
          "scattering = thomson",
          "emission   = freefree",
          "absorption = freefree",
          "polarized  = linear",
          # boosts drives the comoving-to-coordinate conversion at emission, without which
          # the wavevector never becomes a coordinate four-vector
          "boosts     = true",
          "stepsize   = {0!r}".format(stepsize),
          "varystep   = true", "",
          "<problem>",
          "temp     = 1.0e5",
          "taumin   = 1.0e-3",
          "taumax   = 1.0e4",
          "emin      = {:e}".format(emin),
          "emax      = {:e}".format(emax), ""]

    open(file, "w").write("\n".join(o))


# Main function
def main(**kwargs):

    path = kwargs.pop("path")
    athena_path = path + "/bin"
    infile = "athinput.mcsnakeatm"

    mcranks = kwargs['mcranks']
    nstep = kwargs['nstep']
    nphots = [kwargs['nmin']]
    step = kwargs['step']
    for i in range(nstep - 1):
        nphots.append(nphots[i] * step)
    iseed = kwargs['iseed']

    lnorm = np.zeros((nstep, 3))

    ffile = kwargs.pop('ffile')
    intensf = None
    for i, nphot in enumerate(nphots):
        write_athinput(iseed + 99*i, nphot, kwargs['nen'], kwargs['emin'],
                       kwargs['emax'], kwargs['beta'], kwargs['stepsize'], file=infile)
        if mcranks == 1:
            com = athena_path + "/athena -i " + infile
        else:
            com = "mpirun -np {:d} ".format(mcranks) + athena_path + "/athena -i " + infile
        print(com)
        specfile = "mcsnakeatm.out1.00000.spec"
        if os.path.exists(specfile):
            os.remove(specfile)
        system(com)
        if not os.path.exists(specfile):
            raise RuntimeError("athena produced no spectrum; the run failed.  Check that"
                               " the build is configured for mc_snake_atm and that the"
                               " mesh has at least as many blocks as mpi ranks.")

        spectrum = mcspec.read_spectrum(specfile)
        mumid = 0.5*(spectrum['mufaces'][1:] + spectrum['mufaces'][:-1])
        if intensf is None:
            if ffile is None:
                print("Computing feautrier transfer")
                xfaces = spectrum['xfaces']
                h = 6.62607015e-27
                everg = 1.6021772e-12
                nu = 0.5*(xfaces[1:] + xfaces[:-1])*everg/h
                feaut.transfer(nd=128, na=32, nu=nu)
                ffile = "feautrier.out"
            nuf, muf, intensf, polf = feaut.read_feautrier(ffile)

        # average over phi
        intensity = spectrum['intensity']
        intens = np.sum(intensity[0, :, :, :], axis=0)/float(spectrum['nphi'])
        qpol = np.sum(intensity[1, :, :, :], axis=0)/float(spectrum['nphi'])
        # Bins that collected no photons have intens == 0; the polarization fraction is
        # undefined there, so mask them out rather than letting a nan poison the norm.
        empty = (intens == 0.)
        qpol = np.divide(qpol, intens, out=np.zeros_like(qpol), where=~empty)

        normi = 0.
        normp = 0.
        n = 0
        npol = 0
        for j in range(spectrum['nmu']):
            iinterp = AREA*interp_feaut(mumid[j], muf, intensf)
            pinterp = interp_feaut(mumid[j], muf, polf)
            for k in range(spectrum['nx']):
                normi += abs(intens[j, k] - iinterp[k])/iinterp[k]
                n += 1
                if empty[j, k]:
                    continue
                normp += abs(qpol[j, k] - pinterp[k])
                npol += 1
        if empty.any():
            print("  {:d} of {:d} spectral bins were empty and are excluded"
                  " from the polarization norm".format(int(empty.sum()), n))
        lnorm[i, 0] = float(nphot)
        lnorm[i, 1] = normi/float(n)
        lnorm[i, 2] = normp/float(npol) if npol > 0 else np.nan
        print("  nphot {:d}: intensity norm {:.4e}  polarization norm {:.4e}"
              .format(nphot, lnorm[i, 1], lnorm[i, 2]))

    np.savetxt(kwargs['outfile'], lnorm)

    print("\n     nphot     intensity norm   rate    polarization norm   rate")
    for i in range(nstep):
        if i == 0:
            print("{:10.0f}   {:.4e}       -      {:.4e}         -"
                  .format(lnorm[i, 0], lnorm[i, 1], lnorm[i, 2]))
        else:
            ratio = lnorm[i, 0]/lnorm[i - 1, 0]
            ri = np.log(lnorm[i - 1, 1]/lnorm[i, 1])/np.log(ratio)
            rp = np.log(lnorm[i - 1, 2]/lnorm[i, 2])/np.log(ratio)
            print("{:10.0f}   {:.4e}   {:5.2f}      {:.4e}     {:5.2f}"
                  .format(lnorm[i, 0], lnorm[i, 1], ri, lnorm[i, 2], rp))
    print("\nThe norms are means of the fractional error against Feautrier over every")
    print("populated bin, so they mix photon shot noise with any systematic offset.  The")
    print("cartesian version of this test converges at a rate near 0.3 rather than the")
    print("0.5 of pure shot noise, so that is the number to compare against, not 0.5.")
    print("Refining --stepsize 4x was measured not to move either norm, so at these")
    print("photon numbers the residual is shot noise rather than the geodesic step.")

    plt.plot(lnorm[:, 0], lnorm[:, 1], '+-', label='intensity')
    plt.plot(lnorm[:, 0], lnorm[:, 2], 'x-', label='polarization')
    plt.plot(lnorm[:, 0], lnorm[-1, 1]*(lnorm[-1, 0]/lnorm[:, 0])**0.5, 'k--',
             label=r'$N^{-1/2}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('photons')
    plt.ylabel('mean fractional error vs Feautrier')
    plt.legend()
    plt.savefig("conv.pdf")


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('iseed', type=int, help='random seed')
    parser.add_argument('nmin', type=int, help='minimum photon number')
    parser.add_argument('nstep', type=int, help='number of steps')
    parser.add_argument('step', type=int, default=2,
                        help='factor by which to increase nphot')
    parser.add_argument('--beta', type=float, default=0.3,
                        help='peak shear snake_a*snake_k; 0 degenerates to Minkowski')
    parser.add_argument('--stepsize', type=float, default=1.0e-2,
                        help='geodesic step, as a fraction of the zone crossing time')
    parser.add_argument('--mcranks', type=int, default=1, help='mpi ranks to use')
    parser.add_argument('--ffile', default=None, help='feautrier input file')
    parser.add_argument('--nen', type=int, default=100,
                        help='number of photon energy bins')
    parser.add_argument('--emin', type=float, default=1., help='minimum photon energy')
    parser.add_argument('--emax', type=float, default=100., help='maximum photon energy')
    parser.add_argument('--path', default="/home/swd8g/athena-swdavis",
                        help='path to Athena++ distribution')
    parser.add_argument('--outfile', default="conv.out",
                        help='output filename for storing convergence rate')

    args = parser.parse_args()
    main(**vars(args))
