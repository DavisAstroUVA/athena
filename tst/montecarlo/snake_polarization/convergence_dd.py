#! /usr/bin/env python

"""
Parallel-transport accuracy test for the polarization tensor in snake coordinates.

Snake coordinates (White, Stone & Gammie 2016) are flat spacetime written in a sheared
frame, so they carry a nonzero connection while still admitting a global covariantly
constant frame.  Projected onto that frame the coherency tensor of a freely propagating
photon is exactly constant, whatever the connection does to its coordinate components.
mc_snake records the running maximum departure from the emitted value for each photon;
that spread is transport error and nothing else.

Like the geodesic test this is deterministic in the quantity being refined, so the
convergence knob is the integration step size rather than the photon number.  The
geodesic itself is integrated by RK4, while GeneralPusher::AdvanceStep transports the
tensor with Heun's method -- the rate taken once at each end of the geodesic step and
averaged -- so the drift should fall as stepsize**2 and the reported order should sit near
2.0.  It read 1.0 before that change, when the connection was evaluated only after the
step and applied as a single Euler update.
"""

# python standard modules
import argparse
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
from os import system

# Athena++ modules
import athena_mc as mcspec

# The photon list slot mc_snake writes the polarization drift into.  Slots 0 and 1 are the
# geodesic invariant, which frames.py reads.
IPOLDRIFT = 2

# Box and shear.  beta = snake_a*snake_k*cos(snake_k*x) is the only free parameter of the
# metric; 0.3 is a substantial shear.  snake_k puts exactly one period across the box, so
# the deck stays legal if the boundaries are later switched to periodic.
SNAKE_L = 1.0e11

KB = 1.380649e-16
TGAS = 1.0e6
DENS = 1.0e-10
PGAS_CODE = 1.0e-6

# Free-free emission bounds as x = E/(kT), which is what mc_snake's tnorm = true branch
# expects.  The energy scale cancels out of the transport -- StepSize divides by |k| and
# RK4 multiplies back by it -- but it must not be absurdly small in cgs, because
# MeridianBasis rejects a wavevector whose magnitude falls below TINY_NUMBER and then
# quietly leaves the coherency tensor at zero.  See the comment in read_drift.
XMIN = 0.01
XMAX = 20.0


def write_athinput(iseed, nphot, stepsize, beta, polang, polcirc, nx,
                   file='athinput.mcsnakepol'):
    """
    Write the athinput file for one step size.
    """
    snake_k = 2.0 * math.pi / SNAKE_L
    snake_a = beta / snake_k

    # polcirc is a Stokes V, so it only round-trips through the photon list when the run
    # tracks V.  The test reads user variables rather than Stokes columns, but keeping the
    # tag honest means a list file dumped from this deck says what it carries.
    polarized = "circular" if polcirc != 0.0 else "linear"

    o = ["<comment>",
         "problem   =  Polarization parallel transport in snake coordinates",
         "reference =",
         "configure = --prob=mc_snake --coord=gr_user -g -mc",
         "",
         "<job>", "problem_id = mcsnakepol", "",
         # user[2] is the drift; the reference tensor occupies the slots above it and is
         # deliberately not written out.
         "<output1>", "file_type  = phlist", "nuser      = 3", "dt         = 1.0e-11", "",
         "<time>", "cfl_number = 0.3", "nlim       = 1", "tlim       = 1.0", "",
         # GRUser reads m and a unconditionally and neither means anything here.
         "<coord>", "m = 0.0", "a = 0.0",
         "snake_a = {0!r}".format(snake_a), "snake_k = {0!r}".format(snake_k), ""]

    o += ["<mesh>"]
    for d in (1, 2, 3):
        o += ["nx{0} = {1:d}".format(d, nx),
              "x{0}min = {1!r}".format(d, -0.5 * SNAKE_L),
              "x{0}max = {1!r}".format(d, 0.5 * SNAKE_L),
              "ix{0}_bc = periodic".format(d), "ox{0}_bc = periodic".format(d),
              # Photons leave rather than wrap, so every photon travels a fixed physical
              # path set by the box and the step size only controls how it is resolved.
              "ix{0}_mc_bc = escape".format(d), "ox{0}_mc_bc = escape".format(d), ""]

    o += ["<meshblock>", "nx1 = {0:d}".format(nx), "nx2 = {0:d}".format(nx),
          "nx3 = {0:d}".format(nx), "",
          "<hydro>", "gamma = 1.666666666666667", "",
          "<montecarlo>",
          "nphot          = {0:d}".format(nphot),
          "iseed          = {0:d}".format(iseed),
          "general_pusher = true",
          "mc_coord       = snake",
          "scattering     = none",
          "absorption     = none",
          "emission       = freefree",
          "polarized      = " + polarized,
          "stepsize       = {0!r}".format(stepsize),
          "varystep       = true",
          # boosts drives TransformToCoordinate, without which the emitted direction never
          # becomes a coordinate four-vector and the coherency tensor is never built.
          "boosts         = true",
          "tmax           = 1.0e36", "",
          "<problem>",
          "dens_code = 1.0", "pgas_code = {0:e}".format(PGAS_CODE),
          "rho_cgs = {0:e}".format(DENS),
          "tgas_cgs = {0:e}".format(TGAS / PGAS_CODE),
          "l_cgs = 1.0", "velocity = 0.0",
          "polang  = {0!r}".format(polang),
          "polcirc = {0!r}".format(polcirc),
          "tnorm = true",
          "emin = {0!r}".format(XMIN),
          "emax = {0!r}".format(XMAX), ""]

    open(file, "w").write("\n".join(o))


def read_drift(pattern="mcsnakepol.out1.proc*.00000.list"):
    """
    Read every rank's photon list and return the per-photon polarization drift, already
    normalized inside the pgen by the largest reference component.
    """
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        raise RuntimeError("no photon list files matching " + pattern)

    drift = []
    for fname in files:
        reader = mcspec.read_list_generator(fname)
        result = next(reader)
        header = result['header']
        for result in reader:
            phlist = header.copy()
            phlist['list'] = result['chunk']
            phlist['length'] = result['length']
            photons = mcspec.Photons(phlist)
            if photons.nuser <= IPOLDRIFT:
                raise RuntimeError("photon list carries {:d} user variables, need {:d}"
                                   " -- is nuser = 3 set on the output block?"
                                   .format(photons.nuser, IPOLDRIFT + 1))
            drift.append(photons.user[:, IPOLDRIFT])

    drift = np.concatenate(drift)

    # Photons that never got a coherency tensor keep the pgen's -1 sentinel; a drift of
    # exactly 0 is a real measurement, and the one an unsheared run is supposed to give.
    # Losing photons in bulk means MeridianBasis declined to build the tensor, most likely
    # because the emitted wavevector fell below its absolute TINY_NUMBER magnitude test,
    # and that failure is otherwise invisible in the output.
    live = drift >= 0.
    nlive = int(live.sum())
    if nlive < 0.5 * drift.size:
        raise RuntimeError(
            "only {:d} of {:d} photons carry a coherency tensor.  Check that the emitted"
            " photon energies are not so small that MeridianBasis rejects the wavevector."
            .format(nlive, drift.size))
    return drift[live], nlive, drift.size


# Main function
def main(**kwargs):

    path = kwargs.pop("path")
    athena_path = path + "/bin"
    infile = "athinput.mcsnakepol"

    mcranks = kwargs['mcranks']
    nstep = kwargs['nstep']

    steps = [kwargs['step0'] / float(kwargs['refine'])**i for i in range(nstep)]

    results = np.zeros((nstep, 3))
    for i, stepsize in enumerate(steps):
        write_athinput(kwargs['iseed'], kwargs['nphot'], stepsize, kwargs['beta'],
                       kwargs['polang'], kwargs['polcirc'], kwargs['nx'], file=infile)
        system("rm -f mcsnakepol.out1.proc*.list")
        if mcranks == 1:
            com = athena_path + "/athena -i " + infile
        else:
            com = "mpirun -np {:d} ".format(mcranks) + athena_path + "/athena -i " + infile
        print(com)
        system(com)

        drift, nlive, ntot = read_drift()

        results[i, 0] = stepsize
        results[i, 1] = np.median(drift)
        results[i, 2] = np.percentile(drift, 99.)

        print("  stepsize {:e}: {:d}/{:d} photons, median/p99 drift {:.3e}/{:.3e}"
              .format(stepsize, nlive, ntot, results[i, 1], results[i, 2]))

    np.savetxt(kwargs['outfile'], results)

    # observed order of convergence between successive refinements
    print("\nstepsize      median drift   order    p99 drift     order")
    for i in range(nstep):
        if i == 0:
            print("{:.4e}   {:.4e}      -     {:.4e}      -"
                  .format(results[i, 0], results[i, 1], results[i, 2]))
        else:
            ratio = results[i - 1, 0] / results[i, 0]

            def order(a, b):
                # An unsheared run has no connection to integrate and lands on exactly
                # zero, where an order is undefined rather than badly behaved.
                if a <= 0. or b <= 0.:
                    return "    -"
                return "{:5.2f}".format(np.log(a / b) / np.log(ratio))

            print("{:.4e}   {:.4e}   {:s}   {:.4e}   {:s}"
                  .format(results[i, 0], results[i, 1],
                          order(results[i - 1, 1], results[i, 1]), results[i, 2],
                          order(results[i - 1, 2], results[i, 2])))

    plt.plot(results[:, 0], results[:, 1], 'r+-', label='median')
    plt.plot(results[:, 0], results[:, 2], 'bx-', label='99th percentile')
    plt.plot(results[:, 0], results[-1, 1] * (results[:, 0] / results[-1, 0]),
             'k--', label='1st order')
    plt.plot(results[:, 0], results[-1, 1] * (results[:, 0] / results[-1, 0])**2,
             'k:', label='2nd order')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('stepsize')
    plt.ylabel('flat-frame coherency tensor drift')
    plt.legend()
    plt.savefig("conv.pdf")


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('iseed', type=int, help='random seed')
    parser.add_argument('nstep', type=int, help='number of step size refinements')
    parser.add_argument('--step0', type=float, default=1.0e-2,
                        help='coarsest stepsize')
    parser.add_argument('--refine', type=float, default=2.0,
                        help='factor to reduce stepsize by each iteration')
    parser.add_argument('--nphot', type=int, default=1024,
                        help='number of photons')
    parser.add_argument('--nx', type=int, default=8,
                        help='zones per dimension')
    parser.add_argument('--beta', type=float, default=0.3,
                        help='peak shear beta = snake_a*snake_k')
    parser.add_argument('--polang', type=float, default=0.3,
                        help='emitted linear polarization angle in radians')
    parser.add_argument('--polcirc', type=float, default=0.0,
                        help='emitted Stokes V')
    parser.add_argument('--mcranks', type=int, default=1,
                        help='mpi ranks to use')
    parser.add_argument('--path', default="/home/swd8g/athena-swdavis",
                        help='path to Athena++ distribution')
    parser.add_argument('--outfile', default="conv.out",
                        help='output filename for storing convergence rate')

    args = parser.parse_args()
    main(**vars(args))
