#! /usr/bin/env python

"""
Frame-consistency test for the fluid four-velocity in Kerr-Schild spacetime.

The Monte Carlo used to store a four-velocity built and normalized at the cell center and
then contract it with the metric at the photon, so u.u was -1 where the vector was built
and -1 + O(dx dg) where it was used.  MonteCarloBlock::FluidFourVelocity now rebuilds it
from the stored primitives at whatever point is asked for, and this test measures how far
u.u departs from -1 at the two points that matter:

  IUUDEV -- at the photon, the frame the transport uses;
  IUUCEN -- at the photon's cell center, the frame the per-cell moment matrices use.

Both should sit at roundoff.  They are separate numbers because they are separate frames:
the transport follows the photon while the moments are cell averages, and that split is
deliberate (see UpdateMoments).  Before the reconstruction IUUDEV read ~7e-2 and fell only
as fast as the cell width.

No other test in the tree can see this.  Snake has g_tt = -1 with no time cross terms and
a static fluid, so u.u is -1 identically; Minkowski is flat.  Every polarized test runs in
snake.  Hence Kerr-Schild.

The deviation is first order in the photon's offset from the cell center, so refining the
radial grid should shrink it in proportion to the cell width -- that is the check that the
number being measured is the offset error and not something else.  Once the four-velocity
is rebuilt at the photon from the primitives (see
doc/monte_carlo/velocity_reconstruction_plan.md) the deviation should collapse to roundoff
at every resolution, and the reported order becomes meaningless rather than 1.

Usage:
    python kerr_frames.py [--nx1 64] [--nres 3] [--path /path/to/athena]

Needs: python configure.py --prob=mc_gr_simple --coord=kerr-schild -g -mc && make
"""

# python standard modules
import argparse
import glob
import numpy as np
from os import system

# Athena++ modules
import athena_mc as mcspec

# Photon list slots written by mc_gr_simple's InsideHorizon hook.  IUUDEV is measured at
# the photon, the frame the transport uses; IUUCEN at the photon's cell center, the frame
# ComputeTransformations and ComovingFrameMatrix build the per-cell moment matrices on.
# The two are deliberately different points -- see UpdateMoments -- so they are reported
# separately, but both must sit at roundoff.
IUUDEV = 0
IUURAD = 1
IUUCEN = 2

RMIN = 2.0
RMAX = 20.0
SPIN = 0.9375


def write_athinput(nx1, nphot, iseed, file='athinput.kerrframes'):
    """
    Write the athinput for one radial resolution.
    """
    # Keep the block count fixed as nx1 grows so the ghost-cell population, and hence what
    # the diagnostic skips, does not change with resolution.
    nx1_block = nx1 // 2

    o = ["<comment>",
         "problem   =  Kerr-Schild fluid four-velocity frame consistency",
         "configure = --prob=mc_gr_simple --coord=kerr-schild -g -mc",
         "",
         "<job>", "problem_id = kerrframes", "",
         "<output1>", "file_type = phlist", "nuser     = 3", "",
         "<time>", "cfl_number = 0.3", "nlim       = 1", "tlim       = 1.0", "",
         "<coord>", "m = 1.0", "a = {0!r}".format(SPIN), "",
         "<mesh>",
         "nx1        = {0:d}".format(nx1),
         "x1min      = {0!r}".format(RMIN),
         "x1max      = {0!r}".format(RMAX),
         "ix1_mc_bc  = absorb", "ox1_mc_bc  = escape",
         "nx2        = 16", "x2min      = 0.0",
         "x2max      = 3.14159265358979",
         "ix2_mc_bc  = polar", "ox2_mc_bc  = polar",
         "nx3        = 8", "x3min      = 0.0",
         "x3max      = 6.28318530717959",
         "ix3_mc_bc  = periodic", "ox3_mc_bc  = periodic", "",
         "<meshblock>",
         "nx1 = {0:d}".format(nx1_block), "nx2 = 8", "nx3 = 4", "",
         "<hydro>", "gamma = 1.666666666666667", "",
         "<montecarlo>",
         "nphot          = {0:d}".format(nphot),
         "iseed          = {0:d}".format(iseed),
         "scattering     = none",
         "emission       = freefree",
         "absorption     = freefree",
         "polarized      = none",
         "general_pusher = true",
         # boosts drives GetVelocity(); without it vel holds the normal observer and the
         # fluid three-velocity never enters, which is not the case being tested.
         "boosts         = true",
         "stepsize       = 1.0e-2",
         "varystep       = true", "",
         "<problem>",
         "temp      = 1.0e6",
         "constdens = true",
         "dens      = 1.0e-6",
         "tnorm     = true",
         "emin      = 0.3",
         "emax      = 20.",
         "l_cgs     = 1476625.0380501247",
         "vel_cgs   = 29979245800.0",
         "tgas_cgs  = 1.08101416735e+13", ""]

    open(file, "w").write("\n".join(o))


def read_deviation(pattern="kerrframes.out1.proc*.00000.list"):
    """
    Return the per-photon maximum of |u.u + 1| and the radius at which it peaked.
    """
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        raise RuntimeError("no photon list files matching " + pattern)

    user = []
    for fname in files:
        photons = mcspec.Photons(mcspec.read_list(fname))
        if photons.nuser <= IUUCEN:
            raise RuntimeError("photon list carries {:d} user variables, need {:d} --"
                               " is nuser = 3 set on the output block?"
                               .format(photons.nuser, IUUCEN + 1))
        user.append(photons.user)

    user = np.concatenate(user, axis=0)
    if user.shape[0] == 0:
        raise RuntimeError("photon lists are empty -- every photon was destroyed or"
                           " retired rather than escaping.  If <montecarlo>/capmove is"
                           " set, check it is well above the steps a photon needs to"
                           " cross the domain: with varystep the step is a fraction of a"
                           " cell, so a finer grid needs more of them.")
    return user[:, IUUDEV], user[:, IUURAD], user[:, IUUCEN]


def main(**kwargs):

    athena = kwargs.pop("path") + "/bin/athena"
    infile = "athinput.kerrframes"

    resolutions = [kwargs['nx1'] * 2**i for i in range(kwargs['nres'])]
    results = np.zeros((len(resolutions), 3))

    for i, nx1 in enumerate(resolutions):
        write_athinput(nx1, kwargs['nphot'], kwargs['iseed'], file=infile)
        system("rm -f kerrframes.out1.proc*.list")
        com = athena + " -i " + infile
        print(com)
        system(com)

        dev, rad, cen = read_deviation()
        dr = (RMAX - RMIN) / float(nx1)
        results[i] = (dr, dev.max(), cen.max())
        print("  nx1 {:4d}  dr {:.4f}  photon-frame |u.u+1| {:.4e}"
              "  cell-center {:.4e}  peak at r {:.3f}"
              .format(nx1, dr, results[i, 1], results[i, 2], np.median(rad)))

    print("\n   dr       photon frame   order     cell center   order")
    for i in range(len(resolutions)):
        if i == 0:
            print("{:.4e}   {:.4e}      -     {:.4e}      -"
                  .format(results[i, 0], results[i, 1], results[i, 2]))
        else:
            ratio = results[i - 1, 0] / results[i, 0]

            def order(a, b):
                # Once the velocity is rebuilt at the photon both values sit at roundoff
                # and the ratio is noise, so report a dash rather than a spurious order.
                if a <= 1.0e-13 or b <= 1.0e-13:
                    return "    -"
                return "{:5.2f}".format(np.log(a / b) / np.log(ratio))

            print("{:.4e}   {:.4e}   {:s}   {:.4e}   {:s}"
                  .format(results[i, 0], results[i, 1],
                          order(results[i - 1, 1], results[i, 1]), results[i, 2],
                          order(results[i - 1, 2], results[i, 2])))

    np.savetxt(kwargs['outfile'], results)

    worst_phot = results[:, 1].max()
    worst_cen = results[:, 2].max()
    print("\nlargest |u.u + 1|:  photon frame {:.4e}   cell center {:.4e}"
          .format(worst_phot, worst_cen))
    tol = 1.0e-13
    if worst_phot < tol and worst_cen < tol:
        print("PASS: both frames are unit timelike where they are used")
    elif worst_phot >= tol:
        print("FAIL: the transport frame is not normalized at the photon")
    else:
        print("FAIL: the moment frame is not normalized at the cell center")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nx1', type=int, default=64,
                        help='coarsest radial resolution')
    parser.add_argument('--nres', type=int, default=3,
                        help='number of resolution doublings')
    parser.add_argument('--nphot', type=int, default=5000,
                        help='number of photons')
    parser.add_argument('--iseed', type=int, default=121500,
                        help='random seed')
    parser.add_argument('--path', default="/home/swd8g/athena-swdavis",
                        help='path to Athena++ distribution')
    parser.add_argument('--outfile', default="kerr_frames.out",
                        help='output filename for the resolution scan')

    args = parser.parse_args()
    main(**vars(args))
