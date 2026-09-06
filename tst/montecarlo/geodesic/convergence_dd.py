#! /usr/bin/env python

"""
Geodesic integrator accuracy test in Kerr-Schild coordinates.

The Kerr metric is stationary and axisymmetric, so k_t and k_phi are exactly conserved
along a null geodesic.  mc_geodesic records the running minimum and maximum of both for
each photon; the spread is integration error.  Unlike the other Monte Carlo tests this is
deterministic, so the convergence knob is the integration step size rather than the
photon number, and drift should fall as stepsize**4 for RK4.
"""

# python standard modules
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
from os import system

# Athena++ modules
import athena_mc as mcspec

# sentinel written by the pgen for photons that never took a tracked step
BIG = 1.0e300


def write_athinput(iseed, nphot, stepsize, spin, rmin, rmax, nmu, nphi,
                   file='athinput.mcgeo'):
    """
    Write the athinput file for one step size.
    """
    outfile = open(file, 'w')
    outfile.write("<comment>\n")
    outfile.write("problem   =  Null geodesic invariants in Kerr-Schild\n")
    outfile.write("reference =\n")
    outfile.write("configure = --prob=mc_geodesic --coord=kerr-schild -g -mc\n")
    outfile.write("\n")
    outfile.write("<job>\n")
    outfile.write("problem_id = mcgeo\n")
    outfile.write("\n")
    outfile.write("<output1>\n")
    outfile.write("file_type  = phlist\n")
    outfile.write("nuser      = 4\n")
    outfile.write("dt         = 1.0e-11\n")
    outfile.write("\n")
    outfile.write("<time>\n")
    outfile.write("cfl_number = 0.3\n")
    outfile.write("nlim       = 1\n")
    outfile.write("tlim       = 1.0\n")
    outfile.write("\n")
    outfile.write("<coord>\n")
    outfile.write("m          = 1.0\n")
    outfile.write("a          = {:e}\n".format(spin))
    outfile.write("\n")
    outfile.write("<mesh>\n")
    outfile.write("nx1        = 32\n")
    outfile.write("x1min      = {:e}\n".format(rmin))
    outfile.write("x1max      = {:e}\n".format(rmax))
    outfile.write("x1rat      = 1.05\n")
    outfile.write("ix1_bc     = outflow\n")
    outfile.write("ox1_bc     = outflow\n")
    outfile.write("ix1_mc_bc  = absorb\n")
    outfile.write("ox1_mc_bc  = escape\n")
    outfile.write("\n")
    outfile.write("nx2        = 16\n")
    outfile.write("x2min      = 0.0\n")
    outfile.write("x2max      = 3.141592653589793\n")
    outfile.write("ix2_bc     = polar\n")
    outfile.write("ox2_bc     = polar\n")
    outfile.write("ix2_mc_bc  = polar\n")
    outfile.write("ox2_mc_bc  = polar\n")
    outfile.write("\n")
    outfile.write("nx3        = 16\n")
    outfile.write("x3min      = 0.0\n")
    outfile.write("x3max      = 6.283185307179586\n")
    outfile.write("ix3_bc     = periodic\n")
    outfile.write("ox3_bc     = periodic\n")
    outfile.write("ix3_mc_bc  = periodic\n")
    outfile.write("ox3_mc_bc  = periodic\n")
    outfile.write("\n")
    outfile.write("<meshblock>\n")
    outfile.write("nx1 = 16\n")
    outfile.write("nx2 = 8\n")
    outfile.write("nx3 = 8\n")
    outfile.write("\n")
    outfile.write("<hydro>\n")
    outfile.write("gamma = 1.333333333333333\n")
    outfile.write("\n")
    outfile.write("<montecarlo>\n")
    outfile.write("nphot          = {:d}\n".format(nphot))
    outfile.write("iseed          = {:d}\n".format(iseed))
    outfile.write("general_pusher = true\n")
    outfile.write("boosts         = false\n")
    outfile.write("scattering     = none\n")
    outfile.write("absorption     = none\n")
    outfile.write("emission       = user\n")
    outfile.write("polarized      = none\n")
    outfile.write("stepsize       = {:e}\n".format(stepsize))
    outfile.write("varystep       = true\n")
    outfile.write("tmax           = 1.0e36\n")
    outfile.write("\n")
    outfile.write("<problem>\n")
    outfile.write("nmu       = {:d}\n".format(nmu))
    outfile.write("nphi      = {:d}\n".format(nphi))
    outfile.write("dens      = 1.0e-10\n")
    outfile.write("pgas      = 1.0e-10\n")
    outfile.close()


def read_drift(pattern="mcgeo.out1.proc*.00000.list"):
    """
    Read every rank's photon list and return the per-photon fractional drift in the two
    conserved quantities.  Both are normalized by |k_t|, since k_phi passes through zero
    for radially launched photons.
    """
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        raise RuntimeError("no photon list files matching " + pattern)

    kt_min, kt_max, kp_min, kp_max = [], [], [], []
    for fname in files:
        reader = mcspec.read_list_generator(fname)
        result = next(reader)
        header = result['header']
        for result in reader:
            phlist = header.copy()
            phlist['list'] = result['chunk']
            phlist['length'] = result['length']
            photons = mcspec.Photons(phlist)
            if photons.nuser < 4:
                raise RuntimeError("photon list carries {:d} user variables, need 4"
                                   " -- is nuser = 4 set on the output block?"
                                   .format(photons.nuser))
            kt_min.append(photons.user[:, 0])
            kt_max.append(photons.user[:, 1])
            kp_min.append(photons.user[:, 2])
            kp_max.append(photons.user[:, 3])

    kt_min = np.concatenate(kt_min)
    kt_max = np.concatenate(kt_max)
    kp_min = np.concatenate(kp_min)
    kp_max = np.concatenate(kp_max)

    # drop photons that never took a tracked step (sentinels untouched)
    live = (kt_min < 0.5 * BIG) & (kt_max > -0.5 * BIG)
    kt_min, kt_max = kt_min[live], kt_max[live]
    kp_min, kp_max = kp_min[live], kp_max[live]

    scale = np.abs(0.5 * (kt_min + kt_max))
    good = scale > 0.
    drift_e = (kt_max - kt_min)[good] / scale[good]
    drift_l = (kp_max - kp_min)[good] / scale[good]

    return drift_e, drift_l, int(live.sum())


# Main function
def main(**kwargs):

    path = kwargs.pop("path")
    athena_path = path + "/bin"
    infile = "athinput.mcgeo"

    mcranks = kwargs['mcranks']
    nstep = kwargs['nstep']
    nphot = kwargs['nphot']
    spin = kwargs['spin']

    steps = [kwargs['step0'] / float(kwargs['refine'])**i for i in range(nstep)]

    results = np.zeros((nstep, 5))
    for i, stepsize in enumerate(steps):
        write_athinput(kwargs['iseed'], nphot, stepsize, spin, kwargs['rmin'],
                       kwargs['rmax'], kwargs['nmu'], kwargs['nphi'], file=infile)
        system("rm -f mcgeo.out1.proc*.list")
        if mcranks == 1:
            com = athena_path + "/athena -i " + infile
        else:
            com = "mpirun -np {:d} ".format(mcranks) + athena_path + "/athena -i " + infile
        print(com)
        system(com)

        drift_e, drift_l, nlive = read_drift()

        results[i, 0] = stepsize
        results[i, 1] = np.median(drift_e)
        results[i, 2] = np.percentile(drift_e, 99.)
        results[i, 3] = np.median(drift_l)
        results[i, 4] = np.percentile(drift_l, 99.)

        print("  stepsize {:e}: {:d} photons, median/p99 drift"
              "  k_t {:.3e}/{:.3e}   k_phi {:.3e}/{:.3e}"
              .format(stepsize, nlive, results[i, 1], results[i, 2],
                      results[i, 3], results[i, 4]))

    np.savetxt(kwargs['outfile'], results)

    # observed order of convergence between successive refinements
    print("\nstepsize      median k_t    order   median k_phi   order")
    for i in range(nstep):
        if i == 0:
            print("{:.4e}   {:.4e}      -     {:.4e}      -"
                  .format(results[i, 0], results[i, 1], results[i, 3]))
        else:
            ratio = results[i - 1, 0] / results[i, 0]
            oe = np.log(results[i - 1, 1] / results[i, 1]) / np.log(ratio)
            ol = np.log(results[i - 1, 3] / results[i, 3]) / np.log(ratio)
            print("{:.4e}   {:.4e}   {:5.2f}   {:.4e}   {:5.2f}"
                  .format(results[i, 0], results[i, 1], oe, results[i, 3], ol))

    plt.plot(results[:, 0], results[:, 1], 'r+-', label=r'$k_t$ (median)')
    plt.plot(results[:, 0], results[:, 3], 'bx-', label=r'$k_\phi$ (median)')
    plt.plot(results[:, 0], results[-1, 1] * (results[:, 0] / results[-1, 0])**4,
             'k--', label=r'4th order')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('stepsize')
    plt.ylabel('fractional drift')
    plt.legend()
    plt.savefig("conv.pdf")


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('iseed', type=int, help='random seed')
    parser.add_argument('nstep', type=int, help='number of step size refinements')
    parser.add_argument('--step0', type=float, default=0.5,
                        help='coarsest stepsize')
    parser.add_argument('--refine', type=float, default=2.0,
                        help='factor to reduce stepsize by each iteration')
    parser.add_argument('--nphot', type=int, default=4096,
                        help='number of photons')
    parser.add_argument('--spin', type=float, default=0.9,
                        help='black hole spin')
    parser.add_argument('--rmin', type=float, default=1.6,
                        help='inner radius of domain')
    parser.add_argument('--rmax', type=float, default=50.0,
                        help='outer radius of domain')
    parser.add_argument('--nmu', type=int, default=8,
                        help='polar angles in the launch direction grid')
    parser.add_argument('--nphi', type=int, default=8,
                        help='azimuthal angles in the launch direction grid')
    parser.add_argument('--mcranks', type=int, default=8,
                        help='mpi ranks to use')
    parser.add_argument('--path', default="/home/swd8g/athena-swdavis",
                        help='path to Athena++ distribution')
    parser.add_argument('--outfile', default="conv.out",
                        help='output filename for storing convergence rate')

    args = parser.parse_args()
    main(**vars(args))
