#! /usr/bin/env python

"""
Emergent polarization from a vertically stratified disk atmosphere in spherical-polar
coordinates, compared against the plane-parallel Feautrier solution.

This is the end-to-end counterpart to tst/montecarlo/spherical_polarization, which checks
transport and the comoving round trip pointwise on single photons.  Here a full population
scatters its way out of an optically thick atmosphere and the emergent polarization is
compared against an independent solution of the same transfer problem.

Geometry.  The density depends only on z = r cos(theta), so every cylindrical radius
carries the same plane-parallel atmosphere, stratified from tau = taumax at the midplane to
taumin at z = zmax.  The midplane (outer_x2) absorbs and the upper surface (inner_x2)
escapes, which is the spherical analogue of the one-sided slab in
tst/montecarlo/thomson_polarized_spectrum: same optical depths, same emission, same
Feautrier reference, different chart.

Angles.  The spectrum is binned with polar_axis = true, so mu is measured against the
global z axis rather than against whatever boundary the photon crossed.  That is the
physically meaningful angle for a disk: mu = 1 looks down the rotation axis and mu -> 0
grazes the disk edge-on.  Chandrasekhar's result for a semi-infinite electron-scattering
atmosphere is that the emergent polarization vanishes at mu = 1 and rises monotonically
toward mu = 0, reaching 11.7 per cent, so the sign and slope of the mu dependence are as
much of a check as the amplitude.

What this test can and cannot show.  Unlike the pointwise tests it is statistical, so it
converges as 1/sqrt(nphot) and needs a tolerance rather than an exact answer.  It is also
not an exact realization of the plane-parallel problem: a cell of angular size dtheta spans
dz = r dtheta, so the vertical resolution degrades with radius, and the domain reaches
above the atmosphere at large radius.  Both are controlled by keeping the radial range
narrow relative to its midpoint.  Treat a disagreement here as a reason to look, not as
proof of a bug, and use the spherical_polarization tests to localize anything it finds.

Requires a build configured for this problem generator:

    python configure.py --prob=mc_isoth --coord=spherical_polar -mc && make

PYTHONPATH must include vis/python/montecarlo for athena_mc.
"""

# python standard modules
import argparse
import math
import os
import subprocess
import sys

import numpy as np

# Athena++ modules
import athena_mc as mcspec
import feautrier as feaut

# Atmosphere, matched to tst/montecarlo/thomson_polarized_spectrum so the same Feautrier
# solution describes both.  ZMAX is the slab thickness there.
TEMP = 1.0e5
TAUMIN = 1.0e-3
TAUMAX = 1.0e4
ZMAX = 1.0e11

# Disk geometry.  RIN is chosen so that zmax/RIN sets the opening angle, and the radial
# range is kept narrow so that the vertical cell thickness r*dtheta varies little across
# the domain.  ROUT/RIN = 2 leaves the vertical resolution varying by a factor of two.
RIN = 1.0e12
ROUT = 2.0e12

EVERG = 1.6021772e-12
HPLANCK = 6.62607015e-27


def theta_min():
    """Polar angle of the upper surface: the atmosphere reaches z = ZMAX at r = RIN."""
    return math.acos(ZMAX / RIN)


def write_athinput(path, iseed, nphot, nen, emin, emax, ncth, nphi,
                   nx1=16, nx2=64, nx3=16, general_pusher=False, stepsize=0.5):
    """Write one deck.  Mirrors the cartesian slab test, in spherical-polar."""
    o = ["<comment>",
         "problem   = vertically stratified disk atmosphere",
         "configure = --prob=mc_isoth --coord=spherical_polar -mc", "",
         "<job>", "problem_id = mcdisk", "",
         "<output1>",
         "file_type  = spec",
         # photons that leave through the upper surface; the midplane absorbs
         "face       = inner_x2",
         # mu against the global z axis, not the boundary normal
         "polar_axis = true",
         "ne         = {0:d}".format(nen),
         "emin       = {0:e}".format(emin),
         "emax       = {0:e}".format(emax),
         "ncth       = {0:d}".format(ncth),
         "nphi       = {0:d}".format(nphi), "",
         "<time>", "cfl_number = 0.1", "nlim = 1", "tlim = 1.0", "",
         "<mesh>",
         "nx1        = {0:d}".format(nx1),
         "x1min      = {0:e}".format(RIN),
         "x1max      = {0:e}".format(ROUT),
         "ix1_bc     = outflow", "ox1_bc     = outflow",
         "ix1_mc_bc  = escape", "ox1_mc_bc  = escape", "",
         "nx2        = {0:d}".format(nx2),
         # theta runs from the upper surface down to the midplane
         "x2min      = {0!r}".format(theta_min()),
         "x2max      = {0!r}".format(0.5 * math.pi),
         "ix2_bc     = outflow", "ox2_bc     = outflow",
         "ix2_mc_bc  = escape", "ox2_mc_bc  = absorb", "",
         "nx3        = {0:d}".format(nx3),
         "x3min      = 0.0", "x3max      = {0!r}".format(2.0 * math.pi),
         "ix3_bc     = periodic", "ox3_bc     = periodic",
         "ix3_mc_bc  = periodic", "ox3_mc_bc  = periodic", "",
         "<hydro>", "gamma = 1.666666666666667", "iso_sound_speed = 1.0", "",
         "<montecarlo>",
         "nphot      = {0:d}".format(nphot),
         "iseed      = {0:d}".format(iseed),
         "scattering = thomson",
         "emission   = freefree",
         "absorption = freefree",
         "polarized  = linear"]
    if general_pusher:
        # The geodesic integrator, which in flat spacetime should reproduce the legacy
        # pusher.  Note that it also changes what the stored Stokes parameters are
        # referenced to: CoherencyToObserverStokes runs only under the general pusher and
        # refers them to the normal observer's meridian, whose third spatial leg here is
        # e_phi rather than the global z axis.
        o += ["general_pusher = true",
              "stepsize   = {0!r}".format(stepsize),
              "varystep   = true",
              "capmove    = 2000000"]
    o += ["",
         "<problem>",
         # the non-radial branch of the spherical setup in mc_isoth
         "radial     = false",
         "zmax       = {0:e}".format(ZMAX),
         "temp       = {0:e}".format(TEMP),
         "taumin     = {0:e}".format(TAUMIN),
         "taumax     = {0:e}".format(TAUMAX),
         "emin       = {0:e}".format(emin),
         "emax       = {0:e}".format(emax), ""]
    open(path, "w").write("\n".join(o))


def run(athena, workdir, **kwargs):
    """Run one case; returns the combined output."""
    infile = os.path.join(workdir, "athinput.mcdisk")
    write_athinput(infile, **kwargs)
    p = subprocess.Popen([athena, "-i", infile, "-d", workdir],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = p.communicate()[0].decode("utf-8", "replace")
    if p.returncode != 0:
        raise RuntimeError("mc_isoth failed:\n" + out[-4000:])
    return out


def interp_feaut(mu0, mu, varin):
    """Interpolate the Feautrier solution in angle, as the cartesian test does."""
    nnu = len(varin[:, 0])
    varout = np.zeros(nnu)
    for i in range(nnu):
        varout[i] = np.interp(mu0, mu, varin[i, :])
    return varout


def polarization_norm(spectrum, muf, polf):
    """Mean |P(mc) - P(feautrier)| over every (mu, energy) bin that collected photons.

    This is the norm convergence_dd.py accumulates, and the reason for it is that at any
    fixed photon count the per-bin scatter is large: the quantity that carries the signal
    is not the value of the norm but the rate at which it falls.  Also returns the
    intensity-weighted band average per mu bin, which is far less noisy and is what the
    monotonicity check uses.
    """
    intensity = spectrum["intensity"]
    intens = np.sum(intensity[0, :, :, :], axis=0) / float(spectrum["nphi"])
    qpol = np.sum(intensity[1, :, :, :], axis=0) / float(spectrum["nphi"])
    empty = intens == 0.0
    qpol = np.divide(qpol, intens, out=np.zeros_like(qpol), where=~empty)
    mumid = 0.5 * (spectrum["mufaces"][1:] + spectrum["mufaces"][:-1])

    total, npol = 0.0, 0
    band = []
    for j in range(spectrum["nmu"]):
        pint = interp_feaut(mumid[j], muf, polf)
        for k in range(spectrum["nx"]):
            if empty[j, k]:
                continue
            total += abs(qpol[j, k] - pint[k])
            npol += 1
        w = intens[j, :]
        if w.sum() > 0.0:
            band.append((mumid[j],
                         float(np.sum(qpol[j, :] * w) / w.sum()),
                         float(np.sum(pint * w) / w.sum())))
    return (total / npol if npol else float("nan")), band, int(empty.sum())


def main(**kwargs):

    athena = os.path.join(kwargs["path"], "bin", "athena")
    if not os.path.exists(athena):
        print("no binary at " + athena)
        return 1
    workdir = kwargs["workdir"]
    if not os.path.isdir(workdir):
        os.makedirs(workdir)
    cwd = os.getcwd()

    emin, emax = kwargs["emin"], kwargs["emax"]
    counts = [int(kwargs["nphot"] * kwargs["refine"]**i)
              for i in range(kwargs["nstep"])]

    muf = polf = None
    norms, bands = [], []
    for n in counts:
        run(athena, workdir, iseed=kwargs["iseed"], nphot=n,
            nen=kwargs["nen"], emin=emin, emax=emax,
            ncth=kwargs["ncth"], nphi=kwargs["nphi"],
            nx1=kwargs["nx1"], nx2=kwargs["nx2"], nx3=kwargs["nx3"],
            general_pusher=kwargs["general_pusher"],
            stepsize=kwargs["stepsize"])
        spectrum = mcspec.read_spectrum(
            os.path.join(workdir, "mcdisk.out1.00000.spec"))

        if muf is None:
            ffile = os.path.join(workdir, "feautrier.out")
            if not os.path.exists(ffile):
                print("Computing feautrier transfer")
                xfaces = spectrum["xfaces"]
                nu = 0.5 * (xfaces[1:] + xfaces[:-1]) * EVERG / HPLANCK
                os.chdir(workdir)
                feaut.transfer(tconst=TEMP, trange=[TAUMIN, TAUMAX], l0=ZMAX,
                               outfile="feautrier.out", nu=nu)
                os.chdir(cwd)
            nuf, muf, intensf, polf = feaut.read_feautrier(ffile)

        nrm, band, nempty = polarization_norm(spectrum, muf, polf)
        norms.append(nrm)
        bands.append(band)
        print("  nphot = {0:>10d}   mean |dP| = {1:.5f}   ({2:d} empty bins)".format(
            n, nrm, nempty))

    print("\nconvergence of the polarization norm")
    print("  nphot        mean |dP|     order")
    orders = []
    for i, n in enumerate(counts):
        if i == 0:
            print("  {0:>10d}   {1:.5f}       -".format(n, norms[i]))
            continue
        o = math.log(norms[i - 1] / norms[i]) / math.log(kwargs["refine"])
        orders.append(o)
        print("  {0:>10d}   {1:.5f}    {2:5.2f}".format(n, norms[i], o))

    print("\nemergent polarization at the largest photon count")
    print("   mu       P(mc)      P(feautrier)")
    for mu, pmc, pfe in bands[-1]:
        print("  {0:5.3f}   {1:9.5f}   {2:9.5f}".format(mu, pmc, pfe))

    # Monte Carlo noise falls as 1/sqrt(N), so the norm should too once it is
    # noise-dominated.  A norm that stalls means a systematic difference the photon
    # count cannot remove, which is what a genuine geometry or basis error looks like.
    got = sum(orders) / len(orders) if orders else float("nan")
    band = bands[-1]
    lo = min(band, key=lambda r: r[0])
    hi = max(band, key=lambda r: r[0])

    results = [
        ("norm falls as 1/sqrt(nphot)",
         "{0:.2f}".format(got), "0.50", abs(got - 0.5) < kwargs["order_tol"]),
        ("P rises toward mu = 0",
         "yes" if lo[1] > hi[1] else "no", "yes", lo[1] > hi[1]),
    ]

    print("\n" + "-" * 62)
    print("{0:<30} {1:>12} {2:>8}   {3}".format("check", "measured", "expect", "result"))
    print("-" * 62)
    nfail = 0
    for label, measured, expect, ok in results:
        if not ok:
            nfail += 1
        print("{0:<30} {1:>12} {2:>8}   {3}".format(label, measured, expect,
                                                    "PASS" if ok else "FAIL"))
    print("-" * 62)
    print("disk atmosphere polarization converges to feautrier" if nfail == 0
          else "{0:d} check(s) FAILED".format(nfail))
    return 1 if nfail else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", default="../../..", help="path to athena root")
    p.add_argument("--workdir", default="/tmp/mc_disk_atm", help="scratch run directory")
    p.add_argument("--nphot", type=int, default=1000000,
                   help="photon samples at the coarsest point of the ladder")
    p.add_argument("--nstep", type=int, default=3, help="points in the ladder")
    p.add_argument("--general-pusher", dest="general_pusher", action="store_true",
                   help="use the geodesic pusher instead of the legacy spherical one")
    p.add_argument("--stepsize", type=float, default=0.5,
                   help="general pusher step, as a fraction of a cell crossing")
    p.add_argument("--nx1", type=int, default=16, help="radial cells")
    p.add_argument("--nx2", type=int, default=64,
                   help="polar cells; the vertical resolution of the atmosphere")
    p.add_argument("--nx3", type=int, default=16, help="azimuthal cells")
    p.add_argument("--refine", type=float, default=4.0,
                   help="photon-count refinement factor between points")
    p.add_argument("--order-tol", dest="order_tol", type=float, default=0.15,
                   help="how close the measured convergence order must be to 0.5")
    p.add_argument("--iseed", type=int, default=1300431, help="random seed")
    p.add_argument("--nen", type=int, default=32, help="energy bins")
    p.add_argument("--emin", type=float, default=1.0, help="lowest photon energy, eV")
    p.add_argument("--emax", type=float, default=1.0e2, help="highest photon energy, eV")
    p.add_argument("--ncth", type=int, default=8, help="mu bins")
    p.add_argument("--nphi", type=int, default=8, help="phi bins")
    sys.exit(main(**vars(p.parse_args())))
