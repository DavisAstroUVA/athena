#! /usr/bin/env python

"""
Regression checks for polarized transport in spherical-polar coordinates, driven through
mc_sphpol.  Two independent things are measured, selected by problem/scatopac.

  POLRESID (scatopac = 0)

    One photon, no scattering.  The spacetime is flat, so the coherency tensor referred to
    the global cartesian frame is exactly constant however the connection moves its stored
    spherical components; POLRESID is the departure from that and must fall as stepsize**2.

    The chart matters here in a way snake cannot show.  MCSphericalPolar::InverseTetrad
    carries the coordinate components onto the *local* orthonormal legs, which rotate along
    the ray, so the reference frame has to be rotated once more into global cartesian
    before the constancy holds.  A test that stopped at the orthonormal legs would measure
    a quantity that genuinely varies and would report order 0.

    alpha and chi choose which connection components the ray exercises: chi = 0 sweeps
    theta, chi = 90 sweeps phi, and alpha = 0 is a purely radial ray.  All three must read
    order 2.  In particular the radial ray is *not* an exact-zero control -- unlike
    snake_a = 0, which degenerates to cartesian Minkowski where the connection vanishes
    identically.  Along a radial ray the orthonormal legs do not rotate, but
    Gamma^theta_(r theta) = Gamma^phi_(r phi) = 1/r still act on the coordinate components
    through the scale factors.  It is checked at order 2 rather than against zero so that
    anyone who later "fixes" it to zero finds out here that the expectation was wrong.

  POLDEG (scatopac > 0)

    The comoving round trip at a scattering, which the transport checks above never touch.
    The photon is emitted unpolarized and Thomson-scatters exactly once through a right
    angle in the comoving frame, which leaves it fully linearly polarized.  The degree of
    polarization is invariant under both Lorentz boosts and parallel transport, so the
    value measured in the observer frame at escape must be 1 whatever the fluid velocity.

    That single number covers the whole chain the scatter sits inside: coordinate to
    comoving, the meridian basis built in the comoving frame, tensor to Stokes and back,
    and comoving to coordinate.  An error in any of them shows up as a degree of
    polarization that is not 1, and one that grows with velocity.  The residual measured
    here in fact *falls* with beta, because relativistic beaming shortens the post-scatter
    path and so accumulates less transport truncation; it is truncation, not boost error.

Requires a build configured for this problem generator:

    python configure.py --prob=mc_sphpol --coord=spherical_polar -mc && make
"""

# python standard modules
import argparse
import math
import os
import re
import subprocess
import sys

# The measured order has to be this close to 2 to pass.  Wide enough not to be flaky, far
# too tight to admit a first-order scheme.
ORDER_TOL = 0.05

# Ceiling on |P - 1| at escape.  The transport truncation behind it is ~2e-9 at the default
# stepsize, while getting the comoving frame wrong misses by O(beta), so anything in
# between separates the two cleanly.
POLDEG_TOL = 1.0e-6

# Emission point and mesh.  The launch radius sits well inside the outer boundary so the
# ray has room to sweep, and well away from the axis so the meridian basis is comfortable.
R0 = 10.0
RMIN, RMAX = 1.0, 100.0
PI = math.pi


def write_athinput(path, stepsize, alpha, chi, scatopac=0.0, velocity=0.0, polcirc=0.0,
                   nx1=32, nx2=16, nx3=16):
    """Write one deck.  Mirrors athinput.sphpol."""
    polarized = "circular" if polcirc != 0.0 else "linear"
    scattering = "user" if scatopac > 0.0 else "none"

    o = ["<job>", "problem_id = sphpol", "",
         "<time>", "cfl_number = 0.3", "nlim = 1", "tlim = 1.0", "",
         "<mesh>",
         "nx1 = {0:d}".format(nx1),
         "x1min = {0!r}".format(RMIN), "x1max = {0!r}".format(RMAX),
         "ix1_bc = outflow", "ox1_bc = outflow",
         "ix1_mc_bc = escape", "ox1_mc_bc = escape", "",
         "nx2 = {0:d}".format(nx2),
         "x2min = 0.0", "x2max = {0!r}".format(PI),
         "ix2_bc = polar", "ox2_bc = polar",
         "ix2_mc_bc = escape", "ox2_mc_bc = escape", "",
         "nx3 = {0:d}".format(nx3),
         "x3min = 0.0", "x3max = {0!r}".format(2.0 * PI),
         "ix3_bc = periodic", "ox3_bc = periodic",
         "ix3_mc_bc = periodic", "ox3_mc_bc = periodic", "",
         "<hydro>", "gamma = 1.666666666666667", "",
         "<montecarlo>",
         "general_pusher = true",
         "nphot = 1", "iseed = 12345",
         "emission = none", "absorption = none",
         "scattering = " + scattering,
         "polarized = " + polarized,
         "stepsize = {0!r}".format(stepsize),
         "varystep = true",
         # both required: they gate the comoving-to-coordinate conversion the problem
         # generator hands its emission over for
         "boosts = true", "initialize_comoving = true",
         "capmove = 2000000", "tmax = 1.0e36", "",
         "<problem>",
         "r0 = {0!r}".format(R0), "th0 = 90.0", "ph0 = 0.0",
         "alpha = {0!r}".format(alpha), "chi = {0!r}".format(chi),
         "polang = 0.0", "polcirc = {0!r}".format(polcirc),
         "scatopac = {0!r}".format(scatopac),
         "velocity = {0!r}".format(velocity),
         "rho_cgs = 1.0", "tgas_cgs = 1.0e6",
         "vel_cgs = 29979245800.0", "l_cgs = 1.0", ""]
    open(path, "w").write("\n".join(o))


def run(athena, workdir, tag, **kwargs):
    """Run one case.  Returns (POLRESID, POLDEG, nscat, combined output), each None when
    the run failed or did not print it."""
    infile = os.path.join(workdir, "athinput.sphpol")
    write_athinput(infile, **kwargs)
    p = subprocess.Popen([athena, "-i", infile, "-d", workdir],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = p.communicate()[0].decode("utf-8", "replace")
    if p.returncode != 0:
        return None, None, None, out
    mres = re.search(r"POLRESID\s+(\S+)", out)
    mdeg = re.search(r"POLDEG\s+(\S+)\s+nscat\s+(\d+)", out)
    res = float(mres.group(1)) if mres else None
    deg = float(mdeg.group(1)) if mdeg else None
    nsc = int(mdeg.group(2)) if mdeg else None
    return res, deg, nsc, out


def measure_order(athena, workdir, alpha, chi, step0, nstep, refine):
    """Residuals over a step refinement, and the order between successive pairs."""
    steps = [step0 / float(refine)**i for i in range(nstep)]
    res = []
    for st in steps:
        r, _, _, out = run(athena, workdir, "order", stepsize=st, alpha=alpha, chi=chi)
        if r is None:
            raise RuntimeError("mc_sphpol failed at stepsize {0!r}:\n{1}".format(st, out))
        res.append(r)
    orders = []
    for i in range(1, nstep):
        if res[i - 1] <= 0. or res[i] <= 0.:
            orders.append(None)
        else:
            orders.append(math.log(res[i - 1] / res[i]) / math.log(refine))
    return steps, res, orders


def main(**kwargs):

    athena = os.path.join(kwargs["path"], "bin", "athena")
    if not os.path.exists(athena):
        print("no binary at " + athena)
        return 1
    workdir = kwargs["workdir"]
    if not os.path.isdir(workdir):
        os.makedirs(workdir)

    results = []          # (label, measured, expected-text, ok)

    # --- transport order, for each family of connection components
    rays = (("theta sweep", kwargs["alpha"], 0.0),
            ("phi sweep", kwargs["alpha"], 90.0),
            ("radial", 0.0, 0.0))
    for name, alpha, chi in rays:
        steps, res, orders = measure_order(athena, workdir, alpha, chi,
                                           kwargs["step0"], kwargs["nstep"],
                                           kwargs["refine"])
        print("\ntransport, {0} (alpha = {1!r}, chi = {2!r})".format(name, alpha, chi))
        print("  stepsize      POLRESID      order")
        for i, st in enumerate(steps):
            o = "    -" if i == 0 or orders[i - 1] is None \
                else "{0:5.2f}".format(orders[i - 1])
            print("  {0:.4e}   {1:.4e}   {2}".format(st, res[i], o))
        got = orders[-1]
        ok = got is not None and abs(got - 2.0) < ORDER_TOL
        results.append(("order  {0}".format(name),
                        "-" if got is None else "{0:.2f}".format(got), "2.00", ok))

    # --- the comoving round trip at a scattering, over a range of fluid velocities
    print("\nscattering, degree of polarization at escape")
    print("  velocity       POLDEG       |P-1|   nscat")
    for beta in kwargs["beta"]:
        _, deg, nsc, out = run(athena, workdir, "scat",
                               stepsize=kwargs["step0"], alpha=kwargs["alpha"], chi=0.0,
                               scatopac=kwargs["scatopac"], velocity=beta)
        if deg is None:
            print("  {0:<8} run produced no POLDEG".format(beta))
            results.append(("poldeg beta = {0!r}".format(beta), "-", "1 +- 1e-6", False))
            continue
        err = abs(deg - 1.0)
        print("  {0:<8}   {1:.10f}   {2:.3e}   {3:d}".format(beta, deg, err, nsc))
        # Exactly one scatter is part of the claim: the right-angle result is only
        # parameter-free for a single scatter, so a second one would invalidate it
        # silently rather than loudly.
        ok = err < POLDEG_TOL and nsc == 1
        results.append(("poldeg beta = {0!r}".format(beta),
                        "{0:.2e}".format(err), "<1e-6", ok))

    print("\n" + "-" * 66)
    print("{0:<34} {1:>12} {2:>8}   {3}".format("check", "measured", "expect", "result"))
    print("-" * 66)
    nfail = 0
    for label, measured, expect, ok in results:
        if not ok:
            nfail += 1
        print("{0:<34} {1:>12} {2:>8}   {3}".format(label, measured, expect,
                                                    "PASS" if ok else "FAIL"))
    print("-" * 66)
    print("all spherical-polar polarization checks passed" if nfail == 0
          else "{0:d} check(s) FAILED".format(nfail))
    return 1 if nfail else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", default="../../..", help="path to athena root")
    p.add_argument("--workdir", default="/tmp/mc_sphpol", help="scratch run directory")
    p.add_argument("--alpha", type=float, default=30.0,
                   help="angle of the ray from radial, in degrees")
    p.add_argument("--beta", type=float, nargs="+", default=[0.0, 0.3, 0.8],
                   help="fluid velocities to run the scattering check at")
    p.add_argument("--scatopac", type=float, default=0.2,
                   help="scattering coefficient, tuned for about one optical depth")
    p.add_argument("--step0", type=float, default=1.0e-2, help="coarsest stepsize")
    p.add_argument("--refine", type=float, default=2.0, help="step refinement factor")
    p.add_argument("--nstep", type=int, default=4, help="number of refinements")
    sys.exit(main(**vars(p.parse_args())))
