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
    polarization that is not 1.  Two regimes are distinguished by refinement: at zero
    velocity the residual converges at second order and is transport truncation; at
    non-zero velocity it is frozen under refinement and is a genuine error in the
    comoving round trip.  As of this writing the velocity cases fail, and are meant to
    stay red until that is fixed -- they used to read 1e-9 only because an unpaired
    basis rotation made P degenerate, which is why a check that beats the integrator's
    own truncation should be distrusted rather than celebrated.

    What POLDEG cannot see is a wrong choice of reference axes: the degree of
    polarization is invariant under rotating them.  The next two checks cover that.

  REFRESID (scatopac = 0, generic geometry)

    The Stokes parameters the outputs carry, against the same coherency tensor decomposed
    by hand in the global cartesian frame against the meridian of the global z axis.
    Exact algebra on both sides, so the tolerance is round-off.  The geometry is
    deliberately generic -- off-axis colatitude, non-zero azimuth, a ray out of the
    meridional plane, a polarization angle between the axes -- because the default ray
    lies in a symmetry plane where Q = -1, U = 0 would come out of almost any referencing.

  list wavevector basis (same runs)

    The photon list's wavevector columns against the escape direction the problem
    generator computes independently, and the header's basis line.  The list writes k
    on the global cartesian legs, the frame whose z axis Q and U are referenced to, so
    that a reader can rebuild the meridian plane from the file alone.

Requires a build configured for this problem generator:

    python configure.py --prob=mc_sphpol --coord=spherical_polar -mc && make
"""

# python standard modules
import argparse
import glob
import math
import os
import re
import subprocess
import sys

# Athena++ modules
import athena_mc as mcspec

# The measured order has to be this close to 2 to pass.  Wide enough not to be flaky, far
# too tight to admit a first-order scheme.
ORDER_TOL = 0.05

# Ceiling on |P - 1| at escape.  At zero velocity the residual is pure transport
# truncation and converges at exactly second order -- measured at 5.3e-6, 1.3e-6,
# 3.3e-7, 8.2e-8 over stepsizes 1e-2, 5e-3, 2.5e-3, 1.25e-3 -- so ~2e-5 at the default
# stepsize of this deck is the honest floor.  A wrong comoving frame misses by O(beta),
# 0.12 at beta = 0.3, and does not move under refinement, so 1e-4 separates the two by
# three orders of magnitude.  The earlier 1e-6 was met only while a basis error made P
# degenerate and insensitive to truncation; a check that reads better than the
# integrator can deliver is the thing to distrust.
POLDEG_TOL = 1.0e-4

# Ceiling on the disagreement between the Stokes parameters the outputs carry and the same
# tensor decomposed by hand in the global cartesian frame.  Both are exact algebra on the
# same transported tensor, with no truncation in between, so this is a round-off tolerance
# rather than a physics one.
REFRESID_TOL = 1.0e-12

# Emission point and mesh.  The launch radius sits well inside the outer boundary so the
# ray has room to sweep, and well away from the axis so the meridian basis is comfortable.
R0 = 10.0
RMIN, RMAX = 1.0, 100.0
PI = math.pi


def write_athinput(path, stepsize, alpha, chi, scatopac=0.0, velocity=0.0, polcirc=0.0,
                   th0=90.0, ph0=0.0, polang=0.0, nx1=32, nx2=16, nx3=16):
    """Write one deck.  Mirrors athinput.sphpol."""
    polarized = "circular" if polcirc != 0.0 else "linear"
    scattering = "user" if scatopac > 0.0 else "none"

    o = ["<job>", "problem_id = sphpol", "",
         # the photon list is what the wavevector-basis check reads back
         "<output1>", "file_type = phlist", "dt = 1.0e-20", "",
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
         "r0 = {0!r}".format(R0),
         "th0 = {0!r}".format(th0), "ph0 = {0!r}".format(ph0),
         "alpha = {0!r}".format(alpha), "chi = {0!r}".format(chi),
         "polang = {0!r}".format(polang), "polcirc = {0!r}".format(polcirc),
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


def run_refresid(athena, workdir, **kwargs):
    """REFRESID and the escape direction KCART for one configuration.  Either is None
    when the run did not print it."""
    infile = os.path.join(workdir, "athinput.sphpol")
    write_athinput(infile, **kwargs)
    # A stale list from the previous configuration would be read back as this one's
    for old in glob.glob(os.path.join(workdir, "sphpol.out1*.list")):
        os.remove(old)
    p = subprocess.Popen([athena, "-i", infile, "-d", workdir],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = p.communicate()[0].decode("utf-8", "replace")
    if p.returncode != 0:
        return None, None, out
    m = re.search(r"REFRESID\s+(\S+)", out)
    mk = re.search(r"KCART\s+(\S+)\s+(\S+)\s+(\S+)", out)
    kcart = tuple(float(mk.group(i)) for i in (1, 2, 3)) if mk else None
    return (float(m.group(1)) if m else None), kcart, out


def list_wavevector(workdir):
    """The wavevector columns of the one photon in the list the run just wrote, and the
    basis the header declares for them.  (None, None) if there is no list."""
    files = sorted(glob.glob(os.path.join(workdir, "sphpol.out1*.list")))
    if not files:
        return None, None
    reader = mcspec.read_list_generator(files[-1])
    header = next(reader)["header"]
    for result in reader:
        if result["chunk"] is None or not result["length"]:
            continue
        ph = header.copy()
        ph["list"] = result["chunk"]
        ph["length"] = result["length"]
        p = mcspec.Photons(ph)
        return (float(p.k1[0]), float(p.k2[0]), float(p.k3[0])), header.get("basis")
    return None, header.get("basis")


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

    # --- how the Stokes parameters are referenced at output
    #
    # CoherencyToObserverStokes sets sqp and sup from the transported tensor, and nothing
    # else here can see whether it got the axes right: POLRESID reads the tensor and never
    # looks at the Stokes parameters, and the degree of polarization is invariant under a
    # rotation of the reference axes, so the scattering check is blind to it too.
    #
    # The configurations below are deliberately generic.  The default ray lies in a plane
    # containing the z axis with the polarization along a principal direction, which forces
    # Q = -1 and U = 0 by symmetry and would be reproduced by almost any referencing,
    # right or wrong.  Off-axis theta, a non-zero azimuth, a ray that sweeps out of the
    # meridional plane and a polarization angle between the axes give a Q and U that pin
    # the convention down.
    # The same runs also check the photon list's wavevector columns.  Those are written
    # on the global cartesian legs, the frame whose z axis the Stokes parameters are
    # referenced to, and the header says so with basis=cartesian; the problem generator
    # prints the direction it computed independently as KCART.  Wavevector and Stokes
    # parameters in one frame is what lets a reader rebuild the meridian plane from the
    # file alone, which is the guarantee this row enforces.
    print("\nStokes referencing at output, against the tensor in the global cartesian frame,")
    print("and the list's wavevector against the same direction")
    print("  th0    ph0   alpha  chi   polang     REFRESID     |k_list - k|   basis")
    refcases = ((90.0, 0.0, 30.0, 0.0, 0.0),
                (57.0, 40.0, 35.0, 63.0, 27.0),
                (125.0, 200.0, 55.0, 37.0, 71.0),
                (70.0, 300.0, 80.0, 20.0, 15.0),
                (40.0, 95.0, 25.0, 140.0, 50.0))
    worst, kworst = 0.0, 0.0
    nref, nk = 0, 0
    bases = set()
    for th0, ph0, al, ch, pa in refcases:
        r, kcart, out = run_refresid(athena, workdir, stepsize=kwargs["step0"],
                                     alpha=al, chi=ch, th0=th0, ph0=ph0, polang=pa)
        klist, basis = list_wavevector(workdir)
        bases.add(basis)
        if r is None:
            print("  {0:<6} {1:<5} {2:<6} {3:<5} {4:<8}  no REFRESID".format(
                th0, ph0, al, ch, pa))
            continue
        nref += 1
        worst = max(worst, r)
        if kcart is not None and klist is not None:
            kn = math.sqrt(sum(c*c for c in klist))
            dk = max(abs(klist[i]/kn - kcart[i]) for i in range(3))
            nk += 1
            kworst = max(kworst, dk)
            ktxt = "{0:.4e}".format(dk)
        else:
            ktxt = "no list" if klist is None else "no KCART"
        print("  {0:<6} {1:<5} {2:<6} {3:<5} {4:<8}  {5:.4e}   {6:<12}   {7}".format(
            th0, ph0, al, ch, pa, r, ktxt, basis))
    ok = (nref == len(refcases)) and worst < REFRESID_TOL
    results.append(("stokes referencing",
                    "{0:.1e}".format(worst) if nref else "-",
                    "<{0:.0e}".format(REFRESID_TOL), ok))
    okk = (nk == len(refcases)) and kworst < REFRESID_TOL and bases == {"cartesian"}
    results.append(("list wavevector basis",
                    "{0:.1e}".format(kworst) if nk else "-",
                    "<{0:.0e}".format(REFRESID_TOL), okk))

    # --- the comoving round trip at a scattering, over a range of fluid velocities
    print("\nscattering, degree of polarization at escape")
    print("  velocity       POLDEG       |P-1|   nscat")
    for beta in kwargs["beta"]:
        _, deg, nsc, out = run(athena, workdir, "scat",
                               stepsize=kwargs["step0"], alpha=kwargs["alpha"], chi=0.0,
                               scatopac=kwargs["scatopac"], velocity=beta)
        if deg is None:
            print("  {0:<8} run produced no POLDEG".format(beta))
            results.append(("poldeg beta = {0!r}".format(beta), "-",
                            "<{0:.0e}".format(POLDEG_TOL), False))
            continue
        err = abs(deg - 1.0)
        print("  {0:<8}   {1:.10f}   {2:.3e}   {3:d}".format(beta, deg, err, nsc))
        # Exactly one scatter is part of the claim: the right-angle result is only
        # parameter-free for a single scatter, so a second one would invalidate it
        # silently rather than loudly.
        ok = err < POLDEG_TOL and nsc == 1
        results.append(("poldeg beta = {0!r}".format(beta),
                        "{0:.2e}".format(err), "<{0:.0e}".format(POLDEG_TOL), ok))

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
