#! /usr/bin/env python

"""
Regression checks for polarized transport, driven through mc_poltest.

mc_poltest launches a single photon with an exactly specified polarization state and
reports POLRESID: the largest departure of its coherency tensor, projected onto the
globally covariantly constant frame, from the value the problem generator seeded.  The
spacetime is flat, so the exact answer is zero and everything left is numerical.

This complements tst/montecarlo/snake_polarization, which refines the same quantity over a
free-free photon population.  Here the initial state is deterministic and known, which
makes the test sensitive to two things the population test cannot separate:

  order      GeneralPusher::AdvanceStep transports the tensor with Heun's method, so
             POLRESID must fall as stepsize**2.  It fell as stepsize**1 when the connection
             was evaluated only after the step.

  emission   The reference is taken from the tensor the problem generator seeds, before
             TransferPhotonsOnBlock rebuilds it from the handed-over Stokes parameters.  A
             rebuild that references the polarization to the wrong basis -- what happens if
             the coherency tensor is built after k has been transformed to the coordinate
             frame, or if a generator's own tensor is overwritten -- leaves a constant
             O(g_xy) offset that no amount of refinement removes, so the order checks read
             0 rather than 2.  Both bugs were reintroduced deliberately to confirm this
             test reports them; the first-order transport reads 1.

Note that the order checks, not the unsheared one, are what catch a bad emission basis.
With snake_a = 0 the coordinate and comoving frames coincide, so the ordering inside the
conversion makes no difference and the unsheared run passes either way.  What it does
pin down is the handover round trip in isolation, with no transport error on top of it.

Requires a build configured for this problem generator:

    python configure.py --prob=mc_poltest --coord=gr_user -g -mc && make
"""

# python standard modules
import argparse
import math
import os
import re
import subprocess
import sys

# Snake shear.  beta = snake_a*snake_k*cos(snake_k*x) is the only free parameter of the
# metric; the launch point sits where it is comfortably nonzero, so the connection the
# transport has to integrate is not accidentally small.
SNAKE_L = 1.0e11
SNAKE_K = 2.0 * math.pi / SNAKE_L

# The measured order has to be this close to 2 to pass.  Wide enough not to be flaky, far
# too tight to admit the first-order scheme, which reads 1.0.
ORDER_TOL = 0.05

# Floor for the unsheared run.  The reference tensor is the one the problem generator
# seeds, taken before TransferPhotonsOnBlock rebuilds it from the Stokes parameters handed
# over, so even with no connection to integrate the residual sits at the round-off of that
# round trip -- a few times 1e-16.  Keeping the reference on the generator's side is
# deliberate: it is what makes POLRESID sensitive to the handover as well as the transport.
# The tolerance is far below the 1e-8 the sheared runs produce.
FLAT_TOL = 1.0e-13


def write_athinput(path, stepsize, beta, polcirc, zeta):
    """Write one deck.  Mirrors inputs/mc/athinput.poltest."""
    snake_a = beta / SNAKE_K
    polarized = "circular" if polcirc != 0.0 else "linear"

    o = ["<job>", "problem_id = poltest", "",
         "<time>", "cfl_number = 0.3", "nlim = 1", "tlim = 1.0", "",
         "<coord>", "m = 0.0", "a = 0.0",
         "snake_a = {0!r}".format(snake_a), "snake_k = {0!r}".format(SNAKE_K), "",
         "<mesh>"]
    for d in (1, 2, 3):
        o += ["nx{0} = 8".format(d),
              "x{0}min = {1!r}".format(d, -0.5 * SNAKE_L),
              "x{0}max = {1!r}".format(d, 0.5 * SNAKE_L),
              "ix{0}_bc = periodic".format(d), "ox{0}_bc = periodic".format(d),
              "ix{0}_mc_bc = escape".format(d), "ox{0}_mc_bc = escape".format(d), ""]
    o += ["<meshblock>", "nx1 = 8", "nx2 = 8", "nx3 = 8", "",
          "<hydro>", "gamma = 1.666666666666667", "",
          "<montecarlo>",
          "mc_coord = snake", "general_pusher = true",
          "nphot = 1", "iseed = 125787",
          "scattering = none", "emission = none", "absorption = none",
          "polarized = " + polarized,
          "stepsize = {0!r}".format(stepsize),
          "varystep = true",
          # both required: they gate the comoving-to-coordinate conversion the problem
          # generator hands its emission over for
          "boosts = true", "initialize_comoving = true",
          "tmax = 1.0e36", "",
          "<problem>",
          "x0 = {0!r}".format(-0.4 * SNAKE_L), "y0 = 0.0", "z0 = 0.0",
          "zeta = {0!r}".format(zeta), "psi = 25.0",
          "polang = 30.0", "polcirc = {0!r}".format(polcirc),
          "scatopac = 0.0", "outsphere = false", ""]
    open(path, "w").write("\n".join(o))


def run(athena, workdir, stepsize, beta, polcirc=0.0, zeta=80.0):
    """Run one case and return (residual, combined output).  residual is None if the run
    failed or printed no POLRESID."""
    infile = os.path.join(workdir, "athinput.poltest")
    write_athinput(infile, stepsize, beta, polcirc, zeta)
    p = subprocess.Popen([athena, "-i", infile, "-d", workdir],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = p.communicate()[0].decode("utf-8", "replace")
    m = re.search(r"POLRESID\s+(\S+)", out)
    if p.returncode != 0 or m is None:
        return None, out
    return float(m.group(1)), out


def measure_order(athena, workdir, beta, polcirc, step0, nstep, refine):
    """Residuals over a step refinement, and the order between successive pairs."""
    steps = [step0 / float(refine)**i for i in range(nstep)]
    res = []
    for st in steps:
        r, out = run(athena, workdir, st, beta, polcirc)
        if r is None:
            raise RuntimeError("mc_poltest failed at stepsize {0!r}:\n{1}".format(st, out))
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

    # --- transport order, linear and circular
    for name, polcirc in (("linear", 0.0), ("circular", 0.6)):
        steps, res, orders = measure_order(athena, workdir, kwargs["beta"], polcirc,
                                           kwargs["step0"], kwargs["nstep"],
                                           kwargs["refine"])
        print("\n{0} polarization, beta = {1!r}".format(name, kwargs["beta"]))
        print("  stepsize      POLRESID      order")
        for i, st in enumerate(steps):
            o = "    -" if i == 0 or orders[i - 1] is None \
                else "{0:5.2f}".format(orders[i - 1])
            print("  {0:.4e}   {1:.4e}   {2}".format(st, res[i], o))
        got = orders[-1]
        ok = got is not None and abs(got - 2.0) < ORDER_TOL
        results.append(("order  {0}  (Heun -> 2)".format(name),
                        "-" if got is None else "{0:.2f}".format(got), "2.00", ok))

    # --- No shear means no connection, so the transport contributes exactly nothing and
    #     what is left is the Stokes handover round trip on its own.  This does not test
    #     the emission basis -- with no shear the frames coincide and any ordering passes --
    #     but it does pin the round trip's fidelity with nothing on top of it.
    r, out = run(athena, workdir, kwargs["step0"], 0.0)
    results.append(("flat   snake(a=0) handover",
                    "-" if r is None else "{0:.3e}".format(r),
                    "<1e-13", r is not None and r < FLAT_TOL))

    # --- the meridian basis is undefined along the frame z axis, and mc_poltest must say
    #     so rather than quietly emitting a photon with the wrong polarization
    r, out = run(athena, workdir, kwargs["step0"], kwargs["beta"], zeta=0.0)
    guarded = r is None and "comoving frame z axis" in out
    results.append(("guard  zeta = 0 rejected", "yes" if guarded else "no", "yes", guarded))

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
    print("all polarized transport checks passed" if nfail == 0
          else "{0:d} check(s) FAILED".format(nfail))
    return 1 if nfail else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", default="../../..", help="path to athena root")
    p.add_argument("--workdir", default="/tmp/mc_poltest", help="scratch run directory")
    p.add_argument("--beta", type=float, default=0.3, help="peak shear snake_a*snake_k")
    p.add_argument("--step0", type=float, default=1.0e-2, help="coarsest stepsize")
    p.add_argument("--refine", type=float, default=2.0, help="step refinement factor")
    p.add_argument("--nstep", type=int, default=4, help="number of refinements")
    sys.exit(main(**vars(p.parse_args())))
