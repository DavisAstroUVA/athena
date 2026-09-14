#! /usr/bin/env python

"""
Emergent polarization from a vertically stratified disk atmosphere in spherical-polar
coordinates, compared against the plane-parallel Feautrier solution.

This is the end-to-end counterpart to tst/montecarlo/spherical_polarization, which checks
transport, the comoving round trip and the output referencing pointwise on single photons.
Here a full population scatters its way out of an optically thick atmosphere and the
emergent polarization is compared against an independent solution of the same transfer
problem.  It runs either pusher, so it is also the one place the legacy spherical pusher and
the general pusher are held to the same physical answer.

Geometry.  The density depends only on z = r cos(theta), so every cylindrical radius carries
the same plane-parallel atmosphere, stratified from tau = taumax at the midplane to taumin at
z = zmax and continuing exponentially above.  The midplane (outer_x2) absorbs, as the slab's
floor does in tst/montecarlo/thomson_polarized_spectrum: same optical depths, same emission,
same Feautrier reference, different chart.  theta runs all the way from the pole to the
midplane, so the only boundaries the slab does not have are the two radial walls; there is no
artificial cone above the atmosphere, and the region between the atmosphere's top and the
walls is optically thin.

The two things a spherical grid gets wrong about a planar problem, and what is done about
each:

  The walls.  A photon crossing either wall *inside* the atmosphere, below zmax, has not
  emerged from the surface: it has leaked out sideways, having sampled only part of the
  column it should have.  Those photons must be dropped.  A photon crossing a wall *above*
  zmax has emerged and merely drifted out radially through the thin region; those are
  legitimate and must be kept -- and they are preferentially the grazing ones, exactly where
  the polarization is largest, so discarding them (which binning on a boundary face did)
  biases the answer where it matters most.  A single criterion does both: every escaping
  photon is binned if it left at z > zmax and dropped otherwise, whichever wall it crossed.
  The photon list carries the escape position, so this is a mask on the list.

  The inner hole.  Dropping side leaks removes them from the spectrum but not from the
  physics: in a real disk a photon crossing the inner radius at z < zmax would continue
  through disk material and could scatter back out.  So the hole also has to be small.  The
  inner radius is twice zmax, which makes the hole half a percent of the emitting area.

Resolution.  Density is laid down from the cell-center z, so what has to resolve the scale
height (6.2e9 cm) is the vertical cell size r dtheta inside the atmosphere, and the
atmosphere sits in the last few degrees above the midplane at the outer radius while
spanning thirty at the inner one.  Uniform theta cells over the full quadrant would waste
most of them near the pole, so the spacing is geometric, x2rat < 1, shrinking toward the
midplane.  The driver prints the vertical cell size at the atmosphere's top and at the
midplane for the nx2 and x2rat it was given; with the defaults it is about a fifth of a
scale height at the top and a twelfth at the midplane at the outer radius, and finer inside.

Angles.  mu is measured against the global z axis, the meaningful angle for a disk: mu = 1
looks down the rotation axis and mu -> 0 grazes it edge-on.  Chandrasekhar's result for a
semi-infinite electron-scattering atmosphere is that the emergent polarization vanishes at
mu = 1 and rises monotonically toward mu = 0, perpendicular to the meridian plane, which in
the code's convention (Q > 0 along the meridian direction) is Q < 0.  The sign and the slope
in mu are as much of a check as the amplitude.

Emission sampling.  Samples are placed uniformly over cells with the emissivity carried in
the weight, the code's default, and that is the right choice here: emission goes as density
squared, so the escapers come from a thin skin holding a negligible share of it, and
placing samples in proportion to emission (equal_weight = true) sends nearly all of them to
the absorbing midplane -- 38 escapers out of 4 million when measured.  The price of the
default is that escaping weights span decades, so the effective sample size is well below
the escape count and any *fraction of escaping weight* fluctuates from run to run (the
side-leak fraction read 8, 15 and 12 percent on identical decks).  Fractions by photon
count are reported alongside for that reason, and the convergence of the norm, not any
single fraction, is what the test gates on.

The general pusher and the axis.  With theta reaching the pole, the general pusher loses
about 0.02 percent of its photons where the legacy pusher loses none: the Monte Carlo polar
boundary (Polar in mcbvals.cpp) is a stub that destroys whatever reaches it, and within a
few hundredths of a degree of the axis the geodesic integrator runs away (photons have been
caught at theta of thousands of degrees and radii far outside the domain).  A runaway photon
that happens to escape instead of being caught carries its full weight and a meaningless
direction, and one deep-midplane photon of that kind is enough to swamp a pooled band --
the general pusher's grazing sign check can come out under-powered for that reason while
its norm converges normally and agrees with the legacy pusher's.  The escaper weight
distributions of the two pushers are otherwise identical.  This is a defect of the general
pusher at the axis, not of this test's geometry, and it applies to the Kerr-Schild decks,
which also run theta from 0 to pi.

What this test can and cannot show.  It is statistical, so it converges as 1/sqrt(nphot) and
needs a tolerance rather than an exact answer; the quantity that carries the signal is the
rate at which the norm falls, not its value at any one count.  Treat a disagreement here as a
reason to look, and use the spherical_polarization tests to localize anything it finds.

Requires a build configured for this problem generator:

    python configure.py --prob=mc_isoth --coord=spherical_polar -mc -mpi && make

PYTHONPATH must include vis/python/montecarlo for athena_mc and feautrier.
"""

# python standard modules
import argparse
import glob
import math
import os
import subprocess
import sys

import numpy as np

# Athena++ modules
import athena_mc as mcspec
import feautrier as feaut

# Atmosphere, matched to tst/montecarlo/thomson_polarized_spectrum so the same Feautrier
# solution describes both.  ZMAX is the slab thickness there; L0 its density scale height.
TEMP = 1.0e5
TAUMIN = 1.0e-3
TAUMAX = 1.0e4
ZMAX = 1.0e11
L0 = 6.2e9

# Disk geometry; see the module docstring for why these values.
RIN = 2.0e11
ROUT = 1.6e12

# Energy axis, eV, matching the slab test
EMIN_EV, EMAX_EV, NEN = 1.0, 100.0, 32

EVERG = 1.6021772e-12
HPLANCK = 6.62607015e-27


def theta_cells(nx2, x2rat):
    """Cell widths of the geometric theta grid over [0, pi/2]: at the pole, at the
    midplane, and at the atmosphere's top (z = zmax) at radius r, for the resolution
    report.  Athena++ makes consecutive cells differ by the factor x2rat, so the widths are
    d1, d1*q, ..., d1*q**(n-1) with d1 (1 - q**n)/(1 - q) = pi/2."""
    q, n = x2rat, nx2
    d1 = 0.5 * math.pi * (1.0 - q) / (1.0 - q**n) if q != 1.0 else 0.5 * math.pi / n
    dmid = d1 * q**(n - 1)

    def width_at(theta):
        # the cell containing theta: partial sums of the widths from the pole
        s, k = 0.0, 0
        while k < n - 1 and s + d1 * q**k < theta:
            s += d1 * q**k
            k += 1
        return d1 * q**k
    return d1, dmid, width_at


def write_athinput(path, iseed, nphot, nx1=32, nx2=512, nx3=8, x2rat=0.993,
                   mb1=8, mb2=64, mb3=8, general_pusher=False, stepsize=0.5,
                   equal_weight=False):
    """Write one deck.  Mirrors the cartesian slab test, in spherical-polar, with a photon
    list as the output the driver bins."""
    o = ["<comment>",
         "problem   = vertically stratified disk atmosphere",
         "configure = --prob=mc_isoth --coord=spherical_polar -mc -mpi", "",
         "<job>", "problem_id = mcdisk", "",
         # the list carries the escape position, which the height mask needs
         "<output1>", "file_type = phlist", "",
         "<time>", "cfl_number = 0.1", "nlim = 1", "tlim = 1.0", "",
         "<mesh>",
         "nx1        = {0:d}".format(nx1),
         "x1min      = {0:e}".format(RIN),
         "x1max      = {0:e}".format(ROUT),
         "ix1_bc     = outflow", "ox1_bc     = outflow",
         "ix1_mc_bc  = escape", "ox1_mc_bc  = escape", "",
         # from the pole to the midplane, cells shrinking toward the midplane
         "nx2        = {0:d}".format(nx2),
         "x2min      = 0.0",
         "x2max      = {0!r}".format(0.5 * math.pi),
         "x2rat      = {0!r}".format(x2rat),
         "ix2_bc     = polar", "ox2_bc     = outflow",
         "ix2_mc_bc  = polar", "ox2_mc_bc  = absorb", "",
         "nx3        = {0:d}".format(nx3),
         "x3min      = 0.0", "x3max      = {0!r}".format(2.0 * math.pi),
         "ix3_bc     = periodic", "ox3_bc     = periodic",
         "ix3_mc_bc  = periodic", "ox3_mc_bc  = periodic", "",
         "<meshblock>",
         "nx1 = {0:d}".format(mb1), "nx2 = {0:d}".format(mb2), "nx3 = {0:d}".format(mb3), "",
         "<hydro>", "gamma = 1.666666666666667", "iso_sound_speed = 1.0", "",
         "<montecarlo>",
         "nphot      = {0:d}".format(nphot),
         "iseed      = {0:d}".format(iseed),
         "scattering = thomson",
         "emission   = freefree",
         "absorption = freefree",
         "polarized  = linear",
         # How emission samples are placed.  false (the code default) picks the cell
         # uniformly at random and carries the emissivity in the weight, which is the right
         # importance sampling for emergent light here: it oversamples the thin skin the
         # escapers come from.  true places samples in proportion to emission, and emission
         # goes as density squared, so nearly every sample starts at the absorbing midplane:
         # measured, 38 escapers out of 4 million.  The option is kept for that record.
         "equal_weight = {0}".format("true" if equal_weight else "false")]
    if general_pusher:
        o += ["general_pusher = true",
              "stepsize   = {0!r}".format(stepsize),
              "varystep   = true",
              "capmove    = 2000000"]
    o += ["",
         "<problem>",
         # the z-stratified branch of the spherical setup in mc_isoth
         "radial     = false",
         "zmax       = {0:e}".format(ZMAX),
         "temp       = {0:e}".format(TEMP),
         "taumin     = {0:e}".format(TAUMIN),
         "taumax     = {0:e}".format(TAUMAX),
         "emin       = {0:e}".format(EMIN_EV),
         "emax       = {0:e}".format(EMAX_EV), ""]
    open(path, "w").write("\n".join(o))


def run(athena, workdir, mcranks, **kwargs):
    """Run one case; returns the combined output."""
    infile = os.path.join(workdir, "athinput.mcdisk")
    write_athinput(infile, **kwargs)
    for old in glob.glob(os.path.join(workdir, "mcdisk.out1*.list")):
        os.remove(old)
    cmd = [athena, "-i", infile, "-d", workdir]
    if mcranks > 1:
        cmd = ["mpirun", "-np", str(mcranks)] + cmd
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = p.communicate()[0].decode("utf-8", "replace")
    if p.returncode != 0:
        raise RuntimeError("mc_isoth failed:\n" + out[-4000:])
    return out


def above_zmax(phots):
    """The mask make_spectrum takes: True for photons to drop.  A photon that left the
    domain at z <= zmax crossed a wall inside the atmosphere rather than emerging from its
    top.  screen.py carries the same function for make_spectrum.py --screen."""
    z = phots.x1 * np.cos(phots.x2)
    return z <= ZMAX * (1.0 - 1.0e-6)


def bin_lists(workdir, ncth, nphi, keep_lists):
    """Stream every rank's list through make_spectrum with the height mask, accumulate, and
    delete the lists.  Also returns an accounting of where the escaping weight went."""
    files = sorted(glob.glob(os.path.join(workdir, "mcdisk.out1*.list")))
    if not files:
        raise RuntimeError("no photon list written in " + workdir)
    spectrum = {}
    # Dropped weight is split by wall because only the inner part is a physical bias: a
    # photon lost through the inner wall below zmax would, in a real disk, have gone on
    # through disk material and could have scattered back out.  A photon lost through the
    # outer wall below zmax is merely a photon the finite domain could not follow, and
    # dropping it leaves the polarization at each mu unbiased, since that depends on the
    # emergent angle and the column, not on where in radius the photon was born.
    acct = dict(total=0.0, dropped_inner=0.0, dropped_outer=0.0,
                kept_inner=0.0, kept_outer=0.0, kept_other=0.0)
    for fn in files:
        reader = mcspec.read_list_generator(fn)
        header = next(reader)["header"]
        for result in reader:
            ph = header.copy()
            ph["list"] = result["chunk"]
            ph["length"] = result["length"]
            phots = mcspec.Photons(ph)
            mask = above_zmax(phots)
            w = phots.weight * phots.energy
            r = phots.x1
            inner = r <= RIN * (1.0 + 1.0e-6)
            outer = r >= ROUT * (1.0 - 1.0e-6)
            acct["total"] += float(w.sum())
            acct["dropped_inner"] += float(w[mask & inner].sum())
            acct["dropped_outer"] += float(w[mask & ~inner].sum())
            acct["kept_inner"] += float(w[~mask & inner].sum())
            acct["kept_outer"] += float(w[~mask & outer].sum())
            # with the domain reaching the pole every escape is through a wall; anything
            # else here would mean a photon left somewhere it should not have
            acct["kept_other"] += float(w[~mask & ~inner & ~outer].sum())
            # The same by photon count.  Escaping weights span decades, since the sampling
            # carries the emissivity in the weight, so fractions of weight fluctuate from
            # run to run while fractions of count do not; both are worth seeing.
            acct["n_total"] = acct.get("n_total", 0) + int(len(w))
            acct["n_dropped"] = acct.get("n_dropped", 0) + int(mask.sum())
            spec = mcspec.make_spectrum(phots, NEN, EMIN_EV * 1.0e-3, EMAX_EV * 1.0e-3,
                                        xaxis="kev", logx=True, nmu=ncth, nphi=nphi,
                                        mask=mask, anglebin="cartesian")
            spectrum = mcspec.add_spectra(spectrum, spec)
            if result["done"]:
                break
        if not keep_lists:
            os.remove(fn)
    return spectrum, acct


def interp_feaut(mu0, mu, varin):
    """Interpolate the Feautrier solution in angle, as the cartesian test does."""
    nnu = len(varin[:, 0])
    varout = np.zeros(nnu)
    for i in range(nnu):
        varout[i] = np.interp(mu0, mu, varin[i, :])
    return varout


def polarization_norm(spectrum, muf, polf):
    """Mean |Q/I(mc) - Q/I(feautrier)| over every (mu, energy) bin that collected photons,
    plus the intensity-weighted band average per mu bin, which is far less noisy and is
    what the sign and slope checks use.  Q/I is a ratio, so whatever per-bin weighting the
    spectrum carries cancels."""
    intensity = spectrum["intensity"]
    intens = np.sum(intensity[0, :, :, :], axis=0)
    qpol = np.sum(intensity[1, :, :, :], axis=0)
    empty = intens == 0.0
    qpol = np.divide(qpol, intens, out=np.zeros_like(qpol), where=~empty)
    mumid = 0.5 * (spectrum["mufaces"][1:] + spectrum["mufaces"][:-1])

    total, npol = 0.0, 0
    band = []
    for j in range(len(mumid)):
        pint = interp_feaut(mumid[j], muf, polf)
        for k in range(intens.shape[1]):
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


def band_aggregate(spectrum, muf, polf, mulo, muhi):
    """Q/I pooled over every (phi, mu, energy) bin with mulo <= mu < muhi, its one-sigma
    error from the spectrum's per-bin errors added in quadrature, and the Feautrier value
    pooled with the same intensity weights.  Q/I is a ratio, so the per-bin weighting the
    spectrum carries cancels; the error on the ratio takes the denominator as exact, which
    is fine when the pooled intensity is well determined, as it is here."""
    intensity, errors = spectrum["intensity"], spectrum["errors"]
    mumid = 0.5 * (spectrum["mufaces"][1:] + spectrum["mufaces"][:-1])
    sel = [j for j in range(len(mumid)) if mulo <= mumid[j] < muhi]
    itot = float(np.sum(intensity[0][:, sel, :]))
    qtot = float(np.sum(intensity[1][:, sel, :]))
    qerr = float(np.sqrt(np.sum(errors[1][:, sel, :]**2)))
    # Feautrier, weighted by the same intensities
    ftot = 0.0
    for j in sel:
        pint = interp_feaut(mumid[j], muf, polf)
        w = np.sum(intensity[0][:, j, :], axis=0)
        ftot += float(np.sum(pint * w))
    if itot <= 0.0:
        return float("nan"), float("nan"), float("nan")
    return qtot / itot, qerr / itot, ftot / itot


def main(**kwargs):

    athena = os.path.join(kwargs["path"], "bin", "athena")
    if not os.path.exists(athena):
        print("no binary at " + athena)
        return 1
    workdir = kwargs["workdir"]
    if not os.path.isdir(workdir):
        os.makedirs(workdir)
    cwd = os.getcwd()

    counts = [int(kwargs["nphot"] * kwargs["refine"]**i) for i in range(kwargs["nstep"])]
    pusher = "general" if kwargs["general_pusher"] else "legacy"
    d1, dmid, width_at = theta_cells(kwargs["nx2"], kwargs["x2rat"])
    top_in = width_at(math.acos(ZMAX / RIN))
    top_out = width_at(math.acos(ZMAX / ROUT))
    print("disk atmosphere, {0} pusher; rin/zmax = {1:.1f}, rout/rin = {2:.1f}; theta from the "
          "pole, {3:d} cells, ratio {4!r}".format(pusher, RIN / ZMAX, ROUT / RIN,
                                                   kwargs["nx2"], kwargs["x2rat"]))
    print("  vertical cell in scale heights: atmosphere top at rin {0:.2f}, at rout {1:.2f};"
          " midplane at rout {2:.2f}; pole cell {3:.2f} deg".format(
              RIN * top_in / L0, ROUT * top_out / L0, ROUT * dmid / L0, math.degrees(d1)))
    if ROUT * top_out > L0 / 3.0:
        print("  WARNING: the atmosphere top at rout is under-resolved; raise nx2 or lower x2rat")

    muf = polf = None
    norms, bands = [], []
    for n in counts:
        out = run(athena, workdir, kwargs["mcranks"], iseed=kwargs["iseed"], nphot=n,
                  nx1=kwargs["nx1"], nx2=kwargs["nx2"], nx3=kwargs["nx3"],
                  x2rat=kwargs["x2rat"], mb1=kwargs["mb1"], mb2=kwargs["mb2"],
                  mb3=kwargs["mb3"], general_pusher=kwargs["general_pusher"],
                  stepsize=kwargs["stepsize"], equal_weight=kwargs["equal_weight"])
        spectrum, acct = bin_lists(workdir, kwargs["ncth"], kwargs["nphi"],
                                   kwargs["keep_lists"])
        # The run's own tally, and the escaping weight per photon.  Both should be
        # independent of nphot up to shot noise; a drift with nphot means the samples are
        # being distributed or capped differently at different counts, which would make the
        # ladder points different physical experiments rather than the same one resolved
        # better.
        tally = [ln.strip() for ln in out.splitlines() if ln.startswith("ntot:")]
        print("      run tally: " + (tally[-1] if tally else "(not found in output)"))
        print("      escaping weight per photon: {0:.6e};  escapers dropped by count: {1:.2%}"
              .format(acct["total"] / n, acct["n_dropped"] / max(acct["n_total"], 1)))

        if muf is None:
            ffile = os.path.join(workdir, "feautrier.out")
            if not os.path.exists(ffile):
                print("Computing feautrier transfer")
                xfaces = spectrum["xfaces"]                     # keV
                nu = 0.5 * (xfaces[1:] + xfaces[:-1]) * 1.0e3 * EVERG / HPLANCK
                os.chdir(workdir)
                feaut.transfer(tconst=TEMP, trange=[TAUMIN, TAUMAX], l0=ZMAX,
                               outfile="feautrier.out", nu=nu)
                os.chdir(cwd)
            nuf, muf, intensf, polf = feaut.read_feautrier(ffile)

        # Keep every point's spectrum, so a result can be re-examined with its own error
        # estimates rather than reconstructed from memory of the printout.
        mcspec.write_spectrum(os.path.join(workdir, "mcdisk_{0:d}.spec".format(n)), spectrum)

        nrm, band, nempty = polarization_norm(spectrum, muf, polf)
        norms.append(nrm)
        bands.append(band)
        last_spectrum = spectrum
        t = acct["total"]
        print("  nphot = {0:>10d}   mean |dP| = {1:.5f}   ({2:d} empty bins)".format(
            n, nrm, nempty))
        print("      escaping weight: side leaks below zmax, inner wall {0:.2%} (a physical bias),"
              " outer wall {1:.1%};  kept above zmax: inner wall {2:.2%}, outer wall {3:.1%},"
              " elsewhere {4:.1%}".format(
                  acct["dropped_inner"] / t, acct["dropped_outer"] / t,
                  acct["kept_inner"] / t, acct["kept_outer"] / t, acct["kept_other"] / t))

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

    print("\nemergent Q/I at the largest photon count, band averaged over energy")
    print("   mu       Q/I (mc)   Q/I (feautrier)")
    for mu, pmc, pfe in bands[-1]:
        print("  {0:5.3f}   {1:+9.5f}   {2:+9.5f}".format(mu, pmc, pfe))

    # Monte Carlo noise falls as 1/sqrt(N), so the norm should too once it is noise
    # dominated.  A norm that stalls means a systematic difference the photon count cannot
    # remove, which is what a geometry or basis error looks like.
    got = sum(orders) / len(orders) if orders else float("nan")

    # Sign and slope, on aggregated bands with the spectrum's own errors.  A single mu bin's
    # band average is the noisiest number in the test -- the grazing bin at the general
    # pusher's affordable photon count carries an uncertainty comparable to the signal --
    # so the lowest three bins (mu < 0.375) and the highest three (mu > 0.625) are pooled,
    # and the sign is required at two sigma.  A run that cannot resolve the sign at two
    # sigma is reported as under-powered rather than as physics.
    gq, gs, gf = band_aggregate(last_spectrum, muf, polf, 0.0, 0.375)
    fq, fs, ff = band_aggregate(last_spectrum, muf, polf, 0.625, 1.0)
    print("\npooled bands at the largest photon count, Q/I +- sigma, and Feautrier")
    print("  grazing  mu < 0.375 : {0:+.5f} +- {1:.5f}   feautrier {2:+.5f}   pull {3:+.1f}"
          .format(gq, gs, gf, (gq - gf) / gs if gs > 0 else float("nan")))
    print("  face-on  mu > 0.625 : {0:+.5f} +- {1:.5f}   feautrier {2:+.5f}   pull {3:+.1f}"
          .format(fq, fs, ff, (fq - ff) / fs if fs > 0 else float("nan")))

    sign_ok = gq + 2.0 * gs < 0.0
    sign_txt = "{0:+.4f}+-{1:.4f}".format(gq, gs)
    if not sign_ok and abs(gq) < 2.0 * gs:
        sign_txt += " (under-powered)"
    results = [
        ("norm falls as 1/sqrt(nphot)",
         "{0:.2f}".format(got), "0.50", abs(got - 0.5) < kwargs["order_tol"]),
        ("Q < 0 at grazing, 2 sigma", sign_txt, "< 0", sign_ok),
        ("|Q/I| rises toward mu = 0",
         "yes" if abs(gq) > abs(fq) else "no", "yes", abs(gq) > abs(fq)),
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
    p.add_argument("--mcranks", type=int, default=8, help="MPI ranks (1 runs serially)")
    p.add_argument("--nphot", type=int, default=1000000,
                   help="photon samples at the coarsest point of the ladder")
    p.add_argument("--nstep", type=int, default=3, help="points in the ladder")
    p.add_argument("--general-pusher", dest="general_pusher", action="store_true",
                   help="use the geodesic pusher instead of the legacy spherical one")
    p.add_argument("--stepsize", type=float, default=0.5,
                   help="general pusher step, as a fraction of a cell crossing")
    p.add_argument("--nx1", type=int, default=32, help="radial cells")
    p.add_argument("--nx2", type=int, default=512,
                   help="polar cells from the pole to the midplane")
    p.add_argument("--x2rat", type=float, default=0.993,
                   help="ratio of consecutive polar cell widths; < 1 shrinks toward the midplane")
    p.add_argument("--nx3", type=int, default=8, help="azimuthal cells")
    p.add_argument("--mb1", type=int, default=8, help="meshblock cells, x1")
    p.add_argument("--mb2", type=int, default=64, help="meshblock cells, x2")
    p.add_argument("--mb3", type=int, default=8, help="meshblock cells, x3")
    p.add_argument("--refine", type=float, default=4.0,
                   help="photon-count refinement factor between points")
    p.add_argument("--order-tol", dest="order_tol", type=float, default=0.15,
                   help="how close the measured convergence order must be to 0.5")
    p.add_argument("--iseed", type=int, default=1300431, help="random seed")
    p.add_argument("--ncth", type=int, default=8, help="mu bins")
    p.add_argument("--nphi", type=int, default=8, help="phi bins")
    p.add_argument("--equal-weight", dest="equal_weight", action="store_true",
                   help="place emission samples in proportion to emission with equal "
                        "weights, instead of uniformly over cells with weighted samples")
    p.add_argument("--keep-lists", dest="keep_lists", action="store_true",
                   help="do not delete the photon lists after binning them")
    sys.exit(main(**vars(p.parse_args())))
