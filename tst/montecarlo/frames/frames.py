#! /usr/bin/env python

"""
Frame-consistency tests for the Monte Carlo radiation moments.

These check identities that must hold exactly, so they need no analytic solution, no
convergence study and no reference data -- only that two numbers agree.  They exist
because the moment machinery has two independent frame conventions (the lab basis the
moments are accumulated in, and the comoving basis they are boosted to) which can drift
apart silently: a bug that put a spurious Doppler factor into the general relativistic
comoving moments went unnoticed precisely because no test compared the two.

  identity  With the fluid at rest the boost to the comoving frame is the identity, so
            moments_com must equal moments_lab zone by zone.  Run on both pushers.

  pushers   In flat spacetime the general pusher, geodesic integration and ConstructTetrad
            must reproduce the legacy cartesian answer exactly, boosted or not.  Any
            disagreement is a bug in the general-pusher machinery, not a modelling
            difference.  This is the same claim tst/montecarlo/minkowski makes, checked
            here directly against the cartesian run rather than against a blackbody.

  traceless The stress tensor of a gas of null particles is traceless, so in any
            orthonormal basis Er must equal P11+P22+P33.  This is the check with real
            teeth in curved spacetime: the tetrad is orthonormal with respect to the
            metric at the zone centre while k is null at the photon, so normalising the
            direction by the time component rather than by the spatial magnitude breaks
            it by the cell-scale variation of the metric -- 3e-2 for the angular
            resolution the shell problem uses.

  redshift  The normal observer in Kerr-Schild measures n_mu n_nu T^{mu nu} = alpha^2 T^tt,
            so Ermc_lab / Ermc_coord must equal 1/(1+2M/r).  This ties the tetrad
            projection to the coordinate accumulation, which share no code.

  lapse     Snake coordinates are flat spacetime sheared by y = y_M + a sin(k x_M).  The
            lapse is exactly 1, so the normal observer is the coordinate-time observer and
            the two bases share e_(0) = d_t: Er must agree between them exactly.  The
            spatial legs still shear, e_(x) = d_x + beta d_y, so the flux and pressure must
            NOT agree -- checked separately, since a regression that collapsed the lab
            tetrad onto the coordinate basis would otherwise pass the Er check by accident.
            No other supported metric separates the lapse from the spatial tetrad.

  flat      Snake with zero amplitude is Minkowski in disguise, so every moment must match
            the Minkowski run.  This is what makes the snake checks trustworthy: it
            separates "the shear is handled correctly" from "the snake path is broken".

  geodesic  The contravariant k^x is constant along every snake geodesic, because x = x_M
            and the underlying Minkowski path is a straight line.  This is the only check
            here that is not self-referential, and it is the one with teeth: zeroing g_xy
            takes the drift from 1e-12 to 1.7, and a 5 percent error in d(g^-1)/dx gives
            2e-2.  It is checked as a convergence rate rather than a tolerance -- halving
            the step must cut the drift by ~16 -- because a bare tolerance cannot tell a
            correct integration from a test that has stopped measuring anything.

            Note what it does NOT test.  RK4 evolves the covariant components with
            dk_mu/dl = -1/2 d_mu g^ab k_a k_b, so k_t, k_y and k_z are conserved to machine
            precision whatever the metric is; only k^x survives non-trivially.  And it
            constrains the metric routines against each other, not against reality: a
            Metric and InverseMetric that consistently described some other spacetime would
            pass.  The flat check above is what rules that out, at zero amplitude.

  A caution about traceless.  PhotonFrames normalises the direction by its spatial
  magnitude, so |n| = 1 by construction and P11+P22+P33 = sum(w) = Er identically, whatever
  the metric or tetrad.  The residual near 1e-6 is tab-writer output precision.  It is a
  useful guard on the accumulator arithmetic but it does not validate any geometry -- both
  deliberate metric bugs above leave it unchanged.  The same caveat applies to the flat,
  spherical and kerr-schild traceless entries, not just snake.

  Snake also carries the traceless check into the only geometry with a non-orthogonal
  coordinate basis (g_xy != 0), though for the reason just given that says less than it
  appears to.

  derived   The comoving moments are a per-zone tensor transform of the lab moments, so
            deriving them at output must reproduce accumulating them photon by photon --
            sum L p L p == L (sum p p) L, exactly, because the transform is the same matrix
            for every photon in the zone.  Checked in flat spacetime, where it is an
            identity.  It is deliberately not checked in curved spacetime: there the
            accumulated path renormalises each photon direction to unit in the comoving
            frame, which the tensor transform cannot replicate, and the two differ by the
            cell-centring error.  That difference is a useful diagnostic but not a bound.

Deliberately uses tab output rather than hdf5 so it runs without an HDF5 build.

Usage:
  python frames.py [--path /path/to/athena] [--nphot N] [--tol 1e-12]
"""

import argparse
import glob
import math
import re
import os
import sys
import numpy as np

KB = 1.380649e-16
TGAS = 1.0e6
DENS = 1.0e-10
PGAS_CODE = 1.0e-6


# Snake shear.  beta = snake_a*snake_k*cos(snake_k*x) is the only free parameter of the
# metric; 0.3 is a substantial shear.  snake_k is fixed by the box so exactly one period
# spans it, which is what makes a periodic x1 boundary legal -- mc_snake refuses otherwise.
SNAKE_BETA = 0.3
SNAKE_L = 1.0e11                       # box side used by the cartesian/mink/snake kinds
SNAKE_K = 2.0 * math.pi / SNAKE_L
SNAKE_A = SNAKE_BETA / SNAKE_K


def write_athinput(path, kind, vel, nphot, iseed, accumulate_comoving=False,
                   snake_amp=None, traj_steps=0, stepsize=None, geodesic=False):
    """kind is 'cart' (legacy pusher), 'mink' (general pusher, GR machinery), 'sphr'
    (legacy pusher in spherical-polar, where the direction is rotated into the zone-centre
    orthonormal basis before it is accumulated) or 'snake' (general pusher on a flat but
    sheared metric with g_xy != 0).

    snake_amp overrides the shear amplitude; pass 0.0 to degenerate snake to Minkowski.
    traj_steps > 0 adds a trajectory output, used by the geodesic check."""
    general = "true" if kind in ("mink", "snake") else "false"
    pid = {"cart": "mciso", "mink": "mcmink", "sphr": "mcsphr", "snake": "mcsnake",
           }[kind]
    o = ["<job>", "problem_id = " + pid, "",
         "<output1>", "file_type = tab", "variable = mclab", "id = lab", "dt = 1.0", "",
         "<output2>", "file_type = tab", "variable = mccom", "id = com", "dt = 1.0", ""]
    if kind == "snake":
        # the coordinate-basis moments are only needed for the unit-lapse check
        o += ["<output3>", "file_type = tab", "variable = mccoord", "id = crd",
              "dt = 1.0", ""]
    if traj_steps:
        o += ["<output4>", "file_type = traj", "steplimit = {0:d}".format(traj_steps),
              "dt = 1.0", ""]
    if geodesic:
        # user[1] carries the largest relative drift in k^x seen along each photon path
        o += ["<output5>", "file_type = phlist", "nuser = 2", ""]
    o += ["<time>", "cfl_number = 0.3", "nlim = 1", "tlim = 1.0", "", "<mesh>"]
    if kind == "sphr":
        # A shell rather than a box.  The angular cells are deliberately coarse, since
        # that is where the rotation into the zone-centre basis actually does something.
        o += ["nx1 = 8", "x1min = 1.0e10", "x1max = 2.0e10",
              "ix1_bc = outflow", "ox1_bc = outflow",
              "ix1_mc_bc = escape", "ox1_mc_bc = escape", "",
              "nx2 = 8", "x2min = 0.0", "x2max = {0!r}".format(math.pi),
              "ix2_bc = polar", "ox2_bc = polar",
              "ix2_mc_bc = polar", "ox2_mc_bc = polar", "",
              "nx3 = 8", "x3min = 0.0", "x3max = {0!r}".format(2.0*math.pi),
              "ix3_bc = periodic", "ox3_bc = periodic",
              "ix3_mc_bc = periodic", "ox3_mc_bc = periodic", ""]
    else:
        for d in (1, 2, 3):
            o += ["nx{0} = 8".format(d),
                  "x{0}min = -5.0e10".format(d), "x{0}max = 5.0e10".format(d),
                  "ix{0}_bc = periodic".format(d), "ox{0}_bc = periodic".format(d),
                  "ix{0}_mc_bc = {1}".format(d, "escape" if geodesic else "periodic"),
                  "ox{0}_mc_bc = {1}".format(d, "escape" if geodesic else "periodic"), ""]
    o += ["<meshblock>", "nx1 = 4", "nx2 = 4", "nx3 = 4", "",
          "<hydro>", "gamma = 1.666666666666667", ""]
    if kind == "snake":
        amp = SNAKE_A if snake_amp is None else snake_amp
        # GRUser reads m and a unconditionally and neither means anything here.
        o += ["<coord>", "m = 0.0", "a = 0.0",
              "snake_a = {0!r}".format(amp), "snake_k = {0!r}".format(SNAKE_K), ""]
    o += ["<montecarlo>",
          "nphot = {0:d}".format(nphot), "iseed = {0:d}".format(iseed),
          "general_pusher = " + general]
    if kind == "snake":
        # gr_user does not determine the Monte Carlo metric, so it has to be named
        o += ["mc_coord = snake"]
    o += [
          "scattering = none", "emission = freefree",
          "absorption = " + ("none" if geodesic else "freefree"),
          "polarized = none",
          "stepsize = {0!r}".format(1.0e-2 if stepsize is None else stepsize),
          "varystep = true", "abs_method = weight",
          # boosts stay on even at zero velocity: that is what makes the comoving
          # transform run and the identity meaningful rather than vacuous.
          "boosts = true",
          "accumulate_comoving = " + ("true" if accumulate_comoving else "false"),
          "", "<problem>"]
    if kind in ("cart", "sphr"):
        o += ["temp = {0:e}".format(TGAS), "dens = {0:e}".format(DENS),
              "velocity = {0:e}".format(vel), "constdens = true"]
        if kind == "sphr":
            o += ["radial = true"]
    else:
        o += ["dens_code = 1.0", "pgas_code = {0:e}".format(PGAS_CODE),
              "rho_cgs = {0:e}".format(DENS),
              "tgas_cgs = {0:e}".format(TGAS / PGAS_CODE),
              "l_cgs = 1.0", "velocity = {0:e}".format(vel)]
    o += ["emin = {0:e}".format(0.001 * KB * TGAS),
          "emax = {0:e}".format(50.0 * KB * TGAS), ""]
    open(path, "w").write("\n".join(o))


def read_moments(rundir, tag):
    """Concatenate the per-block tab files for one output id."""
    rows, hdr = [], None
    for f in sorted(glob.glob(os.path.join(rundir, "*.{0}.*.tab".format(tag)))):
        for line in open(f):
            if line.startswith("#"):
                if "x1v" in line:
                    hdr = line.split()[1:]
                continue
            rows.append([float(x) for x in line.split()])
    if not rows:
        raise RuntimeError("no {0} output found in {1}".format(tag, rundir))
    return np.array(rows), hdr


# Ermc + Frmc1..3 + the nine Prmc components.  The tab writer emits the pressure as a
# nine-column slice under a single "Prmc" header word, so the header has fewer names than
# the data has columns and name matching silently picks up only Prmc11.  The moment block
# is always the trailing NMOMENT columns, which is what makes this robust; mclab carries
# tgas and rho ahead of it while mccom and mccoord do not.
NMOMENT = 13
# The tab writer emits limited digits, so exact identities land near 1e-6 rather than at
# round-off.  The redshift check additionally carries the cell-centring difference between
# the zone-centred tetrad and the photon-position coordinate accumulation.
TRACE_TOL = 1.0e-4
REDSHIFT_TOL = 1.0e-4
# snake(a=0) and mink are the same spacetime reached through different Coordinates classes
# (GRUser evaluates numerically, Minkowski analytically), so they agree to round-off rather
# than bit-for-bit.
FLAT_TOL = 1.0e-8
# Measured drift at stepsize 1e-2 is ~1e-12 and falls as step^4; a broken metric gives
# O(1) and a 5 percent error in d(g^-1)/dx gives ~2e-2, so this bound has ten orders of
# headroom.
GEODESIC_TOL = 1.0e-9
# RK4 predicts 16 per halving; measured 15.8-15.9.  Loosened for Monte Carlo sampling
# noise in which photon happens to be the worst case.
GEODESIC_RATIO = 8.0


def moment_columns(arr, hdr):
    """Return the moment columns only, dropping index, coordinate and extra columns."""
    if arr.shape[1] < NMOMENT:
        raise RuntimeError("expected at least {0} columns, got {1}"
                           .format(NMOMENT, arr.shape[1]))
    return arr[:, -NMOMENT:]


def max_rel(a, b):
    den = np.maximum(np.abs(a), np.abs(b))
    den[den == 0.0] = 1.0
    return float(np.max(np.abs(a - b) / den))


def read_photon_list(rundir):
    """Photon list: text header, then big-endian float64 rows of npars columns.  The user
    variables are the trailing columns, so user[1] -- the geodesic drift -- is the last."""
    rows = []
    for f in sorted(glob.glob(os.path.join(rundir, "*.list"))):
        raw = open(f, "rb").read()
        end = raw.index(b"\n", raw.index(b"coord=")) + 1
        hdr = dict(l.split("=", 1) for l in raw[:end].decode().strip().split("\n"))
        n, npars = int(hdr["length"]), int(hdr["npars"])
        if n == 0:
            continue
        rows.append(np.frombuffer(raw[end:end + n*npars*8],
                                  dtype=">f8").reshape(n, npars))
    if not rows:
        raise RuntimeError("no photon list found in " + rundir)
    return np.vstack(rows)


# Which (prob, coord) the binary in bin/ was last built from.  configure.py rewrites
# src/defs.hpp, which the Makefile does not track as a dependency, so switching either one
# demands a full "make clean" -- but repeating the same pair does not, and the suite runs
# several consecutive cases per configuration.
_built = [None]


def ensure_build(src, prob, coord, extra):
    """Rebuild only when the configuration actually changed."""
    import subprocess
    key = (prob, coord, tuple(extra))
    if _built[0] == key:
        return
    subprocess.call(["make", "clean"], cwd=src,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call(["python", "configure.py", "--prob=" + prob,
                           "--coord=" + coord, "-mc"] + list(extra),
                          cwd=src, stdout=subprocess.DEVNULL)
    subprocess.check_call(["make", "-j8"], cwd=src,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _built[0] = key


def write_gr_athinput(src, path, nphot, iseed):
    """Reuse the repository's Kerr-Schild shell input, with tab moment outputs."""
    ref = os.path.join(src, "inputs", "mc", "athinput.mcisogr")
    text = open(ref).read()
    # drop every existing output block; hdf5 is not assumed to be available
    text = re.sub(r"<output\d+>.*?(?=<[a-z])", "", text, flags=re.S)
    text = re.sub(r"^nphot\s*=.*$", "nphot = {0:d}".format(nphot), text, flags=re.M)
    text = re.sub(r"^iseed\s*=.*$", "iseed = {0:d}".format(iseed), text, flags=re.M)
    outs = ""
    for i, (var, tag) in enumerate((("mclab", "lab"), ("mccom", "com"),
                                    ("mccoord", "crd")), start=1):
        outs += ("<output{0}>\nfile_type = tab\nvariable = {1}\nid = {2}\n"
                 "dt = 1.0\n\n".format(i, var, tag))
    open(path, "w").write(outs + text)


def build_and_run_gr(src, workdir, nphot, iseed):
    import subprocess
    ensure_build(src, "mc_isoth_gr", "kerr-schild", ["-g"])
    os.makedirs(workdir, exist_ok=True)
    inp = os.path.join(workdir, "athinput.frames_gr")
    write_gr_athinput(src, inp, nphot, iseed)
    with open(os.path.join(workdir, "run.log"), "w") as log:
        subprocess.check_call([os.path.join(src, "bin", "athena"), "-i", inp],
                              cwd=workdir, stdout=log, stderr=subprocess.STDOUT)
    return workdir


def build_and_run(src, kind, vel, workdir, nphot, iseed,
                  accumulate_comoving=False, snake_amp=None, traj_steps=0,
                  stepsize=None, geodesic=False):
    import subprocess
    athena_path = os.path.join(src, "bin")
    prob, coord, extra = {"cart": ("mc_isoth", "cartesian", []),
                          "sphr": ("mc_isoth", "spherical_polar", []),
                          "mink": ("mc_isoth_mink", "minkowski", ["-g"]),
                          "snake": ("mc_snake", "gr_user", ["-g"])}[kind]
    ensure_build(src, prob, coord, extra)
    os.makedirs(workdir, exist_ok=True)
    inp = os.path.join(workdir, "athinput.frames")
    write_athinput(inp, kind, vel, nphot, iseed, accumulate_comoving,
                   snake_amp, traj_steps, stepsize, geodesic)
    with open(os.path.join(workdir, "run.log"), "w") as log:
        subprocess.check_call([os.path.join(athena_path, "athena"), "-i", inp],
                              cwd=workdir, stdout=log, stderr=subprocess.STDOUT)
    return workdir


def main(**kwargs):
    path = kwargs.pop("path")
    nphot = kwargs.pop("nphot")
    tol = kwargs.pop("tol")
    iseed = kwargs.pop("iseed")
    work = kwargs.pop("workdir")
    src = os.path.abspath(path)

    failures = []
    runs = {}
    for kind in ("cart", "mink", "sphr", "snake"):
        for vel, label in ((0.0, "zero"), (0.1, "boost")):
            name = "{0}_{1}".format(kind, label)
            runs[name] = build_and_run(src, kind, vel,
                                       os.path.join(work, name), nphot, iseed)
    # snake with no shear must reduce to the Minkowski run exactly
    runs["snake_flat"] = build_and_run(src, "snake", 0.0,
                                       os.path.join(work, "snake_flat"), nphot, iseed,
                                       snake_amp=0.0)

    print("{0:<34}{1:>14}   {2}".format("check", "max rel diff", "result"))
    print("-" * 66)

    # identity: fluid at rest => comoving moments equal lab moments
    for name in ("cart_zero", "mink_zero", "sphr_zero", "snake_zero"):
        lab, hl = read_moments(runs[name], "lab")
        com, hc = read_moments(runs[name], "com")
        d = max_rel(moment_columns(lab, hl), moment_columns(com, hc))
        ok = d < tol
        failures += [] if ok else ["identity/" + name]
        print("{0:<34}{1:>14.3e}   {2}".format(
            "identity  " + name + "  com==lab", d, "PASS" if ok else "FAIL"))

    # pushers: general pusher in flat spacetime reproduces the legacy cartesian answer
    for label in ("zero", "boost"):
        for tag in ("lab", "com"):
            a, ha = read_moments(runs["cart_" + label], tag)
            b, hb = read_moments(runs["mink_" + label], tag)
            d = max_rel(moment_columns(a, ha), moment_columns(b, hb))
            ok = d < tol
            failures += [] if ok else ["pushers/" + label + "/" + tag]
            print("{0:<34}{1:>14.3e}   {2}".format(
                "pushers   " + label + "  " + tag + "  cart==mink", d,
                "PASS" if ok else "FAIL"))

    # traceless: Er == P11+P22+P33 in any orthonormal basis, for a null-photon gas
    gr = build_and_run_gr(src, os.path.join(work, "gr"), nphot, iseed)
    for label, rundir, tags in (("flat", runs["mink_boost"], ("lab", "com")),
                                ("spherical", runs["sphr_boost"], ("lab", "com")),
                                ("kerr-schild", gr, ("lab", "com")),
                                ("snake", runs["snake_boost"], ("lab", "com"))):
        for tag in tags:
            a, h = read_moments(rundir, tag)
            M = moment_columns(a, h)
            er, ptr = M[:, 0], M[:, 4] + M[:, 5] + M[:, 6]
            scale = np.abs(er) + np.abs(M[:, 4]) + np.abs(M[:, 5]) + np.abs(M[:, 6])
            keep = scale > 0
            d = float(np.max(np.abs(er - ptr)[keep] / scale[keep]))
            ok = d < TRACE_TOL
            failures += [] if ok else ["traceless/" + label + "/" + tag]
            print("{0:<34}{1:>14.3e}   {2}".format(
                "traceless " + label + "  " + tag, d, "PASS" if ok else "FAIL"))

    # derived: in flat spacetime, deriving the comoving moments must reproduce accumulating
    acc = build_and_run(src, "cart", 0.1, os.path.join(work, "cart_boost_acc"),
                        nphot, iseed, accumulate_comoving=True)
    a, ha = read_moments(acc, "com")
    b, hb = read_moments(runs["cart_boost"], "com")
    d = max_rel(moment_columns(a, ha), moment_columns(b, hb))
    ok = d < tol
    failures += [] if ok else ["derived"]
    print("{0:<34}{1:>14.3e}   {2}".format(
        "derived   flat  accum==derived", d, "PASS" if ok else "FAIL"))

    accs = build_and_run(src, "sphr", 0.1, os.path.join(work, "sphr_boost_acc"),
                         nphot, iseed, accumulate_comoving=True)
    a, ha = read_moments(accs, "com")
    b, hb = read_moments(runs["sphr_boost"], "com")
    d = max_rel(moment_columns(a, ha), moment_columns(b, hb))
    ok = d < tol
    failures += [] if ok else ["derived/spherical"]
    print("{0:<34}{1:>14.3e}   {2}".format(
        "derived   spherical  accum==derived", d, "PASS" if ok else "FAIL"))

    # redshift: Ermc_lab / Ermc_coord == alpha^2 = 1/(1+2M/r) for Kerr-Schild with a = 0
    lab, hl = read_moments(gr, "lab")
    crd, hc = read_moments(gr, "crd")
    rad = crd[:, hc.index("x1v")]
    el = moment_columns(lab, hl)[:, 0]
    ec = moment_columns(crd, hc)[:, 0]
    keep = ec > 0
    alpha2 = 1.0 / (1.0 + 2.0 / rad)
    d = float(np.max(np.abs(el[keep] / ec[keep] - alpha2[keep]) / alpha2[keep]))
    ok = d < REDSHIFT_TOL
    failures += [] if ok else ["redshift"]
    print("{0:<34}{1:>14.3e}   {2}".format(
        "redshift  kerr-schild  lab/coord", d, "PASS" if ok else "FAIL"))

    # lapse: snake has alpha = 1, so the normal observer IS the coordinate-time observer
    # and the two bases share e_(0) = d_t.  Er must therefore agree exactly.  The flux and
    # pressure must NOT: the spatial legs still shear, e_(x) = d_x + beta d_y, and they
    # differ by O(beta).  Checking only Er is the whole point -- it isolates the lapse from
    # the spatial tetrad, which no other supported metric can do.
    lab, hl = read_moments(runs["snake_boost"], "lab")
    crd, hc = read_moments(runs["snake_boost"], "crd")
    el = moment_columns(lab, hl)[:, 0]
    ec = moment_columns(crd, hc)[:, 0]
    d = max_rel(el, ec)
    ok = d < tol
    failures += [] if ok else ["lapse"]
    print("{0:<34}{1:>14.3e}   {2}".format(
        "lapse     snake  Er lab==coord", d, "PASS" if ok else "FAIL"))

    # and the converse: with a real shear the spatial parts must genuinely differ, so a
    # regression that silently collapsed the lab tetrad onto the coordinate basis would be
    # caught rather than passing the Er check above by accident.
    fl = moment_columns(lab, hl)[:, 1:4]
    fc = moment_columns(crd, hc)[:, 1:4]
    d = max_rel(fl, fc)
    ok = d > 1.0e-3
    failures += [] if ok else ["shear-acts"]
    print("{0:<34}{1:>14.3e}   {2}".format(
        "shear     snake  Fr lab!=coord", d, "PASS" if ok else "FAIL"))

    # flat: snake with zero amplitude is Minkowski in disguise, so every moment must match
    # the Minkowski run.  This is what makes the snake checks above trustworthy -- it
    # separates "the shear is handled correctly" from "the snake path is broken".
    for tag in ("lab", "com"):
        a, ha = read_moments(runs["snake_flat"], tag)
        b, hb = read_moments(runs["mink_zero"], tag)
        d = max_rel(moment_columns(a, ha), moment_columns(b, hb))
        ok = d < FLAT_TOL
        failures += [] if ok else ["flat/" + tag]
        print("{0:<34}{1:>14.3e}   {2}".format(
            "flat      snake(a=0)==mink  " + tag, d, "PASS" if ok else "FAIL"))

    # geodesic: k^x is conserved along every snake geodesic, and unlike the identities
    # above it is not conserved structurally -- see TrackGeodesicInvariant in mc_snake.cpp.
    # What is left is RK4 truncation error, so the check is the convergence rate rather
    # than a tolerance: halving the step must cut the drift by ~16.  A bare tolerance would
    # pass for a test that had quietly stopped measuring anything.
    drift = {}
    for st in (2.0e-2, 1.0e-2):
        d = build_and_run(src, "snake", 0.0,
                          os.path.join(work, "geo_{0:g}".format(st)),
                          min(nphot, 2000), iseed, stepsize=st, geodesic=True)
        drift[st] = float(np.max(read_photon_list(d)[:, -1]))
    ratio = drift[2.0e-2] / drift[1.0e-2] if drift[1.0e-2] > 0 else float("inf")
    ok = drift[1.0e-2] < GEODESIC_TOL and ratio > GEODESIC_RATIO
    failures += [] if ok else ["geodesic"]
    print("{0:<34}{1:>14.3e}   {2}".format(
        "geodesic  snake  k^x drift", drift[1.0e-2], "PASS" if ok else "FAIL"))
    print("{0:<34}{1:>14.2f}   {2}".format(
        "geodesic  convergence (RK4 -> 16)", ratio, "PASS" if ok else "FAIL"))

    print("-" * 66)
    if failures:
        print("FAILED: " + ", ".join(failures))
        return 1
    print("all frame-consistency checks passed")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", default="../../..", help="path to athena root")
    p.add_argument("--workdir", default="/tmp/mc_frames", help="scratch run directory")
    p.add_argument("--nphot", type=int, default=200000, help="photons per run")
    p.add_argument("--iseed", type=int, default=17, help="random seed")
    p.add_argument("--tol", type=float, default=1.0e-12, help="relative tolerance")
    sys.exit(main(**vars(p.parse_args())))
