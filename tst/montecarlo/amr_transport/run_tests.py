#! /usr/bin/env python
"""
Does photon transport survive a refinement boundary?

Runs mc_isoth in a uniform medium of prescribed optical depth on three meshes that differ
only in how they are refined: uniform, one level refined above the mid-plane, and one
level refined below it.  The medium has no gradient, so the escaping fraction cannot
depend on the mesh, and the three runs must agree to within Poisson noise.

The run uses <montecarlo>/equal_weight = true, and that is not incidental.  With the
default variable-weight sampling every block emits the same number of photons whatever
its volume, so a refined block emits as many photons as a coarse one while carrying an
eighth of the energy.  The emitted energy per block is still correct -- the per-cell
emission array carries the cell volume -- but the photon counts no longer track it, and
nesc/ntot then changes with refinement even when the transport is exact.  Comparing
counts across meshes is only meaningful when every photon carries the same weight.

Usage:
    python run_tests.py [--workdir DIR] [--path /path/to/athena] [--nphot N]
                        [--tau TAU] [--sigma N] [--keep-build]

Configures and builds its own binary, so it does not disturb an existing build.
"""

import argparse
import math
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

# refinement region for each case, as (x3min, x3max); None means no refinement
CASES = [
    ("uniform", None),
    ("fine above interface", ("5.0e10", "1.0e11")),
    ("fine below interface", ("0.0", "5.0e10")),
]


def build(root, workdir):
    """Configure and build mc_isoth in workdir, leaving the caller's build alone."""
    src = os.path.join(workdir, "build")
    if os.path.exists(src):
        shutil.rmtree(src)
    os.makedirs(src)
    for item in ("src", "inputs", "configure.py", "Makefile.in"):
        s = os.path.join(root, item)
        d = os.path.join(src, item)
        shutil.copytree(s, d) if os.path.isdir(s) else shutil.copy(s, d)
    subprocess.check_call([sys.executable, "configure.py", "--prob=mc_isoth", "-mc"],
                          cwd=src, stdout=subprocess.DEVNULL)
    subprocess.check_call(["make", "-j8"], cwd=src, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)
    return src


def write_input(build_dir, workdir, name, region, tau):
    """athinput.mciso with the heavy outputs dropped and a uniform medium."""
    text = open(os.path.join(build_dir, "inputs", "mc", "athinput.mciso")).read()
    kept, skip = [], False
    for line in text.split("\n"):
        if line.strip().startswith("<"):
            skip = line.strip().startswith(("<output2>", "<output3>"))
        if not skip:
            kept.append(line)
    text = "\n".join(kept)
    text = text.replace("<problem>", "<problem>\nconstdens = true\ntau       = %s"
                        % tau, 1)
    # see the module docstring: photon counts only track energy at equal weight
    text = text.replace("<montecarlo>", "<montecarlo>\nequal_weight = true", 1)
    if region is not None:
        block = ("refinement = static\n\n<refinement1>\n"
                 "x1min = -5.0e10\nx1max =  5.0e10\n"
                 "x2min = -5.0e10\nx2max =  5.0e10\n"
                 "x3min = %s\nx3max = %s\nlevel = 1\n\n<meshblock>" % region)
        text = text.replace("<meshblock>", block, 1)
    path = os.path.join(workdir, "athinput.%s" % name)
    open(path, "w").write(text)
    return path


def run(build_dir, workdir, tag, inp, nphot):
    rundir = os.path.join(workdir, "run_%s" % tag)
    os.makedirs(rundir, exist_ok=True)
    out = subprocess.run([os.path.join(build_dir, "bin", "athena"), "-i", inp,
                          "montecarlo/nphot=%d" % nphot],
                         cwd=rundir, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True).stdout
    m = re.search(r"ntot:\s*(\d+)\s+nesc:\s*(\d+)\s+nabs:\s*(\d+)", out)
    if not m:
        print(out[-800:])
        raise RuntimeError("could not parse photon totals for case %r" % tag)
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=os.path.join(HERE, "work"))
    ap.add_argument("--path", default=DEFAULT_ROOT)
    ap.add_argument("--nphot", type=int, default=200000)
    ap.add_argument("--tau", default="3.0")
    ap.add_argument("--sigma", type=float, default=4.0,
                    help="tolerance in standard deviations (default 4)")
    ap.add_argument("--keep-build", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    print("building mc_isoth ...")
    build_dir = build(args.path, args.workdir)

    results = []
    for i, (name, region) in enumerate(CASES):
        tag = "c%d" % i
        inp = write_input(build_dir, args.workdir, tag, region, args.tau)
        ntot, nesc, nabs = run(build_dir, args.workdir, tag, inp, args.nphot)
        if ntot != args.nphot or nesc + nabs != ntot:
            print("  %-22s ACCOUNTING ERROR ntot=%d nesc=%d nabs=%d"
                  % (name, ntot, nesc, nabs))
            return 1
        results.append((name, nesc))
        print("  %-22s nesc = %d" % (name, nesc))

    # the uniform run is the reference; the medium has no gradient, so refinement
    # cannot change the escaping fraction
    ref = results[0][1]
    p = ref / float(args.nphot)
    sigma = math.sqrt(args.nphot * p * (1.0 - p))
    print("\nreference nesc = %d, sigma = %.0f, tolerance = %.1f sigma"
          % (ref, sigma, args.sigma))

    nfail = 0
    for name, nesc in results[1:]:
        dev = (nesc - ref) / sigma
        ok = abs(dev) <= args.sigma
        print("  %-22s %+6d  (%+.1f sigma)  %s"
              % (name, nesc - ref, dev, "PASS" if ok else "FAIL"))
        nfail += int(not ok)

    if not args.keep_build:
        shutil.rmtree(build_dir, ignore_errors=True)
    print("\n%d of %d refined cases agree with the uniform mesh"
          % (len(results) - 1 - nfail, len(results) - 1))
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
