#! /usr/bin/env python
"""
Regression test for MCGridFile, the athdf grid reader.

Builds a standalone driver around MCGridFile, generates athdf fixtures, and checks that
each is either accepted or rejected for the expected reason.  Optionally also runs the
reader over a real snapshot given with --athdf.

The driver is compiled directly from source rather than from obj/, so this test does not
depend on the tree having been built, and it is compiled against a copy of defs.hpp whose
COORDINATE_SYSTEM matches the fixtures.

Usage:
    python run_tests.py [--workdir DIR] [--athdf FILE] [--path /path/to/athena]

Requires h5py and an MPI C++ compiler.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

# fixture -> (accepted?, substring the rejection message must contain)
EXPECTED = {
    "uniform_ok.athdf":       (True, None),
    "ratio_ok.athdf":         (True, None),
    "amr_ok.athdf":           (True, None),
    "coord_compatible.athdf": (True, None),
    "bad_ghost.athdf":    (False, "not divisible by the block size"),
    "bad_slice.athdf":    (False, "sliced or summed output"),
    "bad_nomesh.athdf":   (False, "mesh_data = false"),
    "bad_coord.athdf":    (False, "different grid topology"),
    "bad_tiling.athdf":   (False, "gaps or overlaps"),
    "bad_usergen.athdf":  (False, "user-defined mesh generator"),
    "bad_intdata.athdf":  (False, "stores cell data as integers"),
}


def coord_of(defs_hpp):
    with open(defs_hpp) as f:
        m = re.search(r'#define\s+COORDINATE_SYSTEM\s+"([^"]+)"', f.read())
    return m.group(1) if m else None


def hdf5_flags():
    """Return (include, lib) dirs for HDF5, preferring what h5cc reports."""
    for probe in ("h5pcc", "h5cc"):
        exe = shutil.which(probe)
        if exe is None:
            continue
        try:
            out = subprocess.check_output([exe, "-show"], text=True)
        except (subprocess.CalledProcessError, OSError):
            continue
        inc = re.findall(r"-I(\S+)", out)
        lib = re.findall(r"-L(\S+)", out)
        if inc or lib:
            return inc, lib
    return [], []


def build(root, workdir, coord):
    """Compile the driver against a defs.hpp patched to `coord`."""
    srccopy = os.path.join(workdir, "src")
    if os.path.exists(srccopy):
        shutil.rmtree(srccopy)
    shutil.copytree(os.path.join(root, "src"), srccopy)

    defs = os.path.join(srccopy, "defs.hpp")
    with open(defs) as f:
        text = f.read()
    text = re.sub(r'(#define\s+COORDINATE_SYSTEM\s+")[^"]+(")',
                  r"\g<1>%s\g<2>" % coord, text)
    with open(defs, "w") as f:
        f.write(text)

    # test_mcgrid.cpp includes ../../../src/..., so it has to sit at the same depth
    # below workdir as it does below the repo root, next to the patched src copy
    driverdir = os.path.join(workdir, "tst", "montecarlo", "mcgrid")
    os.makedirs(driverdir, exist_ok=True)
    driver = os.path.join(driverdir, "test_mcgrid.cpp")
    shutil.copy(os.path.join(HERE, "test_mcgrid.cpp"), driver)

    inc, lib = hdf5_flags()
    exe = os.path.join(workdir, "test_mcgrid")
    cmd = [os.environ.get("CXX", "mpicxx"), "-O2", "-std=c++11"]
    cmd += ["-I%s" % d for d in inc]
    cmd += [
        driver,
        os.path.join(srccopy, "monte_carlo", "mcgrid.cpp"),
        os.path.join(srccopy, "globals.cpp"),
        os.path.join(srccopy, "parameter_input.cpp"),
        os.path.join(srccopy, "outputs", "io_wrapper.cpp"),
        "-o", exe,
    ]
    cmd += ["-L%s" % d for d in lib]
    cmd += ["-lhdf5", "-lz", "-ldl"]
    subprocess.check_call(cmd)
    return exe


def run(exe, path):
    p = subprocess.run([exe, path], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=os.path.join(HERE, "work"))
    ap.add_argument("--athdf", default=None,
                    help="also run the reader over this real snapshot")
    ap.add_argument("--path", default=DEFAULT_ROOT,
                    help="path to the athena source tree")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    # Build the fixture driver at a fixed coordinate system rather than whatever the tree
    # happens to be configured for.  The coordinate check compares grid topologies, so a
    # tree configured for gr_user -- which implies no topology -- would skip it entirely
    # and quietly turn bad_coord.athdf into a no-op.
    coord = "cartesian"
    print("building driver (COORDINATE_SYSTEM = %s)" % coord)
    exe = build(args.path, args.workdir, coord)

    fixdir = os.path.join(args.workdir, "fixtures")
    subprocess.check_call([sys.executable, os.path.join(HERE, "make_fixtures.py"),
                           fixdir, "--coord", coord])

    npass = nfail = 0
    for name in sorted(EXPECTED):
        want_ok, want_msg = EXPECTED[name]
        rc, out = run(exe, os.path.join(fixdir, name))
        got_ok = (rc == 0)
        ok = (got_ok == want_ok)
        if ok and not want_ok:
            ok = want_msg in out
        print("  %-22s %s" % (name, "PASS" if ok else "FAIL"))
        if not ok:
            print("    expected %s%s" % ("accept" if want_ok else "reject",
                                         "" if want_ok else " (%r)" % want_msg))
            for line in out.splitlines()[:6]:
                print("    | " + line)
            nfail += 1
        else:
            npass += 1

    if args.athdf:
        # a real snapshot need not share this build's coordinate system, so give it a
        # driver compiled to match rather than tripping the coordinate check
        import h5py
        with h5py.File(args.athdf, "r") as f:
            fcoord = f.attrs["Coordinates"]
        if isinstance(fcoord, bytes):
            fcoord = fcoord.decode()
        exe_real = exe
        if fcoord != coord:
            print("rebuilding driver for %s (COORDINATE_SYSTEM = %s)"
                  % (os.path.basename(args.athdf), fcoord))
            exe_real = build(args.path, os.path.join(args.workdir, "real"), fcoord)
        rc, out = run(exe_real, args.athdf)
        ok = (rc == 0)
        print("  %-22s %s" % (os.path.basename(args.athdf), "PASS" if ok else "FAIL"))
        for line in out.splitlines():
            print("    | " + line)
        npass += int(ok)
        nfail += int(not ok)

    print("\n%d passed, %d failed" % (npass, nfail))
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
