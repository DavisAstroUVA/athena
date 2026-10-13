#!/usr/bin/env python3
"""
Average Monte Carlo moment outputs (athdf) written at several output intervals.

A static Monte Carlo run with <montecarlo>/nout > 1 emits the same number of photons in
each interval and writes an independent estimate of the same moments each time.  This
script combines these into a single, lower-noise estimate

Output format
-------------
The first input is copied and its data overwritten in place, so every attribute and every
mesh array (Levels, LogicalLocations, x1f ... x3v) is inherited byte-for-byte.

How the moments combine
-----------------------
For M intervals of equal nphot and tint, each variable is averaged, e.g.

    Jbar = (1/M) sum_i J_i

and a variable whose name carries '_err' (e.g. the per-interval standard error that
MonteCarloBlock::NormalizeMoments writes for the scattering moments) combines in
quadrature and is then divided by M, not averaged:

    sigma_bar = sqrt( sum_i sigma_i^2 ) / M

which is RMS(sigma_i)/sqrt(M).

The empirical error
-------------------
With --empirical a second file is written whose '_err' slots instead hold

    sigma_emp = std(J_i, ddof=1) / sqrt(M)

the scatter of the intervals about their own mean. The per-interval error is a path-length
estimator that squares one photon's contribution to a cell once per free flight.
Comparing the two files cell by cell measures how optimistic the per-interval estimate is.
Note, however, that at small M, sigma_emp is itself noisy so its own fractional error is
about 1/sqrt(2(M-1)), so 24% at M = 10. 

Note what the empirical file holds for a variable that has NO error partner, which (at
the time of this writing)includes moments mclab, mccom, mccoord and mcsrc. If --emprical
is selected the error goes into the variable's own slot.

usage:
  average_moments.py -o mean.athdf run.out4.00000.athdf run.out4.00001.athdf ...
  average_moments.py -o mean.athdf --empirical emp.athdf run.out4.0000*.athdf
"""
import argparse
import os
import re
import shutil
import sys

import h5py
import numpy as np

# --- temporary: names from before the <value>_err convention was settled ---------------
# The scattering moments were written as sscat5 / sscat_err5.  Translating them here, once,
# at read time does three things: old files can be averaged, old and new files can be
# averaged together, and the rest of this script only ever sees the new convention.  The
# output always carries the new names whatever went in.
#
# To remove this support, delete LEGACY_VAL, LEGACY_ERR and canonical_names, and replace
# the two calls to canonical_names with plain decode().
LEGACY_VAL = re.compile(r"^sscat(\d+)$")
LEGACY_ERR = re.compile(r"^sscat_err(\d+)$")


def canonical_names(names):
    """Legacy scattering-moment names -> the <value>_err convention; others unchanged."""
    out = []
    for n in names:
        m = LEGACY_VAL.match(n)
        if m:
            out.append("Jnu" + m.group(1))
            continue
        m = LEGACY_ERR.match(n)
        if m:
            out.append("Jnu" + m.group(1) + "_err")
            continue
        out.append(n)
    return out


def decode(a):
    """HDF5 fixed-length string attribute -> list of str."""
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def var_names(f):
    """This file's variable names, in its own order, translated to the convention."""
    return canonical_names(decode(f.attrs["VariableNames"]))


def check_same_mesh(files):
    """Every input must describe the same grid and the same variables."""
    ref = files[0]
    rname = ref.filename
    for key in ("MeshBlockSize", "RootGridSize", "NumMeshBlocks", "MaxLevel",
                "NumVariables"):
        a = np.atleast_1d(ref.attrs[key])
        for f in files[1:]:
            b = np.atleast_1d(f.attrs[key])
            if not np.array_equal(a, b):
                raise SystemExit(f"{f.filename}: attribute {key} = {b} does not match "
                                 f"{rname} ({a})")
    a = decode(ref.attrs["DatasetNames"])
    for f in files[1:]:
        b = decode(f.attrs["DatasetNames"])
        if a != b:
            raise SystemExit(f"{f.filename}: DatasetNames does not match {rname}\n"
                             f"  {rname}: {a}\n  {f.filename}: {b}")
    # Compared as sets of canonical names, so a file written before the naming convention
    # changed matches one written after.  Order is allowed to differ; combine() reads each
    # file through its own permutation.
    a = set(var_names(ref))
    for f in files[1:]:
        b = set(var_names(f))
        if a != b:
            raise SystemExit(
                f"{f.filename}: variable names do not match {rname}\n"
                f"  only in {rname}: {sorted(a - b)}\n"
                f"  only in {f.filename}: {sorted(b - a)}")
    for key in ("Levels", "LogicalLocations"):
        a = ref[key][:]
        for f in files[1:]:
            if not np.array_equal(a, f[key][:]):
                raise SystemExit(f"{f.filename}: {key} differs from {rname}; the inputs "
                                 "are not on the same mesh")


def variable_groups(names):
    """
    Pair each error variable with the value variable it belongs to.

    Returns a list of (value_index, error_index_or_None).  The convention is that an error
    variable is named for its value variable with '_err' appended (e.g. Jnu5 and Jnu5_err).
    A variable with no partner (e.g. mclab or mcsrc) is averaged on its own.
    """
    index = {n: i for i, n in enumerate(names)}
    used = set()
    groups = []
    for i, n in enumerate(names):
        if n.endswith("_err"):
            continue
        j = index.get(n + "_err")
        groups.append((i, j))
        used.add(i)
        if j is not None:
            used.add(j)
    orphans = [i for i, n in enumerate(names) if i not in used]
    if orphans:
        raise SystemExit(
            "these look like error variables but no value variable matches them: "
            + ", ".join(names[i] for i in orphans)
            + "\nThe convention is <value>_err, e.g. Jnu5 and Jnu5_err.")
    return groups


def combine(files, perm, dset, groups, names, out_val, out_err, emp_err, verbose):
    """
    Stream one variable group at a time.

    Files on the inner loop and variables on the outer one: a single variable of the
    largest grid here is about 0.5 GB in float64, where a whole file is many times that.
    Total read is M x filesize either way, which is unavoidable.

    perm[k][c] is where canonical variable c sits in file k, so files that name or order
    their variables differently still line up.  The output is written in canonical order.

    Memory is flat in the number of files -- each slab is read, consumed and released
    inside one iteration -- and proportional to the cells in one variable.  Keeping that
    constant small is the goal:

      - s2 exists only to feed the empirical error, so it is not built at all unless that
        was asked for;
      - slabs stay in the float32 they are stored as.  Adding float32 into a float64
        accumulator widens element-wise inside the ufunc and is bitwise identical to
        converting the whole slab first, which would cost a full-size temporary;
      - masking and squaring are done in place, and the one float64 buffer the squares
        need is allocated once per variable and reused for every file.  np.multiply needs
        dtype=np.float64 to be told to compute in float64 rather than compute in float32
        and cast the result, which is not the same number.
    """
    nfile = len(files)
    shape = files[0][dset].shape[1:]
    want_emp = emp_err is not None
    for g, (iv, ie) in enumerate(groups):
        s1 = np.zeros(shape, dtype=np.float64)     # sum of J_i
        s2 = np.zeros(shape, dtype=np.float64) if want_emp else None  # sum of J_i^2
        qerr = np.zeros(shape, dtype=np.float64) if ie is not None else None
        cnt = np.zeros(shape, dtype=np.int32)
        sq = (np.empty(shape, dtype=np.float64)
              if (want_emp or ie is not None) else None)

        for k, f in enumerate(files):
            v = f[dset][perm[k][iv]]               # float32, and ours to modify
            ok = np.isfinite(v)
            e = None
            if ie is not None:
                e = f[dset][perm[k][ie]]
                np.logical_and(ok, np.isfinite(e), out=ok)
            bad = ~ok
            # Zeroing has to be an assignment, not a multiply by the mask: the excluded
            # entries are the non-finite ones and NaN*0 is NaN.
            v[bad] = 0
            s1 += v
            if want_emp:
                np.multiply(v, v, out=sq, dtype=np.float64)
                s2 += sq
            if e is not None:
                # A cell dropped from the mean must be dropped from the error too, or the
                # two stop describing the same sample.
                e[bad] = 0
                np.multiply(e, e, out=sq, dtype=np.float64)
                qerr += sq
            cnt += ok
            del v, e, ok, bad

        good = cnt > 0

        # Before s1 stops being the sum.  Worked through s2 and the scratch buffer so that
        # no full-size array is allocated here either.  Two-pass would be kinder
        # numerically but it doubles the reads; the variance is clamped at zero so
        # round-off cannot put a NaN into the square root.
        if want_emp:
            two = cnt > 1
            np.multiply(s1, s1, out=sq)
            np.divide(sq, cnt, out=sq, where=good)
            np.subtract(s2, sq, out=s2)                  # sum of squares about the mean
            np.divide(s2, np.maximum(cnt - 1, 1), out=s2, where=two)
            np.maximum(s2, 0.0, out=s2)
            np.sqrt(s2, out=s2)                          # standard deviation
            np.sqrt(cnt, out=sq, dtype=np.float64)
            np.divide(s2, sq, out=s2, where=two)         # standard error of the mean
            s2[~two] = np.nan
            emp_err[iv if ie is None else ie] = s2.astype(emp_err.dtype)

        # These two go in place as well, so the mean lands in s1 and the combined error in
        # qerr rather than in fresh arrays.  Both hold zero where good is false, which is
        # exactly where the division is skipped, so the NaN fill is what shows through.
        np.divide(s1, cnt, out=s1, where=good)
        s1[~good] = np.nan
        out_val[iv] = s1.astype(out_val.dtype)

        if ie is not None:
            np.sqrt(qerr, out=qerr)
            np.divide(qerr, cnt, out=qerr, where=good)
            qerr[~good] = np.nan
            out_err[ie] = qerr.astype(out_err.dtype)

        dropped = int((cnt < nfile).sum())
        if verbose:
            tag = names[iv] + (f" (+{names[ie]})" if ie is not None else "")
            extra = f", {dropped} cell(s) missing an interval" if dropped else ""
            print(f"  [{g+1}/{len(groups)}] {tag}{extra}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="athdf moment files from the same run")
    p.add_argument("-o", "--out", required=True, help="averaged output file")
    p.add_argument("--empirical", metavar="FILE", default=None,
                   help="also write this file, with the '_err' slots holding the scatter "
                        "between intervals instead of the propagated error")
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args()

    if len(args.inputs) < 2:
        raise SystemExit("need at least two files to average")
    if args.empirical and len(args.inputs) < 3:
        print("warning: the empirical error from two intervals is barely meaningful "
              "(70% uncertain on itself)", file=sys.stderr)
    for path in (args.out, args.empirical):
        if path and os.path.abspath(path) in [os.path.abspath(i) for i in args.inputs]:
            raise SystemExit(f"refusing to write {path}: it is one of the inputs")

    files = [h5py.File(f, "r") for f in args.inputs]
    try:
        check_same_mesh(files)
        # Canonical order is the first file's, translated; every other file is read
        # through a permutation onto it.
        names = var_names(files[0])
        perm = []
        for f in files:
            here = {n: i for i, n in enumerate(var_names(f))}
            perm.append([here[n] for n in names])
        dsets = decode(files[0].attrs["DatasetNames"])
        if len(dsets) != 1:
            raise SystemExit(f"expected a single dataset, found {dsets}; moment outputs "
                             "write one")
        dset = dsets[0]
        groups = variable_groups(names)
        npair = sum(1 for _, e in groups if e is not None)
        renamed = [f.filename for f in files
                   if decode(f.attrs["VariableNames"]) != var_names(f)]
        if not args.quiet:
            print(f"averaging {len(files)} file(s), dataset '{dset}', {len(names)} "
                  f"variable(s), {npair} with an error partner")
            if renamed:
                print(f"note: {len(renamed)} file(s) use the old sscat names; the output "
                      "is written with the current ones")
            nlone = sum(1 for _, e in groups if e is None)
            if args.empirical and nlone:
                print(f"note: {nlone} variable(s) have no error partner; in "
                      f"{args.empirical} their own slot holds the empirical error, not "
                      "the value")

        # Inherit the format by copying, then overwrite the data in place.
        shutil.copyfile(args.inputs[0], args.out)
        emp_path = args.empirical
        if emp_path:
            shutil.copyfile(args.inputs[0], emp_path)

        out = h5py.File(args.out, "r+")
        emp = h5py.File(emp_path, "r+") if emp_path else None
        try:
            combine(files, perm, dset, groups, names, out[dset], out[dset],
                    emp[dset] if emp else None, not args.quiet)
            times = [float(np.atleast_1d(f.attrs["Time"])[0]) for f in files]
            for h in (out, emp):
                if h is None:
                    continue
                # The copy carries the first input's names, which may be the old ones and
                # are in its order; the data was written in canonical order, so the
                # attribute has to say so.
                width = max(files[0].attrs["VariableNames"].dtype.itemsize,
                            max(len(n) for n in names) + 1)
                h.attrs.create("VariableNames",
                               np.array([n.encode() for n in names], dtype=f"|S{width}"))
                h.attrs["Time"] = np.array(float(np.mean(times)),
                                           dtype=np.atleast_1d(files[0].attrs["Time"]).dtype)
                h.attrs["NumAveraged"] = np.int32(len(files))
                h.attrs["AveragedFrom"] = np.array(
                    [os.path.basename(i).encode() for i in args.inputs])
        finally:
            out.close()
            if emp:
                emp.close()
    finally:
        for f in files:
            f.close()

    if not args.quiet:
        print(f"wrote {args.out}" + (f" and {args.empirical}" if args.empirical else ""))


if __name__ == "__main__":
    main()
