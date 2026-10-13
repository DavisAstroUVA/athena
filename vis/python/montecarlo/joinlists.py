#! /usr/bin/env python

"""
Join the photon lists the ranks of an MPI run wrote into one list per run or per output.

    joinlists.py BASENAME NPROC START END [options]

A run writes one list per rank and output, BASENAME.proc<rank>.<output>.list, for
example xrb.out1.proc3.00002.list.  This reads the lists of ranks 0 to NPROC-1 for
outputs START to END and writes their photons one after another into a single file
with one header, so that later tools see one list rather than many.  The header of the
joined list is the first input's, with the counts combined:

    length   the photons in the file, summed over the inputs
    ntot     the photons launched, summed over the inputs (each rank records its own)
    dt       the integration time, summed over the outputs joined; the ranks of one
             output share a dt and it is counted once

Every input must agree on the number of columns, polarization, coordinates, metric
parameters, frame and wavevector basis, and the ranks of one output on dt.  The data
is streamed through in chunks, so a list of any size can be joined in bounded memory.

Examples
--------
Ranks 0-15 of output 0 into xrb.out1.list:

    joinlists.py xrb.out1 16 0 0

Outputs 0-9 of the same run into one list of all ten intervals, xrb.out1.list, whose dt
is the total of the ten:

    joinlists.py xrb.out1 16 0 9

The same, but one joined list per output, xrb.out1.00000.list ... xrb.out1.00009.list:

    joinlists.py xrb.out1 16 0 9 --multiout

A run in which some ranks wrote no list (none of their photons escaped), skipping the
missing files instead of stopping, and deleting the inputs once the join is written:

    joinlists.py xrb.out1 16 0 0 --skip --rm

--outfile names the output; with --multiout the output number is put before its
extension.  make_spectrum.py does not need joined lists, since it sums the ranks of
an output itself, but a joined list is smaller to keep and simpler to hand on.
"""

# python standard modules
import argparse
import collections
import os

import numpy as np

# Athena++ modules
import athena_mc as athenamc

# the list files of one output
Output = collections.namedtuple('Output', ['number', 'files'])


def list_names(basename, nproc, start, end):
    """
    The outputs START to END, each with the names of its rank lists in rank order.
    """

    return [Output(j, [f"{basename}.proc{i:d}.{j:05d}.list" for i in range(nproc)])
            for j in range(start, end + 1)]


def read_header(infile):
    """
    The header of a list file, read without touching its data.
    """

    reader = athenamc.read_list_generator(infile)
    try:
        return next(reader)['header']
    except StopIteration:
        raise SystemExit(f"{infile} could not be read") from None
    finally:
        reader.close()


def check_headers(header, other, infile, first):
    """
    Exit if a list cannot be joined onto the first one: header_match compares the
    columns, polarization, coordinates, metric and frame, and the wavevector basis is
    compared here because it also decides what the columns mean.
    """

    if not athenamc.header_match(header, other, 'list'):
        raise SystemExit(f"{infile} cannot be joined with {first}: the lists differ in "
                         "columns, polarization, coordinates, metric or frame")
    if header.get('basis') != other.get('basis'):
        raise SystemExit(f"{infile} cannot be joined with {first}: wavevector basis "
                         f"{other.get('basis')} vs {header.get('basis')}")


def fix_length(outfile, length):
    """
    Overwrite the length field of a written list in place.  Called when an input held
    fewer photons than its header promised, so the total written to the header first
    was too large.  The new value has no more digits than the old, and is padded to the
    same width with trailing spaces, which the readers ignore as the code's own files
    already carry them.
    """

    with open(outfile, 'r+b') as f:
        head = f.read(10000)
        start = head.index(b'length=') + len(b'length=')
        end = head.index(b'\n', start)
        field = str(length).encode().ljust(end - start)
        if len(field) != end - start:
            raise SystemExit(f"{outfile}: cannot correct length to {length} in place")
        f.seek(start)
        f.write(field)


def find_lists(outputs, skip):
    """
    The input lists that exist, in order, with their headers.  A missing list is dropped
    with --skip and stops the join otherwise.
    """

    present = []
    headers = {}
    for out in outputs:
        for infile in out.files:
            if os.path.isfile(infile):
                present.append(infile)
                headers[infile] = read_header(infile)
            elif skip:
                print(f"{infile} not found, skipping")
            else:
                raise SystemExit(f"{infile} not found; --skip joins without it")
    if not present:
        raise SystemExit("none of the input lists exist")
    return present, headers


def joined_totals(outputs, headers, first):
    """
    The photons in the file, the photons launched and the integration time of the join,
    checking on the way that every list can be joined with the first.
    """

    ntot = 0
    length = 0
    dt = 0.0
    for out in outputs:
        files = [f for f in out.files if f in headers]
        if not files:
            continue
        for infile in files:
            check_headers(headers[first], headers[infile], infile, first)
            ntot += headers[infile]['ntot']
            length += headers[infile]['length']
        # the ranks of one output ran the same interval, so its dt counts once
        dts = {headers[f]['dt'] for f in files}
        if len(dts) > 1:
            raise SystemExit(f"the ranks of output {out.number} have different dt: "
                             f"{sorted(dts)}")
        dt += dts.pop()
    return length, ntot, dt


def join(outputs, outfile, skip):
    """
    Join the lists of the given outputs into outfile.  Returns the files that were read,
    so that --rm removes only those.
    """

    # Headers first, so the output header can carry the totals before any data is
    # written.
    present, headers = find_lists(outputs, skip)
    first = present[0]
    header = headers[first]
    length, ntot, dt = joined_totals(outputs, headers, first)
    print(f"joining {len(present)} list(s): {length} photons out of {ntot} launched, "
          f"dt = {dt:.8e}")

    # Then the data of each input appended chunk by chunk, so no more than one chunk
    # of one file is in memory at a time.
    athenamc.write_list(outfile, dict(header, ntot=ntot, dt=dt,
                                      list=np.empty((0, header['npars']))),
                        header=True, length=length)
    written = 0
    for infile in present:
        print(f"  reading {infile}")
        reader = athenamc.read_list_generator(infile)
        next(reader)
        for result in reader:
            athenamc.write_list(outfile, dict(header, list=result['chunk']), header=False)
            written += result['length']
            if result['done']:
                break
    if written != length:
        print(f"warning: {written} photons written but the headers promised {length}; "
              f"the header of {outfile} is corrected")
        fix_length(outfile, written)
    print(f"wrote {outfile}")

    return present


def main(args):
    """
    Join the lists into one file, or one file per output with --multiout, from the
    options returned by parse_args().
    """

    outputs = list_names(args.basename, args.nproc, args.start, args.end)

    if args.multiout:
        stem, ext = os.path.splitext(args.outfile)
        joins = [([out], f"{stem}.{out.number:05d}{ext}") for out in outputs]
    else:
        joins = [(outputs, args.outfile)]

    for outs, outfile in joins:
        read = join(outs, outfile, args.skip)
        if args.rm:
            for infile in read:
                os.remove(infile)
            print(f"  removed {len(read)} input list(s)")


def parse_args(argv=None):
    """
    Parse and check command-line options; exits with a usage message on bad input
    """

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('basename',
                        help='name of the lists up to .proc, for example xrb.out1')
    parser.add_argument('nproc', type=int,
                        help='number of ranks, lists .proc0 to .proc<nproc-1>')
    parser.add_argument('start', type=int,
                        help='first output number')
    parser.add_argument('end', type=int,
                        help='last output number')
    parser.add_argument('--multiout', action='store_true',
                        help='one joined list per output rather than one for all')
    parser.add_argument('--skip', action='store_true',
                        help='join without any list that does not exist, instead of '
                             'stopping')
    parser.add_argument('--rm', action='store_true',
                        help='delete the input lists once their join is written')
    parser.add_argument('--outfile',
                        help='output filename (default: <basename>.list, or '
                             '<basename>.<output>.list per output with --multiout)')

    args = parser.parse_args(argv)

    if args.nproc < 1:
        parser.error(f"nproc must be positive, got {args.nproc}")
    if args.start < 0 or args.end < args.start:
        parser.error(f"need 0 <= start <= end, got {args.start} and {args.end}")
    if args.outfile is None:
        args.outfile = args.basename + '.list'

    # never write a join over one of its inputs
    inputs = {os.path.realpath(f)
              for out in list_names(args.basename, args.nproc, args.start, args.end)
              for f in out.files}
    if os.path.realpath(args.outfile) in inputs:
        parser.error(f"--outfile {args.outfile} is one of the input lists")

    return args


if __name__ == '__main__':
    main(parse_args())
