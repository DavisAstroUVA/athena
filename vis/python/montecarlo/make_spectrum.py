#! /usr/bin/env python

"""
Bin photon list files into spectra with athena_mc.make_spectrum().

    make_spectrum.py NX XMIN XMAX LIST [LIST ...] [options]

The photons in the list files are binned into NX bins between XMIN and XMAX, spaced
logarithmically unless --linearx is given.  The x variable is set by --xunit: photon
energy in ev (the default) or kev, frequency nu in Hz, or wavelength lambda in Angstrom.
The spectra written are read back by plot_spectrum.py and athena_mc.read_spectrum().

How the lists are grouped
-------------------------
A run writes one list per rank and output, named <base>.proc<rank>.<output>.list, for
example xrb.out1.proc3.00002.list.  The ranks of one output together hold the photons
of that interval, so their spectra are summed.  Each output then becomes a spectrum of
its own, <base>.<output>.spec, or with --combine the outputs are averaged, weighted by
their integration times, into a single spectrum <base>.spec.  A file not named this
way is treated as an output on its own and written to its own name with .spec.

Examples
--------
One list, 100 bins from 0.1 to 100 keV, written to xrb.out1.00000.spec:

    make_spectrum.py 100 0.1 100. xrb.out1.proc0.00000.list --xunit kev

An MPI run with one output: all ranks of output 00000 summed into one spectrum, with
the statistical error of each bin, which plot_spectrum.py --ploterr needs:

    make_spectrum.py 100 0.1 100. xrb.out1.proc*.00000.list --xunit kev --yerror

A run with nout > 1: one spectrum per output, xrb.out1.00000.spec, xrb.out1.00001.spec,
and so on, for looking at the variation between intervals:

    make_spectrum.py 100 0.1 100. xrb.out1.proc*.list --xunit kev --yerror

The same lists averaged over all outputs into the lowest-noise spectrum, xrb.out1.spec:

    make_spectrum.py 100 0.1 100. xrb.out1.proc*.list --xunit kev --yerror --combine

Resolved in angle: ten bins in the cosine of the polar angle and four in azimuth, so
that plot_spectrum.py --imu and --iphi can select a viewing direction.  --anglebin
chooses how the direction is measured: cartesian (default), spherical or hybrid.

    make_spectrum.py 100 0.1 100. xrb.out1.proc*.list --xunit kev --yerror --combine \\
        --nmu 10 --mumin 0. --mumax 1. --nphi 4

Leaving photons out with a screen function.  screen.py in the working directory (or
on PYTHONPATH) defines functions that take a Photons chunk and return True for the
photons to drop; see tst/montecarlo/disk_atmosphere/screen.py for an example:

    make_spectrum.py 100 0.1 100. disk.out1.proc*.list --screen above_zmax

--calclum also prints the luminosity of each output summed directly from its lists,
independent of the binning, and --outfile overrides the default output name.
"""

# python standard modules
import argparse
import collections
import glob
import os
import re
import sys

import numpy as np

# Athena++ modules
import athena_mc as athenamc
from athena_mc import Photons

# Optional user module supplying --screen functions.  A screen.py is a per-dataset input
# kept next to the lists, so the working directory is searched as well as PYTHONPATH.
# Appended rather than prepended, so it cannot shadow an installed module.
sys.path.append(os.getcwd())
try:
    import screen
except ModuleNotFoundError:
    screen = None

# values accepted on the command line
XUNITS = ['ev', 'kev', 'nu', 'lambda']
ANGLEBINS = ['cartesian', 'spherical', 'hybrid']

# how the code names a photon list: <base>.proc<rank>.<output>.list
LIST_NAME = re.compile(r'^(?P<base>.+)\.proc(?P<rank>\d+)\.(?P<output>\d+)\.list$')

# chunks read between progress messages
PROGRESS_EVERY = 20

# the list files of one output of one run; output is None for a file not named like one
Output = collections.namedtuple('Output', ['base', 'output', 'files'])


def load_screen(name):
    """
    Return the function of the given name from the user's screen.py, or None when no
    screen was asked for.  Exits with a message saying what was looked for and where.
    """

    if name is None:
        return None
    if screen is None:
        raise SystemExit(
            f"--screen={name} needs a screen.py module providing a function "
            f"{name}(phots), and none was importable.\n"
            "Looked in the working directory, on PYTHONPATH, "
            "and alongside make_spectrum.py.\n"
            "It takes a Photons chunk and returns a boolean array that is True for "
            "the photons to leave out of the spectrum.")
    function = getattr(screen, name, None)
    if not callable(function):
        available = sorted(n for n in dir(screen)
                           if not n.startswith('_') and callable(getattr(screen, n)))
        raise SystemExit(
            f"screen.py ({getattr(screen, '__file__', 'location unknown')}) has no "
            f"function named {name!r}.\n"
            f"Defined there: {', '.join(available) if available else 'nothing callable'}")
    return function


def expand_files(patterns):
    """
    Expand any glob patterns the shell left unexpanded, keeping the order given; a
    pattern that matches nothing is kept as a name so that it is reported as missing.
    """

    files = []
    for pattern in patterns:
        files.extend(sorted(glob.glob(pattern)) or [pattern])
    return files


def group_outputs(files):
    """
    Group list files by run and output number, ranks in order within each output and
    outputs in order within each run.
    """

    groups = {}
    for file in files:
        match = LIST_NAME.match(file)
        if match:
            key = (match['base'], match['output'])
        else:
            key = (os.path.splitext(file)[0], None)
        groups.setdefault(key, []).append(file)
    return [Output(base, output, sorted(names))
            for (base, output), names in sorted(groups.items(), key=lambda kv: (
                kv[0][0], kv[0][1] or ''))]


def output_names(outputs, args):
    """
    Names of the spectrum files to write, one per output or a single one with --combine.
    An --outfile given for several outputs has the output number put before its extension.
    """

    if args.combine:
        if args.outfile is not None:
            return [args.outfile]
        return [outputs[0].base + '.spec']

    names = []
    for out in outputs:
        if args.outfile is None:
            tag = '' if out.output is None else f".{out.output}"
            names.append(f"{out.base}{tag}.spec")
        elif len(outputs) == 1:
            names.append(args.outfile)
        else:
            stem, ext = os.path.splitext(args.outfile)
            tag = os.path.basename(out.base) if out.output is None else out.output
            names.append(f"{stem}.{tag}{ext}")
    return names


def spectrum_from_list(infile, args, screen_function):
    """
    Bin one list file, streamed in chunks so that the whole list is never in memory.
    Returns the spectrum and the luminosity of the list, None unless --calclum.
    """

    reader = athenamc.read_list_generator(infile)
    header = next(reader)['header']
    print(f"  reading {infile} ({header['length']} samples)")

    spectrum = {}
    luminosity = 0.0 if args.calclum else None
    for nchunk, result in enumerate(reader):
        if nchunk > 0 and nchunk % PROGRESS_EVERY == 0:
            print(f"    {result['remaining']} samples remain")
        phlist = dict(header, list=result['chunk'], length=result['length'])
        if args.calclum:
            luminosity += athenamc.get_luminosity_list(phlist)

        phots = Photons(phlist)
        mask = None if screen_function is None else screen_function(phots)
        spec = athenamc.make_spectrum(phots, args.nx, args.xmin, args.xmax,
                                      xaxis=args.xunit, logx=not args.linearx,
                                      nmu=args.nmu, mumin=args.mumin, mumax=args.mumax,
                                      nphi=args.nphi, phimin=args.phimin,
                                      phimax=args.phimax, anglebin=args.anglebin,
                                      yerror=args.yerror, mask=mask)
        spectrum = athenamc.add_spectra(spectrum, spec)
        if result['done']:
            break

    return spectrum, luminosity


def spectrum_from_output(out, args, screen_function):
    """
    Sum the ranks of one output into its spectrum; each rank holds a share of the same
    photons, so the sum is the spectrum of all of them.  Also returns the luminosity of
    the output, None unless --calclum.
    """

    spectrum = {}
    luminosity = 0.0 if args.calclum else None
    for infile in out.files:
        spec, lum = spectrum_from_list(infile, args, screen_function)
        spectrum = athenamc.add_spectra(spectrum, spec)
        if args.calclum:
            luminosity += lum
    return spectrum, luminosity


def main(args):
    """
    Make one spectrum per output, or one for all outputs together, from the options
    returned by parse_args().
    """

    screen_function = load_screen(args.screen)

    combined = {}
    lum_dt = 0.0
    total_dt = 0.0
    for i, out in enumerate(args.outputs):
        label = out.base if out.output is None else f"{out.base} output {out.output}"
        print(f"{label}: {len(out.files)} file(s)")
        spectrum, luminosity = spectrum_from_output(out, args, screen_function)
        if args.calclum:
            print(f"  luminosity: {luminosity:e}")
            lum_dt += luminosity*spectrum['dt']
            total_dt += spectrum['dt']

        if args.combine:
            # outputs are independent intervals, so their spectra are averaged in time
            combined = athenamc.add_spectra(combined, spectrum, method='time')
        else:
            athenamc.write_spectrum(args.outnames[i], spectrum)
            print(f"  wrote {args.outnames[i]}")

    if args.combine:
        if args.calclum and len(args.outputs) > 1:
            print(f"combined luminosity: {lum_dt/total_dt:e}")
        athenamc.write_spectrum(args.outnames[0], combined)
        print(f"wrote {args.outnames[0]}")


def parse_args(argv=None):
    """
    Parse and check command-line options; exits with a usage message on bad input
    """

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('nx', type=int,
                        help='number of x bins')
    parser.add_argument('xmin', type=float,
                        help='minimum for x variable')
    parser.add_argument('xmax', type=float,
                        help='maximum for x variable')
    parser.add_argument('infile', nargs='+',
                        help='input photon list filename(s); a quoted glob is expanded')
    parser.add_argument('--xunit', default='ev', choices=XUNITS,
                        help='variable to be used for x axis')
    parser.add_argument('--linearx', action='store_true',
                        help='space the x bins linearly rather than logarithmically')
    parser.add_argument('--nmu', type=int, default=1,
                        help='number of cos(theta) bins')
    parser.add_argument('--mumin', type=float, default=0.0,
                        help='minimum cos polar angle')
    parser.add_argument('--mumax', type=float, default=1.0,
                        help='maximum cos polar angle')
    parser.add_argument('--nphi', type=int, default=1,
                        help='number of phi bins')
    parser.add_argument('--phimin', type=float, default=0.0,
                        help='minimum phi')
    parser.add_argument('--phimax', type=float, default=2.0*np.pi,
                        help='maximum phi')
    parser.add_argument('--anglebin', default='cartesian', choices=ANGLEBINS,
                        help='how the angle bins are defined')
    parser.add_argument('--yerror', action='store_true',
                        help='compute intensity errors')
    parser.add_argument('--calclum', action='store_true',
                        help='report the luminosity of each output from its lists')
    parser.add_argument('--screen',
                        help='name of a function in a user-supplied screen.py that takes '
                             'a Photons chunk and returns True for photons to leave out')
    parser.add_argument('--combine', action='store_true',
                        help='average all outputs into one spectrum, weighted by their '
                             'integration times')
    parser.add_argument('--outfile',
                        help='output filename (default: <base>.<output>.spec per output, '
                             '<base>.spec with --combine); with several outputs the '
                             'output number is put before its extension')

    args = parser.parse_args(argv)

    # checks on the binning
    if args.nx < 1 or args.nmu < 1 or args.nphi < 1:
        parser.error("nx, --nmu and --nphi must be positive")
    if args.xmin >= args.xmax:
        parser.error(f"xmin ({args.xmin}) must be less than xmax ({args.xmax})")
    if not args.linearx and args.xmin <= 0.0:
        parser.error(f"logarithmic bins need xmin > 0, got {args.xmin}; or use --linearx")

    # the inputs, grouped into outputs
    args.infile = expand_files(args.infile)
    missing = [f for f in args.infile if not os.path.isfile(f)]
    if missing:
        parser.error(f"input file(s) not found: {', '.join(missing)}")
    args.outputs = group_outputs(args.infile)

    # a combined spectrum of several runs has no natural name
    bases = {out.base for out in args.outputs}
    if args.combine and args.outfile is None and len(bases) > 1:
        parser.error("--combine over lists from different runs needs --outfile")

    # never write an output over an input list
    args.outnames = output_names(args.outputs, args)
    inputs = {os.path.realpath(f) for f in args.infile}
    clash = [n for n in args.outnames if os.path.realpath(n) in inputs]
    if clash:
        parser.error(f"output file(s) {clash} would overwrite an input file")

    return args


if __name__ == '__main__':
    main(parse_args())
