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
example xrb.out1.proc3.00002.list. The ranks of one output together hold the photons
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

import numpy as np

# Athena++ modules
import athena_mc as athenamc
import mc_cli

# values accepted on the command line
XUNITS = ['ev', 'kev', 'nu', 'lambda']
ANGLEBINS = ['cartesian', 'spherical', 'hybrid']



def main(args):
    """
    Make one spectrum per output, or one for all outputs together, from the options
    returned by parse_args().  The streaming, grouping and writing are mc_cli's; this
    supplies how one chunk of photons becomes a spectrum.
    """

    def bin_chunk(phots, mask):
        return athenamc.make_spectrum(phots, args.nx, args.xmin, args.xmax,
                                      xaxis=args.xunit, logx=not args.linearx,
                                      nmu=args.nmu, mumin=args.mumin, mumax=args.mumax,
                                      nphi=args.nphi, phimin=args.phimin,
                                      phimax=args.phimax, anglebin=args.anglebin,
                                      yerror=args.yerror, mask=mask)

    mc_cli.bin_outputs(args, 'make_spectrum.py', bin_chunk, athenamc.add_spectra,
                       athenamc.write_spectrum)


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

    # the inputs, grouped into outputs, and the names of the spectra to write
    mc_cli.check_inputs(parser, args, '.spec')

    return args


if __name__ == '__main__':
    main(parse_args())
