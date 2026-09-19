#! /usr/bin/env python

"""
Bin photon list files into images with athena_mc.make_image().

    make_image.py XMAX YMAX LIST [LIST ...] [options]

Each photon is placed in the image plane of a distant observer looking back along its
direction, at its impact parameter: the part of its position perpendicular to its
direction, in the units of the list's positions.  The plane is --nx by --ny pixels
from -XMAX to XMAX and -YMAX to YMAX (or from --xmin and --ymin), with the sky's axes,
y toward the +z pole and x toward decreasing azimuth, north up and east left.

Observers are binned by the cosine of the inclination of the direction, --ninc bins
from --imin to --imax (default 16 over -1 to 1, so the two hemispheres are separate),
and photons by energy, --nen logarithmic bins from --emin to --emax in keV (default one
bin over everything).  The pixel values are the surface brightness, erg/s per unit area
per steradian of direction, and per Hz when there is more than one energy bin.  The
images written are read back by plot_image.py and athena_mc.read_image().

How the lists are grouped
-------------------------
As for make_spectrum.py: the ranks of one output, <base>.proc<rank>.<output>.list, are
summed into one image, <base>.<output>.img, and with --combine the outputs are averaged
weighted by their integration times into <base>.img.

Examples
--------
A 64 by 64 pixel image out to 20 (list units) of an MPI run's one output, into
xrb.out1.00000.img:

    make_image.py 20. 20. xrb.out1.proc*.00000.list --nx 64 --ny 64

All outputs of the run averaged into xrb.out1.img, with four inclination bins covering
the upper hemisphere only:

    make_image.py 20. 20. xrb.out1.proc*.list --nx 64 --ny 64 --combine \\
        --ninc 4 --imin 0. --imax 1.

Three energy bands, 1-3, 3-10 and 10-30 keV, so that plot_image.py --ie selects one:

    make_image.py 20. 20. xrb.out1.proc*.list --nx 64 --ny 64 --combine \\
        --nen 3 --emin 1. --emax 30.

A rectangular field, and a screen function from screen.py in the working directory
that returns True for photons to leave out:

    make_image.py 30. 10. disk.out1.proc*.list --nx 96 --ny 32 --screen above_zmax

Positions are in code units and the image axes are left unlabeled unless --unit names
a unit for the plot's axis labels; --outfile overrides the default output name.
"""

# python standard modules
import argparse

# Athena++ modules
import athena_mc as athenamc
import mc_cli


def main(args, screen_function=None):
    """
    Make one image per output, or one for all outputs together, from the options
    returned by parse_args().  The streaming, grouping and writing are mc_cli's; this
    supplies how one chunk of photons becomes an image.  A caller such as a notebook
    can pass its own screen function instead of naming one in screen.py.
    """

    def bin_chunk(phots, mask):
        return athenamc.make_image(phots, args.ninc, args.imin, args.imax,
                                   args.nen, args.emin, args.emax,
                                   args.nx, args.xmin, args.xmax,
                                   args.ny, args.ymin, args.ymax,
                                   unit=args.unit or '', mask=mask)

    mc_cli.bin_outputs(args, 'make_image.py', bin_chunk, athenamc.add_images,
                       athenamc.write_image, screen_function)


def parse_args(argv=None):
    """
    Parse and check command-line options; exits with a usage message on bad input
    """

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('xmax', type=float,
                        help='right edge of the image, in the units of the positions')
    parser.add_argument('ymax', type=float,
                        help='top edge of the image')
    parser.add_argument('infile', nargs='+',
                        help='input photon list filename(s); a quoted glob is expanded')
    parser.add_argument('--xmin', type=float,
                        help='left edge of the image (default: -xmax)')
    parser.add_argument('--ymin', type=float,
                        help='bottom edge of the image (default: -ymax)')
    parser.add_argument('--nx', type=int, default=16,
                        help='number of pixels across')
    parser.add_argument('--ny', type=int, default=16,
                        help='number of pixels down')
    parser.add_argument('--ninc', type=int, default=16,
                        help='number of bins in the cosine of the inclination')
    parser.add_argument('--imin', type=float, default=-1.0,
                        help='minimum cosine of the inclination')
    parser.add_argument('--imax', type=float, default=1.0,
                        help='maximum cosine of the inclination')
    parser.add_argument('--nen', type=int, default=1,
                        help='number of logarithmic energy bins')
    parser.add_argument('--emin', type=float, default=1.0e-300,
                        help='minimum photon energy (keV)')
    parser.add_argument('--emax', type=float, default=1.0e300,
                        help='maximum photon energy (keV)')
    parser.add_argument('--unit',
                        help='unit of the positions in the lists, recorded for the plot '
                             'axis labels; by default they are code units and unlabeled')
    parser.add_argument('--screen',
                        help='name of a function in a user-supplied screen.py that takes '
                             'a Photons chunk and returns True for photons to leave out')
    parser.add_argument('--calclum', action='store_true',
                        help='report the luminosity of each output from its lists')
    parser.add_argument('--combine', action='store_true',
                        help='average all outputs into one image, weighted by their '
                             'integration times')
    parser.add_argument('--outfile',
                        help='output filename (default: <base>.<output>.img per output, '
                             '<base>.img with --combine); with several outputs the '
                             'output number is put before its extension')

    args = parser.parse_args(argv)

    # checks on the binning
    if args.xmin is None:
        args.xmin = -args.xmax
    if args.ymin is None:
        args.ymin = -args.ymax
    if min(args.nx, args.ny, args.ninc, args.nen) < 1:
        parser.error("--nx, --ny, --ninc and --nen must be positive")
    if args.xmin >= args.xmax or args.ymin >= args.ymax:
        parser.error(f"need xmin < xmax and ymin < ymax, got {args.xmin}..{args.xmax} "
                     f"and {args.ymin}..{args.ymax}")
    if not -1.0 <= args.imin < args.imax <= 1.0:
        parser.error(f"need -1 <= imin < imax <= 1, got {args.imin} and {args.imax}")
    if args.emin <= 0.0 or args.emin >= args.emax:
        parser.error(f"need 0 < emin < emax, got {args.emin} and {args.emax}")

    # the inputs, grouped into outputs, and the names of the images to write
    mc_cli.check_inputs(parser, args, '.img')

    return args


if __name__ == '__main__':
    main(parse_args())
