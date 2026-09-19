#! /usr/bin/env python

"""
Plot one plane of an Athena++ Monte Carlo image written by make_image.py.

    plot_image.py IMG [options]

An .img file holds a stack of images: one per inclination bin and energy bin, each with
the intensity and, for a polarized list, the Stokes planes Q, U and possibly V.  --iinc
and --ie pick the inclination and energy bin (both default to 0; the bins are those the
image was made with, cos(inclination) increasing with --iinc).  The plot is saved as a
.png named after the input unless --outfile is given.  The axes are the sky's, north up
and east left, in the units recorded in the image.

What is plotted
---------------
--type chooses the quantity: intensity (default), the surface brightness in erg/s per
unit area per steradian; q, u or v, a Stokes plane; polfrac, sqrt(Q^2 + U^2); or
polangle, the polarization angle in degrees.  The Stokes planes are stored summed like
the intensity, so by default q, u, v and polfrac are divided by the intensity pixel by
pixel and show Q/I, U/I, V/I and the polarization fraction; --unnormalized shows them as
stored, the polarized intensity.  The angle is a ratio and is the same either way.

--xmin, --xmax, --ymin and --ymax zoom in on part of the image; the default is its
full extent.

The color scale is --colormap (hot; twilight suits the cyclic polarization angle) between
--vmin and --vmax, logarithmic with --logc, and --vnorm divides by the maximum first.
Intensity defaults to a range of five decades below its maximum; polfrac and polangle
default to a minimum of 0, which --logc cannot use, so give --vmin with it.

-p draws the polarization direction as a bar in every --step'th pixel, its length the
fraction (or polarized intensity with --unnormalized); -a averages the Stokes planes
over each block of --step pixels instead of sampling one.

Examples
--------
The intensity of the most edge-on bin of a 16-bin image, on a log scale:

    plot_image.py xrb.out1.img --iinc 7 --logc

Face-on from above (the last bin of 16 over -1 to 1), with polarization bars:

    plot_image.py xrb.out1.img --iinc 15 --logc -p --step 4 -a

The polarization fraction and angle in the second of three energy bands:

    plot_image.py xrb.out1.img --iinc 15 --ie 1 --type polfrac --vmax 0.2
    plot_image.py xrb.out1.img --iinc 15 --ie 1 --type polangle -c twilight

The polarized intensity rather than the fraction:

    plot_image.py xrb.out1.img --iinc 15 --type polfrac --unnormalized --logc --vmin 1e-8
"""

# python standard modules
import argparse
import os

import matplotlib.pyplot as plt

# Athena++ modules
import athena_mc as athenamc

# values accepted on the command line
TYPES = ['intensity', 'q', 'u', 'v', 'polfrac', 'polangle']

# command-line options forwarded to athena_mc.plot_image by name
PLOT_OPTS = ('xmin', 'xmax', 'ymin', 'ymax', 'colormap', 'vmin', 'vmax', 'vnorm', 'logc')


def make_figure(args):
    """
    Read the image and plot the requested plane, using the options returned by
    parse_args().  Returns the figure, for a notebook to show or a caller to save.
    """

    image = athenamc.read_image(args.infile)
    if not 0 <= args.iinc < image['ninc'] or not 0 <= args.ie < image['nen']:
        raise SystemExit(f"{args.infile}: --iinc {args.iinc} --ie {args.ie} out of "
                         f"range; the image has {image['ninc']} inclination and "
                         f"{image['nen']} energy bin(s)")

    fig, ax = plt.subplots()
    plot_opts = {key: getattr(args, key) for key in PLOT_OPTS}
    athenamc.plot_image(image, args.iinc, args.ie, itype=args.type, pvec=args.pvec,
                        average=args.average, step=args.step, ax=ax,
                        normalize=not args.unnormalized, **plot_opts)
    return fig


def main(args):
    """
    Make the figure and save it to args.outfile.
    """

    fig = make_figure(args)
    fig.savefig(args.outfile)
    plt.close(fig)


def parse_args(argv=None):
    """
    Parse and check command-line options; exits with a usage message on bad input
    """

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('infile',
                        help='input image filename')
    parser.add_argument('--type', default='intensity', choices=TYPES,
                        help='quantity to plot')
    parser.add_argument('--iinc', type=int, default=0,
                        help='index of the inclination bin to plot')
    parser.add_argument('--ie', type=int, default=0,
                        help='index of the energy bin to plot')
    parser.add_argument('--xmin', type=float,
                        help='left edge of the plot (default: the image edge)')
    parser.add_argument('--xmax', type=float,
                        help='right edge of the plot')
    parser.add_argument('--ymin', type=float,
                        help='bottom edge of the plot')
    parser.add_argument('--ymax', type=float,
                        help='top edge of the plot')
    parser.add_argument('-c', '--colormap', default='hot',
                        help='Matplotlib colormap; twilight suits the cyclic '
                             'polarization angle')
    parser.add_argument('--vmin', type=float,
                        help='data value at the bottom of the color scale; write '
                             '--vmin=<val> for a negative value')
    parser.add_argument('--vmax', type=float,
                        help='data value at the top of the color scale')
    parser.add_argument('--vnorm', action='store_true',
                        help='divide by the maximum before coloring')
    parser.add_argument('--logc', action='store_true',
                        help='logarithmic color scale')
    parser.add_argument('--unnormalized', action='store_true',
                        help='show q, u, v and polfrac as stored, the polarized '
                             'intensity, rather than divided by I into fractions')
    parser.add_argument('-p', '--pvec', action='store_true',
                        help='draw polarization bars')
    parser.add_argument('-a', '--average', action='store_true',
                        help='average the Stokes planes over each block of --step pixels '
                             'for the bars, instead of sampling one pixel')
    parser.add_argument('--step', type=int, default=4,
                        help='pixels between polarization bars')
    parser.add_argument('--outfile',
                        help='output filename for the plot (default: input with .png)')

    args = parser.parse_args(argv)

    if args.step < 1:
        parser.error(f"--step must be positive, got {args.step}")
    if args.logc and args.type in ('polfrac', 'polangle') and args.vmin is None:
        parser.error(f"--logc with --type {args.type} needs --vmin, since the default "
                     "minimum of 0 has no logarithm")

    # never write the plot over the image
    if args.outfile is None:
        args.outfile = os.path.splitext(args.infile)[0] + '.png'
    if os.path.realpath(args.outfile) == os.path.realpath(args.infile):
        parser.error(f"--outfile {args.outfile} would overwrite the input image")

    return args


if __name__ == '__main__':
    main(parse_args())
