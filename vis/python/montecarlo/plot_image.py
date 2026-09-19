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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Athena++ modules
import athena_mc as athenamc

# Main function
def main(**kwargs):

    # Use latex labels
    #plt.rc('text',usetex=True)
    #plt.rc('font', **{'family' :"serif"})

    # filenames for io
    infile = kwargs.pop('infile')
    outfile = kwargs.pop('outfile')
    if outfile is None:
        outfile = infile.replace('.img','.png')

    # read image
    image = athenamc.read_image(infile)

    # Set plot parameters
    iinc = kwargs.pop("iinc")
    ie = kwargs.pop("ie")
    normalize = not kwargs.pop("unnormalized")

    # Set axis to be reused
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    athenamc.plot_image(image,iinc,ie,ax=ax,normalize=normalize,**kwargs)
    #athenamc.plot_image_old(image,iinc,ax=ax)

    # save plot to outfile
    plt.savefig(outfile)
    plt.close()

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('infile',
                        help='input image filename')
    parser.add_argument('--type',
                        default = 'intensity',
                        help='variable to plot')
    parser.add_argument('--iinc',
                        default = 0,
                        type = int,
                        help='index of angle bin to plot')
    parser.add_argument('--ie',
                        default = 0,
                        type = int,
                        help='index of energy to plot')
    parser.add_argument('--xmin',
                        default = None,
                        type = float,
                        help='left edge of the plot (default: the image edge)')
    parser.add_argument('--xmax',
                        default = None,
                        type = float,
                        help='right edge of the plot')
    parser.add_argument('--ymin',
                        default = None,
                        type = float,
                        help='bottom edge of the plot')
    parser.add_argument('--ymax',
                        default = None,
                        type = float,
                        help='top edge of the plot')
    parser.add_argument('-c', '--colormap',
                        default='hot',
                        help='name of Matplotlib colormap to use instead of default; \
                              hot is default, twilight is good for cyclic variables \
                              like polarization angle')
    parser.add_argument('--vmin',
                        type=float,
                        default=None,
                        help='data value to correspond to colormap minimum; use \
                              --vmin=<val> if <val> has negative sign')
    parser.add_argument('--vmax',
                        type=float,
                        default=None,
                        help='data value to correspond to colormap maximum; use \
                              --vmax=<val> if <val> has negative sign')
    parser.add_argument('--vnorm',
                        action='store_true',
                        help='flag indicating that intensity should be normalized \
                              to maximum')
    parser.add_argument('--logc',
                        action='store_true',
                        help='flag indicating data should be colormapped logarithmically')
    parser.add_argument('--unnormalized',
                        action='store_true',
                        help='show q, u, v and polfrac as stored, the polarized \
                              intensity, rather than divided by I into fractions')
    parser.add_argument('-p', '--pvec',
                        action='store_true',
                        default=False,
                        help='flag indicating that polarization should be plotted')
    parser.add_argument('-a', '--average',
                        action='store_true',
                        default=False,
                        help='flag indicating that polarization should be averaged \
                              over steps')
    parser.add_argument('--step',
                        type=int,
                        default=4,
                        help='pixels between polarization bars')
    parser.add_argument('--outfile',
                        default=None,
                        help='output filename for the plot (default: input with .png)')

    args = parser.parse_args()
    main(**vars(args))
