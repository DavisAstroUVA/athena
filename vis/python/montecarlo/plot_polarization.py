#! /usr/bin/env python

"""
Plot a polarized Athena++ Monte Carlo spectrum: intensity over two polarization panels.

    plot_polarization.py SPEC [SPEC ...] [options]

Three panels share the x axis.  The taller top panel is the spectrum, nu L_nu by default
on logarithmic axes.  The two lower panels are, with --panels fracangle (the default), the
polarization fraction in percent and the polarization angle in degrees, or with --panels
qu the normalized Stokes parameters Q/I and U/I; both are linear in y by default.  As in
plot_spectrum.py, several spectra go on the same axes, --imu and --iphi pick angle bins
('sum' integrates over an angle, 'ave' averages over azimuth), and --xunit converts the
x axis.  A spectrum without Stokes parameters is refused.

Feautrier comparison
--------------------
--ffile overlays the plane-parallel Thomson-atmosphere solution feautrier.py writes, the
comparison the thomson_polarized_spectrum and snake_thomson_spectrum tests make, and so
replaces plot_feautrier.py.  The file holds I_nu and the polarization degree against
frequency and cos(theta); each is interpolated to the cos(theta) at the middle of every
integer --imu bin and drawn as a line.  The intensity is scaled by --fnorm, the emitting
area for the code's L_nu.  The polarization degree is signed and is drawn as Q/I with
U = 0: positive along the meridian, negative perpendicular to it as at the limb, so the
fraction panel gets its magnitude and the angle panel 0 or 90 degrees.
--fcompute solves the atmosphere on the spectrum's frequency grid first, writing
feautrier.out in the working directory, for the default atmosphere of feautrier.py.
An --imu of 'sum' has no single cos(theta) and gets no overlay.

Examples
--------
Polarization fraction and angle of one spectrum, all angles summed:

    plot_polarization.py xrb.out1.spec

Two viewing angles with error bars and Q/I, U/I below:

    plot_polarization.py xrb.out1.spec --imu 1 3 --ploterr --panels qu --mulegend

The snake atmosphere test against the Feautrier solution for a 1e11 cm square box:

    plot_polarization.py mcsnakeatm.out1.00000.spec --imu 0 3 7 \\
        --ffile feautrier.out --fnorm 1.e22 --xunit ev --ploterr
"""

# python standard modules
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

# Athena++ modules
import athena_mc as athenamc
import feautrier as feaut
from plot_spectrum import XUNITS, SCALES, mu_bin, phi_bin, check_bins

# values accepted on the command line
IUNITS = ['nulnu', 'lnu', 'counts']
PANELS = {'fracangle': ('polfrac', 'polangle'), 'qu': ('q', 'u')}

# default y limits of the two lower panels: the full range of the fraction in percent and
# of the angle in degrees; Q/I and U/I autoscale
PANEL_LIMITS = {'fracangle': ((0., 100.), (-90., 90.)),
                'qu': ((None, None), (None, None))}

# relative heights of the intensity panel and the two polarization panels
HEIGHT_RATIOS = (3, 2, 2)

# cgs constants, as in athena_mc.make_spectrum
H_CGS = 6.62607015e-27
EVERG = 1.6021772e-12
C_CGS = 2.99792e10


def frequency_to_x(nu, xunit):
    """
    The x-axis value in xunit of a frequency in Hz
    """

    if xunit == 'kev':
        return nu*H_CGS/EVERG/1000.
    if xunit == 'ev':
        return nu*H_CGS/EVERG
    if xunit == 'lambda':
        return C_CGS/nu*1.e8
    return nu


def feautrier_curves(ffile, mu, xunit, yunit, fnorm):
    """
    The Feautrier intensity, in the top panel's yunit and scaled by fnorm, and the
    polarization degree, both at cos(theta) = mu, against the x axis in xunit.
    """

    nuf, muf, intensf, polf = feaut.read_feautrier(ffile)
    inten = np.array([np.interp(mu, muf, row) for row in intensf])
    pol = np.array([np.interp(mu, muf, row) for row in polf])
    if yunit == 'nulnu':
        inten = nuf*inten
    elif yunit == 'counts':
        inten = inten/(H_CGS*nuf)
    return frequency_to_x(nuf, xunit), fnorm*inten, pol


def plot_one(spectrum, axes, args, label=None, ffile=None):
    """
    Plot one spectrum on the three axes, one set of curves per requested (imu, iphi)
    pair, with the Feautrier solution at each integer imu when ffile is given.
    """

    if args.xunit != spectrum['units']:
        athenamc.convert_xaxis(args.xunit, spectrum)

    mumid = 0.5*(spectrum['mufaces'][1:] + spectrum['mufaces'][:-1])
    phimid = 0.5*(spectrum['phifaces'][1:] + spectrum['phifaces'][:-1])
    show_mu = args.mulegend or len(args.imu) > 1
    show_phi = len(args.iphi) > 1
    yunits = (args.yunit,) + PANELS[args.panels]
    scales = (args.yscale, args.pscale, args.pscale)
    limits = ((args.ymin, args.ymax), tuple(args.p1lim), tuple(args.p2lim))

    for iphv in args.iphi:
        for imuv in args.imu:
            parts = [] if label is None else [label]
            if show_mu:
                parts.append(f"μ={mumid[imuv]:.2f}" if isinstance(imuv, int)
                             else f"μ {imuv}")
            if show_phi:
                parts.append(f"φ={phimid[iphv]:.2f}" if isinstance(iphv, int)
                             else f"φ {iphv}")
            curve_label = ", ".join(parts) or None

            for ax, yunit, yscale, (ymin, ymax) in zip(axes, yunits, scales, limits):
                result = athenamc.plot_frequency(spectrum, imuv, iphv,
                                                 plterr=args.ploterr, xunit=args.xunit,
                                                 yunit=yunit, rebinx=args.rebinx)
                if result is None:
                    print(f"  skipping yunit={yunit!r} at imu={imuv}, iphi={iphv}")
                    continue
                x, y, yerr, xlabel, ylabel = result
                athenamc.make_plot(x, y, yerr=yerr, xlabel=xlabel, ylabel=ylabel, ax=ax,
                                   xscale=args.xscale, yscale=yscale,
                                   xmin=args.xmin, xmax=args.xmax, ymin=ymin, ymax=ymax,
                                   label=curve_label if ax is axes[0] else None)

            if ffile is not None and isinstance(imuv, int):
                xf, inten, pol = feautrier_curves(ffile, mumid[imuv], args.xunit,
                                                  args.yunit, args.fnorm)
                # the same color as the points it belongs with
                color = axes[0].get_lines()[-1].get_color()
                first = imuv == args.imu[0] and iphv == args.iphi[0]
                flabel = "Feautrier" if first else None
                axes[0].plot(xf, inten, '-', color=color, label=flabel)
                # pol is signed: it is Q/I with U = 0, negative where the polarization
                # is perpendicular to the meridian, as at the limb of the atmosphere
                if args.panels == 'fracangle':
                    axes[1].plot(xf, 100.*np.abs(pol), '-', color=color)
                    axes[2].plot(xf, np.where(pol < 0., 90., 0.), '-', color=color)
                else:
                    axes[1].plot(xf, pol, '-', color=color)
                    axes[2].plot(xf, np.zeros_like(xf), '-', color=color)


def make_figure(args):
    """
    Read each input spectrum and plot it on the three shared-x panels, using the options
    returned by parse_args().  Returns the figure.
    """

    fig = plt.figure(figsize=(6.4, 8.0))
    grid = fig.add_gridspec(3, 1, height_ratios=HEIGHT_RATIOS, hspace=0.06)
    axes = [fig.add_subplot(grid[0])]
    axes += [fig.add_subplot(grid[i], sharex=axes[0]) for i in (1, 2)]

    for i, file in enumerate(args.infile):
        spectrum = athenamc.read_spectrum(file)
        if not athenamc.check_polarization(spectrum, 'polfrac', kind='spectrum'):
            raise SystemExit(f"{file} carries no Stokes parameters (polarized="
                             f"{spectrum['polarized']}); plot_spectrum.py plots it")
        check_bins(spectrum, args.imu, args.iphi, file)
        print(f"lumin: ({file}) {athenamc.get_luminosity(spectrum)}")
        label = args.labels[i] if args.labels is not None else None
        plot_one(spectrum, axes, args, label=label, ffile=args.ffile)

    # one x label, at the bottom
    for ax in axes[:2]:
        ax.tick_params(labelbottom=False)
        ax.set_xlabel("")
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend()
    return fig


def main(args):
    """
    Make the figure and save it to args.outfile.
    """

    fig = make_figure(args)
    fig.savefig(args.outfile, bbox_inches='tight')
    plt.close(fig)


def parse_args(argv=None):
    """
    Parse and check command-line options; exits with a usage message on bad input
    """

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('infile', nargs='+',
                        help='input spectrum filename(s), with Stokes parameters')
    parser.add_argument('--imu', nargs='+', type=mu_bin, default=['sum'],
                        help='index of angle bin(s) to plot, or sum')
    parser.add_argument('--iphi', nargs='+', type=phi_bin, default=['sum'],
                        help='phi bin(s) to plot, or sum/ave')
    parser.add_argument('--panels', default='fracangle', choices=sorted(PANELS),
                        help='lower panels: polarization fraction and angle, or Q/I '
                             'and U/I')
    parser.add_argument('--xunit', default='kev', choices=XUNITS,
                        help='variable to be used for x axis')
    parser.add_argument('--yunit', default='nulnu', choices=IUNITS,
                        help='quantity in the top panel')
    parser.add_argument('--xscale', default='log', choices=SCALES,
                        help='x-axis scale')
    parser.add_argument('--xmin', type=float,
                        help='x-axis minimum')
    parser.add_argument('--xmax', type=float,
                        help='x-axis maximum')
    parser.add_argument('--yscale', default='log', choices=SCALES,
                        help='y-axis scale of the top panel')
    parser.add_argument('--ymin', type=float,
                        help='y-axis minimum of the top panel')
    parser.add_argument('--ymax', type=float,
                        help='y-axis maximum of the top panel')
    parser.add_argument('--pscale', default='linear', choices=SCALES,
                        help='y-axis scale of the two polarization panels')
    parser.add_argument('--p1lim', nargs=2, type=float, metavar=('MIN', 'MAX'),
                        help='y limits of the first polarization panel (default: 0 to '
                             '100 percent for the fraction, autoscaled for Q/I)')
    parser.add_argument('--p2lim', nargs=2, type=float, metavar=('MIN', 'MAX'),
                        help='y limits of the second polarization panel (default: -90 '
                             'to 90 degrees for the angle, autoscaled for U/I)')
    parser.add_argument('--rebinx', type=int,
                        help='amount to rebin x axis by')
    parser.add_argument('--ploterr', action='store_true',
                        help='plot with error bars')
    parser.add_argument('--mulegend', action='store_true',
                        help='add mu values to the legend')
    parser.add_argument('--labels', nargs='+',
                        help='legend label for each input file')
    parser.add_argument('--ffile',
                        help='Feautrier solution from feautrier.py to overlay')
    parser.add_argument('--fnorm', type=float, default=1.0,
                        help='factor on the Feautrier intensity, the emitting area')
    parser.add_argument('--fcompute', action='store_true',
                        help='solve the default Feautrier atmosphere on the first '
                             'spectrum\'s frequencies, writing feautrier.out here, and '
                             'overlay it')
    parser.add_argument('--outfile',
                        help='output filename for the plot '
                             '(default: first input with .png extension)')

    args = parser.parse_args(argv)

    # the lower panels' default limits depend on what they show
    if args.p1lim is None:
        args.p1lim = PANEL_LIMITS[args.panels][0]
    if args.p2lim is None:
        args.p2lim = PANEL_LIMITS[args.panels][1]

    # checks that involve more than one option
    if args.labels is not None and len(args.labels) != len(args.infile):
        parser.error(f"number of labels ({len(args.labels)}) does not match "
                     f"number of input files ({len(args.infile)})")
    if args.ffile is not None and args.fcompute:
        parser.error("--ffile and --fcompute are alternatives")
    if args.ffile is not None and not os.path.isfile(args.ffile):
        parser.error(f"--ffile {args.ffile} not found")
    if (args.ffile is not None or args.fcompute) and not any(
            isinstance(m, int) for m in args.imu):
        parser.error("a Feautrier overlay needs at least one integer --imu bin; "
                     "'sum' has no single cos(theta)")

    if args.fcompute:
        # the solution on the spectrum's own frequency grid, as the tests do
        spectrum = athenamc.read_spectrum(args.infile[0])
        # get_frequency returns the bin-center frequencies
        nu = athenamc.get_frequency(spectrum['units'], spectrum['xfaces'])
        print("computing the Feautrier solution on", len(nu), "frequencies")
        feaut.transfer(nd=128, na=32, nu=nu)
        args.ffile = "feautrier.out"

    # never write an output over an input spectrum
    if args.outfile is None:
        args.outfile = os.path.splitext(args.infile[0])[0] + '.png'
    inputs = {os.path.realpath(f) for f in args.infile}
    if os.path.realpath(args.outfile) in inputs:
        parser.error(f"--outfile {args.outfile} would overwrite an input file")

    return args


if __name__ == '__main__':
    main(parse_args())
