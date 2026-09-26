#! /usr/bin/env python

"""
Plot Athena++ Monte Carlo spectra against photon energy, frequency or wavelength.

    plot_spectrum.py SPEC [SPEC ...] [options]

Reads one or more .spec files written by make_spectrum.py (or by the code's own
spectrum output) and draws them on one set of axes, saved as a .png named after the
first input unless --outfile is given.  The luminosity of each spectrum is printed.

What is plotted
---------------
--xunit picks the x variable (kev by default, or ev, nu, lambda); the spectrum is
converted if it was binned in another.  --yunit picks the quantity: nulnu (default),
lnu, counts, or for polarized spectra polfrac, polangle, q, u, v.  Axes are logarithmic
by default; --xscale/--yscale, --xmin/--xmax and --ymin/--ymax adjust them, and
--rebinx N merges N adjacent x bins.

Spectra binned in angle by make_spectrum.py --nmu/--nphi hold one curve per (mu, phi)
bin.  --imu and --iphi choose which to draw: a bin index, or 'sum' to integrate over
the angle, and for phi also 'ave' to average over the azimuthal bins.  The defaults,
--imu sum --iphi sum, give the angle-integrated spectrum, that is nu L_nu.  Several
values draw several curves.

Examples
--------
One spectrum, nu L_nu against keV, saved to xrb.out1.png:

    plot_spectrum.py xrb.out1.spec

With error bars (the spectrum must have been made with make_spectrum.py --yerror):

    plot_spectrum.py xrb.out1.spec --ploterr

Two runs overlaid with legend labels, a linear y axis and set limits:

    plot_spectrum.py scatter_1e8/xrb.out1.spec scatter_1e9/xrb.out1.spec \\
        --labels "1e8 photons" "1e9 photons" --yscale linear --xmin 1. --xmax 50.

Viewing-angle dependence: three polar bins of an angle-resolved spectrum, averaged
over azimuth, with the mu of each bin in the legend:

    plot_spectrum.py xrb.out1.spec --imu 0 4 9 --iphi ave --mulegend

Polarization: the polarization fraction, then the polarization angle, in degrees:

    plot_spectrum.py xrb.out1.spec --yunit polfrac --yscale linear
    plot_spectrum.py xrb.out1.spec --yunit polangle --yscale linear

A blackbody for comparison, with its temperature in K and an overall normalization:

    plot_spectrum.py atm.out1.spec --bbtemp 1.0e7 --bbnorm 1.0e30

--txtfile also writes each file's plotted curves to a .txt file named after it, for
plotting elsewhere; the error columns are included only with --ploterr.
"""

# python standard modules
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const

# Athena++ modules
import athena_mc as athenamc

# values accepted on the command line
XUNITS = ['ev', 'kev', 'nu', 'lambda']
YUNITS = ['nulnu', 'lnu', 'counts', 'polfrac', 'polangle', 'q', 'u', 'v']
BB_YUNITS = ['nulnu', 'lnu', 'counts']
SCALES = ['linear', 'log', 'symlog', 'logit']

# command-line options forwarded to athena_mc.make_plot for every curve
AXIS_OPTS = ('xscale', 'xmin', 'xmax', 'yscale', 'ymin', 'ymax')


def mu_bin(s):
    """
    argparse type for mu bins: 'sum' or an integer bin index
    """

    if s == 'sum':
        return s
    try:
        return int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected 'sum' or a bin index, got {s!r}") from None


def phi_bin(s):
    """
    argparse type for phi bins: 'sum', 'ave', or an integer bin index
    """

    if s in ('sum', 'ave'):
        return s
    try:
        return int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected 'sum', 'ave' or a bin index, got {s!r}") from None


def check_bins(spectrum, imu, iphi, filename):
    """
    Exit with a clear message if a requested mu or phi bin is not in the spectrum
    """

    for option, bins, nbins in (('--imu', imu, spectrum['nmu']),
                                ('--iphi', iphi, spectrum['nphi'])):
        bad = [b for b in bins if isinstance(b, int) and not 0 <= b < nbins]
        if bad:
            raise SystemExit(f"{filename}: {option} bin(s) {bad} out of range; "
                             f"file has {nbins} bin(s), indices 0-{nbins-1}")


def plot_one(spectrum, ax, xunit, yunit, imu, iphi, *, plterr=False, rebinx=None,
             mulegend=False, label=None, txtfile=None, axis_opts=None):
    """
    Plot curves corresponding to a single spectrum, one per requested (imu, iphi) pair.
    axis_opts (axis scales and limits) are forwarded to athena_mc.make_plot.
    """

    if axis_opts is None:
        axis_opts = {}

    # Convert xaxis, if needed
    if xunit != spectrum['units']:
        athenamc.convert_xaxis(xunit, spectrum)

    # bin midpoints for legend entries
    mumid = 0.5*(spectrum['mufaces'][1:] + spectrum['mufaces'][:-1])
    phimid = 0.5*(spectrum['phifaces'][1:] + spectrum['phifaces'][:-1])

    # name the angles in the legend whenever curves would otherwise share a label
    show_mu = mulegend or len(imu) > 1
    show_phi = len(iphi) > 1

    # columns and header names for txtfile, collected as curves are plotted so that
    # skipped curves, rebinning, and missing errors do not leave gaps
    columns = []
    names = []

    for iphv in iphi:
        for imuv in imu:

            result = athenamc.plot_frequency(spectrum, imuv, iphv, plterr=plterr, xunit=xunit,
                                             yunit=yunit, rebinx=rebinx)
            # skip if no data
            if result is None:
                print(f"  skipping yunit={yunit!r} at imu={imuv}, iphi={iphv}")
                continue
            x, y, yerr, xlabel, ylabel = result

            # legend entry: file label plus angles, as needed
            parts = [] if label is None else [label]
            if show_mu:
                parts.append(f"μ={mumid[imuv]:.2f}" if isinstance(imuv, int) else f"μ {imuv}")
            if show_phi:
                parts.append(f"φ={phimid[iphv]:.2f}" if isinstance(iphv, int) else f"φ {iphv}")
            athenamc.make_plot(x, y, yerr=yerr, xlabel=xlabel, ylabel=ylabel, ax=ax,
                               label=", ".join(parts) or None, **axis_opts)

            if txtfile is not None:
                if not columns:
                    columns.append(x)
                    names.append('x')
                tag = f"imu={imuv},iphi={iphv}"
                columns.append(y)
                names.append(f"y({tag})")
                # error columns only when errors were plotted
                if yerr is not None:
                    columns.append(yerr)
                    names.append(f"yerr({tag})")

    if columns:
        np.savetxt(txtfile, np.column_stack(columns), header=" ".join(names))
        print(f"  wrote {txtfile}")


def plot_blackbody(ax, xfaces, xunit, yunit, bbtemp, bbnorm, *, imu=None, iphi=None):
    """
    Plot blackbody for comparison on x-axis bin faces xfaces, given in xunit
    """

    # Compute frequency and x
    x = 0.5*(xfaces[1:] + xfaces[:-1])
    nu = athenamc.get_frequency(xunit, xfaces)

    # physical constants (cgs), from astropy
    c_cgs = const.c.cgs.value
    kb_cgs = const.k_B.cgs.value
    h_cgs = const.h.cgs.value
 

    # Plot blackbody spectrum
    ybb = (bbnorm*2*h_cgs/c_cgs**2*nu**3
           / (np.exp(h_cgs*nu/(kb_cgs*bbtemp)) - 1.0))
    if iphi is not None and 'sum' in iphi:
        ybb *= 2 * np.pi
    if imu is not None and 'sum' in imu:
        ybb *= 0.5  # imu = sum return flux
    if yunit == 'nulnu':
        y = ybb*nu
    elif yunit == 'lnu':
        y = ybb
    elif yunit == 'counts':
        y = ybb/(h_cgs*nu)
    else:
        raise ValueError(f"blackbody not available for yunit={yunit!r}")
    ax.plot(x, y, linestyle='-', label=f"blackbody, T={bbtemp:.3g} K")


def make_figure(args):
    """
    Read each input spectrum with athena_mc.py and plot them on a shared axis, using the
    options returned by parse_args().  Returns the figure, for a notebook to show or a
    caller to save.
    """

    axis_opts = {key: getattr(args, key) for key in AXIS_OPTS}

    fig, ax = plt.subplots()

    # plot spectra from all infiles, keeping each x grid for the blackbody
    allfaces = []
    for i, file in enumerate(args.infile):
        # read spectrum as dict from infile
        spectrum = athenamc.read_spectrum(file)
        check_bins(spectrum, args.imu, args.iphi, file)
        print(f"lumin: ({file}) {athenamc.get_luminosity(spectrum)}")

        # plot curves corresponding to this spectrum
        label = args.labels[i] if args.labels is not None else None
        txtfile = os.path.splitext(file)[0] + '.txt' if args.txtfile else None
        plot_one(spectrum, ax, args.xunit, args.yunit, args.imu, args.iphi,
                 plterr=args.ploterr, rebinx=args.rebinx, mulegend=args.mulegend,
                 label=label, txtfile=txtfile, axis_opts=axis_opts)
        allfaces.append(spectrum['xfaces'])

    # one blackbody spanning the x range of all spectra
    if args.bbtemp is not None:
        xfaces = np.unique(np.concatenate(allfaces))
        plot_blackbody(ax, xfaces, args.xunit, args.yunit, args.bbtemp, args.bbnorm,
                       imu=args.imu, iphi=args.iphi)

    # add legend only if some curve was labeled
    if ax.get_legend_handles_labels()[0]:
        ax.legend()

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

    # the raw formatter keeps the examples in the docstring as written in -h
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('infile', nargs='+',
                        help='input photon spectrum filename(s)')
    parser.add_argument('--imu', nargs='+', type=mu_bin, default=['sum'],
                        help='index of angle bin(s) to plot, or sum')
    parser.add_argument('--iphi', nargs='+', type=phi_bin, default=['sum'],
                        help='phi bin(s) to plot, or sum/ave')
    parser.add_argument('--xscale', default='log', choices=SCALES,
                        help='x-axis scale')
    parser.add_argument('--xmin', type=float,
                        help='x-axis minimum')
    parser.add_argument('--xmax', type=float,
                        help='x-axis maximum')
    parser.add_argument('--yscale', default='log', choices=SCALES,
                        help='y-axis scale')
    parser.add_argument('--ymin', type=float,
                        help='y-axis minimum')
    parser.add_argument('--ymax', type=float,
                        help='y-axis maximum')
    parser.add_argument('--rebinx', type=int,
                        help='amount to rebin x axis by')
    parser.add_argument('--xunit', default='kev', choices=XUNITS,
                        help='variable to be used for x axis')
    parser.add_argument('--yunit', default='nulnu', choices=YUNITS,
                        help='variable to be used for y axis')
    parser.add_argument('--ploterr', action='store_true',
                        help='plot intensity with error bar')
    parser.add_argument('--outfile',
                        help='output filename for spectrum '
                             '(default: first input with .png extension)')
    parser.add_argument('--bbtemp', type=float,
                        help='blackbody temperature (K)')
    parser.add_argument('--bbnorm', default=1.0, type=float,
                        help='blackbody normalization')
    parser.add_argument('--mulegend', action='store_true',
                        help='add mu values to the legend')
    parser.add_argument('--labels', nargs='+',
                        help='legend label for each input file')
    parser.add_argument('--txtfile', action='store_true',
                        help='write the plotted curves for each input file to a .txt file '
                             'named after it (.spec replaced by .txt); error columns are '
                             'included only with --ploterr')

    args = parser.parse_args(argv)

    # checks that involve more than one option
    if args.labels is not None and len(args.labels) != len(args.infile):
        parser.error(f"number of labels ({len(args.labels)}) does not match "
                     f"number of input files ({len(args.infile)})")
    if args.bbtemp is not None and args.yunit not in BB_YUNITS:
        parser.error(f"--bbtemp only works with --yunit {', '.join(BB_YUNITS)}")

    # never write an output over an input spectrum
    if args.outfile is None:
        args.outfile = os.path.splitext(args.infile[0])[0] + '.png'
    inputs = {os.path.realpath(f) for f in args.infile}
    if os.path.realpath(args.outfile) in inputs:
        parser.error(f"--outfile {args.outfile} would overwrite an input file")
    if args.txtfile:
        clash = [f for f in args.infile
                 if os.path.realpath(os.path.splitext(f)[0] + '.txt') in inputs]
        if clash:
            parser.error(f"--txtfile would overwrite input file(s) {clash}")

    return args


if __name__ == '__main__':
    main(parse_args())
