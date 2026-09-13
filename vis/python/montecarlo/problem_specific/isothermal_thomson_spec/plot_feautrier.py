#! /usr/bin/env python

"""
Two-panel spectrum plot for the isothermal Thomson atmosphere: nu L_nu above, Q/I below,
with the Feautrier solution optionally overplotted on both.

Carries the same options as plot_spectrum.py -- several input files, mu and phi selection
including sum and ave, rebinning, error bars, a blackbody overlay, a text dump of the
plotted curves -- but fixes the figure to those two panels, which is the comparison this
problem is for.

The Feautrier curve comes from --ffile, or is computed on the spectrum's own frequency grid
when that is omitted.  --fnorm scales the intensity overlay, which is needed because the
Feautrier solution is per unit area while the Monte Carlo spectrum carries the luminosity of
whatever emitting area the run set up.
"""

# python standard modules
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Athena++ modules
import athena_mc as athenamc
import feautrier as feaut

# y units the Feautrier overlay knows how to draw, per panel
FEAUT_INTENS_UNITS = ('nulnu', 'lnu')
FEAUT_POL_UNITS = ('q',)


def imu_handler(imu):
    """
    Parse imu to determine which angles to plot.  Mirrors plot_spectrum.py, so 'sum' and
    'ave' pass through as themselves.
    """

    if imu is None:
        return [0]
    if imu in ('sum', 'ave'):
        return [imu]
    if len(imu) > 1:
        slist = imu.strip(('[]')).split(",")
        ilist = [int(i) for i in slist]
    else:
        ilist = [int(imu)]
    return ilist


def file_handler(infile):
    """
    Parse infile to determine file list
    """

    if len(infile) > 1:
        flist = infile.strip(('[]')).split(",")
    else:
        flist = [infile]
    return flist


def x_from_frequency(xunit, nu):
    """
    Frequency -> the abscissa a panel is drawn against.

    The inverse of athena_mc.get_frequency.  Needed because plot_frequency does not convert
    the spectrum, it only labels the axis: the caller converts with convert_xaxis, and the
    Feautrier curve then has to be placed in that same unit or it lands a factor of 1000
    away with the default keV axis.
    """

    h = 6.62607015e-27
    everg = 1.6021772e-12
    c = 2.99792e10

    if xunit == 'kev':
        return nu*h/everg/1000.
    if xunit == 'ev':
        return nu*h/everg
    if xunit == 'nu':
        return nu
    if xunit == 'lambda':
        return c/nu*1.e8
    raise ValueError("unknown xunit " + repr(xunit))


def interp_feaut(mu0, mu, varin):
    """
    Interpolate feautrier solution in angle
    """
    nnu = len(varin[:, 0])
    varout = np.zeros(nnu)
    for i in range(nnu):
        varout[i] = np.interp(mu0, mu, varin[i, :])
    return varout


def plot_blackbody(spectrum, ax, xunit, yunit, bbtemp, bbnorm, imu=None, iphi=None):
    """
    Plot blackbody for comparison.  Same as plot_spectrum.py.
    """

    if xunit != spectrum['units']:
        athenamc.convert_xaxis(xunit, spectrum)

    xfaces = spectrum['xfaces']
    x = 0.5*(xfaces[1:] + xfaces[:-1])
    nu = athenamc.get_frequency(spectrum['units'], xfaces)

    c = 2.99792458e10
    kb = 1.380649e-16
    h = 6.62607015e-27
    ybb = bbnorm*2*h/c**2*nu**3/(np.exp(h*nu/(kb*bbtemp)) - 1.0)
    if iphi == 'sum':
        ybb *= 2 * np.pi
    if imu == 'sum':
        ybb *= 0.5     # imu = sum returns flux
    if yunit == 'nulnu':
        ax.plot(x, ybb*nu, linestyle='-')
    elif yunit == 'lnu':
        ax.plot(x, ybb, linestyle='-')
    elif yunit == 'counts':
        ax.plot(x, ybb/(h*nu), linestyle='-')


def plot_panel(spectrum, ax, xunit, yunit, mulist, philist, plterr, rebinx, limits):
    """
    Draw one spectrum's curves onto one panel.  Returns the list of (x, y, yerr) actually
    plotted, so the caller can dump them to text.
    """

    if xunit != spectrum['units']:
        athenamc.convert_xaxis(xunit, spectrum)

    curves = []
    for iphv in philist:
        for imuv in mulist:
            # plot_frequency reports its own complaint and returns None -- an unpolarized
            # spectrum asked for q, say.  Unpacking that blindly buries the useful message
            # under a TypeError, so skip the curve and let the panel draw what it can.
            result = athenamc.plot_frequency(
                spectrum, imuv, iphv, plterr=plterr, xunit=xunit, yunit=yunit,
                rebinx=rebinx)
            if result is None:
                print("  skipping yunit={0!r} at imu={1}, iphi={2}".format(
                    yunit, imuv, iphv))
                continue
            x, y, yerr, xlabel, ylabel = result
            athenamc.make_plot(x, y, yerr=yerr, xlabel=xlabel, ylabel=ylabel, ax=ax,
                               **limits)
            curves.append((x, y, yerr))
    return curves


def overlay_feautrier(ax, mulist, mumid, muf, evf, var, fnorm=1.0):
    """
    Overplot the Feautrier solution at each requested mu.

    Silently does nothing for 'sum' or 'ave': the Feautrier solution is angle resolved, so
    there is no single curve to compare a summed or averaged spectrum against.  The caller
    warns once rather than here, so the message does not repeat per panel.
    """

    for imu in mulist:
        if imu in ('sum', 'ave'):
            continue
        ax.plot(evf, var(mumid[imu])*fnorm)


# Main function
def main(**kwargs):

    # filenames for io
    infile = kwargs.pop('infile')
    files = file_handler(infile)
    outfile = kwargs.pop('outfile')
    if outfile is None:
        outfile = files[0].replace('.spec', '.png')

    # plot parameters
    plterr = kwargs.pop('ploterr')
    xunit = kwargs.pop('xunit')
    yunitt = kwargs.pop('yunittop')
    yunitb = kwargs.pop('yunitbot')
    imu = kwargs.pop('imu')
    iphi = kwargs.pop('iphi')
    rebinx = kwargs.pop('rebinx')
    mulegend = kwargs.pop('mulegend')
    txtfile = kwargs.pop('txtfile')

    # blackbody overlay, as in plot_spectrum.py
    bbtemp = kwargs.pop('bbtemp')
    bbnorm = kwargs.pop('bbnorm')
    if bbtemp is not None:
        bbtemp = float(bbtemp)
        bbnorm = 1. if bbnorm is None else float(bbnorm)

    # feautrier overlay
    fnorm = kwargs.pop('fnorm')
    ffile = kwargs.pop('ffile')
    nofeaut = kwargs.pop('nofeaut')

    # per-panel axis control; x is shared, y is not
    xlim = {'xscale': kwargs.pop('xscale'), 'xmin': kwargs.pop('xmin'),
            'xmax': kwargs.pop('xmax')}
    toplim = dict(xlim, yscale=kwargs.pop('yscaletop'), ymin=kwargs.pop('ymintop'),
                  ymax=kwargs.pop('ymaxtop'))
    botlim = dict(xlim, yscale=kwargs.pop('yscalebot'), ymin=kwargs.pop('yminbot'),
                  ymax=kwargs.pop('ymaxbot'))

    mulist = imu_handler(imu)
    philist = imu_handler(iphi)

    fig = plt.figure()
    gs = gridspec.GridSpec(5, 1)
    ax1 = fig.add_subplot(gs[0:3, 0])
    ax2 = fig.add_subplot(gs[3:5, 0])

    feaut_read = False
    for fname in files:
        spectrum = athenamc.read_spectrum(fname)
        print("lumin: (" + fname + ")", athenamc.get_luminosity(spectrum))

        top = plot_panel(spectrum, ax1, xunit, yunitt, mulist, philist, plterr,
                         rebinx, toplim)
        ax1.tick_params(labelbottom=False)
        ax1.set_xlabel("")
        bot = plot_panel(spectrum, ax2, xunit, yunitb, mulist, philist, plterr,
                         rebinx, botlim)

        if bbtemp is not None:
            plot_blackbody(spectrum, ax1, xunit, yunitt, bbtemp, bbnorm, imu, iphi)

        # The Feautrier solution is read or computed once, on the first spectrum's grid,
        # and reused: it describes the atmosphere, not the run, so recomputing it per file
        # would give the same answer at the cost of another solve.
        if not nofeaut and not feaut_read:
            mumid = 0.5*(spectrum['mufaces'][1:] + spectrum['mufaces'][:-1])
            if ffile is None:
                nu = athenamc.get_frequency(spectrum['units'], spectrum['xfaces'])
                print("computing feautrier solution on the spectrum's frequency grid")
                feaut.transfer(nd=128, na=32, nu=nu)
                ffile = "feautrier.out"
            nuf, muf, intensf, polf = feaut.read_feautrier(ffile)
            evf = x_from_frequency(xunit, nuf)
            feaut_read = True

            if any(m in ('sum', 'ave') for m in mulist):
                print("note: feautrier overlay skipped for imu=sum/ave, which have no "
                      "single angle to compare against")

            if yunitt in FEAUT_INTENS_UNITS:
                scale = nuf if yunitt == 'nulnu' else 1.0
                overlay_feautrier(ax1, mulist, mumid, muf, evf,
                                  lambda m: scale*interp_feaut(m, muf, intensf), fnorm)
            if yunitb in FEAUT_POL_UNITS:
                overlay_feautrier(ax2, mulist, mumid, muf, evf,
                                  lambda m: interp_feaut(m, muf, polf))

        if txtfile is not None and top:
            cols = [top[0][0]]
            for panel in (top, bot):
                for x, y, yerr in panel:
                    cols.append(y)
                    cols.append(yerr if yerr is not None else np.zeros_like(y))
            np.savetxt(txtfile, np.column_stack(cols))

    if mulegend:
        nmu = len(mulist)
        labels = [f"mu={(mu+0.5)/nmu:.2f}" for mu in mulist if mu not in ('sum', 'ave')]
        if labels:
            ax1.legend(labels)

    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('infile',
        help = 'input photon spectrum filename(s)')
    parser.add_argument('--imu',
        default = '0',
        help = 'index of angle bin to plot; an int, a [i,j,k] list, or sum/ave')
    parser.add_argument('--iphi',
        default = 'ave',
        help = 'controls phi bin for plot')
    parser.add_argument('--xscale',
        default = 'log',
        help = 'x-axis scale')
    parser.add_argument('--xmin',
        default = None, type = float,
        help = 'x-axis minimum')
    parser.add_argument('--xmax',
        default = None, type = float,
        help = 'x-axis maximum')
    parser.add_argument('--yscaletop',
        default = 'log',
        help = 'y-axis scale, top panel')
    parser.add_argument('--yscalebot',
        default = 'linear',
        help = 'y-axis scale, bottom panel')
    parser.add_argument('--ymintop',
        default = None, type = float,
        help = 'y-axis minimum, top panel')
    parser.add_argument('--ymaxtop',
        default = None, type = float,
        help = 'y-axis maximum, top panel')
    parser.add_argument('--yminbot',
        default = None, type = float,
        help = 'y-axis minimum, bottom panel')
    parser.add_argument('--ymaxbot',
        default = None, type = float,
        help = 'y-axis maximum, bottom panel')
    parser.add_argument('--rebinx',
        default = None, type = int,
        help = 'amount to rebin x axis by')
    parser.add_argument('--xunit',
        default = 'kev',
        help = 'variable to be used for x axis: ev, kev, nu, lambda')
    parser.add_argument('--yunittop',
        default = 'nulnu',
        help = 'top panel y axis: nulnu, lnu, counts')
    parser.add_argument('--yunitbot',
        default = 'q',
        help = 'bottom panel y axis: q, u, v, polfrac, polangle')
    parser.add_argument('-ploterr',
        action = 'store_true',
        help = 'plot intensity with error bar')
    parser.add_argument('--outfile',
        default = None,
        help = 'output filename for spectrum')
    parser.add_argument('--ffile',
        default = None,
        help = 'feautrier input file; computed on the spectrum grid if omitted')
    parser.add_argument('--fnorm',
        default = 1., type = float,
        help = 'feautrier area normalization, applied to the intensity overlay')
    parser.add_argument('-nofeaut',
        action = 'store_true',
        help = 'skip the feautrier overlay entirely')
    parser.add_argument('--bbtemp',
        default = None,
        help = 'blackbody temperature')
    parser.add_argument('--bbnorm',
        default = None,
        help = 'blackbody normalization')
    parser.add_argument('-mulegend',
        action = 'store_true',
        help = 'add a legend for mu values')
    parser.add_argument('--txtfile',
        nargs = '?', const = 'out.txt', default = None,
        help = 'write the plotted curves to this text file; give the flag with no name '
               'to use out.txt, omit it entirely to write nothing')

    args = parser.parse_args()
    main(**vars(args))
