#! /usr/bin/env python

"""
Plot athena++ spectra as function of frequency, angle, photon energy, etc.
"""

# python standard modules
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Athena++ modules
import athena_mc as athenamc


def imu_handler(imu):
    """
    Parse imu to deterimine which angles to plot
    """

    if imu is None:
        return [0]
    if imu == 'sum':
        return [imu]
    if imu == 'ave':
        return [imu]
    if len(imu) > 1:
        # loop over all imu in the array
        slist = imu.strip(('[]')).split(",")
        ilist = [int(i) for i in slist]
    else:
        ilist = [imu]
    return ilist

def file_handler(infile):
    """
    Parse infile to deterimine file list
    """

    if len(infile) > 1:
        # loop over all imu in the array
        flist = infile.strip(('[]')).split(",")
    else:
        flist = [infile]
    return flist

def plot_one(spectrum, ax, xunit, yunit, imu, iphi, plterr, **kwargs):
    """
    Plot curves corresponding to single spectrum
    """

    #Convert xaxis, if needed
    if xunit != spectrum['units']:
        athenamc.convert_xaxis(xunit,spectrum)

    # check for rebinning
    rebinx = kwargs.pop('rebinx')
    mulegend = kwargs.pop('mulegend')

    # Popped unconditionally: whatever is left in kwargs is forwarded to ax.plot via
    # make_plot, which rejects any keyword it does not recognise.  Naming the file is what
    # asks for it to be written, so there is no separate flag to get out of step with it.
    txtfile = kwargs.pop('txtfile', None)

    # plot spectrum as function mu and phi
    mulist = imu_handler(imu)
    philist = imu_handler(iphi)
    if txtfile is not None:
        nmu = len(mulist)
        nphi =len(philist)
        nout = 2*nphi*nmu+1
        nx = spectrum['nx']
        print(nx,nout)
        out_arr = np.zeros((nx,nout))
        counter = 0

    for iphv in philist:
        for imuv in mulist:

            result = athenamc.plot_frequency(spectrum, imuv, iphv,
                                         plterr=plterr, xunit=xunit, yunit=yunit, rebinx=rebinx)
            # skip if no data
            if result is None:
                print("  skipping yunit={0!r} at imu={1}, iphi={2}".format(
                    yunit, imuv, iphv))
                continue
            x, y, yerr, xlabel, ylabel = result
            athenamc.make_plot(x, y, yerr=yerr, xlabel=xlabel, ylabel=ylabel, ax=ax, **kwargs)

            if txtfile is not None:
                if counter == 0:
                    out_arr[:,0] = x
                out_arr[:,2*counter+1] = y
                out_arr[:,2*counter+2] = yerr
                counter += 1

    if mulegend:
        nmu = len(mulist)

        ax.legend([f"μ={(mu+0.5)/nmu:.2f}" for mu in mulist])

    if txtfile is not None:
        np.savetxt(txtfile, out_arr)

def plot_blackbody(spectrum, ax, xunit, yunit, bbtemp, bbnorm, imu = None, iphi = None):
    """
    Plot blackbody for comparison
    """

    # Convert xaxis, if needed
    if xunit != spectrum['units']:
        athenamc.convert_xaxis(xunit, spectrum)

    # Compute frequency and x
    xfaces = spectrum['xfaces']
    x = 0.5*(xfaces[1:] + xfaces[:-1])
    nu = athenamc.get_frequency(spectrum['units'], xfaces)

    # Plot blackbody spectrum
    c = 2.99792458e10
    kb = 1.380649e-16
    h = 6.62607015e-27
    ybb = bbnorm*2*h/c**2*nu**3/(np.exp(h*nu/(kb*bbtemp)) - 1.0)
    if iphi == 'sum':
        ybb *= 2 * np.pi
    if imu == 'sum':
        ybb *= 0.5 # imu = sum return flux
    if yunit == 'nulnu':
        ax.plot(x, ybb*nu, linestyle='-')
    elif yunit == 'lnu':
        ax.plot(x, ybb, linestyle='-')
    elif yunit == 'counts':
        ax.plot(x, ybb/(h*nu), linestyle='-')

# Main function
def main(**kwargs):
    """
    Wrapper for running the plot_spectrum() function in athena_mc.py. Plotting parameters
    are specified at command line using argparse and pass via kwargs.
    """
    # Get blackbody parameters
    bbtemp = kwargs.pop('bbtemp')
    bbnorm = kwargs.pop('bbnorm')
    if bbtemp is not None:
        bbtemp = float(bbtemp)
        if bbnorm is None:
            bbnorm = 1.
        else:
            bbnorm = float(bbnorm)

    # filenames for io
    infile = kwargs.pop('infile')
    files = file_handler(infile)
    outfile = kwargs.pop('outfile')
    if outfile is None:
        outfile = files[0].replace('.spec', '.png')

    # Set plot parameters
    plterr = kwargs.pop("ploterr")
    xunit = kwargs.pop("xunit")
    yunit = kwargs.pop("yunit")
    imu = kwargs.pop("imu")
    iphi = kwargs.pop("iphi")

    # Set axis to be reused
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # plot spectra from all infiles
    for file in files:
        # read spectrum as dict from infile
        spectrum = athenamc.read_spectrum(file)
        print("lumin: ("+file+")",athenamc.get_luminosity(spectrum))

        # plot curves corresponding to this spectrum
        plot_one(spectrum, ax, xunit, yunit, imu, iphi, plterr, **kwargs)

        if bbtemp is not None:
            plot_blackbody(spectrum, ax, xunit, yunit, bbtemp, bbnorm, imu, iphi)

    # save plot to outfile
    plt.savefig(outfile)
    plt.close()

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile',
        help = 'input photon spectrum filename(s)')
    parser.add_argument('--imu',
        default = 'sum',
        help = 'index of angle bin to plot')
    parser.add_argument('--iphi',
        default = 'ave',
        help = 'controls phi bin for plot')
    parser.add_argument('--xscale',
        default = 'log',
        help = 'x-axis scale')
    parser.add_argument('--xmin',
        default = None,
        type = float,
        help = 'x-axis mimimum')
    parser.add_argument('--xmax',
        default = None,
        type = float,
        help = 'x-axis maximum')
    parser.add_argument('--yscale',
        default = 'log',
        help = 'y-axis scale')
    parser.add_argument('--ymin',
        default = None,
        type = float,
        help='y-axis mimimum')
    parser.add_argument('--ymax',
        default = None,
        type = float,
        help = 'y-axis maximum')
    parser.add_argument('--rebinx',
        default = None,
        type = int,
        help = 'amount to rebin x axis by')
    parser.add_argument('--xunit',
        default = 'kev',
        help = 'variable to be used for x axis: ev, kev, nu, lambda')
    parser.add_argument('--yunit',
        default = 'nulnu',
        help = 'variable to be used for y axis: nulnu, lnu, counts')
    parser.add_argument('-ploterr',
        action = 'store_true',
        help = 'plot intensity with error bar')
    parser.add_argument('--outfile',
        default = None,
        help = 'output filename for spectrum')
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
        nargs = '?',
        const = 'out.txt',
        default = None,
        help = 'write the plotted curves to this text file; give the flag with no name '
               'to use out.txt, omit it entirely to write nothing')

    args = parser.parse_args()
    main(**vars(args))
