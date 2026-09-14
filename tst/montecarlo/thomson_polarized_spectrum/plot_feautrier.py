#! /usr/bin/env python

"""
Plot athena++ spectrum
"""

# python standard modules
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#from scipy import interpolate

# Athena++ modules
import athena_mc as athenamc
import feautrier as feaut

def interp_feaut(mu0,mu,varin):
    """
    Interpolate feautrier solution in angle
    """
    nnu = len(varin[:,0])
    varout  = np.zeros(nnu)
    for i in range(nnu):
        varout[i] = np.interp(mu0,mu,varin[i,:])
    return varout

# athena_mc names the polarization y-axis options differently from the retired
# athena_mc_spec module this script used to import
YUNIT_ALIAS = {'frac': 'polfrac', 'angle': 'polangle'}

def plot_panel(spectrum, imu, ax, yunit, iphi, xunit, plterr, **limits):
    """
    Draw one spectrum panel.  athena_mc.plot_frequency returns the data rather than
    plotting it, so make_plot does the drawing.
    """
    yunit = YUNIT_ALIAS.get(yunit, yunit)
    result = athenamc.plot_frequency(spectrum, imu, iphi, xunit=xunit, yunit=yunit,
                                     plterr=plterr)
    if result is None:
        return None
    x, y, yerr, xlabel, ylabel = result
    athenamc.make_plot(x, y, yerr=yerr, xlabel=xlabel, ylabel=ylabel, ax=ax, **limits)
    return x

# Main function
def main(**kwargs):

    # Use latex labels
    plt.rc('text',usetex=True)
    plt.rc('font', **{'family' :"serif"})

    # filenames for io
    infile = kwargs.pop('infile')
    outfile = kwargs.pop('outfile')

    # read spectrum as dict from infile
    spectrum = athenamc.read_spectrum(infile)
    # compute nu, assuming spectrum stored as ev
    xfaces = spectrum['xfaces']
    h = 6.62607015e-27
    everg = 1.6021772e-12
    nu = 0.5*(xfaces[1:]+xfaces[:-1])*everg/h
    mumid = 0.5*(spectrum['mufaces'][1:]+spectrum['mufaces'][:-1])

    fnorm = kwargs.pop('fnorm')
    ffile = kwargs.pop('ffile')
    if ffile is None:
        # Compute feautrier solution
        feaut.transfer(nd=128,na=32,nu=nu)
        ffile = "feautrier.out"
    # read in feautier solution
    nuf,muf,intensf,polf = feaut.read_feautrier(ffile)
    evf = nuf*h/everg

    def imu_handler(imu):
        if (len(imu) > 1):
            # loop over all imu in the array
            slist = imu.strip(('[]')).split(",")
            ilist = [int(i) for i in slist]
        else:
            ilist = [int(imu)]
        return ilist

    # Get imu or imus for plotting different polar angles
    ilist = imu_handler(kwargs.pop('imu'))

    # Split the remaining arguments into what selects the data and what shapes the axes;
    # make_plot forwards anything it does not recognize to matplotlib, so only the axis
    # limits and scales may remain in limits.
    yunit = kwargs.pop('yunit')
    iphi = kwargs.pop('iphi')
    xunit = kwargs.pop('xunit')
    plterr = kwargs.pop('ploterr')
    limits = dict(kwargs)

    # plot spectrum
    if yunit == 'specfrac':
        fig = plt.figure()
        gs = gridspec.GridSpec(5,1)
        ax1 = fig.add_subplot(gs[0:3,0])
        limits1 = dict(limits)
        limits1['yscale'] = 'log'
        for imu in ilist:
            plot_panel(spectrum,imu,ax1,'nulnu',iphi,xunit,plterr,**limits1)
            ax1.tick_params(labelbottom=False)
            ax1.set_xlabel("")
            iinterp = interp_feaut(mumid[imu],muf,intensf)
            ax1.plot(evf,nuf*iinterp*fnorm)

        ax2 = fig.add_subplot(gs[3:5,0])
        limits2 = dict(limits)
        limits2.pop('ymin',None)
        limits2.pop('ymax',None)
        for imu in ilist:
            plot_panel(spectrum,imu,ax2,'q',iphi,xunit,plterr,**limits2)
            pinterp = interp_feaut(mumid[imu],muf,polf)

            ax2.plot(evf,pinterp*100)

        plt.tight_layout()

    else:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for imu in ilist:
            plot_panel(spectrum,imu,ax,yunit,iphi,xunit,plterr,**limits)


    # save plot to outfile
    if outfile is None:
        outfile = infile.replace('.spec','.pdf')
    plt.savefig(outfile)
    plt.close()

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile',
        help='input photon list filename')
    parser.add_argument('--imu',
        default = None,
        help='index of angle bin to plot')
    parser.add_argument('--iphi',
        default = 'ave',
        help='controls phi bin for plot')
    parser.add_argument('--xscale',
        default = 'log',
        help='x-axis scale')
    parser.add_argument('--xmin',
        default = None,
        help='x-axis mimimum')
    parser.add_argument('--xmax',
        default = None,
        help='x-axis maximum')
    parser.add_argument('--yscale',
        default = 'linear',
        help='y-axis scale')
    parser.add_argument('--ymin',
        default = None,
        help='y-axis mimimum')
    parser.add_argument('--ymax',
        default = None,
        help='y-axis maximum')
    parser.add_argument('--xunit',
        default='kev',
        help='variable to be used for x axis: ev, kev, nu, lambda')
    parser.add_argument('--yunit',
        default='frac',
        help='variable to be used for y axis: frac, angle')
    parser.add_argument('--ploterr',
        action='store_true',
        help='plot intensity with error bar')
    parser.add_argument('--fnorm',
        type = float,
        default = 1.,
        help='feautrier area normalization')
    parser.add_argument('--ffile',
        default = None,
        help='feautrier input file')
    parser.add_argument('--outfile',
        default=None,
        help='output filename for spectrum')

    args = parser.parse_args()
    main(**vars(args))
