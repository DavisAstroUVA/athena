"""
Plotting spectra and images (the only submodule that imports matplotlib).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from .lists import is_polarized, parse_polarization, tracks_circular
from .spectra import get_frequency

__all__ = [
    'compute_nulnu_error',
    'compute_lnu_error',
    'compute_counts_error',
    'compute_pol_frac_error',
    'compute_pol_angle_error',
    'compute_q_error',
    'compute_u_error',
    'compute_v_error',
    'compute_flux_frac_error',
    'polarization_requested',
    'circular_requested',
    'check_polarization',
    'plot_frequency',
    'plot_theta',
    'plot_phi',
    'make_plot',
    'subsample_polarization',
    'plot_image',
]

def compute_nulnu_error(intensity, nu, errors=None):
    """
    Compute nu*L_nu and error
    """

    y = intensity[0,:]*nu
    if errors is not None:
        err = errors[0,:]*nu
        return y, err
    return y, None

def compute_lnu_error(intensity, errors=None):
    """
    Compute L_nu and error
    """
    y = intensity[0,:]
    if errors is not None:
        yerr = errors[0,:]
        return y, yerr
    return y, None

def compute_counts_error(intensity, nu, errors=None):
    """
    Compute L_nu and error
    """
    h = 6.62607015e-27
    y = intensity[0,:]/(h*nu)
    if errors is not None:
        yerr = errors[0,:]/(h*nu)
        return y, yerr
    return y, None

def compute_pol_frac_error(intensity,errors=None):
    """
    Compute fractional polarization and error (%)
    """
    i = intensity[0,:]
    q = intensity[1,:]
    u = intensity[2,:]
    frac = np.sqrt(q**2+u**2)/i
    if errors is not None:
        ei = errors[0,:]
        eq = errors[1,:]
        eu = errors[2,:]
        err = np.sqrt(((q*eq)**2+(u*eu)**2)/(q**2+u**2) + (q**2+u**2)*(ei/i)**2)/i
        return frac*100, err*100.
    return frac*100, None

def compute_pol_angle_error(intensity,errors=None):
    """
    Compute polarization angle and error if requested (degrees)
    """
    q = intensity[1,:]
    u = intensity[2,:]
    angle = 90./np.pi*np.arctan2(u,q)
    if errors is not None:
        eq = errors[1,:]
        eu = errors[2,:]
        err = 90./np.pi*np.sqrt(((u*eu)**2+(q*eq)**2))/(q**2+u**2)
        return angle, err
    return angle, None

def compute_q_error(intensity,errors=None):
    """
    Compute q=Q/I and error if requested

    Q > 0 for polarization along the meridian direction l, as stored by the code.
    """
    i = intensity[0,:]
    q = intensity[1,:]
    frac = q/i
    if errors is not None:
        ei = errors[0,:]
        eq = errors[1,:]
        err = np.sqrt(eq**2 + (q**2)*(ei/i)**2)/i
        return frac, err
    return frac, None

def compute_u_error(intensity,errors=None):
    """
    Compute u=U/I and error if requested
    """
    i = intensity[0,:]
    u = intensity[2,:]
    frac = u/i
    if errors is not None:
        ei = errors[0,:]
        eu = errors[1,:]
        err = np.sqrt(eu**2 + (u**2)*(ei/i)**2)/i
        return frac, err
    return frac, None

def compute_v_error(intensity,errors=None):
    """
    Compute v=V/I and error if requested.  Only present for circular spectra.
    """
    i = intensity[0,:]
    v = intensity[3,:]
    frac = v/i
    if errors is not None:
        ei = errors[0,:]
        ev = errors[3,:]
        err = np.sqrt(ev**2 + (v**2)*(ei/i)**2)/i
        return frac, err
    return frac, None

def compute_flux_frac_error(intensity,mufaces,errors=None):
    """
    Compute I/F and error if requested
    """
    i = intensity[0,:]
    mu = 0.5*(mufaces[1:]+mufaces[:-1])
    dmu = mufaces[1:]-mufaces[:-1]
    flux = np.sum(mu*dmu*i)
    frac = i/flux/2.
    if errors is not None:
        ei = errors[0,:]
        err = np.sqrt(ei**2 + np.sum((dmu*mu*ei)**2)*i**2/flux**2)/flux/2.
        return frac, err
    return frac, None

def polarization_requested(yunit):
    """
    Determine whether code requires polarization based on requested y unit
    """
    return yunit in ('polfrac', 'polangle', 'q', 'u', 'v')


def circular_requested(yunit):
    """
    Determine whether the requested y unit needs Stokes V, i.e. a circular file
    """
    return yunit == 'v'


def check_polarization(header, yunit, kind='spectrum'):
    """
    Check that a spectrum or image carries the Stokes parameters yunit needs.

    Returns True when the request can be satisfied, and prints why when it cannot.
    """
    if not polarization_requested(yunit):
        return True
    mode = parse_polarization(header['polarized'])
    if not is_polarized(mode):
        print("Error: polarization output "+yunit+" requested for unpolarized "+kind+".")
        return False
    if circular_requested(yunit) and not tracks_circular(mode):
        print("Error: output "+yunit+" requested but "+kind+" mode is '"+mode
              +"', which carries no Stokes V.")
        return False
    return True

def plot_frequency(spectrum, imu='sum', iphi='ave', xunit='kev', yunit='nulnu',
                   plterr=True, nu=None, rebinx=None):
    """
    Generate plot versus frequency (or equivalent).
    """

    # Set up x axis as bin midpoints
    xfaces = spectrum['xfaces']
    x = 0.5*(xfaces[1:]+xfaces[:-1])
    if nu is None:
        nu = get_frequency(spectrum['units'],xfaces)

    # Initialize x labels
    xlabel = None
    if xunit == 'kev':
        xlabel = r"$E~{\rm (keV)}$"
    if xunit == 'ev':
        xlabel = r"$E~{\rm (eV)}$"
    if xunit == 'nu':
        xlabel = r"$\nu~{\rm (Hz)}$"
    if xunit == 'lambda':
        # the unit as a plain Unicode character outside the math: the mathtext form
        # \AA needs a closing brace this line lacked and is not in every math font
        xlabel = "$\\lambda$ (Å)"

    # Check if error requested and stored
    if plterr:
        if spectrum['yerror'] != "true":
            print("Warning: error requested but not computed in spectrum.\n")
            plterr = False

    # Check whether spectrum has required polarization data
    if not check_polarization(spectrum, yunit):
        return None

    # Compute intensity spectrum
    intensity = spectrum['intensity']
    if plterr:
        errors = spectrum['errors']
    else:
        errors = None

    # Selection for azimuthal angle
    if ((iphi == 'ave') or (iphi == 'sum')):
        norm = 1./float(spectrum['nphi'])
        if iphi == 'sum':
            norm *= 2.*np.pi
        intensity = np.sum(intensity,axis=1)*norm
        if plterr:
            errors = np.sqrt(np.sum((errors)**2,axis=1))*norm
    else:
        iphi = int(iphi)
        intensity = intensity[:,iphi,:,:]
        if plterr:
            errors = errors[:,iphi,:,:]

    # Selection for polar angle
    if imu == 'sum':
        nmu = spectrum['nmu']
        mumid = 0.5*(spectrum['mufaces'][1:]+spectrum['mufaces'][:-1])
        intensity = np.tensordot(mumid,intensity,axes=[0,1])/nmu
        if plterr:
            errors = np.sqrt(np.tensordot((mumid)**2,(errors)**2,axes=[0,1]))/nmu
    else:
        imu = int(imu)
        intensity = intensity[:,imu,:]
        if plterr:
            errors = errors[:,imu,:]


    # Set y, yerr, and ylabel according to input units
    yerr = None
    ylabel = None
    if yunit == 'nulnu':
        # nu L_nu and lambda L_lambda are the same quantity; name it after the axis
        if xunit == 'lambda':
            ylabel = r"$\lambda L_\lambda~{\rm (erg/s)}$"
        else:
            ylabel = r"$\nu L_\nu~{\rm (erg/s)}$"
        y, yerr = compute_nulnu_error(intensity,nu,errors)
    elif yunit == 'lnu':
        ylabel = r"$L_\nu~{\rm (erg/s/Hz)}$"
        y, yerr = compute_lnu_error(intensity,errors)
    elif yunit == 'counts':
        ylabel = r"$N_\nu~{\rm (counts/s/Hz)}$"
        y, yerr = compute_counts_error(intensity,nu,errors)
    elif yunit == 'polfrac':
        ylabel = r"$\rm Pol.\; Fraction \; (\%)$"
        y, yerr = compute_pol_frac_error(intensity,errors)
    elif yunit == 'polangle':
        ylabel = r"$\rm Pol.\; Angle$"
        y, yerr = compute_pol_angle_error(intensity,errors)
    elif yunit == 'q':
        ylabel = r"$Q_\nu/I_\nu$"
        y, yerr = compute_q_error(intensity,errors)
    elif yunit == 'u':
        ylabel = r"$U_\nu/I_\nu$"
        y, yerr = compute_u_error(intensity,errors)
    elif yunit == 'v':
        ylabel = r"$V_\nu/I_\nu$"
        y, yerr = compute_v_error(intensity,errors)
    else:
        print("Error: yunit ("+yunit+") not specified correctly")
        return None

    # Rebin in frequency if requested
    if rebinx is not None:
        nbin = int(rebinx)
        n_new = len(x) // nbin
        x = x[:n_new * nbin].reshape(n_new, nbin).mean(axis=1)
        y = y[:n_new * nbin].reshape(n_new, nbin).mean(axis=1)
        # yerr is None whenever plterr is off, which rebinning has no reason to require
        if yerr is not None:
            y2 = yerr**2
            y2 = y2[:n_new * nbin].reshape(n_new, nbin).mean(axis=1)
            yerr = np.sqrt(y2)

    # Return x and y variables, their labels, and possible error on y
    return x,y,yerr,xlabel,ylabel

def plot_theta(spectrum,ix,iphi='ave',xunit='mu',yunit='lnu',
               plterr=True,nu=None,verbose=False):
    """
    Generate plot versus polar angle (theta)
    """

    # Set up x axis as bin midpoints
    xfaces = spectrum['mufaces']
    x = 0.5*(xfaces[1:]+xfaces[:-1])

    # Initialize x labels
    xlabel = None
    if xunit == 'mu':
        xlabel = r"$\cos \theta$"

    # Check if error requested and stored
    if plterr:
        if spectrum['yerror'] != "true":
            print("Warning: error requested but not computed in spectrum.\n")
            plterr = False

    # Check whether spectrum has required polarization data
    if not check_polarization(spectrum, yunit):
        return None

    intensity = spectrum['intensity']
    if plterr:
        errors = spectrum['errors']
    else:
        errors = None

    # Selection for azimuthal angle
    if ((iphi == 'ave') or (iphi == 'sum')):
        norm = 1./float(spectrum['nphi'])
        if iphi == 'sum':
            norm *= 2.*np.pi
        intensity = np.sum(intensity,axis=1)*norm
        if plterr:
            errors = np.sqrt(np.sum((errors)**2,axis=1))*norm
    else:
        iphi = int(iphi)
        intensity = intensity[:,iphi,:,:]
        if plterr:
            errors = errors[:,iphi,:,:]

    # Selection for frequency
    if ix == 'sum':
        nx = spectrum['nx']
        dx = (spectrum['xfaces'][1:]-spectrum['xfaces'][:-1]).reshape(nx)
        #intensity = np.dot(dx,intensity,axis=1)
        intensity = np.tensordot(dx,intensity,axes=[0,2])
        if plterr:
            errors = np.sqrt(np.tensordot((dx)**2,(errors)**2,axes=[0,2]))
    else:
        ix = int(ix)
        intensity = intensity[:,:,ix]
        if plterr:
            errors = errors[:,:,ix]

    # Set y, yerr, and ylabel according to input units
    yerr = None
    ylabel = None
    if yunit == 'nulnu':
        ylabel = r"$\nu L_\nu {\rm (erg/s)}$"
        y, yerr = compute_nulnu_error(intensity,nu,errors)
    elif yunit == 'lnu':
        ylabel = r"$L_\nu {\rm (erg/s/Hz)}$"
        y, yerr = compute_lnu_error(intensity,errors)
    elif yunit == 'counts':
        ylabel = r"$N_\nu {\rm (counts/s/Hz)}$"
        y, yerr = compute_counts_error(intensity,nu,errors)
    elif yunit == 'polfrac':
        ylabel = r"$\rm Pol.\; Fraction \; (\%)$"
        y, yerr = compute_pol_frac_error(intensity,errors)
    elif yunit == 'polangle':
        ylabel = r"$\rm Pol.\; Angle$"
        y, yerr = compute_pol_angle_error(intensity,errors)
    elif yunit == 'q':
        ylabel = r"$Q_\nu/I_\nu$"
        y, yerr = compute_q_error(intensity,errors)
    elif yunit == 'u':
        ylabel = r"$U_\nu/I_\nu$"
        y, yerr = compute_u_error(intensity,errors)
    elif yunit == 'v':
        ylabel = r"$V_\nu/I_\nu$"
        y, yerr = compute_v_error(intensity,errors)
    elif yunit == 'fluxfrac':
        ylabel = r"$I_\nu/F_\nu$"
        y, yerr = compute_flux_frac_error(intensity,xfaces,errors)

    if verbose:
        print(yunit)
        if isinstance(ix, int):
            print("x: ", spectrum['xfaces'][ix],' ', spectrum['units'])
        else:
            print("x: ", ix)
        if isinstance(iphi, int):
            print("phi: ",spectrum['phifaces'][iphi])
        else:
            print("phi: ",iphi)

    # Return x and y variables, there labels, and possible error on y
    return x, y, yerr, xlabel, ylabel

def plot_phi(spectrum, ix, imu='sum', xunit='phi', yunit='lnu',
               plterr=True, nu=None):
    """
    Generate plot versus azimuthal angle (phi)
    """

    # Set up x axis as bin midpoints
    xfaces = spectrum['phifaces']/(2.*np.pi)
    x = 0.5*(xfaces[1:]+xfaces[:-1])

    # Initialize x labels
    xlabel = None
    if xunit == 'phi':
        xlabel = r"$\phi/(2\pi)$"

    # Check if error requested and stored
    if plterr:
        if spectrum['yerror'] != "true":
            print("Warning: error requested but not computed in spectrum.\n")
            plterr = False

    # Check whether spectrum has required polarization data
    if not check_polarization(spectrum, yunit):
        return None

    intensity = spectrum['intensity']
    if plterr:
        errors = spectrum['errors']
    else:
        errors = None

    # Selection for polar angle
    if imu == 'sum':
        nmu = spectrum['nmu']
        mumid = 0.5*(spectrum['mufaces'][1:]+spectrum['mufaces'][:-1])
        intensity = np.tensordot(mumid,intensity,axes=[0,2])/nmu
        if plterr:
            errors = np.sqrt(np.tensordot((mumid)**2,(errors)**2,axes=[0,2]))/nmu
    else:
        imu = int(imu)
        intensity = intensity[:,:,imu,:]
        if plterr:
            errors = errors[:,:,imu,:]

    # Selection for frequency
    if ix == 'sum':
        nx = spectrum['nx']
        dx = (spectrum['xfaces'][1:]-spectrum['xfaces'][:-1]).reshape(nx)
        #intensity = np.dot(dx,intensity,axis=1)
        intensity = np.tensordot(dx,intensity,axes=[0,2])
        if plterr:
            errors = np.sqrt(np.tensordot((dx)**2,(errors)**2,axes=[0,2]))
    else:
        ix = int(ix)
        intensity = intensity[:,:,ix]
        if plterr:
            errors = errors[:,:,ix]

    # Set y, yerr, and ylabel according to input units
    yerr = None
    ylabel = None
    if yunit == 'nulnu':
        ylabel = r"$\nu L_\nu~{\rm (erg/s)}$"
        y, yerr = compute_nulnu_error(intensity,nu,errors)
    elif yunit == 'lnu':
        ylabel = r"$L_\nu {\rm (erg/s/Hz)}$"
        y, yerr = compute_lnu_error(intensity,errors)
    elif yunit == 'counts':
        ylabel = r"$N_\nu {\rm (counts/s/Hz)}$"
        y, yerr = compute_counts_error(intensity,nu,errors)
    elif yunit == 'polfrac':
        ylabel = r"$\rm Pol.\; Fraction \; (\%)$"
        y, yerr = compute_pol_frac_error(intensity,errors)
    elif yunit == 'polangle':
        ylabel = r"$\rm Pol.\; Angle$"
        y, yerr = compute_pol_angle_error(intensity,errors)
    elif yunit == 'q':
        ylabel = r"$Q_\nu/I_\nu$"
        y, yerr = compute_q_error(intensity,errors)
    elif yunit == 'u':
        ylabel = r"$U_\nu/I_\nu$"
        y, yerr = compute_u_error(intensity,errors)
    elif yunit == 'v':
        ylabel = r"$V_\nu/I_\nu$"
        y, yerr = compute_v_error(intensity,errors)
    elif yunit == 'fluxfrac':
        ylabel = r"$I_\nu/F_\nu$"
        y, yerr = compute_flux_frac_error(intensity,xfaces,errors)

    # Return x and y variables, there labels, and possible error on y
    return x,y,yerr,xlabel,ylabel


def make_plot(x,y,yerr=None,ax=None,xmin=None,xmax=None,ymin=None,ymax=None,
              xlabel=None,ylabel=None,xscale=None,yscale=None,fmt=None,
              **kwargs):
    """
    General wrapper for plotting of Monte Carlo spectral plots
    """

    if ax is None:
        # Create figure, axis and assume a single plot window
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

    if fmt is None:
        fmt = '.'

    if yerr is not None:
        ax.errorbar(x,y,yerr=yerr,fmt=fmt,**kwargs)
    else:
        ax.plot(x,y,fmt,**kwargs)

    # Set axis labelx
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # Set axis scales
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)

    # (re)Set plot ranges
    if xmin is not None:
        if xmax is not None:
            ax.set_xlim(xmin,xmax)
        else:
            ax.set_xlim(left=xmin)
    elif xmax is not None:
        ax.set_xlim(right=xmax)

    if ymin is not None:
        if ymax is not None:
            ax.set_ylim(ymin,ymax)
        else:
            ax.set_ylim(bottom=ymin)
    elif ymax is not None:
        ax.set_ylim(top=ymax)

    return ax


def subsample_polarization(q,u,x,y,step,average):
    """
    Subsampling of itensity array
    """

    nx = len(x)
    ny = len(y)

    if not average:
        x = x[step // 2:nx:step]
        y = y[step // 2:ny:step]
        q = q[step // 2:nx:step,step // 2:ny:step]
        u = u[step // 2:nx:step,step // 2:ny:step]
        return q,u,x,y

    # too lazy to work out pythony way of doing this
    xp = np.zeros(nx // step)
    yp = np.zeros(ny // step)
    qp = np.zeros((nx // step,ny // step))
    up = np.zeros((nx // step,ny // step))
    for i in range(nx // step):
        xp[i] = np.average(x[i*step:(i+1)*step])
    for i in range(ny // step):
        yp[i] = np.average(y[i*step:(i+1)*step])
    for i in range(nx // step):
        for j in range(ny // step):
            qp[i,j] = np.average(q[i*step:(i+1)*step,j*step:(j+1)*step])
            up[i,j] = np.average(u[i*step:(i+1)*step,j*step:(j+1)*step])

    return qp,up,xp,yp


def plot_image(image, iinc, ie, itype='intensity', pvec=False, average=False, step=4,
               ax=None, normalize=True, **kwargs):
    """
    Plot an image: one inclination bin iinc and energy bin ie of it, as the quantity
    itype: 'intensity', a Stokes plane 'q', 'u' or 'v', 'polangle' or 'polfrac'.

    The image holds the Stokes planes summed like the intensity, so Q is polarized
    energy per unit area and solid angle, not a fraction.  With normalize the Stokes
    planes and the polarized intensity are divided by I, pixel by pixel, giving Q/I,
    U/I, V/I and the polarization fraction; without it they are shown as stored.  The
    polarization angle is a ratio and is the same either way.  pvec draws the
    polarization vectors, with the same normalization.
    """
    if ax is None:
        # Create figure, axis and assume a single plot window
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

    vmin = kwargs['vmin']
    vmax = kwargs['vmax']
    cmap = plt.get_cmap(kwargs['colormap'])
    # Draw on the figure that holds ax.  A stray plt.figure() here used to open a second,
    # empty figure and draw into that instead, so the caller's figure stayed blank and a
    # notebook showed two frames.
    plt.sca(ax)

    if not check_polarization(image, itype, kind='image'):
        raise RuntimeError("Polarization type requested ("+itype+
                           ") but image mode is '"
                           +parse_polarization(image['polarized'])+"'")

    # the planes of this inclination and energy; copies, since vals is edited below
    planes = np.array(image['intensity'][:,iinc,ie,:,:])
    inten = planes[0]
    if normalize:
        # a pixel with no intensity has no polarization to show
        scale = np.divide(1.0, inten, out=np.zeros_like(inten), where=inten > 0.0)
        tag = "/I"
    else:
        scale = np.ones_like(inten)
        tag = ""

    if itype == 'intensity':
        vals = inten
        clabel=r"$I$"
        if vmin is None:
            vmin = 1.e-5*np.max(vals)
    elif itype == 'q':
        vals = planes[1]*scale
        clabel=rf"$Q{tag}$"
    elif itype == 'u':
        vals = planes[2]*scale
        clabel=rf"$U{tag}$"
    elif itype == 'v':
        vals = planes[3]*scale
        clabel=rf"$V{tag}$"
    elif itype == 'polangle':
        vals = 90./np.pi*np.arctan2(planes[2],planes[1])
        vals[vals < 0.] += 360.
        if vmin is None:
            vmin = 0.
        clabel = r"$\rm Pol.\; Angle$"
    elif itype == 'polfrac':
        vals = np.sqrt(planes[1]**2+planes[2]**2)*scale
        if vmin is None:
            vmin = 0.
        clabel = r"$\rm Pol.\; Fraction$" if normalize else r"$\rm Pol.\; Intensity$"
    else:
        raise RuntimeError("Type:"+itype+" is not defined.")

    if kwargs['vnorm']:
        vm = np.max(vals)
        vals = vals / vm
        vmax = 1.
        vmin /= vm

    if kwargs['logc']:
        vals[vals <= 0.] = 1.e-20 * np.max(vals)
        norm = colors.LogNorm(vmin=vmin,vmax=vmax)
    else:
        norm = colors.Normalize(vmin=vmin,vmax=vmax)

    x = 0.5*(image['xfaces'][1:]+image['xfaces'][:-1])
    y = 0.5*(image['yfaces'][1:]+image['yfaces'][:-1])
    x_2d, y_2d = np.meshgrid(x,y)
    im = plt.pcolormesh(x_2d, y_2d, vals, cmap=cmap, norm=norm)

    # the image's own edges, or a zoom given as xmin, xmax, ymin, ymax in kwargs
    def limit(key, default):
        value = kwargs.get(key)
        return default if value is None else value
    plt.xlim(limit('xmin', image['xfaces'][0]), limit('xmax', image['xfaces'][-1]))
    plt.ylim(limit('ymin', image['yfaces'][0]), limit('ymax', image['yfaces'][-1]))
    # Positions are in code units and the axes are unlabeled unless the image records a
    # unit; only for cm are the weights known to be erg, so only then does I get units.
    unit = image.get('unit') or ''
    if unit:
        plt.xlabel(rf"$x \; (\rm {unit})$")
        plt.ylabel(rf"$y \; (\rm {unit})$")
        if itype == 'intensity' and unit == 'cm':
            clabel=r"$I \; (\rm erg/s/cm^2/sr)$"
    else:
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")

    plt.colorbar(im,label=clabel)
    plt.gca().set_aspect('equal')
    if pvec:
        if is_polarized(image['polarized']):
            q = planes[1]*scale
            u = planes[2]*scale
            q, u, x, y = subsample_polarization(q,u,x,y,step,average)
            x_pol, y_pol = np.meshgrid(x,y)

            pol_angle = 0.5 * np.arctan2(u,q)
            pol_frac = np.sqrt(q*q+u*u)
            vx = pol_frac*np.cos(pol_angle)
            vy = pol_frac*np.sin(pol_angle)

            plt.quiver(x_pol, y_pol, vx, vy, color='k',headwidth=0, headlength=0,
                       headaxislength=0, scale = None,pivot='middle')
        else:
            raise RuntimeError("Polarization vectors requested but image is unpolarized")

    return ax.figure
