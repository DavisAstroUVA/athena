"""
Images of the escaping light for a distant observer: binning, files, combining.
"""

# standard python modules
import struct

import numpy as np
from .lists import num_stokes_stored, parse_polarization
from .binning import build_bins, get_bins, sum_into_bins, unit_direction_cartesian
from .spectra import _combine_binned

__all__ = [
    'add_images',
    'image_bins',
    'make_image',
    'write_image',
    'read_image',
]


def add_images(image1, image2, method='statistical'):
    """
    Combine two images into one, returned as a new dictionary, with the same two
    methods and the same meaning as add_spectra.  Images carry no errors.
    """

    return _combine_binned(image1, image2, method, 'image', False)

def image_bins(phots, ifaces, xfaces, yfaces):
    """
    Bin photons by the inclination of their direction and by where they fall in the
    image plane of a distant observer looking back along that direction.

    The observer sees a photon at its impact parameter, the part of its position
    perpendicular to its direction, b = r - (r.k)k. The image axes are those of the
    sky: y increases toward the +z pole, x toward decreasing azimuth, so that with
    north up, east is to the left.  Inclination is cos(theta) of the direction, signed,
    so that the two hemispheres fall in different bins.
    """

    ninc = ifaces.size - 1
    nx = xfaces.size - 1
    ny = yfaces.size - 1

    # Cartesian position and direction, whatever chart and wavevector basis the file
    # carries.
    geometry = phots.props['geometry']
    if geometry == 'spherical':
        sth = np.sin(phots.x2)
        xp = phots.x1*sth*np.cos(phots.x3)
        yp = phots.x1*sth*np.sin(phots.x3)
        zp = phots.x1*np.cos(phots.x2)
    elif geometry == 'cylindrical':
        xp = phots.x1*np.cos(phots.x2)
        yp = phots.x1*np.sin(phots.x2)
        zp = phots.x3
    else:
        xp, yp, zp = phots.x1, phots.x2, phots.x3
    kx, ky, kz = unit_direction_cartesian(phots)

    # impact parameter vector
    rdotk = xp*kx + yp*ky + zp*kz
    bx = xp - rdotk*kx
    by = yp - rdotk*ky
    bz = zp - rdotk*kz

    # The sky axes of the direction: theta-hat points away from the +z pole and phi-hat
    # toward increasing azimuth, so up is -theta-hat and left is +phi-hat.  A direction
    # along the pole has no azimuth; phi = 0 is taken, which only rotates that image.
    sth = np.sqrt(np.maximum(1.0 - kz*kz, 0.0))
    safe = np.where(sth > 0.0, sth, 1.0)
    cph = np.where(sth > 0.0, kx/safe, 1.0)
    sph = np.where(sth > 0.0, ky/safe, 0.0)
    y = -(bx*kz*cph + by*kz*sph - bz*sth)
    x = -(-bx*sph + by*cph)

    ibins = get_bins(kz, ifaces, ninc, log=False)
    xbins = get_bins(x, xfaces, nx, log=False)
    ybins = get_bins(y, yfaces, ny, log=False)
    return ibins, xbins, ybins


def make_image(phots, ninc, imin, imax, nen, emin, emax, nx, xmin, xmax, ny, ymin, ymax,
               unit='', mask=None):
    """
    Make an image (dict) of a photon list, as seen by distant observers.

    Photons are binned in the cosine of the inclination of their direction (ninc bins
    from imin to imax), in photon energy (nen logarithmic bins from emin to emax, in
    keV) and in the image plane (nx by ny pixels over xmin..xmax by ymin..ymax, in the
    units of the list's positions, code units unless unit names one for the plot's axis
    labels; see image_bins for the axes).

    intensity[0] is the surface brightness: energy per unit time, per unit pixel area
    and per steradian of observer direction, the solid angle of an inclination bin
    being 2 pi times its width in cos(theta) because the image is summed over azimuth.
    It is also per Hz when nen > 1, and integrated over the band when nen == 1, so that
    a bolometric image is not divided by an enormous bandwidth. The Stokes planes follow
    in the order Q, U, V.  mask is True for photons to leave out.
    """

    # Store the image as a dictionary
    image = {}

    # Store integration time
    image['dt'] = phots.dt

    # Store total number of photons for reference
    image['ntot'] = phots.ntot

    # Create bins for viewer inclination
    ifaces = build_bins(imin, imax, ninc, False)
    image['ninc'] = ninc
    image['ifaces'] = ifaces

    # Create bins for observed photon energy
    efaces = build_bins(emin, emax, nen, True)
    image['nen'] = nen
    image['efaces'] = efaces

    # Create bins for image plane, image will be uniform 2d array
    image['nx'] = nx
    image['ny'] = ny
    xfaces = build_bins(xmin, xmax, nx, False)
    yfaces = build_bins(ymin, ymax, ny, False)
    image['xfaces'] = xfaces
    image['yfaces'] = yfaces

    # set units
    image['unit'] = unit

    ibins, xbins, ybins = image_bins(phots, ifaces, xfaces, yfaces)
    everg = 1.6021772e-12
    ebins = get_bins(phots.energy/everg/1000., efaces, nen, True, True)

    # Create intensity grid and loop over photons to add contribution
    image['polarized'] = parse_polarization(phots.polarized)
    # Provenance from the photon list, as make_spectrum does.
    image['coord'] = getattr(phots, 'coord', None)
    image['metric_params'] = getattr(phots, 'metric_params', None) or {}
    image['frame'] = getattr(phots, 'frame', None)
    npol = num_stokes_stored(image['polarized'])
    nintens = 1 + npol
    image['nintens'] = nintens

    if mask is not None:
        ibins[mask] = -1

    intensity = np.zeros((nintens, ninc, nen, ny, nx))

    # Create mask for valid indices
    valid_phots = (ibins >= 0) & (ebins >= 0) & (xbins >= 0) & (ybins >= 0)

    # Extract valid indices and weights
    valid_i = ibins[valid_phots]
    valid_e = ebins[valid_phots]
    valid_y = ybins[valid_phots]
    valid_x = xbins[valid_phots]
    valid_weights = phots.weight[valid_phots] * phots.energy[valid_phots]

    # Accumulate each plane with a flat bincount; the Stokes planes follow the intensity
    # in the order Q, U, V
    bins = (valid_i, valid_e, valid_y, valid_x)
    shape = (ninc, nen, ny, nx)
    stokes = [phots.q, phots.u] if npol >= 2 else []
    if npol == 3:
        stokes.append(phots.v)
    intensity[0] = sum_into_bins(shape, bins, valid_weights)
    for m, spol in enumerate(stokes):
        intensity[m+1] = sum_into_bins(shape, bins, valid_weights * spol[valid_phots])

    # Normalize: per unit time, per steradian of direction, per pixel area, and per Hz
    # unless the image is a single band
    domega = 2.*np.pi*(ifaces[1:] - ifaces[:-1])
    if nen == 1:
        dnu = np.array([1.])
    else:
        h = 6.62607015e-27
        dnu = (efaces[1:] - efaces[:-1])*1000.*everg/h
    area = np.outer(yfaces[1:] - yfaces[:-1], xfaces[1:] - xfaces[:-1])
    intensity /= (phots.dt * domega[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
                  * dnu[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
                  * area[np.newaxis, np.newaxis, np.newaxis, :, :])

    image['intensity'] = intensity

    return image

def write_image(filename,image):
    """
    Writes image to output file
    """

    # Open outfile
    outfile = open(filename, 'w')

    ninc = image['ninc']
    nen = image['nen']
    nx = image['nx']
    ny = image['ny']

    # Write header information
    outfile.write("dt={:.8e}\n".format(image['dt']))
    outfile.write("ninc={:d}\n".format(ninc))
    outfile.write("nen={:d}\n".format(nen))
    outfile.write("nx={:d}\n".format(nx))
    outfile.write("ny={:d}\n".format(ny))
    # 'none' for code units: the reader scans each value from its second character, so an
    # empty value would swallow the next line
    outfile.write("unit="+(image.get('unit') or 'none')+"\n")
    outfile.write("ntot={:d}\n".format(image['ntot']))
    outfile.write("nintens={:d}\n".format(image['nintens']))
    outfile.write("polarized="+parse_polarization(image['polarized'])+"\n")
    if image.get('coord') is not None:
        outfile.write("coord="+image['coord']+"\n")
    mpars = image.get('metric_params') or {}
    if mpars:
        outfile.write("metric_params="
                      + ",".join(f"{k}={v!r}" for k, v in mpars.items()) + "\n")
    if image.get('frame') is not None:
        outfile.write("frame="+image['frame']+"\n")
    outfile.close()

    # Write binfaces
    outfile = open(filename, 'ab')
    myfmt='>'+'d'*(ninc+1)
    data=struct.pack(myfmt,*(image['ifaces']))
    outfile.write(data)
    myfmt='>'+'d'*(nen+1)
    data=struct.pack(myfmt,*(image['efaces']))
    outfile.write(data)
    myfmt='>'+'d'*(nx+1)
    data=struct.pack(myfmt,*(image['xfaces']))
    outfile.write(data)
    myfmt='>'+'d'*(ny+1)
    data=struct.pack(myfmt,*(image['yfaces']))
    outfile.write(data)
    # Write data
    nelements = image['nintens']*ninc*nen*ny*nx
    myfmt='>'+'d'*nelements
    data=struct.pack(myfmt,*(image['intensity'].reshape(nelements)))
    outfile.write(data)
    outfile.close()

def read_image(filename):
    """
    Read image and return as a dictionary
    """

    # Read raw data
    with open(filename, 'rb') as data_file:
        raw_data = data_file.read()
    raw_data_ascii = raw_data.decode('ascii', 'replace')

    image = {}
    current_index = 0

    # Function for skipping though the file
    def skip_string(expected_string):
        expected_string_len = len(expected_string)
        if raw_data_ascii[current_index:current_index+expected_string_len] != expected_string:
            raise RuntimeError('File not formatted as expected')
        return current_index+expected_string_len

    try:
        current_index = skip_string("dt=")
        end_of_line_index = current_index + 1
        while raw_data_ascii[end_of_line_index] != '\n':
            end_of_line_index += 1
        image['dt'] = list(map(float,raw_data_ascii[current_index:end_of_line_index].split(' ')))[0]
        current_index = end_of_line_index + 1
    except:
        print("Image file contains no dt entry. Setting to 1.")
        image['dt'] = 1.

    current_index = skip_string("ninc=")
    end_of_line_index = current_index + 1
    while raw_data_ascii[end_of_line_index] != '\n':
        end_of_line_index += 1
    image['ninc'] = list(map(int,raw_data_ascii[current_index:end_of_line_index].split(' ')))[0]
    current_index = end_of_line_index + 1

    current_index = skip_string("nen=")
    end_of_line_index = current_index + 1
    while raw_data_ascii[end_of_line_index] != '\n':
        end_of_line_index += 1
    image['nen'] = list(map(int,raw_data_ascii[current_index:end_of_line_index].split(' ')))[0]
    current_index = end_of_line_index + 1

    current_index = skip_string("nx=")
    end_of_line_index = current_index + 1
    while raw_data_ascii[end_of_line_index] != '\n':
        end_of_line_index += 1
    image['nx'] = list(map(int,raw_data_ascii[current_index:end_of_line_index].split(' ')))[0]
    current_index = end_of_line_index + 1

    current_index = skip_string("ny=")
    end_of_line_index = current_index + 1
    while raw_data_ascii[end_of_line_index] != '\n':
        end_of_line_index += 1
    image['ny'] = list(map(int,raw_data_ascii[current_index:end_of_line_index].split(' ')))[0]
    current_index = end_of_line_index + 1

    current_index = skip_string("unit=")
    end_of_line_index = current_index + 1
    while raw_data_ascii[end_of_line_index] != '\n':
        end_of_line_index += 1
    image['unit'] = raw_data_ascii[current_index:end_of_line_index].split(' ')[0]
    if image['unit'] == 'none':   # code units, see write_image
        image['unit'] = ''
    current_index = end_of_line_index + 1

    current_index = skip_string("ntot=")
    end_of_line_index = current_index + 1
    while raw_data_ascii[end_of_line_index] != '\n':
        end_of_line_index += 1
    image['ntot'] = list(map(int,raw_data_ascii[current_index:end_of_line_index].split(' ')))[0]
    current_index = end_of_line_index + 1

    current_index = skip_string("nintens=")
    end_of_line_index = current_index + 1
    while raw_data_ascii[end_of_line_index] != '\n':
        end_of_line_index += 1
    image['nintens'] = list(map(int,raw_data_ascii[current_index:end_of_line_index].split(' ')))[0]
    current_index = end_of_line_index + 1

    current_index = skip_string("polarized=")
    end_of_line_index = current_index + 1
    while raw_data_ascii[end_of_line_index] != '\n':
        end_of_line_index += 1
    image['polarized'] = parse_polarization(
        raw_data_ascii[current_index:end_of_line_index].split(' ')[0])
    current_index = end_of_line_index + 1

    # Optional, as in read_spectrum: added after the format was in use, and metric_params
    # is absent for a metric with no free parameters.
    image['coord'] = None
    image['metric_params'] = {}
    if raw_data_ascii.startswith("coord=", current_index):
        current_index += len("coord=")
        end_of_line_index = raw_data_ascii.find('\n', current_index)
        image['coord'] = raw_data_ascii[current_index:end_of_line_index].split(' ')[0]
        current_index = end_of_line_index + 1
    if raw_data_ascii.startswith("metric_params=", current_index):
        current_index += len("metric_params=")
        end_of_line_index = raw_data_ascii.find('\n', current_index)
        for item in raw_data_ascii[current_index:end_of_line_index].split(','):
            if '=' in item:
                key, _, val = item.partition('=')
                try:
                    image['metric_params'][key.strip()] = float(val)
                except ValueError:
                    image['metric_params'][key.strip()] = val.strip()
        current_index = end_of_line_index + 1

    # One tag for every direction-like quantity in the file: the wavevector, the
    # angle bins and the plane the Stokes parameters are referenced to.  Optional,
    # as it postdates the format.
    image['frame'] = None
    if raw_data_ascii.startswith("frame=", current_index):
        current_index += len("frame=")
        end_of_line_index = raw_data_ascii.find('\n', current_index)
        image['frame'] = raw_data_ascii[current_index:end_of_line_index].split(' ')[0]
        current_index = end_of_line_index + 1

    # Read in faces
    ninc = image['ninc']
    format_string = '>' + 'd'*(ninc+1)
    begin_index = current_index
    end_index = begin_index + 8*(ninc+1)
    image['ifaces'] = np.array(struct.unpack(format_string, raw_data[begin_index:end_index]))
    nen = image['nen']
    format_string = '>' + 'd'*(nen+1)
    begin_index = end_index
    end_index = begin_index + 8*(nen+1)
    image['efaces'] = np.array(struct.unpack(format_string, raw_data[begin_index:end_index]))
    nx = image['nx']
    format_string = '>' + 'd'*(nx+1)
    begin_index = end_index
    end_index = begin_index + 8*(nx+1)
    image['xfaces'] = np.array(struct.unpack(format_string, raw_data[begin_index:end_index]))
    ny = image['ny']
    format_string = '>' + 'd'*(ny+1)
    begin_index = end_index
    end_index = begin_index + 8*(ny+1)
    image['yfaces'] = np.array(struct.unpack(format_string, raw_data[begin_index:end_index]))

    # Read intensities
    nintens = image['nintens']
    nelements = nintens*ninc*nen*ny*nx
    format_string = '>' + 'd'*nelements
    begin_index = end_index
    end_index = begin_index + 8*nelements
    vals = np.array(struct.unpack(format_string, raw_data[begin_index:end_index]))
    image['intensity'] = vals.reshape((nintens,ninc,nen,ny,nx))
    return image
