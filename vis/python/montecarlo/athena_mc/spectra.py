"""
Spectra: binning photons, the spectrum files, combining and luminosities.
"""

# standard python modules
import struct
import math

import numpy as np
from .lists import num_stokes_stored, parse_polarization
from .binning import (
    build_bins, get_angle_bins_cartesian, get_angle_bins_hybrid,
    get_angle_bins_spherical, get_bins, sum_into_bins)

__all__ = [
    'write_spectrum',
    'read_spectrum',
    'header_match',
    'COMBINE_METHODS',
    'ADD_SPECTRA_METHODS',
    'add_spectra',
    'convert_xaxis',
    'get_frequency',
    'get_luminosity',
    'family_key',
    'finalize_family_errors',
    'make_spectrum',
]

def write_spectrum(filename,spectrum):
    """
    Writes spectrum to output file
    """

    # Open outfile
    outfile = open(filename, 'w')

    nx = spectrum['nx']
    nmu = spectrum['nmu']
    nphi = spectrum['nphi']

    # Write header information
    outfile.write("dt={:.8e}\n".format(spectrum['dt']))
    outfile.write("nx={:d}\n".format(nx))
    outfile.write("nmu={:d}\n".format(nmu))
    outfile.write("nphi={:d}\n".format(nphi))
    outfile.write("ntot={:d}\n".format(spectrum['ntot']))
    outfile.write("nintens={:d}\n".format(spectrum['nintens']))
    outfile.write("units="+spectrum.get('units', spectrum.get('xaxis'))+"\n")
    outfile.write("polarized="+parse_polarization(spectrum['polarized'])+"\n")
    outfile.write("yerror="+spectrum['yerror']+"\n")
    # Geometry and metric last, matching the order Spectrum::WriteSpectrum uses, so a
    # spectrum written from python and one written by the code parse identically.
    if spectrum.get('coord') is not None:
        outfile.write("coord="+spectrum['coord']+"\n")
    mpars = spectrum.get('metric_params') or {}
    if mpars:
        outfile.write("metric_params="
                      + ",".join(f"{k}={v!r}" for k, v in mpars.items()) + "\n")
    if spectrum.get('frame') is not None:
        outfile.write("frame="+spectrum['frame']+"\n")
    outfile.close()

    # Write binfaces
    outfile = open(filename, 'ab')
    myfmt='>'+'d'*(nx+1)
    bin=struct.pack(myfmt,*(spectrum['xfaces']))
    outfile.write(bin)
    myfmt='>'+'d'*(nmu+1)
    bin=struct.pack(myfmt,*(spectrum['mufaces']))
    outfile.write(bin)
    myfmt='>'+'d'*(nphi+1)
    bin=struct.pack(myfmt,*(spectrum['phifaces']))
    outfile.write(bin)
    # Write data
    nelements = spectrum['nintens']*nx*nmu*nphi
    myfmt='>'+'d'*nelements
    data=struct.pack(myfmt,*(spectrum['intensity'].reshape(nelements)))
    outfile.write(data)
    if spectrum['yerror'] == 'true':
        bin=struct.pack(myfmt,*(spectrum['errors'].reshape(nelements)))
        outfile.write(bin)
    outfile.close()

def read_spectrum(filename):
    """
    Read spectrum and return as a dictionary
    """

    # Read raw data
    with open(filename, 'rb') as data_file:
        raw_data = data_file.read()
    raw_data_ascii = raw_data.decode('ascii', 'replace')

    spectrum = {}
    current_index = 0

    # Function for skipping though the file
    def skip_string(expected_string):
        expected_string_len = len(expected_string)
        if raw_data_ascii[current_index:current_index+expected_string_len] != expected_string:
            print(raw_data_ascii[current_index:current_index+expected_string_len])
            raise RuntimeError('File not formatted as expected')
        return current_index+expected_string_len

    try:
        current_index = skip_string("dt=")
        end_of_line_index = current_index + 1
        while raw_data_ascii[end_of_line_index] != '\n':
            end_of_line_index += 1
        spectrum['dt'] = list(map(float,raw_data_ascii[current_index:end_of_line_index].split(' ')))[0]
        current_index = end_of_line_index + 1
    except:
        print("Spectrum file contains no dt entry. Setting to 1.")
        spectrum['dt'] = 1.

    current_index = skip_string("nx=")
    end_of_line_index = current_index + 1
    while raw_data_ascii[end_of_line_index] != '\n':
        end_of_line_index += 1
    spectrum['nx'] = list(map(int,raw_data_ascii[current_index:end_of_line_index].split(' ')))[0]
    current_index = end_of_line_index + 1
    current_index = skip_string("nmu=")
    end_of_line_index = current_index + 1
    while raw_data_ascii[end_of_line_index] != '\n':
        end_of_line_index += 1
    spectrum['nmu'] = list(map(int,raw_data_ascii[current_index:end_of_line_index].split(' ')))[0]
    current_index = end_of_line_index + 1
    current_index = skip_string("nphi=")
    end_of_line_index = current_index + 1
    while raw_data_ascii[end_of_line_index] != '\n':
        end_of_line_index += 1
    spectrum['nphi'] = list(map(int,raw_data_ascii[current_index:end_of_line_index].split(' ')))[0]
    current_index = end_of_line_index + 1
    current_index = skip_string("ntot=")
    end_of_line_index = current_index + 1
    while raw_data_ascii[end_of_line_index] != '\n':
        end_of_line_index += 1
    spectrum['ntot'] = list(map(int,raw_data_ascii[current_index:end_of_line_index].split(' ')))[0]
    current_index = end_of_line_index + 1
    current_index = skip_string("nintens=")
    end_of_line_index = current_index + 1
    while raw_data_ascii[end_of_line_index] != '\n':
        end_of_line_index += 1
    spectrum['nintens'] = list(map(int,raw_data_ascii[current_index:end_of_line_index].split(' ')))[0]
    current_index = end_of_line_index + 1
    current_index = skip_string("units=")
    end_of_line_index = current_index + 1
    while raw_data_ascii[end_of_line_index] != '\n':
        end_of_line_index += 1
    spectrum['units'] = raw_data_ascii[current_index:end_of_line_index].split(' ')[0]
    current_index = end_of_line_index + 1
    current_index = skip_string("polarized=")
    end_of_line_index = current_index + 1
    while raw_data_ascii[end_of_line_index] != '\n':
        end_of_line_index += 1
    spectrum['polarized'] = parse_polarization(
        raw_data_ascii[current_index:end_of_line_index].split(' ')[0])
    current_index = end_of_line_index + 1
    current_index = skip_string("yerror=")
    end_of_line_index = current_index + 1
    while raw_data_ascii[end_of_line_index] != '\n':
        end_of_line_index += 1
    spectrum['yerror'] = raw_data_ascii[current_index:end_of_line_index].split(' ')[0]
    current_index = end_of_line_index + 1
    # Geometry and metric are optional: they were added after the format was in use, and
    # metric_params is absent for a metric with no free parameters.  Probed rather than
    # required so one reader handles files written before and after.
    spectrum['coord'] = None
    spectrum['metric_params'] = {}
    if raw_data_ascii.startswith("coord=", current_index):
        current_index += len("coord=")
        end_of_line_index = raw_data_ascii.find('\n', current_index)
        spectrum['coord'] = raw_data_ascii[current_index:end_of_line_index].split(' ')[0]
        current_index = end_of_line_index + 1
    if raw_data_ascii.startswith("metric_params=", current_index):
        current_index += len("metric_params=")
        end_of_line_index = raw_data_ascii.find('\n', current_index)
        for item in raw_data_ascii[current_index:end_of_line_index].split(','):
            if '=' in item:
                key, _, val = item.partition('=')
                try:
                    spectrum['metric_params'][key.strip()] = float(val)
                except ValueError:
                    spectrum['metric_params'][key.strip()] = val.strip()
        current_index = end_of_line_index + 1

    # One tag for every direction-like quantity in the file: the wavevector, the
    # angle bins and the plane the Stokes parameters are referenced to.  Optional,
    # as it postdates the format.
    spectrum['frame'] = None
    if raw_data_ascii.startswith("frame=", current_index):
        current_index += len("frame=")
        end_of_line_index = raw_data_ascii.find('\n', current_index)
        spectrum['frame'] = raw_data_ascii[current_index:end_of_line_index].split(' ')[0]
        current_index = end_of_line_index + 1

    # Read in faces
    nx = spectrum['nx']
    format_string = '>' + 'd'*(nx+1)
    begin_index = current_index
    end_index = begin_index + 8*(nx+1)
    spectrum['xfaces'] = np.array(struct.unpack(format_string,
                                                raw_data[begin_index:end_index]))
    nmu = spectrum['nmu']
    format_string = '>' + 'd'*(nmu+1)
    begin_index = end_index
    end_index = begin_index + 8*(nmu+1)
    spectrum['mufaces'] = np.array(struct.unpack(format_string,
                                                 raw_data[begin_index:end_index]))
    nphi = spectrum['nphi']
    format_string = '>' + 'd'*(nphi+1)
    begin_index = end_index
    end_index = begin_index + 8*(nphi+1)
    spectrum['phifaces'] = np.array(struct.unpack(format_string,
                                                  raw_data[begin_index:end_index]))

    # Read intensities
    nintens = spectrum['nintens']
    nelements = nintens*nx*nmu*nphi
    format_string = '>' + 'd'*nelements
    begin_index = end_index
    end_index = begin_index + 8*nelements
    vals = np.array(struct.unpack(format_string, raw_data[begin_index:end_index]))
    spectrum['intensity'] = vals.reshape((nintens,nphi,nmu,nx))
    if spectrum['yerror'] == 'true':
        begin_index = end_index
        end_index = begin_index + 8*nelements
        vals = np.array(struct.unpack(format_string, raw_data[begin_index:end_index]))
        spectrum['errors'] = vals.reshape((nintens,nphi,nmu,nx))
    return spectrum


def header_match(dict1, dict2, dict_type):
    """
    Check if headers match
    """

    match = True
    if dict1 is None:
        return False
    if dict2 is None:
        return False
    if dict_type == 'list':
        if dict1['npars'] != dict2['npars']:
            match = False
        elif dict1['polarized'] != dict2['polarized']:
            match = False
        # Geometry and metric were not compared before, so lists from different
        # coordinate systems -- or the same one with a different spin -- could be merged
        # silently.  Absent metric_params compares equal to absent, so files written
        # before the header carried it still combine with each other.
        elif dict1.get('coord') != dict2.get('coord'):
            match = False
        elif (dict1.get('metric_params') or {}) != (dict2.get('metric_params') or {}):
            match = False
        elif dict1.get('frame') != dict2.get('frame'):
            match = False
    elif dict_type == 'spec':
        if dict1['nx'] != dict2['nx']:
            match = False
        elif dict1['nmu'] != dict2['nmu']:
            match = False
        elif dict1['nphi'] != dict2['nphi']:
            match = False
        elif dict1['nintens'] != dict2['nintens']:
            match = False
        elif dict1['units'] != dict2['units']:
            match = False
        elif dict1['polarized'] != dict2['polarized']:
            match = False
        elif dict1['yerror'] != dict2['yerror']:
            match = False
        # Absent compares equal to absent, so spectra written before the header carried
        # geometry still combine with each other, while one that records a metric will not
        # silently merge with one that records a different metric.
        elif dict1.get('coord') != dict2.get('coord'):
            match = False
        elif (dict1.get('metric_params') or {}) != (dict2.get('metric_params') or {}):
            match = False
        elif dict1.get('frame') != dict2.get('frame'):
            match = False
    elif dict_type == 'image':
        if dict1['ninc'] != dict2['ninc']:
            match = False
        elif dict1['nen'] != dict2['nen']:
            match = False
        elif dict1['nx'] != dict2['nx']:
            match = False
        elif dict1['ny'] != dict2['ny']:
            match = False
        elif dict1['nintens'] != dict2['nintens']:
            match = False
        elif dict1['unit'] != dict2['unit']:
            match = False
        elif dict1['polarized'] != dict2['polarized']:
            match = False
        elif dict1.get('coord') != dict2.get('coord'):
            match = False
        elif (dict1.get('metric_params') or {}) != (dict2.get('metric_params') or {}):
            match = False
        elif dict1.get('frame') != dict2.get('frame'):
            match = False
    else:
        print("file type: "+dict_type+" not supported. Returning false.")
        match = False

    return match

COMBINE_METHODS = ('statistical', 'time')
ADD_SPECTRA_METHODS = COMBINE_METHODS

# the header fields two spectra or two images must share to be combined
_COMBINE_FIELDS = {
    'spec': ('nx', 'nmu', 'nphi', 'nintens', 'units', 'polarized', 'yerror',
             'coord', 'metric_params', 'frame'),
    'image': ('ninc', 'nen', 'nx', 'ny', 'nintens', 'unit', 'polarized',
              'coord', 'metric_params', 'frame'),
}
_COMBINE_FACES = {
    'spec': ('xfaces', 'mufaces', 'phifaces'),
    'image': ('ifaces', 'efaces', 'xfaces', 'yfaces'),
}


def _combine_binned(bin1, bin2, method, kind, has_errors):
    """
    The combination shared by add_spectra and add_images; kind is 'spec' or 'image'.
    """

    if not bin1:
        return bin2
    if not bin2:
        return bin1

    name = 'add_spectra' if kind == 'spec' else 'add_images'
    if method not in COMBINE_METHODS:
        raise ValueError(f"[{name}]: method {method!r} is not one of {COMBINE_METHODS}")
    if not header_match(bin1, bin2, kind):
        # say which fields, so a wrong file in a glob can be picked out
        differ = [key for key in _COMBINE_FIELDS[kind]
                  if (bin1.get(key) or None) != (bin2.get(key) or None)]
        raise RuntimeError(f'[{name}]: headers do not match in '
                           + ', '.join(f"{k} ({bin1.get(k)} vs {bin2.get(k)})"
                                       for k in differ))
    # header_match compares the bin counts; the bins have to be the same bins as well
    for key in _COMBINE_FACES[kind]:
        if not np.allclose(bin1[key], bin2[key], rtol=1.0e-12, atol=0.0):
            raise RuntimeError(f'[{name}]: {key} do not match')

    # the arrays are replaced, never written into, so a shallow copy leaves the inputs
    # untouched
    out = bin1.copy()

    if method == 'statistical':
        # a file carries dt to eight decimals, so a spectrum read back is compared to
        # one made in memory at that precision, not to the bit
        if not math.isclose(bin1['dt'], bin2['dt'], rel_tol=1.0e-7):
            raise RuntimeError(f'[{name}]: statistical addition requested but the '
                               f"inputs have different integration times, "
                               f"{bin1['dt']:.8e} and {bin2['dt']:.8e}")
        out['intensity'] = bin1['intensity'] + bin2['intensity']
        if has_errors:
            out['errors'] = np.sqrt(bin1['errors']**2 + bin2['errors']**2)
        if 'family_sums' in bin1 and 'family_sums' in bin2:
            out['family_sums'] = _merge_family_sums(bin1['family_sums'],
                                                    bin2['family_sums'])
    else:
        # outputs are independent, so their family errors are finalized first and then
        # combine like any others
        bin1 = finalize_family_errors(dict(bin1))
        bin2 = finalize_family_errors(dict(bin2))
        out = bin1.copy()
        total_dt = bin1['dt'] + bin2['dt']
        w1 = bin1['dt']/total_dt
        w2 = bin2['dt']/total_dt
        out['intensity'] = w1*bin1['intensity'] + w2*bin2['intensity']
        if has_errors:
            out['errors'] = np.sqrt((w1*bin1['errors'])**2
                                    + (w2*bin2['errors'])**2)
        out['dt'] = total_dt

    # each list's ntot is the photons run on its rank, so the total is the sum whether
    # the parts are ranks of one interval or whole intervals
    out['ntot'] = bin1['ntot'] + bin2['ntot']

    return out


def add_spectra(spec1, spec2, method='statistical'):
    """
    Combine two spectra into one, returned as a new dictionary.

    'statistical' sums them, for spectra that each hold a share of the photons of one
    interval, as the lists of the ranks of one output do; their dt must agree.  'time'
    averages them weighted by integration time, for spectra of independent intervals
    or runs that are each already normalized, as the outputs of a run with nout > 1
    are.  Errors combine in quadrature either way, and ntot, the photons run, is
    summed.  An empty dict stands for "nothing yet" and returns the other spectrum, so
    a loop over files can start from {}.
    """

    # 'yerror' is the string the file header carries, "true" or "false", so it has to be
    # compared, not tested: "false" is a non-empty string and used to take this branch,
    # and spectra made without errors could not be combined at all.
    has_errors = bool(spec1) and spec1['yerror'] == 'true'
    return _combine_binned(spec1, spec2, method, 'spec', has_errors)

# Retrun x for desired units
def convert_xaxis(newunit, spectrum):
    """
    Convert the x-axis from one unit type to another type
    """
    
    h = 6.62607015e-27
    everg = 1.6021772e-12
    c = 2.99792e10

    baseunit = spectrum['units']
    xfaces = spectrum['xfaces']
    if baseunit == 'kev':
        nu = xfaces*1000.*everg/h
    elif baseunit == 'ev':
        nu = xfaces*everg/h
    elif baseunit == 'nu':
        nu = xfaces
    elif baseunit == 'lambda':
        # wavelength in Angstrom, so 1e8 per cm; this and the three conversions below
        # used to divide by 1e8 instead, putting nu off by 1e16
        nu = c*1.e8/xfaces
    if newunit == 'kev':
        spectrum['xfaces'] = nu*h/(everg*1000.)
    elif newunit == 'ev':
        spectrum['xfaces'] = nu*h/everg
    elif newunit == 'nu':
        spectrum['xfaces'] = nu
    elif newunit == 'lambda':
        spectrum['xfaces'] = c/nu*1.e8
    spectrum['units'] = newunit

# Return nu
def get_frequency(xunit, xfaces):
    """
    Return frequency given current xunit
    """
    
    h = 6.62607015e-27
    everg = 1.6021772e-12
    c = 2.99792e10

    if xunit == 'kev':
        nu = 0.5*(xfaces[1:]+xfaces[:-1])*1000.*everg/h
    elif xunit == 'ev':
        nu = 0.5*(xfaces[1:]+xfaces[:-1])*everg/h
    elif xunit == 'nu':
        nu = 0.5*(xfaces[1:]+xfaces[:-1])
    elif xunit == 'lambda':
        nu = 0.5*(1./xfaces[:-1]+1./xfaces[1:])*c*1.e8
    return nu


def get_luminosity(spec):
    """
    Computes the integrated luminosity corresponding to a spectrum
    """

    nx = spec['nx']
    nmu = spec['nmu']
    nphi = spec['nphi']
    mumid = 0.5*(spec['mufaces'][1:]+spec['mufaces'][:-1])

    # Compute frequency width and mean energy (in ergs) of bins
    h = 6.62607015e-27
    everg = 1.6021772e-12
    c = 2.99792e10
    xaxis = spec['units']
    xfaces = spec['xfaces']
    if xaxis == 'kev':
        dnu = (xfaces[1:]-xfaces[:-1])*1000.*everg/h
        #emid = 0.5*(xfaces[1:]+xfaces[:-1])*1000.*everg
    elif xaxis == 'ev':
        dnu = (xfaces[1:]-xfaces[:-1])*everg/h
        #emid = 0.5*(xfaces[1:]+xfaces[:-1])*everg
    elif xaxis == 'nu':
        dnu = xfaces[1:]-xfaces[:-1]
        #emid = 0.5*(xfaces[1:]+xfaces[:-1])*h
    elif xaxis == 'lambda':
        # abs: a spectrum converted to a wavelength axis has faces running downward
        dnu = np.abs((1./xfaces[:-1]-1./xfaces[1:])*c*1.e8)
        #emid = 0.5*(1./xfaces[:-1]-1./xfaces[1:])*c*h/1.e8

    # compute sum over frequency and solid angle
    lumin = (np.sum(dnu * mumid[:, np.newaxis] * spec['intensity'][0]) *
             2. * np.pi / nmu / nphi)

    return lumin

def family_key(user, tcol, ecol):
    """
    One integer per photon identifying its birth photon, from the emission temperature
    and emitted energy user columns the mc_readhdf* generators write.  The copies a
    weight window makes share both, so they share the key; two independent photons
    share it only if both values agree to a part in a million.
    """
    return (np.round(np.log(np.maximum(user[:, ecol], 1e-300))*1e6).astype(np.int64)
            * 1000003
            + np.round(np.log(np.maximum(user[:, tcol], 1.))*1e6).astype(np.int64))


def finalize_family_errors(spectrum):
    """
    Replace the errors of a spectrum made with family keys by the family-level ones:
    the weight of each family in each bin is summed, and the error is the root of the
    sum of the squares of those totals.  Copies of one birth photon are correlated, so
    the photon-level sum of squares overstates the statistics of a weight-window run;
    families are independent, so this is the estimator the photon-level one is for
    analog samples.  The raw sums are dropped; a spectrum without them is returned as
    it is, so the call is safe after any combination.
    """
    fam = spectrum.pop('family_sums', None)
    scale = spectrum.pop('error_scale', None)
    if fam is None or spectrum.get('yerror') != 'true':
        return spectrum
    keys, sums = fam
    nbins = spectrum['errors'].size
    bins = (keys % nbins).astype(np.int64)
    sq = np.bincount(bins, weights=sums**2, minlength=nbins)
    spectrum['errors'] = np.sqrt(sq.reshape(spectrum['errors'].shape)) * scale
    return spectrum


def _merge_family_sums(fam1, fam2):
    # each input is a few runs sorted by key (one per Stokes plane from make_spectrum,
    # one after a merge), so a stable sort merges them in linear time, where np.unique
    # would sort the concatenation from scratch
    keys = np.concatenate([fam1[0], fam2[0]])
    sums = np.concatenate([fam1[1], fam2[1]])
    if keys.size == 0:      # every photon of both was screened out
        return keys, sums
    order = np.argsort(keys, kind='stable')
    keys, sums = keys[order], sums[order]
    first = np.flatnonzero(np.concatenate(([True], keys[1:] != keys[:-1])))
    return keys[first], np.add.reduceat(sums, first)


def make_spectrum(phots,nx,xmin,xmax,xaxis='kev',logx=True,nmu=1,mumin=0,mumax=1.,
                  nphi=1,phimin=0,phimax=2.*np.pi,yerror=True,mask=None,
                  xfunc=None,anglebin='cartesian',family=None,**kwargs):
    """
    Makes spectrum (dict) from photon object.  With family, an integer key per photon
    (see family_key), the spectrum also carries the per-family, per-bin weight sums
    that finalize_family_errors turns into family-level errors once every chunk and
    rank of the output has been added in.
    """

    # Store spectrum as a dictionary
    spectrum = {}

    # Store integration time
    spectrum['dt'] = phots.dt

    # Store total number of photons for refernce
    spectrum['ntot'] = phots.ntot

    # Set x binning variable
    h = 6.62607015e-27
    everg = 1.6021772e-12
    c = 2.99792e10
    preset = False
    # 'units' is the canonical key: it is what the file header calls this field and what
    # every consumer reads.  make_spectrum used to set 'xaxis' instead, so a freshly made
    # spectrum could not be passed to convert_xaxis, get_luminosity or the plotting
    # scripts, while one read back from file could not be written out again.
    spectrum['units'] = xaxis
    # Carry the provenance across from the photon list, so a derived spectrum still says
    # which metric produced it and header_match can refuse to combine incompatible ones.
    spectrum['coord'] = getattr(phots, 'coord', None)
    spectrum['metric_params'] = getattr(phots, 'metric_params', None) or {}
    spectrum['frame'] = getattr(phots, 'frame', None)
    if xaxis == 'kev':
        xphots = phots.energy/everg/1000.
        preset = True
    elif xaxis == 'ev':
        xphots = phots.energy/everg
        preset = True
    elif xaxis == 'nu':
        xphots = phots.energy/h
        preset = True
    elif xaxis == 'lambda':
        xphots = c*h/phots.energy*1.e8   # Angstrom
        preset = True
    if not preset:
        if xfunc is None:
            raise RuntimeError('Unrecognized xunit and no xfunc provided')
        xphots = xfunc(phots.energy,True,**kwargs)

    # Create bins
    xfaces = build_bins(xmin,xmax,nx,logx)
    spectrum['nx'] = nx
    spectrum['xfaces'] = xfaces

    # Get x bins
    xbins = get_bins(xphots,xfaces,nx,log=logx)

    # Get angle bins
    spectrum['nmu'] = nmu
    mufaces = build_bins(mumin,mumax,nmu,False)
    spectrum['mufaces'] = mufaces

    spectrum['nphi'] = nphi
    phifaces = build_bins(phimin,phimax,nphi,False)
    spectrum['phifaces'] = phifaces

    if anglebin == 'cartesian':
        mubins, phibins = get_angle_bins_cartesian(phots,nmu,mufaces,nphi,phifaces)
    elif anglebin == 'spherical':
        mubins, phibins = get_angle_bins_spherical(phots,nmu,mufaces)
    elif anglebin == 'hybrid':
        mubins, phibins = get_angle_bins_hybrid(phots,nmu,mufaces,nphi,phifaces)
    else:
        print("Error: anglebin == "+anglebin+". Must be cartesian, spherical, or hybrid.")

    # Create intensity grid and loop over photons to add contribution
    spectrum['polarized'] = parse_polarization(phots.polarized)
    npol = num_stokes_stored(spectrum['polarized'])
    nintens = 1 + npol
    spectrum['nintens'] = nintens
    #count = np.zeros((nphi,nmu,nx))
    intensity = np.zeros((nintens,nphi,nmu,nx))
    if yerror:
        errors = np.zeros((nintens,nphi,nmu,nx))

    # Apply the user define mask
    if mask is not None:
        xbins[mask] = -1

    # determine all valid indices
    valid_phots = (xbins >= 0) & (mubins >= 0) & (phibins >= 0)

    # determine corresponding bin values
    valid_phi = phibins[valid_phots]
    valid_mu = mubins[valid_phots]
    valid_x = xbins[valid_phots]

    valid_weights = phots.weight[valid_phots] * phots.energy[valid_phots]

    # Accumulate each plane with a flat bincount; the Stokes planes follow the intensity
    # in the order Q, U, V
    bins = (valid_phi, valid_mu, valid_x)
    shape = (nphi, nmu, nx)
    stokes = [phots.q, phots.u] if npol >= 2 else []
    if npol == 3:
        stokes.append(phots.v)
    plane_weights = [valid_weights]
    plane_weights += [valid_weights * spol[valid_phots] for spol in stokes]
    for m, w in enumerate(plane_weights):
        intensity[m] = sum_into_bins(shape, bins, w)
        if yerror:
            errors[m] = sum_into_bins(shape, bins, w**2)
    if family is not None and yerror:
        # one composite key per (family, plane, bin); the bin index is recovered from it
        # by finalize_family_errors, so the family key is multiplied by the bin count
        nbins = nintens * nphi * nmu * nx
        fkey = family[valid_phots].astype(np.int64)
        if fkey.size and np.abs(fkey).max() > np.iinfo(np.int64).max // (2*nbins):
            raise OverflowError('family keys too large for the number of spectrum bins')
        flat = (valid_phi * nmu + valid_mu) * nx + valid_x
        keys = []
        sums = []
        for m, w in enumerate(plane_weights):
            composite = fkey * nbins + (m * nphi * nmu * nx + flat)
            uniq, inv = np.unique(composite, return_inverse=True)
            keys.append(uniq)
            sums.append(np.bincount(inv, weights=w))
        spectrum['family_sums'] = (np.concatenate(keys), np.concatenate(sums))

    # Compute frequency width and mean energy (in erg) of bins
    h = 6.62607015e-27
    everg = 1.6021772e-12
    c = 2.99792e10
    if xaxis == 'kev':
        dnu = (xfaces[1:]-xfaces[:-1])*1000.*everg/h
        #emid = 0.5*(xfaces[1:]+xfaces[:-1])*1000.*everg
    elif xaxis == 'ev':
        dnu = (xfaces[1:]-xfaces[:-1])*everg/h
        #emid = 0.5*(xfaces[1:]+xfaces[:-1])*everg
    elif xaxis == 'nu':
        dnu = xfaces[1:]-xfaces[:-1]
        #emid = 0.5*(xfaces[1:]+xfaces[:-1])*h
    elif xaxis == 'lambda':
        dnu = (1./xfaces[:-1]-1./xfaces[1:])*c*1.e8
        #emid = 0.5*(1./xfaces[:-1]-1./xfaces[1:])*c*h/1.e8
    if not preset:
        efaces = xfunc(xfaces,False,**kwargs)
        #emid = 0.5*(efaces[1:]+efaces[:-1])
        dnu  = (efaces[1:]-efaces[:-1])/h

    # Normalize intensities
    mumid = 0.5*(mufaces[1:]+mufaces[:-1])


    # Compute factors with proper broadcasting
    # mumid: (nmu,) -> (1, 1, nmu, 1)
    # dnu: (nx,) -> (1, 1, 1, nx)
    const_factor = nphi * nmu / (2. * np.pi * phots.dt)
    fac = const_factor / (mumid[:, np.newaxis] * dnu[np.newaxis, :])  # Shape: (nmu, nx)

    # Rescale intensity and errors
    intensity *= fac

    if yerror:
        errors *= fac**2
        if family is not None:
            spectrum['error_scale'] = np.broadcast_to(fac, intensity.shape).copy()

    spectrum['intensity'] = intensity

    # Finish computing errors on intensities
    if yerror:
        """
        # Compute masks once
        inds_gt1 = count > 1.
        inds_le1 = count <= 1.
        inds_gt1 = inds_gt1[np.newaxis, :]
        inds_le1 = inds_le1[np.newaxis, :]
        count_bcast = count[np.newaxis, :]

        # Vectorized operation
        errors[inds_gt1] = np.sqrt(errors[inds_gt1] -
                                   intensity[inds_gt1]**2 / count_bcast[inds_gt1])
        errors[inds_le1] = 0.
        """
  
        spectrum['errors'] = np.sqrt(errors)

    if yerror:
        spectrum['yerror'] = "true"
    else:
        spectrum['yerror'] = "false"

    return spectrum
