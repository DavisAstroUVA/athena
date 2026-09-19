"""
Support for manipulating and plotting Monte Carlo outputs
"""

# standard python modules
from multiprocessing.resource_sharer import stop
import struct
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from io import BytesIO

# Canonical geometry tags written by MonteCarlo::SetGeometryTag (montecarlo.cpp).
#
# Every consumer of a photon list needs three facts that the Athena++ COORDINATE_SYSTEM
# name does not carry: whether the run was relativistic, what the spatial basis is, and
# how the wavevector is stored.  The legacy pushers keep a unit orthonormal three-vector
# alongside the energy; GeneralPusher keeps genuine contravariant components, so in
# spherical coordinates k^theta and k^phi need factors of r and r sin(theta) before they
# can be treated as orthonormal.
#
#   relativistic : energy column holds the conserved -k_t rather than ep
#   geometry     : spatial basis, 'cartesian' | 'spherical' | 'cylindrical'
#   kvec         : 'unit' (orthonormal three-vector) | 'coord' (contravariant components)
#   metric       : 'minkowski' | 'kerr_schild' | 'boyer_lindquist'
#   layout       : 'v2' for tags below; 'v1' for pre-tag files, whose energy column held
#                  k^t in GR runs and whose spherical entries are ambiguous about kvec
_COORD_TAGS = {
    'cartesian':      dict(relativistic=False, geometry='cartesian',   kvec='unit',
                           metric='minkowski',       layout='v2'),
    'cartesian_gp':   dict(relativistic=False, geometry='cartesian',   kvec='coord',
                           metric='minkowski',       layout='v2'),
    'spherical_polar':dict(relativistic=False, geometry='spherical',   kvec='unit',
                           metric='minkowski',       layout='v2'),
    'spherical_gp':   dict(relativistic=False, geometry='spherical',   kvec='coord',
                           metric='minkowski',       layout='v2'),
    'cylindrical_gp': dict(relativistic=False, geometry='cylindrical', kvec='coord',
                           metric='minkowski',       layout='v2'),
    'minkowski_cart': dict(relativistic=True,  geometry='cartesian',   kvec='coord',
                           metric='minkowski',       layout='v2'),
    'ks_spherical':   dict(relativistic=True,  geometry='spherical',   kvec='coord',
                           metric='kerr_schild',     layout='v2'),
    'ks_cartesian':   dict(relativistic=True,  geometry='cartesian',   kvec='coord',
                           metric='kerr_schild',     layout='v2'),
    'bl_spherical':   dict(relativistic=True,  geometry='spherical',   kvec='coord',
                           metric='boyer_lindquist', layout='v2'),
    'snake_cart':     dict(relativistic=True,  geometry='cartesian',   kvec='coord',
                           metric='snake',           layout='v2'),
}

# Tags written before SetGeometryTag existed.  These were just COORDINATE_SYSTEM, so the
# pusher is unknowable: 'cartesian' and 'spherical_polar' may hold either convention.  We
# assume the legacy pusher, which is what those runs almost always used, and the GR ones
# carried k^t in the energy column rather than -k_t.
_COORD_TAGS_LEGACY = {
    'cartesian':       dict(relativistic=False, geometry='cartesian', kvec='unit',
                            metric='minkowski',   layout='v1'),
    'spherical_polar': dict(relativistic=False, geometry='spherical', kvec='unit',
                            metric='minkowski',   layout='v1'),
    'cylindrical':     dict(relativistic=False, geometry='cylindrical', kvec='coord',
                            metric='minkowski',   layout='v1'),
    'minkowski':       dict(relativistic=True,  geometry='cartesian', kvec='coord',
                            metric='minkowski',   layout='v1'),
    'kerr-schild':     dict(relativistic=True,  geometry='spherical', kvec='coord',
                            metric='kerr_schild', layout='v1'),
    'gr_user':         dict(relativistic=True,  geometry='cartesian', kvec='coord',
                            metric='kerr_schild', layout='v1'),
}


def coord_properties(coord):
    """
    Return how to interpret a photon list, given its coord header tag.

    This is the single place that decides relativistic vs non-relativistic, the spatial
    basis, and the wavevector convention.  Callers should branch on the returned fields
    rather than comparing the coord string directly.
    """
    if coord in _COORD_TAGS:
        return dict(_COORD_TAGS[coord])
    if coord in _COORD_TAGS_LEGACY:
        return dict(_COORD_TAGS_LEGACY[coord])
    raise ValueError(f"Unrecognized coord tag '{coord}' in photon list header")


# Polarization modes, mirroring MCPolarization in src/monte_carlo/polarization.hpp.
# 'none' stores no Stokes parameters, 'linear' stores Q and U, 'circular' adds V.
POLARIZATION_MODES = ('none', 'linear', 'circular')

# What the two file formats used to write before the modes existed.  Lists carried an
# integer, spectra and images the strings true/false.  Both meant "linear" when set.
_POLARIZATION_LEGACY = {'0': 'none', '1': 'linear', '2': 'circular',
                        'false': 'none', 'true': 'linear',
                        'False': 'none', 'True': 'linear'}


def parse_polarization(value):
    """
    Normalize a polarized header field to one of POLARIZATION_MODES.

    Accepts the current mode names and the legacy spellings both writers used.
    """
    if isinstance(value, bool):
        return 'linear' if value else 'none'
    if isinstance(value, (int, np.integer)):
        value = str(int(value))
    tag = str(value).strip()
    if tag in POLARIZATION_MODES:
        return tag
    if tag in _POLARIZATION_LEGACY:
        return _POLARIZATION_LEGACY[tag]
    raise ValueError(f"Unrecognized polarized tag '{value}' in file header")


def is_polarized(mode):
    """True when any Stokes parameters are stored"""
    return parse_polarization(mode) != 'none'


def tracks_circular(mode):
    """True when Stokes V is stored"""
    return parse_polarization(mode) == 'circular'


def num_stokes_stored(mode):
    """Number of Stokes planes stored alongside the intensity: 0, 2 (Q,U) or 3 (Q,U,V)"""
    mode = parse_polarization(mode)
    if mode == 'circular':
        return 3
    if mode == 'linear':
        return 2
    return 0


#SWD: Maybe photons be rewritten simply as dictionary
#SWD: Add error control
class Photons:
    """
    Class for storing photon list data
    """
    #Initialization from dictionary
    def __init__(self, phlist):
        self.dt = phlist['dt']
        self.polarized = parse_polarization(phlist['polarized'])
        self.npars = 10 + num_stokes_stored(self.polarized)
        self.ntot = phlist['ntot']
        self.coord = phlist['coord']
        # Free parameters of the metric. Empty for a flat metric
        self.metric_params = phlist.get('metric_params', {})
        # The frame the wavevector and the Stokes parameters are both measured in.
        self.frame = phlist.get('frame')
        # The spatial basis of k1,k2,k3: 'cartesian' when the writer has already rotated
        # the local legs onto the global ones, None for older files that carry the local
        # orthonormal legs and need the rotation applied here.
        self.basis = phlist.get('basis')
        # How to interpret this list: relativistic?, spatial basis, wavevector convention
        self.props = coord_properties(self.coord)
        self.relativistic = self.props['relativistic']
        ncol = phlist['npars']
        self.nphot = phlist['length']
        if ncol < self.npars:
            raise ValueError(f"Error creating photon: ncol {ncol} < npars {self.npars}")

        if ncol > self.npars:
            self.nuser = ncol - self.npars
            self.user = np.zeros((self.nphot, self.nuser))
        else:
            self.nuser = 0

        # allocate arrays for each variable
        self.weight = phlist['list'][:,0]
        self.energy = phlist['list'][:,1]
        self.x0 = phlist['list'][:,5]
        self.x1 = phlist['list'][:,2]
        self.x2 = phlist['list'][:,3]
        self.x3 = phlist['list'][:,4]
        self.k0 = phlist['list'][:,9].copy()
        self.k1 = phlist['list'][:,6].copy()
        self.k2 = phlist['list'][:,7].copy()
        self.k3 = phlist['list'][:,8].copy()
        if is_polarized(self.polarized):
            self.q = phlist['list'][:,10]
            self.u = phlist['list'][:,11]
        if tracks_circular(self.polarized):
            self.v = phlist['list'][:,12]
        if self.nuser > 0:
            for i in range(self.nuser):
                self.user[:,i] = phlist['list'][:,i+self.npars]

def ks_cartesian_metric(pos, M, a):
    """Covariant Kerr-Schild Cartesian metric at pos = [t, x, y, z]."""
    a2 = a**2
    x1, x2, x3 = pos[1], pos[2], pos[3]
    rr2 = x1**2 + x2**2 + x3**2
    r2 = 0.5 * (rr2 - a2 + np.hypot(rr2 - a2, 2.0 * a * x3))
    r = np.sqrt(r2)
    f = 2.0 * M * r2 * r / (r2**2 + a2 * x3**2)

    l = np.array([1.0,
                  (r * x1 + a * x2) / (r2 + a2),
                  (r * x2 - a * x1) / (r2 + a2),
                  x3 / r])

    return np.diag([-1., 1., 1., 1.]) + f * np.outer(l, l)


def resolve_metric_params(phots, **kwargs):
    """
    Mass and spin of a Kerr metric, taken from the photon list unless the caller overrides.

    The list header records the metric's free parameters, so these no longer have to be
    supplied by hand and cannot silently disagree with the run that produced the file.  An
    explicit keyword still wins, because lists written before the header carried the field
    have no parameters at all, but a value that contradicts the file is reported rather
    than quietly honoured.
    """
    params = getattr(phots, 'metric_params', None) or {}

    def disagrees(given, key):
        return key in params and not np.isclose(given, params[key], rtol=1e-12, atol=0.)

    # Mass: geometric units with M = 1 are the convention throughout and the previous code
    # hardcoded it, so fall back to that silently.
    mass = kwargs.get('mass')
    if mass is None:
        mass = params.get('m', 1.0)
    elif disagrees(mass, 'm'):
        print(f"Warning: mass={mass!r} was supplied but the file says {params['m']!r};"
              " using the supplied value.")

    # Spin: changes the answer materially and used to be a required keyword, so refuse to
    # guess when neither the file nor the caller supplies it rather than quietly assuming
    # Schwarzschild.
    spin = kwargs.get('spin')
    if spin is None:
        if 'a' not in params:
            raise ValueError(
                "spin is neither given nor recorded in the photon list header.  Lists"
                " written before the header carried metric_params must be passed"
                " spin=<a> explicitly.")
        spin = params['a']
    elif disagrees(spin, 'a'):
        print(f"Warning: spin={spin!r} was supplied but the file says {params['a']!r};"
              " using the supplied value.")

    return mass, spin


def read_list(filename, data=True, header=True):
    """
    Faster version of read_list. Read unformated list output and return list
    as a dictionary
    """

    mxh_ = 1000

    try:
        with open(filename, 'rb') as data_file:
            raw_data = data_file.read()
        raw_data_ascii = raw_data[0:mxh_].decode('ascii', 'replace')
    except:
        print(f"Could not open {filename} for reading.")
        return None

    phlist = {}
    current_index = 0

    def skip_string(expected_string):
        expected_string_len = len(expected_string)
        if raw_data_ascii[current_index:current_index+expected_string_len] != \
           expected_string:
            raise RuntimeError('File not formatted as expected')
        return current_index + expected_string_len

    def parse_line_value(prefix, value_type=float):
        nonlocal current_index
        current_index = skip_string(prefix)
        end_of_line_index = raw_data_ascii.find('\n', current_index)
        if end_of_line_index == -1:
            raise RuntimeError(f"Could not find end of line for {prefix}")

        value_str = raw_data_ascii[current_index:end_of_line_index].split(' ')[0]
        current_index = end_of_line_index + 1
        return value_type(value_str)

    if header:
        try:
            phlist['dt'] = parse_line_value("dt=", float)
        except:
            print("List file contains no dt entry. Setting to 1.")
            phlist['dt'] = 1.

        phlist['length'] = parse_line_value("length=", int)
        phlist['npars'] = parse_line_value("npars=", int)
        phlist['ntot'] = parse_line_value("ntot=", int)
        phlist['polarized'] = parse_polarization(parse_line_value("polarized=", str))

        # Handle coord separately as it's a string
        current_index = skip_string("coord=")
        end_of_line_index = raw_data_ascii.find('\n', current_index)
        phlist['coord'] = raw_data_ascii[current_index:end_of_line_index].split(' ')[0]
        current_index = end_of_line_index + 1

        # Free parameters of the metric, written since the spin and mass are needed to
        # lower an index or form -k_t.  Absent for metrics with no parameters
        phlist['metric_params'] = {}
        if raw_data_ascii.startswith("metric_params=", current_index):
            current_index += len("metric_params=")
            end_of_line_index = raw_data_ascii.find('\n', current_index)
            for item in raw_data_ascii[current_index:end_of_line_index].split(','):
                if '=' in item:
                    key, _, val = item.partition('=')
                    try:
                        phlist['metric_params'][key.strip()] = float(val)
                    except ValueError:
                        phlist['metric_params'][key.strip()] = val.strip()
            current_index = end_of_line_index + 1

        # One tag for every direction-like quantity in the file: the wavevector, the
        # angle bins and the plane the Stokes parameters are referenced to.  Optional,
        # as it postdates the format.
        phlist['frame'] = None
        if raw_data_ascii.startswith("frame=", current_index):
            current_index += len("frame=")
            end_of_line_index = raw_data_ascii.find('\n', current_index)
            phlist['frame'] = raw_data_ascii[current_index:end_of_line_index].split(' ')[0]
            current_index = end_of_line_index + 1

        # The spatial basis of the wavevector columns.  'cartesian' means the writer has
        # already rotated the local legs onto the global ones; absent means an older file
        # that still carries the local orthonormal legs, which the reader rotates itself.
        phlist['basis'] = None
        if raw_data_ascii.startswith("basis=", current_index):
            current_index += len("basis=")
            end_of_line_index = raw_data_ascii.find('\n', current_index)
            phlist['basis'] = raw_data_ascii[current_index:end_of_line_index].split(' ')[0]
            current_index = end_of_line_index + 1

    if data:
        npars = phlist['npars']
        length = phlist['length']

        # Calculate optimal chunk size based on available memory
        # Aim for ~100MB chunks
        target_chunk_bytes = 100 * 1024 * 1024  # 100MB
        samples_per_chunk = max(1, target_chunk_bytes // (8 * npars))
        nchunk = int(length/samples_per_chunk)
        if nchunk > 10:
            print(f"Reading {filename}. Estimated chunks: {nchunk:d}.")
        begin_index = current_index
        remaining_samples = length
        chunks = []

        ti = time.time()
        chunk_num = 0

        while remaining_samples > 0:
            chunk_size = min(samples_per_chunk, remaining_samples)
            bytes_to_read = chunk_size * npars * 8
            end_index = begin_index + bytes_to_read

            # Check bounds
            if end_index > len(raw_data):
                deficit_bytes = end_index - len(raw_data)
                deficit = math.ceil(deficit_bytes / (8 * npars))
                print(f"Warning: raw_data smaller than expected by {deficit} samples.")
                chunk_size -= deficit
                end_index = len(raw_data)
                remaining_samples = chunk_size  # This will be the last chunk
                phlist['length'] = length - deficit + chunk_size

            if chunk_size <= 0:
                break

            # Using frombuffer for faster reading
            # dtype='>f8' big-endian double precision
            chunk_data = np.frombuffer(raw_data[begin_index:end_index],
                                       dtype='>f8').reshape(chunk_size, npars)

            chunks.append(chunk_data)
            begin_index = end_index
            remaining_samples -= chunk_size
            chunk_num += 1

            # Report progress every 10 chunks
            if (nchunk > 10) and (chunk_num % 10 == 0):
                tf = time.time()
                print(f" Processing chunk {chunk_num}, time: {tf-ti:.2f}s")
                ti = tf

        # Concatenate all chunks at once
        if chunks:
            phlist['list'] = np.concatenate(chunks, axis=0)
        else:
            phlist['list'] = np.array([]).reshape(0, npars)

    return phlist

def read_list_generator(filename, chunk_size=None):
    """
    Generator that yields chunks of data from the list file.
    First yield returns the header dict, subsequent yields return data chunks.
    """
    mxh_ = 10000
    
    try:
        data_file = open(filename, 'rb')
    except:
        print(f"Could not open {filename} for reading.")
        return
    
    try:
        # Read only the header portion
        header_bytes = data_file.read(mxh_)
        raw_data_ascii = header_bytes.decode('ascii', 'replace')
    except:
        data_file.close()
        print(f"Could not read header from {filename}.")
        return
    
    phlist = {}
    current_index = 0
    
    def skip_string(expected_string):
        nonlocal current_index
        expected_string_len = len(expected_string)
        if raw_data_ascii[current_index:current_index+expected_string_len] != \
           expected_string:
            data_file.close()
            raise RuntimeError('File not formatted as expected')
        current_index += expected_string_len
    
    def parse_line_value(prefix, value_type=float):
        nonlocal current_index
        skip_string(prefix)
        end_of_line_index = raw_data_ascii.find('\n', current_index)
        if end_of_line_index == -1:
            data_file.close()
            raise RuntimeError(f"Could not find end of line for {prefix}")
        
        value_str = raw_data_ascii[current_index:end_of_line_index].split(' ')[0]
        current_index = end_of_line_index + 1
        return value_type(value_str)
    
    # Parse header
    try:
        phlist['dt'] = parse_line_value("dt=", float)
    except:
        print("List file contains no dt entry. Setting to 1.")
        phlist['dt'] = 1.
    
    phlist['length'] = parse_line_value("length=", int)
    phlist['npars'] = parse_line_value("npars=", int)
    phlist['ntot'] = parse_line_value("ntot=", int)
    phlist['polarized'] = parse_polarization(parse_line_value("polarized=", str))
    
    skip_string("coord=")
    end_of_line_index = raw_data_ascii.find('\n', current_index)
    phlist['coord'] = raw_data_ascii[current_index:end_of_line_index].split(' ')[0]
    current_index = end_of_line_index + 1

    # Free parameters of the metric, written since the spin and mass are needed to lower
    # an index or form -k_t.  Absent for metrics with no parameters
    phlist['metric_params'] = {}
    if raw_data_ascii.startswith("metric_params=", current_index):
        current_index += len("metric_params=")
        end_of_line_index = raw_data_ascii.find('\n', current_index)
        for item in raw_data_ascii[current_index:end_of_line_index].split(','):
            if '=' in item:
                key, _, val = item.partition('=')
                try:
                    phlist['metric_params'][key.strip()] = float(val)
                except ValueError:
                    phlist['metric_params'][key.strip()] = val.strip()
        current_index = end_of_line_index + 1

    # One tag for every direction-like quantity in the file: the wavevector, the
    # angle bins and the plane the Stokes parameters are referenced to.  Optional,
    # as it postdates the format.
    phlist['frame'] = None
    if raw_data_ascii.startswith("frame=", current_index):
        current_index += len("frame=")
        end_of_line_index = raw_data_ascii.find('\n', current_index)
        phlist['frame'] = raw_data_ascii[current_index:end_of_line_index].split(' ')[0]
        current_index = end_of_line_index + 1

    # see read_list: absent means the local orthonormal legs, 'cartesian' the global ones
    phlist['basis'] = None
    if raw_data_ascii.startswith("basis=", current_index):
        current_index += len("basis=")
        end_of_line_index = raw_data_ascii.find('\n', current_index)
        phlist['basis'] = raw_data_ascii[current_index:end_of_line_index].split(' ')[0]
        current_index = end_of_line_index + 1

    # Yield header first
    yield {'header': phlist, 'chunk': None, 'remaining': None, 'length': None, 'done': False}
    
    # Calculate chunk size
    npars = phlist['npars']
    length = phlist['length']
    
    if chunk_size is None:
        target_chunk_bytes = 100 * 1024 * 1024
        chunk_size = max(1, target_chunk_bytes // (8 * npars))
    
    # Move file pointer to start of binary data
    data_file.seek(current_index)
    
    remaining_samples = length
    
    # Yield data chunks
    try:
        while remaining_samples > 0:
            current_chunk_size = min(chunk_size, remaining_samples)
            bytes_to_read = current_chunk_size * npars * 8
            
            # Read only the bytes needed for this chunk
            chunk_bytes = data_file.read(bytes_to_read)
            
            # Check if we got all the data we expected
            if len(chunk_bytes) < bytes_to_read:
                deficit_bytes = bytes_to_read - len(chunk_bytes)
                deficit = math.ceil(deficit_bytes / (8 * npars))
                print(f"Warning: File smaller than expected by {deficit} samples.")
                current_chunk_size = len(chunk_bytes) // (8 * npars)
            
            if current_chunk_size <= 0:
                break
            
            chunk_data = np.frombuffer(chunk_bytes, dtype='>f8').reshape(current_chunk_size, npars)
            
            remaining_samples -= current_chunk_size
            
            yield {'header': phlist, 'chunk': chunk_data, 'remaining': remaining_samples,
                   'length': current_chunk_size, 'done': remaining_samples <= 0}
    
    finally:
        data_file.close()


def write_list(filename, phlist, header=True, length=None):
    """
    Faster version of write_list. Write photon list (dictionary) to file
    """

    if header:
        # Write header in text mode for readable ASCII
        with open(filename, 'w') as outfile:
            outfile.write(f"dt={phlist['dt']:.8e}\n")
            outfile.write(f"length={length if length is not None else phlist['length']:d}\n")
            outfile.write(f"npars={phlist['npars']:d}\n")
            outfile.write(f"ntot={phlist['ntot']:d}\n")
            outfile.write(f"polarized={parse_polarization(phlist['polarized'])}\n")
            outfile.write(f"coord={phlist['coord']}\n")
            # Carry the metric through a read-modify-write, so a filtered list is still
            # self-describing.  Written only when present, matching the C++ writer.
            mpars = phlist.get('metric_params') or {}
            if mpars:
                outfile.write("metric_params="
                              + ",".join(f"{k}={v!r}" for k, v in mpars.items()) + "\n")
            if phlist.get('frame') is not None:
                outfile.write("frame="+phlist['frame']+"\n")
            # Carried through unchanged: the columns are copied verbatim, so the basis
            # they were written in is the basis they are still in.
            if phlist.get('basis') is not None:
                outfile.write("basis="+phlist['basis']+"\n")

    # Append binary data using numpy's tobytes() - faster than struct.pack
    with open(filename, 'ab') as outfile:
        data_array = phlist['list'].astype('>f8')  # big-endian double precision
        outfile.write(data_array.tobytes())

def print_list(infile, start = 0, stop = None):
    """
    Print contents of list
    """

    phlist = read_list(infile, data=True)

    print(f"dt={phlist['dt']:.8e}\n")
    print(f"length={phlist['length']:d}\n")
    print(f"npars={phlist['npars']:d}\n")
    print(f"ntot={phlist['ntot']:d}\n")
    print(f"polarized={parse_polarization(phlist['polarized'])}\n")
    print(f"coord={phlist['coord']}\n")
    print(f"metric_params={phlist.get('metric_params') or {}}\n")
    print(f"frame={phlist.get('frame')}\n")

    if stop is None:
        stop = phlist['length']
    for i in range(start, stop):
        print(f"Photon {i}: {phlist['list'][i]}")

def get_luminosity_list(phlist):
    """
    Read in list file and compute luminoisty
    """
    phots = Photons(phlist)
    lumin = np.sum(phots.weight*phots.energy)/phots.dt

    return lumin

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
    else:
        print("file type: "+dict_type+" not supported. Returning false.")
        match = False

    return match

def add_spectra(spec1, spec2, method='statistical'):
    """
    Add two spectra to create single spectrum
    """

    if spec1 == {}:
        return spec2
    elif spec2 == {}:
        return spec1
    
    if not header_match(spec1, spec2, 'spec'):
        raise RuntimeError('[add_specta]: headers do not match')

    # copy spectra to new dictionaries to avoid modification of originals
    spec1c = spec1.copy()
    spec2c = spec2.copy()

    if method == 'statistical':
        if spec1c['dt'] != spec2c['dt']:
            raise RuntimeError('[add_specta]: statistical averaging requested' \
                               'but spectra have different integration times')
    
    # intialize spec_out as copy for simplicity
    spec_out = spec1c.copy()

    has_errors = spec_out['yerror'] == 'true'

    if method == 'statistical':
        # sum unormalized intensity and error arrays
        spec_out['intensity'] = spec1c['intensity'] + spec2c['intensity']
        if has_errors:
            spec_out['errors'] = np.sqrt((spec1c['errors'])**2 + (spec2c['errors'])**2)
    elif method == 'time':
        # weighted sum of intensities and errors
        total_dt = spec1c['dt'] + spec2c['dt']
        w1 = spec1c['dt']/total_dt
        w2 = spec2c['dt']/total_dt
        spec_out['intensity'] = w1*spec1c['intensity'] + w2*spec2c['intensity']
        if has_errors:
            spec_out['errors'] = np.sqrt((w1*spec1c['errors'])**2 + (w2*spec2c['errors'])**2)
        spec_out['dt'] = total_dt

    return spec_out

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
        nu = c/1.e8/xfaces
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
        nu = 0.5*(1./xfaces[:-1]+1./xfaces[1:])*c/1.e8
    return nu

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
        xlabel = r"$\lambda~{\rm (\AA)$"

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
        dnu = (1./xfaces[:-1]-1./xfaces[1:])*c/1.e8
        #emid = 0.5*(1./xfaces[:-1]-1./xfaces[1:])*c*h/1.e8

    # compute sum over frequency and solid angle
    lumin = (np.sum(dnu * mumid[:, np.newaxis] * spec['intensity'][0]) *
             2. * np.pi / nmu / nphi)

    return lumin

def build_bins(xmin, xmax, nx, logx):
    """
    Builds a x-axis grid for binning the photons
    """
    if logx:
        return np.logspace(np.log10(xmin),np.log10(xmax),nx+1)
    return np.linspace(xmin,xmax,nx+1)

def get_bins(xphots, xfaces, nx, uniform=True, log=True):
    """
    Returns the bin numbers corresponding photon samples
    """
    xbins = np.zeros(len(xphots), dtype=int)-1
    if uniform:
        if log:
            xlfaces = np.log10(xfaces)
            xwidth = xlfaces[nx] - xlfaces[0]

            # get integer bin number
            log_xphots = np.log10(xphots)
            xbins = np.floor((log_xphots - xlfaces[0]) / xwidth * nx).astype(int)

            # catch out-of-bounds values
            xbins = np.where((xbins < 0) | (xbins >= nx), -1, xbins)
        else:
            xwidth = xfaces[nx] - xfaces[0]

            # get integer bin number
            xbins = np.floor((xphots - xfaces[0]) / xwidth * nx).astype(int)

            # catch out-of-bounds values
            xbins = np.where((xbins < 0) | (xbins >= nx), -1, xbins)

        return xbins
    else:
        # use binary search for non uniform data (generally much slower)
        return get_bins_binary_search(xphots, xfaces, nx)

def get_bins_binary_search(xphots, xfaces, nx):
    """
    Returns x bin numbers corresponding to xphots for non-uniformly
    binned data via binary search.
    """
    # Exclude values outside of search range
    indsp = (xphots > xfaces[nx]).nonzero()
    indsm = (xphots < xfaces[0]).nonzero()
    xbins = np.searchsorted(xfaces,xphots)-1
    xbins[indsp] = -1
    xbins[indsm] = -1

    return xbins

def effective_kvec(photons):
    """
    Whether the stored wavevector is an orthonormal three-vector or contravariant
    coordinate components.

    _COORD_TAGS records what a coord tag meant when the tag was introduced.  A file that
    carries a frame field says it directly and wins: both 'normal' and 'lab' store
    components in an orthonormal frame, so the scale factors and metric projections the
    coordinate form needs must not be applied on top.
    """
    if getattr(photons, 'frame', None) in ('normal', 'lab'):
        return 'unit'
    return photons.props['kvec']


def unit_direction_cartesian(photons):
    """
    Cartesian components of the photon propagation direction, as a unit vector.

    Handles every wavevector convention the format has carried.  A file whose header
    declares basis=cartesian already holds the global cartesian components and needs
    only normalizing.  Otherwise the columns are on the local orthonormal legs: under the
    legacy pushers directly, under GeneralPusher as contravariant components whose
    spherical k^theta and k^phi carry factors of 1/r and 1/(r sin(theta)), and whose
    cylindrical k^phi carries 1/R, all removed before rotating.  The result is normalized
    either way: for a null vector the orthonormal spatial components have norm equal to
    the time component, not unity.
    """
    props = photons.props
    k1, k2, k3 = photons.k1, photons.k2, photons.k3

    if getattr(photons, 'basis', None) == 'cartesian':
        norm = np.sqrt(k1*k1 + k2*k2 + k3*k3)
        return k1/norm, k2/norm, k3/norm

    if props['geometry'] == 'spherical':
        cth = np.cos(photons.x2)
        sth = np.sin(photons.x2)
        cph = np.cos(photons.x3)
        sph = np.sin(photons.x3)
        if effective_kvec(photons) == 'coord':
            k2 = k2 * photons.x1
            k3 = k3 * photons.x1 * sth
        kx = k1*sth*cph + k2*cth*cph - k3*sph
        ky = k1*sth*sph + k2*cth*sph + k3*cph
        kz = k1*cth - k2*sth
    elif props['geometry'] == 'cylindrical':
        cph = np.cos(photons.x2)
        sph = np.sin(photons.x2)
        if effective_kvec(photons) == 'coord':
            k2 = k2 * photons.x1
        kx = k1*cph - k2*sph
        ky = k1*sph + k2*cph
        kz = k3
    else:
        kx, ky, kz = k1, k2, k3

    norm = np.sqrt(kx*kx + ky*ky + kz*kz)
    return kx/norm, ky/norm, kz/norm


def unit_direction_radial(photons):
    """
    Component of the photon propagation direction along the local radial unit vector.
    """
    props = photons.props

    # Go through the cartesian direction, which already resolves every basis the file
    # can carry, and dot it with the radial unit vector built from the position.  The
    # spherical shortcut of reading k1 directly is gone: it was only right while the
    # columns were on the local legs.
    kx, ky, kz = unit_direction_cartesian(photons)
    if props['geometry'] == 'spherical':
        sth = np.sin(photons.x2)
        x = sth*np.cos(photons.x3)
        y = sth*np.sin(photons.x3)
        z = np.cos(photons.x2)
        return kx*x + ky*y + kz*z
    if props['geometry'] == 'cylindrical':
        x = photons.x1*np.cos(photons.x2)
        y = photons.x1*np.sin(photons.x2)
        z = photons.x3
    else:
        x, y, z = photons.x1, photons.x2, photons.x3
    rnorm = np.sqrt(x*x + y*y + z*z)
    return (kx*x + ky*y + kz*z)/rnorm


def get_angle_bins_cartesian(photons,nmu, mufaces, nphi, phifaces):
    """
    Bin angles in theta, phi defined relative to x,y,z axes
    """

    if ((nmu == 1) and (mufaces[0] <= 0.) and (mufaces[1] >= 1.0)):
        skipmu = True
    else:
        skipmu = False
    if ((nphi == 1) and (phifaces[0] <= 0.) and (phifaces[1] >= 2.*np.pi)):
        skipphi = True
    else:
        skipphi = False

    if (skipmu and skipphi):
        return np.zeros(photons.nphot,dtype=int),np.zeros(photons.nphot,dtype=int)

    kx, ky, kz = unit_direction_cartesian(photons)

    if skipmu:
        # return 0
        mubins = np.zeros(photons.nphot,dtype=int)
    else:
        # Bin based on k . z
        mu = abs(kz)
        mubins = get_bins(mu,mufaces,nmu,log=False)
    if skipphi:
        # return 0
        phibins = np.zeros(photons.nphot,dtype=int)
    else:
        phi = np.arctan2(ky, kx)
        phi[(phi<0.).nonzero()] += 2.*np.pi
        phibins = get_bins(phi, phifaces, nphi, log=False)

    return mubins, phibins


def get_angle_bins_spherical(photons,nmu,mufaces):
    """
    Bin angles relative to local radial direction.  Here we only bin
    in polar angle.
    """
    if ((nmu == 1) and (mufaces[0] <= 0.) and (mufaces[1] >= 1.0)):
        skipmu = True
    else:
        skipmu = False

    if skipmu:
        return np.zeros(photons.nphot,dtype=int),np.zeros(photons.nphot,dtype=int)

    # Bin based on k_r
    mu = unit_direction_radial(photons)
    mubins = get_bins(mu, mufaces ,nmu, log=False)

    # return 0 for phi
    phibins = np.zeros(photons.nphot, dtype=int)

    return mubins, phibins

def get_angle_bins_hybrid(photons, nmu, mufaces, nphi, phifaces):
    """
    Bin angles in theta, phi defined relative to x,y,z axes, but with phi
    defined by local azimuthal angle
    """
    if nmu == 1:
        print("Error: this function requires nmu > 1")
        return  np.zeros(photons.nphot,dtype=int),np.zeros(photons.nphot,dtype=int)
    if nphi == 1:
        print("Error: this function requires nphi > 1")
        return np.zeros(photons.nphot,dtype=int),np.zeros(photons.nphot,dtype=int)

    if photons.props['geometry'] != 'spherical':
        print("Error: this function only works with spherical geometry")
        return np.zeros(photons.nphot,dtype=int),np.zeros(photons.nphot,dtype=int)

    # The z and phi-hat components of the direction, from the cartesian form so that the
    # basis the file was written in is resolved in one place.
    kx, ky, kz = unit_direction_cartesian(photons)
    cph = np.cos(photons.x3)
    sph = np.sin(photons.x3)
    kph = -kx*sph + ky*cph

    # Bin based on k . z
    mu = abs(kz)
    mubins = get_bins(mu,mufaces,nmu,log=False)

    phi = np.arcsin(kph)
    # inds should be empty for outgoing photons
    inds = (kr < 0.).nonzero()
    phi[inds] = np.pi - phi[inds]
    phi[(phi<0.).nonzero()] += 2.*np.pi
    phibins = get_bins(phi, phifaces, nphi, log=False)

    return mubins,phibins

def make_spectrum(phots,nx,xmin,xmax,xaxis='kev',logx=True,nmu=1,mumin=0,mumax=1.,
                  nphi=1,phimin=0,phimax=2.*np.pi,yerror=True,mask=None,
                  xfunc=None,anglebin='cartesian',**kwargs):
    """
    Makes spectrum (dict) from photon object
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
        xphots = c*h/(phots.energy*1.e8)
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

    # Use np.add.at for accumulation
    #np.add.at(count, (valid_phi, valid_mu, valid_x), 1.0)
    np.add.at(intensity, (0, valid_phi, valid_mu, valid_x), valid_weights)

    # Stokes planes follow the intensity in the order Q, U, V
    stokes = [phots.q, phots.u] if npol >= 2 else []
    if npol == 3:
        stokes.append(phots.v)
    for m, spol in enumerate(stokes):
        np.add.at(intensity, (m+1, valid_phi, valid_mu, valid_x),
                  valid_weights * spol[valid_phots])

    if yerror:
        np.add.at(errors, (0, valid_phi, valid_mu, valid_x), valid_weights**2)
        for m, spol in enumerate(stokes):
            np.add.at(errors, (m+1, valid_phi, valid_mu, valid_x),
                      (valid_weights * spol[valid_phots])**2)

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
        dnu = (1./xfaces[:-1]-1./xfaces[1:])*c/1.e8
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

def get_image_bins(phots, rcam, ifaces, xfaces, yfaces):
    """
    Bin photons in image plane coordinates
    """
 
    ninc = ifaces.size - 1
    nx  = xfaces.size - 1
    ny = yfaces.size - 1

    # Cartesian position and direction, whatever chart and wavevector basis the file
    # carries.  Previously keyed on the raw coord string, which knew only the two legacy
    # tags and read the spherical wavevector as if it were always on the local legs.
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
    rp = np.sqrt(xp**2 + yp**2 + zp**2)
    kx, ky, kz = unit_direction_cartesian(phots)
    kdx = xp*kx + yp*ky + zp*kz


    dl = np.sqrt(rcam**2-rp**2+kdx**2)-kdx
    xf = xp + dl * kx
    yf = yp + dl * ky
    zf = zp + dl * kz

    cthf = zf/rcam
    sthf = np.sqrt(1.-cthf**2)
    phf =  np.arctan2(yf,xf)
    cphf = np.cos(phf)
    sphf = np.sin(phf)
    np.set_printoptions(threshold=1000)
    print(len(np.where(kz > 0.95)[0]))
    print(len(np.where(kz < -0.95)[0]))
    ibins = get_bins(cthf, ifaces, ninc, log=False)

    kth = kx*cthf*cphf + ky*cthf*sphf - kz*sthf
    kph = -kx*sphf + ky*cphf
    norm = np.sqrt(1.+kth**2+kph**2)
    y = kth*rcam*norm
    x = kph*rcam*norm

    xbins = get_bins(x,xfaces,nx,log=False)
    ybins = get_bins(y,yfaces,ny,log=False)
    #for i,q in enumerate(cthf):
    #    print(q,ibins[i],x[i],y[i],xbins[i],ybins[i])
    return ibins, xbins, ybins

def make_image_mc(phots, rcam, ninc, imin, imax, nen, emin, emax,
                  nx, xmin, xmax, ny, ymin, ymax, unit, mask=None):
    """
    Create a binned image from photon list
    """

    # Store the image as a dictionary
    image = {}

    # Store integration time
    image['dt'] = phots.dt

    # Store total number of photons for refernce
    image['ntot'] = phots.ntot

    # Create bins for viewer inclination
    ifaces = build_bins(imin,imax,ninc,False)
    image['ninc'] = ninc
    image['ifaces'] = ifaces

    # Create bins for observed frequency/photon energy
    efaces = build_bins(emin,emax,nen,True)
    image['nen'] = nen
    image['efaces'] = efaces

    # Create bins for image plane, image will be uniform 2d array
    image['nx'] = nx
    image['ny'] = ny
    xfaces = build_bins(xmin,xmax,nx,False)
    yfaces = build_bins(ymin,ymax,ny,False)
    image['xfaces'] = xfaces
    image['yfaces'] = yfaces

    # set units
    image['unit'] = unit

    ibins, xbins, ybins = get_image_bins(phots, rcam, ifaces, xfaces, yfaces)
    #set ebins temporarily to 0
    everg = 1.6021772e-12
    xphots = phots.energy/everg/1000.
    ebins = get_bins(xphots, efaces, nen, True, True)
    #ebins = np.zeros(len(ibins),dtype=int)

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

    intensity = np.zeros((nintens,ninc,nen,ny,nx))

    # Create mask for valid indices
    valid_phots = (ibins >= 0) & (ebins >= 0) & (xbins >= 0) & (ybins >= 0)

    # Extract valid indices and weights
    valid_i = ibins[valid_phots]
    valid_e = ebins[valid_phots]
    valid_y = ybins[valid_phots]
    valid_x = xbins[valid_phots]
    valid_weights = phots.weight[valid_phots] * phots.energy[valid_phots]

    np.add.at(intensity, (0, valid_i, valid_e, valid_y, valid_x), valid_weights)

    # Stokes planes follow the intensity in the order Q, U, V
    stokes = [phots.q, phots.u] if npol >= 2 else []
    if npol == 3:
        stokes.append(phots.v)
    for m, spol in enumerate(stokes):
        np.add.at(intensity, (m+1, valid_i, valid_e, valid_y, valid_x),
                  valid_weights * spol[valid_phots])
    # Normalize intensities
    mumid = abs(0.5*(ifaces[1:]+ifaces[:-1]))
    dmu = ifaces[1:]-ifaces[:-1]

    if image['nen'] == 1:
        dnu = np.array([1.])
    else:
        h = 6.62607015e-27
        everg = 1.6021772e-12
        dnu = (efaces[1:]-efaces[:-1])*1000.*everg/h

    dx = xfaces[1:]-xfaces[:-1]
    dy = yfaces[1:]-yfaces[:-1]
    area = np.outer(dy,dx)
    for k in range(nintens):
        for j in range(ninc):
            for i in range(nen):
                # not divided by dnu for now
                fac = dnu[i]*dmu[j]*mumid[j]*2.*np.pi*phots.dt
                intensity[k,j,i,:,:] /= fac*area


    image['intensity'] = intensity

    return image

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
               ax=None, **kwargs):
    """
    Plot an image
    """
    if ax is None:
        # Create figure, axis and assume a single plot window
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

    vmin = kwargs['vmin']
    vmax = kwargs['vmax']
    cmap = plt.get_cmap(kwargs['colormap'])
    plt.figure()

    if not check_polarization(image, itype, kind='image'):
        raise RuntimeError("Polarization type requested ("+itype+
                           ") but image mode is '"
                           +parse_polarization(image['polarized'])+"'")
    if itype == 'intensity':
        vals = image['intensity'][0,iinc,ie,:,:]
        clabel=r"$I$"
        if vmin is None:
            vmin = 1.e-5*np.max(vals)
    elif itype == 'q':
        vals = image['intensity'][1,iinc,ie,:,:]
        clabel=r"$Q/I$"
    elif itype == 'u':
        vals = image['intensity'][2,iinc,ie,:,:]
        clabel=r"$U/I$"
    elif itype == 'v':
        vals = image['intensity'][3,iinc,ie,:,:]
        clabel=r"$V/I$"
    elif itype == 'polangle':
        q = image['intensity'][1,iinc,ie,:,:]
        u = image['intensity'][2,iinc,ie,:,:]
        vals = 90./np.pi*np.arctan2(u,q)
        vals[vals < 0.] += 360.
        if vmin is None:
            vmin = 0.
        clabel = r"$\rm Pol.\; Angle$"
    elif itype == 'polfrac':
        q = image['intensity'][1,iinc,ie,:,:]
        u = image['intensity'][2,iinc,ie,:,:]
        vals = np.sqrt(q**2+u**2)
        if vmin is None:
            vmin = 0.
        clabel = r"$\rm Pol.\; Fraction$"
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

    plt.xlim(image['xfaces'][0],image['xfaces'][-1])
    plt.ylim(image['yfaces'][0],image['yfaces'][-1])
    if image['unit'] == 'cm':
        plt.xlabel(r"$x \; (\rm cm)$")
        plt.ylabel(r"$y \; (\rm cm)$")
        if itype == 'intensity':
            clabel=r"$I \; (\rm erg/s/cm^2)$"
    else:
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")

    plt.colorbar(im,label=clabel)
    plt.gca().set_aspect('equal')
    if pvec:
        if is_polarized(image['polarized']):
            q = image['intensity'][1,iinc,ie,:,:]
            u = image['intensity'][2,iinc,ie,:,:]
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
    outfile.write("unit="+image['unit']+"\n")
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
