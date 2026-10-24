"""
Photon lists: the geometry tags, the Photons chunk and the list readers and writers.
"""

# standard python modules
import time
import math

import numpy as np

__all__ = [
    'coord_properties',
    'POLARIZATION_MODES',
    'parse_polarization',
    'is_polarized',
    'tracks_circular',
    'num_stokes_stored',
    'Photons',
    'ks_cartesian_metric',
    'resolve_metric_params',
    'read_list',
    'read_list_generator',
    'write_list',
    'print_list',
    'get_luminosity_list',
]

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

    @classmethod
    def from_chunk(cls, header, chunk):
        """
        The photons of one chunk yielded by read_list_generator, given the header it
        yielded first; what every reader of a list does with each chunk.
        """
        return cls(dict(header, list=chunk['chunk'], length=chunk['length']))

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
