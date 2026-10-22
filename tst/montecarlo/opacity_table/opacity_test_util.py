"""
Shared pieces of the table-opacity tests: athinput writers for the two generators the
tests run (mc_readhdf on the small isothermal-atmosphere snapshot, mc_isoth_opac on its
own atmosphere), a sequential runner, and bitwise comparisons of photon lists and athdf
files.  Runs are sequential on purpose: two MPI-built binaries started at the same
moment fail in PMIx at startup.
"""
import filecmp
import glob
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))

SNAPSHOT_DIR = os.path.expanduser('~/mc_test/read_hdf5')
SNAPSHOT = 'input_mciso.athdf'
SNAPSHOT_GR = 'ks_torus.athdf'  # from make_ks_snapshot.py
TABLE = 'out_opacity_table_nfreq32.txt'


# the keys the code reads from <opacitytable>; every other key of extra goes to <montecarlo>
TABLE_KEYS = ('opacity_file', 'table_groups', 'table_above', 'table_emax', 'table_offgrid',
              'table_gray', 'table_fill_freefree', 'table_storage', 'nfree_file', 'table_ne',
              'two_band', 'two_band_w', 'two_band_floor', 'two_band_closure',
              'two_band_thick', 'two_band_decorrelate')


def mc_block(nphot, iseed, scattering, emission, absorption, extra):
    """the <montecarlo> block and, when extra holds table keys, the <opacitytable> block"""
    lines = ['<montecarlo>',
             'nphot      = {:d}'.format(nphot),
             'iseed      = {:d}'.format(iseed),
             'scattering = ' + scattering,
             'emission   = ' + emission,
             'absorption = ' + absorption,
             'polarized  = false',
             'abs_method = weight']
    table = []
    for k, v in extra.items():
        (table if k in TABLE_KEYS else lines).append('{} = {}'.format(k, v))
    if table:
        lines += ['', '<opacitytable>'] + table
    return '\n'.join(lines) + '\n'


def write_readhdf_athinput(path, snapshot, table, nphot, iseed, scattering='thomson',
                           emission='table', absorption='table', extra=None,
                           table_key='montecarlo'):
    """
    mc_readhdf on the snapshot: the small isothermal atmosphere (64x64x128 cells, eight
    blocks).  table_key = 'problem' writes the table under <problem>/opacity_filename, the
    key the pre-module generator read; 'montecarlo' writes <opacitytable>/opacity_file.
    """
    extra = dict(extra or {})
    if table_key == 'montecarlo':
        extra['opacity_file'] = table
    text = """<comment>
problem   = isothermal atmosphere from a snapshot, table opacities
configure = --prob=mc_readhdf -mc -mpi -hdf5

<job>
problem_id = readhdf

<output1>
file_type = phlist

<output2>
file_type = hdf5
variable  = mclab
dt = 1.e-11

<mesh>
nx1        = 64
x1min      = -5.0e10
x1max      = 5.0e10
ix1_mc_bc  = periodic
ox1_mc_bc  = periodic

nx2        = 64
x2min      = -5.0e10
x2max      = 5.0e10
ix2_mc_bc  = periodic
ox2_mc_bc  = periodic

nx3        = 128
x3min      = 0.0
x3max      = 1.0e11
ix3_mc_bc  = absorb
ox3_mc_bc  = escape

<meshblock>
nx1 = 32
nx2 = 32
nx3 = 64

<hydro>
gamma = 1.666666666666667

""" + mc_block(nphot, iseed, scattering, emission, absorption, extra) + """
<problem>
input_filename = {snapshot}
{table_line}rho_cgs = 1.
vel_cgs = 1.
l_cgs = 1.
tnorm    = true
emin     = 1.0e-02
emax     = 3.0e+01
""".format(snapshot=snapshot,
           table_line=('opacity_filename = ' + table + '\n') if table_key == 'problem' else '')
    with open(path, 'w') as f:
        f.write(text)


def write_readhdf_gr_athinput(path, snapshot, table, nphot, iseed, scattering='thomson',
                              emission='table', absorption='table', extra=None,
                              table_key='montecarlo', nx=16, nb=2, half=24.):
    """
    mc_readhdf_gr on the synthetic Kerr-Schild torus of make_ks_snapshot.py (its mesh
    and unit conversions repeated here), general pusher, spin 0.9.  tcut = 5e6 K masks
    the hottest cells of the torus (its peak is 1e7 K) so that the mask path is exercised.
    """
    extra = dict(extra or {})
    if table_key == 'montecarlo':
        extra['opacity_file'] = table
    text = """<comment>
problem   = Kerr-Schild torus from a synthetic snapshot, table opacities
configure = --prob=mc_readhdf_gr --coord=gr_user -g -mc -mpi -hdf5

<job>
problem_id = readhdf_gr

<time>
cfl_number = 0.3
nlim       = 1
tlim       = 1.0

<coord>
m = 1.0
a = 0.9

<output1>
file_type = phlist

<output2>
file_type = hdf5
variable  = mclab
dt = 1.e-11

<mesh>
nx1        = {ntot}
x1min      = {lo}
x1max      = {hi}
ix1_bc     = outflow
ox1_bc     = outflow
ix1_mc_bc  = escape
ox1_mc_bc  = escape

nx2        = {ntot}
x2min      = {lo}
x2max      = {hi}
ix2_bc     = outflow
ox2_bc     = outflow
ix2_mc_bc  = escape
ox2_mc_bc  = escape

nx3        = {ntot}
x3min      = {lo}
x3max      = {hi}
ix3_bc     = outflow
ox3_bc     = outflow
ix3_mc_bc  = escape
ox3_mc_bc  = escape

<meshblock>
nx1 = {nx}
nx2 = {nx}
nx3 = {nx}

<hydro>
gamma = 1.666666666666667

""".format(ntot=nx*nb, nx=nx, lo=-half, hi=half) + mc_block(
        nphot, iseed, scattering, emission, absorption,
        dict({'mc_coord': 'kerr_schild_cartesian', 'general_pusher': 'true',
              'varystep': 'true', 'stepsize': '0.1', 'boosts': 'true',
              'initialize_comoving': 'true', 'tmax': '1.0e36'}, **extra)) + """
<problem>
input_filename = {snapshot}
{table_line}rho_cgs  = 1.0e-3
tgas_cgs = 1.0e8
vel_cgs  = 2.99792458e10
l_cgs    = 1.4766e6
heabund  = 0.09
tcut     = 5.0e6
dcut     = 1.0e-20
tnorm    = true
emin     = 1.0e-02
emax     = 3.0e+01
nescape  = 16
""".format(snapshot=snapshot,
           table_line=('opacity_filename = ' + table + '\n') if table_key == 'problem' else '')
    with open(path, 'w') as f:
        f.write(text)


def write_isoth_opac_athinput(path, table, nphot, iseed, scattering='compton',
                              emission='table', absorption='table', extra=None,
                              temp=3.e5, dens=1.e-9, taumin=1.e-3, taumax=3.e3,
                              table_key='montecarlo'):
    """
    mc_isoth_opac: the isothermal atmosphere whose density falls exponentially with
    height between the two scattering depths, escaping through the top.  32x32x64 cells
    in eight blocks, so that the test runs in seconds.
    """
    extra = dict(extra or {})
    if table_key == 'montecarlo':
        extra['opacity_file'] = table
    text = """<comment>
problem   = isothermal atmosphere, table opacities
configure = --prob=mc_isoth_opac -mc -mpi -hdf5

<job>
problem_id = mciso

<output1>
file_type = phlist

<output2>
file_type = hdf5
variable  = mclab
dt = 1.e-11

<mesh>
nx1        = 32
x1min      = -5.0e11
x1max      = 5.0e11
ix1_mc_bc  = periodic
ox1_mc_bc  = periodic

nx2        = 32
x2min      = -5.0e11
x2max      = 5.0e11
ix2_mc_bc  = periodic
ox2_mc_bc  = periodic

nx3        = 64
x3min      = 0.0
x3max      = 1.0e12
ix3_mc_bc  = absorb
ox3_mc_bc  = escape

<meshblock>
nx1 = 16
nx2 = 16
nx3 = 32

<hydro>
gamma = 1.666666666666667

""" + mc_block(nphot, iseed, scattering, emission, absorption, extra) + """
<problem>
{table_line}temp     = {temp:e}
dens     = {dens:e}
taumin   = {taumin:e}
taumax   = {taumax:e}
tnorm    = true
emin     = 1.0e-02
emax     = 3.0e+01
""".format(temp=temp, dens=dens, taumin=taumin, taumax=taumax,
           table_line=('opacity_filename = ' + table + '\n') if table_key == 'problem' else '')
    with open(path, 'w') as f:
        f.write(text)


def run(binary, athinput, workdir, mcranks=1, log='run.log'):
    """run the binary in workdir on the athinput file; returns the wall time in seconds"""
    os.makedirs(workdir, exist_ok=True)
    cmd = [os.path.abspath(binary), '-i', os.path.basename(athinput)]
    if mcranks > 1:
        cmd = ['mpirun', '-np', str(mcranks)] + cmd
    t0 = time.time()
    with open(os.path.join(workdir, log), 'w') as f:
        rc = subprocess.call(cmd, cwd=workdir, stdout=f, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    if rc != 0:
        tail = open(os.path.join(workdir, log)).read().splitlines()[-15:]
        raise RuntimeError('{} failed in {} (exit {}):\n{}'.format(
            ' '.join(cmd), workdir, rc, '\n'.join(tail)))
    return wall


def link_inputs(workdir, *paths):
    """symlink the snapshot and table into the run directory"""
    os.makedirs(workdir, exist_ok=True)
    for p in paths:
        dst = os.path.join(workdir, os.path.basename(p))
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(p), dst)


def list_files(workdir):
    return sorted(glob.glob(os.path.join(workdir, '*.list')))


def athdf_files(workdir):
    return sorted(glob.glob(os.path.join(workdir, '*.athdf')))


def lists_identical(dir_a, dir_b):
    """every photon list of dir_a has a byte-identical partner in dir_b"""
    la, lb = list_files(dir_a), list_files(dir_b)
    if len(la) != len(lb) or not la:
        return False, 'list count {} vs {}'.format(len(la), len(lb))
    for a, b in zip(la, lb):
        if not filecmp.cmp(a, b, shallow=False):
            return False, '{} differs from {}'.format(a, b)
    return True, '{} list file(s) identical'.format(len(la))


def athdf_identical(dir_a, dir_b):
    """every dataset of each athdf file of dir_a equals its partner's (raw h5py read, so
    that a file athena_read cannot place, such as one on a GR mesh, still compares)"""
    import h5py
    import numpy as np
    fa, fb = athdf_files(dir_a), athdf_files(dir_b)
    if len(fa) != len(fb):
        return False, 'athdf count {} vs {}'.format(len(fa), len(fb))
    for a, b in zip(fa, fb):
        with h5py.File(a, 'r') as ha, h5py.File(b, 'r') as hb:
            for k in ha:
                if k not in hb or not np.array_equal(ha[k][()], hb[k][()]):
                    return False, '{} differs in {}'.format(os.path.basename(a), k)
    return True, '{} athdf file(s) identical'.format(len(fa))


def grep(path, key):
    """the lines of a log holding key"""
    return [l.rstrip() for l in open(path) if key in l]
