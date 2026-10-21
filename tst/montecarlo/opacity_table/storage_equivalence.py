#! /usr/bin/env python
"""
table_storage = cell against table_storage = global.

The two storage modes of the table opacities compute the same extinctions and emission
spectra with the same functions at different times (per cell at the build, or per call
from the rank's one table), so with the same seed they must give byte-identical photon
lists and moment files.  Each configuration below is run in both modes and compared;
the wall times are printed, since the global mode trades speed for memory.  PASS/FAIL.

  python storage_equivalence.py --path /path/to/athena --workdir DIR
         --pgen readhdf|readhdf_gr|isoth_opac [--configs NAME ...] [--nphot N]

The binary must be built for the generator named: --prob=mc_readhdf -mc -hdf5 (the
snapshot in --snapshot-dir), --prob=mc_readhdf_gr --coord=gr_user -g -mc -hdf5 (the
synthetic snapshot make_ks_snapshot.py writes, as ks_torus.athdf in --snapshot-dir) or
--prob=mc_isoth_opac -mc -hdf5.  Configurations are
added here as the phases of doc/monte_carlo/opacity_table_plan.md add features.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import opacity_test_util as u  # noqa: E402

# name: (scattering, emission, absorption, extra <montecarlo> keys)
CONFIGS = {
    'readhdf': {
        'edge_thomson': ('thomson', 'table', 'table',
                         {'table_groups': 'edge', 'table_gray': 'freefree'}),
        'edge_none': ('none', 'table', 'table',
                      {'table_groups': 'edge', 'table_gray': 'freefree'}),
        'edge_compton': ('compton', 'table', 'table',
                         {'table_groups': 'edge', 'table_gray': 'freefree'}),
        'mean_thomson': ('thomson', 'table', 'table', {'table_gray': 'freefree'}),
        'mean_compton_ext': ('compton', 'table', 'table',
                             {'table_gray': 'freefree', 'table_emax': '100.'}),
    },
    'isoth_opac': {
        'edge_compton': ('compton', 'table', 'table',
                         {'table_groups': 'edge', 'table_gray': 'freefree'}),
        'edge_none': ('none', 'table', 'table',
                      {'table_groups': 'edge', 'table_gray': 'freefree'}),
        # absorption from the table under free-free emission, and the reverse (no
        # scattering there: free-free absorption alone lets a Thomson atmosphere this
        # thick random-walk for 1e4 scatterings per photon)
        'edge_ffemis': ('thomson', 'freefree', 'table',
                        {'table_groups': 'edge', 'table_gray': 'freefree'}),
        'edge_ffabs': ('none', 'table', 'freefree',
                       {'table_groups': 'edge', 'table_gray': 'freefree'}),
        'mean_compton': ('compton', 'table', 'table', {'table_gray': 'freefree'}),
        'mean_none_ext': ('none', 'table', 'table',
                          {'table_gray': 'freefree', 'table_emax': '100.'}),
        # electron density from the table's free electrons per atom: needs a table with
        # an nfree file beside it (--snapshot-dir /home/swd8g/opacity_table
        # --table out_uniform_nfreq32_solabun.txt)
        'mean_ne': ('thomson', 'table', 'table', {'table_ne': 'true', 'nfree_file': 'auto'}),
        # the two-band split, plain and with the high band thermalized (same table)
        'mean_twoband': ('thomson', 'table', 'table',
                         {'table_ne': 'true', 'nfree_file': 'auto', 'two_band': 'true',
                          'abs_method': 'prob'}),
        'mean_twoband_thick': ('compton', 'table', 'table',
                               {'table_ne': 'true', 'nfree_file': 'auto', 'two_band': 'true',
                                'two_band_thick': '10.', 'abs_method': 'prob'}),
    },
    'readhdf_gr': {
        # the synthetic Kerr-Schild torus of make_ks_snapshot.py (--snapshot-dir holds it
        # as ks_torus.athdf): the general pusher's per-step opacity refresh, the tcut
        # mask, and the escape tables of the biased scheme
        'edge_thomson': ('thomson', 'table', 'table',
                         {'table_groups': 'edge', 'table_gray': 'freefree'}),
        'edge_compton_biased': ('compton', 'table', 'table',
                                {'table_groups': 'edge', 'table_gray': 'freefree',
                                 'weights': 'biased'}),
        'mean_compton_biased': ('compton', 'table', 'table',
                                {'table_gray': 'freefree', 'weights': 'biased',
                                 'table_emax': '100.'}),
    },
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--path', default=os.path.join(HERE, '..', '..', '..'),
                    help='root of the tree (bin/athena), or a binary')
    ap.add_argument('--workdir', default='.', help='where to run')
    ap.add_argument('--pgen', default='readhdf', choices=sorted(CONFIGS))
    ap.add_argument('--configs', nargs='*', default=None,
                    help='names from the table in this script; default all')
    ap.add_argument('--snapshot-dir', default=u.SNAPSHOT_DIR)
    ap.add_argument('--table', default=u.TABLE, help='opacity table in snapshot-dir')
    ap.add_argument('--nphot', type=int, default=100000)
    ap.add_argument('--iseed', type=int, default=132639)
    ap.add_argument('--mcranks', type=int, default=1)
    args = ap.parse_args()

    binary = args.path if os.path.isfile(args.path) else os.path.join(args.path, 'bin', 'athena')
    snap_name = u.SNAPSHOT_GR if args.pgen == 'readhdf_gr' else u.SNAPSHOT
    snapshot = os.path.join(args.snapshot_dir, snap_name)
    table = os.path.join(args.snapshot_dir, args.table)
    for p in (binary, table) + ((snapshot,) if args.pgen != 'isoth_opac' else ()):
        if not os.path.exists(p):
            sys.exit('missing: ' + p)
    configs = CONFIGS[args.pgen]
    names = args.configs or sorted(configs)
    workdir = os.path.abspath(args.workdir)
    nfree = table[:-4] + '_nfree.txt'

    allok = True
    for name in names:
        scattering, emission, absorption, extra = configs[name]
        times = {}
        dirs = {}
        for mode in ('cell', 'global'):
            d = os.path.join(workdir, args.pgen, name, mode)
            dirs[mode] = d
            keys = dict(extra, table_storage=mode)
            if keys.get('nfree_file') == 'auto':
                if not os.path.exists(nfree):
                    sys.exit('{} needs {}'.format(name, nfree))
                keys['nfree_file'] = os.path.basename(nfree)
            if args.pgen == 'readhdf':
                u.link_inputs(d, snapshot, table)
                ath = os.path.join(d, 'athinput.readhdf')
                u.write_readhdf_athinput(ath, u.SNAPSHOT, args.table, args.nphot, args.iseed,
                                         scattering, emission, absorption, keys)
            elif args.pgen == 'readhdf_gr':
                u.link_inputs(d, snapshot, table)
                ath = os.path.join(d, 'athinput.readhdf_gr')
                u.write_readhdf_gr_athinput(ath, u.SNAPSHOT_GR, args.table, args.nphot,
                                            args.iseed, scattering, emission, absorption,
                                            keys)
            else:
                u.link_inputs(d, table)
                if os.path.exists(nfree):
                    u.link_inputs(d, nfree)
                ath = os.path.join(d, 'athinput.mciso')
                u.write_isoth_opac_athinput(ath, args.table, args.nphot, args.iseed,
                                            scattering, emission, absorption, keys)
            times[mode] = u.run(binary, ath, d, args.mcranks)
        ok_l, msg_l = u.lists_identical(dirs['cell'], dirs['global'])
        ok_a, msg_a = u.athdf_identical(dirs['cell'], dirs['global'])
        stats = u.grep(os.path.join(dirs['cell'], 'run.log'), 'ntot:')
        print('{}/{}: {}'.format(args.pgen, name, stats[-1] if stats else '?'))
        print('  cell {:.1f} s, global {:.1f} s; {}; {}'.format(
            times['cell'], times['global'], msg_l, msg_a))
        allok = allok and ok_l and ok_a
    print('PASS' if allok else 'FAIL')
    sys.exit(0 if allok else 1)


if __name__ == '__main__':
    main()
