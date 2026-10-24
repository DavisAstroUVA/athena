#! /usr/bin/env python
"""
Table opacities across a load-balancing repack.

With <loadbalancing> test_repack = local|mesh every block is packed, deleted, built anew
and unpacked at the top of each transport (no block changes rank), and the new block
rebuilds its table arrays from the fluid it receives.  A run with the knob must give
byte-identical photon lists and moment files to the plain run; the table arrays are
not carried, so this is the check that the rebuild reproduces them, with the two-band
split's five arrays included.  Each configuration below runs plain, local and mesh and
compares; on one rank the lists must be identical, and that is the PASS/FAIL.  On
several ranks the asynchronous photon exchange orders the random draws by arrival, two
plain runs already differ (proc2 and proc3 of four on this snapshot), so a run there
only prints the escaper counts of the three modes for the eye.

  python repack_equivalence.py --path /path/to/athena --workdir DIR
         [--configs NAME ...] [--mcranks 1 [4 ...]] [--nphot N]

The binary is --prob=mc_readhdf -mc -mpi -hdf5 on the small snapshot of
--snapshot-dir; the table (with its _nfree file beside it, for the configurations that
need one) comes from --table in that directory.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import opacity_test_util as u  # noqa: E402

# name: (scattering, emission, absorption, extra keys; mc_block puts the table's in <opacitytable>)
CONFIGS = {
    'mean_compton': ('compton', 'table', 'table', {}),
    'mean_twoband_thick': ('compton', 'table', 'table',
                           {'table_ne': 'true', 'nfree_file': 'auto', 'two_band': 'true',
                            'two_band_thick': '10.', 'abs_method': 'prob'}),
    'mean_twoband_global': ('compton', 'table', 'table',
                            {'table_ne': 'true', 'nfree_file': 'auto', 'two_band': 'true',
                             'two_band_thick': '10.', 'abs_method': 'prob',
                             'table_storage': 'global'}),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--path', default=os.path.join(HERE, '..', '..', '..'),
                    help='root of the tree (bin/athena), or a binary')
    ap.add_argument('--workdir', default='.', help='where to run')
    ap.add_argument('--configs', nargs='*', default=None,
                    help='names from the table in this script; default all')
    ap.add_argument('--snapshot-dir', default=u.SNAPSHOT_DIR)
    ap.add_argument('--table-dir', default=None,
                    help='directory of the table; default the snapshot directory')
    ap.add_argument('--table', default=u.TABLE)
    ap.add_argument('--nphot', type=int, default=100000)
    ap.add_argument('--iseed', type=int, default=132639)
    ap.add_argument('--mcranks', type=int, nargs='+', default=[1],
                    help='rank counts; the comparison is bitwise on one rank only')
    args = ap.parse_args()

    binary = args.path if os.path.isfile(args.path) else os.path.join(args.path, 'bin', 'athena')
    snapshot = os.path.join(args.snapshot_dir, u.SNAPSHOT)
    table = os.path.join(args.table_dir or args.snapshot_dir, args.table)
    for p in (binary, table, snapshot):
        if not os.path.exists(p):
            sys.exit('missing: ' + p)
    nfree = table[:-4] + '_nfree.txt'
    names = args.configs or sorted(CONFIGS)
    workdir = os.path.abspath(args.workdir)

    allok = True
    for name in names:
        scattering, emission, absorption, extra = CONFIGS[name]
        keys = dict(extra)
        if keys.get('nfree_file') == 'auto':
            if not os.path.exists(nfree):
                sys.exit('{} needs {}'.format(name, nfree))
            keys['nfree_file'] = os.path.basename(nfree)
        for nr in args.mcranks:
            dirs = {}
            times = {}
            for mode in ('none', 'local', 'mesh'):
                d = os.path.join(workdir, name, 'np{}'.format(nr), mode)
                dirs[mode] = d
                u.link_inputs(d, snapshot, table)
                if os.path.exists(nfree):
                    u.link_inputs(d, nfree)
                ath = os.path.join(d, 'athinput.readhdf')
                u.write_readhdf_athinput(ath, u.SNAPSHOT, args.table, args.nphot, args.iseed,
                                         scattering, emission, absorption,
                                         dict(keys, test_repack=mode))
                times[mode] = u.run(binary, ath, d, nr)
            stats = u.grep(os.path.join(dirs['none'], 'run.log'), 'ntot:')
            print('{} on {} rank(s): {}'.format(name, nr, stats[-1] if stats else '?'))
            if nr > 1:
                for mode in ('local', 'mesh'):
                    stats = u.grep(os.path.join(dirs[mode], 'run.log'), 'ntot:')
                    print('  {:5s} {:.1f} s: {} (not bitwise on several ranks)'.format(
                        mode, times[mode], stats[-1] if stats else '?'))
                continue
            for mode in ('local', 'mesh'):
                ok_l, msg_l = u.lists_identical(dirs['none'], dirs[mode])
                ok_a, msg_a = u.athdf_identical(dirs['none'], dirs[mode])
                print('  {:5s} {:.1f} s against {:.1f} s: {}; {}'.format(
                    mode, times[mode], times['none'], msg_l, msg_a))
                allok = allok and ok_l and ok_a
    print('PASS' if allok else 'FAIL')
    sys.exit(0 if allok else 1)


if __name__ == '__main__':
    main()
