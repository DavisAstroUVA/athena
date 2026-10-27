#! /usr/bin/env python
"""
The module's table opacities against the generator-owned ones they replaced.

mc_readhdf is run on the small isothermal-atmosphere snapshot twice with the same seed:
once with a binary built from a tree that predates the port (emission = user,
absorption = user, <problem>/opacity_filename, the tables inside the generator), once
with the ported tree (emission = table, absorption = table, <opacitytable>/opacity_file,
table_groups = edge, table_gray = freefree).  The two must give byte-identical photon
lists and identical moment files: the port keeps the arithmetic and the order of the
random draws, so nothing weaker than identity is expected.  PASS/FAIL.

  python port_equivalence.py --reference /path/to/old/bin/athena --path /path/to/athena
         --workdir DIR [--snapshot-dir ~/mc_test/read_hdf5]
         [--table out_opacity_table_nfreq32.txt] [--nphot 200000]
         [--scattering thomson|none|compton] [--storage cell|global]
         [--pgen readhdf|readhdf_gr] [--weights emission|biased]

With --scattering none the two are not identical: the reference binary roulettes samples
whose weight the absorption has taken to zero, and a tenth of them survive to be
destroyed by the no-scattering handler, whereas the module absorbs them outright.
With --pgen readhdf_gr identity holds only where no cell is excluded: the reader's hook
now zeroes the species of cells above tcut, below dcut and inside the horizon, where
the pre-port generator gave them the densities of a 1e-30 g/cm^3 floor, and the
extinction of 1e-31 against 0 in those cells moves a few flights by round-off.  There
the comparison is of the escaped, absorbed and removed counts within four times their
Poisson scatter, and the output says so.
Both binaries must be built with --prob=mc_readhdf -mc -hdf5 (and -mpi if --mcranks > 1).
The old one comes from a checkout of main before the port, e.g.
  git archive <commit> | tar -x -C DIR && cd DIR && python configure.py ... && make
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import opacity_test_util as u  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reference', required=True, help='binary built before the port')
    ap.add_argument('--path', default=os.path.join(HERE, '..', '..', '..'),
                    help='root of the ported tree (bin/athena), or a binary')
    ap.add_argument('--workdir', default='.', help='where to run')
    ap.add_argument('--snapshot-dir', default=u.SNAPSHOT_DIR)
    ap.add_argument('--table', default=u.TABLE, help='opacity table in snapshot-dir')
    ap.add_argument('--nphot', type=int, default=200000)
    ap.add_argument('--iseed', type=int, default=132639)
    ap.add_argument('--scattering', default='thomson', choices=['thomson', 'none', 'compton'])
    ap.add_argument('--storage', default='cell', choices=['cell', 'global'],
                    help='table_storage of the ported run')
    ap.add_argument('--mcranks', type=int, default=1)
    ap.add_argument('--pgen', default='readhdf', choices=['readhdf', 'readhdf_gr'],
                    help='readhdf_gr runs the synthetic snapshot of make_ks_snapshot.py')
    ap.add_argument('--weights', default='emission', choices=['emission', 'equal', 'biased'])
    args = ap.parse_args()

    gr = (args.pgen == 'readhdf_gr')
    writer = u.write_readhdf_gr_athinput if gr else u.write_readhdf_athinput
    snap_name = u.SNAPSHOT_GR if gr else u.SNAPSHOT
    ath_name = 'athinput.' + args.pgen
    new_bin = args.path if os.path.isfile(args.path) else os.path.join(args.path, 'bin', 'athena')
    snapshot = os.path.join(args.snapshot_dir, snap_name)
    table = os.path.join(args.snapshot_dir, args.table)
    for p in (args.reference, new_bin, snapshot, table):
        if not os.path.exists(p):
            sys.exit('missing: ' + p)

    workdir = os.path.abspath(args.workdir)
    dirs = {'old': os.path.join(workdir, 'old'), 'new': os.path.join(workdir, 'new')}
    for d in dirs.values():
        u.link_inputs(d, snapshot, table)
    # the pre-port generator: tables under <problem>, user emission and absorption
    writer(os.path.join(dirs['old'], ath_name), snap_name, args.table, args.nphot,
           args.iseed, args.scattering, emission='user', absorption='user',
           extra={'weights': args.weights}, table_key='problem', legacy=True)
    # the module: the same table through the shared path, reading it the old way
    writer(os.path.join(dirs['new'], ath_name), snap_name, args.table, args.nphot,
           args.iseed, args.scattering, emission='table', absorption='table',
           extra={'table_groups': 'edge', 'table_gray': 'freefree',
                  'table_storage': args.storage, 'weights': args.weights})
    t_old = u.run(args.reference, ath_name, dirs['old'], args.mcranks)
    t_new = u.run(new_bin, ath_name, dirs['new'], args.mcranks)

    ok_l, msg_l = u.lists_identical(dirs['old'], dirs['new'])
    ok_a, msg_a = u.athdf_identical(dirs['old'], dirs['new'])
    old_stats = u.grep(os.path.join(dirs['old'], 'run.log'), 'ntot:')
    new_stats = u.grep(os.path.join(dirs['new'], 'run.log'), 'ntot:')
    print('reference: {}  ({:.1f} s)'.format(old_stats[-1] if old_stats else '?', t_old))
    print('ported:    {}  ({:.1f} s, storage {})'.format(
        new_stats[-1] if new_stats else '?', t_new, args.storage))
    print('photon lists: ' + msg_l)
    print('moment files: ' + msg_a)
    ok = ok_l and ok_a
    if not ok and gr and old_stats and new_stats:
        # excluded cells carry zero species in the ported reader, a 1e-30 density floor
        # in the pre-port one: a statistical comparison of the counts is what is left
        ok = counts_agree(old_stats[-1], new_stats[-1])
        print('readhdf_gr excludes cells with zero species, the reference with a density '
              'floor; counts {} within four Poisson sigma'.format(
                  'agree' if ok else 'DIFFER'))
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


def counts_agree(old_line, new_line, nsigma=4.):
    """the nesc, nabs and nrem of two 'ntot:' report lines within nsigma of Poisson"""
    import re
    import math
    for key in ('nesc', 'nabs', 'nrem'):
        a = re.search(key + r': (\d+)', old_line)
        b = re.search(key + r': (\d+)', new_line)
        if a is None or b is None:
            continue
        a, b = int(a.group(1)), int(b.group(1))
        if abs(a - b) > nsigma*math.sqrt(max(a, b, 1)):
            return False
    return True


if __name__ == '__main__':
    main()
