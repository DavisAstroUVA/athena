#! /usr/bin/env python
"""
A multigroup opacity table of free-free absorption and Thomson scattering, in the layout
tops_regrid.py writes (nfre, ntem, nrho; group lower bounds [keV]; T [keV]; rho [g/cm^3];
grey Rosseland; grey Planck; Rosseland group means including scattering; Planck group
means, absorption only; cm^2/g), with the free electrons per atom beside it
(<out>_nfree.txt).  The gas is the Monte Carlo module's fully ionized hydrogen and helium
with heabund helium nuclei per hydrogen nucleus, and the free-free opacity is the
module's own,

    chi_ff = ffnrm n_e n_ion T^-1/2 nu^-3 (1 - e^-x),  ffnrm = 3.692146e8,

so a run on this table can be held against emission = freefree, absorption = freefree.
Each Planck group mean is the B_nu-weighted mean over the group and each Rosseland mean
the dB/dT-weighted harmonic mean of chi_ff + chi_es, both integrated numerically on a
fine grid in ln nu; the grey means integrate over all frequencies.

  python make_ff_table.py OUT.txt [--ngroups 32 --emin 1e-3 --emax 7.4989
                                   --nt 120 --tmin 5.8e3 --tmax 1e9
                                   --nrho 121 --rhomin 1e-18 --rhomax 1e-6
                                   --tgrid-from TABLE.txt --heabund 0.09 --digits 8]

--emin and --emax are the lowest and highest group lower bounds in keV (the top group's
upper bound continues the geometric spacing); --tgrid-from takes the temperature grid of
an existing table instead of a uniform one.  The functions are importable: planck_group
and rosseland_group give the means for any (T, rho, group) the tests need.
"""
import argparse

import numpy as np

H = 6.62607015e-27
C = 2.99792458e10
KB = 1.380649e-16
KEV = 1.602176634e-9
MP = 1.67262192369e-24
MU = 1.66053906660e-24
SIGMA_T = 6.65248e-25
FFNRM = 3.692146e8
KEV_K = KEV/KB


def number_densities(rho, heabund):
    nh = rho/(MP*(1. + 4.*heabund))
    nhe = nh*heabund
    return nh + 2.*nhe, nh + 4.*nhe  # n_e, n_ion (sum Z^2 n)


def kappa_ff(nu, temp, rho, heabund):
    """free-free opacity per gram [cm^2/g] at frequency nu (array)"""
    ne, nion = number_densities(rho, heabund)
    x = H*nu/(KB*temp)
    return ne*nion*FFNRM/np.sqrt(temp)/nu**3*(-np.expm1(-x))/rho


def kappa_es(rho, heabund):
    ne, _ = number_densities(rho, heabund)
    return SIGMA_T*ne/rho


def planck_nu(nu, temp):
    x = H*nu/(KB*temp)
    return 2.*H*nu**3/C**2/np.expm1(x)


def dplanck_dt(nu, temp):
    """dB_nu/dT up to a factor common to all nu at fixed T (x^4 e^x/(e^x-1)^2 in nu^0)"""
    x = H*nu/(KB*temp)
    ex = np.exp(-x)
    return nu**4*ex/(-np.expm1(-x))**2


def _lognu(elo, ehi, n):
    """n-point grid in ln nu over [elo, ehi] (erg) with trapezoid weights d ln nu"""
    lnu = np.linspace(np.log(elo/H), np.log(ehi/H), n)
    w = np.full(n, lnu[1] - lnu[0])
    w[0] *= 0.5
    w[-1] *= 0.5
    return np.exp(lnu), w


def planck_group(elo, ehi, temp, rho, heabund, n=401):
    """Planck mean of kappa_ff over [elo, ehi] (erg), cm^2/g"""
    nu, w = _lognu(elo, ehi, n)
    b = planck_nu(nu, temp)*nu*w  # B_nu dnu
    den = b.sum()
    if den <= 0.:
        return float(kappa_ff(np.sqrt(elo*ehi)/H, temp, rho, heabund))
    return float((kappa_ff(nu, temp, rho, heabund)*b).sum()/den)


def rosseland_group(elo, ehi, temp, rho, heabund, n=401):
    """Rosseland mean of kappa_ff + kappa_es over [elo, ehi] (erg), cm^2/g"""
    nu, w = _lognu(elo, ehi, n)
    d = dplanck_dt(nu, temp)*nu*w
    den = d.sum()
    ktot = kappa_ff(nu, temp, rho, heabund) + kappa_es(rho, heabund)
    if den <= 0.:
        return float(ktot[len(ktot)//2])
    return float(den/(d/ktot).sum())


def grey_means(temp, rho, heabund, n=2001):
    """grey Planck (absorption) and Rosseland (with scattering) means over all nu"""
    kt = KB*temp
    nu, w = _lognu(1.e-8*kt, 60.*kt, n)
    b = planck_nu(nu, temp)*nu*w
    d = dplanck_dt(nu, temp)*nu*w
    kff = kappa_ff(nu, temp, rho, heabund)
    kp = (kff*b).sum()/b.sum()
    kr = d.sum()/(d/(kff + kappa_es(rho, heabund))).sum()
    return float(kp), float(kr)


def group_edges(ngroups, emin_kev, emax_kev):
    """the ngroups+1 edges in erg, the first ngroups being the lower bounds written"""
    dle = (np.log10(emax_kev) - np.log10(emin_kev))/(ngroups - 1)
    return 10.**(np.log10(emin_kev) + dle*np.arange(ngroups + 1))*KEV


def write_tables(out, edges, tgrid, rgrid, heabund=0.09, digits=8):
    """edges in erg (ngroups+1), tgrid in K, rgrid in g/cm^3"""
    ng, nt, nr = len(edges) - 1, len(tgrid), len(rgrid)
    plan = np.zeros((ng, nt, nr))
    ross = np.zeros((ng, nt, nr))
    gp = np.zeros((nt, nr))
    gr = np.zeros((nt, nr))
    for j, temp in enumerate(tgrid):
        for i, rho in enumerate(rgrid):
            gp[j, i], gr[j, i] = grey_means(temp, rho, heabund)
            for g in range(ng):
                plan[g, j, i] = planck_group(edges[g], edges[g+1], temp, rho, heabund)
                ross[g, j, i] = rosseland_group(edges[g], edges[g+1], temp, rho, heabund)
    fmt = '{:.%dE}' % digits
    with open(out, 'w') as f:
        f.write('{}\n{}\n{}\n'.format(ng, nt, nr))
        f.writelines(fmt.format(e/KEV) + '\n' for e in edges[:-1])
        f.writelines(fmt.format(t/KEV_K) + '\n' for t in tgrid)
        f.writelines(fmt.format(r) + '\n' for r in rgrid)
        for block in [gr, gp] + [ross[g] for g in range(ng)] + [plan[g] for g in range(ng)]:
            for j in range(nt):
                f.write(' '.join(fmt.format(v) for v in block[j]) + '\n')
    # free electrons per atom: atoms are the nuclei, electrons those of full ionization
    nfree = (1. + 2.*heabund)/(1. + heabund)
    abar = (1. + 4.*heabund)/(1. + heabund)*MP/MU
    stem = out[:-4] if out.endswith('.txt') else out
    with open(stem + '_nfree.txt', 'w') as f:
        f.write('# N_free, free electrons per atom, fully ionized H + He with {:g} He per H '
                '(make_ff_table.py)\n'.format(heabund))
        f.write('# Abar = {:.5f}  Z_full = {:.5f}  n_e = rho * N_free / (Abar * m_u)\n'.format(
            abar, nfree))
        f.write('# layout: nT, nrho, T [keV] (nT lines), rho [g/cm^3] (nrho lines), '
                'then nT rows of nrho values\n')
        f.write('{}\n{}\n'.format(nt, nr))
        f.writelines(fmt.format(t/KEV_K) + '\n' for t in tgrid)
        f.writelines(fmt.format(r) + '\n' for r in rgrid)
        for j in range(nt):
            f.write(' '.join(fmt.format(nfree) for _ in range(nr)) + '\n')
    return plan, ross


def read_tgrid(path):
    """the temperature grid [K] of an existing table"""
    with open(path) as f:
        ng, nt, nr = (int(f.readline()) for _ in range(3))
        for _ in range(ng):
            f.readline()
        return np.array([float(f.readline()) for _ in range(nt)])*KEV_K


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out')
    ap.add_argument('--ngroups', type=int, default=32)
    ap.add_argument('--emin', type=float, default=1.e-3, help='lowest lower bound [keV]')
    ap.add_argument('--emax', type=float, default=7.4989, help='highest lower bound [keV]')
    ap.add_argument('--nt', type=int, default=120)
    ap.add_argument('--tmin', type=float, default=5.8e3, help='[K]')
    ap.add_argument('--tmax', type=float, default=1.e9, help='[K]')
    ap.add_argument('--nrho', type=int, default=121)
    ap.add_argument('--rhomin', type=float, default=1.e-18)
    ap.add_argument('--rhomax', type=float, default=1.e-6)
    ap.add_argument('--tgrid-from', default=None, help='take the T grid of this table')
    ap.add_argument('--heabund', type=float, default=0.09)
    ap.add_argument('--digits', type=int, default=8)
    args = ap.parse_args()
    edges = group_edges(args.ngroups, args.emin, args.emax)
    if args.tgrid_from:
        tgrid = read_tgrid(args.tgrid_from)
    else:
        tgrid = np.geomspace(args.tmin, args.tmax, args.nt)
    rgrid = np.geomspace(args.rhomin, args.rhomax, args.nrho)
    write_tables(args.out, edges, tgrid, rgrid, args.heabund, args.digits)
    print('{}: {} groups {:.4g} to {:.4g} keV, {} temperatures, {} densities'.format(
        args.out, args.ngroups, edges[0]/KEV, edges[-1]/KEV, len(tgrid), len(rgrid)))


if __name__ == '__main__':
    main()
