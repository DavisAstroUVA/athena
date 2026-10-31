#! /usr/bin/env python
"""
The frequency lattice in a moving medium, in one dimension (frequency only): the
discretization of the adiabatic drift and of the frequency interface, kinetic against
lattice, with no spatial transport.  Path is in units of R0 and x in Doppler widths.

Kinetic photon at x: a flight of length l ~ Exp(1 / k(x)) along which x drifts by -u l (a
homologous expansion shifts every flight by the same amount per path), an escape with
probability 1 - exp(-l / tau(x)) during the flight (the sink that stands for the spatial
escape, tau(x) = k(x) over R0), then a scattering: in the Doppler core (|x| < xcw) the
resonant atom's velocity is x and x' = x cos(theta) + u_perp sin(theta) cos(phi) (no
memory of the drift), in the wing x' = x - 1/x + N(0, 1) (coherent, the drift kept,
with the mean shift toward the core of the wing kernel).  The
core's scatterings are many and carry no path, so the kinetic core is replaced by the
distribution of its exit frequency, tabulated once from the same kernel.

Lattice walker in bin b (the core bin and the wing bins to the band edge xe): a dwell
l ~ Exp(1 / R) with R the two frequency rates of Smith et al. (2018) eq 11 from the
bin-averaged k plus the escape rate 1 / kbar_b; during the dwell x drifts by -u l in a
wing bin (nothing in the core, whose scatterings redraw the frequency) and a drift that
reaches the bin edge is a crossing: a hop keeping x into a node bin, an emergence at the
edge into a thin bin.  A frequency hop by the rate ('uniform': x redrawn in the new bin,
'offset': the fractional offset kept); a hop into a thin bin is an emergence at the
crossed edge plus a step's overshoot; an escape leaves at x uniform in the bin (as the
spatial emergence of the code).  Entry from a thin bin: a scattering that lands in a wing
node converts with P = sqrt(2 pi) / (2 dx) and otherwise reflects about the edge; one
that lands in the core converts; a photon whose flight's drift carried it into the band
converts at its next scattering if that leaves it in a node.

Reported for each rule set: the red fraction, the mean x and the median |x| of the
escapers, the Kolmogorov distance from the kinetic walk everywhere, the mean path, the
conversions per photon and the fraction escaping from the lattice.  With --variants the
interface rules are varied instead (no reflection, emergence exactly at the edge, drift
entries tested too, the probability halved and doubled).

What it showed (R2 of compton_ddmc_plan.md, 2000 walkers, band to 18.9): at u = 0 only
the entry probability agrees with the kinetic walk (K at the sampling scale; admitting
every entry gives K 0.15); at u = 1 the rules of the code (uniform redraw at a hop, the
probability with reflection, the overshoot) are 1.6 sampling scales from the kinetic
walk, 0.01 redder in the red fraction and 1 Doppler width redder in the mean, with a path
5 percent short, which is the code's own departure from the analog run; keeping the
walker's offset at a hop is redder (0.05, K 3 scales), no reflection and an emergence at
the edge itself are worse, testing drift entries or scaling the probability moves the
numbers within the noise; at u = 10 every rule set agrees (K 0.02).  The departure at a
moderate drift is the lattice's, not a code error.

    python freq_drift_toy.py [--u 0 1 10] [--xe 18.9] [--nwalk 2000] [--variants]
"""
import argparse
import numpy as np

A = 0.0149          # Voigt a at 10 K
TAU0 = 1.e6         # line-centre depth over R0
XCW = 2.6           # core-wing transition at this a
CORE = 3.3          # the core bin's half-width
DX = 5.2            # wing bin width (0.21 (a tau0)^(1/3) at a tau0 = 1.5e4)


def kx(x):
    x = np.asarray(x, float)
    return TAU0*(np.exp(-x*x) + A/(np.sqrt(np.pi)*(x*x + 1e-30)))


def kbar(lo, hi):
    """bin average of k over [lo, hi] (|x|), by the midpoint rule on 64 points"""
    xs = lo + (np.arange(64) + 0.5)*(hi - lo)/64.
    return kx(xs).mean()


def core_exit_table(rng, n=200000):
    """|x| at which a photon entering the Doppler core leaves it, from the core kernel"""
    x = rng.normal(size=n)*0.7
    alive = np.ones(n, bool)
    out = np.zeros(n)
    while alive.any():
        idx = np.nonzero(alive)[0]
        cth = rng.uniform(-1., 1., idx.size)
        sth = np.sqrt(1. - cth*cth)
        uperp = rng.normal(size=idx.size)/np.sqrt(2.)
        xn = x[idx]*cth + uperp*sth*np.cos(rng.uniform(0., 2.*np.pi, idx.size))
        done = np.abs(xn) >= XCW
        out[idx[done]] = np.abs(xn[done])
        alive[idx[done]] = False
        x[idx] = xn
    return np.sort(out)


def kinetic_scatter(rng, x, exit_tab):
    """one scattering of kinetic photons at x: wing step, or the core's exit frequency"""
    core = np.abs(x) < XCW
    # the wing kernel: unit variance and the mean shift -1/x toward the core
    # (Osterbrock 1962), which with k ~ 1/x^2 makes the process the diffusion
    # d/dx (k/2 dJ/dx) that the lattice's rates discretize
    xn = x - 1./np.where(core, np.inf, x) + rng.normal(size=x.size)
    ncore = np.count_nonzero(core)
    if ncore:
        sign = np.where(rng.uniform(size=ncore) < 0.5, -1., 1.)
        xn[core] = sign*exit_tab[rng.integers(0, exit_tab.size, ncore)]
    return xn


def run(rng, u, xe, hop, entry, n, exit_tab, lattice=True, emerge='overshoot', pf_scale=1.,
        stats=None):
    nb = int(round((xe - CORE)/DX))           # wing node bins per side
    lo = np.concatenate([[0.], CORE + np.arange(nb + 1)*DX])   # |x| edges: bin 0 = core
    hi = np.concatenate([[CORE], CORE + (np.arange(nb + 1) + 1)*DX])
    width = hi - lo
    kb = np.array([kbar(l_, h_) for l_, h_ in zip(lo, hi)])
    # Smith eq 11 between bin m and m+1 (bin nb+1 is the first thin bin)
    r_up = np.array([1./(width[m]*(width[m]/kb[m] + width[m+1]/kb[m+1])) for m in range(nb + 1)])
    r_dn = np.array([0.] + [1./(width[m]*(width[m]/kb[m] + width[m-1]/kb[m-1]))
                            for m in range(1, nb + 1)])
    esc = 1./kb
    pf = min(1., pf_scale*np.sqrt(2.*np.pi)/(2.*DX))
    nconv = 0
    nesc_lat = 0
    x = np.zeros(n)
    path = np.zeros(n)
    alive = np.ones(n, bool)
    on = np.full(n, lattice)        # on the lattice (start in the core bin)
    m = np.zeros(n, int)            # bin index (0 core, 1..nb wing nodes)
    s = np.ones(n)                  # side of the wing bin
    xesc = np.zeros(n)
    while alive.any():
        # ---- lattice walkers: one dwell each
        w = alive & on
        if w.any():
            idx = np.nonzero(w)[0]
            mm = m[idx]
            ss = s[idx]
            xx = x[idx]
            rate = np.where(mm == 0, 2.*r_up[0], r_up[mm] + r_dn[mm]) + esc[mm]
            dt = rng.exponential(1./rate)
            # drift toward the red inside a wing bin: the path to the edge it moves to
            wing = mm > 0
            edge = np.where(ss > 0, lo[mm], hi[mm])      # |x| of the edge the drift reaches
            dist = np.where(ss > 0, np.abs(xx) - edge, edge - np.abs(xx))
            tdrift = np.where(wing & (u > 0.), dist/max(u, 1e-300), np.inf)
            cross = tdrift < dt
            dtt = np.minimum(dt, tdrift)
            path[idx] += dtt
            xx = np.where(wing, xx - u*dtt, xx)
            for k in range(idx.size):
                i = idx[k]
                mi, si = mm[k], ss[k]
                if cross[k]:
                    if si > 0:                     # blue side: inward, toward the core
                        if mi == 1:
                            m[i] = 0
                            x[i] = rng.uniform(-CORE, CORE)
                        else:
                            m[i] = mi - 1
                            x[i] = hi[mi - 1] - 1e-9
                    else:                          # red side: outward
                        if mi == nb:               # emergence at the band edge
                            on[i] = False
                            x[i] = -(hi[nb] + 1e-9)
                        else:
                            m[i] = mi + 1
                            x[i] = -(lo[mi + 1] + 1e-9)
                    continue
                # an event at the rate: escape, or a frequency hop
                r = rng.uniform()*rate[k]
                if r < esc[mi]:
                    alive[i] = False
                    nesc_lat += 1
                    xesc[i] = (rng.uniform(-CORE, CORE) if mi == 0
                               else si*rng.uniform(lo[mi], hi[mi]))
                    continue
                if mi == 0:
                    si = 1. if rng.uniform() < 0.5 else -1.
                    mn = 1
                else:
                    mn = mi + 1 if (r - esc[mi]) < r_up[mi] else mi - 1
                if mn > nb:                        # emergence into the thin bin
                    on[i] = False
                    over = (rng.uniform()*np.sqrt(-2.*np.log(rng.uniform()))
                            if emerge == 'overshoot' else 1e-9)
                    x[i] = si*(hi[nb] + over)
                    continue
                m[i] = mn
                s[i] = si
                if mn == 0:
                    x[i] = rng.uniform(-CORE, CORE)
                elif hop == 'uniform' or mi == 0:
                    x[i] = si*rng.uniform(lo[mn], hi[mn])
                else:
                    frac = (np.abs(xx[k]) - lo[mi])/width[mi]
                    x[i] = si*(lo[mn] + min(max(frac, 0.), 1.)*width[mn])
        # ---- kinetic photons: one flight and one scattering each
        w = alive & ~on
        if w.any():
            idx = np.nonzero(w)[0]
            x0 = x[idx]
            k0 = kx(x0)
            l = rng.exponential(1./k0)
            path[idx] += l
            x1 = x0 - u*l                          # the drift along the flight
            # escape along the flight
            pesc = 1. - np.exp(-l/k0)
            gone = rng.uniform(size=idx.size) < pesc
            xesc[idx[gone]] = x1[gone]
            alive[idx[gone]] = False
            x2 = kinetic_scatter(rng, x1, exit_tab)
            if lattice:
                inband1 = np.abs(x1) < xe          # in the band before the scattering (by drift)
                inband2 = np.abs(x2) < xe
                for k in np.nonzero(inband2 & ~gone)[0]:
                    i = idx[k]
                    ax = abs(x2[k])
                    mn = 0 if ax < CORE else int((ax - CORE)//DX) + 1
                    by_drift = inband1[k] and entry != 'pf_alldrift'
                    admit = by_drift or mn == 0 or entry == 'all' or rng.uniform() < pf
                    if admit:
                        on[i] = True
                        m[i] = mn
                        s[i] = 1. if x2[k] > 0 else -1.
                        nconv += 1
                    elif entry == 'pf' or (entry == 'pf_alldrift' and not inband1[k]):
                        x2[k] = np.sign(x2[k])*(2.*xe - ax)   # reflected about the band edge
                    # pf_noreflect (and a drift entry refused): stays kinetic where it is
            x[idx] = x2
    if stats is not None:
        stats['path'] = path.mean()
        stats['conv'] = nconv/n
        stats['esc_lat'] = nesc_lat/n
    return xesc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--u', type=float, nargs='+', default=[0., 1., 10.])
    ap.add_argument('--xe', type=float, default=18.9)
    ap.add_argument('--nwalk', type=int, default=2000)
    ap.add_argument('--variants', action='store_true',
                    help='variants of the interface rules instead of the hop/entry matrix')
    args = ap.parse_args()
    rng = np.random.default_rng(7)
    tab = core_exit_table(rng)
    print(f"core exit |x|: median {np.median(tab):.2f}, mean {tab.mean():.2f}")
    if args.variants:
        cases = [('uniform', 'pf', 'overshoot', 1.), ('uniform', 'pf_noreflect', 'overshoot', 1.),
                 ('uniform', 'pf', 'edge', 1.), ('uniform', 'pf_alldrift', 'overshoot', 1.),
                 ('uniform', 'pf', 'overshoot', 0.5), ('uniform', 'pf', 'overshoot', 2.),
                 ('offset', 'pf_noreflect', 'overshoot', 1.)]
    else:
        cases = [(hop, entry, 'overshoot', 1.) for hop in ('uniform', 'offset')
                 for entry in ('pf', 'all')]
    for u in args.u:
        st = {}
        xr = run(rng, u, args.xe, 'uniform', 'pf', args.nwalk, tab, lattice=False, stats=st)
        print(f"u = {u:g}: kinetic  red fraction {np.mean(xr < 0):.3f}, mean x {xr.mean():+.2f}, "
              f"|x| median {np.median(np.abs(xr)):.2f}, mean path {st['path']:.2f}")
        for hop, entry, emerge, pfs in cases:
            st = {}
            xl = run(rng, u, args.xe, hop, entry, args.nwalk, tab, emerge=emerge, pf_scale=pfs,
                     stats=st)
            dk = np.max(np.abs(np.searchsorted(np.sort(xr), np.sort(xl))/xr.size
                               - np.arange(1, xl.size + 1)/xl.size))
            print(f"   hop {hop:8s} entry {entry:12s} emerge {emerge:9s} pf x{pfs:g}: "
                  f"red {np.mean(xl < 0):.3f} ({np.mean(xl < 0) - np.mean(xr < 0):+.3f}), "
                  f"mean x {xl.mean():+.2f} ({xl.mean() - xr.mean():+.2f}), |x| median "
                  f"{np.median(np.abs(xl)):.2f}, K {dk:.3f} (scale {np.sqrt(2./args.nwalk):.3f}), "
                  f"path {st['path']:.2f}, conversions {st['conv']:.1f}, escaped on lattice "
                  f"{st['esc_lat']:.2f}")


if __name__ == '__main__':
    main()
