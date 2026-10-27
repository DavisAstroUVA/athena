#!/usr/bin/env python
"""
tops_regrid.py -- resample a LANL TOPS multigroup opacity table onto a grid
that is uniform in log T and log rho, extrapolating off the TOPS table where
needed, and test the extrapolation by hold-out.

Input is the raw TOPS results page (the long text file saved from
aphysics2.lanl.gov, containing the WARNING block, the grey "Rosseland and
Planck opacities and free electrons" blocks and the "Multigroup opacities"
blocks).

TOPS table structure (measured from the raw output)
---------------------------------------------------
* Temperatures are TOPS's own irregular grid.
* Requested densities are returned exactly when they lie on the table.
  Requests below the table edge are clamped to the exact edge density, which
  is listed in the "Den Used" column of the WARNING block.  The clamped cells
  therefore hold the exact opacity at (T, rho_min(T)).
* The edge is a line of constant electron degeneracy,
      rho_min(T) * N_free / T^1.5 = const        (T in K),
  so the table is rectangular in (T, n_e T^-3/2).

Method
------
inside the table : interpolate between temperature rows along lines of
                   constant height above the edge (constant n_e T^-3/2),
                   log-log linear.
below the edge   : at fixed T, power law in rho for the absorption part,
                   kappa_abs = kappa_abs(edge) * (rho/rho_min)^alpha, with
                   alpha fitted over the first --fit-decades (default 0.35,
                   about 4 grid points) above the edge and clipped to
                   --alpha-clip.  Rosseland keeps electron scattering as a
                   floor, capped at the edge value so the extrapolation is
                   continuous:
                     grey   sigma_T n_e/rho / (1 + (T/T0)^p), T0 and p fitted
                            to TOPS's own edge values (close to Buchler &
                            Yueh 1976);
                     groups sigma_T n_e/rho x (1 + a kT/m_e c^2)(1 - b h nu/m_e c^2),
                            a and b fitted to TOPS (about 2.2 and 3).
above T_max      : fixed rho; continue each field's measured slope in T over
                   the top --tslope-rows rows (default).  --highT ff uses the
                   free-free shape kappa_nu ~ T^-1/2 (1 - exp(-h nu/kT)) per
                   group and T^-3.5 for the grey means.
below T_min      : clamp to the lowest row.

See README.md for motivation, validation results and caveats.

Examples
--------
  python tops_regrid.py info TOPS_n32_TDEephot_0.0001kev_100kev.txt
  python tops_regrid.py regrid TOPS_n32_TDEephot_0.0001kev_100kev.txt \\
         -o out_uniform_nfreq32.txt --rho-range 1e-16 1e-6 --n-rho 100 \\
         --T-range 5.8e3 1e9 --n-T 80 --plot out_uniform_nfreq32.png
  python tops_regrid.py holdout TOPS_n32_TDEephot_0.0001kev_100kev.txt \\
         --cut 1 2 --plot holdout.png

The output of "regrid" has the same layout as the out_opacity_table_*.txt
files read by FreOpacity (old_method/freq_opacity_table_interpolator.py).
"""

import argparse
import sys
from collections import OrderedDict

import numpy as np

KEV_K = 1.160451812e7          # 1 keV / k_B in K
SIGMA_T_OVER_MU = 0.400617     # sigma_T / m_u  [cm^2 g^-1]
ME_C2_K = 5.9299e9             # m_e c^2 / k_B in K
ME_C2_KEV = 511.0
LOG_FLOOR = 1e-300

# bit flags for the branch map
B_BELOW_EDGE = 1
B_ABOVE_TMAX = 2
B_BELOW_TMIN = 4
B_ABOVE_RHOMAX = 8


def kn_factor(T, T0=4.5e8, p=0.86):
    """Relativistic reduction of the Rosseland scattering opacity,
    1/(1 + (T/T0)^p).  Defaults are Buchler & Yueh (1976) with the density
    term dropped; TopsTable refits T0, p to the table's own edge values."""
    return 1.0 / (1.0 + (np.asarray(T) / T0) ** p)


def _approx_mass(Z):
    return {1: 1.008, 2: 4.0026}.get(Z, 2.0 * Z)


# --------------------------------------------------------------------------
# Raw TOPS reader
# --------------------------------------------------------------------------
class TopsTable:
    """Everything in a raw TOPS results page.

    Attributes
    ----------
    T, T_keV  : (nT,) temperatures [K], [keV]
    rho       : (nrho,) requested densities [g/cm^3]
    E_lo      : (nfre,) group lower bounds [keV]; E_hi, E_c upper bound, centre
    F         : (nf, nT, nrho) fields: [grey Ross, grey Planck,
                Ross group 0..nfre-1, Planck group 0..nfre-1]
    is_ross   : (nf,) bool
    nfree     : (nT, nrho) free electrons per atom
    clamped   : (nT, nrho) bool, True where TOPS substituted the edge density
    rho_min   : (nT,) exact edge density per row (rho[0] if the row has no
                clamped cells; edge_known is then False)
    """

    def __init__(self, path):
        self.path = path
        with open(path) as fh:
            lines = fh.read().splitlines()

        self.T_keV = self._grid(lines, 'Temperature grid used')
        self.rho = self._grid(lines, 'Density grid used')
        self.E_lo = self._grid(lines, 'Photon grid used')
        self.T = self.T_keV * KEV_K
        nT, nr, ng = len(self.T), len(self.rho), len(self.E_lo)
        self.nT, self.nrho, self.nfre = nT, nr, ng
        # the last group's upper bound is not in the file; continue the
        # geometric spacing (same convention as FreOpacity)
        self.E_hi = np.append(self.E_lo[1:], self.E_lo[-1] ** 2 / self.E_lo[-2])
        self.E_c = np.sqrt(self.E_lo * self.E_hi)

        it = {v: k for k, v in enumerate(self.T_keV)}
        ir = {v: k for k, v in enumerate(self.rho)}

        # WARNING block: (T, rho requested, rho used)
        self.clamped = np.zeros((nT, nr), dtype=bool)
        self.rho_min = self.rho[0] * np.ones(nT)
        self.edge_known = np.zeros(nT, dtype=bool)
        for k, l in enumerate(lines):
            if l.split() == ['Temp', 'Den', 'Req', 'Den', 'Used']:
                for l2 in lines[k + 1:]:
                    p = l2.split()
                    try:
                        t, rq, ru = (float(v) for v in p)
                    except ValueError:
                        break
                    j, i = it[t], ir[rq]
                    self.clamped[j, i] = True
                    self.rho_min[j] = ru
                    self.edge_known[j] = True
                break

        # composition
        self.composition = []   # (number fraction, mass fraction, Z, symbol)
        for k, l in enumerate(lines):
            if l.startswith('Normalized composition'):
                for l2 in lines[k + 2:]:
                    p = l2.split()
                    if len(p) != 5:
                        break
                    self.composition.append((float(p[0]), float(p[1]), int(p[2]), p[3]))
                break
        if self.composition:
            x = np.array([c[0] for c in self.composition])
            Z = [c[2] for c in self.composition]
            self.abar = float(np.sum(x * np.array([_approx_mass(z) for z in Z])) / np.sum(x))
            self.X_H = sum(c[1] for c in self.composition if c[2] == 1)
            self.Z_full = float(np.sum(x * np.array(Z)) / np.sum(x))
        else:
            print('warning: no composition block; assuming solar-like Abar=1.3, X=0.7',
                  file=sys.stderr)
            self.abar, self.X_H, self.Z_full = 1.3, 0.7, None

        # grey blocks
        greyR = np.full((nT, nr), np.nan)
        greyP = np.full((nT, nr), np.nan)
        self.nfree = np.full((nT, nr), np.nan)
        for k, l in enumerate(lines):
            if l.lstrip().startswith('Density') and 'Ross opa' in l and 'T=' in l:
                j = it[float(l.split('T=')[1])]
                blk = np.array([lines[k + 1 + i].split()[:4] for i in range(nr)], dtype=float)
                greyR[j], greyP[j], self.nfree[j] = blk[:, 1], blk[:, 2], blk[:, 3]

        # multigroup blocks
        try:
            img = lines.index('Multigroup opacities')
        except ValueError:
            raise SystemExit(f'{path}: no "Multigroup opacities" section; '
                             'was the table requested with a photon grid?')
        mgR = np.full((ng, nT, nr), np.nan)
        mgP = np.full((ng, nT, nr), np.nan)
        hdr, data = [], []
        for l in lines[img + 1:]:
            if 'for T, density' in l:
                p = l.split()
                hdr.append((it[float(p[-2])], ir[float(p[-1])]))
            elif len(l.split()) == 3:
                data.append(l)
            elif l.strip() and len(data) >= len(hdr) * ng:
                break   # web-page footer after the last block
        data = np.array(' '.join(data).split(), dtype=float).reshape(len(hdr), ng, 3)
        if not np.allclose(data[0, :, 0], self.E_lo, rtol=1e-4):
            raise SystemExit(f'{path}: multigroup energies do not match the photon grid')
        for b, (j, i) in enumerate(hdr):
            mgR[:, j, i] = data[b, :, 1]
            mgP[:, j, i] = data[b, :, 2]

        # Free electrons per atom are carried as one more field, the ionization
        # deficit Z_full - N_free, so that extrapolating it to low density or
        # high T can never exceed full ionization.
        if self.Z_full is None:
            self.Z_full = float(np.nanmax(self.nfree))
        deficit = np.maximum(self.Z_full - self.nfree, 0.0)
        self.F = np.concatenate([greyR[None], greyP[None], mgR, mgP, deficit[None]])
        if np.isnan(self.F).any():
            raise SystemExit(f'{path}: {np.isnan(self.F).sum()} missing table values')
        self.nf = self.F.shape[0]
        self.i_nfree = self.nf - 1      # index of the deficit field
        self.is_ross = np.zeros(self.nf, dtype=bool)
        self.is_ross[0] = True
        self.is_ross[2:2 + ng] = True

        # In clamped cells TOPS fills every group with the grey mean, so the
        # per-group values there are not the edge values.
        c = self.clamped
        self.mg_clamped_is_grey = bool(c.any() and
            np.allclose(self.F[2:2 + ng][:, c], self.F[0][c]) and
            np.allclose(self.F[2 + ng:2 + 2 * ng][:, c], self.F[1][c]))

        # edge invariant log10(rho_min N_free / T^1.5) using N_free at the edge
        self.nfree_edge = np.array([self.nfree[j, self._edge_index(j)] for j in range(nT)])
        self.log_sigma_edge = np.log10(self.rho_min * self.nfree_edge / self.T ** 1.5)

        # Electron scattering: Thomson x N_free/Abar x 1/(1+(T/T0)^p).  At the
        # edge above ~1e6 K the grey Rosseland mean is pure scattering, so fit
        # T0, p there; fall back to Buchler & Yueh if that is not possible.
        self.kn_T0, self.kn_p, self.kn_fitted = 4.5e8, 0.86, False
        thom = SIGMA_T_OVER_MU * self.nfree_edge / self.abar
        ratio = np.array([self.F[0, j, self._edge_index(j)] for j in range(nT)]) / thom
        y = 1.0 / ratio - 1.0
        sel = self.edge_known & (self.T > 3e6) & (y > 1e-3)
        if sel.sum() >= 3:
            p, b = np.polyfit(np.log10(self.T[sel]), np.log10(y[sel]), 1)
            if 0.3 < p < 2.0:
                self.kn_T0, self.kn_p, self.kn_fitted = 10 ** (-b / p), p, True

        # Per-group Rosseland scattering: at the first grid point above the
        # edge for T > 3e6 K, (Ross_g - Planck_g) / Thomson is pure scattering;
        # fit log of it as a*theta - b*x.
        self.gs_a, self.gs_b, self.gs_fitted = 2.2, 3.0, False
        rows = [j for j in range(nT) if self.edge_known[j] and self.T[j] > 3e6]
        if len(rows) >= 3:
            th_, x_, y_ = [], [], []
            for j in rows:
                i = int(np.argmax(~self.clamped[j]))
                thom = SIGMA_T_OVER_MU * self.nfree[j, i] / self.abar
                r = (self.F[2:2 + ng, j, i] - self.F[2 + ng:2 + 2 * ng, j, i]) / thom
                ok = (r > 0) & (self.E_c * KEV_K / self.T[j] < 20)
                th_ += [self.T[j] / ME_C2_K] * int(ok.sum())
                x_ += list(self.E_c[ok] / ME_C2_KEV)
                y_ += list(np.log(r[ok]))
            A = np.column_stack([th_, -np.array(x_)])
            (a, b), *_ = np.linalg.lstsq(A, np.array(y_), rcond=None)
            if 0 < a < 10 and 0 < b < 10:
                self.gs_a, self.gs_b, self.gs_fitted = a, b, True

    def kn(self, T):
        return kn_factor(T, self.kn_T0, self.kn_p)

    def group_scat_shape(self, T):
        """Per-group Rosseland scattering relative to Thomson,
        (1 + a kT/m_e c^2)(1 - b h nu/m_e c^2)."""
        return (1 + self.gs_a * T / ME_C2_K) * (1 - self.gs_b * self.E_c / ME_C2_KEV)

    def kes_fields(self, T, nfree):
        """Electron-scattering opacity [cm^2/g] for every field (0 for Planck).
        Grey Rosseland: Thomson x thermal relativistic reduction (fitted).
        Rosseland groups: Thomson x group_scat_shape (fitted); unlike the grey
        mean these rise slightly with T."""
        th = SIGMA_T_OVER_MU * nfree / self.abar
        out = np.zeros(self.nf)
        out[0] = th * self.kn(T)
        out[2:2 + self.nfre] = th * self.group_scat_shape(T)
        return out

    @staticmethod
    def _grid(lines, prefix):
        for k, l in enumerate(lines):
            if l.startswith(prefix):
                n = int(l.split()[-2])
                vals = []
                k += 1
                while len(vals) < n:
                    vals += [float(v) for v in lines[k].split()]
                    k += 1
                return np.array(vals)
        raise SystemExit(f'no "{prefix}" line found; is this a raw TOPS results page?')

    def _edge_index(self, j):
        """Index of a grid cell that holds the edge value of row j."""
        return int(np.argmax(self.clamped[j])) if self.clamped[j].any() else 0

    def row_points(self, j):
        """(log10 rho, values (nf, n), N_free) of the genuine table data in
        row j.  The exact edge is the first point; fields whose edge value
        TOPS did not supply are NaN there."""
        valid = ~self.clamped[j]
        logr = np.log10(self.rho[valid])
        vals = self.F[:, j, valid]
        nfr = self.nfree[j, valid]
        if self.edge_known[j]:
            le = np.log10(self.rho_min[j])
            if len(logr) == 0 or logr[0] - le > 1e-6:
                i0 = self._edge_index(j)
                logr = np.insert(logr, 0, le)
                edge = self.F[:, j, i0:i0 + 1].copy()
                if self.mg_clamped_is_grey:
                    edge[2:self.i_nfree] = np.nan   # only grey means and N_free are genuine
                vals = np.concatenate([edge, vals], axis=1)
                nfr = np.insert(nfr, 0, self.nfree[j, i0])
        return logr, vals, nfr

    def field_label(self, f):
        if f == 0:
            return 'grey Rosseland'
        if f == 1:
            return 'grey Planck'
        g = (f - 2) % self.nfre
        kind = 'Ross' if f < 2 + self.nfre else 'Planck'
        return f'{kind} group {g} ({self.E_c[g]:.3g} keV)'

    def field_sets(self):
        g = self.nfre
        return OrderedDict([
            ('Planck groups', np.arange(2 + g, 2 + 2 * g)),
            ('Planck grey', np.array([1])),
            ('Ross groups', np.arange(2, 2 + g)),
            ('Ross grey', np.array([0])),
            ('N_free', np.array([self.i_nfree])),
        ])


# --------------------------------------------------------------------------
# Method options
# --------------------------------------------------------------------------
class Options:
    """How to extrapolate/interpolate.  See add_method_args for meanings."""

    def __init__(self, slope='fit', fit_decades=0.35, alpha_fixed=1.0,
                 alpha_clip=(0.0, 1.2), alpha_smooth=1, alpha_relax=None, alpha_far=1.0,
                 highT='tslope', tslope_rows=3):
        self.__dict__.update(locals())
        del self.__dict__['self']


# --------------------------------------------------------------------------
# One temperature row
# --------------------------------------------------------------------------
class Row:
    def __init__(self, tab, j, opt, cut=0.0):
        self.T = tab.T[j]
        self.j = j
        logr, v, nfr = tab.row_points(j)
        if cut > 0:
            keep = logr >= np.log10(tab.rho_min[j]) + cut - 1e-9
            if keep.sum() < 2:
                keep[-2:] = True
            logr, v, nfr = logr[keep], v[:, keep], nfr[keep]
        v = v.copy()
        self.logr = logr
        self.lb = logr[0]
        ross = tab.is_ross
        bad0 = np.isnan(v[:, 0])
        ref = np.where(bad0, v[:, min(1, v.shape[1] - 1)], v[:, 0])  # first genuine value

        # electron-scattering floor (Rosseland fields only)
        self.kes = np.zeros(tab.nf)
        self.kes[ross] = np.minimum(tab.kes_fields(self.T, nfr[0])[ross], ref[ross])

        # slopes d log(kappa - kes) / d log rho at the edge
        a = v - self.kes[:, None]
        if opt.slope == 'fixed':
            alpha = np.full(tab.nf, opt.alpha_fixed)
        else:
            sel = logr <= self.lb + opt.fit_decades + 1e-9
            if sel.sum() < 3:
                sel[:3] = True
            x = logr[sel]
            alpha = np.full(tab.nf, opt.alpha_fixed)
            for f in range(tab.nf):
                y = a[f, sel]
                ok = np.isfinite(y) & (y > 1e-6 * v[f, sel])
                if ok.sum() >= 2:
                    alpha[f] = np.polyfit(x[ok], np.log10(y[ok]), 1)[0]
        self.alpha = np.clip(alpha, *opt.alpha_clip)

        # fill missing edge values by stepping down from the first grid point
        self.relax, self.alpha_far = opt.alpha_relax, opt.alpha_far
        if bad0.any():
            dx = np.array([self.lb - logr[1]])
            v[bad0, 0] = (self.kes[bad0]
                          + np.maximum(ref[bad0] - self.kes[bad0], 0.0)
                          * 10 ** self._dlog(dx)[bad0, 0])
        self.v = v
        self.logv = np.log10(np.maximum(v, LOG_FLOOR))
        self.a0 = np.maximum(v[:, 0] - self.kes, 0.0)

    def _dlog(self, dx):
        """Change of log10 kappa_abs for steps dx <= 0 below the edge, (nf, n).
        The slope relaxes from alpha at the edge to alpha_far over
        alpha_relax dex:  s(d) = a_far + (alpha - a_far) exp(-d / D)."""
        a = self.alpha[:, None]
        if not self.relax:
            return a * dx[None, :]
        D, af = self.relax, self.alpha_far
        d = -dx[None, :]
        return -(af * d + (a - af) * D * (1 - np.exp(-d / D)))

    def eval(self, logq):
        """Values (nf, nq) at log10 rho = logq, at this row's temperature."""
        logq = np.atleast_1d(logq)
        out = np.empty((self.v.shape[0], len(logq)))
        ins = logq >= self.lb
        if ins.any():
            for f in range(out.shape[0]):
                out[f, ins] = 10 ** np.interp(logq[ins], self.logr, self.logv[f])
        if (~ins).any():
            dx = logq[~ins] - self.lb
            out[:, ~ins] = self.kes[:, None] + self.a0[:, None] * 10 ** self._dlog(dx)
        return out


# --------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------
class Model:
    def __init__(self, tab, opt, cut=0.0, rows=None):
        self.tab, self.opt = tab, opt
        rows = range(tab.nT) if rows is None else rows
        self.rows = [Row(tab, j, opt, cut) for j in rows]
        self.logT = np.log10([r.T for r in self.rows])
        self.logrho_max = np.log10(tab.rho[-1])
        if opt.alpha_smooth > 1:
            A = np.array([r.alpha for r in self.rows])
            h = opt.alpha_smooth // 2
            for k, r in enumerate(self.rows):
                r.alpha = np.median(A[max(0, k - h):k + h + 1], axis=0)
        # group centre energies for each field (nan for grey)
        g = tab.nfre
        self.E_field = np.full(tab.nf, np.nan)
        self.E_field[2:2 + g] = tab.E_c
        self.E_field[2 + g:2 + 2 * g] = tab.E_c

    def _high_T(self, T, logq):
        b = self.rows[-1]
        vb = b.eval(logq)
        r = T / b.T
        tab, opt = self.tab, self.opt
        kes_T = b.kes.copy()
        kes_T[0] *= tab.kn(T) / tab.kn(b.T)                      # grey Rosseland
        g = tab.nfre
        kes_T[2:2 + g] *= tab.group_scat_shape(T) / tab.group_scat_shape(b.T)
        a = np.maximum(vb - b.kes[:, None], 0.0)
        if opt.highT == 'ff':
            s = np.full(tab.nf, r ** -3.5)
            grp = ~np.isnan(self.E_field)
            xT = self.E_field[grp] * KEV_K / T
            xb = self.E_field[grp] * KEV_K / b.T
            s[grp] = r ** -0.5 * (-np.expm1(-xT)) / (-np.expm1(-xb))
            return kes_T[:, None] + a * s[:, None]
        # tslope: continue each field's slope over the top rows, at fixed rho
        K = min(opt.tslope_rows, len(self.rows))
        lt = self.logT[-K:]
        la = np.array([np.log10(np.maximum(rw.eval(logq) - rw.kes[:, None], 1e-30))
                       for rw in self.rows[-K:]])          # (K, nf, nq)
        lt_c = lt - lt.mean()
        beta = np.tensordot(lt_c, la - la.mean(axis=0), axes=(0, 0)) / np.sum(lt_c ** 2)
        beta = np.clip(beta, -4.0, 0.5)
        return kes_T[:, None] + a * r ** beta

    def evaluate(self, T, logq):
        """Values (nf, nq) and branch flags (nq,) at temperature T [K]."""
        logq = np.atleast_1d(np.asarray(logq, dtype=float))
        logT = np.log10(T)
        lT = self.logT
        codes = np.zeros(len(logq), dtype=np.int8)
        if logT > lT[-1] + 1e-9:
            vals = self._high_T(T, logq)
            codes |= B_ABOVE_TMAX
            codes[logq < self.rows[-1].lb] |= B_BELOW_EDGE
        elif logT < lT[0] - 1e-9:
            vals = self.rows[0].eval(logq)
            codes |= B_BELOW_TMIN
            codes[logq < self.rows[0].lb] |= B_BELOW_EDGE
        else:
            j = int(np.clip(np.searchsorted(lT, logT, side='right') - 1, 0, len(lT) - 2))
            w = (logT - lT[j]) / (lT[j + 1] - lT[j])
            r1, r2 = self.rows[j], self.rows[j + 1]
            # interpolate at constant height above the edge (constant n_e T^-1.5)
            x = logq - ((1 - w) * r1.lb + w * r2.lb)
            v1, v2 = r1.eval(r1.lb + x), r2.eval(r2.lb + x)
            below = x < 0
            vals = 10 ** ((1 - w) * np.log10(np.maximum(v1, LOG_FLOOR))
                          + w * np.log10(np.maximum(v2, LOG_FLOOR)))
            codes[below] |= B_BELOW_EDGE
        codes[logq > self.logrho_max + 1e-9] |= B_ABOVE_RHOMAX
        return vals, codes

    def edge_logrho(self, T):
        """log10 rho_min at temperature T (rows interpolated in log T;
        held at the end rows outside the table)."""
        return np.interp(np.log10(T), self.logT, [r.lb for r in self.rows])


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def write_nfree(path, tab, T_K, rho, nfree):
    """Free electrons per atom on the output grid."""
    fmt = '{:.5E}'.format
    with open(path, 'w') as fh:
        fh.write(f'# N_free, free electrons per atom, from {tab.path} (tops_regrid.py)\n')
        fh.write(f'# Abar = {tab.abar:.5f}  Z_full = {tab.Z_full:.5f}  '
                 'n_e = rho * N_free / (Abar * m_u)\n')
        fh.write('# layout: nT, nrho, T [keV] (nT lines), rho [g/cm^3] (nrho lines), '
                 'then nT rows of nrho values\n')
        fh.write(f'{len(T_K)}\n{len(rho)}\n')
        fh.writelines(fmt(t / KEV_K) + '\n' for t in T_K)
        fh.writelines(fmt(r) + '\n' for r in rho)
        for j in range(len(T_K)):
            fh.write(' '.join(fmt(x) for x in nfree[j]) + '\n')


def write_table(path, tab, T_K, rho, vals, rossabs=None, digits=4):
    """vals: (nf, nT, nrho).  Same layout as the out_opacity_table files; rossabs
    (nfre, nT, nrho), the Rosseland absorption, is appended as a further nfre blocks
    after the Planck means (--rossabs), which readers that stop at the Planck block
    never reach."""
    fmt = ('{:.%dE}' % digits).format
    g = tab.nfre
    with open(path, 'w') as fh:
        fh.write(f'{g}\n{len(T_K)}\n{len(rho)}\n')
        fh.writelines(fmt(e) + '\n' for e in tab.E_lo)
        fh.writelines(fmt(t / KEV_K) + '\n' for t in T_K)
        fh.writelines(fmt(r) + '\n' for r in rho)
        order = [0, 1] + list(range(2, 2 + g)) + list(range(2 + g, 2 + 2 * g))
        for f in order:
            for j in range(len(T_K)):
                fh.write(' '.join(fmt(x) for x in vals[f, j]) + '\n')
        if rossabs is not None:
            for k in range(g):
                for j in range(len(T_K)):
                    fh.write(' '.join(fmt(x) for x in rossabs[k, j]) + '\n')


# --------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Two-band (picket-fence) split of each group
# --------------------------------------------------------------------------
TB_GAP0 = 1        # kappa_low = 0 closure used (chosen, or fallback from 'w')
TB_SINGLE = 2      # kappa_R >= kappa_P + kappa_es: one band, kappa_h = kappa_l = kappa_P
TB_FLOOR = 4       # Rosseland absorption below the floor; raised to the floor


def two_band(kP, kR, s, closure='w', w0=0.01, floor=0.003):
    """Split one group into a high band (fraction w, absorption kh) and a low
    band (fraction 1-w, absorption kl), with the same scattering s in both,
    so that

        w kh + (1-w) kl                         = kP   (Planck: emission)
        [ w/(kh+s) + (1-w)/(kl+s) ]^-1          = kR   (Rosseland, total)

    closure 'w'   : w = w0 fixed; falls back to 'gap0' where that would need kl < 0.
    closure 'gap0': kl = 0 (gaps hold scattering only); w follows.
    Where the Rosseland absorption kR - s is below floor*s (too small to be
    known), it is raised to floor*s.  Where kR >= kP + s (no line structure to
    represent), one band is returned.

    All inputs are arrays of the same shape.  Returns w, kh, kl, status flags.
    """
    kP, kR, s = (np.asarray(x, dtype=float) for x in (kP, kR, s))
    status = np.zeros(kP.shape, dtype=np.int8)
    Ra = kR - s
    low = Ra < floor * s
    status[low] |= TB_FLOOR
    Ra = np.where(low, floor * s, Ra)
    R = Ra + s
    Pt = kP + s
    single = R >= Pt * (1 - 1e-9)
    status[single] |= TB_SINGLE

    # gap0 closure: kl = 0, kh = kP R / Ra - s, w = kP / kh
    with np.errstate(divide='ignore', invalid='ignore'):
        kh_g = kP * R / Ra - s
        w_g = np.where(kh_g > 0, kP / kh_g, 1.0)
    w, kh, kl = w_g, kh_g, np.zeros_like(kP)

    if closure == 'w':
        # with a = kh + s:  w0 a^2 + a [R(1-2w0) - Pt] + R w0 Pt = 0, larger root
        b = Pt - R * (1 - 2 * w0)
        D = np.maximum(b ** 2 - 4 * w0 ** 2 * R * Pt, 0.0)
        a = (b + np.sqrt(D)) / (2 * w0)
        kh_w = a - s
        kl_w = (Pt - w0 * a) / (1 - w0) - s
        ok = kl_w >= 0
        w = np.where(ok, w0, w_g)
        kh = np.where(ok, kh_w, kh_g)
        kl = np.where(ok, kl_w, 0.0)
        status[~ok & ~single] |= TB_GAP0
    else:
        status[~single] |= TB_GAP0

    w = np.where(single, 1.0, w)
    kh = np.where(single, kP, kh)
    kl = np.where(single, kP, kl)
    return w, kh, kl, status


def two_band_check(w, kh, kl, s):
    """Planck and Rosseland means reproduced by a two-band split."""
    P = w * kh + (1 - w) * kl
    R = 1.0 / (w / (kh + s) + (1 - w) / (kl + s))
    return P, R


def write_twoband(path, tab, T_K, rho, w, kh, kl, s, closure, w0, floor):
    """w, kh, kl, s: (nfre, nT, nrho)."""
    fmt = '{:.5E}'.format
    with open(path, 'w') as fh:
        fh.write(f'# two-band split of each group, from {tab.path} (tops_regrid.py)\n')
        fh.write(f'# closure = {closure}' + (f', w = {w0:g}' if closure == 'w' else '')
                 + f', Rosseland-absorption floor = {floor:g} x kappa_es\n')
        fh.write('# in each group g: high band, frequency fraction w, absorption kappa_h;\n')
        fh.write('#                  low band, fraction 1-w, absorption kappa_l;\n')
        fh.write('#                  electron scattering kappa_es in both bands.\n')
        fh.write('# w kappa_h + (1-w) kappa_l = Planck group mean;\n')
        fh.write('# [w/(kappa_h+kappa_es) + (1-w)/(kappa_l+kappa_es)]^-1 = Rosseland group mean.\n')
        fh.write('# Monte Carlo: emit in band b with probability w_b kappa_b / kappa_P, absorb\n')
        fh.write('# with kappa_b, and redraw the band with probability w_b after electron scattering.\n')
        fh.write('# layout: nfre, nT, nrho, group lower bounds [keV], T [keV], rho [g/cm^3],\n')
        fh.write('#   then for each group: w (nT rows of nrho), kappa_h, kappa_l, kappa_es [cm^2/g]\n')
        g = tab.nfre
        fh.write(f'{g}\n{len(T_K)}\n{len(rho)}\n')
        fh.writelines(fmt(e) + '\n' for e in tab.E_lo)
        fh.writelines(fmt(t / KEV_K) + '\n' for t in T_K)
        fh.writelines(fmt(r) + '\n' for r in rho)
        for k in range(g):
            for arr in (w, kh, kl, s):
                for j in range(len(T_K)):
                    fh.write(' '.join(fmt(x) for x in arr[k, j]) + '\n')


def default_twoband_path(output):
    stem = output[:-4] if output.endswith('.txt') else output
    return stem + '_twoband.txt'


def default_nfree_path(output):
    stem = output[:-4] if output.endswith('.txt') else output
    return stem + '_nfree.txt'


def cmd_info(args):
    tab = TopsTable(args.raw)
    print(f'file            : {tab.path}')
    print(f'temperatures    : {tab.nT}, {tab.T[0]:.4g} - {tab.T[-1]:.4g} K '
          f'({tab.T_keV[0]:.4g} - {tab.T_keV[-1]:.4g} keV), irregular in log T')
    print(f'densities       : {tab.nrho}, {tab.rho[0]:.4g} - {tab.rho[-1]:.4g} g/cm^3')
    print(f'photon groups   : {tab.nfre}, {tab.E_lo[0]:.4g} - {tab.E_hi[-1]:.4g} keV '
          '(last upper bound extrapolated)')
    print(f'composition     : {len(tab.composition)} elements, X = {tab.X_H:.4f}, '
          f'Abar = {tab.abar:.4f}, full ionization N_free = {tab.Z_full:.4f}')
    print(f'clamped cells   : {tab.clamped.sum()} of {tab.clamped.size}; '
          f'{tab.edge_known.sum()} of {tab.nT} rows have a known edge')
    ls = tab.log_sigma_edge[tab.edge_known]
    if len(ls):
        print('edge invariant  : log10(rho_min N_free / T^1.5) = '
              f'{ls.mean():.4f} +- {ls.std():.1e} (min {ls.min():.4f}, max {ls.max():.4f})')
        lr = (np.log10(tab.rho_min) - 1.5 * np.log10(tab.T))[tab.edge_known]
        print(f'                 log10(rho_min / T^1.5) ranges {lr.min():.4f} to {lr.max():.4f} '
              '(differs only where N_free < its high-T value)')
    if tab.mg_clamped_is_grey:
        print('clamped cells   : per-group values are copies of the grey mean '
              '(not edge values); only grey edge values are used')
    j = tab.nT - 1
    src = 'fitted to TOPS edge' if tab.kn_fitted else 'Buchler & Yueh default'
    print(f'scattering      : sigma_T n_e/rho / (1 + (T/{tab.kn_T0:.3g} K)^{tab.kn_p:.3f})  ({src})')
    print(f'                  at T = {tab.T[j]:.3g} K: TOPS edge grey Ross = '
          f'{tab.F[0, j, tab._edge_index(j)]:.4f}, '
          f'formula = {tab.kes_fields(tab.T[j], tab.nfree_edge[j])[0]:.4f}')
    src = 'fitted to TOPS' if tab.gs_fitted else 'default'
    print('                  per-group Rosseland: Thomson x (1 + '
          f'{tab.gs_a:.2f} kT/mc^2)(1 - {tab.gs_b:.2f} h nu/mc^2)  ({src})')
    if args.verbose:
        print('\n   T [K]      rho_min      N_free  log10(rho_min N_free/T^1.5)')
        for j in range(tab.nT):
            print(f'{tab.T[j]:10.4e}  {tab.rho_min[j]:10.4e}  {tab.nfree_edge[j]:7.4f}  '
                  f'{tab.log_sigma_edge[j]:9.4f}{"" if tab.edge_known[j] else "  (no clamping)"}')


def options_from_args(args):
    return Options(slope='fit', fit_decades=args.fit_decades,
                   alpha_clip=tuple(args.alpha_clip), alpha_smooth=args.alpha_smooth,
                   alpha_relax=args.alpha_relax or None,
                   highT=args.highT, tslope_rows=args.tslope_rows)


def cmd_regrid(args):
    tab = TopsTable(args.raw)
    opt = options_from_args(args)
    model = Model(tab, opt)
    T_range = args.T_range or (tab.T[0], tab.T[-1])
    T_out = np.logspace(np.log10(T_range[0]), np.log10(T_range[1]), args.n_T)
    rho_out = np.logspace(np.log10(args.rho_range[0]), np.log10(args.rho_range[1]), args.n_rho)
    logq = np.log10(rho_out)

    vals = np.empty((tab.nf, len(T_out), len(rho_out)))
    codes = np.empty((len(T_out), len(rho_out)), dtype=np.int8)
    for j, T in enumerate(T_out):
        vals[:, j], codes[j] = model.evaluate(T, logq)
    nfree = np.clip(tab.Z_full - vals[tab.i_nfree], 0.0, tab.Z_full)
    rossabs = None
    if args.rossabs:
        # the Rosseland group means less the scattering of this script's model, so that
        # a reader can add its own scattering back without the ill-conditioned
        # subtraction of two numbers near kappa_es written to a few digits
        g = tab.nfre
        thom = SIGMA_T_OVER_MU * nfree / tab.abar
        shape = np.array([tab.group_scat_shape(T) for T in T_out])
        s_model = np.transpose(shape)[:, :, None] * thom[None]
        rossabs = vals[2:2 + g] - s_model
    write_table(args.output, tab, T_out, rho_out, vals, rossabs, args.digits)
    nfree_path = None
    if not args.no_nfree:
        nfree_path = args.nfree or default_nfree_path(args.output)
        write_nfree(nfree_path, tab, T_out, rho_out, nfree)

    tb_report = None
    if args.twoband is not None:
        g = tab.nfre
        tb_path = args.twoband or default_twoband_path(args.output)
        thom = SIGMA_T_OVER_MU * nfree / tab.abar                       # (nT, nrho)
        shape = np.array([tab.group_scat_shape(T) for T in T_out])     # (nT, nfre)
        s = np.transpose(shape)[:, :, None] * thom[None]              # (nfre, nT, nrho)
        kR, kP = vals[2:2 + g], vals[2 + g:2 + 2 * g]
        w, kh, kl, st = two_band(kP, kR, s, args.twoband_closure, args.twoband_w,
                                 args.twoband_floor)
        write_twoband(tb_path, tab, T_out, rho_out, w, kh, kl, s,
                      args.twoband_closure, args.twoband_w, args.twoband_floor)
        P2, R2 = two_band_check(w, kh, kl, s)
        ok = (st & TB_SINGLE) == 0
        tiny = kP < args.twoband_floor * s
        tb_report = (tb_path, st, tiny, np.max(np.abs(P2 / kP - 1)),
                     np.max(np.abs(R2[ok] / np.maximum(kR[ok], s[ok] * (1 + args.twoband_floor)) - 1))
                     if ok.any() else 0.0)

    depth = np.array([[max(0.0, model.edge_logrho(min(T, tab.T[-1])) - lq) for lq in logq]
                      for T in T_out])
    n = codes.size
    print(f'wrote {args.output}: {tab.nfre} groups, {len(T_out)} T x {len(rho_out)} rho')
    print(f'  T   {T_out[0]:.4g} - {T_out[-1]:.4g} K   (table {tab.T[0]:.4g} - {tab.T[-1]:.4g} K)')
    print(f'  rho {rho_out[0]:.4g} - {rho_out[-1]:.4g} g/cm^3')
    print(f'  inside table          : {np.mean(codes == 0) * 100:5.1f}% of cells')
    for bit, name in [(B_BELOW_EDGE, 'below density edge'), (B_ABOVE_TMAX, 'above T_max'),
                      (B_BELOW_TMIN, 'below T_min (clamped)'),
                      (B_ABOVE_RHOMAX, 'above rho_max (held)')]:
        m = (codes & bit) != 0
        if m.any():
            print(f'  {name:22s}: {m.sum() / n * 100:5.1f}% of cells')
    if (depth > 0).any():
        print(f'  deepest extrapolation below the edge: {depth.max():.2f} dex in rho '
              '(the hold-out test only checks the first ~1-2 dex)')
    if nfree_path:
        print(f'wrote {nfree_path} (free electrons per atom on the same grid)')
    if tb_report:
        path, st, tiny, eP, eR = tb_report
        n = st.size
        print(f'wrote {path} (two-band split of every group)')
        if args.twoband_closure == 'w':
            print(f'  w = {args.twoband_w:g} closure used : '
                  f'{np.mean((st & (TB_GAP0 | TB_SINGLE)) == 0) * 100:5.1f}% of (group, T, rho) cells')
        print(f'  kappa_low = 0 (gap0)      : {np.mean((st & TB_GAP0) != 0) * 100:5.1f}%')
        sb = (st & TB_SINGLE) != 0
        print(f'  single band               : {np.mean(sb) * 100:5.1f}%, of which')
        print(f'    absorption negligible   : {np.mean(sb & tiny) * 100:5.1f}%  '
              f'(kappa_P < {args.twoband_floor:g} kappa_es)')
        print(f'    no line structure       : {np.mean(sb & ~tiny) * 100:5.1f}%  '
              '(Rosseland absorption >= Planck: smooth continuum or Wien tail)')
        print(f'  Rosseland absorption floor: {np.mean((st & TB_FLOOR) != 0) * 100:5.1f}%')
        print(f'  check: max |Planck error| {eP:.1e}, max |Rosseland error| {eR:.1e} '
              '(single-band cells keep Planck exact and are excluded from the Rosseland check)')
    if args.branches:
        np.savetxt(args.branches, codes, fmt='%d',
                   header=f'branch flags, rows = T ({len(T_out)}), cols = rho ({len(rho_out)}); '
                          '0 table, 1 below edge, 2 above T_max, 4 below T_min, 8 above rho_max')
        print(f'wrote {args.branches}')
    if args.plot:
        plot_regrid(args.plot, tab, model, T_out, rho_out, vals, codes)
        print(f'wrote {args.plot}')


def plot_regrid(path, tab, model, T_out, rho_out, vals, codes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lr, lt = np.log10(rho_out), np.log10(T_out)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    im = ax[0].pcolormesh(lr, lt, codes, shading='auto', cmap='tab10', vmin=0, vmax=10)
    ax[0].set_title('branch: 0 table, 1 <edge, 2 >Tmax, 3 both, 4 <Tmin', fontsize=10)
    fig.colorbar(im, ax=ax[0])
    for a, f, name in [(ax[1], 1, r'grey Planck $\log_{10}\kappa_P$'),
                       (ax[2], 0, r'grey Rosseland $\log_{10}\kappa_R$')]:
        im = a.pcolormesh(lr, lt, np.log10(vals[f]), shading='auto', cmap='viridis')
        fig.colorbar(im, ax=a)
        a.set_title(name)
    for a in ax:
        a.plot(np.log10(tab.rho_min), np.log10(tab.T), 'w--', lw=1.2, label='TOPS edge')
        a.axhline(np.log10(tab.T[-1]), c='w', ls=':', lw=1)
        a.set_xlabel(r'$\log_{10}\rho$ [g cm$^{-3}$]')
        a.set_ylabel(r'$\log_{10}T$ [K]')
        a.set_xlim(lr[0], lr[-1])
        a.set_ylim(lt[0], lt[-1])
    ax[0].legend(loc='upper left', fontsize=8)
    fig.savefig(path, dpi=130)


# ---------------------------- hold-out ------------------------------------
def _stats(err):
    e = np.abs(np.concatenate([np.ravel(x) for x in err])) if err else np.array([])
    e = e[np.isfinite(e)]
    if e.size == 0:
        return np.nan, np.nan, np.nan, 0
    return np.median(e), np.percentile(e, 90), e.max(), e.size


def _errors(tab, j, pred, truth, max_x):
    """log10(pred/truth); NaN where TOPS gave no genuine value or where the
    group is deep in the Wien tail (h nu > max_x kT)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        err = np.log10(np.maximum(pred, LOG_FLOOR) / truth)
    wien = tab.E_c * KEV_K / tab.T[j] > max_x
    g = tab.nfre
    err[2:2 + g][wien] = np.nan
    err[2 + g:2 + 2 * g][wien] = np.nan
    # N_free: compare N_free itself, not the deficit
    k = tab.i_nfree
    err[k] = np.log10(np.maximum(tab.Z_full - pred[k], 1e-30)
                      / np.maximum(tab.Z_full - truth[k], 1e-30))
    return err


def _print_table(title, results, sets):
    print(f'\n{title}')
    print('  |error| in dex = |log10(predicted / TOPS)|:  median / 90th pct / max  (N points)')
    for q in sets:
        print(f'  {q}')
        for m, res in results.items():
            md, p90, mx, n = _stats(res[q])
            print(f'    {m:26s} {md:8.3f} {p90:8.3f} {mx:8.3f}   ({n})')


def holdout_low_rho(tab, methods, cut, max_x):
    """Hide the first `cut` decades above the edge, extrapolate into them."""
    sets = tab.field_sets()
    res = OrderedDict()
    perT = OrderedDict()
    for name, opt in methods.items():
        model = Model(tab, opt, cut=cut)
        res[name] = {q: [] for q in sets}
        perT[name] = np.full(tab.nT, np.nan)
        for j, row in enumerate(model.rows):
            logr, truth, _ = tab.row_points(j)
            hid = logr < row.lb - 1e-9
            if not hid.any():
                continue
            err = _errors(tab, j, row.eval(logr[hid]), truth[:, hid], max_x)
            for q, fs in sets.items():
                res[name][q].append(err[fs])
            e = np.abs(err[sets['Planck groups']])
            if np.isfinite(e).any():
                perT[name][j] = np.nanpercentile(e, 90)
    return res, perT


def holdout_high_T(tab, methods, hide, max_x):
    """Hide the top `hide` temperature rows, extrapolate up into them."""
    sets = tab.field_sets()
    keep = list(range(tab.nT - hide))
    res, perG = OrderedDict(), OrderedDict()
    for name, opt in methods.items():
        model = Model(tab, opt, rows=keep)
        res[name] = {q: [] for q in sets}
        eg = []
        for j in range(tab.nT - hide, tab.nT):
            logr, truth, _ = tab.row_points(j)
            pred, _ = model.evaluate(tab.T[j], logr)
            err = _errors(tab, j, pred, truth, max_x)
            for q, fs in sets.items():
                res[name][q].append(err[fs])
            eg.append(np.abs(err[sets['Planck groups']]))
        perG[name] = np.nanmedian(np.concatenate(eg, axis=1), axis=1)
    return res, perG


def holdout_interp(tab, methods, max_x, near_edge=1.0):
    """Drop every other T row, interpolate back to it."""
    sets = tab.field_sets()
    keep = list(range(0, tab.nT, 2))
    if keep[-1] != tab.nT - 1:
        keep.append(tab.nT - 1)
    test = [j for j in range(tab.nT) if j not in keep]
    res_all, res_edge = OrderedDict(), OrderedDict()
    for name, opt in methods.items():
        model = Model(tab, opt, rows=keep)
        res_all[name] = {q: [] for q in sets}
        res_edge[name] = {q: [] for q in sets}
        for j in test:
            logr, truth, _ = tab.row_points(j)
            pred, _ = model.evaluate(tab.T[j], logr)
            err = _errors(tab, j, pred, truth, max_x)
            ne = logr < logr[0] + near_edge
            for q, fs in sets.items():
                res_all[name][q].append(err[fs])
                res_edge[name][q].append(err[fs][:, ne])
    return res_all, res_edge


def cmd_holdout(args):
    tab = TopsTable(args.raw)
    sets = tab.field_sets()
    base = options_from_args(args)

    def variant(**kw):
        d = dict(base.__dict__)
        d.update(kw)
        return Options(**d)

    low_methods = OrderedDict([
        ('kramers (alpha=1)', variant(slope='fixed', alpha_fixed=1.0, alpha_relax=None)),
        ('flat (alpha=0)', variant(slope='fixed', alpha_fixed=0.0, alpha_relax=None)),
        ('fit 0.35 dex, no relax', variant(fit_decades=0.35, alpha_relax=None)),
        ('fit 1 dex, no relax', variant(fit_decades=1.0, alpha_relax=None)),
        ('fit 0.35, relax 0.5 dex', variant(fit_decades=0.35, alpha_relax=0.5)),
        ('fit 0.35, relax 1 dex', variant(fit_decades=0.35, alpha_relax=1.0)),
        ('fit 0.35, relax 2 dex', variant(fit_decades=0.35, alpha_relax=2.0)),
    ])
    print(f'Errors exclude groups with h nu > {args.max_x:g} kT (Wien tail) and the '
          'per-group values TOPS copies from the grey mean in clamped cells.')
    figdata = {}
    for cut in args.cut:
        res, perT = holdout_low_rho(tab, low_methods, cut, args.max_x)
        _print_table(f'=== low-density hold-out: first {cut:g} dex above the edge hidden, '
                     'extrapolated at fixed T ===', res, sets)
        figdata[f'low{cut}'] = perT

    hi_methods = OrderedDict([
        ('free-free shape (ff)', variant(highT='ff')),
        (f'measured T-slope ({args.tslope_rows} rows)', variant(highT='tslope')),
    ])
    res, perG = holdout_high_T(tab, hi_methods, args.hide_top, args.max_x)
    j0 = tab.nT - args.hide_top
    _print_table(f'=== high-T hold-out: top {args.hide_top} rows hidden '
                 f'({tab.T[j0]:.3g} - {tab.T[-1]:.3g} K), extrapolated from '
                 f'{tab.T[j0 - 1]:.3g} K at fixed rho (on-table points only) ===', res, sets)
    figdata['high'] = perG

    in_methods = OrderedDict([('constant n_e T^-1.5', base)])
    res_all, res_edge = holdout_interp(tab, in_methods, args.max_x)
    _print_table('=== interpolation hold-out: every other T row dropped and '
                 're-interpolated (all on-table points) ===', res_all, sets)
    _print_table('    ... only points within 1 dex of the density edge', res_edge, sets)

    if args.plot:
        plot_holdout(args.plot, tab, figdata, args.cut)
        print(f'\nwrote {args.plot}')


def plot_holdout(path, tab, figdata, cuts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(cuts) + 1
    fig, ax = plt.subplots(1, n, figsize=(5.5 * n, 4.5), constrained_layout=True)
    for k, cut in enumerate(cuts):
        for name, v in figdata[f'low{cut}'].items():
            ax[k].plot(tab.T, v, label=name, marker='.', lw=1)
        ax[k].set_xscale('log')
        ax[k].set_yscale('log')
        ax[k].set_xlabel('T [K]')
        ax[k].set_ylabel(r'90th pct $|\Delta\log_{10}\kappa_P|$ over groups')
        ax[k].set_title(f'low-density hold-out, {cut:g} dex hidden')
        ax[k].legend(fontsize=8)
    for name, v in figdata['high'].items():
        ax[-1].plot(tab.E_c, v, label=name, marker='.', lw=1)
    ax[-1].set_xscale('log')
    ax[-1].set_yscale('log')
    ax[-1].set_xlabel('group centre [keV]')
    ax[-1].set_ylabel(r'median $|\Delta\log_{10}\kappa_P|$')
    ax[-1].set_title('high-T hold-out')
    ax[-1].legend(fontsize=8)
    fig.savefig(path, dpi=130)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def add_method_args(p):
    g = p.add_argument_group('method options')
    g.add_argument('--fit-decades', type=float, default=0.35,
                   help='fit the low-density power-law slope over this many dex '
                        'above the edge (default: %(default)s)')
    g.add_argument('--alpha-clip', type=float, nargs=2, default=[0.0, 1.2],
                   metavar=('LO', 'HI'),
                   help='clip the fitted slope d log kappa_abs / d log rho '
                        '(default: %(default)s)')
    g.add_argument('--alpha-smooth', type=int, default=1, metavar='NROWS',
                   help='running median of the slope over this many T rows; '
                        '1 = off (default: %(default)s)')
    g.add_argument('--alpha-relax', type=float, default=0.0, metavar='DEX',
                   help='below the edge, relax the slope from its fitted value '
                        'to 1 (kappa_abs ~ rho, all absorbers minority ions) '
                        'over this many dex; 0 = keep the fitted slope '
                        '(default: %(default)s)')
    g.add_argument('--highT', choices=['tslope', 'ff'], default='tslope',
                   help='extrapolation above T_max: continue each field\'s measured '
                        'slope in T over the top rows, or the free-free shape per '
                        'group (default: %(default)s)')
    g.add_argument('--tslope-rows', type=int, default=3,
                   help='rows used for --highT tslope (default: %(default)s)')


def main(argv=None):
    p = argparse.ArgumentParser(
        description='Resample a raw LANL TOPS multigroup opacity table onto a '
                    'uniform log T / log rho grid, with extrapolation off the table.',
        epilog='Run "%(prog)s <command> -h" for the options of each command.')
    sub = p.add_subparsers(dest='command', metavar='command')
    sub.required = True

    pi = sub.add_parser('info', help='describe a raw TOPS file and its table edge')
    pi.add_argument('raw', help='raw TOPS results page (.txt)')
    pi.add_argument('-v', '--verbose', action='store_true', help='print the edge for every T row')
    pi.set_defaults(func=cmd_info)

    pr = sub.add_parser('regrid', help='write a table uniform in log T and log rho',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pr.add_argument('raw', help='raw TOPS results page (.txt)')
    pr.add_argument('-o', '--output', required=True, help='output table file')
    pr.add_argument('--rho-range', type=float, nargs=2, default=[1e-16, 1e-6],
                    metavar=('MIN', 'MAX'), help='density range [g/cm^3]')
    pr.add_argument('--n-rho', type=int, default=100, help='number of densities')
    pr.add_argument('--T-range', type=float, nargs=2, default=None, metavar=('MIN', 'MAX'),
                    help='temperature range [K]; None = the TOPS table range')
    pr.add_argument('--n-T', type=int, default=69, help='number of temperatures')
    pr.add_argument('--plot', metavar='PNG', help='save a diagnostic figure')
    pr.add_argument('--nfree', metavar='TXT',
                    help='where to write N_free (free electrons per atom) on the output '
                         'grid; default <output>_nfree.txt')
    pr.add_argument('--no-nfree', action='store_true', help='do not write the N_free file')
    pr.add_argument('--rossabs', action='store_true',
                    help='append the Rosseland absorption (Rosseland means less the '
                         'scattering model) as further per-group blocks')
    pr.add_argument('--digits', type=int, default=4,
                    help='significant digits after the first in the table (default 4)')
    pr.add_argument('--twoband', nargs='?', const='', default=None, metavar='TXT',
                    help='also write a two-band (picket-fence) split of every group that '
                         'reproduces both the Planck and Rosseland group means; '
                         'default path <output>_twoband.txt')
    pr.add_argument('--twoband-closure', choices=['w', 'gap0'], default='w',
                    help='w: fix the high-band fraction (--twoband-w), falling back to '
                         'gap0 where needed; gap0: low band has no absorption')
    pr.add_argument('--twoband-w', type=float, default=0.01,
                    help='high-band frequency fraction for the w closure')
    pr.add_argument('--twoband-floor', type=float, default=0.003,
                    help='Rosseland absorption below this fraction of the scattering is '
                         'raised to it (below it the scattering model cannot resolve it)')
    pr.add_argument('--branches', metavar='TXT',
                    help='save the per-cell branch flags (which rule produced each value)')
    add_method_args(pr)
    pr.set_defaults(func=cmd_regrid)

    ph = sub.add_parser('holdout', help='test the extrapolation by hiding part of the table',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ph.add_argument('raw', help='raw TOPS results page (.txt)')
    ph.add_argument('--cut', type=float, nargs='+', default=[1.0],
                    help='dex above the edge to hide in the low-density test')
    ph.add_argument('--hide-top', type=int, default=6,
                    help='number of top T rows to hide in the high-T test')
    ph.add_argument('--max-x', type=float, default=20.0,
                    help='leave groups with h nu > MAX_X kT out of the error statistics')
    ph.add_argument('--plot', metavar='PNG', help='save a summary figure')
    add_method_args(ph)
    ph.set_defaults(func=cmd_holdout)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
