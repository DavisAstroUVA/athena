# Kompaneets Green's function tables: format and sampling algorithm

This file describes the tables written by `kompaneets_greens.py` and the
algorithm a Monte Carlo code needs to sample from them. The reference
implementation is `kompaneets_greens.Sampler`. Any port should reproduce its
output to float32 round-off; see *Verifying a port* at the end.

## 1. What the table represents

A photon of initial energy `xi = h nu / kT` spends a Compton parameter `y`
in a uniform, isothermal region. Here `y = theta * tau_path`, where
`theta = kT / m_e c^2` and `tau_path` is the Thomson optical path length
travelled. The photon may also be absorbed by an energy-dependent opacity

    kappa_abs / kappa_es = theta * lam * a(x)

with `a(x)` the free-free shape (Gaunt factor 1, stimulated emission included,
normalised to a(1) = 1):

    a(x) = (1 - exp(-x)) / x^3 / (1 - exp(-1))

The absorption parameter is

    lam = kappa_ff(x=1) / (kappa_es * theta)

which is `kompaneets_greens.lam_freefree(rho, T)`. It is the same quantity as
`tfac` in the old `compton_greens.py`.

For each `(lam, y, xi)` the table gives two things:

1. **The survival fraction** `S(lam, y, xi)`: the probability that the photon
   has not been absorbed.
2. **The energy distribution of the survivors**, stored as quantiles of
   `ln(xf)`.

The random-walk (MRW) step uses them as follows. After sampling the escape
path length from the sphere, it sets `y = theta * tau_path`, draws `xf` from
the quantiles, and multiplies the photon weight by `S`. You can use Russian
roulette with probability `S` instead of the weight.

## 2. Files

| file | contents |
|---|---|
| `kgreens_table.npz` | no absorption: a single `lam = 0` slice (4.6 MB) |
| `kgreens_table_ff.npz` | free-free absorption: 28 `lam` slices (129 MB) |

Both are NumPy `.npz` archives, i.e. zip files of `.npy` arrays. All arrays
are little-endian and C-ordered (last index fastest). To regenerate them:

    python kompaneets_greens.py --out kgreens_table.npz
    python kompaneets_greens.py --absorption --nproc 8 --out kgreens_table_ff.npz

Options: `--lamlim LMIN LMAX`, `--nlam`, `--ny`, `--nxi`, `--nq`.

## 3. Arrays

Dimensions in the current files: `NL = 28` (or 1), `NY = 73`, `NX = 61`,
`NQ = 257`.

| name | dtype | shape | meaning |
|---|---|---|---|
| `lam` | float64 | (NL) | `lam[0] = 0`; `lam[1:]` log-spaced, 1e-10 ... 1e3 (half-decade steps) |
| `y` | float64 | (NY) | log-spaced, 1e-6 ... 15 (10 per decade) |
| `xi` | float64 | (NX) | log-spaced, 1e-3 ... 60 (~12.6 per decade) |
| `levels` | float64 | (NQ) | cumulative probabilities p_k, increasing, `levels[0] = 0`, `levels[NQ-1] = 1` |
| `uq` | float32 | (NL, NY, NX, NQ) | scaled quantiles `u = ln(xf/xi) / sqrt(2 y)` at each level |
| `abar` | float64 | (NL, NY, NX) | mean absorption `abar = -ln S / (lam y)`; zero in the `lam = 0` slice |
| `dead` | bool | (NL, NY, NX) | true where `S < 1e-12` (fully absorbed; see below) |
| `ys_coef` | float64 | () | generation parameter (bookkeeping only) |
| `sink` | str | () | name of the absorption shape (`freefree_shape`) |

Notes:

- **Grids:** all interpolation is done in `ln lam` (for lam > 0), `ln y` and
  `ln xi`. Every grid is exactly log-spaced, so a port may compute the index
  arithmetically instead of by binary search. Reading the stored arrays is
  safer.
- **Levels** are uniform in logit p between `p_min = 1e-7` and `1 - p_min`:

      z_k = zlo + (-2 zlo) * k/(NQ-1),  zlo = ln(p_min/(1-p_min)),  p_k = 1/(1+exp(-z_k))

  The first and last entries are then overwritten with exactly 0 and 1. The
  quantiles at levels 0 and 1 are the edges of the numerically resolved
  support.
- **Dead cells:** `uq` holds the previous-y shape rescaled, which is
  harmless. `abar` there is computed from `max(ln S, -700)` and should not be
  trusted except to say "S is negligible".
- **Why `S` is stored as `abar`:** `abar` varies slowly with `lam` and
  interpolates well, whereas `S` itself can change by many decades between
  adjacent `lam` values.

## 4. Sampling algorithm

Inputs: `xi`, `y`, `lam >= 0`, and a uniform random number `r` in [0, 1).
Outputs: `xf` and `S`.

### 4.1 Helpers

`axis_weights(grid, v)`: clamp `v` to `[grid[0], grid[N-1]]`, find `i` with
`grid[i] <= v <= grid[i+1]` and `0 <= i <= N-2`, and return `i` and
`t = (v - grid[i]) / (grid[i+1] - grid[i])`. Use this for linear
interpolation; it is always applied to the log of the grid.

`cubic_weights(grid, v)`: first `(i, t) = axis_weights(grid, v)`. If
`i == 0` or `i >= N-2` (first or last interval), return the two linear
points `{i: 1-t, i+1: t}`. Otherwise return 4-point Lagrange weights on nodes
`i-1, i, i+1, i+2`:

    w_k = prod_{m != k} (v - g_m) / (g_k - g_m)

`lam_slices(lam)` returns the slices and weights used for the energy
quantiles:

- `lam <= 0`, or a table with NL = 1: `{0: 1}`
- `0 < lam <= lam[1]`: linear in `lam` (not in `ln lam`),
  `b = lam/lam[1]`, giving `{0: 1-b, 1: b}`
- `lam > lam[1]`: `(k, b) = axis_weights(ln lam[1:], ln lam)`, giving
  `{k+1: 1-b, k+2: b}` (clamped at `lam[NL-1]`)

### 4.2 Final energy `xf`

**Case A, `y < y[0]` (below the table):** use the small-y Gaussian limit in
`ln x` (drift 3 - x, variance 2y):

    xf = xi * exp( (3 - xi) y + sqrt(2 y) * Phi^{-1}(r) )

`Phi^{-1}` is the inverse standard normal CDF. The reference code evaluates it
as `sqrt(2) erfinv(2r - 1)`.

**Case B, `y >= y[0]`:** let `ye = min(y, y[NY-1])`. Above the table the
shape no longer depends on y: it is Wien without absorption, or the
slowest-decaying absorbed mode with it. So the `y_max` slice is reused, and
`ye` also goes into the final rescaling.

1. `(iy, a) = axis_weights(ln y, ln ye)` and
   `(ix, b) = axis_weights(ln xi_grid, ln xi)`.
2. Find the level interval: `k` with `levels[k] <= r <= levels[k+1]`, and
   `c = (r - levels[k]) / (levels[k+1] - levels[k])`. Use a binary search
   on `levels`.
3. For each slice `(l, w_l)` in `lam_slices(lam)`, and for each of the four
   `(y, xi)` corners with bilinear weights `(1-a)(1-b)`, `a(1-b)`, `(1-a)b`,
   `ab`, accumulate:

       u += w_l * w_corner * ((1-c) * uq[l, jy, jx, k] + c * uq[l, jy, jx, k+1])

   Interpolation is linear in every direction, so interpolating only the two
   bracketing levels gives the same result as the reference code, to float32
   round-off (checked: relative difference 3e-9).
   It interpolates the whole 257-vector and then the level. This costs at
   most 2 x 4 x 2 = 16 table reads.
4. `xf = xi * exp(u * sqrt(2 ye))`

`xi` outside `[1e-3, 60]` is clamped for the table lookup, but the true `xi`
is used in `xf = xi * exp(...)`. Callers should keep `xi` inside the table
range; see *Validity* below.

### 4.3 Survival `S`

If `lam <= 0` or the table has NL = 1: `S = 1`.

**Case A, `y < y[0]`:** `S = exp(-lam * a(xi) * y)`.

**Case B, `y[0] <= y <= y[NY-1]`:** `S = exp(-lam * y * abar_interp(lam, y, xi))`,
where `abar_interp` works as follows:

1. `ll = ln(max(lam, lam[1]))`. Below the smallest positive table `lam`,
   `abar` is held at its `lam[1]` value; there `S` is ~1 anyway.
2. The `lam` axis is `ln lam[1:]`. Table slice `l` corresponds to axis index
   `l - 1`.
   - `(IL, WL) = cubic_weights(ln lam[1:], ll)`
   - `(IX, WX) = cubic_weights(ln xi_grid, ln xi)`
   - `(iy, a) = axis_weights(ln y, ln y_in)`
3. **Dead-cell fallback:** if any `dead[i+1, jy, jx]` is true, for `i` in IL,
   `jy` in `{iy, iy+1}` and `jx` in IX, replace both the `lam` and `xi`
   stencils by the linear 2-point ones from `axis_weights`. Near fully
   absorbed cells `ln abar` is not smooth, and a cubic stencil would
   overshoot.
4. Interpolate in log space, linearly in `ln y`:

       L = sum_{i in IL} sum_{jx in IX} WL_i * WX_jx *
           ( (1-a) * ln(max(abar[i+1, iy,   jx], 1e-300))
           +    a  * ln(max(abar[i+1, iy+1, jx], 1e-300)) )
       abar_interp = exp(L)

**Case C, `y > y[NY-1]`:** `ln S` becomes linear in `y` (a single decaying
mode), so extrapolate from the last two table points:

    l1 = -lam * y[NY-2] * abar_interp(lam, y[NY-2], xi)
    l2 = -lam * y[NY-1] * abar_interp(lam, y[NY-1], xi)
    ln S = l2 + (l2 - l1) / (y[NY-1] - y[NY-2]) * (y - y[NY-1])

### 4.4 Use in the random-walk step

    y   = theta * tau_path          # tau_path: Thomson path length of the sphere crossing
    lam = lam_freefree(rho, T)      # per zone; constant within a uniform sphere
    xf  = sample_xf(xi, y, lam, r)
    S   = survival(xi, y, lam)
    weight *= S                     # or: absorb with probability 1 - S

In the old `compton_greens.get_nx_spec` convention, where `t` is the
dimensionless escape time of the sphere,
`y = t * 3 tau^2 theta / pi^2` for a point source at the centre.

## 5. Accuracy (current tables)

Against direct solutions at 200 random off-grid `(lam, y, xi)`
(`python check_table.py kgreens_table_ff.npz --npts 200`):

- **Survival:** the median relative error is 4e-11 and the 90th percentile
  is 5e-5. Errors of 1.5–5% occur only where `S < 1e-3`; even there the error
  in `ln S` is below 0.5%.
- **Energy quantiles** (1%, 50%, 99%): errors are below 0.7% of the
  1%–99% width of the distribution.
- **Solver vs Becker (2003) exact solution:** agreement is within 2.5e-5 of
  the peak value.
- **Solver vs direct Compton Monte Carlo** (`mc_sphere_check.py`): survival
  agrees to 0.1–0.3%, and escaping spectra to under 1% in CDF.

## 6. Validity

- **Kompaneets regime:** `theta <~ 0.03` and `xf * theta <~ 0.05`
  (Klein–Nishina and relativistic corrections are O(theta) and are
  neglected). The table's upper `xi` = 60 is the ceiling. At high
  temperature the physical ceiling `0.05/theta` is lower, and the caller
  must enforce it.
- **Separation of energy and space:** this holds only for a uniform sphere
  with `kappa_abs << kappa_es`, i.e. `theta * lam * a(x) << 1` at the
  energies that matter. When it fails, don't use the random walk.
- **Photons that escape almost immediately:** photons leaving after a path of
  only 1–4 sphere radii are biased. Their spectra differ from the table by
  2–3% in CDF at `tau = 15`, and by ~1% at `tau = 40`. Prefer sphere optical
  depths `>~ 20–30`, or use ordinary Monte Carlo for the first few optical
  depths.
- **Outside the table:** `xi` is clamped at 1e-3 and 60. `lam` above 1e3 is
  clamped; that is extrapolation, and such photons are essentially always
  absorbed. `y` beyond 15 is handled analytically (Case C).

## 7. Verifying a port

Compare against the Python reference on a set of `(xi, y, lam, r)` values:

```python
import numpy as np
import kompaneets_greens as kg
smp = kg.Sampler("kgreens_table_ff.npz")
for xi, y, lam, r in [(0.03, 0.2, 2e-3, 0.37), (1.0, 3.0, 0.0, 0.9),
                      (5.0, 1e-7, 1.0, 0.5), (0.3, 40.0, 0.1, 0.01)]:
    print(xi, y, lam, r, smp.quantile(xi, y, np.array([r]), lam)[0],
          smp.survival(xi, y, lam))
```

The cases cover Cases A–C and both `lam` regimes. Also exercise a point next
to dead cells (e.g. `lam = 100, xi = 0.01`) and `0 < lam < lam[1]`.
