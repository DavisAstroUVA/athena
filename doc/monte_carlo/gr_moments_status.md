# GR radiation moments: current state and options

Status: **deferred deliberately.** Everything else that used to be evaluated at the cell center
has been moved to the photon; the moment *tensors* in a curved metric are the one place that
still is not. This note records why, how big it is, and what was already tried and rejected.

Related: `doc/monte_carlo/velocity_reconstruction_plan.md` (the four-velocity work this came out
of), and the comment at `MonteCarloBlock::UpdateMoments`.

---

## 1. What the code does now

`ComputeTransformations` builds `boost_lab` and `boost_cmv` once per cell, at the cell center,
from a four-velocity reconstructed there. `PhotonFrames::Fill`'s `gr_tetrad_` branch then applies
those matrices to the photon's coordinate four-vector, which is carried at the *photon*:

```
p^(a) = e^(a)_mu(x_center) k^mu(x_photon)
```

That is neither a local measurement (the tetrad at the photon) nor a transported one (carry k to
the center first). Because the tetrad is orthonormal with respect to the metric at the center
while k is null with respect to the metric at the photon, the projection is not null, and the code
divides the spatial part by its own magnitude to keep `n` a unit vector and the moment tensor
traceless. That fixes the magnitude, not the direction.

## 2. How big the error is

Measured on the Kerr-Schild deck (a = 0.9375, r in [2,20], 64 x 16 x 8, so dphi = 0.785):

| quantity | value |
|---|---|
| per-cell \|dF\|/\|F\| from applying a transport correction | median **0.13**, p90 0.31, max 0.42 |
| Er change | 0.33% |

The size is set by the **curvilinear** part of the connection, not by gravity. In Kerr-Schild
spherical, Gamma contains terms like `Gamma^r_thth = -r` and `Gamma^phi_thphi = cot(theta)`, which
are O(1/r); the genuinely gravitational terms are O(M/r^2) and contribute only ~3% at r = 3. So
this is largely the same effect the flat curvilinear code already handles exactly -- it is the
angular width of a cell, and it does not average away.

**It scales with the angular grid.** The deck above is a hard case. On a grid with dphi ~ 0.1 the
effect is roughly 8x smaller, and a first-order treatment would probably be adequate away from the
axis. Before investing in any option below, check the phi resolution of the runs that matter.

## 3. What was tried and rejected

First-order parallel transport of k to the cell center,

```
k^mu -> k^mu - Gamma^mu_(nu rho) k^nu dx^rho = k^mu - acon^mu_rho dx^rho, dx = x_center - x_photon
```

reusing the `acon` that `GeneralPusher::AdvanceStep` already caches. It was implemented, built and
run, then reverted.

Parallel transport preserves `k.k = 0` exactly, so the null violation after transport *is* the
truncation error. Over 30M samples:

```
at the photon      |k.k|/(k0)^2  max 7.3e-16      geodesic is exactly null
after transport    1e-3 .. 1e-2 : 12.8M samples
                   1e-2 .. 1e-1 : 17.2M samples
                   > 1          : ~3700 samples   (polar cells, max 1.1e5)
```

Two independent reasons it fails:

- **The expansion parameter is not small.** With `dphi/2 = 0.39` rad the transport truncates a
  0.39-radian basis rotation at first order. The error is ~`alpha^2/2 = 0.076`, larger than the
  ~0.03 gravitational correction it was meant to capture.
- **It diverges at the axis.** `cot(theta)` blows up in the polar cells regardless of resolution.

Layering the existing renormalization on top of a 10%-level nullity violation would muddle the
frame rather than fix it.

Cost, measured on `mc_snake_atm`, for reference if it is revisited:

| case | overhead |
|---|---|
| polarized, reusing the cached `acon` | +0.9% |
| unpolarized, computing `acon` per step | +11.5% |

The asymmetry is structural: `AdvanceStep`'s corrector evaluates the connection *after* `RK4Step`,
so a polarized run already has it at the right point; an unpolarized run pays for `Connect` plus a
dense 4x4x4 contraction. Roughly 3/4 of that cost is the contraction, not the metric evaluation,
so sharing work with `InverseMetricDerivative` was checked and would save under 3% of the
correction -- not worth it. Gamma is 1/64 sparse in snake and fully dense in Kerr-Schild, so a
`ConnectContract` that skips materializing the array would help snake and do nothing for Kerr.

## 4. Options

1. **Leave it.** The behavior is bounded: the renormalization keeps `|n| = 1` and the moment
   tensor traceless. Adequate whenever GR moments are diagnostic rather than a science product.
   This is the current choice.
2. **Split the connection** -- exact rotation for the curvilinear part, first-order Gamma for the
   gravitational remainder. Principled, and it removes the axis divergence because the divergent
   terms are exactly the ones then handled exactly. Cost is coordinate-specific work in each
   `MCCoord`.
3. **Build the tetrad at the photon.** Exact per photon, `k` stays null, nothing diverges, no
   connection needed. The plumbing is easy -- `PhotonFrames` already localizes frame construction,
   so it is ~20 lines. The cost is a Gram-Schmidt per photon per step, ~300 flops and 4 square
   roots, about 3x the transport that was rejected. Photons in a cell are then measured in
   slightly different frames, an O(cell) inconsistency of the same order as the binning itself.
4. **Sub-step the transport.** Not recommended: it pays more to converge a scheme that still
   diverges at the axis.
5. **Refine the angular grid**, if the science allows. Cheapest mitigation, no code change.

Options 2 and 3 are the real candidates. 3 is the safer one (exact, no divergence); 2 is the more
accurate one if the goal is a genuine per-cell tensor.

## 5. Already done, and not affected by any of this

- **Flat spacetime** (spherical polar, cylindrical, cartesian without `-g`) is handled **exactly**
  by `PhotonFrames::ToCellCenterBasis`. In flat space the Cartesian components of a
  parallel-transported vector are constant, so converting at the photon and back at the cell
  center is parallel transport to all orders -- no connection, no expansion.
- **The opacity frequency shift** is evaluated at the photon, via `ObserverEnergy`.
- **The scattering mean intensity** (`mom_flag_scat`) takes its comoving energy at the photon
  rather than through the cell-center tetrad. This matters disproportionately because `e_scat`
  selects the frequency bin, so an error there misfiles the contribution instead of averaging away.

So what remains is only `mclab`, `mccom` and `mccoord` in a curved metric.

## 6. Notes for whoever picks this up

Hard-won, and easy to lose:

- **HDF5 builds here**, despite `h5cc -show` advertising `/usr/include/hdf5/serial`, which does not
  exist. The openmpi variant is installed:
  ```
  python configure.py ... -mpi -hdf5 \
      --include=/usr/include/hdf5/openmpi \
      --lib_path=/usr/lib/x86_64-linux-gnu/hdf5/openmpi
  ```
  Without a moment output nothing here is observable at all, so this is the prerequisite.
- **`cmp` on a `.athdf` file is not a valid comparison** -- the metadata differs between runs.
  Compare the datasets.
- **Know the noise floor before believing a difference.** Across seeds at fixed settings,
  `nscat/ntot` spreads by 14.8% and `sum(Ermc)` by 4.15%. A 3% "discrepancy" between the legacy
  and general pushers was chased at length and turned out to be one seed's fluctuation; over four
  seeds they agree to 0.5%. Use four seeds for anything compared across runs. `nscat` is useless
  as a diagnostic at these photon counts.
- **Before/after on the same pusher is noise-free**, because the moment frame does not feed back
  into transport -- the two runs have bit-identical photon paths and counters. This is the
  strongest test available and should be preferred over any cross-pusher comparison.
- **`|k.k|/(k0)^2` is the sharp diagnostic** for any transport scheme, since exact transport
  preserves it.
- **`call_moments` must be true** or `PhotonFrames` is never constructed and nothing is
  measurable. It is true in every current problem generator only because they all call
  `AllocateUserMoments`.
