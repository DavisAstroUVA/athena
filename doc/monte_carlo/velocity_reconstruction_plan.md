# Reconstructing the fluid four-velocity at the photon

Status: **complete.** All five phases done. The GR four-velocity is rebuilt at the point it is
used rather than stored as a cell-center quantity.

### Phase 5 -- moments guard and the deliberate split. **Done.**

The transport frames and the moment frames are now built at different points on purpose, so the
test measures both. `mc_gr_simple` writes a third user slot, `IUUCEN`, holding the running maximum
of `|u.u + 1|` evaluated at the photon's *cell center* -- the frame `ComputeTransformations` and
`ComovingFrameMatrix` build `boost_lab` and `boost_cmv` on -- beside `IUUDEV` at the photon itself.
`kerr_frames` reports the two separately and passes only when both are at roundoff:

```
  nx1   64  photon-frame |u.u+1| 1.1102e-15   cell-center 8.8818e-16
  nx1  128  photon-frame |u.u+1| 1.1102e-15   cell-center 6.6613e-16
  nx1  256  photon-frame |u.u+1| 1.3323e-15   cell-center 8.8818e-16
PASS: both frames are unit timelike where they are used
```

The split is documented at `UpdateMoments`: the moments are cell averages, so one frame per cell
is what makes them a well defined tensor and what lets `DeriveComovingMoments` transform the
accumulated sum once rather than projecting each photon twice. The price is that a photon's
comoving frame during transport and the frame its contribution is booked into differ at
`O(dx dg)` -- the same order as the cell averaging itself.

**What could not be checked here.** The plan asked for the moment *arrays* to be compared byte for
byte. That needs the HDF5 `mcmom`/`mccom` output, and this machine has the HDF5 runtime libraries
but not the headers (`h5cc -show` advertises `/usr/include/hdf5/serial`, which does not exist), so
`-hdf5` will not build. What was verified instead is stronger for the specific risk: the change
from reading `vel` to calling `FluidFourVelocity` at the cell center was confirmed **bitwise**
identical (`rel 0.000e+00`, 17 digits, every cell) with a temporary comparison inside
`ComputeTransformations`, so the matrices feeding the moments are unchanged. Run the array
comparison when HDF5 headers are available.

### Opacity refreshed every step. **Done.**

`GeneralPusher::Move` now calls `UpdateOpacities` at the top of each step instead of only when the
photon crosses a cell face. It sits above the step because `tau_step` multiplies `chi` by the
length of the step about to be taken, and it subsumes the refresh that used to hang off
`UpdateZone`.

It is skipped inside a cell when `MonteCarloBlock::shift_unity` holds -- a flat metric (so the
lapse is one) and a fluid at rest (so there is no Doppler term). The comoving shift is then
identically one, the opacity depends on the cell alone, and refreshing within a cell is provably a
no-op. The first step of every `Move` still refreshes regardless, which is the part that actually
corrects results.

The right axis for that test is **not** GR versus non-GR. The shift varies within a cell exactly
when the fluid has a velocity component along a direction the metric depends on, because `u^nu` is
constant per cell while `k_nu` is conserved only for the coordinates the metric ignores. So
spherical polar with any radial flow varies, and so does any curved metric even with a static
fluid; snake does not, and cannot be made to with `mc_snake`, whose metric depends on x1 alone
while its `velocity` drives flow along x3.

Cost on `mc_snake_atm`: unconditional refresh is **12% of wall clock** (15.2 s -> 17.1 s), 8.6% by
instruction count -- the gap is cache effects callgrind does not see. It was 28% before
`FrequencyShiftComoving` stopped building a full tetrad to extract one component. With the
`shift_unity` skip that deck returns to **14.8 s** while producing `.spec` and `.list` **byte
identical** to the unconditional version.

What the measurement actually showed, which is not what was expected:

- **Within a cell the refresh is exactly a no-op in snake** -- the relative change in `chi` was
  `0.000e+00` over 1.2e7 same-cell steps. That is correct rather than disappointing: snake is a
  static metric with a static fluid, so the comoving shift is identically one. It confirms the
  refresh introduces no spurious variation.
- **The result change comes from the first step of each `Move` call**, 2.5e5 occurrences moving
  `chi` by up to 28%. The opacities carried *into* `Move` were stale, and refreshing at the top
  corrects them. `mc_snake_atm` shifts from 1038 to 1047 escaped photons and 209.6 to 199.9
  scatters per photon.
- **The within-cell benefit is argued, not measured.** It should appear in a curved metric with a
  moving fluid, where the shift varies along the geodesic through the lapse and through aberration
  as `k` turns. No deck in `tst/montecarlo` currently exercises that: the Kerr deck has
  `scattering = none` so `chi` is identically zero, and turning scattering on absorbs every photon
  at the inner boundary before anything accumulates. **A deck that would demonstrate it is still
  missing** and would be the natural next test to write.

**Still deferred, by choice:** the cell-center tetrad applied to a wavevector carried at the photon
without parallel transport (the cached `acon` correction).

### Phase 4 -- retire the GR four-velocity. **Done.**

There is no longer a stored four-velocity in GR. `GetVelocity` writes only `uprim` and touches no
metric; `SetNormalObserver` just zeroes it, since zero primitives *are* the normal observer under
`FluidFourVelocity`. `vel` is no longer allocated in GR at all, and its declaration now says
plainly that it is the flat-spacetime `(gamma, gamma*beta^i)` and is not interchangeable with a
four-velocity -- the two meanings sharing one name is what made this hard to see.

The two remaining consumers that legitimately want a *cell-center* four-velocity --
`ComputeTransformations` and `ComovingFrameMatrix`, which build the per-cell moment frames -- call
`FluidFourVelocity` at the cell center instead of reading `vel`. That was the one real risk in this
phase, because the old path took the metric from Athena's `CellMetric` while `FluidFourVelocity`
uses `MCCoord::Metric`/`InverseMetric`. Checked directly with a temporary comparison in
`ComputeTransformations`: the two agree **bitwise** (`rel 0.000e+00`, identical to 17 digits) at
every cell of a Kerr-Schild run, so the moment frames are unchanged.

`ObserverEnergy`'s `sqrt(|u.u|)` division is kept for generality but is now a no-op for the
module's own callers; its comment says so rather than quoting the old error figures.

Also fixed in passing: `vel` was allocated under `boosts || GR || polarized` but freed under
`boosts || GR`, leaking on flat polarized runs. Both are now unconditional
`DeleteAthenaArray()` calls, which handle the never-allocated case.

Gates: `mc_snake_atm` byte-identical; `poltest` 4/4; `snake_polarization` order 2.00;
`kerr_frames` 1.11e-15; the acceleration deck unchanged; cpplint at baseline on all five files.

---

## 1. The problem

`MonteCarloBlock::vel` stores a contravariant four-velocity built at the **cell center**, normalized
there with `CellMetric` (`montecarloblock.cpp`, the fluid branch and the normal-observer branch).
Every GR consumer then evaluates it against the metric at the **photon's** position. So `u.u = -1`
holds where the vector was built but not where it is used, and the shortfall is first order in the
offset. Measured on a Kerr-Schild grid (a = 0.9375, 128 logarithmic radial cells from 1.2M to 100M,
64 in theta, photon half a cell off center):

```
  r/M     u.u at photon      |sqrt(|u.u|) - 1|
   2.0   -1.006291831781     3.14e-03
   4.0   -1.005277813416     2.64e-03
   8.0   -1.003337918198     1.67e-03
  20.0   -1.001539608294     7.70e-04
  50.0   -1.000652910319     3.26e-04
```

That shortfall is currently absorbed by a `sqrt(|u.u|)` division inside `ConstructTetrad` and
`ObserverEnergy`. The division corrects the observer's *magnitude* but not its *direction*, so it is
half a fix, and it converts a fluid-sampling question into a violated normalization that contaminates
every frequency shift.

## 2. What the imaging codes do

blacklight and ipole/grmonty never store or interpolate a four-velocity. They carry the **primitive**
velocity -- a spatial three-vector with no normalization constraint -- and rebuild `u^mu` at the
geodesic point with the metric at that point.

blacklight, `simulation_coefficients.cpp`:

```cpp
CovariantSimulationMetric(x1, x2, x3, gcov_sim);      // at the SAMPLE POINT
double uu0_sim = std::sqrt(1.0 + gcov_sim[1][1]*uu1_sim*uu1_sim + ... );
ucon_sim[0] = uu0_sim / lapse_sim;
ucon_sim[1] = uu1_sim - shift1_sim * uu0_sim / lapse_sim;
```

ipole `model.c` and grmonty `harm_model.c`, identical algebra with `Vfac = alpha*gamma`:

```c
Vcon[1] = interp_scalar(p[U1], ...);                   // PRIMITIVES
gcon_func(X, gcon);                                    // at the PHOTON
Vfac = sqrt(-1./gcon[0][0] * (1. + fabs(VdotV)));
Ucon[0] = -Vfac * gcon[0][0];
Ucon[i] =  Vcon[i] - Vfac * gcon[0][i];
```

This is the same algebra our own `vel` setup already uses -- the difference is only *where* it is
evaluated. `u.u = -1` is then exact by construction, and neither code contains any renormalization.
blacklight's `simulation_interp = false` mode does this with piecewise-constant primitives, which is
the configuration adopted here.

## 3. Scope: only the GR regime

| regime | `vel` holds | normalization | change |
|---|---|---|---|
| flat + legacy pusher | `(gamma, gamma*beta^i)`, orthonormal | metric constant, exact everywhere | none |
| flat curvilinear + general pusher (`tetrads`) | same; coordinate step is `MCCoord::InverseTetrad` | exact in the orthonormal sense | none |
| GR + general pusher | contravariant `u^mu` normalized at cell center | fails at O(dx dg) | **yes** |

`vel` cannot simply be dropped for flat spacetimes. Four non-GR sites read it directly, and the flat
transform path is one of them, not a cache: the shared flat boost helper behind
`TransformToComoving`/`TransformToCoordinate`, `LorentzTransformFrequencyShift`,
`UpdateMomentsAcceleration`, and `MRWAcceleration`. `boost_cmv`/`boost_lab` are *derived* from `vel`
in `ComputeTransformations` and are consumed only by `PhotonFrames` and the moments.

## 4. Decisions

- Nearest-cell primitives, no interpolation (blacklight's `simulation_interp = false`).
- `vel`'s GR contents become the MC-owned primitives; `MonteCarloBlock` keeps owning its velocity data.
- `UpdateMoments` stays on the cell-center frames. Out of scope, with a guard test.

## 5. Phases

### Phase 0 -- MRW bugs and this document. **Done.**

Two bugs, both in `PhotonPusher`, both found while building the gate:

1. `MRWAcceleration` read `pmcb->vel(m, i3, i2, i1)`. `vel` is allocated
   `(ncells3, ncells2, ncells1, 4)`, so the correct order is `vel(i3,i2,i1,m)`; the transposed form
   indexes another cell's memory and runs off the end of the array once `i1` reaches 4. It also
   omitted the `/vel(...,0)` division that every other site uses to turn `gamma*beta` into `beta`.
2. `ReadTimeDistribution` sat behind `if (time_acc)`, but the tables it loads back
   `InterpPathTime`, which `MRWAcceleration` calls unconditionally. With `time_acc` at its default
   an acceleration run read three unallocated arrays and hung. The destructor already frees them
   under `acceleration` alone, which is the evidence of intent.

Gate, `mc_isoth` with `boosts = true`, `velocity = 0.2`, `scattering = thomson`, `taumax = 1e4`,
20000 photons, against the same deck with `acceleration = false`:

| run | nesc | nabs | ndes | nscat/ntot |
|---|---|---|---|---|
| reference, no acceleration | 11277 | 8723 | 0 | 101.6 |
| acceleration, before fix | 11080 | 7768 | **1152** | 43.7 |
| acceleration, after fix | 11290 | 8709 | **1** | 68.5 |

Destroyed photons 1152 -> 1; absorbed from 11% below reference to 0.2%; escaped from 1.7% below to
0.1% above. The reduced scatter count is MRW working as intended. Note the escaping *spectrum* does
not discriminate at this photon count -- the seed-to-seed spread of total flux is 14.9% -- so the
counters are the evidence, not the flux.

Running MRW needs `time_table_tau.out`, `time_table_p.out` and `time_table_t.out` in the working
directory.

### Phase 1 -- Build the test that can see the effect. **Done.**

No existing test can. Snake has `g_tt = -1` with no time cross terms and a static fluid, giving
`u.u = -1` identically; Minkowski is flat. Every polarized test runs in snake.

`mc_gr_simple` built and ran unmodified. It carries a radially varying primitive velocity,
`uu1 = 1/sqrt(1-2/r) * 2/(r+2)`, so both the normalization and the direction error are live. Two
user slots were added to it -- `IUUDEV`, the running maximum of `|u.u + 1|` along each photon's
path, and `IUURAD`, the radius where it peaked -- read by `tst/montecarlo/kerr_frames/`.

Baseline, a = 0.9375, r in [2, 20], 5000 photons, resolution scan:

| dr | max \|u.u + 1\| | order |
|---|---|---|
| 0.2812 | 7.0994e-02 | -- |
| 0.1406 | 3.3734e-02 | 1.07 |
| 0.0703 | 1.8483e-02 | 0.87 |

Nonzero, and **first order in the cell width**, which is the signature of an offset error rather
than anything else. The peak sits at r = 3.124 on the coarse grid; the cell edges are at
`2.0 + n*0.2812`, so 3.125 is exactly a cell boundary -- the point farthest from a cell center.
After Phase 3 this column should collapse to roundoff at every resolution.

Two things surfaced while building this:

- **`vel` was zero in ghost cells. Fixed.** Every fluid-derived array was filled over active
  cells only, but a photon can hold ghost indices while it waits to be handed to the neighboring
  block, and the pusher goes on reading at those indices -- so `vel` returned a null four-velocity
  to `FrequencyShiftComoving` and the two transform routines. `MonteCarloBlock::FillBounds` now
  gives the active-plus-ghost range and `GetDensity`, `GetTemperature`, `GetNumberDensity`,
  `GetScalars`, `GetVelocity`, `SetNormalObserver` and `ComputeTransformations` all use it.
  `GetBField` is a shallow slice and needs nothing.

  Two guards came with it, because extending into ghosts means reading cells a problem generator
  may never have written: `GetTemperature` would form `0/0`, a NaN its floor and ceiling cannot
  clamp since every comparison against NaN is false, so it falls back to the floor; and the
  non-GR `GetVelocity` would divide by zero density, so it leaves the fluid at rest there.

  The source primitives have to be valid in the ghosts for this to mean anything. They are for a
  problem generator that fills its full range, which is the Athena++ convention, and non-GR runs
  get them refreshed by `ConservedToPrimitive` in `Mesh::Initialize`. **Note that
  `ConservedToPrimitive` is deliberately skipped when `MONTE_CARLO_ENABLED` and
  `GENERAL_RELATIVITY` are both set**, so in GR the ghosts are exactly what the problem generator
  wrote and nothing else.
- The `velocity` input to `mc_gr_simple` is read and never used, and the four-velocity block it
  would feed is commented out. Left alone.

These runs used to need a large `<montecarlo>/checkmove`, because with `varystep` the step is a
fraction of a cell and refining the grid multiplied the steps per crossing until the default cap
destroyed every photon. `checkmove` has since been removed; nothing needs setting now. If
`capmove` is ever turned on for such a run, it needs the same generous margin.

### Phases 2 and 3 -- reconstruct at the photon, switch the consumers. **Done.**

`uprim` holds the primitive relative velocity `uu^i` per cell in GR, filled beside `vel` in
`GetVelocity` and zeroed in `SetNormalObserver`.
`MonteCarloBlock::FluidFourVelocity(x, i3, i2, i1, ucon)` redoes `GetVelocity`'s arithmetic
against the metric at `x`, so `u.u = -1` holds where the vector is used. With boosts off `uprim`
is zero and the same formula returns the normal observer at `x` -- exact, since no cell-center
quantity enters.

Switched: `FrequencyShiftComoving`, `TransformToComoving`, `TransformToCoordinate`, and the
comoving tetrad in `MeridianBasis`. The last needs a `GENERAL_RELATIVITY` guard, because `vel`
exists for flat polarized runs but `uprim` is GR only. The transform pair stays exact: they
bracket `Scatter`, which does not move the photon, so both rebuild the same tetrad at the same `x`.

| dr | before | after |
|---|---|---|
| 0.2812 | 7.0994e-02 | 1.1102e-15 |
| 0.1406 | 3.3734e-02 | 1.1102e-15 |
| 0.0703 | 1.8483e-02 | 1.3323e-15 |

Gates: `mc_snake_atm` byte-identical (the negative control -- snake has `uprim = 0`, so the
reconstruction returns exactly what `vel` held, proving the change touched only the curved path);
`poltest` 4/4; `snake_polarization` order 2.00; `snake_thomson_spectrum` converging against
Feautrier; the Phase 0 acceleration deck unchanged.

A note for whoever runs these next: **check `PROBLEM_FILE` in the Makefile before believing a
test result.** Running one pgen's deck against another pgen's binary produces confusing failures
-- an `MCOutput` error reporting `user variables: 0` is the signature -- and rebuilding without
reconfiguring reproduces it, which makes it look like a pre-existing bug rather than a stale
binary.

### Phase 2 (original text) -- Add the reconstruction, call it from nowhere

```
void MonteCarloBlock::FluidFourVelocity(const Real x[4], int i3, int i2, int i1,
                                        Real ucon[4]) const;
```

GR only. Rebuilds `u^mu` from the cell's stored primitives with the metric at `x`, using the same
algebra as the existing setup (`gamma2 = 1 + g_ij uu^i uu^j`, `u^0 = -gamma*alpha*g^{00}`,
`u^i = uu^i - gamma*alpha*g^{0i}`). The `x` argument makes a later interpolated variant a
signature-compatible change. Keep `vel` populated; add a debug assertion that the two agree at cell
centers to roundoff.

*Gate:* every existing test byte-identical -- nothing calls it yet.

### Phase 3 -- Switch GR consumers, one per commit

`FrequencyShiftComoving`; `TransformToComoving`; `TransformToCoordinate`; the comoving tetrad in
`polarization.cpp`. The middle two are the round-trip pair and must be tested together.

The round trip stays exact: `TransformToComoving` and `TransformToCoordinate` bracket `Scatter`,
which does not move the photon, so both calls see the same position and rebuild the same tetrad.
Emission calls `TransformToCoordinate` alone.

*Gate, every step:* snake decks **byte-identical** (any change means the edit escaped the curved
path); `poltest` 4/4; `snake_polarization` order 2.00; `snake_thomson_spectrum` vs Feautrier
unchanged; the Phase 1 Kerr metric strictly improves.

### Phase 4 -- Retire the GR four-velocity, document the flat convention

`vel`'s GR branch stores primitives; `ComputeTransformations` reconstructs a cell-center `u^mu`
locally for `boost_cmv`/`boost_lab`. Non-GR `vel` is unchanged but documented as holding
`(gamma, gamma*beta^i)` -- two conventions sharing one array name is what made this confusing.
Replace the `sqrt(|u.u|)` justification in `ObserverEnergy` with a note that the normalization is now
exact by construction.

*Gate:* full suite; moments outputs byte-identical to Phase 0.

### Phase 5 -- Moments guard

Assert that `moments`, `moments_com` and `moments_coord` did not change across Phases 3-4, and
document the deliberate remaining inconsistency: the photon's comoving frame is defined at its own
position, the comoving moments at the cell center. The cell-center-tetrad question and the cached
`acon` transport correction stay deferred.

## 6. Risks

- **Phase 1 is the real risk.** If `mc_gr_simple` needs substantial work, everything downstream is
  unverifiable.
- `phydro->w(IVX..IVZ)` is already read by the GR velocity setup, so primitives are available in GR
  mode. Re-confirm for the VTK-driven `mc_disksim` path.
- The `sqrt(|u.u|)` division must stay until Phase 4; while `vel` and the reconstruction coexist,
  some call sites still need it.

Remember `make clean` before `make` after any header edit -- Phase 2 touches `montecarlo.hpp`.
