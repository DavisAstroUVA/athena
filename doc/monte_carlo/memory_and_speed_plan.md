# Memory footprint and speed of general-relativistic Monte Carlo runs: plan

Status: **Phase M implemented (M1 to M5), Phase S not started.** Written 2026-09-12 from
a survey of the code as it stands on `polarization_update`; Phase M landed 2026-09-13 on
`memory_reduction`. Measured on one rank, `/usr/bin/time -v` maximum resident set:

| deck | baseline | after M1 to M3 | after M5 | photon list / spectrum |
|---|---|---|---|---|
| snake atmosphere, `gr_user`, 32x32x128 in 16 blocks, 200000 photons | 603 MB | 285 MB | | spectrum byte-identical |
| same, 20000 photons | 503 MB | 187 MB | 182 MB | transport identical; Q, U differ at 5e-16 |
| isothermal shell, `kerr-schild`, 128x8x8, 200000 photons, polarized | 122 MB | | 92 MB | transport columns byte-identical; Q, U differ by 2e-16 |

M4 removes nothing on either deck because both problem generators enroll user moments,
which keeps the moment deposition and with it the tetrad cache; it applies to the XRB
generator, which enrolls none. `mc_poltest` (gr_user, no moments) exercises the skipped
path. The end-of-run line `peak resident memory = ... MB per rank (max), ... MB total` is
the in-code measurement from section 0. Target run: `src/pgen/mc_xrb_hdf_gr.cpp` on `gr_user` with the
Cartesian Kerr-Schild metric, `emission = freefree`, `absorption = freefree`, Thomson or
Compton scattering, `polarized = linear`, general pusher, MC post-processing (`dynamic =
false`). Everything below is scoped to changes that leave the code structure alone: no
change to the photon loop nest, the pusher interface, or the exchange.

The order is deliberate: memory first, because those items are independent of each other
and of the speed work and each is verifiable by a resident-memory number; then speed, in
the order of expected gain per line changed.

---

## 0. Baseline and verification, before touching anything

Every item below is checked against the same two decks, so record their baselines first.

- **GR deck:** `tst/montecarlo/kerr_frames/athinput.kerrframes` (Kerr-Schild, general
  pusher, polarized). Configure with `--prob=mc_isoth_gr --coord=kerr-schild -g -mc` unless
  the item concerns `gr_user`, in which case use a `gr_user` build of the same problem.
  The XRB deck itself needs a snapshot and is the final check, not the development loop.
- **Flat control:** `tst/montecarlo/disk_atmosphere` with the general pusher, to see that a
  change to shared code does not move a non-GR result.

Record for each deck, single rank (`mpirun -n 1`, because MPI runs are not bit
reproducible):

| quantity | how |
|---|---|
| peak resident memory | `VmHWM` from `/proc/self/status`, printed at the end of `main` (add this print first; it is a few lines and stays) |
| wall time of the transport phase | `cpu time used` already in the log; also `zone-cycles`-style photons per second, already printed |
| the photon list | `cmp` against the baseline list; a bit-identical list is the pass criterion for every item marked *bitwise* below |
| the spectrum | `make_spectrum.py` on both lists, compared to the statistical error; the pass criterion for items marked *statistical* |

Items marked **bitwise** rearrange no arithmetic and must reproduce the baseline list
exactly. Items marked **statistical** change rounding (a precomputed prefactor, a fused
expression) and are checked against the spectrum error and the existing test gates in
`tst/montecarlo/spherical_polarization` and `tst/montecarlo/poltest`.

---

## Phase M: memory

Per ghost-inclusive cell, a GR post-processing run carries roughly (Reals, 8 bytes each):

| owner | Reals per cell | read by the MC transport |
|---|---|---|
| `Coordinates` for `gr_user` (`src/coordinates/gr_user.cpp:141-159`) | ~180 | `metric_cell_kji_` (20), once, in `ComputeTransformations` and `DeriveComovingMoments` |
| Hydro `u`, `w`, `u1`, `w1`, `flux[3]` (`src/hydro/hydro.cpp`) | ~35 | `w` only |
| Field with `-b`: `b`, `b1`, `bcc`, `e`, `wght`, edge EMFs, `cc_e_` (`src/field/field.cpp`) | ~22 | `bcc` only, by shallow slice |
| MC `boost_cmv` + `boost_lab` (`montecarloblock.cpp:352-353`) | 32 | moment deposition only, in GR |
| MC `rho`, `tgas`, `species(2)`, `uprim(3)`, `emission`, `vol` | 9 | yes |
| MC `moments` when any moment output is on | 13 | yes |

About 2.3 kB per cell; an 8x8x8 block with `NGHOST = 2` multiplies that by 3.4 for the
ghost layer. Photons cost 472 bytes each (8 ints, 23 Reals, 16 complex), so the default
`max_resident_photons = 1000000` is about half a gigabyte per rank.

The hydro side is never advanced in this mode: `main.cpp:524-538` runs the time integrator
only when `!MONTE_CARLO_ENABLED || pmc->dynamic`. So everything the integrator needs and
the MC does not is dead weight for the whole run.

### M1. Skip the face-metric arrays in `gr_user` when hydro never runs

**What.** In the `GRUser` constructor (`gr_user.cpp:142-161`) allocate
`coord_src_kji_`, `metric_face1/2/3_kji_` and `trans_face1/2/3_kji_` only when the run
will integrate hydro. Keep `metric_cell_kji_`, the volumes, areas, lengths and widths:
`CellMetric`, `GetCellVolume` and the face areas are read at setup, and the cost of the
kept set is about 30 Reals per cell against 150 removed.

**Gate.** `Mesh` is constructed before `MonteCarlo` (`main.cpp:282` vs `:335`), so the
constructors cannot ask `pmc`. Read the same inputs `MonteCarlo` will read:
`MONTE_CARLO_ENABLED && !pin->GetOrAddBoolean("montecarlo", "dynamic", false)`. Put that
test in one inline helper (`bool HydroIsStatic(ParameterInput *pin)`, in a small header
under `src/monte_carlo/`) so M1, M2 and M3 cannot drift apart. The helper returns `false`
whenever `MONTE_CARLO_ENABLED` is 0, so a hydro build is untouched.

**Also fill the guard downstream.** `Face1Metric`, `Face2Metric`, `Face3Metric`, the
`PrimToLocal*`/`FluxToGlobal*` transforms and `AddCoordTermsDivergence` read these arrays.
None is reached without the integrator, but a stray call would then read an unallocated
`AthenaArray`. Add a one-line fatal check at the top of each when the arrays are not
allocated, so a future caller gets an error instead of a segfault.

**Expected gain.** ~150 Reals per cell, the single largest item. **Check:** `VmHWM` on
the `gr_user` build of the GR deck; the list is *bitwise*.

### M2. Same gate for the hydro integrator's registers

**What.** In the `Hydro` constructor skip `u1` and `flux[0..2]` when `HydroIsStatic`.
Keep `u` and `w`: the problem generator calls `PrimitiveToConserved` into `u`, and hydro
outputs may be requested from an MC run. **Keep `w1` as well**, found during
implementation: `Mesh::Initialize` passes it to `ConservedToPrimitive` as the previous
state, restarts pack it (`meshblock.cpp`, `restart.cpp`), and `mc_isoth_gr` and
`mc_snake_atm` write it alongside `w`. `u2`, `u0`, `fl_div`, `u_cc`, `w_cc` are already
conditional on the integrator and order and stay as they are.

**Expected gain.** ~20 Reals per cell. **Check:** as M1, *bitwise*.

### M3. Same gate for the field's integrator registers

**What.** In the `Field` constructor skip `b1`, `e`, `wght`, the six edge-EMF face arrays
and `cc_e_` when `HydroIsStatic`. Keep `b` and `bcc`: the snapshot reader fills `bcc`
directly (`mcsnapshot.cpp:392`) and MC slices it.

**Expected gain.** ~17 Reals per cell when `-b` is on. **Check:** as M1, *bitwise*.

### M4. Allocate the per-cell tetrads only when moments are deposited

**What.** In GR, `boost_cmv` and `boost_lab` are read only by `PhotonFrames::Fill`
(`photon_frames.cpp:65`) and `DeriveComovingMoments` (`photon_frames.cpp:226`). The scatter
path builds its tetrad on the fly through `ComovingFrame`, and the GR branch of
`FrequencyShiftComoving` never touches them. The non-GR branches of `TransformToComoving`,
`TransformToCoordinate` and `FrequencyShiftComoving` do read them, so the condition is:

    allocate and fill when (boosts || tetrads) && (call_moments || !GENERAL_RELATIVITY)

with `call_moments` already defined at `montecarloblock.cpp:93`. `ComputeTransformations`
takes the same condition; the destructor at `:462` follows the allocation.

**Expected gain.** 32 Reals per cell in any GR run without moment output. **Check:** GR
deck with and without a moment output; both lists *bitwise*; the moment output itself
unchanged when on.

### M5. Store the coherency tensor in its independent real components

**What.** `polten` is 16 complex properties, 256 bytes per photon, for a Hermitian tensor
with 16 real degrees of freedom (4 real diagonal, 6 complex off-diagonal). Register 16
real properties instead and give `Photon` two inline helpers, `LoadTensor(ip, N[4][4])`
and `StoreTensor(ip, N[4][4])`, that expand to and contract from the full complex 4x4.
Convert the readers and writers to go through them: `AdvanceStep`
(`generalpusher.cpp:563, 595`), `PolarizationToTetrad`/`PolarizationToCoord`
(`photon.cpp:208-244`), `IsNanPhoton`, `PrintPhoton`, `polarization.cpp`, and the
exchange's complex buffer (`ParticleBuffer::cbuf`, which then carries nothing and can be
skipped when `ncplx == 0`, as its constructor already allows).

**Why it is last in this phase.** It touches the photon layout, so it is the one item in
the phase that needs `make clean` and a full re-run of the polarization gates:
`spherical_polarization`, `poltest`, `snake_polarization`, `disk_atmosphere`.

**What it turned out to be.** The transport in `AdvanceStep` preserves Hermiticity
exactly in floating point, so on the geodesic and the scattering draws the change is
*bitwise*. The frame transforms (`PolarizationToCoord` and friends) sum the same terms
in a different order for (i,j) and (j,i), so the mirrored lower triangle can differ from
the full product in its last bit; that reaches the Stokes parameters at the 1e-16 level
and nothing else. The complex property machinery in `Particles` and the exchange is
left in place with a count of zero.

**Expected gain.** 128 bytes per photon, 27 percent of the photon footprint. It also
halves the exchange traffic per polarized photon and sets up S4.

### M6. Not done, and why

- `dk0p..dk3p` (32 bytes per photon) are written only by `VerletStep`, which nothing
  calls. Dropping them is easy but the header comment at `mccoord.hpp:106` says Verlet
  may be revived; leave them until that is decided.
- The four Stokes columns duplicate the tensor during general-pusher transport but are
  the legacy pushers' only polarization state and are what the outputs read. Making them
  conditional on the pusher is a layout change across every output path; not worth it
  for 32 bytes.
- `NGHOST = 1` would cut the ghost factor from 3.4 to 1.95 for 8x8x8 blocks, but the
  reconstruction order check and the snapshot reader's ghost fill both assume 2. Note
  only.
- Switching the XRB run from `gr_user` to a built-in coordinate would make M1 moot, but
  there is no Cartesian Kerr-Schild `Coordinates` class, only the MC-side
  `MCKerrSchildCartesian`. Out of scope.

---

## Phase S: speed

The step loop makes fifteen virtual metric-family calls per step, six of them at the same
point, and each recomputes `r`, `f`, the null vector, a `hypot` and a `sqrt` from scratch:

| caller | `Metric` | `InverseMetric` | `InverseMetricDerivative` | `Connect` |
|---|---|---|---|---|
| `FrequencyShiftComoving` (`montecarloblock.cpp:2481`) | 1 | | | |
| `FluidFourVelocity` (`montecarloblock.cpp:1943-1944`) | 1 | 1 | | |
| `RK4Step` start, four `SubStep`s, end (`generalpusher.cpp:361-414`) | 2 | 5 | 4 | |
| `ConnectionContraction` (`generalpusher.cpp:487`) | | | | 1 |

On top of that the free-free opacity (`opacity.cpp:58-66`) runs every step in GR, because
`shift_unity` is false for any curved metric (`montecarloblock.cpp:2225`), and each call
does an `exp`, a `sqrt` and a libm `pow(nu, 3)`.

Measure before and after each item on the GR deck; an item that does not move the photon
rate measurably is reverted, not kept on principle.

### S1. One metric evaluation per point

Three sub-steps, each *bitwise* if the expressions are kept in the same order.

**S1a. Fuse `InverseMetric` and `InverseMetricDerivative` in `SubStep`.** Add a virtual
`InverseMetricAndDerivative(x, gcon, dgcon)` to `MCCoord` with a default that calls the
two existing routines, and override it in `MCKerrSchildCartesian` and `MCKerrSchild` with
one body that computes `r`, `f`, `l`, `dr`, `df` once. `SubStep` calls it. Four calls
become four, but each does half the transcendental work.

**S1b. Fuse the end-of-step pair.** `RK4Step:406` and `:414` evaluate `InverseMetric` and
`Metric` at the same point. Add `MetricAndInverse(x, gcov, gcon)` with the same
default-plus-override pattern; in Kerr-Schild both come from the same `f` and `l`.

**S1c. Carry the end-point metric into the next step.** The step ends with `gcov`,
`gcon` at `x`; the next step's `FrequencyShiftComoving`, `FluidFourVelocity` and
`RK4Step:361` all want exactly those at exactly that `x`. Give `GeneralPusher` a small
cached struct `{x[4], gcov[4][4], gcon[4][4], valid}` next to the existing `acon_valid`,
filled at the end of `RK4Step`, invalidated wherever `acon_valid` is, and consulted by a
`MetricAt(x, ...)` helper that compares `x` against the cache before calling the
coordinate object. `FrequencyShiftComoving` and `FluidFourVelocity` take the metric as an
argument from the pusher rather than calling `pcoord` themselves; their other callers
(setup, scattering) pass a freshly evaluated one.

**Expected gain.** Fifteen evaluations to roughly five per step. The geodesic and the
shift are the bulk of a free-flight step, so this is the largest single item.

### S2. Precompute the free-free per-cell prefactor

**What.** Add a per-cell array `ff_prefactor(k,j,i) = ffnrm * n_e * n_ion / sqrt(T)`,
filled in `MonteCarloBlock` after `GetNumberDensity`, and only when the absorption or
emission method is free-free. The per-step body of `FreeFreeOpacity` becomes

    Real nu3 = nu*nu*nu;
    return ff_prefactor(i3,i2,i1) * (1. - exp(-energy/(kb*tgas))) / nu3;

One array read instead of three, no `sqrt`, no `pow`. The `tgas` read stays for the
exponential; storing `1/(kb*T)` alongside removes the division as well. One Real per
cell, which M1 to M4 have more than paid for.

**Expected gain.** Removes a libm `pow` and a `sqrt` per step. *Statistical*: the
prefactor is rounded once per cell instead of per call.

### S3. Drop the duplicated work around the frequency shift

Three small items, each *bitwise*:

- `FrequencyShiftComoving` evaluates `Metric(x)` (`:2481`) and then `FluidFourVelocity`
  evaluates it again (`:1943`). Covered by S1c once both take the metric as an argument;
  if S1c is deferred, pass `gcov` from the first to the second.
- With `mom_flag_scat` on, `UpdateMoments` (`montecarloblock.cpp:942`) recomputes the
  shift `UpdateOpacities` produced forty lines earlier in the same step. Store the shift
  in the pusher for the step and hand it to `UpdateMoments`.
- `UpdateOpacities` applies the shift as five multiply/divide round-trips on `ep`, `acp`,
  `scp` (`generalpusher.cpp:237-245`). Compute the comoving energy into a local and pass
  it to the opacity functions, which already take the block and photon, so that `ep` is
  never touched. That is an interface change to `OpacFunc_t` (add an energy argument), so
  do it only if the profile shows the divisions.

### S4. Exploit Hermiticity in the tensor transport

**What.** `ApplyPolarizationRate` (`generalpusher.cpp:515-528`) runs twice per step over
all sixteen `(i,j)` pairs. `dN/dl = -(A N + N A^T)` is Hermitian when `N` is, so compute
the ten pairs with `j >= i` and mirror. Combined with M5, the tensor is loaded and stored
once as 16 reals rather than gathered from sixteen separate vectors.

**Expected gain.** About 40 percent of the polarization block per step. *Statistical*
only if the mirrored entries were previously computed with a different summation order;
otherwise *bitwise*. Gate: `spherical_polarization` TRANSV and POLDEG, `poltest`.

### S5. De-virtualize the concrete metric

**What.** After S1 the pusher makes four or five virtual calls per step into a large
body. Two options that keep the class structure:

- template `GeneralPusher` on the concrete `MCCoord` type and instantiate it in the
  `MonteCarloBlock` constructor switch that already picks the coordinate object, or
- give `MCCoord` a non-virtual `Dispatch` that switches on `MCCoordSystem` and calls the
  derived body directly, marked `inline` in the header.

Prefer the template: the switch adds a branch per call and prevents nothing. The virtual
interface stays for everything outside the pusher.

**Expected gain.** Inlining lets the compiler hoist the shared `r`, `f`, `hypot` across
the fused routines and drop the zero-fill loops that precede every body. Worth measuring
before and after S1 to see whether it is still needed. *Bitwise*.

### S6. Compiler flags

**What.** The build is `-O3 -std=c++11`. Configure with `--cxx=g++-simd` and drop
`-ffast-math` from the flag string in `configure.py:488-491`, since `IsNanPhoton` and the
volume check depend on `isnan`. Keep `-march=native -flto`. Measure; report the photon
rate with and without.

**Expected gain.** Unknown until measured; `hypot` and the Kerr-Schild bodies are
division heavy, so `-march=native` may matter. *Bitwise* is not guaranteed under `-flto`
with fused multiply-add; use the *statistical* criterion.

### S7. Hygiene in the step loop

All *bitwise*, all one or two lines, done together at the end:

- `GetOpticalDepth` draws a random number and takes a `log` before the `IsOnBlock` test
  (`generalpusher.cpp:65-66`). Swap the two lines. This changes the random sequence for
  blocks with off-block photons, so it is *bitwise* only for a single-block run; check on
  the flat control with `mpirun -n 1` and one block.
- `acon_valid` is cleared on every cell crossing (`generalpusher.cpp:177-179`) although
  `x` and `k` have not moved; only a periodic remap moves them. Clear it only when
  `UpdateZone` actually remapped the position (it can return that), which saves one
  `Connect` per cell crossing.
- Hoist `IsPolarized(pmy_mcb->pmy_mc->polarized)`, `pmcb->shift_unity`,
  `pmcb->call_moments`, `capmove` and `1./c_code` into locals above the `while` in `Move`.
- Remove `pmy_mcb = pmcb;` from `UpdateOpacities` (`generalpusher.cpp:225`).
- Replace `pow(nu,3)` in `emission.cpp` and in the pgen's `FreeFreeOpacity` as in S2.

### S8. Latent bug to fix in passing

In the MRW branch, `step = dl;` at `generalpusher.cpp:165` reads the base-class member
`PhotonPusher::dl` (`photonpusher.hpp:38`), not the block-local `dl` declared at line 118,
which is out of scope there. `GeneralPusher::Move` never writes the member. Hoist the
local above the `if` and remove the member if nothing else uses it. Does not affect the
free-free runs; fix it because it is two lines and will bite the resonance path.

### S9. Out of scope, recorded so it is not re-derived

The loop nest is photon-major: `Move` runs one photon's entire free flight before the
next, so the struct-of-arrays photon layout gains nothing, every step gathers about twenty
scalars from twenty strided vectors, and there is no thread or SIMD parallelism over
photons. Inverting to step-major, taking one step for every photon in a block per pass,
is the real speedup: it would let same-cell photons share the metric and the opacity
prefactor and open the loop to `omp simd`. It is a restructure of `Move`, `UpdateZone`
and the interaction bookkeeping and is not part of this plan.

Also not here: a faster generator than MT19937 (draws are once per flight and per
scatter, not per step, so it is not on the step path), and the RK4 versus Verlet
question.

---

## Order of execution and checkpoints

1. Baseline prints and numbers (section 0). Commit.
2. M1, M2, M3 together behind the one helper. Check `VmHWM` and *bitwise* lists on both
   decks, plus a hydro-only build to confirm nothing changed there. Commit.
3. M4. Same checks. Commit.
4. M5, with `make clean` and the full polarization gate set. Commit.
5. S1a, S1b, S1c in turn, each *bitwise* against the previous. Commit after S1c.
6. S2, *statistical*. Commit.
7. S3, S7, S8 together, *bitwise* except as noted. Commit.
8. S4, gates. Commit.
9. S5 and S6, measured; keep what pays. Commit.
10. Final: the XRB deck on a real snapshot, before-and-after memory and photon rate, and
    a spectrum comparison within its statistical error.
