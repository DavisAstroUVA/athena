# Reconstructing the Athena++ grid from an athdf snapshot: implementation plan

Status: **phases A through D implemented** (`src/monte_carlo/mcgrid.{hpp,cpp}`, gated hooks
in `main.cpp` and `mesh.cpp`, the `FileIndex` lookup in the three athdf-reading problem
generators, and the test in `tst/montecarlo/mcgrid/`). Uniform, SMR and AMR snapshots all
set up their own grid; the block tree is replayed from the file. **Phase E is not started,
so photon transport across refinement boundaries is unverified — do not draw physics
conclusions from a refined snapshot yet.** This note records the survey and the design so
the rest can be picked up without re-deriving the context.

On the gid/file-index question of section 2: measured on the real snapshot, the mapping is
the identity for all 83231 blocks, as the argument there predicts. `MCGridFile::FileIndex`
therefore changes no behaviour today. It is kept because the identity holds only as long
as the writer and the reader both order blocks by `MeshBlockTree::GetMeshBlockList`, which
no part of the athdf format guarantees, and because a silent mismatch would misattribute
every cell in the run.

Phase C was verified against a real 83231-block, 5-level spherical-polar AMR snapshot: the
rebuilt tree reproduces the file's per-level block counts exactly (288 / 1075 / 2948 /
14216 / 64704), and `MapBlocks` matched every block by logical location. A conventional
SMR mesh built from `<refinement>` regions is byte-identical before and after the
`mesh.cpp` change, in both stdout and `mesh_structure.dat`.

Known cost: `MCGridFile::MapBlocks` is a linear scan per gid, so O(N^2) in the block
count. It runs once at setup and takes roughly 4 s for 83231 blocks. Sort the file's
locations and bisect if that ever matters.

Motivation: the Monte Carlo module reads athdf snapshots of earlier Athena++ runs. Today
the reader requires the MC athinput to describe a grid that matches the snapshot exactly,
which means having the original athinput and a run that was uniform or statically refined.
AMR snapshots have no workable path at all; the current workaround is to flatten them to a
uniform grid at some level with `vis/python/uniform.py` (or `athena_read`) and read the
flattened file, which throws away the resolution that motivated the AMR in the first
place.

The goal is to set up the Athena++ mesh directly from the athdf file, for uniform, SMR and
AMR snapshots alike, without a hand-written grid description.

---

## 1. Prior art: this does not exist upstream, and was deliberately declined

Checked against `PrincetonUniversity/athena` `main` at `823614c9` (2026-08-13).

Nothing upstream does this:

- Two `Mesh` constructors only — `Mesh(pin, mesh_test)` and `Mesh(pin, resfile, mesh_test)`.
  No athdf variant.
- `MeshBlockTree::AddMeshBlockWithoutRefine` has exactly one caller in the entire tree: the
  restart constructor. The tree-replay path exists only for `.rst`.
- `src/inputs/hdf5_reader.hpp` exposes only `HDF5ReadRealArray` and `HDF5TableLoader`.
  There is no attribute reader and no integer-dataset reader, so upstream cannot parse an
  athdf header from C++ at all.
- Outside the writer, no source file anywhere reads `LogicalLocations`, `Levels`,
  `RootGridSize` or `MeshBlockSize`.
- The only athdf-consuming pgen is still `from_array.cpp`, indexing by `gid`. Its shipped
  input has `refinement = none`, and the regression tests pass `mesh/nx1` and
  `meshblock/nx1` by hand — upstream's supported workflow is the manual one we are trying
  to replace.
- `vis/python/uniform.py` is upstream's answer to AMR, and it is the flattening workaround
  described above.

It was also considered and rejected as a direction. Issue #143 ("include input parameters
in hdf5 output?", 2018) is this exact discussion; it was closed by merging PR #146, which
*is* today's `hdf5_reader.cpp` + `from_array.cpp` and nothing more. The maintainers'
position was that athdf and restart files should stay distinct, that athdf should remain
compact, and that `.rst` is the vehicle for full run state since it embeds the input deck.
Two counterpoints from that thread are worth carrying: restart files are architecture
dependent and so are not portable between machines, and the long-term intent upstream is
to move off HDF5 entirely.

Consequences for us: there is no upstream code to borrow, little prospect of upstreaming
this, and no upstream design that blocks doing it fork-locally. It also means the athdf
layout should be treated as a **versioned contract we validate on read**, not as a stable
format — see §4.

---

## 2. What an athdf actually contains

From `src/outputs/athena_hdf5.cpp`, attributes at lines 579-655 and datasets at 687-703:

| Item | Kind | Meaning |
|---|---|---|
| `Coordinates` | attr, string | the `COORDINATE_SYSTEM` macro |
| `RootGridX1/X2/X3` | attr, 3x real | `(xmin, xmax, xrat)` of the root mesh |
| `RootGridSize` | attr, 3x int | `mesh_size.nx1/2/3` |
| `MeshBlockSize` | attr, 3x int | `block_size.nx1/2/3` |
| `MaxLevel` | attr, int | `current_level - root_level`, physical |
| `NumMeshBlocks` | attr, int | `nbtotal` |
| `Levels` | dset `[nb]` | `loc.level - root_level`, **physical** level |
| `LogicalLocations` | dset `[nb][3]` | `loc.lx1/lx2/lx3`, **logical** indices at `loc.level` |
| `x1f,x2f,x3f` | dset `[nb][nx+1]` | per-block face coordinates |
| `x1v,x2v,x3v` | dset `[nb][nx]` | per-block cell centers |

Note the asymmetry at `athena_hdf5.cpp:390-393`: `Levels` is offset by `root_level` and
`LogicalLocations` is not.

**The .xdmf is never needed.** It is derived XML written by `MakeXDMF()` from data already
in the athdf; `athena_read.athdf()` ignores it entirely.

### The reconstruction identity

The athdf carries exactly the tree information the restart ID list carries:

```
nrbx{1,2,3} = RootGridSize / MeshBlockSize
root_level  = smallest r with (1<<r) >= max(nrbx1,nrbx2,nrbx3)     // mesh.cpp:341
loclist[i]  = { lx = LogicalLocations[i], level = Levels[i] + root_level }
```

which then replays through the code that already exists for restarts, `mesh.cpp:857-870`:

```cpp
tree.CreateRootGrid();
for (int i=0; i<nbtotal; i++) tree.AddMeshBlockWithoutRefine(loclist[i]);
tree.GetMeshBlockList(loclist, nullptr, nnb);
```

**AMR is the easy case, not the hard one.** An AMR tree is just an arbitrary `loclist`, and
`AddMeshBlockWithoutRefine` accepts any of them. What is awkward today is SMR, because the
current constructor *re-derives* the tree from `<refinement>` blocks via the bisection
search at `mesh.cpp:434-516`. Replaying `loclist` bypasses that and covers uniform, SMR and
AMR through one path.

A useful consequence: the block index in the file **is** the gid of the reconstructed mesh.
`first_block = nslist[my_rank]` (`athena_hdf5.cpp:103`), and both runs order blocks by
`tree.GetMeshBlockList` over the same tree, so `start_cons_file[1] = gid` in
`from_array.cpp:65` and `mc_readhdf.cpp:483` keeps working. We should still build an
explicit permutation rather than lean on the invariant — cheap insurance, see §5 Phase B.

---

## 3. What the athdf does *not* contain

This is the answer to "is the file alone enough": enough for the **grid**, not for the
physics.

- **Physics parameters.** No GR spin or mass, no `gamma`, no units, no opacity settings, no
  boundary flags. Boundary flags matter to us specifically:
  `Photon::ApplyBoundaryConditions` (`photon.cpp:478`) branches on mesh extent for periodic
  wrapping, and `mesh.cpp:144` already derives fluid BCs from the MC `ix1_mc_bc` settings.
  An athinput is still required; it just no longer has to describe the grid.
- **User mesh generators.** If the source run called `EnrollUserMeshGenerator`, the athdf
  records only `x1rat`, and `SetBlockSizeAndBoundaries` (`mesh.cpp:1759`) will regenerate
  the wrong face positions. Detectable by comparing against the stored `x1f` — do it.
- **Precision.** Default athdf output is single precision (`athena_hdf5.cpp:43-53`;
  `-h5double` opts in), so `x1min/x1max/x1rat` carry only about seven digits. The error
  does **not** compound over `nx1`, contrary to what this note first assumed:
  `DefaultMeshGeneratorX1` evaluates `rat^(x*nx)` on a normalized fraction rather than
  multiplying iteratively, so what is left is just the float32 storage of the attributes.
  Measured on a real 5-level spherical-polar AMR snapshot with a logarithmic radial grid
  (r = 2.7 to 400, 83231 blocks), the worst disagreement between regenerated and stored
  faces is 2.1e-7 relative, essentially float32 epsilon. The face check therefore uses a
  1e-6 relative tolerance. `-h5double` is still worth recommending for new runs, but it is
  not needed for the grid to be recovered correctly.
- **`costlist`.** Restart has per-block costs; athdf does not. Uniform costs are fine for
  post-processing.
- **NGHOST / xorder.** Not stored, and with `multilevel` the constructor enforces
  `block_size.nx >= max(2*(xorder-1), NGHOST)` (`mesh.cpp:300`). An MC build with
  `NGHOST=4` reading a snapshot with 8^3 blocks will trip this. Needs a clear error.

---

## 4. Two upstream writer changes to defend against

Our `athena_hdf5.cpp` is behind upstream, and both deltas touch what a reader consumes.
Files produced by newer upstream builds may arrive here, so validate rather than assume.

1. **`mesh_data` output option** (upstream `outputs.cpp:345`, default `true`). When false,
   `Coordinates`, `RootGridX1/2/3`, `RootGridSize`, `NumMeshBlocks`, `MeshBlockSize`,
   `MaxLevel` *and* `x1f..x3v` are all omitted. `Levels` and `LogicalLocations` are still
   written unconditionally, so the tree survives but the root extent and the face
   cross-check do not. Detect and fail with a clear message.
2. **`ATHDF5Output` templated on output type**, with a per-output `data_format`
   (`f16`/`f32`/`f64`/`f128`/`u8`..`u64`). Mesh coordinates follow a separate `mesh_t` that
   tracks the cell-data type, except that unsigned dumps fall back to the compile-time
   `H5Real` — so the precision caveat in §3 is not fixed upstream. The `u8`/`u16` modes
   store cell data as integers scaled by `vmin`/`vmax` and should be rejected outright.

The reconstruction identity itself is unchanged upstream: `Levels` is still
`loc.level - root_level` and `LogicalLocations` still `loc.lx1/2/3`.

---

## 5. Containment: how this respects the module boundary

The MC module is a separate code built on the Athena++ mesh, and development stays inside
`src/monte_carlo/` wherever possible. Where other parts of the tree must be touched, the
change is wrapped in `if (MONTE_CARLO_ENABLED)`. This plan keeps that.

Three facts make full containment practical here:

- `src/monte_carlo/*.cpp` is compiled **unconditionally** (`Makefile.in:55` wildcard). MC
  translation units always link.
- `MONTE_CARLO_ENABLED` is `#define`d to `0` or `1` (`defs.hpp.in:52`, `configure.py:797`),
  so the house style outside the module is a runtime `if (MONTE_CARLO_ENABLED)` — see
  `mesh.cpp:144`, `main.cpp:325`, `outputs.cpp:133`, `time_integrator.cpp:106`. The
  preprocessor form appears only in pgens, as a whole-file `#error` guard.
- Because both branches of a runtime `if` are compiled, MC types may be named freely in
  gated code. `main.cpp:324` declares `MonteCarlo *pmc;` in plain scope and constructs it
  inside the gate. The dead branch is optimized away when MC is off.

Together these mean the entire reader lives in `src/monte_carlo/`, and the footprint
outside it collapses to two small gated hooks. `mesh.cpp:144` is a direct precedent: an
`if (MONTE_CARLO_ENABLED)` block *inside the Mesh constructor* that reads MC-specific input
and adjusts mesh setup, with a comment about enforcing mesh refinement consistency.

One deliberate departure from the earlier sketch of this work: the HDF5 attribute and
integer-dataset readers do **not** go into `src/inputs/hdf5_reader.cpp`. They are needed
only by this feature, so they go in the new MC file with their own `#ifdef HDF5OUTPUT`
guard, mirroring how `hdf5_reader.cpp` guards itself. See §8 for the alternative.

### Where the code goes

| piece | file | in/out of module |
|---|---|---|
| `MCGridFile` — reads and validates the athdf header, owns `loclist`, `file_index` | `monte_carlo/mcgrid.cpp/.hpp` (new) | in |
| HDF5 attribute + int/int64 dataset helpers, `#ifdef HDF5OUTPUT` | `monte_carlo/mcgrid.cpp` | in |
| `MCGridFile::Requested(pin)`, `MCGridFile::InjectMeshParameters(pin)` | `monte_carlo/mcgrid.cpp` | in |
| gated call to inject `<mesh>`/`<meshblock>` params before `new Mesh(...)` | `main.cpp` ~line 270 | **out, gated** |
| gated branch replacing the `<refinement>` loop with the `loclist` replay | `mesh.cpp` ~383-519 | **out, gated** |
| read cell data by `file_index[gid]` instead of `gid` | `pgen/mc_readhdf.cpp`, `pgen/mc_readhdf_gr.cpp` | pgen, already MC-only |

---

## 6. Phased plan

### Phase A — `MCGridFile`, header reading and validation (~250 lines, in module)

New `src/monte_carlo/mcgrid.{hpp,cpp}`. A small class holding `mesh_size`, `block_size`,
`nbtotal`, `root_level`, `max_level`, `loclist[]`, `file_index[]` and the `Coordinates`
string, plus the HDF5 helpers to fill it. All HDF5 code inside `#ifdef HDF5OUTPUT`; the
non-HDF5 build compiles to a stub that `ATHENA_ERROR`s if the feature is requested.
HDF5 converts the BE on-disk types, so pass `H5T_NATIVE_INT` / `H5T_NATIVE_INT64`.

Validation is most of the value here. All of it fails loudly rather than silently
producing a wrong grid:

- `RootGridSize % MeshBlockSize == 0`; catches `ghost_zones=true` outputs where
  `MeshBlockSize` includes ghosts.
- Reject sliced and summed outputs. These break the file-index/gid identity at
  `athena_hdf5.cpp:311-318`, where `first_block` is recounted over active blocks only.
- Reject files missing the §4 `mesh_data` attributes, and integer-quantized `data_format`.
- `Coordinates` against `COORDINATE_SYSTEM`; abort on mismatch.
- Reconstructed faces against the stored `x1f/x2f/x3f`, per block, at float tolerance.
  This is the check that catches a user mesh generator (§3).
- Block-size floor against `NGHOST`/`xorder` when the tree is multilevel (§3).

Build `file_index[]` by matching `(level,lx1,lx2,lx3)` to the post-`GetMeshBlockList`
`loclist` rather than assuming file order equals gid order.

### Phase B — parameter injection (~10 lines out of module, gated)

The `Mesh` constructor reads `<mesh>` in its member-initializer list, before the body runs,
so injection must happen earlier. In `main.cpp`, immediately before
`pmesh = new Mesh(pinput, mesh_flag)`:

```cpp
if (MONTE_CARLO_ENABLED) {
  if (MCGridFile::Requested(pinput)) MCGridFile::InjectMeshParameters(pinput);
}
```

which `pin->SetInteger`/`SetReal`s the twelve `<mesh>`/`<meshblock>` values and sets
`refinement`. **Uniform snapshots are fully automatic at the end of this phase, with zero
changes to `Mesh`** — worth landing and testing on its own.

### Phase C — tree replay (~20 lines out of module, gated)

One gated branch in the `Mesh` constructor selecting between the existing `<refinement>`
loop (`mesh.cpp:383-519`) and the three-line replay from §2, with `adaptive` left false so
`max_level = current_level` at `mesh.cpp:520` and no refinement happens during the MC run.
Shaped like the `mesh.cpp:144` precedent. This is where SMR and AMR start working.

### Phase D — pgen (~5 lines, already MC-only)

Swap `start_cons_file[1] = gid` for `file_index[gid]`, and drop the hand-maintained grid
parameters from the MC athinputs.

### Phase E — photon transport across refinement boundaries

**Status: transport across level jumps tested and found correct.  Nothing else in this
phase has been tested.**

The machinery is level-aware: `Photon::SendToNeighbors` calls
`Particles::FindTargetNeighbor`, which walks the neighbor list to pick the right fine
child on a coarse-to-fine crossing, and `Photon::GetPositionIndices` recomputes cell
indices from the photon's position against the *receiving* block, on both the local
(`photon.cpp:380`) and the MPI (`photon.cpp:650`) path.

`tst/montecarlo/amr_transport/run_tests.py` exercises it: a uniform medium of fixed
optical depth on three meshes that differ only in refinement, where the escaping fraction
cannot depend on the mesh.  200000 photons, sigma = 206:

| mesh | nesc | delta |
|---|---|---|
| uniform, 8 blocks | 61628 | - |
| one level refined **above** the interface | 61259 | -1.8 sigma |
| one level refined **below** the interface | 61361 | -1.3 sigma |

### The trap this test fell into first

The same comparison run **without** `<montecarlo>/equal_weight = true` shows +15 and -16
sigma deviations, and looks exactly like a transport bug at the interface: antisymmetric,
reproducible, and surviving controls for resolution, for the problem generator, and for
geometry (a transparent medium agrees to 0.6 sigma, so photons do cross level jumps and
land in the right place).

It is not a bug.  Under the default variable-weight sampling, `montecarlo.cpp:958` gives
every active block an equal share of the photons, `1/nb_active`, whatever its volume.  The
emitted *energy* per block is still right, because `ComputeEmissionArray` folds the cell
volume into the per-cell emission array and the photon weight is drawn from it -- but a
refined block emits as many photons as a coarse one while carrying an eighth of the
energy.  So `nesc/ntot` is a photon-count fraction, not an energy fraction, and it is
simply not mesh-invariant.  With `equal_weight = true` every photon carries `em_tot/ntot`,
counts track energy, and the discrepancy vanishes.

Two things follow.  Any mesh-convergence check on this code has to be energy-weighted or
run at equal weight; comparing photon counts across different meshes measures the sampling
scheme, not the physics.  And the variable-weight scheme deliberately spends photons per
block rather than per unit energy, which is a variance-reduction choice, not an accident.

### The variable-weight scheme checked directly, in energy

The default scheme was then checked on its own terms, integrating the escaping spectrum
rather than counting photons, on the stratified atmosphere.  1e6 photons:

| mesh | E_escape vs uniform |
|---|---|
| SMR, fine below the interface | -0.46% (-0.2 sigma) |
| SMR, fine above the interface | -0.89% (-0.2 sigma, mean of 3 seeds) |
| uniform at 2x resolution | -1.64% (-0.7 sigma) |

The photon counts for the same three runs are 238785, 374168 and 105138 -- a 57% spread
that collapses to under a percent once measured in energy.

**Refining costs precision where the emission is not.**  The `fine above` configuration
scatters by 6.1% from seed to seed, against 1.3% for the uniform mesh: refining the upper
half turns 4 coarse blocks and 32 fine ones, so the dense emitting base falls from half the
photons to an eleventh, and the few photons it does get carry correspondingly large
weights.  Rare deep photons that escape then dominate the variance.  For an atmosphere
whose emission is dominated by its base, refining the optically thin region above it is
statistically counterproductive under this sampling scheme.  That is a property of the
scheme, not a bug, but it is worth knowing before refining a disk atmosphere and wondering
why the spectrum got noisier.

A methodological note, learned the hard way twice here.  Re-running with more photons but
the **same seed** does not give an independent realization: the two runs share an RNG
stream and are strongly correlated, so a deviation that fails to shrink with N proves
nothing about whether it is a bias.  Only changing the seed does, and the code's own
reported error bars were, if anything, optimistic compared with the seed-to-seed scatter.

### Moment arrays on a refined mesh

Checked with the lab-frame moments (`<output3> variable = mclab`) on the uniform medium,
comparing the horizontally averaged `Ermc(z)` from an SMR run against a uniform one.  1e6
photons, two seeds:

- **`E_r` is continuous across the interface.**  The last coarse layer below z = 5e10 and
  the first fine layer above it read 7088 and 7185, on a profile whose neighbouring layers
  differ by about that much anyway.  A level-dependent error in the `1/(tint*vol)`
  normalization would show as a factor of two or eight here, and there is none.
- **The profiles agree within run-to-run noise.**  Two uniform runs that differ only in
  seed already scatter by 2.6% layer to layer, and the SMR-versus-uniform offsets sit
  inside that: coarse region 1.014 and 1.017 on the two seeds, fine region 0.973 and 1.036
  -- the fine-region offset changes sign with the seed, so it is noise, not a bias.

This constrains a systematic error in the moments on a refined mesh to roughly the couple
of percent that two seeds can resolve.  It is not a precision validation, and it would not
catch a sub-percent bias.

### Ghost zones at fine/coarse interfaces

This one matters because the opacity is evaluated at the photon's own cell
(`opacity.cpp:48`), and `Photon::GetPositionIndices` allows that index to be `is-1` or
`ie+1` -- a photon can sit in the first ghost cell, and at a level jump that cell's density
came from prolongation or restriction.  For a snapshot-driven run it matters more still:
the problem generator fills only active zones, so every ghost cell at a level jump is
interpolated file data.

Tested directly rather than through photon statistics: dump `prim` with
`<output3> ghost_zones = 1` on a refined mesh carrying the stratified profile, which spans
seven decades in density, and compare the ghost cells against `DensityProfile` evaluated
analytically.  Deterministic, one run, no Monte Carlo noise.  Note that the earlier
constant-density tests could not have caught anything here: prolongation of a uniform field
is exact.

| quantity | measured | expected |
|---|---|---|
| active cells | 8.9e-8 | exact, at float32 output precision |
| restriction into coarse ghosts | 4.957e-4 | 4.955e-4, `(dz/4 l0)^2 / 2` |
| prolongation into fine ghosts | 2.0e-3 | order `(dz/l0)^2 / 8` = 2.0e-3 |
| convergence at 2x resolution | 3.90x, 4.01x | 4.00x, second order |

The restriction figure agrees with the analytic prediction to four significant figures,
which is what it should be: restriction produces a cell *average* and the analytic profile
gives a point value at the cell centre, and the difference between them is exactly
`(dz/4 l0)^2 / 2` for an exponential.  Both operators converge at second order.  There are
no zeros, no uninitialized ghosts and no one-cell offsets.  **Prolongation and restriction
at level jumps are correct.**

Practically, the residual is a 0.2% density error in the single ghost layer a photon can
occupy, at the coarsest resolution tested, and it converges away as `dz^2`.

### The scattering moments, and the J_nu / E_r identity

`moments_scat` is meant to be a monochromatic mean intensity, so that
`sum_n J_nu(n) dnu_n = c E_r / (4 pi)` in every cell.  Checked directly, cell by cell,
with `mclab` and `mcscat` dumped from the same run of the uniform medium:

| mesh | cells | ratio of the two sides |
|---|---|---|
| uniform | 524288 | 1.00000 +- 0.00000 |
| refined, level 0 | 262144 | 1.000000 +- 4.7e-8 |
| refined, level 1 | 2097152 | 1.000000 +- 5.6e-8 |

The identity is exact to the single precision the athdf is written in, on coarse and fine
blocks alike.  So the `c/(4 pi nu dloge ln10)` binning factor and the `1/(tint vol)`
normalization are mutually consistent, and neither picks up a level dependence.

**Integrate with the same dnu the estimator assumes.**  The code divides by
`nu_mid * ln(10) * dloge`, not by the exact bin width, and
`sourceterm_frequencies.txt` reports the exact edges `nu_lo` and `nu_hi`.  Using those
edges instead gives a uniform +0.19% offset with *zero* scatter -- which is exactly
`(ln10 * dloge)^2 / 24 = 1.94e-3` for the 128 bins over 12 decades used here.  It is the
log-midpoint approximation, not an error, but it will show up in any hand-rolled
integration of the output and it grows as `dloge^2`.

Two limits on what this establishes.  It is an *internal consistency* check: both sides
are accumulated from the same `w e dl / c` weight, so a common factor wrong in both would
still pass.  And `mc_isoth` is Cartesian and non-relativistic, so this exercises the flat
`else` branch of the estimator -- **not** the GR branch that carried the
`Coordinate4Vector()[IMC0]` bug fixed in b69ebf3a.

### The GR branch, after b69ebf3a

Checked with `mc_isoth_gr` (Kerr-Schild, a = 0, a thin shell at r = 5.95 to 6 M,
`general_pusher` and `boosts` on, so the `GRTetrad() && boosts` branch is the one taken),
1e6 photons, 8192 cells:

| `sum_n J_nu dnu_n` compared against | ratio |
|---|---|
| **comoving** `E_r` (`mccom`) | **1.00055 +- 0.00495** |
| lab `E_r` (`mclab`) | 0.858 +- 0.039 |

The fixed GR estimator satisfies the identity to 0.06%, inside the 0.5% cell-to-cell
scatter.  The 14% miss against the lab moments is not an error: that branch bins and
weights comoving quantities, so the identity is against `mccom` and only against `mccom`.
Comparing scattering moments to `mclab` in a relativistic run will look wrong by roughly
the shift factor, and at r = 6M with a = 0 that is about what is seen.

### A crash found on the way: mccom output alone segfaults

Requesting comoving moments **without also requesting the lab moments** dereferences an
unallocated array:

- `moments` (lab) is allocated only under `if (mom_flag_lab)`, `montecarloblock.cpp:383`,
  and accumulated only under the same flag, `montecarloblock.cpp:891`.
- `accumulate_comoving` defaults to false, `montecarloblock.cpp:74`.
- so `montecarloblock.cpp:1118` calls `DeriveComovingMoments()`, which derives the comoving
  moments *by transforming the lab ones* and reads `moments(...)` at
  `photon_frames.cpp:246`.

With `<output> variable = mccom` and no `mclab` output, that array does not exist.  The
flat case fails earlier with a clean "comoving frame moments requested but boosts set to
false", so the crash needs `boosts = true`, which every relativistic run has.  Workaround:
request `mclab` alongside `mccom`.  Fix: derive a flag
`need_lab = mom_flag_lab || (mom_flag_com && !accumulate_com)` and use it for both the
allocation and the accumulation gate.  Not done here -- it is unrelated to the athdf work
and touches a file outside `monte_carlo/`'s recent churn.

### Still untested

- Anything at more than one level of refinement, or with MPI ranks split across a level
  jump.
- The comoving and coordinate moment arrays compared *across meshes*.  Only `mclab` was.
- `mcscat` on a refined mesh in GR.  The identity was checked on a refined mesh in flat
  space and in GR on a uniform mesh, but not both at once.


**Budget most of the schedule here.** The grid reading is mechanical; this is the part that
is genuinely unknown.

The neighbor machinery is already level-aware and `Photon` inherits all of it:
`Particles::LinkNeighbors` has a `multilevel` branch filling in missing fine-to-coarse
directions from the tree (`particles.cpp:513-540`), and `FindTargetNeighbor` walks the
neighbor list to pick the right fine block on coarse-to-fine (`particles.cpp:1045-1066`).
But nothing in `src/monte_carlo/` mentions `level` at all, so **none of this has ever been
exercised with photons**, and the pushers' cell-index arithmetic at a level jump is
unverified.

Also in this phase: with `multilevel` true the blocks allocate coarse buffers and
`Mesh::Initialize` runs prolongation/restriction to fill ghost zones. The pgens set
`phydro->w` on active zones only, so ghost values at a fine/coarse interface come from
prolongation. Confirm that actually runs on the MC path before trusting opacities near a
level jump.

Suggested tests, smallest first: a two-level static box with a known analytic solution
where photons must cross the interface in both directions; then the same with MPI and the
blocks split across ranks; then energy conservation of the accumulated moments across the
jump.

---

## 7. Open decisions

- **Input parameter naming.** Something like `<montecarlo> grid_from_file = <path>`, or
  reuse the existing `<problem> input_filename` that the MC pgens already take. The latter
  avoids naming the same file twice but couples grid setup to a `<problem>` key.
- **Whether to also snap `x1rat`.** With single-precision attributes we could round a
  near-rational ratio to a clean value, or leave it and accept float-level face offsets.
  Leaving it is simpler and the face check bounds the error; revisit only if it bites.
- **How hard to fail on a `Coordinates` mismatch.** An error is right for a genuine
  mismatch, but `gr_user` snapshots may legitimately carry a name that differs from the MC
  build's. May need an override flag.

## 8. Alternatives considered

- **Use `.rst` instead.** A restart embeds the complete input deck via
  `pin->ParameterDump` (`restart.cpp:70`) *and* the exact tree at double precision, so it
  sidesteps Phases A-C entirely. Rejected as the primary path because restart files are
  architecture dependent (per §1) and because we frequently have athdf snapshots without
  matching restarts. Still the better route when a `.rst` is available on the same
  architecture, and worth documenting for users.
- **Put the HDF5 helpers in `src/inputs/hdf5_reader.cpp`.** More reusable and the natural
  home if this were ever upstreamed, but it adds ungated API surface outside the module for
  a feature only MC uses. Revisit if a non-MC caller appears.
- **A third `Mesh` constructor** taking a filename. Cleaner separation than a gated branch,
  but duplicates roughly 250 lines of the restart constructor's post-tree bookkeeping
  (load balance, block creation, `SearchAndSetNeighbors`) and puts a large ungated block of
  MC-motivated code in `mesh.cpp`. Rejected on both counts.

## 9. Notes for whoever picks this up

- `mcgrid.hpp` is a new header. Per the repo build rule, `make clean` before `make` after
  touching it — the Makefile tracks no header dependencies, and a partial rebuild after a
  layout change produces a silently mismatched binary.
- The local `amr_reader` branch is currently byte-identical to `main`; there is no
  work-in-progress there to build on.
