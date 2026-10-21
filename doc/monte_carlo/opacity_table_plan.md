# Table opacities as a module option: plan

**Status (2026-10-19): Phases 1 to 4 built; Phases 5 and 6 not started.** Work goes
on the `opacity_table` branch in the `athena-opacity` worktree. Each phase ends with its
tests passing and the status block here updated.

Phase 1 as built: `src/monte_carlo/opacity_table.hpp/.cpp` (`OpacityTable`, the
per-block build, `TableCell`), `TableAbsorptionOpacity`, `GetEmissionTable`,
`PhotonEmitTable`, the `EMISTABLE`/`ABSTABLE` flags, the keys `opacity_file`,
`table_groups = edge`, `table_offgrid`, `table_above = clamp`, `table_gray`,
`table_storage`, `table_fill_freefree`, `MonteCarloBlock::tab_mask`, and the run report
line. `mc_readhdf`, `mc_readhdf_gr` and `mc_isoth_opac` use the module (the readers keep
`GetNel`, the GR reader its `tcut` mask and escape tables on the module's accessors);
`mc_isoth_gr` and `mc_gr_simple`, which carried a dead copy of `mc_isoth_opac`'s table
code, are stripped too. The load-balancing refusal of the readers is gone, the tables
being rebuilt in `SetupBlockFromFluid`. Tests in `tst/montecarlo/opacity_table/`:
`port_equivalence.py` gives byte-identical photon lists and moment files against a
binary built from `main` for `mc_readhdf` on the small snapshot (Thomson, none and
Compton scattering) and for `mc_readhdf_gr` on the synthetic Kerr-Schild torus of
`make_ks_snapshot.py` (plain, and `weights = biased` with the escape tables);
`storage_equivalence.py` gives identical lists between `cell` and `global` storage for
every configuration of both readers and of `mc_isoth_opac` (table emission with
free-free absorption and the reverse included). Wall times of the two storage modes
were the same within noise on these small runs, the per-call interpolation being lost
in the transport cost; the hires XRB (Phase 5) is where the memory and the cost will be
measured. `inputs/mc/athinput.isoth_opac` is the reference deck. Found on the way and fixed:
`RouletteOrAbsorb` rolled the roulette on samples whose weight the absorption step had
taken to exactly zero (every sample at its first interaction under `scattering = none`
with weight absorption), so a tenth survived at zero weight, reached the no-scattering
handler and were counted destroyed with a warning each; a zero-weight sample is now
absorbed without a draw, which leaves the random stream of every scattering run
unchanged (the Thomson and Compton equivalences still hold) and the `none` case no
longer identical to the pre-module binary.

Phase 2 as built: `table_groups = mean` is the default (piecewise constant per group,
emission kappa_g times the photon Planck integral over the group from a tabulated
cumulative, `PlanckPhotonCumulative`, energies within a group from B_nu/(h nu));
`table_above = freefree|clamp|none` and `table_emax` (keV) with the free-free extension
groups above the table; the uniform axes are detected and reported, and the (T, rho)
bracket comes from the uniform-spacing guess corrected by a walk, so a non-uniform
temperature grid needs no separate path. The emission, like the free-free one, is a
photon rate (the weights are photon numbers). Tool: `make_ff_table.py` (exact Planck
and Rosseland group means of the module's free-free opacity with Thomson scattering,
plus the N_free file). Tests: `ff_table.py` on a 60 x 31 free-free table over five
temperatures (1e4 to 1e8 K) and three densities (1e-12 to 1e-6, snapped to interior
nodes): the mean reading's photon rate agrees with the analytic integral to 1e-6 at
every point, every group energy is within the sampling noise of the free-free integral,
and the closed box's energy density and window Planck mean are within 3.4% of Planck;
the edge reading's photon rate agrees with its own trapezoid integral to 1e-6 and its
window Planck mean comes out 0.60 to 0.65 of the true one (the F1 deficit), and at
1e8 K, where kT sits at its top node, it emits 0.6 of the energy. With `--emax-ext 100`
the extension (41 groups to 133 keV) passes the same checks at 1e7 and 1e8 K.
The same table on the old 69-row non-uniform temperature grid passes at three
temperatures. `ff_atmosphere.py` (Thomson atmosphere to depth 100, table against
free-free, equal-weight sampling) agrees in total luminosity within its sampling error
at 1e5, 1e6 and 1e7 K and in 31 or 32 of 32 groups within the two-stream estimate of
the discretization plus 3%. `storage_equivalence.py` holds `cell` and `global`
byte-identical for the mean reading with and without the extension, on both readers
(biased weights included) and the atmosphere generator. Two things the tests taught: the closed box's energy density
under analog Planck sampling has a heavy tail (a Wien-tail photon's contribution grows
as E^4 at an E^-3 opacity while the counts fall as e^-x), so it needs a million photons
for a few percent, whereas the free-free sampler's flat-in-ln E draw has no such tail;
and the free-free sampler's window must stop near 60 kT, since samples of no weight at
hundreds of kT walk astronomically far in a closed box.

Phase 3 as built: `nfree_file` (read by `OpacityTable::ReadNFree`, grids checked
against the main table's, Abar from the header) and `table_ne`; `ElectronDensity(T, rho)`
is bilinear in the cell's bracket; the default `GetNumberDensity` and the readers'
`GetNel` use it when `table_ne` is on, the ion density stays that of the ionized
mixture, and the free-free fallback of off-grid cells keeps full ionization (a
departure from the plan's wording, deliberate: those cells are hot or empty, and the
fallback has no cell state). `mc_isoth_opac` gained the `nelJ` user moment, whose ratio
to Er is the cell's n_e exactly. Test `ne_table.py` on the solar TOPS table and its
nfree file at five (T, rho) points on and between nodes, where N_free runs from 0.77 to
1.10: the run's n_e matches rho N_free / (Abar m_u) to 1e-5 (the opacity table's
four-digit temperature grid sets the bracket) and the run without `table_ne` the
ionized H/He value to 1e-7; on the free-free table's nfree file (full ionization) the
two agree. `storage_equivalence.py` has a `mean_ne` configuration.

Phase 4 as built (2026-10-19): `two_band`, `two_band_w`, `two_band_floor`,
`two_band_closure`, `two_band_thick`; `OpacityTable::Split` (the handoff's closed form),
`Bands`, the Rosseland tables and the optional Rosseland absorption block; per-cell
`tab_abs_hi`, `tab_abs_lo`, `tab_w`, `tab_thick` in cell storage and the same per call in
global storage; `Photon::ubp` (band rank) and `grpp` (group of the last absorption);
the redraw after every scattering (the absorption is refreshed after a coherent
scattering too) and on a change of group; the band at emission; the thermalized high
band of 3.5 (emission to the low band with the scattering leakage, the high band's own
value absorbing on the spot); the random-walk step declines; the split's moments are
checked at every build; the report counts absorbed samples that died in their first
step. `tops_regrid.py` is copied into `vis/python/montecarlo/problem_specific/opacity_table/`
with `--rossabs` (a further nfre blocks after the Planck means) and `--digits`; the
reader takes the block when present. On the TOPS table the block and the subtraction
agree to 1e-3 in the median but differ by more than 1% in a third of the two-band
(group, T, rho) cells and disagree on one band or two in 11%, which is the
ill-conditioning the handoff foresaw, so tables for two-band runs should be written
with `--rossabs`. `mc_isoth_opac` gained `tgrad`. Tools: `make_picket_table.py`
(every group the same two bands, exact targets). Tests: `twoband_box.py` on the TOPS
table at four line-dominated cells, where the emission rate under the split and under
the thermalized high band matches the analytic integral to 1e-4 and the group energies
the sampling noise, and the thermalized run's closed box is Planck in the transported
field; `rosseland_slab.py` on the picket table (kappa_h = 300, kappa_l = 1, w = 0.01,
Thomson 0.35, a slab ten depths across with T rising by half), where the single-band
run carries 1.08 +- 0.03 of the Planck-only flux (0.34 of the Rosseland one), the
two-band run 0.98 +- 0.02 of the Rosseland flux and the thermalized run 0.96 +- 0.02, with
76% and 10% of the absorbed samples dying in their first step; `storage_equivalence.py`
configurations `mean_twoband` and `mean_twoband_thick`, byte-identical. What the runs showed, in the handoff's example
cell (2e4 K, 1e-15): with the split alone 100% of the absorbed samples die in their
first step, and the equilibrium field rests on no sample at all; with the high band
thermalized the field is Planck to 0.4% at a million photons and 8% of the samples die
in place (the ones scattered back into the high band). With `abs_method = weight` and
the default `minweight` a sample in a scattering-dominated low band is followed through
hundreds of scatterings at no weight, so the tests use `prob`. Not built: the
two-band Feautrier atmosphere (a picket table and the slab give sharper checks of the
Rosseland limit, and the free-free atmosphere test covers the scattering coupling of a
single band), the `two_band_decorrelate` knob, and the closure sensitivity run (the
picket table recovers the split exactly for any w0; the sensitivity question belongs
to the TOPS science runs of Phase 5). Decisions taken (2026-10-18): the present `mc_readhdf*` reading is
the `edge` mode and `mc_isoth_opac`'s variant is retired untested; `table_ne` is off by
default; the thermalized high band is built in Phase 4; the regrid script is changed to
write the Rosseland absorption and checked; both storage modes of 3.6 are built.

## 1. Goal

Take the tabulated (TOPS multigroup) emission and absorption out of the problem
generators and make them a module option, `emission = table` and `absorption = table`,
beside `freefree`. Then, on that shared path:

1. read tables that are uniform in log T and log rho in O(1), keeping the bisection
   for the old non-uniform temperature grid;
2. set the electron density from the table's free electrons per atom where the gas is
   partially ionized;
3. add the two-band (picket-fence) split of each group so that emission uses the Planck
   mean and diffusion the Rosseland mean without breaking Kirchhoff's law.

The tables come from `/home/swd8g/opacity_table/` (`tops_regrid.py`; main table, `_nfree`
and `_twoband` files; `MC_TWOBAND_HANDOFF.md`). The generators affected are `mc_readhdf`,
`mc_readhdf_gr` and `mc_isoth_opac`; the last is the test bed.

## 2. What exists, and what the survey found

Three generators each carry their own copy of the table machinery: a reader for the
text table into file-scope arrays, a per-cell build of the extinction per group
(`opact`), an emissivity integral per cell (`emis_tot`, `emis_cum`), a `TableOpacity`
user opacity, a `TableEmission` user emission and a `SampleEmissivity` called from the
generator's own `InitializePhoton`. `mc_readhdf_gr` adds the escape tables for biased
sampling on top of `opact`, the `tcut`/`dcut` cuts and the `GetNel` ionization heuristic.

Findings that the port has to settle, not just move:

- **F1. Group means are used as point values at the group's lower edge.** The table's
  k-th value is the Planck mean over [E_k, E_{k+1}). `mc_readhdf*` put it at E_k and
  interpolate linearly in log E to the next value; `mc_isoth_opac` puts it at the upper
  edge of the interval below (it prepends an extrapolated edge), so the two generators
  disagree with each other by one group. For a nu^-3 opacity the Planck group mean is
  0.67 to 0.73 of the value at the lower edge (checked numerically for a 1.3335 ratio),
  so the `mc_readhdf*` reading under-absorbs by about 30% everywhere free-free dominates,
  and the top group (7.5 to 10 keV) is never emitted because the nodes end at its lower
  edge. The consistent reading is piecewise constant: kappa_g over its group, emissivity
  kappa_P,g B_g with B_g integrated exactly, and the energy within a group drawn from
  B_nu. The module gets both: `table_groups = edge` reproduces the present `mc_readhdf*`
  numbers for the equivalence test, `mean` is the default.
- **F2. Gray-row repair.** Rows of the old tables where every group holds the same
  value (TOPS's clamped edge cells, see the opacity_table README) are detected by a
  max/min ratio and replaced by analytic free-free (`mc_readhdf*`) or by 1e-60
  (`mc_isoth_opac`). The regridded tables no longer have such rows. Kept as
  `table_gray = freefree|keep`, default `keep`, with the count still reported.
- **F3. Tables are built once.** `MonteCarloProblemGenerator` runs at startup and on a
  block's arrival after a redistribution, so in a dynamic run the per-cell tables never
  follow the fluid. The module builds them in `SetupBlockFromFluid`, with the rest of
  the fluid-derived arrays, which also makes the per-lid arrays and the load-balancing
  refusal in the readers go away.
- **F4. Ionization.** `GetNel` sets n_e to zero where the grey Rosseland mean falls
  below 0.34 and T < 1e5 K, else to full ionization: a step. The `_nfree` table gives
  N_free(T, rho) continuously; with it n_e = rho N_free / (Abar m_u) with Abar from its
  header.
- **F5. Composition.** The code's free-free and default n_e assume hydrogen and helium
  with `heabund` = 0.09 per hydrogen nucleus (n_e/rho = 0.868/m_u fully ionized). The
  TOPS tables are solar (Z_full/Abar = 0.845/m_u). Setting n_e from the table changes
  the Thomson depth of fully ionized gas by 2.7% relative to today, and the
  temperature inversion in `GetTemperature` uses the ionized H/He mean weight at every
  temperature. Neither is wrong for this project, but the free-free test tables must be
  generated with the code's composition, and the choice of n_e source has to be the
  user's (`table_ne`, default off).
- **F6. Energy range.** Photon energies outside the table are clamped to the end groups.
  Above 10 keV in gas at 1e8 K that over-absorbs by the ratio of the clamp to the
  nu^-3 tail, and emission above 10 keV is dropped altogether, which the free-free path
  never did. `table_above = clamp|freefree|none` for absorption; `freefree` also adds
  extension groups for emission up to `emax`.
- **F7. Memory.** `opact` is nfre Reals per cell (256 B at 32 groups), on the active
  cells only, a choice `mc_readhdf_gr` made for the hires mesh. The two-band split
  needs three per group. Keep active cells only, and put the storage behind one
  accessor with two modes (3.6): per-cell arrays, or one table per rank interpolated at
  every call. The XRB's memory budget (`memory_and_speed_plan.md`) is what the second
  mode is for.

## 3. Design

### 3.1 Files and classes

- `src/monte_carlo/opacity_table.hpp/.cpp`: `class OpacityTable`, one per rank,
  owned by `MonteCarlo`. Holds the group edges (nfre+1, the top edge continued
  geometrically as the tables assume), the T and rho grids, `uniform_t`/`uniform_rho`
  flags set by checking the log spacing to 1e-6, kappa_P (nfre, nT, nrho), kappa_R
  (same), the grey means, and N_free (nT, nrho) with Abar if an `nfree_file` is given.
  Methods: `Locate(T, rho, idx, wgt, on_grid)` (O(1) on a uniform axis, bisection
  otherwise; clamps and reports off-grid), `Planck(g, idx, wgt)` and `Rosseland(...)`
  (log-log bilinear, linear where a corner is non-positive, as `mc_readhdf*` do now),
  `NFree(...)`, and the free-free fallback for off-grid cells using the block's own
  n_e and n_ion.
- `MonteCarloBlock` gains `BuildOpacityTables()`, called from
  `MonteCarlo::SetupBlockFromFluid` after `GetNumberDensity`, so dynamic runs, restarts
  and redistributions all rebuild. It always stores per cell the table bracket and
  weights of the cell's (T, rho) (`tab_loc`: two ints and two Reals) and the emission
  total `tab_emis_tot`; in `cell` storage mode (3.6) it also stores `tab_abs` (ncell,
  nfre), or (ncell, nfre, 3) with the split under two-band, and `tab_emis_cum` (ncell,
  nfre). Every read of an opacity or a group probability goes through one inline
  accessor, `TableKappa(cell, g)` and friends, which in `cell` mode returns the stored
  value and in `global` mode calls `OpacityTable::Planck(g, loc)` with the cell's
  cached bracket. A per-cell `tab_mask` (like `accel_mask`) lets a generator exclude
  cells: `mc_readhdf_gr`'s `tcut`/`dcut`/horizon cells get `CUT_VALUE`, as now.
- `opacity.cpp`: `TableAbsorptionOpacity(pmcb, pphot, ip)`, the group of the photon's
  energy (O(1) in log E, the group grid is uniform) and the cell's value, or the band's
  value under two-band.
- `emission.cpp`: `GetEmissionTable(pmcb, k, j, i, etype)` returning `tab_emis_tot`, and
  `PhotonEmitTable(pmcb, pphot, ip)`: group from `tab_emis_cum` (with the biased
  energy-group branch of `mc_readhdf_gr::SampleEmissivity`, so `nescape = nfre` keeps
  working), energy within the group from B_nu through a shared cumulative Planck table
  in x = E/kT (also what gives B_g exactly, no polylog at run time), isotropic direction;
  under `table_groups = edge` the present trapezoid-in-ln-nu cumulative and log-linear
  draw instead. Generators call it from `InitializePhoton` as they call
  `PhotonEmitFreeFree`.
- `montecarlo.hpp/.cpp`: `EMISTABLE`, `ABSTABLE` in the flag enums and parsers; the
  input keys of 3.2; the end-of-run report (cells from the table, off grid by axis, cut,
  gray rows) moved from the readers' `UserWorkAfterLoop`.
- `GetNumberDensity`: with `table_ne` on and no user hook, species(0) from N_free and
  species(1) as now. The user hook still overrides (the readers keep their horizon and
  cut handling by zeroing rho in the hook as today, or by the mask).

### 3.2 Input keys (`<montecarlo>`)

| key | default | meaning |
|---|---|---|
| `emission = table`, `absorption = table` | | the shared path |
| `opacity_file` | required | main table (old or regridded layout) |
| `nfree_file` | none | free electrons per atom; required by `table_ne` and `two_band` |
| `table_storage` | `cell` | `cell`: per-cell arrays built once per transport; `global`: one table per rank, interpolated at every call (3.6) |
| `table_groups` | `mean` | `mean`: piecewise constant per group; `edge`: the present `mc_readhdf*` reading |
| `table_offgrid` | `freefree` | off the (T, rho) grid: analytic free-free or clamp to the edge |
| `table_above` | `freefree` | above the top group: free-free tail (and extension groups for emission), clamp, or none |
| `table_gray` | `keep` | gray-row replacement of the old tables |
| `table_ne` | false | electron density from N_free |
| `two_band` | false | the split of 3.4 |
| `two_band_w`, `two_band_floor`, `two_band_closure` | 0.01, 0.003, `w` | its closure |
| `two_band_thick` | 0 (off) | cell optical depth in the high band above which it is thermalized analytically (3.5) |

`<problem> opacity_filename` is read as a fallback for existing athinput files, with a
deprecation line.

### 3.3 Uniform and non-uniform grids

Same file layout for both (the regrid script kept it). The reader measures the spacing
of each axis; a uniform axis indexes by `(log x - log x0)/dlog`, the old 69-row
temperature grid (spacing 0.02 to 0.125 dex) by bisection. Interpolation is log-log in
both cases. A test writes the same free-free table on both kinds of grid and checks the
two paths agree to round-off at the nodes and to the interpolation error between them.

### 3.4 Two-band split

Per cell and group at build time, from the interpolated kappa_P,g and kappa_R,g and the
scattering the code will actually use in that cell, s_g = sigma(E_g, T) n_e / rho
(Thomson, or the thermal Klein-Nishina table of `ComptonOpacity` at the group centre):
the closed-form solve of the handoff Section 4 (ported from `tops_regrid.two_band`,
checked on the moments). Stored per cell: w_g, kappa_h,g, kappa_l,g (three Reals per
group; single-band cells store w = 1 and kappa_P twice).

Photon state: one new Real column `Photon::ubp`, the band rank u in [0, 1), packed
with the others (layout change, `make clean`), and one int column `igp`, the comoving
group at the last opacity evaluation. Rules:

- emission: group from kappa_P,g B_g as before, then high band with probability
  w kappa_h / kappa_P and u uniform in [0, w), else u in [w, 1);
- absorption: kappa_h if u < w(cell) else kappa_l, at the current cell's split;
- after every electron scattering: u redrawn uniform in [0, 1);
- whenever the comoving group changes between two opacity evaluations (Compton energy
  change, Doppler or gravitational shift along a flight): u redrawn;
- `two_band_decorrelate` (test knob, default 0): probability of redrawing u at a cell
  crossing, to measure how much the correlated-k assumption carries (Section 5, C2).

### 3.5 Thermalized high band (the addition Section 5 argues for)

In a cell whose depth across its smallest width in the high band, rho kappa_h W,
exceeds `two_band_thick`, the high band is in LTE with the gas and is not transported:

- its emission w kappa_h rho B_g is dropped from the emission array, and in its place the
  low band receives the energy the high band would have passed to it by scattering,
  (1 - w) p_s / (1 - w p_s) times that, with p_s = s / (kappa_h + s), which is
  w (1 - w) s rho B_g to leading order;
- a photon that enters such a cell in the high band, or is scattered into it there, is
  absorbed on the spot (its energy deposited; the compensating emission is already in
  the source above);
- the gas source term is the difference of the two, as it is for every other process.

Detailed balance holds because the high band field is B_g: the rate low to high,
w (1 - w) s rho J_g,low, equals the rate high to low when J_low = B. What is lost is
the high band's share of the diffusive flux, w / (kappa_h + s) against
(1 - w) / (kappa_l + s), negligible by construction wherever the split is two-band.

### 3.6 Storage modes

`table_storage = cell` is the present scheme: the interpolated extinction per group and
the cumulative emission spectrum are stored per cell, so a lookup is one array read and
an emission draw one bisection. It costs nfre Reals per cell (three nfre under two-band)
and a rebuild of every cell on every transport.

`table_storage = global` stores nothing per cell beyond the cached bracket and weights
(`tab_loc`, 24 B) and the emission total. Each rank holds the one table, as log kappa
so that the log-log interpolation is four reads, four multiplies and one exp per value:

- absorption: the photon's group from its energy (O(1)), then `Planck(g, loc)`; under
  `edge` the two node values and the log-linear step in energy, the same arithmetic as
  `cell` mode does at build time; under two-band `Planck`, `Rosseland` and the cell's s,
  then the closed-form split (a square root), per call;
- emission: the group probabilities are computed per emitted photon, nfre interpolations
  and nfre group Planck integrals from the shared cumulative Planck table, then the
  cumulative draw. The same loop computes `tab_emis_tot` at build time, so the total a
  cell is allocated samples by and the spectrum its samples draw are the same numbers
  in both modes. At 32 groups this is of order a microsecond per photon, once per
  photon, against the many lookups of its flights.

The two modes compute the same quantities with the same functions at different times,
so they agree to round-off, and if the per-photon draw order is kept identical they
produce identical photon lists for the same seed. That is the acceptance test for the
mode, and it is what keeps the second mode from doubling the physics tests: the
physics runs in `cell` mode, and one script per phase re-runs its configurations in
`global` mode and compares the lists bitwise (Section 4). The run report prints the
transport time, so the same script records the speed cost of `global`; the XRB in
Phase 5 is where the memory saving and the cost are both measured on the production
mesh. The rebuild cost of `cell` mode in a dynamic run (nfre exps per cell per
transport) is the other number worth having.

Where `global` is expected to hurt: the general pusher's per-step opacity refresh in a
Compton run (energy changes every step, so the group changes and the lookup cannot be
skipped), and two-band, where the split is solved per call. If that cost matters, the
split's per-cell result can be cached under a third mode later; not planned now.

## 4. Phases and tests

Tests live in `tst/montecarlo/opacity_table/`; each is a script that writes its athinput,
runs the binary from `--workdir` and prints PASS/FAIL against stated tolerances, in
the style of `sphere_compton.py`. All but Phase 5 run on pella in minutes.

**Phase 1. Port the present scheme.** `OpacityTable`, the block arrays, the three
functions, `table_groups = edge`, `table_gray = freefree`, `table_offgrid = freefree`,
`table_above = clamp`; `mc_readhdf` and `mc_readhdf_gr` switched to it (they keep the
snapshot reading, cuts, horizon, escape tables and user columns); `mc_isoth_opac`
stripped of its table code. Test: `port_equivalence.py` runs `mc_readhdf` on the small
snapshot in `~/mc_test/read_hdf5` with the old binary (built from `main`) and the new
one on the same seed and compares photon lists; identical if the draw order is kept,
else the per-cell emission totals to round-off and the band luminosities within the
output scatter. `mc_isoth_opac`'s old reading is not reproduced (F1); it is validated
against free-free in Phase 2 instead. Both storage modes are built in this phase, and
`storage_equivalence.py` (shared by every later phase through a list of configurations)
runs the port configuration in `cell` and `global` mode on the same seed and requires
identical photon lists and emission arrays; it also prints the two transport times.

**Phase 2. Uniform grids and the group-mean reading.** O(1) lookup, `table_groups =
mean`, the cumulative Planck table, `table_above = freefree`. Tool:
`vis/python/montecarlo/problem_specific/opacity_table/make_ff_table.py`, which writes a
table in the regrid layout whose kappa_P,g and kappa_R,g are the exact Planck and
Rosseland group means of free-free with the code's H/He composition, on a uniform or
on the old non-uniform grid, plus the matching `_nfree` file (full ionization). Tests:
- `ff_table.py`: periodic uniform boxes with `mc_isoth_opac` on a grid of T (1e4 to
  1e8 K) and rho (1e-12 to 1e-6), table against `emission = freefree, absorption =
  freefree`: total emission equals sum_g 4 pi rho kappa_P,g B_g to 1e-3 (deterministic),
  the `kapJ` moment and the per-group spectrum agree within the group-mean discretization
  the script computes analytically, and `mean` beats `edge` by the 30% of F1;
- `ff_atmosphere.py`: the emergent spectrum of the isothermal atmosphere
  (`absorption_spectrum` setup) at three temperatures, table against free-free;
- the uniform/non-uniform agreement of 3.3;
- `storage_equivalence.py` on the `mean` reading, the free-free tail and both grids.

**Phase 3. Electron density from the table.** `table_ne`. Tests: species(0) against
rho N_free / (Abar m_u) at the nodes (unit), and a scattering slab whose escaping
fraction matches the Thomson depth from N_free at T = 1e4 K, where N_free is well below
full ionization; the free-free fallback and emission in off-grid cells use the same n_e;
`storage_equivalence.py` with `table_ne`.

**Phase 4. Two-band.** 3.4 then 3.5. Tests:
- `twoband_split.py`: a debug dump of every cell's split against
  `tops_regrid.two_band`, and the two moments reproduced to round-off (handoff Section 6.6);
- `equilibrium_box.py`: closed reflecting box at T = 2e4 K, rho = 1e-15 (line-dominated)
  with the TOPS table: energy density a T^4 over the table's window, spectrum B_nu per
  group, net source term zero within noise, single and two-band;
- `thin_cooling.py`: escaping box, luminosity sum_g 4 pi rho kappa_P,g B_g V, both modes;
- `twoband_atmosphere.py`: isothermal scattering atmosphere with two-band absorption
  against a two-band Feautrier solution (two coupled band equations sharing J_g, the
  deterministic counterpart of handoff Section 8; `feautrier.py` extended): the
  emergent I/B per band and total, the sqrt(epsilon) law in each band. This is the test
  that exercises the band redraw at scattering;
- `rosseland_slab.py`: `mc_isoth_opac` gains an optional linear temperature profile;
  flux through a slab thick in both bands against -(c / 3 rho kappa_R,g) dE_g/dx;
- closure sensitivity: w0 = 0.1, 0.01, 0.001 on the atmosphere; forced w = 1 reproduces
  Phase 2 statistically;
- 3.5: the atmosphere and box repeated with `two_band_thick = 10`, same answers, and the
  report's count of emissions absorbed in their birth cell as the cost measure;
- `storage_equivalence.py` with two-band on and off, and with the thermalized high
  band; the per-call split of `global` mode must give the stored split of `cell` mode
  bitwise, which holds if both call the same function on the same interpolated inputs;
- the regrid change of C4: `tops_regrid.py` (copied into
  `vis/python/montecarlo/problem_specific/opacity_table/` so it is versioned with the
  reader; the original in `~/opacity_table` is left alone) writes the Rosseland
  absorption as a fourth per-group block appended after the Planck block, which the old
  readers never reach; `regrid`'s own moment check and the hold-out are rerun, the
  reader's split from the absorption block is compared with the split from the total
  (identical away from the floor, cleaner near it), and the plan records whether the
  change is worth keeping.

**Phase 5. XRB.** On andes: `mc_readhdf_gr` with the old table through the new path
against the last table run's bands and against `d01_emis2`; then the regridded table,
`table_ne` and two-band as science runs; load balancing with tables now allowed, checked
first with `lb_test_repack` on the small snapshot. Both storage modes on the hires mesh:
`VmHWM` per rank and the transport time, which is the comparison 3.6 is for.

**Phase 6. Documentation.** CLAUDE.md (module paragraph, test list), this file's
status, and a short `opacity_table_guide.md` with the keys, the file layouts, and the
reasoning of Section 5.

## 5. The two-band handoff, re-examined

The construction is sound: two bands each obeying Kirchhoff reproduce both means, the
closed-form split is cheap, the redraw after electron scattering is justified (the
electron Doppler shift, 2e-3 in nu/nu at 2e4 K, is many line spacings), and the
group-change rule is right. The points below are what it leaves out or gets at risk.

- **C1. The high band is unsampleable in the regime the method is for.** In the handoff's
  example cell kappa_h = 1.4e6 and s = 0.33 cm^2/g, and nearly all the group's emission,
  w kappa_h / kappa_P, is in the high band. Each such photon is absorbed within
  1/(rho kappa_h) and scatters first with probability p_s = s / (kappa_h + s) = 2e-7;
  only that fraction of its weight ever reaches the low band, which is the only band
  that moves energy. With `abs_method = weight` the weight decays by the same factor
  per flight and is then rouletted. So the energy that actually leaves the cell is
  carried by one photon in 5e6, and the emission scheme spends its samples on photons
  that are born and die in place. Smaller w0 makes it worse (kappa_h ~ kappa_P / w0).
  This is not a defect of the physics, which is correct, but it decides whether the
  method is usable, and the handoff's validation list does not measure it. Section 3.5
  is the fix: where the high band is optically thick within the cell, thermalize it
  analytically and emit its scattering leakage straight into the low band. Build it
  in Phase 4, and have the Phase 4 report print the fraction of emitted energy absorbed
  in the birth cell so the plain scheme's cost is on record.
- **C2. Correlated k across cells is optimistic.** A band is "the same frequencies in
  every cell", but the lines that dominate a group change with ionization stage over a
  few tenths of a dex in T. Where they decorrelate, a gap photon in one cell meets lines
  in the next, so the truth lies between the correlated two-band answer and the
  Planck-only one, and the two-band scheme errs toward escape. The handoff does not
  say so. The `two_band_decorrelate` knob of 3.4 bounds the effect; if the atmosphere
  test moves a lot between 0 and 1, the group grid is too coarse for the method and a
  finer TOPS grid binned into bands per coarse group (handoff Section 5) is the remedy.
- **C3. The split must use the code's scattering, so it needs N_free.** The handoff says
  to use the code's own s (Section 4), and the code's s is sigma n_e / rho with n_e from
  species(0). With the default full ionization, s is wrong by orders of magnitude below
  1e4 K, exactly where the split matters, and the split with it. So `two_band` requires
  `table_ne` or at least an `nfree_file` for s. The `_twoband` file's own kappa_es
  column is then not used by the code at all; the code reads the main table and the
  `_nfree` file and splits at run time.
- **C4. Store the Rosseland absorption, not the total.** The split works on
  kappa_R - s, a difference of two numbers near 0.34 written to five digits; the handoff
  notes the ill-conditioning and floors it. Writing kappa_R,abs = kappa_R - kappa_es (as
  TOPS defines kappa_es) in the regrid output, in a third per-group block or in place
  of the total, removes the subtraction and the dependence on TOPS's scattering fit
  from the code side. A one-line change in `tops_regrid.py`; the reader accepts both.
- **C5. Fast flows.** Handoff Section 8.4 says it: where the velocity changes by more
  than a thermal line width along a path, a photon sweeps through lines and the static
  split fails; the TDE outflow at 0.1c is such a flow, and the right treatment there is
  an expansion opacity, which neither TOPS means nor the split provide. The two-band
  option is for static and slow regions (disk, atmosphere). The group-change redraw of
  3.4 handles the bulk shift's group crossings correctly but does not make the split
  right inside a group.
- **C6. The diffusion test needs a slab thick in both bands.** Section 6.3 of the
  handoff expects the Rosseland flux per group. That holds only where the low band is
  also in the diffusion limit; a slab thin in kappa_l gives a free-streaming low band
  and no Rosseland mean. The Phase 4 slab is sized on kappa_l.
- **C7. Dynamic coupling.** The code has no Fleck factor (handoff Section 3.5 does not
  apply). In a coupled run the high band's absorption and emission are each huge and
  cancel, so the held source term is a small difference of large noisy numbers and the
  cell's heating time can be shorter than the transport cadence. 3.5 removes both:
  the high band contributes no source term where it is thermalized.
- **C8. Interpolating w, kappa_h, kappa_l from the `_twoband` file** (the handoff's
  "acceptable first pass") loses the moments and ties the code to one closure and one
  scattering model. Not worth building; the run-time split is a few lines.

## 6. Other suggestions and open decisions

- **Legacy reading.** `mc_readhdf*`'s semantics are `table_groups = edge`, for the
  equivalence test only; `mc_isoth_opac`'s one-group shift is retired without a test
  of its own (decided).
- **Default n_e source.** `table_ne` off by default keeps every present run's Thomson
  depth (decided). For the TOPS tables the table's n_e is the consistent choice (F5);
  the athinput chooses.
- **Storage.** Both modes (decided); `cell` is the default because it is the present
  behavior and the faster one. The XRB numbers of Phase 5 decide whether `global`
  becomes the production setting there.
- **Composition for free-free tables.** `make_ff_table.py` uses `heabund`; the test
  compares like with like. The TOPS tables stay solar.
- **Temperature inversion.** Unchanged (ionized H/He mean weight). If snapshots from a
  hydro run with a different EOS are post-processed, `tgas_cgs` is the hook, as now.
- **Energy window for emission.** `table_above = freefree` so the hot XRB gas keeps its
  bremsstrahlung tail above 10 keV; below 1 eV nothing is added.
- **Biased sampling.** The escape table on `nescape = nfre` groups and the energy-group
  draw carry over unchanged; the per-band weights do not interact with the window.
- **MRW.** The step's free-free survival assumes the grey exp(-alpha c t) with
  coherent scattering and the Kompaneets table's lambda with Compton; with table
  absorption alpha is the group's (or the band's) value at the photon's energy, which
  the step reads through `AbsorptionOpacity` as now. Under two-band the walk is in one
  band until a scattering, which the step represents many of, so the step must decline
  in two-band cells (or redraw u and use the band-averaged alpha; decline first, measure
  later).
