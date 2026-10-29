# Discrete diffusion Monte Carlo for Compton and resonance scattering: plan

Status: **D0 and D1 done 2026-10-28 on branch `ddmc`; D2 next.**  The 2026-10-06
outline is folded in below with the changes of Section 2.  Order decided: the Compton
phases D0 to D3 first (they build the lattice, the interface and the block machinery that
both scatterings share, where an exact reference exists), the XRB phases D4 and D5 as a
separate decision after D3, and the resonance phases R0 to R3 from D3 on.  D0
(`ddmc_guide.tex`): the formulas derived with their units; ghost zones verified to hold
the neighbours' primitives across block faces and levels before the Monte Carlo arrays
are built; the per-cycle time budget found not to be a stop today (a photon that
exhausts `dtp` is scattered in place every pass).  The six decisions of the guide's
Section 14 were settled with the user the same day: both emergence distributions built;
level-jump faces are boundary faces (split-face rates as follow-up); threshold on the
smallest width, one key; periodic faces as boundary faces; the `ddmcp` column only when
enabled; and a held status for photons that exhaust the cycle budget is wanted as a
general feature of dynamic runs, outside the lattice's hook list, so it is a separate
item and not a phase here.  D1 (`src/monte_carlo/ddmc.cpp`, hooks in the Cartesian and
spherical-polar pushers; the guide's Section 13 lists where the build departs from the
design): grey Thomson scattering, free-free or grey absorption over the dwell, both
interface directions, the moments and force by the face-hop estimators, block-face hops
and conversions, the counters and the report line.  Tests, against the analog run of the
same deck (`tst/montecarlo/sphere_compton/ddmc_cube.py`, `sphere_compton.py --ddmc`,
`disk_atmosphere.py --ddmc`): the cube that fills the mesh reproduces the lattice's exact
first-passage solution in the mean path and the dwell count to the statistics at 8, 16
and 32 cells; the Thomson sphere agrees in the mean escape path to 0.5 percent at 16 and
32 cells across for cell depths 2 to 16 and to 2 to 3 percent at 8 cells across, the
distributions within the Kolmogorov scale; absorption agrees under both methods; the
cube's moments agree per cell to 5 percent in the energy density with the lattice's
expected shape and to a percent on average in the flux and force, chi F / c equal to
the deposited force in both runs; a dwell costs what a scattering costs, so the gain is
the scatterings a dwell stands for (100 at depth 10, 7 to 19 on the gridded spheres).
The test that mattered: the D0 conversion probability was half the right value and the
sphere alone could not tell that from discretization; the cube could (Section 2.2).
Default threshold set to 3 (refused below 2).  The disk atmosphere against Feautrier (the end-to-end gate, spherical polar,
polarized, free-free emission and absorption): the ladder of 1, 4 and 16 million
photons passes with the lattice as without it, grazing Q/I -0.0395 +- 0.0090 against
-0.0401 (norm order 0.49); the moments agree layer by layer within the interval-scatter errors, the deep
fluxes and forces being noise in both runs.  Two things that test taught, both built:
a walker with an absorption opacity forms its rates from the total extinction at its
energy (the table holds the pure Thomson rates), and a photon emitted in a diffusion
cell flies its first flight before it can be converted.  The wiki page Discrete
Diffusion was written 2026-10-29.  Next step: D2.

## 1. Why this and not the in-cell step

The random walk confined to one cell (`compton_mrw_plan.md`, `mrw_guide.tex`) is exact
for the diffusion equation inside a very thick cell and pays a dead zone at every face.
On the hires XRB the `accel_report` histogram put 9 percent of the scatterings in cells
thick enough to step in at the threshold and 60 percent in cells of one to sixteen mean
free paths across their half-width; the step replaced 6 percent of the scatterings and
saved nothing.  Discrete diffusion Monte Carlo (DDMC; Densmore, Urbatsch, Evans & Buksas
2007; frequency dependence in Densmore, Thompson & Urbatsch 2012; the resonant-line form
in Smith, Tsang, Bromm & Milosavljevic 2018) works cell by cell at optical depths of a
few, needs no cell-crossing approximation because crossing cells is what the walk is,
keeps every cell's own temperature, density and velocity, and costs per cell traversed
rather than per scattering.  That is the regime the XRB's work sits in, and it is the
only method that reaches the resonant-line problem (`mc_hotjupiter`), where a core
photon scatters 1e4 to 1e6 times in one cell.  The region step of the random-walk plan
(its Section 6.7) is superseded.

## 2. Review of the 2026-10-06 outline: what changes and why

### 2.1 Energy by the Kompaneets Green's function over the dwell, not by a lattice

The outline made (cell, energy group) one lattice with Chang-Cooper rates between
adjacent groups.  That is the form Smith et al. need for Lyman alpha, where the spatial
leakage depends on the frequency bin through the line profile.  For Compton scattering
below the Klein-Nishina regime the extinction is Thomson and grey: the spatial leakage
rates do not depend on the photon's energy at all, so space and energy decouple.  The
dwell in a cell is then drawn from the grey leakage rates alone, and the energy after a
dwell of comoving path c t at the cell's theta is drawn from the Kompaneets Green's
function at y = theta n_e sigma_T c t, with the free-free survival from the same table.
This is exactly what `MRWStep` already does (`kgreens->Quantile(xi, y, lam, u)` and
`kgreens->Survival(xi, y, lam)`): the validated `KompaneetsTable` is reused unchanged,
no energy groups exist, and the open question of the group energy at the interface
disappears.

Cost is the reason beyond simplicity.  On the group lattice a photon takes about
y / (Delta s)^2 frequency hops per dwell (a random walk in ln x with unit diffusion
coefficient per unit y); on the XRB a deep photon gains y of order 1 in a dwell, so at
the 0.05 group spacing of the solver's grid that is hundreds of lattice events for what
is one table lookup.  Smith et al.'s "matching condition" between the cell size and the
bin width (their equation 37) is the same cost made explicit; it does not arise here.

Where the guards of the Green's function fail (theta > 0.03, x theta > 0.05, x outside
the table, free-free sink too strong; the corona of the XRB), the walker declines into
transport as `MRWTrigger` does now; the report counts the declines.  A Klein-Nishina
energy dependence of the leakage, if it is ever wanted above 50 keV, is a per-group rate
table added later; the XRB bands end at 10 keV.

### 2.2 The interface needs the asymptotic-limit probabilities in both directions

The outline said a transport photon entering a thick cell "becomes a diffusion
particle" and a leaking walker "emerges on the face".  At a threshold of a few
optical depths the accuracy of DDMC rests on the asymptotic diffusion-limit boundary
condition (Habetler & Matkowsky, as used by Densmore 2007), in both directions:

- A transport photon crossing into a diffusion cell n with direction cosine mu to the
  face normal converts with probability P = (4 / (3 chi_n Delta_n + 6 lambda))
  (1 + 3 mu / 2), lambda = 0.7104, and is otherwise returned to the cell it came from
  with an isotropic inward direction (Smith et al. 2018 equation 18, Densmore 2007;
  derived in `ddmc_guide.tex` Section 8 from the same finite difference as the boundary
  leakage rate, so the two are consistent, with the equilibrium check that fixes the
  factor).  The D0 draft had a 2, from counting the hemisphere twice: built that way the
  lattice lost energy at every interface and the Thomson sphere escaped 5 to 13 percent
  too fast at every resolution, while a cube with no interface reproduced its exact
  lattice solution; with the 4 the sphere agrees with the analog run to a percent
  (D1, 2026-10-28).
- A walker leaking through a face into a transport cell emerges at a uniform point
  on the face with its outward direction drawn either from the asymptotic
  distribution, W(mu) proportional to mu + (3/2) mu^2 on the outward hemisphere, the
  flux-weighted emergent intensity of the diffusion limit, or isotropically, which is
  what Densmore 2007, Abdikamalov 2012 and Smith 2018 all do.  Both are built
  (`<ddmc> emergence`) and the disk atmosphere against Feautrier decides the default in
  D1, since the emergent intensity versus mu is the observable that sees the choice.
- The leakage rate into a transport cell is the boundary form with the extrapolation
  length, per unit path: k_f = (2 D_c / Delta_c) / (Delta_c + 2 lambda / chi_c) with
  D = 1 / (3 chi).  The outline wrote the denominator as Delta_c + 2 lambda D_c, a
  factor three short in the extrapolation distance (2 lambda / chi = 6 lambda D).  D0
  writes every formula with its units.

### 2.3 What the XRB histogram says the gain can be

DDMC removes the dead zone but not the thin cells.  The analog histogram
(`compton_mrw_plan.md`, XRB first pass) has 27 percent of the scatterings in cells of
optical half-width below 1 and 15 percent at 1 to 2.  With the threshold on the full
width at 4 (half-width 2), the 42 percent of scatterings in thinner cells stay analog
and the ceiling on the speedup is about 2.4; at the Smith et al. minimum of 2 (half
width 1), 27 percent stay and the ceiling is 3.7.  The hops inside the lattice are
nearly free next to a scattering (no geodesic steps, no tetrad, one table lookup per
dwell), so the ceiling is the number to expect, before the load balancer's part.  That
is a real but modest saving on a transport that the measured-cost partition already cut
from 980 s to about 380 s.  It is the go/no-go for D4 and D5, measured again in D3 with
the DDMC's own report before either is started; the resonant line is where the method
pays by orders of magnitude (Section 4).

### 2.4 Neighbour data and refinement boundaries, which the outline did not mention

The leakage rate through a face needs the neighbour cell's extinction and width.  The
block's `rho`, `tgas` and `species` arrays are allocated with ghost zones and filled
from the hydro primitives over `FillBounds`, which spans the ghosts; the hydro's own
boundary functions fill those, including prolongation across levels.  D0 verifies on a
two-block run (and at a level jump) that the ghost values of `species` match the
neighbour's active cells after a snapshot read, since a static run from `grid_from_file`
applies the boundaries once at startup.  The neighbour's width comes from `pcoord`,
which also covers the ghosts.

The XRB snapshot has eight levels of static refinement.  At a face whose neighbour is at
another level one coarse face touches four fine cells that do not align with it.
Densmore 2007 handles unequal neighbours by splitting the face and weighting the
sub-face rates by area; that is the complete treatment.  The first build does the
simple exact thing instead: a face to another level is a transport interface in both
directions (a walker reaching it emerges through it with W(mu), a transport photon
crossing it into a thick cell converts by P), whatever the neighbour's thickness.  The
report counts hops stopped at level jumps; the split-face rates are the follow-up only
if that count is large.  This is why block and rank crossings come in D3 and not at
the end.

### 2.5 Walker bookkeeping: the photon with a flag, at the cell centre

A walker is a `Photon` with a new integer column `ddmcp` (0 transport, 1
diffusing), its position set to the cell centre and its direction unused.  Then
`PackAll`, the block exchange, the outputs, the load balancer's payload and the restart
record carry it unchanged, `UpdateZone` and the boundary functions work on the centre,
and a hop into a cell on another block is the existing BUFFERED path landing in the
neighbour's face cell.  The alternative, a separate list, would duplicate all of that.
The block loop needs one guard: `TransferPhotonsOnBlock` scatters every photon that
returns from `Move` as EVOLVING, so a walker that returns on the cycle
budget must be skipped there.  (D0 also records how a transport photon that exhausts
`dtp` is held between cycles today, since the pushers leave it EVOLVING with
`dtp <= 0`; the walker follows the same path.)

### 2.6 Accuracy is first order in the cell size, and the dwell is Markov

The lattice reproduces the diffusion equation to first order in the cell size, and the
exponential dwell replaces a first-passage time that is not exponential.  Both errors
shrink with resolution, not with the threshold, so every verification gate is run at
two or three resolutions (the sphere over 8^3, 16^3 and 32^3 cells) and the acceptance
is convergence toward the analog result at the rate of the cell size, not agreement at
one resolution.  The disk atmosphere against Feautrier is the test of the interface
angular distribution, which no sphere test sees.

### 2.7 What a walker loses

Direction (the pressure tensor inside a diffusion cell is isotropic by construction and
the flux is the net leakage, so the scattering moments `Jnu*` in deep cells lose their
angular content, as they should for the diffusion solution), and polarization (emerging
unpolarized, as the random walk does).  Images and spectra are unaffected: a photon
escapes as a transport photon with a direction.  The energy density and the source
terms use the time-spent estimator, which is `UpdateMomentsMRW` and
`UpdateSourceTerms` as they stand.

### 2.8 The random walk and the resonance stub

With DDMC on, the random walk never fires: its threshold (20 and above) lies above the
DDMC threshold, and a walker in a thick cell has no position to walk from.  The two
keys stay independent; the wiki says they do not combine.  The resonance-line
acceleration in `PhotonPusher::MRWResonanceAcceleration` (a sphere step with the
Dijkstra et al. 2006 frequency solution and an escape-time fit, spherical polar only,
threshold hard-coded at 1000) is retired when R2 passes, the way the legacy random
walk was.

### 2.9 Containment (constraint set 2026-10-28)

The feature is self-contained in the way the random walk is: off by default, switched
by `<ddmc> enable`, byte-identical lists and moments with it off against a build before
it existed (the `port_equivalence.py` standard), and the physics in one file,
`ddmc.cpp` with `ddmc.hpp`, as an alternative branch in the photon mover.  The
difference from the random walk is that a walker is a persistent state, not
a step, which costs these hooks and no others outside the file:

- Pushers: a branch at the top of the per-photon loop that runs the lattice walk while
  `ddmcp` is set, and the conversion test after `UpdateZone` enters a new cell; both
  invalidate the general pusher's cached metric and connection, like the random-walk
  branch.
- `Photon`: the integer column `ddmcp`, registered only when the feature is on.
- `TransferPhotonsOnBlock`: the guard that skips the scatter for a walker.
- `SetupBlockFromFluid`: one call that builds the per-cell rate table, with the
  level-jump test from the block's neighbour information.
- `montecarlo.cpp`: the `<ddmc>` keys, the counters and the report lines.

Mesh boundaries need no code: a lattice face on the domain edge is a transport
interface, so the ordinary boundary function sees a photon.  A hop to another block
lands the walker at the cell centre and the BUFFERED exchange carries it with its
flag.  D0 checks every design choice against this list; anything that needs a hook
beyond it is a decision for the user.

### 2.10 Literature

No published DDMC for Compton scattering or the Kompaneets equation was found
(search of 2026-10-28).  The nearest relatives: Densmore, Thompson & Urbatsch 2012
(frequency-dependent DDMC: grey diffusion below a frequency threshold, transport
above); Wollaeger et al. 2013 (SuperNu: multigroup DDMC with Doppler shifts between
groups in an expanding medium, the closest moving-medium precedent); Abdikamalov et al.
2012 (energy-dependent DDMC for neutrinos); Cleveland & Gentile 2014 (teleportation
error in frequency-dependent hybrids).  Smith et al. 2018 is the resonant-line method
and the only published frequency-lattice DDMC for a scattering line.

## 3. The method in this code

Everything is a cell-centre quantity in the cell's comoving frame; under the general
pusher the frame is the cell-centre tetrad and lengths are proper widths along the
cell's edges, which `CellGeometry` and `MRWFrame` already supply.

- **Lattice and rates.**  Per cell, per transport cycle, in `SetupBlockFromFluid` after
  the number densities (rebuilt on restart and redistribution, so nothing is carried):
  the Thomson extinction chi = n_e sigma_T, the full widths Delta_m, the cell optical
  depth tau_cell = chi min_m Delta_m, and six face leakage rates per unit comoving path,
  k_f = (2 D_c / Delta_c) / (Delta_c + Delta_n D_c / D_n) into a diffusion neighbour and
  k_f = (2 D_c / Delta_c) / (Delta_c + 2 lambda / chi_c) into a transport neighbour or
  across a level jump, with D = 1 / (3 chi).  Stored as `ddmc_rate(7, k, j, i)` with the
  total.  Grey, so one table per cell.
- **Dwell.**  Path c t = -ln(xi) / k_tot, capped by the cycle budget `dtp`, by the
  Kompaneets table's y range and by `accel_pmax`-style survival if wanted; the channel
  by the k_f proportions.  A capped dwell leaves the walker in its cell.
- **Energy and absorption.**  y = theta n_e sigma_T c t at the cell's theta; new energy
  from `Quantile(xi, y, lam, u)`, weight times `Survival(xi, y, lam)` (free-free) or
  exp(-kappa_a c t) (grey absorption that is not free-free); `abs_method = prob` draws
  absorption from the survival instead.  The scatterings replaced, n_e sigma_T c t, go
  to `nscp` and the report.
- **Moments and source terms** (required: fluxes and radiation forces must come out of
  diffusion cells, unlike the random walk's, which deposits zero comoving flux and no
  force in walked cells).  Energy density by the time-spent estimator per dwell,
  `UpdateMomentsMRW(pphot, ct, mean energy, beta_tet, gam, ip)`.  Comoving flux by the
  face-hop estimator: each hop through a face adds plus or minus w E to that face's
  counter; the net over `tint` divided by the face area is the flux through the face,
  and the cell-centred flux is the mean of the cell's two faces per direction, which is
  the discrete Fick flux of the lattice and needs no model.  Pressure E/3 on the
  diagonal in the comoving frame, the diffusion closure.  Radiation force: the momentum
  deposited in the gas is chi F / c, equal to minus the gradient of E/3 by Fick's law,
  so each hop deposits plus or minus w E chi Delta_i / (2 c) into `MCRS1..3` of both
  cells sharing the face, each with its own chi and width; `MCRF1..3` from the same
  counters.  Energy exchange `MCRS0` from the weight times the energy change over the
  dwell plus the absorbed energy, through `UpdateSourceTerms` as now.  Lab and
  coordinate moments by the existing frame transformations of (E, F, P), the flux a
  tetrad-frame vector under the general pusher.  The held-back per-walker
  contributions of `sourceterms_error` accumulate over a walker's hops in a cell
  unchanged.  Known limits: the flux in a thick cell is the difference of two large hop
  counts (as the analog flux there is the residual of near-isotropic flights); the
  scattering-binned `Jnu` moments see only the energy before and after a dwell.
  Fick's law applied to the energy-density estimator is a test cross-check, not the
  product, since it fails in interface cells.
- **Interface.**  Section 2.2.  Thresholds: a cell is a diffusion cell when tau_cell
  exceeds `<ddmc> tau`; one key, the asymmetry between entering and leaving carried by
  the probabilities, a second key only if the tests ask for it.
- **Moving medium.**  Rates in the comoving frame of each cell.  At a hop the energy
  changes frame: draw a direction from W(mu) at the face, transform the energy to the
  lab with the leaving cell's velocity and to the comoving frame of the entering cell
  with its velocity (the existing `LorentzTransformFrequencyShift` and the tetrad
  transforms), then discard the direction.  Exact in distribution, no first-order
  shift formula; a few operations per hop.  The dwell's lab time is gamma times the
  comoving time; the advection of the cell's fluid past the lattice is not represented
  within a cell (the lattice is the grid), which is the approximation Smith et al. make
  and the moving sphere measures.
- **Curved spacetime.**  Proper widths and theta in the tetrad frame per cell; the dwell's
  proper time to coordinate time by u^t as `MRWStep` does (`gam * ct / lapse`); the hop
  transform through the two cells' tetrads with the sampled face direction.  The
  walker's coordinate position is the cell centre.
- **Time.**  The waiting times are physical; the cycle budget is a stop; a walker
  stays diffusive across cycles; the moments normalize by `tint` as now.
- **Where it lives.**  A branch at the top of each pusher's per-photon loop: if
  `ddmcp`, hop until it emerges, buffered, absorbed or out of budget, then continue the loop
  or return; the conversion test sits where a transport photon's `UpdateZone` enters a
  new cell.  `TransferPhotonsOnBlock` skips the scatter for `ddmcp` walkers.

## 4. Compton or resonance first

Compton first, as proposed, but with a narrow Compton scope and the resonance work
starting from D3, not after D6.  The reasons:

- The grey leakage decouples space from energy, so the lattice, the interface, the
  thresholds and the block machinery, which is most of the code and all of the risk, are
  built and validated without any frequency complication, against an exact reference
  (the Kompaneets Green's function on the uniform sphere) with a test suite that exists
  (`tst/montecarlo/sphere_compton`).  The resonant line has only asymptotic references
  (Harrington and Neufeld for the slab, Dijkstra et al. 2006 for the sphere) and its own
  analog kernel.
- The energy update reuses a validated table; the resonance needs a frequency lattice
  whose leakage couples to the spatial leakage, a Fokker-Planck form that holds only in
  the wings, a core bin wider than the core, a frequency hybrid that Smith et al. did not
  implement, and a treatment of velocity gradients that they flag as their weak point.
  The published reference lowers the design risk of the resonance work, not its size.
- The payoff ranking is the reverse: the XRB ceiling is a factor 2.4 to 3.7 (Section
  2.3); Smith et al. report 1e2 to 1e6 on resonant lines, and the hot Jupiter's cost is
  set by the handful of core photons `scattering_histogram.py` shows.  So D4 and D5 are a
  decision taken on D3's report, and the resonance phases start once D3 passes.

## 5. Phases

Each phase ends with its tests passing and a status block here; a phase that fails its
gate stops the plan at that point for a decision rather than continuing.

### D0. Design note (no code)

Write `doc/monte_carlo/ddmc_guide.tex` in the pattern of `mrw_guide.tex`: every formula
of Section 3 with its units in this code's conventions (comoving extinction per cm,
`l_cgs`, `vel_cgs`, the general pusher's coordinate components, the proper widths of
`CellGeometry`), the two interface probabilities and the emergence distribution checked
against Densmore 2007 and Smith et al. 2018 equation by equation, the moving-medium hop,
the GR time conversion, and the walker's life cycle through `Move`, `UpdateZone`, the
boundary functions, the block exchange and the block loop (with the EVOLVING-on-budget
guard of Section 2.5).  Verify and record: the ghost values of `species` and `tgas`
after a snapshot read match the neighbours' active cells across a block face and a level
jump; how an exhausted `dtp` is held between cycles.  Decide and record the key names
(Section 6).  Gate: the user approves the note.

### D1. Grey DDMC in flat cells, one block, static

Build: the `ddmcp` column, the per-cell rate table, the dwell and leakage draw, the
interface in both directions, the moments, the counters and the report, in the Cartesian
and spherical-polar pushers (a spherical-polar cell is the box `CellGeometry` already
makes of it).  Tests, each against the analog run of the same deck and at three
resolutions:

- Thomson sphere (`mc_sphere_isoth` at tiny theta, point source), tau 20, 40, 100 over
  8^3, 16^3, 32^3 cells: escape-path CDF quantiles and escaping fraction; the error
  falling with the cell size.
- The same with grey absorption (`abs_method = weight` and `prob`): absorbed fraction.
- Disk atmosphere against Feautrier (`tst/montecarlo/disk_atmosphere`): the emergent
  intensity versus mu, which tests W(mu) and the conversion probability; and its
  moments, where the flux is the same through every layer and the force is minus the
  pressure gradient, against the analog `mcsrc` and `mccom` outputs and their
  `--empirical` errors: the face-hop flux and the force through the diffusion layers,
  Fick's law from the energy density as the cross-check.
- Threshold scan at 2, 3, 5, 10 on the sphere: accuracy and the fraction of scatterings
  replaced; sets the default.
- Cost: wall time and scatterings replaced versus the analog run; the per-hop cost.

Gate: analog agreement converging with resolution; Feautrier within the analog errors at
the default threshold; a measured per-hop cost.

### D2. Compton energy and free-free

Build: the Green's-function dwell update, the survival, the decline guards and their
counts.  Tests: the Compton sphere of `sphere_compton.py` (tau 15, 40, 100; x0 0.1, 1,
3; with and without free-free) against the Green's function and the analog run at the
three resolutions: the per-y separation test, the escaping spectrum against the
convolution, the survival; the `mc_isoth` Thomson regression unchanged.  Gate: the
random-walk plan's Phase E standard, at every resolution, with the error converging.

### D3. Blocks, ranks, refinement

Build: the BUFFERED hop across block faces, the level-jump interface of Section 2.4,
the counters in the exchange, the report's level-jump count.  Tests: the gridded sphere
over 8 and 64 blocks on 1, 4 and 16 ranks bitwise against itself on one rank where the
code promises that (statistically otherwise); photon conservation
`nesc + nabs + ndes + nrem == ntot` on every run; the sphere with a static refinement
block added; `repack_equivalence.py` with `<loadbalancing> test_repack` on a DDMC deck.
Then the **decision run**: the XRB short deck (`d01_emis2` snapshot, flat-spacetime
DDMC is not possible there, so this run is the analog deck with the DDMC report only,
counting what the lattice would reach at each threshold given the level-jump rule) and
the ceiling of Section 2.3 recomputed from it.  Gate: tests pass; the user decides on
D4 and D5 against the resonance phases.

### D4. Moving medium

Build: per-cell comoving rates (already, since the rates are comoving), the hop
transform of Section 3.  Tests: the moving sphere of the random-walk plan (beta 0.01,
0.1, 0.3): lab-frame spectrum and escaping fraction against analog; the Doppler hop
against a run with the hop shift switched off, to see its size.  Gate: analog agreement
within errors at beta 0.1, the beta 0.3 deviation recorded.

### D5. Kerr-Schild and the general pusher; the XRB

Build: the tetrad-frame widths and the time conversion, the hop through two tetrads.
Tests: `mc_isoth_gr` disk atmosphere; the XRB at the production thresholds against
`d01_emis2`: the four bands within the analog output scatter, the image, escape and
absorption counts, the cost ratio with the measured-cost partition on.  Gate: bands
within the scatter; a speedup near the ceiling of D3's decision run.

### D6. Production features

Free-free photons born inside diffusion cells as walkers; weight window and roulette
on the lattice (`WeightWindow` on the cell, `RouletteOrAbsorb` after survival); dynamic
runs (the held rates, the restart record carrying `ddmcp` through `PackForRestart`,
and the budget stop once the held status exists as a feature of dynamic runs in
general, which is a separate item decided 2026-10-28 and not part of this phase); the
wiki page.  Tests: `mc_hotjupiter`-style dynamic cadence on a
Compton deck; one-rank bitwise restart.

### R0. Resonance design note

Append to `ddmc_guide.tex`: the frequency lattice in x with bin widths, the bin-averaged
Voigt extinction (Smith et al. equations 12 to 14, or quadrature of `XsecVoigt`), the
spatial leakage per (cell, bin), the frequency leakage k_x between neighbouring bins
(their equation 11), the core bin at or above twice the core-wing transition, the
frequency hybrid: a (cell, bin) with tau below threshold is transport, a walker
leaking into it emerges with its frequency drawn in the bin, a transport photon
entering a thick (cell, bin) converts by P; the Doppler and temperature rescaling at a
hop by the sampled-direction transform of D4; what the Fokker-Planck form drops (core
redistribution, recoil, detailed balance) and where that matters for the hot Jupiter.
The per-(cell, bin) rate table and its memory (nbin times the grey table).  Gate: user
approval.

### R1. Static line

Build: the frequency lattice on the D1 and D3 machinery, the resonance opacity per bin.
Tests: the Neufeld slab (`J(x)` of Harrington 1973 and Neufeld 1990) and the uniform
sphere against Dijkstra et al. 2006 and against the analog kernel at matched a tau_0;
the time-dependent pulse of Smith et al. section 5; resolution in x and z scanned
together per their matching condition.  Gate: analog agreement converging with both
resolutions.

### R2. Moving line, and retiring the stub

Build: the hop rescaling.  Tests: an expanding shell against analog; the red and blue
peak asymmetry, which Smith et al. could not reproduce with a core bin that blocks
leakage across the line centre, measured against analog so the limitation is known
here.  Then `MRWResonanceAcceleration` is removed.  Gate: the asymmetry error recorded
and judged.

### R3. The hot Jupiter

Tests: `mc_hotjupiter` against its analog run at equal photon count: the Lyman alpha
emergent spectrum and transmission, the photoionization coupling unchanged
(`ruser_meshblock_data[1]`), the cost and `scattering_histogram.py`.  Gate: agreement
within the analog scatter and the speedup recorded; the wiki page updated.

## 6. Keys and the wiki

A block of its own, `<ddmc>`: `enable`, `tau` (the cell threshold on the full width,
default from D1), `report`, and in R1 `nbin`, `xmax`, `core_width`.  Nothing is read
from `<montecarlo>` or `<randomwalk>`.  The report lines follow the random walk's
(`accel_report`): conversions each way, hops, scatterings replaced, budget stops,
declines by reason, level-jump stops, the histogram of scatterings by cell thickness.
A wiki page Discrete Diffusion (beside Random Walk Acceleration) is proposed for
approval when D1 passes, with the keys, the report, what a walker loses (Section 2.7)
and the rule that the two accelerations do not combine.

## 7. What exists to reuse

- `KompaneetsTable` (`kgreens.cpp`) with its `Quantile` and `Survival`, the guards of
  `MRWTrigger`, `UpdateMomentsMRW`, the decline counters and `accel_report`.
- `CellGeometry` and `MRWFrame`: the cell as a box in the comoving or tetrad frame.
- `TransformToComoving`, `TransformToCoordinate`, `LorentzTransformFrequencyShift`.
- The BUFFERED exchange, `PackAll`, the load-balancing payload, `PackForRestart`.
- `tst/montecarlo/sphere_compton` with `sphere_theory.py`, `band_luminosity.py`,
  `disk_atmosphere.py`, the XRB decks and `d01_emis2`, `scattering_histogram.py`.
- `XsecVoigt`, `ResonanceLineOpacity`, `ScatterResonanceLine`, `mc_hotjupiter`.

## 8. References

Densmore, J. D., Urbatsch, T. J., Evans, T. M., Buksas, M. W. 2007, J. Comput. Phys.
222, 485.  Densmore, J. D., Thompson, K. G., Urbatsch, T. J. 2012, J. Comput. Phys. 231,
6924.  Smith, A., Tsang, B. T.-H., Bromm, V., Milosavljevic, M. 2018, MNRAS 479, 2065
(arXiv:1709.10187).  Wollaeger, R. T., van Rossum, D. R., et al. 2013, ApJS 209, 36.
Abdikamalov, E. et al. 2012, ApJ 755, 111.  Cleveland, M. A., Gentile, N. 2014,
J. Comput. Theor. Transp. 43, 6.  Gentile, N. A. 2001, J. Comput. Phys. 172, 543.
Habetler, G. J., Matkowsky, B. J. 1975, J. Math. Phys. 16, 846.  Harrington, J. P. 1973,
MNRAS 162, 43.  Neufeld, D. A. 1990, ApJ 350, 216.  Dijkstra, M., Haiman, Z., Spaans,
M. 2006, ApJ 649, 14.
