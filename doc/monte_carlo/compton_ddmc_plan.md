# Discrete diffusion Monte Carlo for Compton and resonance scattering: plan

Status: **D0 and D1 done 2026-10-28, D2 and D3 done 2026-10-29, D4 done 2026-10-30, R0 approved, R1 done and R2 done 2026-10-30 (the frequency lattice, `<ddmc> line = lya`, guide Sections "What R1 built" and "What R2 built"; `MRWResonanceAcceleration` removed), R3 done 2026-10-31 (spherical polar: the face-area rates, the polar axis, the scattering column's overflow; the core bin's kinetic exit and its edge criterion; the hot Jupiter within the analog scatter at a 21 percent saving; guide Section "What R3 built"), all on branch `ddmc`; the XRB count that decides the split-face rates and D5 is in: on the hires snapshot at threshold 3, 132 million diffusion cells, 13 percent of the mesh, 0.9 percent of them at level jumps (no split-face rates), and 67 percent of the scatterings in cells at least 3 deep, a ceiling near 2.8 on the whole run.**  The 2026-10-06
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
Diffusion was written 2026-10-29.

D2 (Compton energy and free-free, 2026-10-29): the dwell draws the walker's energy from
the Kompaneets table at y = theta times the scatterings the dwell stands for, in chunks
of the table's y range, and its free-free survival from the same table; `kgreens_file`
is read from `<ddmc>` or `<randomwalk>`; a cell with theta above 0.03 is not a diffusion
cell; a photon whose x lies outside the table or whose free-free sink is too strong is
not converted, and a walker whose dwell takes it there is released (both counted).  The
Compton sphere (1e5 photons, x0 1, theta 2e-3 unless said) passes the Green's-function
gate on the lattice as on the analog run at tau 15 (8 cells), 40 (8 and 16 cells) and
100 (32 cells), at x0 0.1 and 3, with free-free at a radius where it absorbs 14 percent
(escaping weight 0.8606 against 0.8602 analog and 0.8579 theory), and the two agree in
the spectrum within 0.5 to 1.8 times the two-sample scale; at tau 100 the analog's 4.8e8
scatterings become 4.8e6 plus 1.6e7 dwells.  Two things the test taught, both built: a
bound on x theta at which walkers are released and photons are not converted (the
random walk's 0.05) is a boundary in energy between 20-scattering dwells and single
scatterings, and the uneven steps across it deplete the walkers' side (6 percent low in
the mean energy at theta 0.01, the spectrum empty above x = 5), so the lattice has no
such bound and relies on the table's x range and the cell's theta; and the leakage
rates and y take the walker's scattering opacity, the thermally averaged Klein-Nishina
value under Compton, which the analog run uses, without which the paths run 7 to 12
percent long at theta 0.01 to 0.02.  With both, the lattice agrees with the analog run
at theta 0.01 and 0.02 to 0.5 percent in the mean energy and 1 to 1.4 times the
two-sample scale in the spectrum, while both sit 2 to 3 percent above the Thomson
Kompaneets theory of the driver (the exact kernel's hardening, not the lattice's).

D3 (blocks, ranks, refinement, 2026-10-29): the block hop and the level-jump rule were
already built in D1, so D3 added the report's level-jump counts (`ddmc_jump`, the faces of
each cell that lead to another level; diffusion cells with such a face; emergences
through them) and the mask built with `<ddmc> report = true` alone, so a run that cannot
enable the lattice (the XRB under the general pusher) still counts its reach.  The test
driver `ddmc_blocks.py` passes: the Thomson sphere over 8 and 64 blocks on 1, 4 and 16
ranks conserves photons and agrees with the 8-block one-rank run within 1.1 times the
two-sample scale; one rank plain against `test_repack = local` and `mesh` gives
byte-identical lists; with a static level-1 block inside the sphere the lattice agrees
with the analog run on the same mesh within 0.7 times the scale (318 of 6511 diffusion
cells touch the jump, and 121 thousand of 229 thousand emergences go through it, the
number the split-face rates would save).  The test found a generator bug, not an
exchange bug: `mc_sphere_isoth` kept the origin cell's indices in globals that every
block's setup overwrote, so with 27 or 64 blocks the point source was born with the last
block's indices and 20 to 50 percent of the photons died on arrival in both pushers;
fixed in `InitializePhoton`.  The decision run is three decks in the user's hands (the
hires snapshot is 114 GB): the `d01_emis2` deck with `<ddmc> report = true`, 1000 photons
and `tau` of 2, 3 and 4, whose report line gives the diffusion cells and the share at
level jumps; the split-face rates are built if that share is above about a fifth.
D4 (moving medium, 2026-10-30): the design note's hop transform (a W(mu) direction
drawn at the hop, the energy taken to the lab and into the neighbour's frame) with the
walker at rest in the lab failed the moving Compton sphere outright: at beta 0.1 the
escape paths were 2.1 times the analog's in a uniform flow and 1.8 times in a homologous
expansion (`--profile homologous`, added to `mc_sphere_isoth` and the sphere driver), since
nothing carried the walkers with the fluid.  Built instead, each step forced by a
measurement: the walker drifts with its cell's fluid during a dwell at the velocity
interpolated linearly across the cell, and crosses a face the drift reaches
(`DDMCAdvect`; into a diffusion cell as a walker owed nothing, `ddmcp = 8 + f` across a
block face; otherwise an emergence isotropic outward in the comoving frame), depositing
no flux because the dwell's boosted isotropic field already carries the advective flux;
over a dwell the energy loses exp(-c t div(beta) / 3), the work done on an expanding
fluid, with the divergence stored in the rate table (`DDMC_RATE_DIV`); a hop keeps the
walker's comoving energy and its offset from the centre (the design's hop transform
double counts the adiabatic loss by 1.42 / tau_cell and on its own gives only that
fraction of it; a hop that recentred the walker set a drifting walker back a quarter
cell per hop, 8 to 18 percent slow in the interface-free cube, which `ddmc_cube.py
--beta [--profile homologous]` now runs); and the conversion test at a face reads the
photon's cosine and energy in the diffusion cell's frame and returns photons isotropic
in that frame (the lab-frame test left the energies 3 to 8 percent low).  Result against
analog at beta 0.1: paths within 1 percent in both flows, mean energy +1.7 percent
(uniform) and -3.4 percent (expansion), spectra 2.6 and 4.3 times the two-sample scale
(the static lattice gives 0.6 to 1.2); at beta 0.3 paths within 1.2 percent, energy
+2.8 and -10.7 percent.  The cube's exact check, the static sphere and the D3 block
tests are unchanged.  The remaining error is first order in beta at the interface.
The hires count (2026-10-30, one Riker node, 749 GB resident, three decks at
thresholds 2, 3 and 4): 157, 132 and 116 million diffusion cells of the 1015 million on
the mesh (15, 13 and 11 percent), with 1.49, 1.22 and 1.06 million of them at level
jumps, under 1 percent, so the split-face rates are not built; the histogram of the
same run puts 61 percent of the scatterings in cells at least 4 deep and about 67
percent at least 3 deep, so with a dwell costing about a twentieth of the scatterings
it stands for the ceiling on the XRB run is a factor near 2.8.  (A first run on a
128-cubed snapshot of the same name gave 7 percent of the mesh, 3 percent at jumps and
80 percent of the scatterings.)  Next: the resonance phases, D5 deferred.

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
- **Moving medium** (as built in D4; the 2026-10-06 version had a frame change at
  each hop and no advection, which the moving sphere rejected).  Rates in the comoving
  frame of each cell.  The walker is carried by its fluid during a dwell, at the
  velocity interpolated linearly across the cell, and crosses a face the drift reaches
  (into a diffusion cell as a walker, otherwise emerging isotropic outward in the
  comoving frame), depositing no flux, since the boosted isotropic field of the dwell
  carries the advective flux; a hop is a translation by one cell that keeps the
  walker's offset.  Over a dwell the energy loses exp(-c t div(beta) / 3), the work on
  an expanding fluid, and keeps its value at a hop; the conversion test at a face is
  made in the diffusion cell's frame.  The dwell's lab time is gamma times the comoving
  time.  Design note Section "A moving medium" and its D4 section.
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
counts.  The table file is read as `kgreens_file` from either `<ddmc>` or `<randomwalk>`
(decided 2026-10-29), so the lattice runs with the walk off; the constructor's refusal of
`scattering = compton` under `<ddmc>` is lifted.  Tests: the Compton sphere of
`sphere_compton.py` (tau 15, 40, 100; x0 0.1, 1, 3; with and without free-free) against
the Green's function and the analog run at the three resolutions: the per-y separation
test, the escaping spectrum against the convolution, the survival; the `mc_isoth`
Thomson regression unchanged.  Gate: the random-walk plan's Phase E standard, at every
resolution, with the error converging.

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

Done 2026-10-30, not as written.  The hop transform as designed was measured and
rejected (see the status block and the design note's D4 section): the build is the
drift of the walker with its fluid, the adiabatic energy change over a dwell, no frame
change at a hop, and the interface test in the diffusion cell's frame.

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

Written 2026-10-30 as Section "R0: the resonance line, design note" of the guide, with
six decisions put to the user: one comoving-frequency grid for the mesh set by
`line_temp` (not Smith's per-cell Doppler grid); the hop keeps the comoving frequency
and the velocity gradient acts over the dwell (D4's rule, not their equations 19 and
20); entry into the lattice in frequency by scattering, no interface probability at
frequency faces; Smith's core bin rather than a redistribution matrix; the hot Jupiter
Lyman alpha runs stay on the legacy spherical-polar pusher (their decks already do);
the line code in a new `ddmc_line.cpp`.  Keys proposed in `<ddmc>`: `line`,
`line_temp`, `xmax`, `dx`, `core_width`.  Also found: the hot Jupiter's core-skipping
hook runs on walkers through `UserWorkInMove` and must be off with the lattice, and
the hot Jupiter snapshots on disk are ionization-only, so the lattice's reach there
is measured in R3 on a wind snapshot.

### R1. Static line

Build: the frequency lattice on the D1 and D3 machinery, the resonance opacity per bin.
Tests: the Neufeld slab (`J(x)` of Harrington 1973 and Neufeld 1990) and the uniform
sphere against Dijkstra et al. 2006 and against the analog kernel at matched a tau_0;
the time-dependent pulse of Smith et al. section 5; resolution in x and z scanned
together per their matching condition.  Gate: analog agreement converging with both
resolutions.

Built 2026-10-30 (guide Section "What R1 built"): `src/monte_carlo/ddmc_line.cpp` on
a `ddmc.cpp` refactored to one membership test (`DDMCIsNode`), bitwise on the grey
runs; keys `line`, `line_temp`, `xmax`, `dx`, `core_width` in `<ddmc>`; the report
adds the nodes, the frequency hops, emergences and returns.  Departures from R0:
the frequency interface has the conversion probability sqrt(2 pi) / (2 dx) with the
rest reflected about the bin edge, and a frequency emergence lands at the crossed
edge plus a step's overshoot (without them the spectra were 5 to 7 percent low where
the node band ends on the peak; `freq_interface_toy.py` is the one-dimensional check
that chose the pair); a virtual bin beyond each end of the grid; a frequency hop draws
the energy in the new bin; `nscp` saturates; the analog sampler's `u_0` follows
Laursen et al. 2009 (the old 0.96 x cost 94 percent of a run).  Tests
(`tst/montecarlo/sphere_lya/sphere_lya.py`, 4000 photons, 32 cells, dx on the
matching line): slab against Neufeld and sphere against Dijkstra at a tau_0 = 1e5 to
1e7, medians within 0.1 bin and Kolmogorov distances 0.016 to 0.027 (scale 0.016);
against the analog kernel on the slab at tau_0 = 1e6 and 10 K with the band edge at
1.1, 0.7 and 0.5 of the peak, within 1.1 times the two-sample scale, 1000 to 5000
times cheaper; the resolution scan converging, with the finding that the node band
narrows as cells shrink (x_edge ~ tau_cell^1/2) and the cost returns once the band
ends inside the peak (31000 analog scatterings per photon at 64 cells).  The pulse
test of Smith et al. section 5 is not run (static runs only).  `mc_sphere_lya`'s
medium is the cells whose centres lie inside the radius, the rest at 1e-20 of the
density with the deck's floors below it (the default `dfloor`, 3.5e-18 cgs, made the
empty cells diffusion cells).

### R2. Moving line, and retiring the stub

Build: the hop rescaling.  Tests: an expanding shell against analog; the red and blue
peak asymmetry, which Smith et al. could not reproduce with a core bin that blocks
leakage across the line centre, measured against analog so the limitation is known
here.  Then `MRWResonanceAcceleration` is removed.  Gate: the asymmetry error recorded
and judged.

Built 2026-10-30 (guide Section "What R2 built"): no hop rescaling (D4's drift and
adiabatic shift already act on the walker's comoving energy and R1 made a shift across
a bin edge a frequency crossing); the one rule added is that the core bin takes no
adiabatic shift, since a core scattering redraws the frequency and erases it (with the
shift the lattice was 8 percent redder than analog at one Doppler width; without, 4).
`MRWResonanceAcceleration`, its escape-time sampler and `UpdateMomentsAcceleration`
removed; `<randomwalk> enable` with `scattering = resonance` refused.  Tests
(`sphere_lya.py --checks moving`, the sphere at tau_0 = 1e6 and 10 K, 4000 photons,
lattice at 16 and 32 cells against analog): a uniform flow (frame bookkeeping, no
drift) agrees to 1.2 scales; the homologous expansion at 3 and 10 Doppler widths to
0.9 to 1.5 scales in the spectrum, at 1 Doppler width to 2.0 to 2.6 with the red
fraction 0.01 to 0.02 high and the mean x one Doppler width redder out of 13; paths
within 5 percent except 13 percent long at 16 cells in the strongest flow.  The
one-dimensional toy `freq_drift_toy.py` (the lattice's rules against the kinetic walk
with the real wing kernel x' = x - 1/x + N(0,1), no spatial transport) reproduces the
code's departure at a moderate drift and finds no rule variant that removes it
(keeping the walker's offset at a frequency hop, D4's spatial rule, is redder, so the
uniform redraw stays); the departure is the lattice's discretization, judged
acceptable for the hot Jupiter's lattice region, where the flow is below the thermal
width.  Asymmetry: the drift in the wing bins and its erasure in the core carry the
red/blue asymmetry, and the core bin does not block it.

### R3. The hot Jupiter

Tests: `mc_hotjupiter` against its analog run at equal photon count: the Lyman alpha
emergent spectrum and transmission, the photoionization coupling unchanged
(`ruser_meshblock_data[1]`), the cost and `scattering_histogram.py`.  Gate: agreement
within the analog scatter and the speedup recorded; the wiki page updated.

Status (2026-10-31): done; the record is the guide's Section "What R3 built".  The
phase was mostly the lattice in spherical polar coordinates, which the hot Jupiter
needs and which no test had exercised: `sphere_lya.py --checks rsphere` (a second
build, `--path-spherical`) found a polar-axis defect (`DDMCPoleFace`), the overflow of
the scattering column past `INT_MAX` (now saturating), and the leakage rates in the box
form, wrong for a curved cell (`DDMCFaceFactors`); with them the spherical sphere
agrees with Dijkstra et al. to 0.01 bin at 1e6 and 1e7, and the spherical expansion
with its analog run to 0.5 to 1.3 scales.  The core bin's exit into a non-node bin is
the kinetic rate (a visit table at setup, `DDMCLineCoreVisit`; the harmonic rate made
the core-only slab's path 1.5 times the analog one), and the core bin is a node only
where the cell is thick at the core's edge, not on the bin's average.  The hot Jupiter
(2000 recombination photons, equal weights, which the comparison needs: the emission
weights span fifteen decades and two analog seeds then differ as much as the lattice)
agrees with both analog seeds within the two-sample scale with the transmission equal,
the lattice holding one percent of the cells and 30 percent of the scatterings, at
1220 CPU seconds against 1552.  The snapshot's base is at the edge of the lattice's
regime (a cell must be about 7e4 deep at line centre for the core bin at 8000 K);
a denser base would gain more.  The photoionization coupling was not exercised
(`nion = 0`).  Deferred: a graded bin width next to the core (the harmonic
core-to-node rate at a wide `dx` misstates the core residence), the core bin's memory
of its entry side, the emergence and reconversion at periodic mesh faces, and an
angular decomposition of the spherical test mesh (its radial shells serialize the
ranks).

### R4. The core edge inside the Doppler core

Why: on the production hot Jupiter deck (stellar Lyman alpha and photoionization,
equal weights, the analytic 7000 K atmosphere, `~/mc_test/hot_jupiter/combo`) the
lattice of R3 holds no cell at all: a cell must be thick at $|x| = 3.3$, and the base
is $7\times10^4$ deep at line centre, 1.3 at the core's edge.  Yet 84 percent of the
run's scatterings are made in cells more than 4000 mean free paths across at the
photon's frequency, and 68 percent by one photon of a thousand (330\,000
scatterings, 157 of the 157 wall seconds of the cycle; 96 percent of the photons
never scatter).  The deep photon sits in the Doppler core of a cell that is thick
at $|x| < 2.7$ and thin outside, and the single global core bin cannot hold it.

Design (the toy is `tst/montecarlo/sphere_lya/core_edge_toy.py`, kinetic core visits
against a hybrid walk, frequency only):

- Each cell gets its own core edge $x_c$ (`ddmc_line_xc`, reference Doppler units):
  the largest $|x|$ at which $k(x)\,W \ge$ `tau`, by bisection on the Voigt
  extinction, capped at `core_width`; a cell whose edge in its own Doppler units is
  below `<ddmc> core_min` (default 2, where a kinetic visit is 260 scatterings) is
  not a diffusion cell.  The wing bins are unchanged and are nodes only where the
  full core is (the extinction falls outward, so a partial core has no wing nodes).
  The core node is the band $|x| < x_c$ of the core bin; `DDMCLineNode(cell, e)`
  tests the energy against it and the bin-level test keeps its meaning.
- The core walker's extinction for the scatterings a dwell stands for is the mean of
  $k$ over the node (flat $J$ within a thick core), its exit through either frequency
  face the kinetic rate $k/(2N)$ with $N(x_{\rm loc}, s)$ from a two-dimensional
  visit table over the local edge and the Doppler scale (the R3 table was one row,
  $x_{\rm loc} = 3.3\,s$), bilinear in $\log N$.  The emergence frequency is the
  edge plus an overshoot drawn from the quantiles of the exit overshoots of the same
  visits: the toy's exits land at a mean 0.19 to 0.23 Doppler widths beyond the edge
  for $x_c$ of 1.5 to 2.5 (near exponential, $d_{99} \approx 0.95$) and 0.31 to 0.46
  with a heavier tail at 3.0 to 3.3, where the Lorentzian exits join; the R3 rule, a
  uniform fraction of a Rayleigh step (mean 0.63), places the photon three times too
  far out, into a cell $e^{-2 x_c d}$ thinner.  An entry into the node from a
  scattering is admitted with probability 1 (Kac: the mean sojourn from the
  stationary entrance distribution is $1/p_{\rm out}$, so the stationary exit rate
  balances the kinetic entries exactly).
- The visit table is written to `<ddmc> visit_file` after its build and read from it
  when the header (line temperature, core width, core minimum, visits, grids) matches,
  so that the setup, 40 s today and about twice that with the second dimension, is
  paid once per deck.
- The core node's spatial leakage: the R3 code diffused the core walker with the
  bin-mean $k$, which is $e^{x_c^2}/(2 x_c^2)$ too large at the edge frequencies
  that carry the flux (a factor 20 at $x_c$ 2.6, 700 at 3.3; the slab and sphere
  tests cannot see it, their core mean free path being $10^{-6}$ of the medium).
  With $J$ flat across the node, each face's rate is the average over the node of
  the per-frequency rate $2/(3W(k(x)W + k_n(x)W_n))$ (Simpson in $x = x_c\cos\theta$,
  64 intervals, which resolves the $e^{x^2}$ rise at the edge), the neighbour's
  node or the boundary form by frequency; the frequency of a hop is drawn in
  proportion to that face's per-frequency rate and then tested against the
  neighbour's node (hop or emerge at that frequency), the force deposit at a hop
  uses $k$ at that frequency, an advective crossing draws uniformly in the node,
  and the conversion of a transport photon at a face uses $k$ at its own frequency.
- The report counts the cells with a partial core edge and their mean edge.

Tests and gates: `core_edge_toy.py` (the visit lengths and exit law, and the hybrid
walk's scatterings and visits to reach $|x| \ge X$ against the kinetic walk, within
the sampling scale for the quantile exit and the measured failure of the Rayleigh
rule); the sphere_lya checks unchanged (slab, sphere, rsphere, analog, moving);
a new `coreedge` check, the slab at $\tau_0$ of $10^4$ and $10^5$ against its analog run
at 8 and 32 cells, where every cell has a partial edge; the production deck's Lyman
alpha type against its analog run (spectrum, transmission, the scattering
histogram, the per-photon lifetimes) and the saving.  Gate: the slab within the
analog scatter; the hot Jupiter within the two-sample scale with the longest-lived
photon no longer setting the cycle's wall time.

Status (2026-11-22): built; the record is the guide's Section "What R4 built".  The
visit table is a stationary chain per point (200 visits and 1e6 scatterings, or 20 000
visits), its exits kept as 65 tail-weighted quantiles, cached in `<ddmc> visit_file`;
the core walker's spatial rates are the node averages.  `coreedge` passes at tau0 1e4
and 1e5 (x within 1.3 scales, paths within 1.4, 20 to 400 times cheaper); the slab and
sphere checks are unchanged; the production hot Jupiter's Lyman alpha type agrees with
its analog run to K = 0.001 at 20 000 photons with the transport wall time 6.4 s against
377 (the longest-lived photon 6 s against 345).  Open: the entry-exit correlation of
short visits (5 to 10 percent at an edge of 1.5), the 1.2 ms per scattering of the
analog hot Jupiter deck (a profile is in hand), the table's cost above x_loc of 5.

To do: the radiation force of the lattice against analog.  Nothing in R1 to R4 measured
the force; the slab and sphere tests have no force gate and the hot Jupiter comparison
was the escaping spectrum, the transmission and the costs.  On the lattice the force is
chi F/c with F the face-hop flux (`DDMCHopDeposit`), deposited only at face crossings,
and in a core cell chi is the Voigt extinction at a frequency drawn in proportion to the
face's per-frequency rate (`DDMCLineCoreDraw`, `DDMCLineKPhoton`), so the deposit is
the node integral of k(x) F(x) in expectation under a flat J; the R3 code weighted every
crossing with the bin-mean k, wrong by the Planck-to-Rosseland ratio over the node.
What is expected: right in the mean to the lattice's first order in the cell size at
the interface and marginal where the cell is only `tau` deep at the edge frequencies,
which carry the flux; the two estimators meet at every node mismatch between
neighbours; and far noisier per photon in a core cell, where a walker leaves by the
frequency faces about ten times more often than by a spatial face, so a cell's force
is a few hops each carrying a whole visit's energy where analog sums thousands of
kicks.  The measurement: the Lyman alpha force profile of the production hot Jupiter,
lattice against analog in the `mcsrc` output with the `RadforceF_err` arrays (the
20 000-photon runs in `$TMPDIR/hj_core/lat20k` and `an20k` both wrote it, so this is a
post-processing step on runs in hand), then a force gate in `coreedge` on the slab's
force profile if the hot Jupiter shows a bias.  Until this is done the lattice's force
is not to be trusted in a coupled run.

## 6. Keys and the wiki

A block of its own, `<ddmc>`: `enable`, `tau` (the cell threshold on the full width,
default from D1), `report`, and in R1 `line`, `line_temp`, `xmax`, `dx`, `core_width`.  Nothing is read
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
