# Discrete diffusion Monte Carlo for Compton scattering: summary plan

Status: **deferred, not started (2026-10-06).**  Written as the outline to flesh out when
the deep-gas cost of the X-ray binary runs is worth attacking; nothing here is built.
The in-cell random walk of `compton_mrw_plan.md` is complete and is the reference this
would be tested against.

## 1. Why this and not the in-cell step

The random walk confined to one cell is exact for the diffusion equation inside a very
thick cell and pays a dead zone at every face.  On the hires XRB the `accel_report`
histogram put 9 percent of the scatterings in cells thick enough to step in at the
threshold and 60 percent in cells of one to sixteen mean free paths across their
half-width; the step replaced 6 percent of the scatterings and saved nothing.  Discrete
diffusion Monte Carlo (DDMC; Densmore, Urbatsch, Evans & Buksas 2007; frequency
dependence in Densmore, Thompson & Urbatsch 2012; the resonant-line form in Smith,
Tsang, Bromm & Milosavljevic 2018) works cell by cell at optical depths of a few, needs
no cell-crossing approximation because crossing cells is what the walk is, keeps every
cell's own temperature, density and velocity, and costs per cell and energy group
traversed rather than per scattering.  That is the regime the XRB's work sits in.  The
region step of `compton_mrw_plan.md` Section 6.7 is a crude version of the same idea
and is superseded by this plan.

## 2. The method in one page

A photon entering a cell whose optical depth exceeds a threshold becomes a diffusion
particle: it has a cell, an energy group, a weight and a time, and no position or
direction.  It evolves as a continuous-time random walk:

- **Spatial leakage.**  For each face f of cell c the rate at which diffusing radiation
  leaks through it is the finite-volume diffusion flux per unit energy in the cell,
  L_f = (2 D_c / Δx_c) / (Δx_c + Δx_n D_c/D_n) per unit length, with D = 1/(3 chi) in the
  cell and in the neighbour n, the standard Densmore form, so that in the limit of many
  cells the walk reproduces the diffusion equation on the grid to first order in the
  cell size.  Into a transport (thin) cell the face is a diffusion boundary with the
  asymptotic extrapolation length, L_f = (2 D_c / Δx_c) / (Δx_c + 2 λ D_c), λ = 0.7104.
- **Energy.**  The Kompaneets operator in its Chang-Cooper finite-volume form, which
  `vis/python/montecarlo/kompaneets_greens.py` already builds (the `Propagator` class),
  gives rates between adjacent energy groups proportional to theta n_e sigma_T c, and the
  free-free sink gives a per-group absorption rate.  Energy and space are a single
  lattice (cell, group) with a single total rate; the exponential waiting time and the
  choice of transition are drawn from it, as in the resonant DDMC of Smith et al.
- **Absorption and moments.**  Absorption is a rate on the lattice (or a weight decay
  over the waiting time); the energy density of the cell receives weight times group
  energy times the time spent, which is the exact estimator of the diffusion solution;
  the Compton source terms receive the group transitions.
- **Interface to transport.**  A particle that leaks into a thin cell is reborn as a
  transport photon on the face, direction from the diffusion-limit boundary
  distribution, energy drawn within its group, weight carried; a transport photon that
  enters a thick cell becomes a diffusion particle with its energy assigned to a group.
  The thresholds for the two conversions, the cell optical depth and the asymmetry
  between entering and leaving, are the main tuning of the method.
- **Moving medium and curved spacetime.**  Everything is a cell-centre quantity: rates in
  the comoving frame of each cell, the Doppler shift between neighbours' frames applied
  as a group transition at each hop, the tetrad at the cell centre once per cell in
  Kerr-Schild.  No per-particle frame construction.
- **Time.**  Native: the waiting times are physical times, so the cycle budget of a
  dynamic run is a stop, and the time a particle spends in a cell is its contribution.

## 3. What exists to reuse

- The Chang-Cooper Kompaneets operator with the free-free sink and its validation
  (`kompaneets_greens.py`, `doc/monte_carlo/kompaneets/`): the energy rates.
- The sphere test (`tst/montecarlo/sphere_compton`): the Compton sphere against the
  Green's function and against analog, with and without absorption, gridded over any
  number of cells, with the band and run-to-run comparison tools; the Lyman-alpha
  validation cases of Smith et al. have direct Compton analogues here.
- `band_luminosity.py` and the XRB decks and runs of the random-walk work: the
  production gate.
- The per-cell frame machinery (`ComovingFrameMatrix`, the normal-observer tetrad) and
  the block-exchange buffers, which a second particle type will need.
- The `accel_mask`, `accel_report` and decline-reason reporting pattern.

## 4. Open design questions, to settle before building

1. Thresholds: cell optical depth for conversion in each direction; whether the face
   threshold of the random walk has an analogue (a transport photon entering a thick
   cell near a face).
2. Energy groups: how many, their spacing, and the error of a group energy against the
   continuous one at the interface; whether the group index is carried across the
   interface or resampled.
3. The interface in a moving medium: the direction distribution at rebirth is in the
   comoving frame of the diffusion cell; the Doppler shift to the lab at the face.
4. Particle bookkeeping: a diffusion particle as a photon with a flag and no position,
   or a separate list; how it crosses blocks and ranks (a hop into a cell on another
   rank is a buffered photon with a cell index and no position).
5. Weight windows and roulette on the lattice; the emission of free-free photons inside
   diffusion cells (born as diffusion particles directly).
6. Verification gates in order: the Compton sphere at tau 100 over 8^3 cells against
   the Green's function and the analog run; the same with free-free; the moving sphere;
   the disk atmosphere against Feautrier; the XRB bands against `d01_emis2` and the
   cost against the analog run with the cost-file partition and the face-capped step.

## 5. Phases, to be fleshed out

- D0. Design note with the leakage, energy-rate and interface formulas written out for
  this code's conventions (comoving opacities, the general pusher's coordinate
  components, the block loop's status flags), and the validation cases adapted from
  Smith et al. to Compton.
- D1. Grey (Thomson) DDMC in flat Cartesian cells with the Densmore interface, tested on
  the Thomson sphere and the disk atmosphere.
- D2. Energy groups with the Kompaneets rates and the free-free sink, tested on the
  Compton sphere with and without absorption.
- D3. Moving medium through per-cell comoving rates and Doppler hops; the moving sphere.
- D4. Kerr-Schild through the per-cell tetrad; the XRB.
- D5. Blocks and ranks, weight control, dynamic runs.

## 6. References

Densmore, J. D., Urbatsch, T. J., Evans, T. M., Buksas, M. W. 2007, J. Comput. Phys.
222, 485.  Densmore, J. D., Thompson, K. G., Urbatsch, T. J. 2012, J. Comput. Phys. 231,
6924.  Smith, A., Tsang, B. T.-H., Bromm, V., Milosavljevic, M. 2018, MNRAS 479, 2065
(arXiv:1709.10187).  Abdikamalov, E. et al. 2012, ApJ 755, 111 (energy-dependent DDMC for
neutrinos).  Gentile, N. A. 2001, J. Comput. Phys. 172, 543 (implicit Monte Carlo
diffusion).
