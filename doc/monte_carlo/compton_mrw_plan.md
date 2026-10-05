# Compton scattering acceleration by a modified random walk: assessment and plan

Status: **Phase A done 2026-10-04 on `compton`.**  `mc_sphere_isoth` had not run since the
photon time budget was introduced: with `emission = none` it never set its block's photon
count (nothing transported) and never set `dtp` (the pusher took no step and the block
loop scattered the stationary photon forever).  Both fixed, with the user columns and the
escape-surface units of Section 3.2.  `tst/montecarlo/sphere_compton/sphere_compton.py`
runs the deck and makes the three comparisons; 1e5 photons at tau 15 take 9 s of
transport (12 million scatterings) and a minute or two of Green's function evaluation.
Results, theta = 2e-3, 1e5 photons:

| case | separation: max CDF difference, gated bins | early bin | S_mc / S_kg | escape path CDF at MC quantiles 0.1 .. 0.9 (mixed boundary) | spectrum vs convolution |
|---|---|---|---|---|---|
| tau 15, x0 1, lam 0 | 0.0065 to 0.010 | 0.027 | 1 | 0.108 0.260 0.503 0.750 0.900 | 0.0072 |
| tau 15, x0 0.03, lam 8.8e-6 | 0.005 to 0.012 | 0.028 | 0.99724 / 0.99722 | 0.109 0.260 0.509 0.757 0.903 | 0.0057 |
| tau 15, x0 0.3, lam 0.020 | 0.003 to 0.007 | 0.024 | 0.9453 / 0.9449 | 0.109 0.259 0.508 0.753 0.900 | 0.0030 |
| tau 15, x0 3, lam 1.0 | 0.003 to 0.007 | 0.019 | 0.9716 / 0.9718 | 0.103 0.250 0.497 0.746 0.897 | 0.0039 |
| tau 40, x0 1, lam 0 | 0.006 to 0.012 | 0.005 | 1 | 0.099 0.246 0.494 0.745 0.897 | 0.0049 |

So Athena's Compton kernel (Pozdnyakov sampling of a relativistic Maxwellian with the
Klein-Nishina kernel, absorption as an albedo weight at each interaction) reproduces the
Kompaneets Green's function and its free-free survival at the level the direct MC in the
kompaneets note reached; the lowest-y bin shows the 2 to 3 percent early-escape deficit at
tau 15 that the note predicted and 0.5 percent at tau 40; the escape paths follow the
mixed-boundary first-passage distribution to 0.01 at every quantile and are clearly not the
absorbing-boundary one (0.138 at the 0.1 quantile at tau 15); and the whole escaping
spectrum matches the convolution to under one percent.  Two small systematics worth a
sentence in the paper: at tau 40 the MC mean energy runs 0.5 to 1.3 percent above the
Green's function in the bins with y > 1, the size of the O(theta) relativistic correction
to the Compton heating rate that Kompaneets drops; and in the absorbed cases the early
bin's survival is 0.06 to 0.13 percent above the table at 10 to 14 sigma, the same
early-escape correlation seen in energy.

**Phase B done the same day.**  `mc_sphere_isoth` now lays the sphere over any grid (cells
in by their centres, a density floor outside, the escape surface still at `radius`), gives
the point source to the block holding the origin, and emits free-free photons through the
standard per-cell emission array; `inputs/mc/athinput.sphere_compton_grid` is 32^3 cells
in eight blocks with the grid offset half a cell so the origin is a cell centre.  Same
comparison at tau 15, x0 1 (`--template`): separation 0.003 to 0.012 in the gated bins,
early bin 0.024, escape-path CDF 0.101 0.245 0.490 0.739 0.893, spectrum 0.0105.  The
cells inside hold 0.994 of the sphere's volume and their staircase surface lets photons
out about 2 percent sooner (median tau_path 100.3 against 102.3 in one cell), which is
where the spectrum's extra 0.3 percent in CDF comes from; a finer grid or an effective
tau closes it.  Transport 11.3 s against 9.2 s in one cell.  Spherical polar is not done:
the chart needs an inner radius, and a destroy or escape boundary there eats the point
source, so it waits for either a reflecting Monte Carlo boundary or a displaced source.
Phase C is in place except the write-up's figures (`doc/monte_carlo/kompaneets/`, with
`make_doc_figures.py` pointed at the moved modules).

**Phase D, static step, built the same day** (`src/monte_carlo/mrw.cpp`, `kgreens.cpp`,
`kgreens.hpp`; the legacy step, its table readers and files, `time_acc`,
`planck_inv_opacity` and `InitializeAccelerationOpacity` are gone; `dmin` stays behind
`compute_dmin` for the hot Jupiter hook).  The C++ sampler reproduces the Python
`Sampler` on 2009 random points to 5e-14 in the table's own variable and 1.4e-14 in S
(`kgreens_port_check.py`).  The analog sphere is unchanged with acceleration off (same
numbers as Phase A to every printed digit).  On the tau 100 sphere over 8^3 cells, x0 1,
2e4 photons, `accel_tau` 10: the random walk passes the separation test and the spectrum
test like the analog run (max CDF 0.0039 and 0.0040 against 0.0027 and 0.0028), its
escape-path quantiles are 0.1 to 1.1 percent below the analog ones
(1772 2452 3693 5529 7974 against 1774 2480 3725 5583 8011 in tau_path), and it replaced
13 percent of the scatterings for a 13 percent saving in wall time: with 8 cells per
2.5 radius the half-cell is 15.6 mean free paths and only the cells wholly inside the
sphere are unmasked, so most attempts are declined (563,585 declined, 173,575 taken).  The
small shortening of the paths is in the direction the diffusion first-passage
distribution errs for a sphere of 10 to 15 mean free paths and is what the `accel_tau`
scan of Phase E is for.  Found and fixed on the way: the pushers handed a photon that a
user hook had ended at a cell face on to the boundary function, which on the gridded
sphere destroyed 9 percent of the escapers at the domain edge.  The plan's Section 6.2
otherwise stands as built; 6.3 and 6.4 next.

## 1. Where things stand

The modified random walk (MRW) of Fleck & Canfield (1984), as used by Min et al. (2009)
and simplified by Robitaille (2010), moves a photon that is deep inside an optically thick
cell to the surface of the largest sphere that fits in the cell in one step: the path
length it would have random-walked is drawn from the first-passage distribution of the
diffusion equation with an absorbing boundary at the sphere, and the photon re-emerges
isotropically at a uniform point on the sphere.  For Thomson scattering that is the whole
story.  For Compton scattering the photon's energy also changes along the way, and the
energy change given the path length is the Green's function of the Kompaneets equation
at Compton parameter y = theta tau_path, theta = kT/m_e c^2.  With free-free absorption
the Green's function acquires an energy-dependent sink and the photon carries the
survival fraction S(y) as a weight.

Three things have been done before this branch, and the branch starts from them:

1. **A legacy MRW implementation** in `src/monte_carlo/photonpusher.cpp`
   (`MRWAcceleration`, lines 278-509, with table readers at 914-1084), called from the
   Cartesian and spherical-polar pushers behind `<montecarlo> acceleration`.  It worked
   for Thomson scattering.  For Compton it drew the final energy from a Green's function
   table and then tried to correct the weight for free-free absorption after the fact,
   and that failed above sphere optical depths of about 50.  Section 2 lists what the
   survey of that code found; the short version is that it is a prototype with enough
   defects that it will be rewritten rather than patched.
2. **A new Green's function solver and quantile tables** in `/home/swd8g/kompaneets/`
   (`kompaneets_greens.py`, `KGREENS_TABLE_FORMAT.md`, `doc/kompaneets_greens_method.tex`),
   written in another session.  It solves the Kompaneets equation in ln x with an
   analytic short-time kernel and a conservative finite-volume scheme, includes the
   free-free sink exactly, and tabulates quantiles of ln(x_f/x_i)/sqrt(2y) over
   (lambda, y, x_i) together with abar = -ln S/(lambda y).  It is validated against
   Becker's (2003) exact solution to 2.5e-5 and against an independent direct Compton
   Monte Carlo (`mc_sphere_check.py`) to under 1 percent in CDF.  Its `Sampler` class
   is the reference for the C++ port.  The handoff note there also records that the
   old Green's functions were wrong at intermediate y (the small-y approximation did not
   conserve photon number: 0.41 at x_i = 1, y = 0.4) and that the old absorption tables
   averaged the opacity over the unabsorbed spectrum.  Either is enough to explain the
   earlier failure with absorption.
3. **The isothermal sphere problem** `src/pgen/mc_sphere_isoth.cpp` with
   `inputs/mc/athinput.sphere_iso`, a uniform sphere inside a single Cartesian cell with
   an escape surface at `radius` applied in `UserWorkInMove`, and twelve analysis
   scripts in `vis/python/montecarlo/problem_specific/sphere_compton/`, all written
   against a photon-list API (`athena_mc_list`, `athena_mc_spec`) that no longer exists
   and seven of them in Python 2 syntax.

The order of work is set by what each step needs from the one before: the sphere test
with ordinary (analog) Compton transport comes first, because it is the reference every
MRW result will be judged against and it is a code-paper test in its own right; the
sphere is then spread over many cells, because that is the geometry in which an MRW that
acts cell by cell can be tested; the Python is rebuilt alongside, because both tests bin
the same lists against the same Green's functions; and the MRW itself comes last.

## 2. What the survey found in the legacy code

Items that make the Compton branch unreachable or wrong, with locations:

- **Compton MRW never triggers.**  For non-coherent scattering the pushers test
  `planck_inv_opacity * dist > tauacc` (`cartesianpusher.cpp:128`,
  `sphericalpolarpusher.cpp:329`).  That array is allocated zero-filled and the only
  function that fills it, `InitializeAccelerationOpacity` (`opacity.cpp:351`), has no
  caller anywhere in `src/`; it is also internally broken (a dangling `Photon *` and an
  energy that is computed but never passed on).  The Planck-mean machinery is Robitaille's
  recipe for absorption-dominated dust and has no role for Compton, where the diffusion
  coefficient is 1/(3 n_e sigma) at the photon's own energy.
- **Wrong photon in the cell update.**  `UpdateZone(pphot,0) //SWDFIX`
  (`photonpusher.cpp:458`) relocates photon 0, not photon `ip`.
- **An extra scattering after every jump.**  `MRWAcceleration` returns with the photon
  `EVOLVING` and `taup > 0`; `TransferPhotonsOnBlock` (`montecarloblock.cpp:857-935`)
  then treats it as an ordinary interaction: albedo factor, `Scatter()` (which replaces
  the isotropic MRW direction), `nscp++`.  The jump should return as a free flight that
  ended on the sphere, not as an interaction.
- **Time is not advanced and moments are not tallied.**  `x0p` and `dtp` are untouched
  (the `path += ct` line is commented out at `:367`), and no moment estimator sees the
  path.  A photon that random-walks for 1e4 mean free paths contributes nothing to the
  energy density of the cell it did it in.
- **Absorption with Compton is a heuristic.**  Lines 388-423 form
  `tauabs = ct (opac_i - opac_f)/(2 or 3 ln(x_f/x_i))` with branches on x_i < 1 and
  x_f > 1, singular at x_f = x_i, with ten alternatives commented out.  This is the
  "adjust weights after the fact" that failed, and the kompaneets work replaces it with
  the survival fraction of the absorbed Green's function.
- **Tables and units.**  The Green's function table (`compton_table_*.out`), the
  path-length table (`time_table_*.out`) and an unused radius table are raw native
  binaries with hard-coded sizes (nt = 200, nxi = 100, np = 100; ntau = 100, np = 400)
  read from the working directory; a size mismatch fails silently.  The trigger and the
  path mix code length with cm^-1 opacities and are right only at `l_cgs = 1`.  The
  spherical-polar branch uses `dmin`, the smallest cell width with angles in radians,
  which is neither a length nor the distance to a face.  `ReadRadiusDistribution` loops
  on an uninitialized member.  The analytic Min et al. CDF built by `InitializeMRWDist`
  is dead code.
- **Boosts.**  The comoving branch (`:300-347`) and the spherical conversion of beta
  (`:444-446`, with sign errors) are untested with Compton and out of scope here.

The resonant-scattering acceleration (`MRWResonanceAcceleration`, general pusher only,
`scattering = resonance`) shares nothing with this but the `acceleration` key and is left
alone.

Items in the sphere problem generator and its tools:

- `SphericalEscape` subtracts `dr/c` from `x0p`, but the Cartesian pusher advances
  `x0p` by `dl`, a length (`cartesianpusher.cpp:159,184`); the correction should be `dr`.
  Negligible numerically, wrong dimensionally.
- The problem assumes the origin's cell holds the whole sphere: every photon starts in
  `i1start,i2start,i3start` and the free-free emission and volume of that one cell are
  rescaled to the sphere.  A multi-cell sphere needs per-cell emission and per-photon
  cell lookup.
- The user columns are weight-integrated path, weight-and-energy-integrated path and a
  weight-integrated absorption coefficient.  With absorption as a weight these are not
  the quantities the Green's function comparison needs (Section 3.2).
- Of the twelve scripts, five are byte-identical to copies under
  `/home/swd8g/kompaneets/inputs/`, the rest are older than those copies, none import
  a module that exists, and `compton_acc.py` (the generator of the legacy binary tables)
  is Python 2.  The escape-time series, the Whittaker-function Green's function and the
  free-free formula are each copied five to eight times across the set.

## 3. Phase A: the Compton sphere as a code-paper test (analog transport)

Goal: show that Athena's Compton scattering (Pozdnyakov sampling of a relativistic
Maxwellian, Klein-Nishina kernel, `ScatterComptonUnpolarized`) reproduces the Kompaneets
Green's function in the regime where Kompaneets holds, with and without free-free
absorption, and in the same runs show that the Green's function plus the diffusion
escape-time distribution predict the escaping spectrum.  Each side checks the other.

### 3.1 The comparison

`mc_sphere_check.py` already does this for its own direct Monte Carlo and is the
template.  For every escaping photon record its birth energy x_i, its escape energy x_f,
its weight, and its Thomson path tau_path = n_e sigma_T times the geometric path.  Then:

1. **Separation test.**  Bin the escapers in y = theta tau_path.  In each bin the
   weighted distribution of ln x_f must match P_Lambda(x_f, y | x_i) averaged over the
   bin's y values, and the mean weight must match S(y).  This is a test of the scattering
   kernel alone: it uses no escape-time theory.  Expected agreement is the direct MC's,
   under 1 percent in CDF and 0.1 to 0.3 percent in S, except in the lowest y bins where
   the early-escape correlation (Section 5 of the kompaneets handoff) gives 2 to 3
   percent at tau = 15 and 1 percent at tau = 40.
2. **Escape-time test.**  The CDF of tau_path against the diffusion first-passage
   distribution for a point source at the centre, both the absorbing-boundary series of
   Min et al. (eq. 7) and the finite-tau mixed-boundary series the old code used
   (`prob_esc_time`, roots of tan lambda = lambda/(1 - 1.5 tau)).  The second is the
   right one for a sphere radiating into vacuum and is what the escaping spectrum needs;
   the first is what the MRW step inside a cell uses.  Report which fits at which tau.
3. **Spectrum test.**  The escaping spectrum against the convolution
   integral p_esc(t) P_Lambda(x_f, y(t) | x_i) dt, and the escaping fraction against the
   integral of p_esc S.

Cases: x_i = 0.1 (the existing deck), 1 and 3 at theta = 2e-3 (T = 1.2e7 K) and
tau = 15, 40, 100; the same with free-free absorption at lambda set by the density
(`lam_freefree(rho, T)`; with `radius = 1e10` and T = 3e6 the existing deck gives
lambda of order 1e-2 to 1e-1, inside the table's range).  Validity caps from the
kompaneets note: theta <= 0.03, x_f theta <= 0.05, theta lambda a(x) << 1.

A monoenergetic point source at the centre (`x0`, `srcdist = false`) is the right source
for 1 and 2; the free-free-emitting sphere (`emission = freefree`, uniform in volume) is
the production-like case and is kept as a spectrum-only check.

### 3.2 Changes to `mc_sphere_isoth.cpp`

- User columns become: `user[0]` geometric path (unweighted; the pusher's `dl` summed
  in `UserWorkInMove`, which already runs every step), `user[1]` birth energy,
  `user[2]` scattering count copied from `nscp` at escape (the `scattering_histogram.py`
  convention, `--nscat-col 2`).  `tau_path` is then `user[0] n_e sigma_T`, with n_e from
  the deck's `tau` and `radius`.  The absorption integral goes: with
  `abs_method = weight` the weight is the survival and nothing else is needed.
- `x0p` correction at the escape surface in length units.
- Free-free absorption on: `absorption = freefree` with the existing
  `FreeFreeAbsorptionOpacity`, which has the same `ffnrm = 3.692146e8`, Gaunt factor 1
  and stimulated-emission factor as `lam_freefree`, so lambda is consistent between the
  code and the table.
- Composition: `heabund` is already a `<problem>` key for the gas constant but
  hard-coded to 0.09 for kappa_es; make kappa_es use the same key.

The deck gains a Compton variant `inputs/mc/athinput.sphere_compton` (and the
free-free one) with `phlist` output carrying the three user columns; `mcmom` and `spec`
outputs stay.

### 3.3 Python

A test driver `tst/montecarlo/sphere_compton/sphere_compton.py` in the pattern of
`disk_atmosphere.py`: takes `--path` and `--workdir`, writes the deck for a case, runs
it, reads the lists with `athena_mc.read_list_generator`, and prints the three
comparisons above as tables with a PASS/FAIL on the separation test (max CDF difference
and S agreement per y bin outside the early-escape bins).  The Green's function side
comes from `kompaneets_greens` (Section 5).

### 3.4 Verification criterion

The separation test passes at the direct MC's level in every case; the spectrum test
agrees to a few percent with the finite-tau escape-time series; the escape-time CDF
matches that series at the 0.1 to 0.9 quantiles to within the quoted 0.005.  Any failure
localises: separation failing with escape time passing points at the scattering kernel;
the reverse points at the escape surface or the time bookkeeping.

## 4. Phase B: the sphere over many cells

Goal: the same physics on a grid where the sphere spans many cells, so that MRW acting
cell by cell has something to be tested on, and so that the test exercises cell
crossings in the analog transport too.

- Density is `rho` for cell centres inside `radius` and zero outside (the cells outside
  only exist to hold the grid; the escape surface stays at `radius` so no photon is
  transported through vacuum).  A partially covered cell is in or out by its centre;
  the sphere's effective tau is then slightly off, and the run reports the volume-weighted
  tau actually laid down so the Python can use it.  Alternatively, in spherical polar
  coordinates the sphere surface is a cell face and the problem is exact; both charts
  are supported, Cartesian first because the escape hook is written for it.
- Photons find their own cell: the point source stays at the origin; the free-free
  source draws a cell in proportion to its emission (the standard emission path, which
  `MonteCarloProblemGenerator` currently overrides) and a position uniform in the cell,
  with the sphere boundary applied by rejection in edge cells.
- The escape hook works in either chart (radius from `x1p..x3p` in Cartesian, `x1p` in
  spherical polar; the back-up along `k` needs the Cartesian direction, which the
  spherical pusher provides through its basis helpers).
- Deck: `athinput.sphere_compton_grid` with, say, 32^3 cells over +/- 1.25 radius and a
  spherical-polar counterpart with 32 radial cells.

Verification: the Phase A comparisons repeated on the gridded sphere agree with the
one-cell sphere within errors; the escaping fraction and spectrum are the same to the
statistical precision of 1e5 photons.  Cost is noted, since the analog gridded run is
the baseline MRW has to beat.

## 5. Phase C: one set of Python tools

Layout under `vis/python/montecarlo/`:

- `kompaneets_greens.py` moved in from `/home/swd8g/kompaneets/` unchanged except for
  `np.trapz` to `np.trapezoid` (NumPy 2.3 here) and a `__main__` that still generates
  tables; with it `test_kompaneets_greens.py` and `check_table.py` under
  `tst/montecarlo/sphere_compton/`.  `mc_sphere_check.py` is kept there as the
  independent reference MC; it imports the escape-time series from the new theory module
  instead of `inputs/compton_greens`.
- `problem_specific/sphere_compton/` is emptied and refilled with:
  - `sphere_theory.py`: constants and composition in one place (the code's values),
    the escape-path CDF and pdf for both boundary conditions, `lam_freefree`, the
    Green's function binned per photon, and the convolution of the Green's function
    with the escape-path pdf.
  - `kgreens_export.py`: converts `kgreens_table*.npz` to the flat binary the C++ reads
    (Section 6.5) and reads it back to check (round trip verified on the 129 MB
    free-free table).
  The binning and plotting live in the test driver itself,
  `tst/montecarlo/sphere_compton/sphere_compton.py --lists ... --athinput ... --plot`,
  which works on any existing run; a second binning CLI would duplicate it.  The ST80
  steady-state spectrum of the old scripts is not carried over: the convolution is the
  exact prediction for this problem and ST80 is not.
- The twelve old scripts are deleted.  Their only functionality not carried over is the
  generator of the legacy `compton_table_*`/`time_table_*` binaries, which Phase D
  retires.
- The write-up, its figure script and figures, and the table format document are in
  `doc/monte_carlo/kompaneets/`; the write-up is the reference the MRW guide will cite.
  The tables themselves (4.6 MB and 129 MB) stay out of the repository: the module
  regenerates them (40 s and 6 minutes on 8 cores) and `kgreens_export.py` writes the
  binary the code reads.

Python 3 only, `import athena_mc`, no hard-coded paths or run names, argparse
throughout, same `parse_args`/`main` layout as `plot_spectrum.py`.

## 6. Phase D: the MRW rebuilt

### 6.1 Why a fixed time rather than a fixed radius

The textbook step draws the first-passage time to a sphere of fixed radius R_0.  That is
the right step only when the medium is at rest on the grid.  With `boosts` the fluid
flows through the cell, the comoving sphere's centre advects by beta c Delta t in the
lab frame, and for a fast enough flow the sphere reaches a cell face by advection before
the photon reaches the sphere's surface by diffusion.  In a dynamic run the photon also
has a time budget `dtp`, the remainder of the cycle, which a first-passage draw ignores
(the legacy code did ignore it).  And in GR the cell is a small patch of a curved
spacetime whose comoving frame is defined at an instant.  All three want the same thing:
choose the time first, then ask where the photon is.  The legacy `time_acc`, the radius
tables and `rad_dist` in the old Python were the start of this and never finished.

The fixed-time step for a point source at the centre of a sphere of radius R_0 in a
medium of extinction chi (comoving), D = 1/(3 chi):

1. Pick the comoving time budget Delta t, the smallest of: the advection limit (the
   sphere plus its advected centre must stay inside the cell, Section 6.3), the
   diffusion limit (Delta t such that the survival P(Delta t) of Min et al. eq. 7 is a
   set fraction, which keeps the step from degenerating into many tiny jumps when the
   budget is long), and `dtp`.
2. With probability 1 - P(Delta t) the photon reached the sphere before the budget ran
   out.  Draw the exit time t_e from the first-passage CDF conditioned on t_e < Delta t
   (uniform u in [0, 1 - P(Delta t)], invert the same CDF) and place the photon on the
   sphere.  Otherwise it is still inside: draw its radius from the fixed-time solution,
   p(r | Delta t) proportional to r sum_n n sin(n pi r/R_0) exp(-n^2 pi^2 D Delta t/R_0^2)
   (Min et al. eq. 6, the series whose radial integral is eq. 7), and set t_e = Delta t.
   Both branches are isotropic in position and in the new direction.
3. tau_path = chi c t_e; y = theta tau_path; energy and survival from the table as
   before (Section 6.5); time and moments advanced by t_e.

Both distributions are analytic and are tabulated once at startup: the first-passage CDF
on a grid in ln Y (what `InitializeMRWDist` already builds) and the radius quantiles on a
grid of (ln t, level).  No files.  At rest with `dtp` unlimited and the diffusion limit
set to P = 0, the step reduces to the fixed-radius one.  The sphere test of Phase E
compares the two on the same deck, which is the check that the fixed-time machinery is
right before it is asked to do anything the fixed-radius step cannot.

The early-escape correlation (Phase A: 2 to 3 percent in the lowest-y bin at tau 15) is
a property of the first-passage branch and is unchanged; the inside branch is if
anything safer, since those photons have had their full budget to thermalize their
direction.

### 6.2 The step in a static medium (Cartesian and spherical polar)

For a photon that is to interact inside a cell (the existing branch in both pushers):

- **Radius.**  R_0 is the distance to the nearest cell face.  Cartesian:
  `DistanceToNearestFace` as now.  Spherical polar: replace `dmin` with the true
  distances, |r - r_f| radially, r times the angle to the theta faces, r sin theta times
  the angle to the phi faces, all in cm (times `l_cgs`).
- **Trigger.**  tau_0 = chi R_0 > `accel_tau` with chi = scp + acp in cm^-1 at the
  photon's current energy (for Compton `scp` is the thermally averaged Klein-Nishina
  coefficient the code already uses).  Default `accel_tau = 20`, from the early-escape
  measurement; Phase E makes it a measured number.
- **Guards.**  Fall back to ordinary transport, not an error, when theta > 0.03,
  x_i theta > 0.05, theta lambda a(x_i) > 0.1, or x_i is outside the table by more than
  its clamp; count the fallbacks in the report.
- **State on exit.**  Position from Section 6.1, direction isotropic, `x0p += c t_e`
  in the pusher's units (length, like `dl`), `nscp` advanced by the Thomson scatterings
  tau_path implies, `UpdateZone(pphot, ip)`, opacities recomputed at the new energy.
  Return as a completed free flight (`taup = -1`, no interaction) so the block loop
  neither applies the albedo nor scatters.  Moments: the path lies in the cell, so the
  path-length estimator gets w c t_e for the energy density at the mean of the birth and
  final energies, zero net flux, pressure J/3; absorbed energy w (1 - S) times the photon
  energy to the heating estimator.  Exact in expectation over many steps, approximate
  for one; the guide says so.
- **Weight window and path stretching** are unaffected: both act at interactions and
  the MRW step is a flight.  `stretch_taucell` already excludes thick cells.

### 6.3 Moving medium (`boosts`)

The walk is done in the comoving frame, where the medium is at rest and the Green's
function applies; the frame changes are the ones the block loop already makes around a
scattering (`TransformToComoving`, `TransformToCoordinate`), so the energy, direction and
opacities entering and leaving the step are in the right frame without new code.  What is
new is the geometry:

- **Advection.**  Over comoving time Delta t the lab time is gamma Delta t and the fluid
  element moves beta c gamma Delta t.  The lab-frame constraint is that the sphere, of
  comoving radius R_0 (contracted by 1/gamma along beta, which for the velocities of a
  disk is ignored and noted), plus the advected centre stays inside the cell: for each
  face, the distance along the face normal must exceed R_0 plus the advection toward it.
  Solved by choosing Delta t so that the advection uses at most a set fraction of the
  nearest-face distance and R_0 is what remains; when the resulting tau_0 falls below
  `accel_tau` the step declines.  This is what the legacy `r0 + delta beta gamma chi r0^2 = dist`
  was after, with the time chosen first instead of the radius.
- **Position and time on exit.**  Lab position = entry + beta c gamma t_e + the drawn
  comoving displacement (rotated isotropically; the contraction is the same 1/gamma
  neglect).  `x0p` advances by the lab time c gamma t_e.  y uses the comoving path.
- **Moments.**  Deposited as an isotropic comoving field and transformed with the
  existing comoving-to-coordinate moment transformation in `photon_frames.cpp`
  (`doc/monte_carlo/frames_and_moments.tex`), which is the one place this needs frame
  care.
- **Spherical polar** has the beta conversion that the legacy branch got wrong
  (`:444-446`); it is written once, through the pusher's basis helpers, not by hand.

Verification: the Phase B sphere with `velocity` set, a uniform flow through the fixed
escape surface, MRW against analog in the lab frame, at beta of 0.01, 0.1 and 0.3.

### 6.4 General relativity (the general pusher)

The idea is the same one step further: the cell is a small patch of spacetime, and
inside it the walk is flat-space diffusion in the fluid's local comoving tetrad, with
the spacetime's effects applied at the ends.  Only in cells that are very thick, so that
the walk is short compared with any scale on which the metric changes; `accel_tau` for
the general pusher is a separate key with a higher default, 50 say, pending Phase E.

- **Frame.**  The photon is projected into the comoving tetrad at its position
  (`MCFRAME_COMOVING` in `photon_frames.cpp`, which `GeneralPusher` already uses around a
  scattering), the step is taken there in proper time and proper length, and the result
  is projected back with the tetrad at the exit point.  Using the exit-point tetrad is
  what the user calls the after-the-fact correction: the change of redshift and of the
  fluid velocity across the displacement enters through the difference between the two
  tetrads, first order in the cell size, and nothing in the step itself knows about it.
- **Geometry.**  Face distances are proper distances in the tetrad frame: the
  coordinate distance to each face along its normal, scaled by the metric at the cell
  (sqrt of the appropriate g_ii for a diagonal spatial metric, the full projection
  otherwise), and the advection constraint of Section 6.3 with the tetrad's spatial
  velocity relative to the coordinate grid.  The displacement is mapped to coordinates
  with the tetrad legs, delta x^mu = e^mu_a delta x^a, and the coordinate time advance is
  the time leg's contribution, u^t times the proper time plus the spatial legs' t
  components, which is where frame dragging enters.  The photon's cell is then found
  from its new coordinates as after any step.
- **What is approximated.**  Curvature over the displacement, the non-uniformity of the
  fluid velocity over the cell, and the second-order redshift across it.  All are
  controlled by the cell size over the local scale and by only triggering in the thick
  cells, and none is assessed by a sphere test.  Which is why:
- **Validation is the XRB** (Phase E): accelerated against non-accelerated runs of the
  mc_readhdf_gr snapshot, spectra and images, with the asymmetric cost savings measured
  cell by cell.  The hard band of the XRB spectrum is made by hundreds of scatterings in
  deep gas (the biased-sampling guide), which is exactly the walk this replaces, so the
  payoff and the risk sit in the same place.
- **Polarization.**  The step returns an unpolarized photon in the comoving frame; the
  tensor machinery then carries it as usual.  Stated, not derived, as before.

Order: 6.2 first on the Cartesian and spherical pushers, then 6.3 on the same two
(the fixed-time machinery is shared), then 6.4 on the general pusher, each with its
verification before the next.

### 6.5 Energy and survival (Compton), and the table file

y = theta tau_path with theta from the cell temperature.  x_f from the quantile table at
(lambda, y, x_i) by the algorithm in `KGREENS_TABLE_FORMAT.md` Section 4, S from abar by
its Section 4.3; weight times S, with the ordinary `minweight` roulette after.  lambda
per cell from the same `ff_cell` prefactor the opacity uses.  Where `absorption` is not
free-free the lambda = 0 slice is used with the grey factor exp(-acp c t_e).  Coherent
scattering: energy unchanged, survival exp(-acp c t_e).

`<montecarlo> kgreens_file` names a flat little-endian binary written by
`kgreens_export.py`: a magic string, the four dimensions, then `lam`, `y`, `xi`,
`levels` as float64, `uq` as float32 in C order, `abar` as float64 and `dead` as uint8;
read with `std::ifstream`, so no HDF5 dependency (decided).  The reader checks the magic
and sizes and refuses the run otherwise; rank 0 prints the table's ranges once.  Port
check: the four `(xi, y, lam, r)` cases of the format document's Section 7 plus a
dead-cell neighbour against the Python `Sampler`, to float32 round-off, in
`tst/montecarlo/sphere_compton/`.

### 6.6 What goes

`time_acc`, `compute_dmin` and `dmin`, `planck_opacity`/`planck_inv_opacity` and
`InitializeAccelerationOpacity`, `ReadComptonGreensFunction`, `ReadRadiusDistribution`,
`ReadTimeDistribution`, `InterpComptonEnergy`, `InterpPathTime`, the `mrw*` table arrays,
the legacy `compton_table_*`/`time_table_*`/`radius_table_*` files and their generator,
and the body of `MRWAcceleration`.  `accel_tau` replaces the hard-coded 10.  The
resonance branch in the general pusher is untouched.

### 6.7 Cells, not spheres: the dead zone and two ways out

The sphere confined to the cell triggers only where the face distance exceeds
`accel_tau`, a fraction ((L - tau_a)/L)^3 of a cubic cell of half-width L mean free
paths, and every step lands the photon on the sphere, at distance zero from a face, from
where it random-walks about tau_a^2 scatterings before it can step again.  Measured on
the gridded sphere (8^3 cells, x0 1): tau 100 with tau_a 10 (L = 15.6, 4.6 percent of
the volume active) replaced 13 percent of the scatterings, tau_a 5 (31 percent active)
37 percent, and tau 1000 with tau_a 20 (L = 156, 66 percent active) 66 percent.  The
dead zone near the faces sets the efficiency, and it is severe unless cells are many
times thicker than the threshold.  Two remedies, in the order they should be tried:

1. **The cell as the diffusion domain.**  For a rectangular box with absorbing walls the
   diffusion Green's function separates: the survival is the product of three
   one-dimensional survivals S_i(xi_i, s), each a function of the start position across
   the cell's width in that direction, S(xi, s) = sum_n c_n(xi) exp(-n^2 pi^2 s') with
   the usual sine series.  The first-passage time from any interior point is drawn by
   inverting 1 - prod S_i (bisection on tabulated ln S_i); the exit direction is i with
   probability proportional to -S_i' prod_{j != i} S_j at that time, the exit side from
   the 1D hitting probabilities, and the positions in the other two directions from the
   1D conditional densities at that time, all 1D tables in (start, time) built once at
   startup like the present ones.  The fixed-time budget carries over (still inside at
   s_b: three 1D positions), advection by shrinking the box on the downstream side as f
   does now, and the medium stays exactly uniform with the moments in one cell.  A
   photon half a mean free path from a face steps straight out through it.  Only a
   Cartesian box, but a spherical-polar cell is a box in its local orthonormal
   coordinates to the accuracy the sphere step already assumes, and so is the
   tetrad-frame cell of Section 6.4.  Cost: three lookups and one inversion per step.
   No new approximation, so this comes first, after the sphere step's validation.
2. **A region of similar neighbouring cells** (the user's suggestion) is what buys
   efficiency where cells are thinner than the threshold but a block of them is thick.
   It needs a similarity test (extinction within a factor, temperature within a few
   percent, velocity difference small against the Doppler scale that matters), the step
   taken with volume-averaged properties, moments split by the region's volume in each
   cell, and the comoving frame or tetrad of the cell the photon lands in; each is an
   approximation whose size is set by the problem.  Its natural form is a superzone box
   of n^3 cells using the machinery of item 1, not a sphere.  Decided after the XRB.

To decide with data, the run report gets a histogram of scatterings by the cell's
optical half-width (`accel_report`): on the XRB it says how much of the deep-gas work
sits in cells thick enough for in-cell steps and how much only a region could reach.

Measured before the cell domain existed, tau 1000 over 8^3 cells, 2000 photons, sphere
domain with `accel_tau` 20: the walk stood for 66 percent of the scatterings and the run
took 245 s against 700 s analog, 2.9 times faster, so the steps themselves cost nothing
visible and the saving is set by the dead zone alone; escape-path quantiles
175k 239k 357k 536k 787k against 169k 239k 354k 531k 792k.  The user chose the cell
domain (item 1) and no cell crossing (item 2) unless it proves necessary; built as
`accel_domain = cell` (default) with the sphere kept as `accel_domain = sphere`, and
`accel_face_tau` the extinction distance from a face below which a photon scatters
normally rather than stepping out through it, since the diffusion first-passage time is
wrong within a mean free path or two of a wall.

**Cell domain measured** (tau 100 over 8^3 cells, L = 15.6, `accel_tau` 10, 2e4 photons,
against the analog run of the same deck): with `accel_face_tau` 2 the walk stood for 61
percent of the scatterings (36 s against 68 s), the spectrum agreed to 0.8 of the
two-sample Kolmogorov scale, but the escape paths came out 3 to 4 percent short at 2.7
times that scale (1720 3582 7726 against 1774 3725 8011), the early-escape error of a
photon stepping from two mean free paths off a wall; with `accel_face_tau` 5, 45 percent
of the scatterings, spectrum 0.7 and paths 1.1 times the scale (1750 3674 8003).  So the
default is 5.  At tau 1000 (L = 156) the cell domain stood for 84 percent of the
scatterings and took 125 s against 700 s analog, 5.6 times faster where the sphere gave
2.9.  Building the box tables adds about 5 s to the setup of every run with
acceleration on.  Rerun at `accel_face_tau` 5, tau 1000: 81 percent of the scatterings,
144 s against 700 s, spectrum 0.9 and paths 1.2 times the two-sample scale (median
3.51e5 against 3.54e5), and `accel_report` puts every scattering in cells of optical
half-width 128 to 256, as it should with L = 156.

**Moving medium measured** (Section 6.3 as built, the sphere with a uniform flow along z
through the fixed escape surface, tau 100 over 8^3 cells, `accel_tau` 10,
`accel_face_tau` 5, 2e4 photons, walk against analog of the same deck): beta 0.1,
spectrum 1.3 times the two-sample scale, mean escaping energy 1.2 percent low, escape
paths 1.5 percent short (median 860 against 873; the flow carries photons out, so the
paths are a quarter of the static ones), 35 percent of the scatterings replaced with
179k attempts declined for want of advection room; beta 0.3, spectrum 0.6, mean energy
0.6 percent low, paths 2 percent short (318 against 323), 32 percent replaced.  The
energy deficit tracks the path shortening (y is 1.7 here, so a percent of path is a few
percent of e^{4y}) rather than growing with beta, so it is the face-threshold
early-escape effect of Section 6.7 seen through Comptonization, not the neglected
contraction.  A face-threshold scan at a Compton-thick setting is the Phase E item that
sets the default for production.

**General pusher step built** (Section 6.4 as written: `GeneralPusher::MRWFrame` makes the
box from the cell's coordinate edges in the normal observer's tetrad, with the fluid's
velocity on its legs; `MRWStep` is otherwise shared).  Tested in flat spacetime with
`general_pusher = true` on the Cartesian sphere, which needed three repairs outside the
step: `vel` was allocated only under boosts or polarization, so the general pusher's
emission transform dereferenced nothing in a static unpolarized run; the sphere
generator's escape hook backed photons up along `k` as a unit vector, which under the
general pusher carries the energy; and the general pusher's fixed default affine step of
1e-3 moved a photon by picometres across a 1e10 cm cell, so the driver sets `varystep`
and `stepsize = 0.02` with `--general-pusher`.  Static tau 100: the general pusher's
analog run passes the Green's function tests like the Cartesian one, and the walk
against it agrees to 0.6 (spectrum) and 0.8 (paths) of the two-sample scale with 44
percent of the scatterings replaced.

A test-geometry trap found on the way, worth knowing for any gridded sphere with a flow:
the cells outside the sphere carried the flow's momentum at a density of 1e-10 of the
sphere's, below the hydro density floor; flooring the density up and keeping the momentum
turns the freed kinetic energy into heat, 3e9 K at 0.1 c, and the few photons that
scatter in those cells come out with absurd energies (a 14 percent tail gaining up to
700 times in a cold-electron test).  The outside cells now carry no velocity and the
default floor is 1e-6 of the sphere density.  The general pusher saw this where the
Cartesian pusher did not, for reasons not pursued once the cause was clear; with the
geometry fixed the two pushers agree.

Final flat-spacetime numbers on the fixed geometry (tau 100 over 8^3 cells, beta 0.1
along z, 1e4 photons, `accel_tau` 10, `accel_face_tau` 5), each against the run named:
Cartesian analog against static, mean energy +4.1 percent (the Doppler shift of the
escapers); general-pusher analog against Cartesian analog, spectrum 0.7 of the two-sample
scale, mean energy +1.1 percent, paths 1.3; Cartesian walk against Cartesian analog,
0.7, +0.4 percent, paths 1.6 (1 percent short); general-pusher walk against its analog,
0.5, -0.8 percent, paths 2.5 (2.3 percent short); the cold-electron pair (theta 1e-5,
x0 100, recoil only) 1.7 and +0.9 percent between pushers.  The walk replaced 35 percent
of the scatterings in both pushers here, with 90k to 150k attempts declined for want of
advection room.  What remains of Phase D is the curved-spacetime validation, which is
the XRB of Phase E.

## 7. Phase E: verification

**Sphere, static** (Phase B grid), MRW against analog on the same deck, with and
without free-free, tau 40 and 100, x_i 0.1, 1, 3: escaping spectrum, escaping fraction
and tau_path CDF within the analog errors; the per-y separation test still passing,
which checks the C++ sampler end to end; fixed-time against fixed-radius step on the
same deck; cost (scatterings per escaper, wall time) against `accel_tau`; the
early-escape bias measured as the spectrum difference at `accel_tau` of 10, 20, 30, 50,
which sets the default.  The one-cell sphere as the degenerate check that MRW declines
when the cell face is beyond the escape surface.

**Sphere, moving** (Section 6.3): beta 0.01, 0.1, 0.3, lab-frame spectrum and escaping
fraction, MRW against analog.

**Thomson regression:** `mc_isoth` with `acceleration` on (the
`velocity_reconstruction_plan.md` Phase 0 deck) unchanged within errors.

**XRB** (Section 6.4): the mc_readhdf_gr snapshot of the biased-sampling work, emission
weights, accelerated against non-accelerated at equal photon count: total and band
luminosities (0.1 to 0.3, 0.3 to 1, 1 to 3, 3 to 10 keV), the spectrum shape, the image,
and the cost ratio; `accel_tau` scanned; the fraction of scatterings replaced and in
which cells.  The non-accelerated run and its errors exist already (`d01_emis2`,
`/PellaShared/swd8g/hires/`).  Acceptance is the biased-sampling standard: bands within
the analog run's own output-to-output scatter.

## 8. Out of scope, recorded so it is not re-asked

- Resonant-scattering acceleration.
- Induced (stimulated) scattering in the Green's function; the kompaneets note explains
  why it was deferred and what it would take.
- Lorentz contraction of the comoving sphere and curvature within a cell; both are
  neglected with a note, and the XRB comparison is what bounds them.

## 9. Decisions

- Tables are flat binaries, not HDF5 (confirmed by the user).
- The legacy binary tables and their generator are retired.
- `accel_tau` defaults to 20 (Cartesian and spherical polar) and 50 (general pusher)
  pending Phase E.
- The kompaneets write-up moves into `doc/monte_carlo/`; the Python module into
  `vis/python/montecarlo/`.
