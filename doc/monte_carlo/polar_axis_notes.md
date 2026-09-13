# The general pusher at the polar axis

Status: known defect, fixes deferred. Written up 2026-09-12 so the work is specified when
it is picked up. Companion to `polarization_conventions.tex` and `faraday_rotation_plan.md`.

## What goes wrong

On a spherical grid whose theta range reaches the pole (`x2min = 0`, `ix2_mc_bc = polar`),
the general pusher loses and corrupts a small fraction of photons at the axis. The legacy
spherical pusher does not, because it moves photons in Cartesian components internally and
never produces a theta outside `[0, pi]`.

Two mechanisms, both seen in `tst/montecarlo/disk_atmosphere` (about 0.02 percent of
photons destroyed under the general pusher, none under the legacy pusher, on the same deck):

1. **The polar boundary is a stub.** `Polar()` in `src/monte_carlo/mcbvals.cpp` prints a
   warning and sets the photon `DESTROYED`. Any general-pusher photon whose theta
   integration crosses zero is lost. About seventy percent of the destroyed photons die
   here. The photons that reach the axis are travelling near `mu = +-1`, so the loss is a
   bias against face-on directions.

2. **The integrator runs away close to the axis.** The spherical connection carries
   `cot(theta)` and `1/(r sin theta)`. Within a few hundredths of a degree of the axis the
   geodesic step goes unstable: photons have been caught by `CheckZone` and
   `GetPositionIndices` at theta of thousands of degrees and radii twenty-five times the
   outer boundary. A runaway photon that happens to leave the domain through an escape face
   instead of being caught is counted as an escaper, carrying its full weight and a
   meaningless direction. One deep-midplane photon of that kind, forty-five times heavier
   than the heaviest legitimate escaper, was enough to inflate the pooled grazing-band error
   of an eight-million-photon run six-fold. This is the reason the general pusher's grazing
   sign check in the disk test can read under-powered while its convergence norm agrees with
   the legacy pusher's.

Neither is a polarization issue. Both apply to the Kerr-Schild decks (`kerr_frames`,
`mc_isoth_gr`), which also run theta from `0` to `pi` with polar boundaries, so production
general-relativistic runs carry the same loss and the same rare outliers.

## Deferred fix 1: a reflecting polar boundary

Replace the stub with the coordinate identity across the axis: `theta -> -theta` (or
`2 pi - theta` at the other pole), `phi -> phi + pi`, with `k^theta -> -k^theta`, the cached
Verlet derivative `dk^theta -> -dk^theta`, the coherency tensor's components with exactly one
theta index negated, the pusher's cached connection invalidated, and the cell indices
recomputed from the new position.

**Why it is deferred.** The `phi + pi` jump is a communication problem, not an algebra
problem. The Monte Carlo exchange (`mcexchange.cpp`) routes a `BUFFERED` photon between
face-neighbor blocks in a symmetric peer scheme; the block across the pole is at `phi + pi`
and is not a face neighbor in that scheme unless the whole azimuthal range lives in one
block. Delivering the reflected photon needs either the polar-neighbor topology Athena++'s
hydro boundary values already know about, or a position-based fallback in the exchange.
That is the piece to design first; the reflection itself is a dozen lines.

A single-photon regression belongs in `tst/montecarlo/spherical_polarization` when this
lands: a ray aimed through the axis, checked for the correct exit direction and an unchanged
coherency tensor in the global Cartesian frame, in the same style as the existing sweeps.

## Deferred fix 2: an axis-aware step cap

`GeneralPusher::StepSize` (`generalpusher.cpp`, near line 600) takes the step as the smallest
coordinate cell crossing with no knowledge of the axis. Directly below it sits a
commented-out block that cut the step a hundred-fold in cells touching `theta = 0` or `pi`
whenever the azimuthal crossing was the limiter: an earlier attempt at this problem.

**Why it is deferred, and the constraint on any retry.** That cap was found only marginally
effective. The design constraint is that it must not trigger too easily: a cap that fires
routinely slows the whole integration, which is the cost that made the earlier version not
worth keeping. Any retry should be keyed on the actual stiffness, the size of the connection
term relative to the step, rather than on cell position, and measured against the runaway
count in the disk test before being adopted.

## Diagnostics that see it

- `ndes` in the run tally: destroyed photons, zero under the legacy pusher on the same deck.
- `verbose = true` in `<montecarlo>` dumps each destroyed photon's position; theta values
  outside `[0, pi]` or radii outside the domain are the runaway signature.
- In the disk test, the escaper-weight accounting **by count** is stable across runs while
  the accounting **by weight** is not; a large heaviest-dropped to heaviest-kept ratio marks
  a runaway carrying a deep photon's weight.
