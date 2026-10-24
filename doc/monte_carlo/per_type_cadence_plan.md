# Per-type transport cadence

Plan for letting each emission type of a `multi` run transport at its own cadence in a
dynamic run, so that cheap photons (the hot Jupiter's photoionizing continuum) run every
cycle while expensive ones (its Lyman alpha) run every N cycles. Written 2026-10-24.

## Status

Built on the branch `cadence_update` (2026-10-24), steps 1 to 5 of the plan; step 6 (the
wiki text) is proposed. Before the work started the `spec` writer's `nsrun / nsamp`
factor was removed (see "Outputs") and `inputs/mc/athinput.hot_jupiter` got `ntype = 2`,
which its two nonzero sample counts require (it had 1, inherited from the deck it
replaced, and could not start), and lost the `*_emis` flag lines that nothing reads.

Tests, one rank, hot Jupiter quick deck (3 x 3 cells, ionizing photons only, coupled):
`cadence = 1` gives photon lists, spectra, `mclab`, `cooling` and `prim` bitwise
identical to the branch before the change, and `mcsrc` to 2e-16 (the held rate is stored
once rather than normalized and un-normalized every cycle); `cadence = 5` gives weights
carrying the five-step interval, moments to 3e-13, and a spectrum luminosity 1.7e-5 off,
the time-weighted average of a slowly varying emissivity against the old step-weighted
one. Two types on the `combo` deck made dynamic (stellar Lyman alpha and photoionization,
frozen fluid, 60 and 1000 photons, 6 cycles): `ion_cadence = 1, lya_cadence = 5` against
`cadence = 1` runs the Lyman alpha at cycles 0 and 5 only (159 s against 521 s), the
domain-averaged neutral fraction agrees to 0.2 percent, the per-type moment totals within
the noise of 60 photons, the ionizing luminosity to 0.1 percent with identical covered
times, and the Lyman alpha energy summed over the run to 0.3 percent. Within an output
interval shorter than the Lyman alpha cadence the emission is credited to the interval in
which the type runs (one step's worth at cycle 0, five at cycle 5), which is inherent:
output intervals should span the longest cadence.

Recombination Lyman alpha is not a test case: it needs an acceleration that does not
exist (2.4 s per photon even for the stellar line on the full mesh, with a heavy tail).

## What the code does today

A dynamic run calls `MonteCarlo::RunMonteCarlo` every cycle (`main.cpp`). The call
returns at once unless `TransportDue()` says a transport is due: the first cycle of a
run or restart, every `cadence` cycles after the last transport, or, with
`cadence_frac > 0`, when the held energy source would change a cell's internal energy by
more than that fraction. On a transport cycle it

1. sets `tint = pmesh->dt` (the emission integration time, in cgs) and advances
   `ttransport`, the simulation time any transport has covered;
2. rebuilds the per-cell arrays from the fluid, resets the moments and the source terms
   of every block;
3. loops `etype` over the `ntype` emission types: `DistributeSamples(etype)`, which
   computes each block's emission array as `GetEmission(...) * tint * vol` (or `area`)
   and deals out `nsamptype[etype]` samples; the transport; the diagnostics; and
   `UserWorkAfterTransfer(etype)`.

Between transports `CoupleMonteCarloToFluid` applies the source terms as rates
(`sourceterms / (tint * vol)`) every cycle, and the outputs normalize spectra and lists
by `ttransport` rather than by elapsed time, since photons were emitted only on
transport cycles.

What is already per type: `nsamptype`, `weight_scheme`, `initialize_comoving`,
`absorption_method`, `GetEmission`, `emission_geometry`, `emission_face`, the moments
(`moments(ntype, nmom, k, j, i)`, likewise `moments_com` and `moments_coord`),
`UserWorkAfterTransfer(etype)`, and the loop body itself. What is per transport, and
therefore shared by the types: the due decision, `tint`, the reset of the moments and
source terms, and the source terms themselves, which are one array summed over types.

## Assessment of the nesting

The worry was that the type loop sitting inside one transport call makes a per-type
cadence a restructuring. It does not. The loop body is already a complete per-type
transport; the four things outside it that are shared can each be made per type by a
small edit, and the loop then simply skips the types that are not due. The one piece of
real work is the source terms: because they are one array that is reset at the start of
every transport, skipping a type would drop its contribution from the coupling. They
need to be held per type. Everything else is bookkeeping.

## Design

### Per-type state in `MonteCarlo`

Three arrays of length `ntype`, beside the existing per-type arrays in
`InitializeEmission`:

- `cadence_type[etype]`, defaulting to `cadence` for every type, so that the global key
  keeps its meaning as the cadence of every type that is not given its own;
- `last_transport_cycle_type[etype]`, `-1` until the type's first transport, as
  `last_transport_cycle` is now;
- `tint_type[etype]`, the simulation time the type's next emission has to cover: the
  sum of `pmesh->dt` over the cycles since the type last ran, accumulated every cycle
  in `RunMonteCarlo` before the due test (in code units; converted to cgs where `tint`
  is today).

A problem generator sets `cadence_type` for the types it defines in
`InitUserMonteCarloData`, where it already sets their weights and geometry; the hot
Jupiter reads `<problem> ion_cadence` and `lya_cadence` (the latter for both Lyman alpha
types) with `cadence` as the default of each. A generic `<montecarlo>` key naming a
cadence per type is not needed for this and is deferred: the type order is the
generator's, so only the generator can name a type meaningfully.

`cadence`, `last_transport_cycle` and `ttransport` keep their roles. `ttransport` stays
the time covered by any transport, which is what the outputs normalize by; see the
argument under "Outputs".

### The due decision

`TransportDue()` becomes a loop that marks each type due when
`last_transport_cycle_type[etype] < 0` or
`ncycle - last_transport_cycle_type[etype] >= cadence_type[etype]`. The `cadence_frac`
trigger stays global: when the held energy source would overshoot, every type is marked
due, since the trigger measures the total heating and cannot attribute it. The call
returns if no type is due. `last_transport_cycle` and `ttransport` are updated whenever
at least one type runs, so a run whose cheapest type has cadence 1 has `ttransport`
equal to the elapsed time, as it should.

### The type loop

For each due type the loop sets `tint = tint_type[etype]` (converted as `tint` is today)
before `DistributeSamples`, so that the emission array, the photon weights and
`UserWorkAfterTransfer(etype)` all see the type's own interval, and resets
`tint_type[etype]` and `last_transport_cycle_type[etype]` after. `tint` as a member is
then "the interval of the type being transported", which is what every reader inside the
loop already assumes; readers outside the loop are dealt with below. Types not due are
skipped, and the per-type diagnostics print only for the types that ran.

### Moments

`ResetMoments()` and `NormalizeMoments()` already loop over `m < ntype`. Reset only the
slot of each type that runs (`ResetMoments(etype)`), and normalize slot `m` by
`tint_type` as it was at that type's last transport rather than by the single `tint`;
this needs the cgs interval of the last transport kept per type (`tint_last_type[etype]`)
since `tint_type` has started accumulating again by output time. The scattering moments
`Jnu*` and their errors have to be checked for the same structure before this step: if
they are summed over types they join the source terms below.

### Source terms

Today `sourceterms` is one unnormalized array summed over all types, reset per
transport, and consumed by `CoupleMonteCarloToFluid` (normalize, apply, unnormalize),
`HeldEnergyFraction`, the `mcsrc` output, and the hot Jupiter's `UserWorkAfterTransfer`
(which reads `MCNABS` and adds its impact cooling to `MCRS0`).

Keep `sourceterms` as the accumulator of the type being transported, and add a held
array of per-type rates, `srcrate(ntype, nsrc, k, j, i)`, plus its error counterpart
for the flux force. At the end of each type's transport (after
`UserWorkAfterTransfer(etype)`, so that the hot Jupiter's cooling is included), convert
the accumulator to a rate, `sourceterms / (tint * vol)`, store it in the type's slot and
reset the accumulator. The three consumers then use the sum over types of the held
rates: `CoupleMonteCarloToFluid` applies `dt * Σ_t srcrate(t)` directly and loses its
normalize/unnormalize pair; `HeldEnergyFraction` reads the summed rate; the `mcsrc`
writer sums the rates and combines the errors in quadrature. With `ntype = 1` the held
array is one copy of the accumulator and the applied source is unchanged to the bit.

The alternative, subtracting a re-run type's previous contribution from the summed array
before adding its new one, needs the same per-type storage and keeps the awkward
unnormalized representation, so it is not simpler.

### Outputs

No change to the normalization of lists and spectra. A type's photon weights carry
`tint_type`, the time since that type last ran, so over an output interval the summed
weight of each type is its emission over the whole interval whatever its cadence, and
dividing by `ttransport - last_ttransport` gives the mean luminosity of every type
alike. The moments are normalized per type as above.

One pre-existing hazard, removed before the work started (2026-10-24): the in-run
`spec` writer scaled its intensity by `nsrun / nsamp` when the two differed, where
`nsrun` counts the photons run since the last output and `nsamp` is the samples of one
transport. That was a leftover of an older scheme: the weights already carry the
integration time of their transport, so the factor overcounted a dynamic run by the number
of transports in the output interval. Measured on the hot Jupiter quick deck at
`cadence = 3` with five transports in one interval, the in-run spectrum was 5.007 times
the spectrum made from the same run's photon list; with the factor removed it is 1.001
times it (the lists of the two binaries are bitwise identical). Under per-type cadences
`nsrun` would in any case have been a mixture of types that `nsamp` does not describe.

### Hot Jupiter

`HoldIonization` tests `pmc->cadence > 1`; it becomes `cadence_type[ion_str] > 1`. With
`ion_cadence = 1` the ionization update returns to the direct path, which is the point of
the exercise: the held-rate machinery in `UserWorkInLoop` is then needed only when the
ionizing photons themselves are run at a cadence. `UserWorkAfterTransfer` reads
`pmy_mc->tint`, which inside the loop is the ion type's interval, so it is unchanged.

### Load balancing and restarts

The held rates are module arrays, so they travel with a block through the Monte Carlo
redistribution hooks like the moments do (`PackArray`/`UnpackArray` beside
`sourceterms`). `last_transport_cycle` is not in the restart file today (a restart
transports on its first cycle); the per-type arrays get the same treatment, so a restart
runs every type once and the cadences resume from there.

## Steps

1. Check whether the scattering moments `Jnu*` are per type or summed over types.
2. The per-type state, the due loop, `tint` set per type inside the loop, per-type reset
   and normalization of the moments. With every `cadence_type` equal to `cadence` the
   code must be bitwise identical to `main` on one rank: `inputs/mc/athinput.hot_jupiter`
   at `cadence = 1` and at `cadence = 5`, and a dynamic `ntype = 1` run.
3. The held per-type source rates and the three consumers. Same bitwise test.
4. The hot Jupiter keys and `HoldIonization`. Test: `ion_cadence = 1`, `lya_cadence = 10`
   against the `cadence = 1` run. The ionization state and the ionizing moments must agree
   to round-off (the ion type runs identically every cycle and the Lyman alpha does not
   feed back on it), and the Lyman alpha spectrum and moments statistically, with the
   cost dropping by close to the Lyman alpha share of the `cadence = 1` run.
5. Pack/unpack of the held rates; the load-balancing test of
   `tst/montecarlo/amr_transport/run_tests.py` (its dynamic case) must pass.
6. Wiki: Dynamic Runs gets a paragraph under "Transport cadence" on per-type cadences and
   how a generator sets them; Problem Generators gets the hot Jupiter keys; The Input File
   is unchanged. Proposed text goes to the user for approval before editing.

## Size

About 150 to 250 lines: the per-type arrays and the due loop in `montecarlo.cpp`, the
held rates and their conversion in `montecarloblock.cpp` with three consumers switched to
the summed rate, the per-type reset and normalization of the moments, the pack/unpack
lines, and a dozen lines in `mc_hotjupiter.cpp`. No function changes its signature except
`ResetMoments`, which gains the type.
