# Load balancing Monte Carlo transport on the Athena++ mesh: assessment and plan

Status: **L0, L1 and L2 implemented 2026-09-14/15 on `load_balance`; L3 onward not
started.**  L2d was run on a dynamic variant of the thin spherical-polar deck
(`dynamic = true`, `tmax` large, six hydro cycles, four ranks) rather than the
hot-Jupiter deck, which does not finish a cycle in useful time at any photon count:
forced moves of 17 to 21 blocks every cycle through the mesh's own per-cycle call with
counts conserved and spectra within noise; `balancer = automatic` (hydro time plus
transport time) 1.23 -> 1.09 after one move and then held.  The prediction guard is
shared with dynamic runs: when the mesh would choose a partition that is no better, the
module holds the mesh's cycle counter one short of the interval so the check waits a
cycle, which removed the rebuild-to-an-identical-layout the mesh otherwise did every
cycle.  Two pre-existing dynamic-mode defects were found and fixed in passing:
`ResetMoments` zeroed only the lab array, which is unallocated when just user moments
are enrolled (a segfault at the top of every dynamic cycle), and left the other frames
accumulating across cycles; and `Photon::ClearBoundary` was called for static runs only,
so a dynamic run with more than one block per rank hung at its first same-rank hand-off.  L2 as built: `MonteCarlo::BalanceStatic`
at the top of `RunMonteCarlo` hands control to the mesh's balancer under
`<loadbalancing> balancer = automatic|manual`, `tolerance`, `interval` (set
`interval = 1` to consider every output interval); a prediction guard runs the mesh's
partitioner on a copy first and redistributes only if the busiest rank improves by
`<montecarlo> lb_min_gain` (default 0.05), because the greedy contiguous partition can
be worse than the current layout at few blocks per rank (the thin deck at 16 ranks,
two blocks per rank, oscillated 1.26 -> 1.59 -> 1.43 -> 1.59 without it and holds at
1.28 with it; a better partitioner is the lever there, section L5); forced moves for
tests via `<montecarlo> lb_test_costs = alternate` with `balancer = manual`;
`<loadbalancing> cost_file` written after every transport and read at startup, in
which case the first transport is already balanced.  Measured on the thin deck: 4
ranks 1.21 -> 1.08 after moving 3 blocks, then held; forced moves of 17 to 29 blocks
per interval at 4 and 16 ranks with counts conserved and spectra within Poisson noise
of the plain runs (max |chi| 3.5, 0.14 percent of bins above 3); cost-file run starts
at 1.08.  Note that under `automatic` the mesh accumulates `MeshBlock::cost_` from one
redistribution to the next rather than per interval (upstream semantics), so between
redistributions the costs are "since the last move"; harmless for the ratios.  L1
as built: `MonteCarloBlock::PackForTransfer`/`UnpackFromTransfer` with `Photon::PackAll`
and `MCRandom::SaveState`; `Mesh::pmc` and the two hooks in
`RedistributeAndRefineMeshBlocks` calling `PackDeparting` and
`RebuildAfterRedistribution`; `RelinkAll`; `UserWorkAfterRebalance`; the table-mode
guard in `mc_readhdf_gr`.  Test knob `<montecarlo> lb_test_repack = local|mesh`: on
the Kerr-Schild shell (one rank, two intervals) and the thin spherical-polar deck (four
ranks) both modes give photon lists and spectra **byte-identical** to the plain run,
so pack, unpack, relink, RNG carry and both hooks are exact; the mesh mode also showed
the inversion at `Initialize(2)` to be idempotent on already round-tripped primitives.
The cross-rank payload path is first exercised by L2b's forced moves.  L0d on `edd_survey.full.00600`, 16 ranks, 200k
photons, polarized Compton, free-free: transport 5740 s summed over ranks, busiest rank
1.29 times the fair share, costliest block 0.116 times the fair share (29 blocks per
rank), 1.35 us per pusher step, 29 ms per photon at 198 scatterings each.  So at 16
ranks balancing can recover up to 29 percent and the granularity ceiling is far away;
at 64 ranks (7 blocks per rank) the ceiling rises to about 0.46 and the imbalance will
be larger, which is the production case to measure in L6.  Also seen: 2.8 percent of
photons retired by `capmove = 50000` in that deck (`nrem`), which is energy lost from
the outputs and worth a look independent of this plan.  Gate results after L0: KS
shell list byte-identical, poltest, kerr_frames and the 16-rank disk atmosphere pass;
block timers sum to the run's CPU time within 0.3 percent on one rank.

Assessment written 2026-09-13, revised the same day so that one mechanism serves
static and dynamic runs.  The first target is the static
(post-processing) Monte Carlo run on a mesh whose block tree does not change: uniform,
static mesh refinement, or a tree replayed from an athdf snapshot.  Dynamic (coupled)
runs use the same mechanism with a smaller payload (section 3.6); adaptive refinement
is deferred but no longer blocked (section L7).

## 1. What Athena++ already provides

Load balancing is a special case of the AMR machinery in `src/mesh/amr_loadbalance.cpp`,
driven once per cycle from `main.cpp:573`:

- `Mesh::LoadBalancingAndAdaptiveMeshRefinement` (line 40) updates the cost list, and if
  `lb_flag_` is set and `step_since_lb >= lb_interval_`, gathers costs and redistributes
  when the most loaded rank exceeds `(1 + lb_tolerance_)` times the mean.
- Costs live in `Mesh::costlist[gid]` (one double per block, replicated on every rank after
  `GatherCostListAndCheckBalance`'s `MPI_Allgatherv`, line 318).  Two sources exist:
  `balancer = automatic` times each task-list stage per block (`MeshBlock::Start/Stop
  TimeMeasurement`, accumulating into `MeshBlock::cost_`); `balancer = manual` lets a
  problem generator call `MeshBlock::SetCostForLoadBalancing`.  Neither runs in a static
  Monte Carlo run: the task list is never entered (`main.cpp:542-557`), and the manual
  setter is private to `MeshBlock` and has an upstream bug (`meshblock.cpp:483` takes
  `std::min(cost, TINY_NUMBER)`, so every manual cost is `TINY_NUMBER`).
- `CalculateLoadBalance` (line 71) cuts the gid-ordered block list, which is the Z-order
  curve, into contiguous rank ranges greedily from the last rank down.  It is a pure
  function of the cost list, and it is what fixes the ceiling on any balancing: a rank
  can never hold less than one block, so the achievable speedup is bounded by
  `total cost / (nranks * max single-block cost)`.
- `RedistributeAndRefineMeshBlocks` (line 350) computes the new rank list, packs the
  registered cell-centered and face-centered arrays of departing blocks (active cells
  only, no ghosts; `PrepareSendSameLevel`, line 686), builds new `MeshBlock` objects for
  arrivals (the `ref_flag = true` constructor, which calls `InitUserMeshBlockData` but not
  `ProblemGenerator`), *reuses* the `MeshBlock` object for a block that stays on its rank
  (only `gid`/`lid` are rewritten), deletes the departed ones, re-searches every block's
  neighbors, calls `Mesh::Initialize(2, pin)` to refill ghost zones and time steps, and
  keeps the (renumbered) cost list.  Without refinement the gid of every block is
  unchanged; only ranks and lids move.
- `MonteCarlo` is already a `friend class` of `Mesh` (`mesh.hpp:210`), so every private
  piece above -- `costlist`, `ranklist`, `nslist`, `nblist`, `CalculateLoadBalance`,
  `RedistributeAndRefineMeshBlocks` -- is callable from the Monte Carlo module with no
  change to the mesh headers.

So the mesh side of "move block g from rank a to rank b, keep the hydro state, fix the
neighbor tables" exists, is exercised by every AMR run, and is reachable from the Monte
Carlo module as it stands.

## 2. What sits on top of it, and what goes stale when blocks move

The Monte Carlo state is not in the `MeshBlock`.  It is a parallel tree owned by
`MonteCarlo::my_blocks` (lid order), each `MonteCarloBlock` pointing at its `MeshBlock`
through `pmy_block`, and each `MeshBlock` pointing back through `pmy_mcb`.  The mesh
never touches it: `RedistributeAndRefineMeshBlocks` will happily delete a `MeshBlock` out
from under its `MonteCarloBlock`.  The inventory of what a `MonteCarloBlock` holds
(`montecarloblock.cpp:33-446`), sorted by what has to happen to it on a move:

**Must travel with the block (irreplaceable accumulated state).**

| item | size on a 64x64x32 block with ghosts (166k cells) | notes |
|---|---|---|
| resident photons: `intprop[nint]`, `rp[nreal]` vectors | up to `loop_max_size` x ~470 B, 4.7 MB at 10k | plus receive buffers, which are empty at a quiescent point |
| `moments` (and `_com`, `_coord`, `_user`, `_scat`, `_scat_error`) | 13 x ntype x 8 B x 166k = 17 MB per type per array | unnormalized sums, so a plain copy is exact |
| `sourceterms` | 10 x 8 B x 166k = 13 MB | only with `call_srcterms` |
| `emit_count_`, `i1_ i2_ i3_` | 0.7 MB | equal-weight emission cursor; sampled, cannot be recomputed |
| `nphremain`, `nphrun`, `nscat nesc nabs ndes nrem`, `minweight`, `emiss_to_weight` | bytes | counters are summed at the end of each type, so moving them is exact |
| `pran` (mt19937 state) | ~5 KB | optional; see section 3.5 |

**Must travel, but belongs to the MeshBlock: `phydro->w` including ghost zones.**  In a
`gr_user` Monte Carlo build the primitives are the ground truth (the pgen reads them from
the athdf, `mc_readhdf_gr.cpp:489-522`), `Mesh::Initialize` deliberately skips
`ConservedToPrimitive` for `MONTE_CARLO_ENABLED && GENERAL_RELATIVITY` (`mesh.cpp:1655`),
and the mesh transfer packs only the registered `u`, active cells only.  A moved block
would arrive with `w == 0`, and every fluid-derived array below would be rebuilt from
zeros.  Ghosts matter too: photons legitimately sit in ghost cells while waiting to be
handed on, and `FillBounds` fills the derived arrays over them.

Update, 2026-09-13: the guard is gone and the inversion runs.  Tested on
`edd_survey.full.00600` at 16 ranks, the upstream inverter failed in 4.8 million of the
60.8 million cells (all and only those with `p <= 1e-12` in code units, handed back at
the density floor with zero velocity) and mis-converged in 5.5 million more, because
`ConservedToPrimitiveNormal` fixes its pressure floor, tolerance and cubic guard at the
absolute value 1e-12.  In a Monte Carlo build those now scale with the cell's
normal-frame energy density and the convergence test is relative to the pressure (see
the comment there); with that, the round trip reproduces the file's density in every
cell, its pressure to float32 rounding, and the Monte Carlo temperature to 5e-7.  Two
consequences for this plan: the mesh transfer of `u` plus `Initialize(2)` does rebuild
`w` on an arriving block, ghosts included, so `w` need not travel in the Monte Carlo
payload; and the physical-boundary ghost values of `w` after a rebalance are the
outflow copies the boundary functions make, not file values, which is what every block
already has today.  The payload list above shrinks to the accumulated Monte Carlo state
and the photons.

**Can be rebuilt from `w` on the receiving side** by the same calls `MonteCarlo::
Initialize` makes (`montecarlo.cpp:583-607`): `rho`, `tgas`, `species`, `ff_cell`,
`uprim`/`vel`, `boost_cmv`/`boost_lab`, `scalars`, `bcc`, `emission`, `planck_*`, and
the `MCCoord` object (it *copies* `x1f`/`x2f`/`x3f` and computes `vol`, `mccoord.cpp:145`,
so nothing dangles).  Rebuilding rather than packing keeps one code path for "derived
state from primitives" and keeps the `MeshBlock`'s `w` correct for outputs.

**Must be re-linked, not moved.**  `Photon` derives from `Particles`, whose constructor
captures `pmy_block` and `pbval_ = pmb->pbval` (`particles.cpp`, constructor), and
`LinkNeighbors` stores pointers into `pbval_->neighbor[]` plus MPI tags built from the
neighbor's *lid* (`particles.cpp:473-540`).  All of that is stale after any
redistribution, including for a block that did not move (its neighbors' ranks and lids
did).  `MCRankExchange` caches the peer set, per-peer buffers, and checks a layout
snapshot (`mcexchange.cpp:53-108`, `CheckLayoutUnchanged` currently makes any
redistribution fatal on purpose).  `MonteCarlo::nblocal`, `my_blocks` order, and the
per-block `loop_max_size` budget (`montecarlo.cpp:557-603`) all depend on the layout.

**Problem-generator state that the module cannot see.**  `mc_readhdf_gr.cpp:275` in table
mode allocates `opact`, `emis_tot`, `emis_cum` as file-scope arrays indexed by *lid*.  A
rebalance breaks that silently.  Free-free mode keeps only scalars and is safe.  The plan
provides a hook (`UserWorkAfterRebalance`) and leaves table mode unsupported until the
pgen uses it.

**Two latent hazards that exist today, independent of this project.**

1. `Mesh::NewTimeStep` (`mesh.cpp:1181`) reads `my_blocks(0)->pmy_mcb->pmy_mc->dynamic`.
   `Mesh::Initialize(2)` ends by calling it, and after a redistribution the first local
   block may be a fresh `MeshBlock` whose `pmy_mcb` has never been set: the regular
   constructor (`meshblock.cpp:49`, the one redistribution uses for arrivals) does not
   initialize it at all, and only the restart constructor nulls it (`meshblock.cpp:264`).
   Any redistribution therefore dereferences garbage before the Monte Carlo module gets a
   chance to reattach.  Fix: initialize `pmy_mcb` in both constructors, and compute a
   `static Monte Carlo` flag once in the `Mesh` constructor from `HydroIsStatic(pin)`
   (`mcstatic.hpp`) so `NewTimeStep` tests that instead of chasing the pointer.
2. If a deck sets `<loadbalancing> balancer = automatic`, the main-loop call at
   `main.cpp:573` is live in a static Monte Carlo run.  Costs stay at `TINY_NUMBER` so it
   never redistributes today, but nothing guarantees that.  In the design below the
   mesh's balancer is the single driver in both modes, with Monte Carlo hooks inside it;
   the static path skips the per-cycle call and invokes the balancer from the module's
   own check points instead.

## 3. Critical assessment

### 3.1 How much modification

Less than it looks, and almost all of it inside `src/monte_carlo/`.

- **Non-MC edits: five small ones.**  (i) The `NewTimeStep` flag and the `pmy_mcb`
  initializer in the regular `MeshBlock` constructor (hazard 1; the flag sits in an
  existing `MONTE_CARLO_ENABLED` block, the initializer is one member-initializer entry).
  (ii) Skipping the per-cycle balancer call in `main.cpp` for static MC (hazard 2, inside
  an existing block).  (iii) A `MonteCarlo *pmc` member on `Mesh`, set by the
  `MonteCarlo` constructor, so the mesh can call back.  (iv) Two hooks in
  `RedistributeAndRefineMeshBlocks`, one before it deletes the old blocks and one after
  `Initialize(2)`, each a single guarded call.  (v) A way for the module to add its
  measured time to `MeshBlock::cost_`: one friend declaration or a public adder.  No
  change to `hydro.cpp` or the particles code; the primitives of an arriving block are
  rebuilt from the transferred `u` by the inversion in `Initialize(2)`, which now runs
  in Monte Carlo builds (see the update in section 2).  The first version of this plan
  had the module compute the partition and call the redistribution itself, with three
  non-MC edits; that only works while the module is the sole balancer, and a dynamic run
  has the hydro balancer and possibly refinement moving blocks on their own.  The hooks
  cost two more small edits and make both modes one code path.
- **New MC code, roughly 600 to 900 lines:** a `MonteCarloBlock` pack/unpack pair, the
  two hook bodies (`PackDeparting`, `RebuildAfterRedistribution`) implementing the
  sequence in section 3.3, per-block cost accounting, a trigger with a collective
  decision for the static mid-transport case, and a quiescence step for the
  asynchronous transport.  The photon pack is the existing `ParticleBuffer` layout
  (`AcceptPhotons` already unpacks that form, `photon.cpp:816`), the array packs are
  `memcpy` of whole `AthenaArray`s.
- **Modified MC code:** `RunMonteCarlo` and `TransportAsync` gain a check point;
  `MCRankExchange::BuildPeerList` becomes re-entrant (it nearly is) and
  `CheckLayoutUnchanged` goes away; the `MonteCarloBlock` constructor is split so that
  a block can be constructed without touching photons (it already takes `pblsize`).

The reason it stays contained: the hooks run inside the routine that owns `newrank`,
`newtoold` and `oldtonew`, so the module never has to reproduce the mesh's decision.
Without refinement the old-to-new gid map is the identity; with it (dynamic runs, L7)
the same arrays say which children or parent a block's photons go to.

### 3.2 What is the right cost function

The question posed -- how to weight a step against a scattering in GR versus flat,
polarized versus not -- has a clean answer: do not weight by hand, measure.

The proposal is to time each block's transport sweep.  `TransferPhotonsOnBlock` is
called per block per sweep from exactly two places (`montecarlo.cpp:1140-1145` and
`1398-1403`); wrapping those calls with a monotonic clock and accumulating into the
owning `MeshBlock`'s `cost_`, the same field the hydro task list times into, captures
pusher steps, scatterings, emission, moment
deposition, opacity refreshes, and polarization transport with the actual weights of
the actual build, with no model to maintain.  This is also what Athena++'s own
`balancer = automatic` does for hydro.  Resolution is not a concern: a sweep of a block
holding thousands of photons is milliseconds, and `std::chrono::steady_clock` resolves
tens of nanoseconds.

Counters are still worth keeping, for two reasons.  First, diagnostics: per block,
count pusher steps (`nmvp` increments, already computed per photon at
`generalpusher.cpp:209`), scatterings (`nscat`, already there), emitted photons, and
photons handed to neighbors.  Second, the trigger can then be argued about in physical
terms.  If a counter model is ever wanted -- for instance to seed the *first* balance
before any timing exists -- the weights should come from a least-squares fit of the
measured block times against the block counters over the blocks on a rank, which the
module can do itself after the first interval.  As priors, from the profiling in
`memory_and_speed_plan.md`: a general-pusher step in Cartesian Kerr-Schild is on the
order of a microsecond (about 660 ns of it metric work), polarized transport adds
roughly half again; a flight in GR is typically hundreds of steps, so a scattering
(a few microseconds, more for Compton or the polarized Stokes round trip) is a percent
or so of the flight it ends.  In the flat Cartesian pusher a cell crossing is tens of
nanoseconds and a flight is tens of crossings, so a scattering can be a large fraction
of a flight.  Counting scatterings alone would therefore misweight GR blocks in thin
regions, where photons take many steps between rare scatterings; steps have to be
counted too, and once both are counted the timing is simpler and more honest.

What a cost measured during one window predicts is the cost of the same block in the
next window.  Early in a transport the work is emission-weighted; late, it concentrates
in the optically thick blocks where the stragglers scatter.  "Several rebalances per
run" is the right cadence for that drift, and the cost list should be aged
(`costlist = w * old + new`, as `UpdateCostList` does) rather than replaced.

Two things timing does not fix.  The granularity bound in section 1: on the XRB
snapshot, 464 blocks over, say, 64 ranks is seven blocks per rank, and one block near
the horizon or the disk midplane can cost more than a rank's fair share.  Diagnostics
(step L1) should print `max block cost / (total / nranks)` so that the ceiling is
known before anything is built on it; if it is close to one, only smaller blocks or
photon-level splitting would help, and that is a different project.  And the initial
distribution: before the first window there is no timing, every block costs one, and
the first interval runs unbalanced.  `DistributeSamples` knows `nphremain` per block
before transport starts, and costs from a previous output interval or a previous run
of the same snapshot are better still; section 4 makes both available.

### 3.3 Where to interrupt `RunMonteCarlo`

A rebalance needs a point where every rank agrees to stop, no photon is in an MPI
buffer, and every rank has the same picture of the load.  The two transport loops
offer this differently.

**Synchronous rounds** (`montecarlo.cpp:1128-1150`).  After `FinishRound` returns, every
rank has passed the same `MPI_Allreduce`, `ExchangeAndDeliver` has completed all
transfers, and `DrainArrivals` has flushed the receive buffers into blocks.  Nothing is
in flight.  Every `check_interval` rounds, add one `MPI_Allgather` of the per-rank cost
accumulated since the last check; every rank evaluates the same predicate on the same
data and either all call the balancer entry point (L2a) or none do.

**Asynchronous transport** (`TransportAsync`, line 1344) has no rounds.  It does have a
sequence of completed `MPI_Iallreduce`s that every rank sees in the same order, and the
termination logic already proves the wire quiet from two consecutive reductions with
`sent == recv` and unchanged totals.  Reuse that: (i) every `check_interval` completed
reductions, piggyback the per-rank cost (an `MPI_Iallgather`, or extend the reduction
buffer with cost sums and use an Allgather only when the sum suggests imbalance); (ii)
when the predicate fires, every rank sets `draining` and stops calling the transport
sweep but keeps calling `CompleteSends`, `DrainIncoming` and `DrainArrivals`; (iii) the
existing quiet test, with the resident-photon term ignored, declares the wire empty;
(iv) all ranks call the balancer entry point, reset the exchange counters (both sums are equal at
that moment, so zeroing them on every rank is consistent), and resume.  Same argument
as termination: each rank posts reduction `k+1` only after completing `k`, so the
decision sequence is identical everywhere.

Mid-transport rebalancing should require `rank_exchange = true` (the default).  The
legacy per-block protocol keeps persistent `MPI_Irecv`s on neighbor links
(`photon.cpp:688`) that would have to be cancelled; not worth supporting.

**The sequence, as the mesh drives it.**  The mesh's `LoadBalancingAndAdaptiveMesh
Refinement` is the entry point in both modes: the main loop calls it every cycle in a
dynamic run, and the module calls it from its own check points in a static run after
gathering its costs (Allgatherv with `nblist`/`nslist`) and setting `lb_flag_`.  When it
decides to redistribute, `RedistributeAndRefineMeshBlocks` runs with two Monte Carlo
hooks in it:

1. **`pmc->PackDeparting(newrank, newtoold, oldtonew, ...)`, before step 7 deletes the
   old blocks.**  Every block leaving this rank is packed (resident photons,
   accumulators, emission cursor, counters, optionally the RNG) and its send posted on
   the Monte Carlo communicator; receives for arrivals are posted (sizes known: fixed
   arrays plus a photon count first, or one probe).  In a dynamic run the accumulators
   are per cycle and already consumed, so the payload is counters and the RNG.
2. The mesh's own steps: hydro `u` moves, new `MeshBlock`s are built, neighbors
   re-searched, `Initialize(2)` refills ghosts and inverts to `w`.
3. **`pmc->RebuildAfterRedistribution()`, after `Initialize(2)`.**  Rebuild
   `MonteCarlo::my_blocks` in the new lid order: a kept block is reattached (its
   `MeshBlock` object was reused, only `lid` changed); an arrival gets a new
   `MonteCarloBlock` on its new `MeshBlock`, is unpacked, and has its derived arrays
   rebuilt from the inverted `w` by the `Initialize` sequence plus
   `MonteCarloProblemGenerator`; departed blocks are deleted.  `nblocal`,
   `loop_max_size`, `pmy_mcb` back-pointers.  Then for every local block:
   `pphot->ClearNeighbors()`, `LinkNeighbors`, `SetOffRankNeighborFlag`,
   `ClearBoundary`; `pexch->BuildPeerList()`, `Reset()`; the `UserWorkAfterRebalance`
   pgen hook; the run log records what moved and what it cost.

The mesh already keeps the cost list across redistributions and ages it
(`UpdateCostList`), and resets `cost_` afterwards (`ResetLoadBalanceVariables`); the
module's timing rides on that rather than duplicating it.

The payload is modest: with lab moments on, 25 to 30 MB per moved block on the XRB mesh;
moving a tenth of 464 blocks is about a gigabyte across the machine, seconds.
`Initialize(2)` is one ghost-zone exchange of `u` over the whole mesh, also seconds.
Against a run of hours, five or ten rebalances are noise, which is the premise of the
project and it holds.

### 3.4 What makes this hard to get right

- **Photons in the wrong place.**  The failure modes are silent: a photon delivered to
  the right rank but the wrong block (stale lid), a photon dropped because its
  destination is no longer a peer, a receive buffer left with `has_incoming_` set across
  the rebalance.  The photon conservation counters (`ntot = nesc + nabs + ndes + nrem`)
  catch the count but not the position.  Section 4's tests therefore compare *outputs*
  (spectra, moment arrays) between rebalanced and unbalanced runs, not just counts, and
  include a one-rank repack test that must be bitwise identical.
- **The two transport modes and the pgen hooks each have their own state.**  The
  checklist in section 2 is the contract; anything added to `MonteCarloBlock` later
  must be classified against it.  A comment at the top of the pack function should say
  so.
- **Statistical non-invariance already present.**  `DistributeSamples` draws per-rank
  photon counts with a multinomial on rank 0 and then per-block counts on each rank
  (`montecarlo.cpp:934-1035`), so the emitted photon distribution depends on the
  block-to-rank assignment.  Rebalanced and unbalanced runs are therefore statistically,
  not bitwise, equivalent even on the emission side.  Optional step L5c makes the draw
  layout-invariant (one multinomial over all `nbtotal` blocks on rank 0); worth doing
  for testability, not required.

### 3.5 Decisions to make now

- **RNG state travels.**  Serializing `std::mt19937` through a stringstream is ~5 KB and
  portable; the GSL variant has `gsl_rng_memcpy`.  Carrying it is what makes the
  one-rank repack test bitwise, which is the most sensitive test of the pack path.
  Reseeding (`iseed + gid*10 + epoch*large`) is the fallback if GSL makes it awkward.
- **Between-interval balancing comes first, mid-transport second.**  A static run with
  `nout > 1` calls `RunMonteCarlo` once per output; balancing at the top of the call
  with costs from the previous interval exercises the entire transfer path with no
  photons resident and no quiescence protocol.  It is also useful on its own: the first
  interval calibrates, the rest run balanced.
- **Costs persist.**  In memory across intervals (the mesh already keeps `costlist`),
  and optionally to a small text file (`<loadbalancing> cost_file`) so that the next
  run on the same snapshot -- the common workflow -- starts balanced.  Reading it back
  is a gid-indexed list, validated against `nbtotal`.
- **Owner of the policy is the mesh's balancer, in both modes.**  `<loadbalancing>
  balancer = automatic`, `tolerance` and `interval` govern as they do for hydro; the
  Monte Carlo time simply adds to the same per-block cost, so in a dynamic run the two
  cannot fight because there is one cost list and one trigger.  What is Monte Carlo
  specific lives in `<montecarlo>`: the mid-transport check cadence and cap
  (`lb_check_interval`, `lb_max_per_transport`, `lb_min_window`), the cost file, the
  test knobs, and `lb_verbose`.  The first version of this plan put the whole policy in
  `<montecarlo>`; that would have left a dynamic run with two balancers.

### 3.6 Dynamic runs

A dynamic (coupled) run differs from the static one in ways that make balancing
simpler, not harder, once the mesh is the driver:

- **The trigger already exists.**  `main.cpp:573` calls the balancer every cycle, after
  `RunMonteCarlo` and the hydro task list.  With the module's time added to `cost_`,
  `balancer = automatic` balances the sum of hydro and transport work with no new
  policy.  Hydro cost per block is nearly uniform; the transport cost is what varies.
- **The Monte Carlo state is disposable at that point.**  Moments are reset at the top
  of every `RunMonteCarlo` (`montecarlo.cpp:1096-1104`) and source terms are consumed
  by the coupling within the cycle, so nothing accumulated has to move.  Photons do not
  persist across cycles either: every `RunMonteCarlo` transports to completion (the
  round loop and `TransportAsync` both run until no block holds a photon).  The
  between-cycle payload is therefore counters and the RNG, a strict subset of the
  static mid-transport payload, and the derived arrays of an arriving block are
  refreshed from `w` at the start of the next `RunMonteCarlo` by the existing dynamic
  path (`montecarlo.cpp:1069-1093`).
- **No mid-transport interrupt is needed**, since the cycle boundary comes often.  L3
  and L4 are static-only.
- **Refinement becomes tractable** for the same reason: with per-cycle state the only
  thing to split or merge is photons, by position, and `Particles::AMRCoarseToFine`
  and `AMRFineToCoarse` (`particles.cpp:57,97`) already do that for same-rank moves;
  cross-rank refinement moves go through `PackDeparting` keyed by destination child,
  using `newtoold`/`oldtonew`, which the hook has.  Still last in the order (L7).

One thing to settle before leaning on `tmax` in dynamic runs, noticed while checking
photon persistence and not chased: when a photon's time budget `dtp` runs out, the
pushers leave the step loop with the status still `EVOLVING`
(`generalpusher.cpp:88-89`, `cartesianpusher.cpp:64`), and `TransferPhotonsOnBlock`
then treats every `EVOLVING` photon as having reached an interaction.  If photons are
ever meant to survive a step boundary, that is where the payload rule above changes.

## 4. Plan

Each step is a mergeable unit with its own verification.  Statistical comparisons use
the existing gates: `disk_atmosphere.py` (spherical polar, MPI, both pushers, Feautrier
reference), `snake_thomson_spectrum` (`gr_user`, polarized), `kerr_frames` and the KS
shell deck for GR lists, `amr_transport` for a static-refined mesh.  A new
`tst/montecarlo/load_balance/` driver runs the same decks with balancing forced and
compares.

### L0. Safety and a baseline measurement (no balancing yet)

- L0a. `Mesh::NewTimeStep`: replace the `pmy_mcb` pointer chase with a flag set in the
  `Mesh` constructor from `HydroIsStatic(pin)`; initialize `pmy_mcb` in the regular
  `MeshBlock` constructor.  `main.cpp`: skip `LoadBalancingAndAdaptiveMeshRefinement`
  when Monte Carlo is static.  With the `pmc` member, the two hooks (L1b) and the cost
  access (L0b), these are all the non-MC edits in the plan.
  *Test:* full rebuild, poltest and kerr_frames unchanged (byte-identical lists).
- L0b. Per-block cost accounting: wall time of each block's transport sweeps added to
  its `MeshBlock::cost_` (friend or adder), so that it flows through `UpdateCostList`
  and is aged and reset by the mesh like the hydro time; counters (steps, scatterings,
  emitted, handed off) kept on `MonteCarloBlock` for the report.  Two timer calls
  around each of the two `TransferPhotonsOnBlock` call sites.
- L0c. End-of-transport report (`lb_verbose`): per rank total, max/mean over ranks,
  max block cost over the fair share (the granularity ceiling), top few blocks by gid.
  *Test:* block times sum to within a few percent of the transport loop's wall time on
  one rank; counters agree with the totals `RunMonteCarlo` already prints.
- L0d. Run the report on the XRB deck at production rank count.  This is the go/no-go
  measurement: it gives the actual imbalance and the ceiling before anything else is
  built.

### L1. Block pack/unpack and rebuild-in-place

- L1a. `MonteCarloBlock::PackForTransfer(buffer)` / `UnpackFromTransfer(buffer)`
  covering the "must travel" list in section 2, with a header carrying sizes and a
  version word.  A constructor path that builds the block without emitting or sampling.
- L1b. `Mesh::pmc`, and the two hooks in `RedistributeAndRefineMeshBlocks` calling
  `MonteCarlo::PackDeparting` and `MonteCarlo::RebuildAfterRedistribution` (section
  3.3).  `BuildPeerList` re-entrant; `CheckLayoutUnchanged` removed.
- L1c. Debug knob `lb_test_repack`: on one rank, at the top of every `RunMonteCarlo`,
  force the balancer with `lb_flag_` set and treat every block as departing and
  arriving: pack, delete, construct anew, unpack, relink.  No block changes rank; the
  mesh routine runs in full.
  *Test:* KS shell and snake decks with `nout = 2`: lists and spectra **bitwise
  identical** with and without the knob (RNG carried).  This validates the payload and
  the relink with no MPI nondeterminism in the way.
- L1d. `UserWorkAfterRebalance` hook (weak default, like the other pgen hooks) and a
  fatal error if `mc_readhdf_gr` is in table mode with balancing on, until its tables
  are rebuilt in that hook.

### L2. Rebalancing between output intervals

- L2a. Static entry point: at the top of `RunMonteCarlo`, when `balancer` is set and
  `nout > 1`, gather the module's costs into the mesh cost list, set `lb_flag_`, and
  call `LoadBalancingAndAdaptiveMeshRefinement`; the mesh decides, and the hooks do the
  rest.
- L2b. Forced-move mode for testing: `lb_test_costs = permute` assigns synthetic costs
  so that a known set of blocks changes rank on 2 and 4 ranks.
  *Tests:* (i) `disk_atmosphere.py` with `--mcranks 4` and forced moves between
  intervals: spectrum passes the Feautrier gate and matches the unbalanced run within
  Poisson noise; (ii) moment arrays (`mclab` athdf) from a two-interval run agree with
  the unbalanced run to statistical precision block by block, including the moved ones;
  (iii) photon conservation holds in every interval; (iv) a run with all costs equal
  moves nothing and is bitwise identical to L0.
- L2c. Cost persistence: `lb_cost_file` write at end of run, read at start; the first
  interval of a second run on the same deck starts from the balanced layout.
  *Test:* second run's first-interval per-rank max/mean is within tolerance from the
  start; the file round-trips (write, read, compare).
- L2d. Dynamic run through the same hooks: `inputs/mc/athinput.hotjupiter` (dynamic,
  coupled) with `balancer = automatic` on 4 ranks and forced moves.
  *Tests:* the run completes; hydro and Monte Carlo outputs agree with the unbalanced
  run to statistical precision; photon conservation per cycle; a one-rank run with the
  repack knob is bitwise identical to the unbalanced one.

### L3. Mid-transport rebalancing, synchronous loop

- L3a. Check point after `FinishRound` every `lb_check_interval` rounds: Allgather of
  per-rank window cost, common predicate (imbalance above `lb_tolerance`, fewer than
  `lb_max_per_transport` rebalances so far, at least `lb_min_window` rounds since the
  last).  On fire, the L2a entry point with photons resident.
- L3b. Payload now carries photons; receive buffers verified empty; `has_incoming_`
  cleared; `nphremain` and the emission cursor move.
  *Tests:* `async_term = false`; forced rebalance every 20 rounds on `disk_atmosphere`
  (4 ranks) and on the snake atmosphere: gates pass, spectra match the unbalanced runs,
  conservation holds; a stress variant forcing a rebalance every round for the first 50
  rounds to shake out ordering bugs.

### L4. Mid-transport rebalancing, asynchronous loop

- L4a. Draining state in `TransportAsync` as in section 3.3; per-rank cost piggybacked
  on the reduction cadence; exchange counters reset consistently after the rebalance.
- L4b. Bound the drain: if the wire is not quiet after a generous number of reductions,
  log and continue transporting rather than hang.
  *Tests:* the L3 tests with `async_term = true`; a 2-rank run where one rank holds all
  the work (forced costs) to exercise the idle rank's drain path; the stress variant.

### L5. Policy, seeding, and polish

- L5a. Aging of the cost list and the window definition; defaults chosen from the L0d
  measurement.
- L5b. Initial balance from `nphremain` (emission-weighted) when no cost file exists;
  optional counter-fit weights as a second estimator, printed for comparison.
- L5c. Optional: layout-invariant `DistributeSamples`.
- L5d. `MCRankExchange` buffers sized to the new peer count; `loop_max_size`
  recomputed for the new `nblocal`; report of what each rebalance moved and cost.

### L6. Production validation

- XRB deck, production rank count, free-free emission and absorption: wall time with and
  without balancing, the per-rank profile before and after each rebalance, and the
  overhead per rebalance.  Compare spectra and moment outputs with the unbalanced run.
  Record the result here.

### L7. Deferred

- Adaptive refinement in dynamic runs: gids renumber and blocks split and merge, but
  with per-cycle Monte Carlo state only photons have to follow, by position (section
  3.6).  Same-rank splits and merges through the existing `Particles` helpers,
  cross-rank ones through `PackDeparting` keyed by destination.  Static runs with
  refinement stay out of scope: their moments accumulate across the run.
- Photon-level balancing (splitting one block's photon population across ranks) if L0d
  shows the granularity ceiling is the binding constraint.

## 5. Verification summary

| step | what is compared | criterion |
|---|---|---|
| L0a | KS shell list, poltest | byte-identical to current main |
| L0c | block timers vs loop wall time; counters vs printed totals | within a few percent; exact |
| L1c | one-rank repack, two intervals | lists and spectra byte-identical |
| L2b | forced moves, 2 and 4 ranks | Feautrier gate; spectra and moments within Poisson noise of unbalanced; conservation exact; no-move case byte-identical |
| L2c | cost file round trip | exact; balanced from the first interval |
| L2d | dynamic coupled deck, 4 ranks, forced moves | completes; outputs statistically identical; one-rank repack bitwise |
| L3, L4 | forced mid-transport rebalances, both loops, stress cadence | gates pass; conservation; no hang |
| L6 | XRB deck | speedup versus the unbalanced run; outputs statistically identical |
