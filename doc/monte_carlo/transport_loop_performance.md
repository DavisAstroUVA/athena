# The transport loop: what made it slow, and what was done about it

This records the work on `MonteCarlo::RunMonteCarlo`'s transfer loop, why each change was
made, and what it did and did not buy. It exists because most of these changes are
invisible from the code alone: they are responses to measurements, and several of the
obvious-looking alternatives are wrong for reasons that only a measurement shows.

The problem that started it was an AMR snapshot read through `grid_from_file`: 83231
blocks of 8x4x8 cells on 32 ranks, which ran far slower than the same physics on a
resampled uniform mesh. A synthetic stand-in reproduces the shape of it and is what the
numbers below are measured on unless stated otherwise.

## 1. The benchmark

`mc_isoth`, 128^3 root grid, 8^3 meshblocks, so 4096 blocks; periodic in x1 and x2,
absorbing at the inner x3 face and escaping at the outer one; 100000 photons.

The periodic sides are the essential part. With a low optical depth a photon emitted
nearly horizontally has a free path far longer than the domain, and the periodic sides
mean it never leaves: it crosses block after block, wrapping, until it eventually drifts
far enough in x3 to reach a boundary. Two variants are used:

- `tau = 1.0e-3` -- the main case, a mix of ordinary photons and long flights.
- `tau = 1.0e-6`, 20000 photons -- a straggler case, where nearly every photon finishes
  at once and a handful fly for a very long time afterwards.

Photon conservation, `nesc + nabs + ndes + nrem == ntot`, is checked on every run and is
the thing that catches the failure mode that matters. Every bug in this area lost photons
silently; none of them announced themselves any other way.

## 2. What the cost actually was

The original protocol had every block send a count message down every neighbor link that
crossed a rank boundary, every round, whether or not it had a photon to go with it, and
then poll each of them. With ~56 neighbor links per block and a third of them crossing a
rank boundary, a rank was issuing on the order of 50000 blocking sends per round. Going
from 1 rank to 8 made the run *more than ten times slower*.

Three separate costs were in play, and they had to be attacked separately:

1. **Per-block, per-link messaging.** Fixed by sweeping only blocks that can take part
   (`ExchangeLocal`), then by aggregating everything bound for a rank into one message per
   peer (`MCRankExchange`).
2. **A barrier per block crossing.** A photon crossing many blocks bought a global
   `Allreduce` for each one. Fixed by the local sub-loop, which keeps transporting while
   photons are only moving between blocks on this rank.
3. **A barrier per round at all.** Even with the sub-loop, a photon crossing *rank*
   boundaries still costs one collective per crossing, and every other rank pays it.
   This is what §4 is about.

The measured chain on the benchmark: original protocol -> active-block lists (22.7x
serial) -> rank aggregation (29x at n=4) -> local sub-loop and a single fused `Allreduce`
(94.6 s to 24.9 s at n=32).

## 3. Why the round count was the thing to look at

Instrumenting the round count settled what was left. On a 32-rank run **99.6% of rounds
were transporting a single photon** while every rank went through the full round for it.
The round count on the benchmark was 4293015; replacing the periodic sides with escaping
ones dropped it to 42. That is the whole story: the cost was not the photons, it was the
rounds, and the rounds were one long flight.

Two things follow, and both were implemented.

## 4. The path cap (`capmove`), replacing `checkmove`

`checkmove` was meant to bound exactly this, and could not. It counted steps *within one
`Move` call*, and a `Move` ends as soon as the photon leaves its block -- about 8 steps on
an 8^3 block. A flight that crosses blocks without end therefore never tripped it, which is
why raising `checkmove` to 10^5 or 10^6 left the round count unchanged at exactly 4293015.
It predates the domain decomposition, which is what broke it: once a trajectory is cut into
one `Move` per block, a per-`Move` counter cannot see a trajectory at all. It has been
removed outright rather than left as a second, weaker cap that no longer means anything.

Removing it also removed the only bound on the pusher's inner `while` loop. That loop
terminates on its own -- each pass either consumes optical depth and time or steps to the
next cell, and a photon that leaves the block ends the loop -- so the bound was doing no
work in any run that behaves. What it did provide was a backstop against a photon that
fails to advance at all, and `capmove` cannot stand in for that, because it is only
evaluated after the loop returns. If that backstop is wanted it should be an explicit
"made no progress" test, not an iteration count.

`capmove` counts the same steps but accumulates them in a photon property, `Photon::nmvp`,
so the count survives both the end of a `Move` and a transfer to another block. It is
reset at each scattering, in `TransferPhotonsOnBlock`: a scattering starts a new free
flight, and a photon that scatters is by definition not in the flight that will not end.

Two details are load-bearing:

- **The status test has to accept `BUFFERED`, not just `EVOLVING`.** A `Move` that ends
  with the photon still `EVOLVING` is one that reached its interaction point inside the
  block. The flight this cap exists to stop leaves the block on every call, and so ends
  every one of them `BUFFERED`. Testing only `EVOLVING` compiles, runs, retires a few
  photons, and does essentially nothing: with `capmove = 200` the longest flight still
  reached 387862 steps.
- **`nmvp` has to be zeroed for reused slots.** `Particles::Resize` value-initializes only
  when the underlying vector grows, and it does not: `RemoveOneParticle` swaps the last
  photon into the freed slot and decrements the count without shrinking the storage. A new
  photon in a reused slot would inherit the previous occupant's count. It is zeroed in
  `Photon::AllocatePhotons`, which both emission sites go through, rather than in the
  problem generators (which is where `nscp` is zeroed, and is easy for a new pgen to miss).

Measured, serial, `tau = 1e-3`:

| `capmove` | rounds | longest flight | `nrem` | wall |
|---|---|---|---|---|
| 0 (off) | 93341 | 746730 | 0 | 80.8 s |
| 2000 | 442 | 2015 | 4160 (4.2%) | 24.5 s |
| 200 | 52 | 220 | 47935 (48%) | 10.1 s |
| 50 | 16 | 71 | 85129 (85%) | 5.7 s |

The longest flight is bounded by `capmove` plus at most one `Move`, since the test is at
the end of one.

### `capmove` is biased, and the reset does not fix that

It discards photons, and not at random: it discards exactly the longest-path ones. In a
scattering-dominated run (`tau = 3`, ~9.9 scatterings per photon) `capmove = 200` still
discarded 8.7% of photons, because free flights are long *horizontally* even when the
vertical depth is 3 -- the periodic directions have no bounding depth. With the cap off
the longest single free flight there was 638 steps.

So it is a safety valve for genuinely unbounded flights, not a general speed knob. It wants
to be set well above the longest physical free flight, and `nrem` in the run summary is the
check on how much was thrown away. **Default is 0, meaning off**, so it never changes
anyone's answer without being asked for.

## 5. Counter-based asynchronous termination (`async_term`)

`FinishRound` settles every round with a blocking `Allreduce`, so the run advances in
lockstep at the pace of whichever rank still holds a photon. `TransportAsync` replaces
that: a rank transports what it holds, posts what is leaving without waiting for it to be
taken, and picks up whatever has arrived.

The only question left is when everyone is finished, and it is answered from two running
counters on `MCRankExchange`: photons handed to other ranks (`nsent_`, incremented at
`Stage`) and photons taken from them (`nrecv_`, incremented at delivery). Their global sums
differ by exactly the number of photons in flight, so the mesh is done when no rank holds a
photon and the two sums agree.

**One reduction is not enough to trust.** Its inputs are read at different instants on
different ranks, so a photon can be counted as received before the rank that sent it
counted it as sent, and a lone reduction can report a quiet mesh that is not quiet. Two
consecutive ones are enough, because a rank posts the second only after the first has
completed: if both report no photons anywhere and the identical pair of totals, then
nothing was sent anywhere between them and nothing was in flight. This is Mattern's
four-counter argument in its usual practical form.

The exchange had to change with it. `ExchangeAndDeliver`'s size handshake tells every peer
"nothing for you", which makes an idle round cost `2*npeers` messages on every rank and
couples the whole domain to the slowest one. `SendStaged`/`DrainIncoming` replace it: the
int stream leads with its own header, so a receiver that has probed that one message can
work out the length of every other stream and post exact receives, and a rank with nothing
to send sends nothing at all.

### Three ways this deadlocks or loses photons

All three were hit while building it. They are recorded because none is obvious from
reading the finished code, and each looked correct when written.

- **Waiting on your own sends without receiving.** A send large enough to go by rendezvous
  does not complete until the receiver matches it. `CompleteSends` doing a bare `Waitall`
  deadlocks on the first round, when every rank has a full complement of photons to hand
  over and all of them wait at once. It has to drain while it waits.
- **Completing sends before taking delivery.** The wait belongs before the *next* reset,
  not after posting, because a rank that posted nothing must still reach the drain.
- **Reading "what landed" from the wrong place.** `CompleteSends` delivers photons too, so
  taking the count from `DrainIncoming`'s return misses everything the former took. The
  flush is then skipped, those photons sit in receive buffers where the activity test
  cannot see them, and the run terminates holding them. This one lost 75% of the photons
  and was caught only by the conservation check.

### What it is worth

| case | ranks | sync | async |
|---|---|---|---|
| `tau=1e-3`, 100k | 8 | 6.87 s | 7.35 s |
| `tau=1e-3`, 100k | 32 | 2.91 s | 2.51 s |
| straggler, 20k | 8 | 4.30 s | 3.81 s |
| straggler, 20k | 32 | 2.41 s | 1.90 s |

Modest, and mixed at low rank counts: it helps where a straggler dominates and where there
are enough ranks for the barrier to be the cost, and it is slightly negative at n=8 on the
mixed case. That is the honest shape of it on this benchmark, which after the earlier
stages is no longer barrier-dominated.

### On the case that motivated it

The tdestream snapshot, 83231 blocks on 5 levels, 32 ranks, 20000 photons, Compton
scattering. Setup is excluded: the timer below runs from "Setup complete" to the summary
line, so it is transport only.

| run | | setup | transport | nesc | nabs |
|---|---|---|---|---|---|
| 1 | sync  | 11.57 s | 327.4 s | 14580 | 5420 |
| 1 | async | 11.55 s | 157.7 s | 14584 | 5416 |
| 2 | sync  | 11.44 s | 327.0 s | 14580 | 5420 |
| 2 | async | 11.27 s | 189.3 s | 14626 | 5374 |

**1.7x to 2.1x on transport**, far more than the 1.0-1.2x the synthetic benchmarks showed,
and the reason is visible in the run: `nscat/ntot` is about 1.3e4. Photons scatter roughly
thirteen thousand times each, with a long tail, so the run really is a handful of
stragglers holding everyone else at the barrier. That is precisely the case counter-based
termination was built for, and the synthetic benchmarks -- with `nscat/ntot` under 0.01 --
never had it.

Take the range seriously rather than the best figure. The synchronous path is
bit-reproducible, repeating its counts and its time to better than 0.2%, which makes it a
clean baseline -- but only against a fixed Compton table. Generating the table rather than
reading the same file changes 622 of its 17901 entries at the level of one part in 1e14,
which is a compiler and libm difference in the numerical integration, not a defect. With
1.3e4 scatterings per photon that is ample to diverge the random walks: it moved `nesc`
from 14580 to 14617, or 0.6 sigma. Reproducing a run bit for bit therefore needs the same
`comptontable.out`, not merely the same source. The asynchronous path varied by 20% between two runs of the same input,
because both the order photons arrive in and the moment the termination reductions complete
are nondeterministic. Two runs bound it loosely; they do not pin it down.

Both repeats conserve exactly (`14580 + 5420` and `14626 + 5374`), which is the first time
the async path has met a mesh of this size. Escaping fractions sit 0.06 and 0.73 Poisson
sigma from the deterministic baseline.

Note that `cpu time used` as printed by the code is `clock()`, i.e. process CPU time on
rank 0, and is the wrong metric for this comparison: a rank spinning in the async
termination loop burns CPU where one blocked in `MPI_Allreduce` may not. Measure wall time.

**Results are not bit-identical to the synchronous path.** Photons arrive in a different
order, so each block consumes its random stream differently. Escaping fractions agree
within Poisson noise, which is the most that can be asked of it.

## 6. Switches

All in `<montecarlo>`:

| name | default | meaning |
|---|---|---|
| `rank_exchange` | on | aggregate off-rank photons per rank rather than per block link |
| `local_max_sweeps` | 1000 | ceiling on consecutive same-rank transport sweeps per round |
| `async_term` | on | counter-based termination instead of a collective per round |
| `capmove` | 0 (off) | cap on steps in one free flight; exceeded photons become `REMOVED` |

`checkmove` is gone (§4). Input files that still set it are harmless -- Athena++ ignores
parameters nothing reads -- but the ones in this repo have been cleaned out.

`async_term` falls back to the synchronous loop automatically whenever the rank exchange is
not active, which includes every serial run.

## 7. Left undone

- Stage 3 as originally sketched, an `MPI_Ibarrier`-based NBX for the sizes handshake, is
  superseded by §5 for the async path but the synchronous path still has the handshake.
- The synchronous path is a second implementation of the same exchange. The two now share
  the delivery walk (`MCRankExchange::Deliver`), which is the part that fails silently, but
  not the message structure. Note that it is not merely a fallback: `TransportAsync` is
  compiled only under `MPI_PARALLEL` and dispatched only when the rank exchange is active,
  so the synchronous loop is the transport loop for every non-MPI build and every serial
  run. Retiring it means first giving those cases an async-equivalent path.
- ~~`GenerateComptonTable` runs once per block~~ **Fixed.** It was called from the
  `MonteCarloBlock` constructor, so it ran about 2600 times per rank on the 83231-block
  snapshot, while `xsect` is a single file-scope global -- every call rebuilt the identical
  table in the identical memory. It now builds at most once per process. On that snapshot:

  | | before | after |
  |---|---|---|
  | `comptonio = 0` (read file) | 11.5 s | 11.2 s |
  | `comptonio = 1` (generate, the default) | >966 s, never finished | 19.2 s |

  One generation costs about 8 s, so the default was spending roughly six hours per rank in
  setup. Reading a table hid the problem, a 143 kB page-cached file being cheap to reread.
  Escape and absorption counts are unchanged to every digit against the pre-fix baseline.
