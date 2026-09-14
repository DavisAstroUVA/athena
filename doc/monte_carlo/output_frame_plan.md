# Retiring the ray-trace branch and giving the user the output frame: implementation plan

Two changes that belong together. Ray tracing is transport with the opacities set to zero,
so it does not need its own loop; and what it actually does differently -- leave the photon
in the observer's frame before the outputs read it -- is something transport runs want too,
for sensible angular binning. Doing the second makes the first a deletion rather than a
port.

Deferred deliberately: the communication work is being merged to main first, and this
should start from a new branch on top of it.

## 1. Ray tracing is already a special case of transport

Verified by reading both loops, not assumed. With `absorption = none` and
`scattering = none`, `chi = scp + acp` is zero, and in every pusher the "photon reaches its
interaction point inside this cell" test is `(chi > 0.) && (dl*l_cgs > tauremaining/chi)`.
With `chi = 0` that is false on every step, so the photon always takes the next-cell branch,
`tauremaining` never decreases, and the photon can only leave the block or the domain. It is
therefore never `EVOLVING` when the post-`Move` loop runs, and both the absorption and the
scattering blocks in `TransferPhotonsOnBlock` are gated on exactly that. They are skipped
for the same reason `RayTracePhotonsOnBlock` omits them.

The remaining differences, in three groups.

**Already better in `TransferPhotonsOnBlock`, so nothing to port:**

- Emission batching. `loop_max_size` caps the *total resident* photons there, but only the
  *newly emitted* ones in the ray-trace loop, which bounds nothing.
- `if (ntot == 0) return;`, absent from the ray-trace loop, which otherwise calls
  `Move(pphot, 0, -1)`.
- A stray `printf("remain: %d")` on every call, and a `PrintPhoton` on every `DESTROYED`
  photon.

**Controlled by input, so they need setting rather than coding:**

- `initialize_comoving`, which gates the comoving-to-coordinate conversion of newly emitted
  photons. It **defaults to true**, and the ray-trace loop hardcodes the opposite; the
  converted inputs must set it false.
- `call_srcterms`, off in an uncoupled run.

**The one real gap, now closed.** `CoherencyToObserverStokes` was called only in the
ray-trace loop, though the comment describing it sits in both. Restoring it to
`TransferPhotonsOnBlock` is a one-line change, already made and merged separately from this
plan. Measured effect on `snake_thomson_spectrum`: mean `|dS|` of 3e-9 and max 1.4e-7, which
is invisible in a polarization norm of 0.1975 -- and it falls to 2e-16 when the snake shear
is set to zero and rises with it, which is the behavior a basis-rotation correction should
have. Real, correct, and small in that regime; untested where the final leg is long and the
spacetime genuinely curved.

## 2. Giving the user the output frame

The motivation is not tidiness. A spectrum bins angles on the photon direction, and that is
only meaningful in an orthonormal frame; `Spectrum::UpdateSpectrum` therefore re-derives the
direction in the normal frame itself. Making the frame a property of the photon at output
time means the binning, the photon list, the Stokes parameters and any user
`FinalizePhoton` hook all read the same numbers instead of three of them reconstructing it.

Most of the machinery exists.

- `MCFRAME_LAB` **is** the normal (Eulerian) observer frame -- see the comment on the
  `MCFrame` enum -- and `PhotonFrames` already projects a photon into a requested frame and
  caches, "at most once each".
- The photon storage already has a notion of a current frame. `GetFourVector`'s contract is
  that `k0p` holds "the photon energy in the frame the photon is currently expressed in",
  with `unit_spatial` distinguishing a dimensional `k^i` from a unit direction, and
  `TransformToComoving`/`TransformToCoordinate` already move photons in place around every
  scattering.

So this is the existing pattern applied at the terminal step, not a new mechanism.

### The constraint that decides the implementation

`PhotonEnergyAtInfinity` computes the conserved `-k_t` by contracting `k0p, k1p, k2p, k3p`
as **coordinate** components with `g_{t mu}`. It is called at *output* time, in
`Spectrum::UpdateSpectrum` and `PhotonList::AddPhoton`, both of which run after the terminal
sweep where the transform would happen.

Transform the photon in place and that contraction silently returns a wrong number: tetrad
components against the coordinate metric. No NaN, no error, just a mis-binned energy in
every spectrum and photon list whenever `relativistic_output` is on. **The conserved energy
has to be computed and stored before the transform**, and that ordering is the single most
important thing to get right here.

## 3. What to build

1. `<montecarlo>/output_frame`, taking `coordinate` (default) or `normal`. The default
   preserves every current result, including `mc_poltest`, which computes a flat-frame
   coherency residual in its `FinalizePhoton` and must keep coordinate components.
2. Somewhere to keep `-k_t` across the transform. A photon real property is the simplest and
   follows the `nmvp` precedent; block scratch would also do, since the value is consumed in
   the same loop iteration.
3. The terminal sweep in `TransferPhotonsOnBlock`, in this order: stash `-k_t`, transform to
   the normal frame if asked, `CoherencyToObserverStokes`, `FinalizePhoton`.
4. `Spectrum::UpdateSpectrum` and `PhotonList::AddPhoton` read the stored energy and the
   already-projected wavevector when the photon is in the normal frame, instead of calling
   `NormalFrameWavevector` themselves. That collapses three tetrad constructions per escaping
   photon to one.
5. Reject `output_frame = normal` unless `general_pusher_flag`. The legacy pushers already
   reference everything to a globally constant orthonormal frame -- it is why
   `CoherencyToObserverStokes` returns immediately for them -- so the option is either a
   no-op or meaningless there, and silently doing nothing is the worse of the two.
6. Guard the `ABSWEIGHT` branch against `scp + acp == 0`. It computes
   `wp *= scp/(scp+acp)`, which is 0/0 with both opacities zero. It is unreachable today for
   the reason in §1, but a zero-opacity run is exactly what ray tracing becomes, and the
   protection is currently a chain of three inferences rather than a test.
7. Delete `RayTracePhotonsOnBlock`, the `raytrace_flag` dispatch in both transport loops, and
   the `raytrace` input option -- preserving its side effect of forcing
   `general_pusher_flag`, which the converted inputs must now set explicitly.

## 4. Test strategy

The ray-trace inputs are `inputs/mc/athinput.img_disk` (`mc_geoimg`, kerr-schild) and
`inputs/mc/athinput.geokerr` (`mc_geokerr`, kerr-schild).

**Capture baselines before touching anything.** Both currently run through
`RayTracePhotonsOnBlock`; their output is the reference the conversion has to reproduce, and
it cannot be recovered once the branch is gone.

Then, in order:

- Convert each input to `scattering = none`, `absorption = none`,
  `initialize_comoving = false`, `general_pusher = true`, and confirm it reproduces its
  baseline. This is the real test of §1, and it should reproduce closely, not merely
  statistically: with no scattering there is no random walk to diverge, so the trajectories
  are deterministic given the same emission.
- With `output_frame = normal`, confirm the spectra and photon lists match the same
  baselines. This is what proves step 4 reads back what step 3 wrote.
- `mc_poltest` unchanged, as the guard on the default: it must keep seeing coordinate
  components in `FinalizePhoton`.
- `snake_thomson_spectrum` and `snake_polarization` unchanged, as the guard on the polarized
  transport path.

## 5. Open decisions

- **Whether `output_frame = normal` should also project the position.** It should not --
  a position is a point, not a frame-dependent vector -- but the flag's name invites the
  question and the answer belongs in the documentation.
- **What `output_frame` means for the comoving frame.** `MCFRAME_COMOVING` exists and is a
  plausible third value for someone who wants comoving spectra. Worth leaving room for in
  the parameter parsing even if only two values are accepted at first.
- **Whether to unify `PhotonFrames` and `NormalFrameWavevector`.** Both construct the
  normal-observer tetrad and neither knows about the other: the first serves the moments
  path with energy, direction and path length; the second serves the output path with the
  tetrad and the projected wavevector. Merging them is the right end state and would remove
  the last duplicate definition of the output frame, but it touches the moments path and
  should not ride along with this.

## 6. Notes for whoever picks this up

- The `#ifdef DEBUG` block in `GetFourVector` asserts that `ep` and `k0p` agree, pending
  `iep` being aliased to `ik0p`. Anything that changes what frame `k0p` is expressed in has
  to keep that invariant, or move `ep` with it.
- `raytrace = true` currently forces `general_pusher_flag = true`. Losing that coupling
  silently would put the converted inputs on a legacy pusher, where the geodesic integration
  they depend on does not exist.
