# Faraday rotation and conversion: implementation plan

Status: **not started**, deliberately deferred. The transport machinery it sits on is
finished and tested; this note records what would be involved so the work can be picked up
without re-deriving the context.

Reference: Dexter (2016), MNRAS 462, 115, especially §2.2.1-2.2.2. Not in the
repository; the copy used when this was written was at `~/refs/dexter2016.pdf`.

---

## 1. What is being added

The general pusher currently transports the coherency tensor in vacuum: only the
connection acts on it, via eq. 16 of Mościbrodzka & Gammie,

```
dN^ij/dl = -(A^i_k N^kj + A^j_k N^ik),    A^i_k = Gamma^i_kl k^l
```

Faraday rotation and conversion add a plasma term. In the Stokes representation used by
Dexter's eq. (46) these are the `rho_Q`, `rho_U`, `rho_V` entries of the transfer matrix:
`rho_V` rotates Q into U (rotation), `rho_Q` and `rho_U` move U into V (conversion). They
act even with no emission or absorption, so they change the polarization along a ray in a
way the vacuum transport does not.

Two things follow that shape the whole design:

- They are defined in the **fluid frame**, referenced to the projected magnetic field, not
  to the meridian basis the code stores Stokes parameters in. Dexter's eqs. (44)–(45) give
  the angle `chi` between the projected field and the polarization basis, and eq. (48) the
  rotation `R(chi)` that aligns the two. Note this is a third frame: the coefficients are
  fluid-frame, the transport is coordinate-frame, and the outputs are normal-observer
  frame. Being explicit about which is which is most of the work.
- They are **frequency dependent**. The vacuum term is not, which is why the current
  integrator gets away with what it does.

---

## 2. Where it goes in the code

The transport was deliberately factored so this term has an obvious home.

| piece | file | role |
|---|---|---|
| `GeneralPusher::ConnectionContraction` | `generalpusher.cpp` | builds `A^i_k` at the photon's state; depends on position and k only |
| `GeneralPusher::ApplyPolarizationRate` | `generalpusher.cpp` | applies a rate to a given tensor |
| `GeneralPusher::AdvanceStep` | `generalpusher.cpp` | Heun predictor / `RK4Step` / corrector |
| `MonteCarloBlock::bcc` | `montecarloblock.cpp` | cell-centred B, shallow slice of `pfield->bcc` |
| `MeridianPair`, `WriteMeridianStokes` | `polarization.cpp` | the basis Stokes parameters are referenced to |

The natural insertion point is inside the rate: `ApplyPolarizationRate` gains a plasma
contribution alongside the connection one, and `AdvanceStep` needs no structural change.

A rotation of the polarization plane is, in tensor language, a commutator `[Omega, N]`
with `Omega` antisymmetric and built from the projected field. Whether to implement it
that way or to convert to Stokes, rotate, and convert back is **an open decision** — see
§6.

---

## 3. The integrator question, which should be settled first

`AdvanceStep` uses Heun's method: the rate is evaluated at both ends of the geodesic step
and averaged, which is second order. That holds for any rate that is a function of the
photon's state, so a Faraday term that depends on position, k and the local field does
**not** by itself break the order.

The real hazard is **stiffness**. Faraday rotation angle per step goes as `rho_V * dl`,
and in a dense magnetised plasma this can be many radians across one step. An explicit
integrator does not merely lose accuracy there, it produces nonsense, and refining the
step until it is resolved can be ruinously expensive. `grtrans` and `ipole` both handle
this by treating the rotation analytically over a step rather than integrating it.

So the first task is not code, it is a decision:

- **(a) Explicit, with a step limit.** Add the term to the rate and cap the step so the
  rotation per step stays small. Simplest; may be unusably slow in the regimes of
  interest.
- **(b) Operator split with an analytic rotation.** Advance the vacuum transport as now,
  then apply a closed-form rotation by `rho_V * dl` about k. Exact for any rotation angle,
  unconditionally stable, but the split reintroduces a first-order error unless it is
  arranged symmetrically (half rotation, transport, half rotation — and note that a naive
  half/full/half split of an *Euler* update does **not** give second order; see the
  comment on `AdvanceStep` for why).
- **(c) Semi-analytic matrix exponential.** Exponentiate the full 4x4 transfer matrix over
  a step, as `grtrans` does with its DELO-style schemes. Most accurate and most work.

Recommendation: measure first. Add the term explicitly (a), use the snake convergence test
to confirm the order is still 2, then find the field strength at which the step limit
becomes intolerable. That number decides whether (b) or (c) is needed.

---

## 4. What has to be built

1. **B field in the fluid frame.** `bcc` is the cell-centred lab-frame field. The transfer
   coefficients need `b^mu` in the comoving frame, and its component projected transverse
   to k. The tetrad machinery for this exists (`ConstructTetrad`, `CoordinateToTetrad`);
   what does not exist is any B-field frame transform in the Monte Carlo module.
2. **The `rho` coefficients themselves.** None are implemented. Decide the electron
   distribution: thermal synchrotron (Dexter's default, with fits in his Appendix A), a
   cold-plasma Faraday rotation measure, or a user hook alongside the existing
   `UserScattering` / `UserAbsorptionOpacity` pointers.
3. **The rotation angle `chi`** between the projected field and the meridian basis —
   Dexter eqs. (44)–(45).
4. **Configuration.** A `<montecarlo>/faraday` flag, and the build needs `-b`. Guard with
   a clear error when polarization is on, Faraday is requested, and the build has no
   field.

---

## 5. Test strategy

This is the part worth getting right, because the existing tests will not catch a Faraday
bug and one of them will actively mislead.

**Regression, first.** With B = 0 or `faraday = false`,
`tst/montecarlo/snake_polarization` must still report order 2.00 and
`tst/montecarlo/poltest` must still pass 4/4. Any change there means the vacuum path was
disturbed.

**The analytic case.** A uniform field in flat spacetime is the reference: the
polarization angle rotates by a known amount linear in path length, with no geometry mixed
in. Snake with `snake_a = 0`, or Minkowski, plus a uniform `bcc`, gives an exact answer to
compare against — angle versus path length, then versus field strength, then versus
frequency (rotation goes as `lambda^2` for the cold-plasma case). This is the test to
write first.

**Convergence.** Once the analytic case passes, refine the step and confirm the order is
still 2. If it degrades, the splitting is wrong — see §3.

**A warning about the existing tests.** Every polarized test in the tree runs in a
*uniform* medium, which is exactly why the shadowed-`chi` opacity bug survived for so
long. Do not assume a passing suite means a correct Faraday term; it means the vacuum term
is intact. The uniform-field test above is deliberately uniform for its analytic answer,
so it will not catch a gradient error either. A varying-field case with no analytic
solution, checked for convergence rather than against a reference, is worth adding once
the uniform one passes.

---

## 6. Open decisions

- Tensor commutator vs Stokes round trip for applying the rotation. The Stokes route
  reuses `MeridianPair` and `WriteMeridianStokes` and is easier to check against Dexter's
  eq. (46) directly; the tensor route avoids a basis round trip per step and is probably
  cheaper.
- Which electron distribution, and whether the coefficients belong behind a user function
  pointer like the opacities do.
- Whether Faraday conversion (`rho_Q`, `rho_U`) is needed at once or rotation (`rho_V`)
  alone is enough for the first science case. Rotation alone is much less work and is the
  larger effect in most regimes.

---

## 7. Prerequisites worth clearing first

- **Output frame — done.** The wavevector, the angle bins and the Stokes meridian are all
  projected onto the normal (Eulerian) observer at the photon's position, and the outputs
  record which frame that is in a single `frame=` header field (`normal` for the general
  pusher, `lab` for the legacy ones). The shared definition is `NormalObserver` in
  `tetrad.cpp`; use it rather than rebuilding the observer, or the direction and the
  polarization angle in a file will drift out of the same frame. This mattered more than
  it looked: before unification the spectrum used `MCCoord::InverseTetrad` while the
  polarization used the normal observer, two different orthonormal frames that agreed only
  because both happened to share a third leg.
- **Opacity within a cell.** Currently refreshed only at cell crossings; measured cost of
  refreshing every step is about 25%. Faraday coefficients are frequency dependent and
  would inherit the same staleness.
