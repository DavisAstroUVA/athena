//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mc_sphpol.cpp
//! \brief single-photon polarized transport test in spherical-polar coordinates
//
// Flat spacetime written in a curvilinear chart, which is the spherical-polar counterpart
// of mc_snake/mc_poltest.  Because the spacetime is flat, a photon travels a straight line
// and its polarization is exactly constant when referred to the global cartesian frame,
// whatever the connection does to the stored spherical components.  That constancy is the
// analytic reference: FinalizePhoton reports the departure from it as POLRESID, which is
// pure truncation error and must fall at second order under step refinement.
//
// The ray is set by the angle `alpha` between the wavevector and the radial direction and
// by `chi`, which selects the plane it bends in: chi = 0 sweeps theta, chi = 90 degrees
// sweeps phi, so the two families of connection components can be exercised separately.
//
// Unlike snake, this chart has no exact-zero control.  Setting snake_a = 0 degenerates to
// Minkowski in cartesian coordinates, where the connection vanishes identically and the
// residual is exactly zero; no choice of ray does that here.  A radial ray (alpha = 0) is
// the closest thing, and it is worth running, but it is not zero: the orthonormal legs do
// not rotate along it, yet Gamma^theta_(r theta) = Gamma^phi_(r phi) = 1/r still act on the
// stored coordinate components through the scale factors, so it converges at second order
// like any other ray.  What alpha = 0 does isolate is that scale-factor part of the
// connection from the theta/phi-sweeping part that larger alpha brings in.
//
// POLRESID is measured from the coherency tensor, which only the general pusher maintains.
// The legacy spherical-polar pusher carries Q and U referenced to the global cartesian
// meridian and never touches the tensor (see polarization.hpp), so POLRESID would read a
// stale tensor there and be meaninglessly zero rather than a passing control.  Comparing
// the two pushers needs a Stokes-based residual instead, which this file does not yet have.
//
// Setting scatopac non-zero switches to the second test in this file, which is about the
// comoving frame rather than the connection; see the comment on FinalizePhoton.

// C++ headers
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <sstream>
#include <stdexcept>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../monte_carlo/montecarlo.hpp"
#include "../monte_carlo/photon.hpp"
#include "../monte_carlo/tetrad.hpp"

#if !MONTE_CARLO_ENABLED
#error "This problem generator requires the Monte Carlo module (-mc)"
#endif

namespace {
  // emission point and ray geometry
  Real r0, th0, ph0;
  Real alpha, chi;
  // polarization state at emission
  Real polang = 0.0;
  Real polcirc = 0.0;
  // photon energy, in the same dimensionless units the rest of the test uses
  Real en0 = 1.0;

  // Reference value of the coherency tensor in the global cartesian frame, captured at
  // emission.  Transport must leave every component unchanged.
  std::complex<Real> nflat0[4][4];
  bool nflat0_set = false;

  // Scattering mode.  Non-zero scatopac turns the run into the comoving-frame test
  // described above FinalizePhoton: the photon is emitted unpolarized, scatters exactly
  // once, and the degree of polarization it escapes with is the measurement.
  Real scatopac = 0.0;
  Real vel_beta = 0.0;

  void PolarizationInFlatFrame(MonteCarloBlock *pmcb, Photon *pphot, int ip,
                               std::complex<Real> nflat[4][4]);
  Real OneScatterOpacity(MonteCarloBlock *pmcb, Photon *pphot, int ip);
  void ThomsonRightAngle(MonteCarloBlock *pmcb, Photon *pphot, int ips, int ipe);
}

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief uniform, static medium; the transport is what is being tested, not the gas
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  Real rho0 = pin->GetOrAddReal("problem", "dens_code", 1.0);
  Real pgas0 = pin->GetOrAddReal("problem", "pgas_code", 1.0e-6);
  // Uniform radial drift, as a fraction of c.  Radial is chosen because e_r is
  // unambiguous in this chart, so the comoving frame is a pure boost of the local
  // orthonormal legs with no extra rotation to disentangle.
  Real beta = pin->GetOrAddReal("problem", "velocity", 0.0);
  if (std::fabs(beta) >= 1.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in mc_sphpol ProblemGenerator" << std::endl
        << "velocity must be a fraction of c, got " << beta << std::endl;
    ATHENA_ERROR(msg);
  }

  AthenaArray<Real> bb;
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        phydro->w(IDN, k, j, i) = phydro->w1(IDN, k, j, i) = rho0;
        phydro->w(IPR, k, j, i) = phydro->w1(IPR, k, j, i) = pgas0;
        phydro->w(IVX, k, j, i) = phydro->w1(IVX, k, j, i) = beta;
        phydro->w(IVY, k, j, i) = phydro->w1(IVY, k, j, i) = 0.0;
        phydro->w(IVZ, k, j, i) = phydro->w1(IVZ, k, j, i) = 0.0;
      }
    }
  }
  peos->PrimitiveToConserved(phydro->w, bb, phydro->u, pcoord, is, ie, js, je, ks, ke);
  return;
}

//========================================================================================
//! \fn void MonteCarlo::InitUserMonteCarloData(ParameterInput *pin)
//========================================================================================

void MonteCarlo::InitUserMonteCarloData(ParameterInput *pin) {

  if (!IsPolarized(polarized)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in mc_sphpol InitUserMonteCarloData" << std::endl
        << "mc_sphpol is a polarized transport test; set polarized = linear or circular"
        << std::endl;
    ATHENA_ERROR(msg);
  }

  // Emission is handed over in the comoving frame -- k as a unit direction, polarization
  // as Stokes parameters -- and TransferPhotonsOnBlock builds the coherency tensor and
  // converts k, which is the path every emitted photon in a production run takes.  Turning
  // either off skips that conversion and leaves the photon in the wrong frame.  It also
  // costs the test its coverage of the comoving round trip: the reference tensor is
  // captured from the coordinate-frame state the handover is *meant* to reproduce, so any
  // error in that conversion shows up in POLRESID rather than being defined away.
  if (!pin->GetOrAddBoolean("montecarlo", "initialize_comoving", true)
      || !pin->GetOrAddBoolean("montecarlo", "boosts", true)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in mc_sphpol InitUserMonteCarloData" << std::endl
        << "mc_sphpol requires boosts = true and initialize_comoving = true" << std::endl;
    ATHENA_ERROR(msg);
  }

  scatopac = pin->GetOrAddReal("problem", "scatopac", 0.0);
  if (scatopac > 0.0) {
    EnrollUserScatteringFunction(ThomsonRightAngle);
    EnrollUserOpacityFunction(OneScatterOpacity, false);
  }

  nuser_var = 0;
  return;
}

//========================================================================================
//! \fn void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin)
//========================================================================================

void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin) {

  r0  = pin->GetOrAddReal("problem", "r0", 10.0);
  th0 = pin->GetOrAddReal("problem", "th0", 90.0) * PI / 180.0;
  ph0 = pin->GetOrAddReal("problem", "ph0", 0.0) * PI / 180.0;
  // alpha = 0 is a radial ray, the exact-zero control
  alpha = pin->GetOrAddReal("problem", "alpha", 30.0) * PI / 180.0;
  chi = pin->GetOrAddReal("problem", "chi", 0.0) * PI / 180.0;
  polang = pin->GetOrAddReal("problem", "polang", 0.0) * PI / 180.0;
  polcirc = pin->GetOrAddReal("problem", "polcirc", 0.0);
  en0 = pin->GetOrAddReal("problem", "energy", 1.0);
  scatopac = pin->GetOrAddReal("problem", "scatopac", 0.0);
  vel_beta = pin->GetOrAddReal("problem", "velocity", 0.0);

  if (std::fabs(polcirc) > 1.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in mc_sphpol MonteCarloProblemGenerator" << std::endl
        << "polcirc must lie in [-1,1], got " << polcirc << std::endl;
    ATHENA_ERROR(msg);
  }

  // With emission = none, DistributeSamples returns without assigning anything and the
  // problem generator has to say which block launches photons.  Exactly one block owns the
  // emission point, and only that one asks for any.
  nphremain = 0;
  nphrun = 0;
  minweight = 0.0;
  if (r0  >= pcoord->x1f(is) && r0  < pcoord->x1f(ie+1) &&
      th0 >= pcoord->x2f(js) && th0 < pcoord->x2f(je+1) &&
      ph0 >= pcoord->x3f(ks) && ph0 < pcoord->x3f(ke+1)) {
    nphremain = pin->GetInteger("montecarlo", "nphot");
  }
  return;
}

//========================================================================================
//! \fn void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype)
//! \brief launch one deterministic photon with a fully polarized state
//========================================================================================

void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype) {

  // The indices are needed before the opacity calls below, but GetPositionIndices screens
  // the photon for NaN, and a freshly claimed slot holds whatever the last photon in it
  // left behind.  Put every field it looks at into a finite state first.
  for (int ip = ips; ip <= ipe; ip++) {
    pphot->x0p[ip] = 0.0;
    pphot->x1p[ip] = r0;
    pphot->x2p[ip] = th0;
    pphot->x3p[ip] = ph0;
    pphot->k0p[ip] = pphot->ep[ip] = en0;
    pphot->k1p[ip] = pphot->k2p[ip] = pphot->k3p[ip] = 0.0;
    pphot->wp[ip] = 1.0;
    pphot->scp[ip] = pphot->acp[ip] = 0.0;
    pphot->sip[ip] = 1.0;
    pphot->sqp[ip] = pphot->sup[ip] = pphot->svp[ip] = 0.0;
  }
  pphot->GetPositionIndices(ips, ipe);

  Real cth = std::cos(th0), sth = std::sin(th0);

  for (int ip = ips; ip <= ipe; ip++) {

    pphot->ep[ip] = en0;
    pphot->wp[ip] = 1.0;
    pphot->dtp[ip] = pmy_mc->tmax;
    pphot->nscp[ip] = 0;
    pphot->statp[ip] = EVOLVING;
    pphot->dk0p[ip] = pphot->dk1p[ip] = pphot->dk2p[ip] = pphot->dk3p[ip] = 0.0;

    // Direction on the local orthonormal spherical legs.  alpha is measured from radial
    // and chi selects the plane the ray leaves the radial direction in.  For a static
    // fluid these legs are the comoving tetrad, so this is what gets handed over; the
    // coordinate form below is only used to seed the polarization and the reference.
    Real nr  = std::cos(alpha);
    Real nth = std::sin(alpha)*std::cos(chi);
    Real nph = std::sin(alpha)*std::sin(chi);

    // the physical wavevector in coordinate components, obtained by dividing out the
    // scale factors that InverseTetrad multiplies back in
    Real kcoord[4];
    kcoord[IMC0] = en0;
    kcoord[IMC1] = en0*nr;
    kcoord[IMC2] = en0*nth/r0;
    kcoord[IMC3] = en0*nph/(r0*sth);

    // Fully polarized, split between linear and circular by polcirc so that
    // Q^2 + U^2 + V^2 = 1 whatever polcirc is set to.
    // In scattering mode the photon is emitted unpolarized, so that a single Thomson
    // scatter through a right angle has a parameter-free answer.
    Real stokes[4];
    stokes[0] = 1.0;
    if (scatopac > 0.0) {
      stokes[1] = stokes[2] = stokes[3] = 0.0;
    } else {
      Real plin = std::sqrt(std::max(0.0, 1.0 - SQR(polcirc)));
      stokes[1] = plin*std::cos(2.0*polang);
      stokes[2] = plin*std::sin(2.0*polang);
      stokes[3] = polcirc;
    }

    // Seed the tensor against a tetrad whose timelike leg is the static observer and whose
    // third spatial leg is along k, which is the transversality StokesToTensor assumes.
    Real x[4];
    x[IMC0] = pphot->x0p[ip];
    x[IMC1] = pphot->x1p[ip];
    x[IMC2] = pphot->x2p[ip];
    x[IMC3] = pphot->x3p[ip];
    Real gcov[4][4];
    pcoord->Metric(x, gcov);

    Real ucon[4], vcon[4], econ[4][4], ecov[4][4];
    ucon[IMC0] = 1.0; ucon[IMC1] = 0.0; ucon[IMC2] = 0.0; ucon[IMC3] = 0.0;
    // any spatial direction not along k will do; the polar leg is the natural one here
    vcon[IMC0] = 0.0; vcon[IMC1] = 0.0; vcon[IMC2] = -1.0/r0; vcon[IMC3] = 0.0;
    ConstructTetrad(ucon, kcoord, vcon, gcov, econ, ecov);

    std::complex<Real> tcopy[4][4];
    StokesToTensor(stokes, tcopy);
    pphot->PolarizationToCoord(tcopy, econ, ip);

    // Reference value, taken from the coordinate-frame state the comoving handover below
    // is meant to reproduce.  Capturing it here rather than after the conversion is what
    // puts the emission-time comoving round trip inside POLRESID.
    if (!nflat0_set) {
      PolarizationInFlatFrame(this, pphot, ip, nflat0);
      nflat0_set = true;
    }

    // Hand the photon over in the frame the emission path expects: k as a unit direction
    // on the comoving legs, polarization as Stokes parameters referenced to the meridian
    // basis.  TransferPhotonsOnBlock inverts both.
    pphot->k0p[ip] = en0;
    pphot->k1p[ip] = nr;
    pphot->k2p[ip] = nth;
    pphot->k3p[ip] = nph;

    pphot->sip[ip] = stokes[0];
    pphot->sqp[ip] = stokes[1];
    pphot->sup[ip] = stokes[2];
    pphot->svp[ip] = stokes[3];

    pphot->acp[ip] = AbsorptionOpacity(this, pphot, ip);
    pphot->scp[ip] = ScatteringOpacity(this, pphot, ip);
  }
  return;
}

//========================================================================================
//! \fn void MonteCarloBlock::FinalizePhoton(Photon *pphot, int ip)
//! \brief report the transport residual
//========================================================================================

void MonteCarloBlock::FinalizePhoton(Photon *pphot, int ip) {

  if (scatopac > 0.0) {
    // Scattering mode.  An unpolarized photon Thomson-scattered through a right angle is
    // fully linearly polarized perpendicular to the scattering plane, so the degree of
    // polarization leaving the scatter is exactly one.  P is invariant under both Lorentz
    // boosts and parallel transport, so it must still be exactly one when the photon is
    // measured in the observer frame at escape, whatever the fluid velocity.  Anything
    // else is an error in the comoving round trip the scatter sits inside:
    // coordinate -> comoving, meridian basis, tensor -> Stokes -> tensor,
    // comoving -> coordinate.  CoherencyToObserverStokes has already refreshed these from
    // the transported tensor by the time this runs.
    if (pphot->nscp[ip] > 0 && std::fabs(pphot->sip[ip]) > 0.0) {
      Real pdeg = std::sqrt(SQR(pphot->sqp[ip]) + SQR(pphot->sup[ip]) + SQR(pphot->svp[ip]))
                  / std::fabs(pphot->sip[ip]);
      printf("POLDEG %.10e nscat %d\n", pdeg, pphot->nscp[ip]);
    }
    return;
  }

  if (!nflat0_set) return;

  std::complex<Real> nflat[4][4];
  PolarizationInFlatFrame(this, pphot, ip, nflat);
  Real dmax = 0.0, scale = 0.0;
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      dmax = std::max(dmax, std::abs(nflat[a][b] - nflat0[a][b]));
      scale = std::max(scale, std::abs(nflat0[a][b]));
    }
  }
  if (scale > 0.0) printf("POLRESID %.10e\n", dmax/scale);
  return;
}

namespace {

//----------------------------------------------------------------------------------------
//! \fn void PolarizationInFlatFrame(...)
//! \brief the coherency tensor in the global cartesian frame
//
// Two steps, and the second is the one that distinguishes this from the snake test.
// MCSphericalPolar::InverseTetrad carries the coordinate components onto the *local*
// orthonormal legs (e_r, e_theta, e_phi), scaling by (1, 1, r, r sin theta).  Those legs
// rotate along the ray, so they are not a parallel-transported frame and the tensor
// referred to them is not constant.  The global cartesian legs are, so the orthonormal
// components are rotated once more by R(theta,phi) -- the same rotation ToScatteringBasis
// applies to the wavevector, here applied to a rank-two tensor as R N R^T.

void PolarizationInFlatFrame(MonteCarloBlock *pmcb, Photon *pphot, int ip,
                             std::complex<Real> nflat[4][4]) {

  Real x[4];
  x[IMC0] = pphot->x0p[ip];
  x[IMC1] = pphot->x1p[ip];
  x[IMC2] = pphot->x2p[ip];
  x[IMC3] = pphot->x3p[ip];

  Real invtet[4][4];
  pmcb->pcoord->InverseTetrad(x, invtet);
  std::complex<Real> nort[4][4];
  pphot->PolarizationToTetrad(nort, invtet, ip);

  Real cth = std::cos(x[IMC2]), sth = std::sin(x[IMC2]);
  Real cph = std::cos(x[IMC3]), sph = std::sin(x[IMC3]);

  Real rot[4][4];
  for (int a = 0; a < 4; ++a)
    for (int b = 0; b < 4; ++b) rot[a][b] = 0.0;
  rot[IMC0][IMC0] = 1.0;
  rot[IMC1][IMC1] = sth*cph; rot[IMC1][IMC2] = cth*cph; rot[IMC1][IMC3] = -sph;
  rot[IMC2][IMC1] = sth*sph; rot[IMC2][IMC2] = cth*sph; rot[IMC2][IMC3] =  cph;
  rot[IMC3][IMC1] = cth;     rot[IMC3][IMC2] = -sth;    rot[IMC3][IMC3] =  0.0;

  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      std::complex<Real> sum(0.0, 0.0);
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) sum += rot[a][i]*rot[b][j]*nort[i][j];
      nflat[a][b] = sum;
    }
  }
}


//----------------------------------------------------------------------------------------
//! \fn Real OneScatterOpacity(MonteCarloBlock *pmcb, Photon *pphot, int ip)
//! \brief a scattering coefficient that switches off after the first scatter
//
// The right-angle result below is exact for a single scatter of unpolarized light; a second
// scatter would leave a degree of polarization that depends on the geometry and is no
// longer a parameter-free prediction.  Returning zero once the photon has scattered lets it
// free-stream out to the boundary and be measured.

Real OneScatterOpacity(MonteCarloBlock *pmcb, Photon *pphot, int ip) {
  return (pphot->nscp[ip] == 0) ? scatopac : 0.0;
}

//----------------------------------------------------------------------------------------
//! \fn void ThomsonRightAngle(MonteCarloBlock *pmcb, Photon *pphot, int ips, int ipe)
//! \brief deterministic Thomson scatter through exactly ninety degrees
//
// Called with the photon in the comoving frame, k stored as a unit direction on the
// comoving legs and the Stokes parameters referenced to the meridian plane containing the
// frame's third spatial leg and k (see polarization.hpp).
//
// The outgoing direction is taken to be the meridian unit vector of the incoming ray,
// l = normalize(z - (z.n) n).  That is perpendicular to n, so the deflection is exactly a
// right angle, and it lies in the plane spanned by z and n -- which means the scattering
// plane, the incoming meridian plane and the outgoing meridian plane are all the same
// plane.  The emergent polarization of an unpolarized beam is perpendicular to the
// scattering plane, so in that basis it is exactly Q = -1, U = V = 0, with the sign fixed
// by StokesToTensor putting I+Q on the meridian leg.
//
// Deterministic on purpose: the point is to check the frame machinery around the scatter,
// so the scatter itself must contribute no sampling noise.

void ThomsonRightAngle(MonteCarloBlock *pmcb, Photon *pphot, int ips, int ipe) {

  for (int ip = ips; ip <= ipe; ++ip) {
    Real n[3];
    n[0] = pphot->k1p[ip];
    n[1] = pphot->k2p[ip];
    n[2] = pphot->k3p[ip];
    Real nmag = std::sqrt(SQR(n[0]) + SQR(n[1]) + SQR(n[2]));
    if (nmag <= TINY_NUMBER) continue;
    for (int i = 0; i < 3; ++i) n[i] /= nmag;

    Real l[3] = {-n[2]*n[0], -n[2]*n[1], 1.0 - n[2]*n[2]};
    Real lmag = std::sqrt(SQR(l[0]) + SQR(l[1]) + SQR(l[2]));
    if (lmag <= TINY_NUMBER) continue;   // incoming ray along the frame z axis
    for (int i = 0; i < 3; ++i) l[i] /= lmag;

    pphot->k1p[ip] = l[0];
    pphot->k2p[ip] = l[1];
    pphot->k3p[ip] = l[2];

    pphot->sip[ip] = 1.0;
    pphot->sqp[ip] = -1.0;
    pphot->sup[ip] = 0.0;
    pphot->svp[ip] = 0.0;

    // belt and braces alongside OneScatterOpacity, so a stale scp cannot buy a second
    // scatter before the opacities are next refreshed
    pphot->scp[ip] = 0.0;
  }
}

} // namespace
