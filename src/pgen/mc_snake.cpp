//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mc_snake.cpp
//! \brief Uniform periodic box in sinusoidal ("snake") coordinates, built with -g.
//
// Snake coordinates are flat spacetime written in the sheared frame of White, Stone &
// Gammie (2016), ApJS 225, 22:
//
//     t = t_M,  x = x_M,  y = y_M + a sin(k x_M),  z = z_M
//
// so with beta = a k cos(k x),
//
//     ds^2 = -dt^2 + (1 + beta^2) dx^2 - 2 beta dx dy + dy^2 + dz^2.
//
// This pgen uses the same physical problem as mc_isoth_mink -- a uniform box,
// free-free emission, optionally drifting along x3. Because the spacetime is genuinely flat,
// every frame-invariant scalar must agree between the two runs to round-off.
//
// Units follow mc_isoth_mink exactly: hydro primitives are dimensionless code values,
// converted for the Monte Carlo by rho_cgs and tgas_cgs, because the GR equation of
// state needs p/rho << 1.  See that file for the arithmetic.
//
// Inputs (in addition to the usual mc_isoth ones):
//   <coord>/snake_a   shear amplitude          (default 0 -> degenerates to Minkowski)
//   <coord>/snake_k   shear wavenumber         (default 0)
//   <coord>/m, a      required by GRUser and ignored here; set both to 0
//   <problem>/polang  emitted linear polarization angle, radians (default 0)
//   <problem>/polcirc emitted Stokes V, |V| <= 1                 (default 0)

// C++ headers
#include <algorithm>  // max
#include <cmath>
#include <complex>
#include <cstring>  // strcmp
#include <sstream>
#include <stdexcept>
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

#if !MONTE_CARLO_ENABLED
#error "This problem requires monte carlo"
#endif
#if !GENERAL_RELATIVITY
#error "mc_snake requires general relativity (-g)"
#endif

#include "../monte_carlo/montecarlo.hpp"
#include "../monte_carlo/tetrad.hpp"
#include "../monte_carlo/photon.hpp"
#include "../monte_carlo/photonpusher.hpp"

namespace {
bool tnorm;
Real logemin, logemax;
// Cached so the metric callback, which runs per cell, does not re-parse the input.
// Set in Mesh::InitUserMeshData before the metric is ever evaluated.
Real snake_a = 0.0;
Real snake_k = 0.0;
void SnakeMetric(Real x1, Real x2, Real x3, ParameterInput *pin,
                 AthenaArray<Real> &g, AthenaArray<Real> &g_inv,
                 AthenaArray<Real> &dg_dx1, AthenaArray<Real> &dg_dx2,
                 AthenaArray<Real> &dg_dx3);
void JMeanOpacity(MonteCarloBlock *pmcb, Photon *pphot, int ip, int imom,
                  const PhotonFrameState &s);
void AverageEnergy(MonteCarloBlock *pmcb, Photon *pphot, int ip, int imom,
                  const PhotonFrameState &s);
void TrackGeodesicInvariant(MonteCarloBlock *pmcb, Photon *pphot, int ip);

// Emitted polarization, shared by every photon.  Set in MonteCarloProblemGenerator,
// which runs before the first InitializePhoton.
Real polang = 0.0;
Real polcirc = 0.0;

// Photon user-variable layout.  The first two slots are the geodesic invariant and are
// what frames.py reads, so they keep their meaning whether or not polarization is on;
// the polarization slots are only allocated when it is.  IPOLREF starts a flat copy of
// the 4x4 complex reference tensor, real part then imaginary part.
constexpr int IPOLDRIFT = 2;
constexpr int IPOLREF = 3;
constexpr int NPOLCOMP = 32;
constexpr int NUSER_POL = IPOLREF + NPOLCOMP;

void PolarizationInFlatFrame(MonteCarloBlock *pmcb, Photon *pphot, int ip,
                             std::complex<Real> nflat[4][4]);
void TrackPolarizationInvariant(MonteCarloBlock *pmcb, Photon *pphot, int ip);
void TrackInvariants(MonteCarloBlock *pmcb, Photon *pphot,
                     PhotonPusher *ppusher, int ip);
} // namespace

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief enroll the snake metric and check the box is compatible with it
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {

  if (std::strcmp(COORDINATE_SYSTEM, "gr_user") != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in mc_snake InitUserMeshData" << std::endl
        << "mc_snake supplies its own metric and requires --coord=gr_user, got "
        << COORDINATE_SYSTEM << std::endl;
    ATHENA_ERROR(msg);
  }

  snake_a = pin->GetOrAddReal("coord", "snake_a", 0.0);
  snake_k = pin->GetOrAddReal("coord", "snake_k", 0.0);

  // The metric depends on x1 alone, through beta = a k cos(k x).  A periodic x1 boundary
  // therefore only makes sense when the shear closes over the box: otherwise the metric
  // is discontinuous across the boundary and the photons see a seam.  x2 and x3 are
  // always safe, since no metric component depends on them.
  std::string ix1 = pin->GetOrAddString("mesh", "ix1_mc_bc", "");
  if (ix1 == "periodic" && snake_a != 0.0 && snake_k != 0.0) {
    Real lx1 = pin->GetReal("mesh", "x1max") - pin->GetReal("mesh", "x1min");
    Real turns = snake_k * lx1 / (2.0 * PI);
    if (std::fabs(turns - std::round(turns)) > 1.0e-10) {
      std::stringstream msg;
      msg << "### FATAL ERROR in mc_snake InitUserMeshData" << std::endl
          << "Periodic x1 requires the shear to close over the box: snake_k * Lx1 must "
          << "be a multiple of 2 pi." << std::endl
          << "Got snake_k = " << snake_k << ", Lx1 = " << lx1
          << ", snake_k*Lx1/(2 pi) = " << turns << std::endl;
      ATHENA_ERROR(msg);
    }
  }

  EnrollUserMetric(SnakeMetric);

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief uniform box, optionally moving at a constant velocity along x3
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  // dimensionless code values; the Monte Carlo scales them via rho_cgs/tgas_cgs
  Real rho0 = pin->GetOrAddReal("problem", "dens_code", 1.0);
  Real pgas0 = pin->GetOrAddReal("problem", "pgas_code", 1.0e-6);

  // constant velocity as a fraction of the speed of light
  Real beta = pin->GetOrAddReal("problem", "velocity", 0.);
  if (std::fabs(beta) >= 1.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in mc_snake ProblemGenerator" << std::endl
        << "velocity must be given as a fraction of c, got " << beta << std::endl;
    ATHENA_ERROR(msg);
  }
  Real lorentz = 1.0 / std::sqrt(1.0 - SQR(beta));

  // Drift is along x3 deliberately.  g_33 = 1 and no metric component mixes x3 with
  // anything, so utilde^3 = gamma*beta reproduces u^mu = (gamma,0,0,gamma*beta) exactly
  // as it does in Minkowski -- which is what lets a snake run and a mink run be
  // compared directly.  A drift along x1 or x2 would have to account for g_xy.
  Real uu3 = lorentz * beta;

  AthenaArray<Real> bb;
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        phydro->w(IDN, k, j, i) = phydro->w1(IDN, k, j, i) = rho0;
        phydro->w(IPR, k, j, i) = phydro->w1(IPR, k, j, i) = pgas0;
        phydro->w(IVX, k, j, i) = phydro->w1(IVX, k, j, i) = 0.0;
        phydro->w(IVY, k, j, i) = phydro->w1(IVY, k, j, i) = 0.0;
        phydro->w(IVZ, k, j, i) = phydro->w1(IVZ, k, j, i) = uu3;
      }
    }
  }
  peos->PrimitiveToConserved(phydro->w, bb, phydro->u, pcoord, is, ie, js, je, ks, ke);

  return;
}

//========================================================================================
//! \fn void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype)
//! \brief free-free emission, identical to mc_isoth_mink
//========================================================================================

void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype) {

  SetEmissionCellWeight(pphot, ips, ipe);

  for (int ip = ips; ip <= ipe; ip++) {

    pphot->dtp[ip] = pphot->pmy_mcb->pmy_mc->tmax;

    // position within the selected zone
    GetZonePosition(pphot, pran, pcoord, ip);
    pphot->x0p[ip] = 0.0;

    // energy, direction and weight from the free-free emissivity
    if (tnorm) {
      Real logtg = log(tgas(pphot->i3p[ip], pphot->i2p[ip], pphot->i1p[ip]));
      PhotonEmitFreeFree(this, pphot, logemin + logtg, logemax + logtg, ip);
    } else {
      PhotonEmitFreeFree(this, pphot, logemin, logemax, ip);
    }

    if (pphot->wp[ip] < 0.0)
      pphot->statp[ip] = DESTROYED;
    else
      pphot->statp[ip] = EVOLVING;

    pphot->nscp[ip] = 0;

    pphot->acp[ip] = AbsorptionOpacity(this, pphot, ip);
    pphot->scp[ip] = ScatteringOpacity(this, pphot, ip);

    // Emitted polarization.  PhotonEmitFreeFree leaves the photon unpolarized, which
    // makes a transport test vacuous for Q and U, so overwrite the Stokes parameters with
    // a definite state.  The direction is still whatever free-free sampled, so the run
    // exercises the connection over the whole sphere rather than one ray.  The block
    // turns these into the coherency tensor once the photon is in coordinate form.
    if (IsPolarized(pmy_mc->polarized)) {
      Real plin = std::sqrt(std::max(0.0, 1.0 - SQR(polcirc)));
      pphot->sip[ip] = 1.0;
      pphot->sqp[ip] = plin*std::cos(2.0*polang);
      pphot->sup[ip] = plin*std::sin(2.0*polang);
      pphot->svp[ip] = polcirc;
    }

    // Arm the invariants.  k1p is still a unit direction here -- the conversion to
    // dimensional contravariant components happens later, and the coherency tensor is not
    // built until after that -- so neither reference can be taken yet.  A negative drift
    // slot marks "not yet seeded"; the trackers take their references on the first call,
    // once the photon is in coordinate form.
    if (pphot->nuser_var > 0) pphot->user[0][ip] = 0.0;
    if (pphot->nuser_var > 1) pphot->user[1][ip] = -1.0;
    if (pphot->nuser_var > IPOLDRIFT) pphot->user[IPOLDRIFT][ip] = -1.0;
  }

  return;
}

//========================================================================================
//! \fn void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin)
//========================================================================================

void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin) {

  // Set the energy boundaries for free-free emission
  tnorm = pin->GetOrAddBoolean("problem", "tnorm", false);
  if (tnorm) {
    // interpret as xmin/xmax with x = E/(kb*T)
    const Real kb = 1.380649e-16;
    logemin = log(kb * pin->GetReal("problem", "emin"));
    logemax = log(kb * pin->GetReal("problem", "emax"));
  } else {
    // interpret as emin/emax in eV
    const Real everg = 1.6021772e-12;
    logemin = log(everg * pin->GetReal("problem", "emin"));
    logemax = log(everg * pin->GetReal("problem", "emax"));
  }

  polang = pin->GetOrAddReal("problem", "polang", 0.0);
  polcirc = pin->GetOrAddReal("problem", "polcirc", 0.0);
  if (std::fabs(polcirc) > 1.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in mc_snake MonteCarloProblemGenerator" << std::endl
        << "polcirc is a Stokes V for a fully polarized photon and must satisfy "
        << "|polcirc| <= 1, got " << polcirc << std::endl;
    ATHENA_ERROR(msg);
  }

  return;
}

//========================================================================================
//! \fn void MonteCarlo::InitUserMonteCarloData(ParameterInput *pin)
//========================================================================================

void MonteCarlo::InitUserMonteCarloData(ParameterInput *pin) {

  if (!pin->GetOrAddBoolean("montecarlo", "general_pusher", false)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in mc_snake InitUserMonteCarloData" << std::endl
        << "mc_snake requires general_pusher = true" << std::endl;
    ATHENA_ERROR(msg);
  }

  // gr_user cannot say which metric the Monte Carlo should use, so mc_coord must name it
  // and must name this one.  Without the check a snake hydro metric would be paired with
  // a Kerr-Schild photon metric
  if (pin->GetOrAddString("montecarlo", "mc_coord", "") != "snake") {
    std::stringstream msg;
    msg << "### FATAL ERROR in mc_snake InitUserMonteCarloData" << std::endl
        << "mc_snake requires <montecarlo>/mc_coord = snake" << std::endl;
    ATHENA_ERROR(msg);
  }

  // user[0] holds the emitted value of the geodesic invariant, user[1] the largest
  // relative departure from it seen along the path.  See TrackGeodesicInvariant.  With
  // polarization on, user[2] carries the same quantity for the coherency tensor and the
  // slots above it hold its reference value; they are never written to the photon list,
  // which caps its columns at the output block's own nuser.
  nuser_var = IsPolarized(polarized) ? NUSER_POL : 2;
  AllocateUserMoments(2);
  EnrollUserMoment(0, JMeanOpacity, "kapJ");
  EnrollUserMoment(1, AverageEnergy, "eave");
  EnrollUserWorkInMove(TrackInvariants);

  return;
}

namespace {

//----------------------------------------------------------------------------------------
//! \fn void SnakeMetric(...)
//! \brief the snake metric, its inverse and its coordinate derivatives
//
// Mirrors MCSnake in mccoord.cpp. The spatial block has unit determinant,
// (1+beta^2) - beta^2 = 1, so the inverse is the diagonal swapped with the off-diagonal
// sign flipped, and sqrt(-g) = 1. beta depends on x1 alone, so dg_dx2 and dg_dx3 vanish
// identically.

void SnakeMetric(Real x1, Real x2, Real x3, ParameterInput *pin,
                 AthenaArray<Real> &g, AthenaArray<Real> &g_inv,
                 AthenaArray<Real> &dg_dx1, AthenaArray<Real> &dg_dx2,
                 AthenaArray<Real> &dg_dx3) {

  Real beta = snake_a * snake_k * std::cos(snake_k * x1);
  Real dbeta = -snake_a * SQR(snake_k) * std::sin(snake_k * x1);

  for (int n = 0; n < NMETRIC; ++n) {
    g(n) = 0.0;
    g_inv(n) = 0.0;
    dg_dx1(n) = 0.0;
    dg_dx2(n) = 0.0;
    dg_dx3(n) = 0.0;
  }

  // covariant
  g(I00) = -1.0;
  g(I11) = 1.0 + SQR(beta);
  g(I12) = -beta;
  g(I22) = 1.0;
  g(I33) = 1.0;

  // contravariant
  g_inv(I00) = -1.0;
  g_inv(I11) = 1.0;
  g_inv(I12) = beta;
  g_inv(I22) = 1.0 + SQR(beta);
  g_inv(I33) = 1.0;

  // only x1 derivatives survive
  dg_dx1(I11) = 2.0 * beta * dbeta;
  dg_dx1(I12) = -dbeta;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void TrackInvariants(...)
//! \brief the enrolled per-step hook: geodesic first, then polarization

void TrackInvariants(MonteCarloBlock *pmcb, Photon *pphot,
                     PhotonPusher *ppusher, int ip) {

  TrackGeodesicInvariant(pmcb, pphot, ip);
  TrackPolarizationInvariant(pmcb, pphot, ip);
}

//----------------------------------------------------------------------------------------
//! \fn void TrackGeodesicInvariant(...)
//! \brief record the largest relative drift in k^x along each photon's path
//
// Every snake geodesic conserves the contravariant k^x exactly.  Writing the covariant
// components, k_x = k_x^M - beta k_y^M and k_y = k_y^M, so
//
//     k^x = g^xx k_x + g^xy k_y = k_x + beta k_y = k_x^M = const,
//
// which is just the statement that x = x_M and the underlying Minkowski geodesic is a
// straight line.
//
// RK4 rescales the spatial components each step
// to re-impose k.k = 0, so the drift measures the truncation error and must fall as the
// step size is reduced.  frames.py checks the convergence.

void TrackGeodesicInvariant(MonteCarloBlock *pmcb, Photon *pphot, int ip) {

  if (pphot->nuser_var < 2) return;

  // First call: the photon is now in coordinate form, so take the reference here. 
  if (pphot->user[1][ip] < 0.0) {
    pphot->user[0][ip] = pphot->k1p[ip];
    pphot->user[1][ip] = 0.0;
    return;
  }

  // Normalised by k^t, not by k^x itself: k^t is conserved and sets the natural scale,
  // whereas k^x passes through zero for any photon travelling perpendicular to x and
  // would make a relative drift meaningless there.
  Real scale = std::fabs(pphot->k0p[ip]);
  if (scale == 0.0) return;

  Real drift = std::fabs(pphot->k1p[ip] - pphot->user[0][ip]) / scale;
  if (drift > pphot->user[1][ip]) pphot->user[1][ip] = drift;
}

//----------------------------------------------------------------------------------------
//! \fn void PolarizationInFlatFrame(...)
//! \brief project the coherency tensor onto the global flat frame
//
// Snake is flat spacetime in curved coordinates, so it has a global frame that is
// covariantly constant, and MCCoord::InverseTetrad is the projector onto it: its legs are
// e_(t) = d_t, e_(x) = d_x + beta d_y, e_(y) = d_y, e_(z) = d_z, orthonormal by
// construction.  Parallel transport leaves the tensor unchanged in that frame, so every
// component must be constant along the ray whatever the connection does to the coordinate
// components.  Needing no Stokes decomposition, and so no basis tied to k, is what makes
// this the reference: an error in the transport cannot hide in a co-rotating basis.

void PolarizationInFlatFrame(MonteCarloBlock *pmcb, Photon *pphot, int ip,
                             std::complex<Real> nflat[4][4]) {

  Real x[4];
  x[IMC0] = pphot->x0p[ip];
  x[IMC1] = pphot->x1p[ip];
  x[IMC2] = pphot->x2p[ip];
  x[IMC3] = pphot->x3p[ip];

  Real invtet[4][4];
  pmcb->pcoord->InverseTetrad(x, invtet);
  pphot->PolarizationToTetrad(nflat, invtet, ip);
}

//----------------------------------------------------------------------------------------
//! \fn void TrackPolarizationInvariant(...)
//! \brief record the largest departure of the flat-frame coherency tensor from its
//!        emitted value
//
// The companion to TrackGeodesicInvariant, and the quantity the snake polarization
// convergence test refines on.  The geodesic is integrated by RK4 and converges at fourth
// order, while GeneralPusher::AdvanceStep transports the tensor with Heun's method and
// converges at second, which is why the two are tracked separately: at any step size
// coarse enough to matter the polarization error is the larger of the two, and the curves
// cannot be confused.
//
// The error reported is the largest absolute departure over the sixteen components,
// divided by the largest reference component.  A single normalisation for all of them,
// rather than a per-component relative error, keeps components that are legitimately
// near zero from dominating the result.

void TrackPolarizationInvariant(MonteCarloBlock *pmcb, Photon *pphot, int ip) {

  if (pphot->nuser_var < NUSER_POL) return;

  std::complex<Real> nflat[4][4];
  PolarizationInFlatFrame(pmcb, pphot, ip, nflat);

  // First call: the coherency tensor has been built and the photon is in coordinate
  // form, so the reference can be taken now.
  if (pphot->user[IPOLDRIFT][ip] < 0.0) {
    Real scale = 0.0;
    for (int a = 0; a < 4; ++a)
      for (int b = 0; b < 4; ++b) scale = std::max(scale, std::abs(nflat[a][b]));

    // Some photons never get a coherency tensor: MeridianBasis declines to build one for a
    // photon travelling along the frame z axis, where the meridian is undefined, and also
    // for one whose wavevector falls below TINY_NUMBER in whatever units the run uses.
    // Leaving the sentinel in place for those keeps them out of the statistics, and keeps
    // them distinguishable from a photon whose drift is legitimately zero -- which is
    // exactly what an unsheared snake produces, since the connection then contributes
    // identically zero and the arithmetic is bit-exact.
    if (scale <= TINY_NUMBER) return;

    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        pphot->user[IPOLREF + 2*(4*a + b)][ip] = nflat[a][b].real();
        pphot->user[IPOLREF + 2*(4*a + b) + 1][ip] = nflat[a][b].imag();
      }
    }
    pphot->user[IPOLDRIFT][ip] = 0.0;
    return;
  }

  Real dmax = 0.0, scale = 0.0;
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      std::complex<Real> ref(pphot->user[IPOLREF + 2*(4*a + b)][ip],
                             pphot->user[IPOLREF + 2*(4*a + b) + 1][ip]);
      dmax = std::max(dmax, std::abs(nflat[a][b] - ref));
      scale = std::max(scale, std::abs(ref));
    }
  }

  Real drift = dmax/scale;
  if (drift > pphot->user[IPOLDRIFT][ip]) pphot->user[IPOLDRIFT][ip] = drift;
}

void JMeanOpacity(MonteCarloBlock *pmcb, Photon *pphot, int ip, int imom,
                  const PhotonFrameState &s) {

  int i1 = pphot->i1p[ip];
  int i2 = pphot->i2p[ip];
  int i3 = pphot->i3p[ip];

  // energy and path length come from the frame this moment was enrolled with
  Real weight = pphot->wp[ip]*s.e*s.dl/MCConstants::c_cgs;
  pmcb->moments_user(imom,i3,i2,i1) += weight*pphot->acp[ip];
}

void AverageEnergy(MonteCarloBlock *pmcb, Photon *pphot, int ip, int imom,
                  const PhotonFrameState &s) {

  int i1 = pphot->i1p[ip];
  int i2 = pphot->i2p[ip];
  int i3 = pphot->i3p[ip];

  // energy and path length come from the frame this moment was enrolled with
  Real weight = pphot->wp[ip]*s.e*s.dl/MCConstants::c_cgs;
  pmcb->moments_user(imom,i3,i2,i1) += weight*s.e;
}

} // namespace
