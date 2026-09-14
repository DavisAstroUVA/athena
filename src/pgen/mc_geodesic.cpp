//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mc_geodesic.cpp
//! \brief Geodesic integrator accuracy test in Kerr-Schild coordinates.
//!
//! Photons propagate with no absorption and no scattering, so they follow pure null
//! geodesics.  The Kerr metric is stationary and axisymmetric, so the covariant
//! components
//!
//!     k_t   = g_{t mu}   k^mu        (photon energy)
//!     k_phi = g_{phi mu} k^mu        (photon angular momentum)
//!
//! are exactly conserved along each geodesic.  Any variation is integration error.
//!
//! Rather than compare against the emitted values -- UserWorkInMove first fires after one
//! step has already been taken -- each photon carries the running minimum and maximum of
//! both invariants.  The spread (max-min) is the drift.  Both are normalized by |k_t|,
//! because k_phi passes through zero for photons launched radially, which would make a
//! relative measure of it meaningless.
//!
//! Note that RK4Step renormalizes the spatial components of k every step to keep the
//! photon exactly on the null cone, so g_{mu nu} k^mu k^nu is enforced rather than
//! conserved and is useless as a diagnostic.  In Kerr-Schild g_{t i} is non-zero, so that
//! renormalization perturbs k_t: integration error the on-shell projection would
//! otherwise absorb silently instead shows up in the drift measured here.
//!
//! The convergence knob is <montecarlo> stepsize, not photon number -- this is a
//! deterministic test and a few thousand photons suffice.  Drift should fall as the
//! fourth power of the step size.
//!
//! Run with boosts = false: there is no fluid, and the comoving tetrad is built on the
//! normal observer (see MonteCarloBlock::SetNormalObserver).

// C++ headers
#include <cmath>
#include <sstream>
#include <stdexcept>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

#if !MONTE_CARLO_ENABLED
#error "This problem generator requires the Monte Carlo module (-mc)"
#endif

#include "../monte_carlo/mccoord.hpp"
#include "../monte_carlo/montecarlo.hpp"
#include "../monte_carlo/photon.hpp"
#include "../monte_carlo/photonpusher.hpp"

namespace {
// user photon variable slots: running extrema of the two invariants
constexpr int IKTMIN = 0;
constexpr int IKTMAX = 1;
constexpr int IKPMIN = 2;
constexpr int IKPMAX = 3;

constexpr Real BIG = 1.0e300;

Real spin;   // black hole spin parameter a
Real r_hor;  // outer horizon radius
int nmu;     // number of polar angles sampled in the launch tetrad
int nphi;    // number of azimuthal angles sampled in the launch tetrad

void TrackInvariants(MonteCarloBlock *pmcb, Photon *pphot, PhotonPusher *ppusher, int ip);
void CovariantK(MCCoord *pco, Photon *pphot, int ip, Real &kt, Real &kphi);
Real UniformEmission(MonteCarloBlock *pmcb, int k, int j, int i, int etype);
} // namespace

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Uniform, essentially massless background.  The hydro state is irrelevant here
//! -- photons neither absorb nor scatter -- but Athena requires a valid one.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  if (COORDINATE_SYSTEM != "kerr-schild") {
    std::stringstream msg;
    msg << "### FATAL ERROR in mc_geodesic ProblemGenerator" << std::endl
        << "mc_geodesic requires Kerr-Schild coordinates, got "
        << COORDINATE_SYSTEM << std::endl;
    ATHENA_ERROR(msg);
  }

  Real rho0 = pin->GetOrAddReal("problem", "dens", 1.0e-10);
  Real pgas0 = pin->GetOrAddReal("problem", "pgas", 1.0e-10);

  AthenaArray<Real> bb;
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        phydro->w(IDN, k, j, i) = phydro->w1(IDN, k, j, i) = rho0;
        phydro->w(IPR, k, j, i) = phydro->w1(IPR, k, j, i) = pgas0;
        // at rest in the normal frame
        phydro->w(IVX, k, j, i) = phydro->w1(IVX, k, j, i) = 0.0;
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
//! \brief Read problem parameters and enroll the invariant tracker.
//========================================================================================

void MonteCarlo::InitUserMonteCarloData(ParameterInput *pin) {

  if (!pin->GetOrAddBoolean("montecarlo", "general_pusher", false)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in mc_geodesic InitUserMonteCarloData" << std::endl
        << "mc_geodesic requires general_pusher = true" << std::endl;
    ATHENA_ERROR(msg);
  }
  if (pin->GetOrAddBoolean("montecarlo", "boosts", false)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in mc_geodesic InitUserMonteCarloData" << std::endl
        << "mc_geodesic is a vacuum geodesic test and must run with boosts = false"
        << std::endl;
    ATHENA_ERROR(msg);
  }

  spin = pin->GetReal("coord", "a");
  r_hor = 1.0 + std::sqrt(1.0 - SQR(spin));

  nmu = pin->GetOrAddInteger("problem", "nmu", 8);
  nphi = pin->GetOrAddInteger("problem", "nphi", 8);

  // four user variables carry the running extrema of k_t and k_phi
  nuser_var = 4;

  // Emission must be "user" rather than "none": the emissivity itself is irrelevant to a
  // vacuum geodesic test, but "none" leaves the per-cell emission weight array
  // unallocated, so no photons are ever seeded.  A constant emissivity spreads photons
  // uniformly by cell volume, which is what we want for geodesic coverage.
  EnrollUserEmissionFunction(UniformEmission);
  EnrollUserWorkInMove(TrackInvariants);

  return;
}

//========================================================================================
//! \fn void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype)
//! \brief Seed photons across the block over a grid of tetrad directions.
//!
//! Emission cells come from the standard machinery so that each block seeds photons in
//! its own zones; sampling many launch radii at once gives better geodesic coverage than
//! a single shell would.  The photon is left in the local orthonormal frame here -- the
//! framework calls TransformToCoordinate immediately afterwards, since tetrads are
//! always on for GR coordinate systems.
//========================================================================================

void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype) {

  SetEmissionCellWeight(pphot, ips, ipe);

  for (int ip = ips; ip <= ipe; ip++) {

    // position within the selected zone
    GetZonePosition(pphot, pran, pcoord, ip);
    pphot->x0p[ip] = 0.0;

    // Direction grid in the launch tetrad.  Cycling over a deterministic grid rather
    // than sampling randomly keeps the test reproducible and guarantees that
    // near-radial, strongly bent and near-circular trajectories are all represented.
    int idir = ip % (nmu * nphi);
    int imu = idir / nphi;
    int iph = idir % nphi;

    // half-offsets keep the sample off the mu = +-1 axis
    Real cth = -1.0 + 2.0 * (static_cast<Real>(imu) + 0.5) / static_cast<Real>(nmu);
    Real sth = std::sqrt(1.0 - SQR(cth));
    Real phi = 2.0 * PI * (static_cast<Real>(iph) + 0.5) / static_cast<Real>(nphi);

    pphot->wp[ip] = 1.0;
    // k0p is the photon energy, so this sets both.  The value is arbitrary: the test
    // measures fractional drift in k_t and k_phi, which is scale invariant.
    pphot->ep[ip] = 1.0;

    // unit direction in the orthonormal tetrad
    pphot->k1p[ip] = sth * std::cos(phi);
    pphot->k2p[ip] = cth;
    pphot->k3p[ip] = sth * std::sin(phi);

    pphot->dtp[ip] = pphot->pmy_mcb->pmy_mc->tmax;
    pphot->nscp[ip] = 0;
    pphot->statp[ip] = EVOLVING;

    // vacuum
    pphot->acp[ip] = 0.0;
    pphot->scp[ip] = 0.0;

    // running extrema, seeded so the first sample always replaces them
    pphot->user[IKTMIN][ip] = BIG;
    pphot->user[IKTMAX][ip] = -BIG;
    pphot->user[IKPMIN][ip] = BIG;
    pphot->user[IKPMAX][ip] = -BIG;
  }

  return;
}

namespace {

//----------------------------------------------------------------------------------------
//! \fn Real UniformEmission(MonteCarloBlock *pmcb, int k, int j, int i, int etype)
//! \brief constant emissivity, so photons are seeded uniformly by cell volume

Real UniformEmission(MonteCarloBlock *pmcb, int k, int j, int i, int etype) {
  return 1.0;
}

//----------------------------------------------------------------------------------------
//! \fn void CovariantK(MCCoord *pco, Photon *pphot, int ip, Real &kt, Real &kphi)
//! \brief lower the photon wavevector and return the two conserved components

void CovariantK(MCCoord *pco, Photon *pphot, int ip, Real &kt, Real &kphi) {

  Real x[4], gcov[4][4], kcon[4];
  x[IMC0] = pphot->x0p[ip];
  x[IMC1] = pphot->x1p[ip];
  x[IMC2] = pphot->x2p[ip];
  x[IMC3] = pphot->x3p[ip];
  kcon[IMC0] = pphot->k0p[ip];
  kcon[IMC1] = pphot->k1p[ip];
  kcon[IMC2] = pphot->k2p[ip];
  kcon[IMC3] = pphot->k3p[ip];

  pco->Metric(x, gcov);

  kt = 0.0;
  kphi = 0.0;
  for (int i = 0; i < 4; i++) {
    kt += gcov[IMC0][i] * kcon[i];
    kphi += gcov[IMC3][i] * kcon[i];
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void TrackInvariants(MonteCarloBlock *pmcb, Photon *pphot,
//!                          PhotonPusher *ppusher, int ip)
//! \brief update the running extrema of k_t and k_phi.  Called once per integration step.

void TrackInvariants(MonteCarloBlock *pmcb, Photon *pphot, PhotonPusher *ppusher,
                     int ip) {

  // Stop tracking once the photon is inside the horizon.  It is causally disconnected,
  // and Kerr-Schild coordinates stay regular there, so the integrator would happily keep
  // stepping and contaminate the statistic with trajectories nobody cares about.
  if (pphot->x1p[ip] < r_hor) {
    pphot->statp[ip] = ABSORBED;
    return;
  }

  Real kt, kphi;
  CovariantK(ppusher->pcoord, pphot, ip, kt, kphi);

  if (kt < pphot->user[IKTMIN][ip]) pphot->user[IKTMIN][ip] = kt;
  if (kt > pphot->user[IKTMAX][ip]) pphot->user[IKTMAX][ip] = kt;
  if (kphi < pphot->user[IKPMIN][ip]) pphot->user[IKPMIN][ip] = kphi;
  if (kphi > pphot->user[IKPMAX][ip]) pphot->user[IKPMAX][ip] = kphi;

  return;
}

} // namespace
