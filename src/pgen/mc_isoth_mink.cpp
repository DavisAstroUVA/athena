//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mc_isoth_mink.cpp
//! \brief Uniform periodic box in Minkowski coordinates, built with -g.
//!
//! This is the mc_isoth uniform box run through the general relativistic machinery.
//! Spacetime is flat, so the answer is exactly the one the cartesian tests already use --
//! a blackbody at the gas temperature, boosted if the fluid moves -- but the code path is
//! completely different: GeneralPusher geodesic integration, ConstructTetrad, the GR
//! branches of TransformToComoving/TransformToCoordinate, and the general-pusher branches
//! of UpdateMoments.  Any deviation from the cartesian answer is a bug in that machinery
//! rather than a modelling difference.
//!
//! Units.  Minkowski has no mass scale, so nothing forces geometrized units: the
//! coordinates are centimetres (l_cgs = 1) and the metric is dimensionless either way.
//! The hydro primitives are a different matter -- with c = 1 the GR equation of state
//! needs p/rho << 1, which CGS values would badly violate.  So the primitives are carried
//! as dimensionless code values and converted for the Monte Carlo by the existing
//! rho_cgs and tgas_cgs inputs:
//!
//!     rho_mc  = rho_cgs  * w(IDN)                 (see GetDensity)
//!     T_mc    = tgas_cgs * w(IPR)/w(IDN)          (see GetTemperature)
//!
//! so set w(IDN) = 1, w(IPR) = pgas_code, rho_cgs = the physical density, and
//! tgas_cgs = T_physical/pgas_code.  The convergence script does this arithmetic.

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
#error "This problem requires monte carlo"
#endif
#if !GENERAL_RELATIVITY
#error "mc_isoth_mink requires general relativity (-g)"
#endif

#include "../monte_carlo/montecarlo.hpp"
#include "../monte_carlo/tetrad.hpp"
#include "../monte_carlo/photon.hpp"
#include "../monte_carlo/photonpusher.hpp"

namespace {
bool tnorm;
Real logemin, logemax;
void JMeanOpacity(MonteCarloBlock *pmcb, Photon *pphot, int ip, int imom,
                  const PhotonFrameState &s);
void AverageEnergy(MonteCarloBlock *pmcb, Photon *pphot, int ip, int imom,
                  const PhotonFrameState &s);
} // namespace

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief uniform box, optionally moving at a constant velocity along x3
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  if (COORDINATE_SYSTEM != "minkowski") {
    std::stringstream msg;
    msg << "### FATAL ERROR in mc_isoth_mink ProblemGenerator" << std::endl
        << "mc_isoth_mink requires Minkowski coordinates, got "
        << COORDINATE_SYSTEM << std::endl;
    ATHENA_ERROR(msg);
  }

  // dimensionless code values; the Monte Carlo scales them via rho_cgs/tgas_cgs
  Real rho0 = pin->GetOrAddReal("problem", "dens_code", 1.0);
  Real pgas0 = pin->GetOrAddReal("problem", "pgas_code", 1.0e-6);

  // constant velocity as a fraction of the speed of light
  Real beta = pin->GetOrAddReal("problem", "velocity", 0.);
  if (std::fabs(beta) >= 1.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in mc_isoth_mink ProblemGenerator" << std::endl
        << "velocity must be given as a fraction of c, got " << beta << std::endl;
    ATHENA_ERROR(msg);
  }
  Real lorentz = 1.0 / std::sqrt(1.0 - SQR(beta));

  // Athena's GR velocity primitive is utilde^i, with
  // gamma = sqrt(1 + g_ij utilde^i utilde^j).  For the flat metric g_33 = 1, so
  // utilde^3 = gamma*beta reproduces u^mu = (gamma, 0, 0, gamma*beta).
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
//! \brief free-free emission, identical to mc_isoth
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

  return;
}

//========================================================================================
//! \fn void MonteCarlo::InitUserMonteCarloData(ParameterInput *pin)
//========================================================================================

void MonteCarlo::InitUserMonteCarloData(ParameterInput *pin) {

  if (!pin->GetOrAddBoolean("montecarlo", "general_pusher", false)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in mc_isoth_mink InitUserMonteCarloData" << std::endl
        << "mc_isoth_mink requires general_pusher = true" << std::endl;
    ATHENA_ERROR(msg);
  }

  nuser_var = 1;
  AllocateUserMoments(2);
  EnrollUserMoment(0, JMeanOpacity, "kapJ");
  EnrollUserMoment(1, AverageEnergy, "eave");

  return;
}

namespace {

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
