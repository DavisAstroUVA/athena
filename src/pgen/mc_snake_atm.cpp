//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mc_snake_atm.cpp
//! \brief Isothermal scattering atmosphere in sinusoidal ("snake") coordinates.
//
// The same physical problem as mc_isoth -- a plane-parallel isothermal atmosphere whose
// density falls exponentially with height, emitting free-free and scattering off
// electrons -- but written in the sheared chart of White, Stone & Gammie (2016),
// ApJS 225, 22:
//
//     t = t_M,  x = x_M,  y = y_M + a sin(k x_M),  z = z_M
//
// so with beta = a k cos(k x),
//
//     ds^2 = -dt^2 + (1 + beta^2) dx^2 - 2 beta dx dy + dy^2 + dz^2.
//
// The point is that this spacetime is genuinely flat: the shear is a change of chart and
// nothing more.  Stratifying along x3 keeps the physics identical to the cartesian case,
// because z = z_M exactly and no metric component mixes x3 with anything, so the emergent
// polarized intensity must reproduce the same Feautrier solution that mc_isoth is checked
// against.  Everything the general pusher does in between -- integrating geodesics
// through a non-diagonal metric, parallel transporting the coherency tensor, and rotating
// into and out of the scattering frame at every scattering -- must cancel exactly for
// that to happen, which is what makes this a test rather than a demonstration.
//
// Units.  The GR equation of state needs p/rho << 1, so the hydro primitives are
// dimensionless and the Monte Carlo scales them: rho_cgs multiplies the code density and
// tgas_cgs multiplies p/rho to give the temperature.  Both are derived here from taumin,
// taumax and temp and written into the input, so a deck specifies the atmosphere the same
// way it would for mc_isoth and cannot get the conversion wrong.  The code density is
// normalised to one at the top of the atmosphere, which keeps p/rho at its smallest where
// the density is lowest.
//
// Inputs:
//   <coord>/snake_a    shear amplitude       (default 0 -> degenerates to Minkowski)
//   <coord>/snake_k    shear wavenumber      (default 0)
//   <coord>/m, a       required by GRUser and ignored here; set both to 0
//   <problem>/taumin   scattering depth at the top of the atmosphere
//   <problem>/taumax   scattering depth at the bottom
//   <problem>/temp     gas temperature in K
//   <problem>/emin,emax  photon energy range, in eV unless tnorm is set

// C++ headers
#include <cmath>
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
#error "mc_snake_atm requires general relativity (-g)"
#endif

#include "../monte_carlo/montecarlo.hpp"
#include "../monte_carlo/photon.hpp"
#include "../monte_carlo/photonpusher.hpp"

namespace {
bool tnorm;
Real logemin, logemax;
// Cached so the metric callback, which runs per cell, does not re-parse the input.
Real snake_a = 0.0;
Real snake_k = 0.0;
// Atmosphere shape, set in InitUserMeshData and used by ProblemGenerator.  l0 is the
// density scale height; x3top is where the code density is normalised to one.
Real l0 = 1.0;
Real x3top = 0.0;
// p/rho held fixed so the atmosphere is isothermal and the GR equation of state stays in
// the regime it is valid in.  Folded into tgas_cgs, so it never reaches the temperature.
const Real kPgasRatio = 1.0e-6;

void SnakeMetric(Real x1, Real x2, Real x3, ParameterInput *pin,
                 AthenaArray<Real> &g, AthenaArray<Real> &g_inv,
                 AthenaArray<Real> &dg_dx1, AthenaArray<Real> &dg_dx2,
                 AthenaArray<Real> &dg_dx3);
void JMeanOpacity(MonteCarloBlock *pmcb, Photon *pphot, int ip, int imom,
                  const PhotonFrameState &s);
}  // namespace

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief enroll the snake metric and derive the unit conversions from the optical depths
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {

  if (std::strcmp(COORDINATE_SYSTEM, "gr_user") != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in mc_snake_atm InitUserMeshData" << std::endl
        << "mc_snake_atm supplies its own metric and requires --coord=gr_user, got "
        << COORDINATE_SYSTEM << std::endl;
    ATHENA_ERROR(msg);
  }

  snake_a = pin->GetOrAddReal("coord", "snake_a", 0.0);
  snake_k = pin->GetOrAddReal("coord", "snake_k", 0.0);

  // beta depends on x1 alone, so a periodic x1 boundary needs the shear to close over the
  // box or the metric is discontinuous across the seam.  x2 is always safe.
  std::string ix1 = pin->GetOrAddString("mesh", "ix1_mc_bc", "");
  if (ix1 == "periodic" && snake_a != 0.0 && snake_k != 0.0) {
    Real lx1 = pin->GetReal("mesh", "x1max") - pin->GetReal("mesh", "x1min");
    Real turns = snake_k * lx1 / (2.0 * PI);
    if (std::fabs(turns - std::round(turns)) > 1.0e-10) {
      std::stringstream msg;
      msg << "### FATAL ERROR in mc_snake_atm InitUserMeshData" << std::endl
          << "Periodic x1 requires the shear to close over the box: snake_k * Lx1 must "
          << "be a multiple of 2 pi." << std::endl
          << "Got snake_k = " << snake_k << ", Lx1 = " << lx1
          << ", snake_k*Lx1/(2 pi) = " << turns << std::endl;
      ATHENA_ERROR(msg);
    }
  }

  // Atmosphere, stratified along x3.  Deliberately x3 and not x1 or x2: z = z_M exactly
  // and no metric component depends on or mixes x3, so the stratification is untouched by
  // the shear and the plane-parallel transfer problem is the cartesian one.
  Real taumin = pin->GetReal("problem", "taumin");
  Real taumax = pin->GetReal("problem", "taumax");
  if (taumin <= 0.0 || taumax <= taumin) {
    std::stringstream msg;
    msg << "### FATAL ERROR in mc_snake_atm InitUserMeshData" << std::endl
        << "need 0 < taumin < taumax, got taumin = " << taumin
        << ", taumax = " << taumax << std::endl;
    ATHENA_ERROR(msg);
  }
  Real x3min = pin->GetReal("mesh", "x3min");
  x3top = pin->GetReal("mesh", "x3max");
  l0 = (x3top - x3min) / std::log(taumax / taumin);

  // Electron scattering opacity, matching mc_isoth
  const Real heabund = 0.09;
  const Real mp = 1.6726e-24;
  const Real sigmat = 6.65248e-25;
  Real kappaes = sigmat * (1.0 + 2.0*heabund) / (mp * (1.0 + 4.0*heabund));

  // The code density is exp((x3top - x3)/l0), which is one at the top, so rho_cgs is the
  // physical density there: taumin = kappaes * rho_cgs * l0.
  pin->SetReal("problem", "rho_cgs", taumin / (l0 * kappaes));
  // temp = tgas_cgs * p/rho, and p/rho is held at kPgasRatio, so this returns temp.
  pin->SetReal("problem", "tgas_cgs", pin->GetReal("problem", "temp") / kPgasRatio);
  // lengths are already cgs
  pin->SetReal("problem", "l_cgs", 1.0);

  EnrollUserMetric(SnakeMetric);

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief exponential isothermal atmosphere, stratified along x3
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  AthenaArray<Real> bb;
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        Real rho = std::exp((x3top - pcoord->x3v(k)) / l0);
        phydro->w(IDN, k, j, i) = phydro->w1(IDN, k, j, i) = rho;
        // pressure tracks density so the atmosphere is isothermal
        phydro->w(IPR, k, j, i) = phydro->w1(IPR, k, j, i) = kPgasRatio * rho;
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
//! \fn void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype)
//! \brief free-free emission, as in mc_isoth
//========================================================================================

void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype) {

  SetEmissionCellWeight(pphot, ips, ipe);

  for (int ip = ips; ip <= ipe; ip++) {

    pphot->dtp[ip] = pphot->pmy_mcb->pmy_mc->tmax;

    GetZonePosition(pphot, pran, pcoord, ip);
    pphot->x0p[ip] = 0.0;

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
    msg << "### FATAL ERROR in mc_snake_atm InitUserMonteCarloData" << std::endl
        << "mc_snake_atm requires general_pusher = true" << std::endl;
    ATHENA_ERROR(msg);
  }

  // gr_user does not say which metric the Monte Carlo integrates on, so it has to be
  // named, and named as this one.
  if (pin->GetOrAddString("montecarlo", "mc_coord", "") != "snake") {
    std::stringstream msg;
    msg << "### FATAL ERROR in mc_snake_atm InitUserMonteCarloData" << std::endl
        << "mc_snake_atm requires <montecarlo>/mc_coord = snake" << std::endl;
    ATHENA_ERROR(msg);
  }

  AllocateUserMoments(1);
  EnrollUserMoment(0, JMeanOpacity, "kapJ");

  return;
}

namespace {

//----------------------------------------------------------------------------------------
//! \fn void SnakeMetric(...)
//! \brief the snake metric, its inverse and its coordinate derivatives
//
// Mirrors MCSnake in mccoord.cpp.  The spatial block has unit determinant,
// (1+beta^2) - beta^2 = 1, so the inverse swaps the diagonal and flips the sign of the
// off-diagonal, and sqrt(-g) = 1.  beta depends on x1 alone, so dg_dx2 and dg_dx3 vanish.

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

  g(I00) = -1.0;
  g(I11) = 1.0 + SQR(beta);
  g(I12) = -beta;
  g(I22) = 1.0;
  g(I33) = 1.0;

  g_inv(I00) = -1.0;
  g_inv(I11) = 1.0;
  g_inv(I12) = beta;
  g_inv(I22) = 1.0 + SQR(beta);
  g_inv(I33) = 1.0;

  dg_dx1(I11) = 2.0 * beta * dbeta;
  dg_dx1(I12) = -dbeta;

  return;
}

void JMeanOpacity(MonteCarloBlock *pmcb, Photon *pphot, int ip, int imom,
                  const PhotonFrameState &s) {

  int i1 = pphot->i1p[ip];
  int i2 = pphot->i2p[ip];
  int i3 = pphot->i3p[ip];

  Real weight = pphot->wp[ip]*s.e*s.dl/MCConstants::c_cgs;
  pmcb->moments_user(imom, i3, i2, i1) += weight*pphot->acp[ip];
}

}  // namespace
