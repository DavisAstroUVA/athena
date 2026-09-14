//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mc_sphere_ion.cpp
//! \brief A hydrostatic atmosphere that tracks ionization using a passive scalar.

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdio>     // fopen(), fprintf(), freopen()
#include <cstring>    // strcmp()
#include <sstream>
#include <stdexcept>
#include <string>
#include <iostream>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../monte_carlo/montecarlo.hpp"
#include "../monte_carlo/photon.hpp"
#include "../monte_carlo/photonpusher.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"

void TrackIonization(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);
namespace {
  Real rin,rout;
  Real energy0;
  int i1,i2,i3;
  Real rhobase;
  Real IonizedEmission(MonteCarloBlock *pmcb, int k, int j, int i, int etype);

}
//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Spherical atmosphere in hydrostatic balance
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {

  EnrollUserExplicitSourceFunction(TrackIonization);
}

void MonteCarlo::InitUserMonteCarloData(ParameterInput *pin){

  EnrollUserEmissionFunction(IonizedEmission);
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  Real kb    = 1.380649e-16;
  Real mp    = 1.6726e-24;
  Real clight = 2.997924589e10;
  Real sigma_pi = 6.3e-18;// cm2

  Real GM    = pin->GetOrAddReal("problem", "GM", 1.2668653e23);
  Real mmw   = pin->GetOrAddReal("problem", "mmw", 1.); // Mean molecular weight of gas
  rin   = pin->GetReal("problem", "rin"); // Radius of base of atmosphere
  rout  = pin->GetReal("problem", "rout"); // Radius of edge of atmosphere
  Real pbase = pin->GetOrAddReal("problem", "pbase", 10.); // Atmospheric pressure at rin
  Real temp  = pin->GetOrAddReal("problem", "temp", 1.e4); // Isothermal temperature
  rhobase = pbase * mmw * mp / kb / temp;
  Real gamma = peos->GetGamma();
  Real gm1   = gamma - 1.0;
  pmy_mcb->tgas_cgs = mmw * mp / kb;
  pmy_mcb->rho_cgs = rhobase;
  // Calculate density where n_H = n_p in optically thin limit
  Real Gamma0 = 1. / (6. * 60. * 60.); // s, photoionization rate coefficient
  Real alpha = 4.18e-13;
  Real neq0 = Gamma0 / alpha;

  // set up ambient medium at equilibrium for an isothermal atmosphere
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      Real column = 0.;
      for (int i=ie+NGHOST; i>=is-NGHOST; --i) {
        Real r = pcoord->x1v(i);
        phydro->u(IDN,k,j,i) = exp(GM * mmw * mp / kb / temp *
                                   (1./pcoord->x1v(i) - 1./rin));
        Real rho = phydro->u(IDN,k,j,i) * rhobase;

        // Calculate neutral hydrogen column
        Real n_H = std::pow(0.5 * (std::sqrt(neq0 + 4.*rho/mp) - std::sqrt(neq0)), 2.);
        Real dr = pcoord->dx1f(i);
        column += n_H * dr;

        // Calculate ionization rate, initialize ionization fraction scalar
        // Trammell et al 2011, Fig 9 powerlaw
        Real Gamma = Gamma0 / (1. + std::pow(sigma_pi * column, 1.5));

        pscalars->s(0,k,j,i) = (std::sqrt(Gamma) * std::sqrt(4.*alpha*rho/mp + Gamma)
                                / alpha - Gamma / alpha) * 0.5 * mp;
        pscalars->r(0,k,j,i) = pscalars->s(0,k,j,i) / rho;
        //printf("%d %d %d %g\n", k, j, i, pscalars->s(0,k,j,i));

        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        // Fluid internal energy density
        phydro->u(IEN,k,j,i) = phydro->u(IDN,k,j,i) * kb * temp
          / mmw / mp / gm1;
        phydro->u(IEN,k,j,i) += 0.5 * SQR(phydro->u(IM1,k,j,i)) / phydro->u(IDN,k,j,i);
        phydro->u(IEN,k,j,i) += 0.5 * SQR(phydro->u(IM2,k,j,i)) / phydro->u(IDN,k,j,i);
        phydro->u(IEN,k,j,i) += 0.5 * SQR(phydro->u(IM3,k,j,i)) / phydro->u(IDN,k,j,i);
      }
    }
  }
}

//========================================================================================
//! \fn void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin)
//! \brief Analogous to problem generator but used in support of InitializePhoton
//========================================================================================

void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin) {

  const Real mp    = 1.6726e-24;
  const Real kb = 1.380649e-16;
  const Real c = 2.99792458e10;
  const Real nu0 = MCConstants::nu_lya;
  const Real lorw = 6.265e8/(4.*PI);
  const Real h = 6.62607015e-27;

  Real x0 = pin->GetReal("problem","x0");
  Real temp = pin->GetReal("problem","temp");
  Real vth = sqrt( 2. * kb * temp / mp);
  Real dopw = nu0 * vth / c;
  energy0 = h * (nu0 + dopw * x0);

  // Set variables
  rin = pin->GetReal("problem","rin");
  rout = pin->GetReal("problem","rout");

}

//========================================================================================
//! \fn void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype)
//! \brief Initializes Photon packets before integration
//========================================================================================

void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype) {

  // Set initial cells and emission weights for all photon samples
  SetEmissionCellWeight(pphot,ips,ipe);

  for (int ip=ips; ip<=ipe; ip++) {

    pphot->dtp[ip] = pphot->pmy_mcb->pmy_mc->tmax;

    // Obtain initial position within zone
    GetZonePosition(pphot,pran,pcoord,ip);

    // Sample isotropic angle in orthonormal basis
    Real mu = 2.*pran->uniform()-1.0;
    Real stheta = std::sqrt(1.0-mu*mu);
    Real phi = 2.*PI*pran->uniform();
    // k0p is the photon energy (set via ep); the light-travel time bookkeeping
    // below now divides by c explicitly instead of stashing 1/c in k0p.
    pphot->k1p[ip] = stheta * cos(phi);
    pphot->k2p[ip] = stheta * sin(phi);
    pphot->k3p[ip] = mu;

    // Resize direction vector to sphpol code coords
    if (pphot->general_pusher_flag) {
      pphot->k2p[ip] /= pphot->x1p[ip];
      pphot->k3p[ip] /= pphot->x1p[ip] * sin(pphot->x2p[ip]);
    }
    pphot->ep[ip] = energy0;

    // Set status flag
    if (pphot->wp[ip] < 0.0)
      pphot->statp[ip] = DESTROYED;
    else
      pphot->statp[ip] = EVOLVING;

    // Initialize the absorption and scattering extinction coefficients
    // to the values appropriate in the emitted zone
    pphot->acp[ip] = AbsorptionOpacity(this,pphot,ip);
    pphot->scp[ip] = ScatteringOpacity(this,pphot,ip);
    //pphot->PrintPhoton(ip);
  } // loop over ip

}


void TrackIonization(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar) {
  Real mp = 1.6726e-24;
  Real sigma_pi = 6.3e-18;// cm2
  Real Gamma0 = 1. / (6. * 60. * 60.); // s, photoionization rate coefficient
  Real alpha = 4.18e-13; // cm3 s-1 for T=1e4 K, recombination rate coefficient
  Real nR = 1. / alpha / dt;

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      Real column = 0.;
      for (int i=pmb->ie+NGHOST; i>=pmb->is-NGHOST; --i) {
        Real rho = rhobase * prim(IDN,k,j,i);
        Real n_p = cons_scalar(0,k,j,i)/mp;
        Real n_H = rho/mp - n_p;
        Real dr = pmb->pcoord->dx1f(i);
        column += n_H * dr;
        // Trammell et al 2011, Fig 9 powerlaw
        Real Gamma = Gamma0 / (1. + std::pow(sigma_pi * column, 1.5));
        Real nC = Gamma / alpha;
        //if (time <= 0.) {
        //  printf("%g %g %g %g %g %g\n", pmb->pcoord->x1v(i), rho, nC, n_H,
        //         cons_scalar(0,k,j,i)/rho, column);
        //}
        cons_scalar(0,k,j,i) = 0.5 * mp * (-(nC + nR) + std::sqrt((nC + nR)*(nC + nR)
                               + 4. * (nC*rho/mp + nR*n_p)));
      }
    }
  }
  return;
}

namespace {
Real IonizedEmission(MonteCarloBlock *pmcb, int k, int j, int i, int etype) {

  const Real mp = 1.6726e-24;
  const Real alpha = 4.18e-13;
  Real np = pmcb->scalars(k,j,i) / mp;
  return alpha * SQR(np);

}

} //end namespace
