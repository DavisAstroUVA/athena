//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mc_sph_tran.cpp
//! \brief Test photon transport through spherical polar grid
//
//========================================================================================

// C++ headers
#include <iostream> // temporary for testing

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../monte_carlo/montecarlo.hpp"
#include "../monte_carlo/photon.hpp"
#include "../monte_carlo/photonpusher.hpp"

#if !MONTE_CARLO_ENABLED
#error "This problem requires monte carlo"
#endif


namespace {
  // user defined functions
  Real StepFunctionOpacity(MonteCarloBlock *pmcb, Photon *pphot, int ip);

  // Global variables
  int irmax;
  Real opac;

} // namespace

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief monte carlo test problem generator
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  Real rideal = 8.314e7;
  Real rho0 = 1.;
  Real tgas = 1.;
  Real gamma = peos->GetGamma();

  // Set nominal values for grid, unused
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real x1 = pcoord->x1f(i+1);
        if (x1 <= 1.0){
          phydro->u(IDN,k,j,i) = rho0;
          phydro->u(IEN,k,j,i) = tgas * phydro->u(IDN,k,j,i)/(gamma-1.0);
        }
        else {
          phydro->u(IDN,k,j,i) = 1e-7;
          phydro->u(IEN,k,j,i) = 0.0;
        }
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
      }
    }
  }

}

//========================================================================================
//! \fn void MonteCarlo::InitUserMonteCarloData(ParameterInput *pin)
//! \brief Initializes user data specific to MonteCarlo class
//========================================================================================

void MonteCarlo::InitUserMonteCarloData(ParameterInput *pin) {

  EnrollUserOpacityFunction(StepFunctionOpacity,true);
}

//========================================================================================
//! \fn void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin)
//! \brief Analogous to problem generator but used in support of InitializePhoton
//========================================================================================

void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin) {

  opac = pin->GetReal("problem","opac");
  irmax = 0;
  for(int i=is; i<=ie; i++) {
    if (pcoord->x1f(i+1) <= 1.0) {
      if (i > irmax)
        irmax = i;
    }
  }

  Real ncells = static_cast<Real>(pmy_mc->ncells);
  Real ratio = static_cast<Real>(ie-is+1)/static_cast<Real>(irmax-is+1);
  printf("ratio: %g\n",ratio);
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        if (i <= irmax)
          emission(k,j,i) = opac*pcoord->vol(k,j,i)*ncells/ratio;
        else
          emission(k,j,i) = 0.;
      }
    }
  }
}


//========================================================================================
//! \fn void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype)
//! \brief Initializes Photon packets before integration
//========================================================================================

void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype) {


  // Get meshblock dimensions

  Real nx1 = static_cast<Real>(irmax-is+1);
  Real nx2 = static_cast<Real>(je-js+1);
  Real nx3 = static_cast<Real>(ke-ks+1);

  for (int ip=ips; ip<=ipe; ip++) {

    // Randomly assign emission zone
    int i1,i2,i3;
    pphot->i1p[ip] = i1 = static_cast<int>(pran->uniform()*nx1)+is;
    pphot->i2p[ip] = i2 = static_cast<int>(pran->uniform()*nx2)+js;
    pphot->i3p[ip] = i3 = static_cast<int>(pran->uniform()*nx3)+ks;

    // emit all photons with same weight and energy
    pphot->wp[ip] = emission(i3,i2,i1);
    pphot->ep[ip] = 1.0;

    // Obtain initial position within zone
    GetZonePosition(pphot,pran,pcoord,ip);
    pphot->x0p[ip] = 0.0;
    Real phi = 2. * PI * pran->uniform();
    Real cph = cos(phi);
    Real sph = sin(phi);
    Real cth = 2. * pran->uniform() - 1.;
    Real sth = sqrt(1. - SQR(cth));

    // Initialize wave vector with isotropic distribution.  k0p carries the photon
    // energy and is set via ep, so only the spatial components are set here.
    pphot->k1p[ip] = sth*cph;
    pphot->k2p[ip] = sth*sph;
    pphot->k3p[ip] = cth;

    // Convert k unit vector to k^\alpha
    if (pmy_mc->general_pusher_flag) {
      pphot->k2p[ip] /= pphot->x1p[ip];
      pphot->k3p[ip] /= (pphot->x1p[ip]*sin(pphot->x2p[ip]));
      pphot->dk0p[ip] = 0.;
      pphot->dk1p[ip] = 0.;
      pphot->dk2p[ip] = 0.;
      pphot->dk3p[ip] = 0.;
    }

    // Initialize Stokes vector
    if (IsPolarized(pphot->polarized)) {
      pphot->sip[ip] = 1.0;
      pphot->sqp[ip] = 0.0;
      pphot->sup[ip] = 0.0;
      pphot->svp[ip] = 0.0;
    }

    // Initialize the absorption and scattering extinction coefficients
    pphot->acp[ip] = opac; // all photons start with this opacity
    pphot->scp[ip] = 0.0;

    //pphot->PrintPhoton(ip);

  } // loop over ip
}

//========================================================================================
//! \fn void MonteCarloBlock::FinalizePhoton(Photon *pphot, ip)
//! \brief Complete work at end of photon packets before integration
//========================================================================================

void MonteCarloBlock::FinalizePhoton(Photon *pphot, int ip) {

}


namespace {

//----------------------------------------------------------------------------------------
//! \fn Real StepFunctionOpacity(MonteCarloBlock *pmcb, Photon *pphot, int ip)
//! \brief return 0 or 10 for extinction coeffictent

Real StepFunctionOpacity(MonteCarloBlock *pmcb, Photon *pphot, int ip) {

  // Opacity is position dependent
  if (pmcb->pcoord->x1f(pphot->i1p[ip]+1) <= 1.)
    return opac;
  else
    return 0.;
}

} //namespace
