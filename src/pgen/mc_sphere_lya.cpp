//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mc_sphere_lya.cpp
//  \brief Problem generator for monte carlo through uniform isothermal sphere with lyman
//         alpha scattering
//
//========================================================================================

// C/C++ headers
#include <iostream>
#include <stdexcept>

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
#include "../globals.hpp"

#if !MONTE_CARLO_ENABLED
#error "This problem requires monte carlo"
#endif

namespace {
  // Global variables
  Real rad0,time0;
  Real energy0;
  int i1start,i2start,i3start;

  // function headers
  void SphericalEscape(MonteCarloBlock *pmcb, Photon *phot, PhotonPusher *ppusher, int ip);
  void TimedEscape(MonteCarloBlock *pmcb, Photon *phot, PhotonPusher *ppusher, int ip);
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief monte carlo test problem generator
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  Real rideal = 8.314e7;
  Real c = 2.99792458e10;
  Real temp = pin->GetReal("problem","temp");
  Real tau = pin->GetReal("problem","tau");
  Real rad0;
  if (COORDINATE_SYSTEM == "cartesian") {
    rad0 = pin->GetReal("problem","radius");
  } else if (COORDINATE_SYSTEM == "spherical_polar") {
    rad0 = pcoord->x1f(ie+1);
    printf("rad0: %g\n",rad0);
  }
  Real vel = pin->GetOrAddReal("problem","velocity",0.);
  Real gamma = peos->GetGamma();
  vel *= c;

  Real kb = 1.380649e-16;
  Real mass = 1.660538782e-24;
  Real vth = sqrt(2.*kb*temp/mass);
  Real nu0 = 2.468e15;
  Real dopw = nu0 * vth / c;
  Real kappa = ResLinePre() / (mass*sqrt(PI)*dopw);
  Real rho = tau / (kappa * rad0);
  //printf("kappa: %g %g %g\n",kappa,ResLinePre(),rho/mass);
  //printf("voigt: %g\n",XsecVoigt(nu0,temp));
  //printf("rho: %g %g %g %g\n",rho,kappaes,rad0,tau);
  // density is non-zero only in sphere
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        phydro->u(IDN,k,j,i) = rho;
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = rho*vel;
        phydro->u(IEN,k,j,i) = rideal*rho*temp/(gamma-1.0);
      }
    }
  }

  // add kinetic energy
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        phydro->u(IEN,k,j,i) += 0.5*SQR(phydro->u(IM1,k,j,i))/phydro->u(IDN,k,j,i);
        phydro->u(IEN,k,j,i) += 0.5*SQR(phydro->u(IM2,k,j,i))/phydro->u(IDN,k,j,i);
        phydro->u(IEN,k,j,i) += 0.5*SQR(phydro->u(IM3,k,j,i))/phydro->u(IDN,k,j,i);
      }}}
}

//========================================================================================
//! \fn void MonteCarlo::InitUserMonteCarloData(ParameterInput *pin)
//! \brief Initializes user data specific to MonteCarlo class
//========================================================================================

void MonteCarlo::InitUserMonteCarloData(ParameterInput *pin){

  nuser_var = 3;
  // If time is set in problem generator, terminate photon integration based on time
  // but if not terminated based on radius
  Real time = pin->GetOrAddReal("problem","time",-1.);
  if (time > 0.) {
    EnrollUserWorkInMove(TimedEscape);
  } else {
    if (COORDINATE_SYSTEM == "cartesian") {
      EnrollUserWorkInMove(SphericalEscape);
    }
  }
}

//========================================================================================
//! \fn void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin)
//! \brief Analogous to problem generator but used in support of InitializePhoton
//========================================================================================

void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin) {

  Real x0 = pin->GetReal("problem","x0");
  Real temp = pin->GetReal("problem","temp");
  Real kb = 1.380649e-16;
  Real mass = 1.660538782e-24;
  Real vth = sqrt( 2. * kb * temp / mass);
  Real c = 2.99792458e10;
  Real nu0 = 2.468e15;
  Real dopw = nu0 * vth / c;
  Real lorw = 6.265e8/(4.*PI);
  //printf("dop, lor, a: %e %e %e\n",dopw,lorw,lorw/dopw);
  Real h = 6.62607015e-27;
  energy0 = h * (nu0 + dopw * x0);

  rad0 = pin->GetReal("problem","radius");
  time0 = pin->GetOrAddReal("problem","time",-1.);

  if (COORDINATE_SYSTEM == "cartesian") {
    // Deterime cell of initial photon, which is asssumed to include
    // if origin if more than one cell is specified for each direction
    i1start = -1;
    for(int i=is; i<=ie; i++) {
      if ((0. > pcoord->x1f(i)) && (0. <= pcoord->x1f(i+1)))
        i1start = i;
    }
    i2start = -1;
    for(int i=js; i<=je; i++) {
      if ((0. > pcoord->x2f(i)) && (0. <= pcoord->x2f(i+1)))
        i2start = i;
    }
    i3start = -1;
    for(int i=ks; i<=ke; i++) {
      if ((0. > pcoord->x3f(i)) && (0. <= pcoord->x3f(i+1)))
        i3start = i;
    }
    if ((i1start < 0) || (i2start < 0) || (i3start < 0)) {
      std::stringstream msg;
      msg << "### FATAL ERROR in InitUserMonteCarloBlockData" << std::endl
          << "Origin not found within domain." << std::endl;
      throw std::runtime_error(msg.str().c_str());
    }
  }
  // Set number of samples per block because emmision_flag is set to EMISNONE
  nphremain = pin->GetInteger("montecarlo", "nphot");
  nphrun = 0;
}


//========================================================================================
//! \fn void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype)
//! \brief Initializes Photon packets before integration
//========================================================================================

void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype) {

  if (COORDINATE_SYSTEM == "spherical_polar") {
    Real nx2 = static_cast<Real>(je-js+1);
    Real nx3 = static_cast<Real>(ke-ks+1);
  }

  for (int ip=ips; ip<=ipe; ip++) {

    pphot->user[0][ip] = 0.;
    pphot->user[1][ip] = 0.;
    pphot->user[2][ip] = 0.;

    // Set status flag
    pphot->statp[ip] = EVOLVING;
    int i1,i2,i3;
    if (COORDINATE_SYSTEM == "cartesian") {
      // Initialize photon at the origin
      pphot->i1p[ip] = i1 = i1start;
      pphot->i2p[ip] = i2 = i2start;
      pphot->i3p[ip] = i3 = i3start;
      pphot->x1p[ip] = 0.;
      pphot->x2p[ip] = 0.;
      pphot->x3p[ip] = 0.;
      pphot->x0p[ip] = 0.; //time

      // Generate initial angle parameters
      Real phi = 2. * PI * pran->uniform();
      Real cphi = cos(phi);
      Real sphi = sin(phi);
      Real cth = 2. * pran->uniform() - 1.;
      Real sth = sqrt(1. - SQR(cth));
      // Initialize wave vector with isotropic distribution
      pphot->k1p[ip] = sth*cphi;
      pphot->k2p[ip] = sth*sphi;
      pphot->k3p[ip] = cth;
      // k0p is the photon energy (set via ep); the light-travel time bookkeeping
      // below now divides by c explicitly instead of stashing 1/c in k0p.

    } else if (COORDINATE_SYSTEM == "spherical_polar") {
      pphot->i1p[ip] = i1 = is;
      pphot->i2p[ip] = i2 = static_cast<int>(pran->uniform()*nx2)+js;
      pphot->i3p[ip] = i3 = static_cast<int>(pran->uniform()*nx3)+ks;
      // Obtain initial position within zone, assumes r(is) << r(is+1)
      pphot->x1p[ip] = pcoord->x1f(pphot->i1p[ip]) * 100.; // at inner edge
      Real cthh = cos(pcoord->x2f(pphot->i2p[ip]));
      Real cthl = cos(pcoord->x2f(pphot->i2p[ip]+1));
      Real cth = cthl + pran->uniform() * (cthh-cthl);
      pphot->x2p[ip] = acos(cth);
      Real pl = pcoord->x3f(pphot->i3p[ip]); Real dp = pcoord->x3f(pphot->i3p[ip]+1)-pl;
      pphot->x3p[ip] = pl+pran->uniform()*dp;
      pphot->x0p[ip] = 0.; //time
      // Initialize wave vector so that it is parallel with r direction
      pphot->k1p[ip] = 1.0;
      pphot->k2p[ip] = 0.;
      pphot->k3p[ip] = 0.;
      // k0p is the photon energy (set via ep); the light-travel time bookkeeping
      // below now divides by c explicitly instead of stashing 1/c in k0p.
    }

    // Initialize Photon weights, energy, direction, polarization
    // TODO: Make this an input rather than hardcoding
    Real target_lum = 1.e20;
    pphot->wp[ip] = target_lum / energy0;
    pphot->ep[ip] = energy0;

    // Initialize Stokes vector
    if (IsPolarized(pphot->polarized)) {
      pphot->sip[ip] = 1.0;
      pphot->sqp[ip] = 0.0;
      pphot->sup[ip] = 0.0;
      pphot->svp[ip] = 0.0;
    }

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



namespace {

// Used to evalue photons time distribution as fixed spherical
// escape surface
void SphericalEscape(MonteCarloBlock *pmcb, Photon *pphot, PhotonPusher *ppusher,
                     int ip) {

  pphot->user[0][ip] += ppusher->dl * pphot->wp[ip];
  pphot->user[1][ip] += ppusher->dl * pphot->wp[ip] * pphot->ep[ip];
  pphot->user[2][ip] += ppusher->dl * pphot->wp[ip] * pphot->acp[ip];

  // First check radius condition
  Real r = sqrt(SQR(pphot->x1p[ip])+SQR(pphot->x2p[ip])+SQR(pphot->x3p[ip]));
  if (r >= rad0) {
    Real dr = r-rad0;
    // assume cartesian for now
    pphot->x0p[ip] -= dr/2.99792458e10;
    pphot->x1p[ip] -= pphot->k1p[ip]*dr;
    pphot->x2p[ip] -= pphot->k2p[ip]*dr;
    pphot->x3p[ip] -= pphot->k3p[ip]*dr;

    pphot->statp[ip] = ESCAPED;
    //pphot->face = BoundaryFace::undef;
  }

}

// Used to test photons radial distributions after a fixed travel time
void TimedEscape(MonteCarloBlock *pmcb, Photon *pphot, PhotonPusher *ppusher,
                 int ip) {

  // First check radius condition
  Real r = sqrt(SQR(pphot->x1p[ip])+SQR(pphot->x2p[ip])+SQR(pphot->x3p[ip]));
  if (r >= rad0) {
    Real dr = r-rad0;
    // assume cartesian for now
    pphot->x0p[ip] -= dr/2.99792458e10;
    pphot->x1p[ip] -= pphot->k1p[ip]*dr;
    pphot->x2p[ip] -= pphot->k2p[ip]*dr;
    pphot->x3p[ip] -= pphot->k3p[ip]*dr;

    pphot->statp[ip] = ESCAPED;
    //pphot->face = BoundaryFace::undef;
  }
  // Then check time condition -- ensures time is not over estimated
  if (pphot->x0p[ip] >= time0) {
    Real dt = pphot->x0p[ip] - time0;
    pphot->x0p[ip] -= dt;
    pphot->x1p[ip] -= pphot->k1p[ip]*dt*2.99792458e10;
    pphot->x2p[ip] -= pphot->k2p[ip]*dt*2.99792458e10;
    pphot->x3p[ip] -= pphot->k3p[ip]*dt*2.99792458e10;

    pphot->statp[ip] = ESCAPED;
    //pphot->face = BoundaryFace::undef;
  }
}

} //namespace
