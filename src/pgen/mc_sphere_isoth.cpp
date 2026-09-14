//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mctest.cpp
//! \brief Problem generator for  monte carlo through uniform sphere
//
//========================================================================================

// C++ headers
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
  Real energy0,tsource;
  bool srcdist,tnorm,planckdist;
  int i1start,i2start,i3start;
  Real logemin, logemax;

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
  Real rad0 = pin->GetReal("problem","radius");
  Real vel = pin->GetOrAddReal("problem","velocity",0.);
  Real gamma = peos->GetGamma();
  vel *= c;

  Real heabund = 0.09; //hardcode for now (should be parameter)
  //Real heabund = 0.;
  Real mp = 1.6726e-24;
  Real sigmat = 6.65248e-25;
  Real kappaes = sigmat * (1. + 2.*heabund) / (mp * (1.+4.*heabund) );

  Real rho = tau / (kappaes * rad0);
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
      }
    }
  }
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
  } else
    EnrollUserWorkInMove(SphericalEscape);

}

//========================================================================================
//! \fn void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin)
//! \brief Analogous to problem generator but used in support of InitializePhoton
//========================================================================================

void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin) {

  // Set variables
  srcdist =pin->GetOrAddBoolean("problem","srcdist",false);
  rad0 = pin->GetReal("problem","radius");
  time0 = pin->GetOrAddReal("problem","time",-1.);

  if (pmy_mc->emission_flag == EMISNONE) {
    planckdist = pin->GetOrAddBoolean("problem","planckdist",false);
    if (planckdist) {
      tsource = pin->GetReal("problem","tsource");
    } else {
      Real x0 = pin->GetReal("problem","x0");
      Real temp = pin->GetReal("problem","temp");
      Real kb = 1.380649e-16;
      energy0 = kb*temp*x0;
    }
  } else if (pmy_mc->emission_flag == EMISFF) {
    // Set the energy boundaries for free-free emission
    tnorm = pin->GetOrAddBoolean("problem","tnorm",false);
    if (tnorm) {
      // interpret as xmin/xmax with x=E/(kb*T)
      Real kb = 1.380649e-16;
      logemin = log(kb*pin->GetReal("problem", "emin"));
      logemax = log(kb*pin->GetReal("problem", "emax"));
    } else {
      // interpret as emin/emax in eV
      Real everg = 1.6021772e-12;
      logemin = log(everg*pin->GetReal("problem", "emin"));
      logemax = log(everg*pin->GetReal("problem", "emax"));
    }
  }

  // Deterime cell of initial photon, which is asssumed to include
  // the origin if more than one cell is specified for each direction
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
    msg << "### FATAL ERROR in MonteCarloProblemGenerator" << std::endl
        << "Origin not found within domain." << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  if (pmy_mc->emission_flag == EMISFF) {
    // Adjust for smaller emission volume
    Real cellvol = pcoord->vol(i3start,i2start,i1start);
    Real spherevol = 4./3.*PI*pow(rad0,3);
    emission(i3start,i2start,i1start) *= (spherevol/cellvol);
    pcoord->vol(i3start,i2start,i1start) *= (spherevol/cellvol);
    minweight *= (spherevol/cellvol);
  }

}

//========================================================================================
//! \fn void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype)
//! \brief Initializes Photon packets before integration
//========================================================================================

void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype) {

  for (int ip=ips; ip<=ipe; ip++) {

    pphot->user[0][ip] = 0.;
    pphot->user[1][ip] = 0.;
    pphot->user[2][ip] = 0.;

    // Set status flag
    pphot->statp[ip] = EVOLVING;

    // Initialize cell number
    int i1,i2,i3;
    pphot->i1p[ip] = i1 = i1start;
    pphot->i2p[ip] = i2 = i2start;
    pphot->i3p[ip] = i3 = i3start;

    // Initialize Photon weights, energy, direction, polarization
    if (pmy_mc->emission_flag == EMISNONE) {
      pphot->wp[ip] = 1.0;
      if (planckdist)
        pphot->ep[ip] = PlanckDist(tsource,pran);
      else
        pphot->ep[ip] = energy0;

      // Initialize Stokes vector
      if (IsPolarized(pphot->polarized)) {
        pphot->sip[ip] = 1.0;
        pphot->sqp[ip] = 0.0;
        pphot->sup[ip] = 0.0;
        pphot->svp[ip] = 0.0;
      }
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

      // Get initial position of photon
      if (srcdist) {
        // Model an exponential weight time distribuion
        Real x = pran->uniform();
        while (x <= 0.)
          x = pran->uniform();
        Real dev = pran->uniform();
        while (sin(PI*x)*x < 0.57923*dev) {
          // Yields nearly exponential escape time distribution
          //while (sin(PI*x)/(x*PI) < dev) {
          x = pran->uniform();
          while (x <= 0.)
            x = pran->uniform();
          dev = pran->uniform();
        }

        Real r0 = x*rad0;
        Real phi = 2. * PI * pran->uniform();
        Real cphi = cos(phi);
        Real sphi = sin(phi);

        Real cth = 2. * pran->uniform() - 1.;
        Real sth = sqrt(1. - SQR(cth));

        pphot->x1p[ip] = r0*sth*cphi;
        pphot->x2p[ip] = r0*sth*sphi;
        pphot->x3p[ip] = r0*cth;
        pphot->x0p[ip] = 0.; //time
      } else {
        // Initialize photon at the origin
        pphot->x1p[ip] = 0.;
        pphot->x2p[ip] = 0.;
        pphot->x3p[ip] = 0.;
        pphot->x0p[ip] = 0.; //time
      }

    } else if (pmy_mc->emission_flag == EMISFF) {
      // Set weight according to the emission array, which is the relative number
      // of photons per unit time emitted in each cell
      pphot->wp[ip] = emission(i3,i2,i1);

      // Obtain intitial energy, polarization, direction and weight
      // Utilize free-free emission function in emission.cpp
      if(tnorm) {
        Real logtg = log(tgas(i3,i2,i1));
        PhotonEmitFreeFree(this,pphot,logemin+logtg,logemax+logtg,ip);
      } else{
        PhotonEmitFreeFree(this,pphot,logemin,logemax,ip);
      }

      Real r0 = pow(pran->uniform()*rad0*rad0*rad0,1./3.);
      Real phi = 2. * PI * pran->uniform();
      Real cphi = cos(phi);
      Real sphi = sin(phi);
      Real cth = 2. * pran->uniform() - 1.;
      Real sth = sqrt(1. - SQR(cth));
      pphot->x1p[ip] = r0*sth*cphi;
      pphot->x2p[ip] = r0*sth*sphi;
      pphot->x3p[ip] = r0*cth;
      pphot->x0p[ip] = 0.; //time
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
