//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mc_sphere_isoth.cpp
//! \brief Monte Carlo transport through a uniform isothermal sphere.  The escape
//! surface is the sphere of radius <problem> radius, applied in UserWorkInMove; the
//! list's user columns are the birth energy, the scattering count at escape and the
//! unweighted path length, which the sphere_compton tools bin against the Kompaneets
//! Green's function.
//
//========================================================================================

// C++ headers
#include <algorithm>
#include <cmath>
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
  Real tau_rho_; // the sphere's density

  // function headers
  void SphericalEscape(MonteCarloBlock *pmcb, Photon *phot, PhotonPusher *ppusher, int ip);
  void TimedEscape(MonteCarloBlock *pmcb, Photon *phot, PhotonPusher *ppusher, int ip);
}


//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief monte carlo test problem generator
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  // Gas constant of the default hydrogen-helium mixture, so the temperature the
  // Monte Carlo inverts from this pressure is the one written here.
  Real heabund = pin->GetOrAddReal("problem","heabund",0.09);
  Real rideal = MonteCarloBlock::GasConstant(heabund);
  Real c = 2.99792458e10;
  Real temp = pin->GetReal("problem","temp");
  Real tau = pin->GetReal("problem","tau");
  Real rad0 = pin->GetReal("problem","radius");
  Real vel = pin->GetOrAddReal("problem","velocity",0.);
  Real gamma = peos->GetGamma();
  vel *= c;

  Real mp = 1.6726e-24;
  Real sigmat = 6.65248e-25;
  Real kappaes = sigmat * (1. + 2.*heabund) / (mp * (1.+4.*heabund) );

  Real rho = tau / (kappaes * rad0);
  // Cells whose centre lies outside the sphere get a density floor (a single cell
  // holding the whole sphere is inside); the escape surface at rad0 keeps photons
  // out of the floor region anyway
  Real floor_frac = pin->GetOrAddReal("problem","rho_floor_frac",1.e-10);
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real r = std::sqrt(SQR(pcoord->x1v(i)) + SQR(pcoord->x2v(j)) + SQR(pcoord->x3v(k)));
        Real rhoc = (r < rad0) ? rho : rho*floor_frac;
        phydro->u(IDN,k,j,i) = rhoc;
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = rhoc*vel;
        phydro->u(IEN,k,j,i) = rideal*rhoc*temp/(gamma-1.0);
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
  // With <problem> time set (a path length, c = 1) the photons stop there; otherwise
  // at the sphere
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
  {
    Real heabund = pin->GetOrAddReal("problem","heabund",0.09);
    Real mp = 1.6726e-24;
    Real sigmat = 6.65248e-25;
    Real kappaes = sigmat * (1. + 2.*heabund) / (mp * (1.+4.*heabund) );
    tau_rho_ = pin->GetReal("problem","tau") / (kappaes * rad0);
  }

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
  const bool has_origin = (i1start >= 0) && (i2start >= 0) && (i3start >= 0);

  // With no emission function the block has to be given its photon count; the point
  // source goes to the block holding the origin.  Free-free emission uses the standard
  // per-cell emission array, which is already zero (to the floor) outside the sphere.
  if (pmy_mc->emission_flag == EMISNONE)
    nphremain = has_origin ? pin->GetInteger64("montecarlo","nphot") : 0;

  // The random-walk step is for cells wholly inside the sphere: a cell the escape surface
  // cuts is not the uniform medium the step assumes
  if (accel_mask.GetSize() > 0) {
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          Real rmax = 0.;
          for (int c=0; c<8; c++) {
            Real x = (c & 1) ? pcoord->x1f(i+1) : pcoord->x1f(i);
            Real y = (c & 2) ? pcoord->x2f(j+1) : pcoord->x2f(j);
            Real z = (c & 4) ? pcoord->x3f(k+1) : pcoord->x3f(k);
            rmax = std::max(rmax, std::sqrt(x*x + y*y + z*z));
          }
          accel_mask(k,j,i) = (rmax < rad0) ? 1 : 0;
        }
      }
    }
  }

  // Report the volume actually laid down inside the sphere, since the cells are in or
  // out by their centres
  Real vin = 0.;
  for (int k=ks; k<=ke; k++)
    for (int j=js; j<=je; j++)
      for (int i=is; i<=ie; i++)
        if (rho(k,j,i) > 0.5*tau_rho_) vin += pcoord->vol(k,j,i);
  if (pmy_mc->verbose && vin > 0.)
    std::cout << "sphere block " << pmy_block->gid << ": volume inside / cell volume = "
              << vin/(4./3.*PI*rad0*rad0*rad0) << (has_origin ? " (holds the origin)" : "")
              << std::endl;

}

//========================================================================================
//! \fn void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype)
//! \brief Initializes Photon packets before integration
//========================================================================================

void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype) {

  // Free-free photons are spread over the cells in proportion to their emission
  if (pmy_mc->emission_flag == EMISFF)
    SetEmissionCellWeight(pphot,ips,ipe);

  for (int ip=ips; ip<=ipe; ip++) {

    // user[0] birth energy (set below), user[1] scattering count at escape,
    // user[2] unweighted path length
    pphot->user[1][ip] = 0.;
    pphot->user[2][ip] = 0.;

    // Set status flag
    pphot->statp[ip] = EVOLVING;

    // Initialize Photon weights, energy, direction, polarization
    if (pmy_mc->emission_flag == EMISNONE) {
      // the point source: this block holds the origin
      pphot->i1p[ip] = i1start;
      pphot->i2p[ip] = i2start;
      pphot->i3p[ip] = i3start;
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
      // Cell and weight were set above; position uniform in the cell, energy from the
      // free-free emission function
      GetZonePosition(pphot,pran,pcoord,ip);
      pphot->x0p[ip] = 0.; //time
      if(tnorm) {
        Real logtg = log(tgas(pphot->i3p[ip],pphot->i2p[ip],pphot->i1p[ip]));
        PhotonEmitFreeFree(this,pphot,logemin+logtg,logemax+logtg,ip);
      } else{
        PhotonEmitFreeFree(this,pphot,logemin,logemax,ip);
      }
    }

    // Set status flag and the time budget; with no budget the pusher takes no step and
    // the block loop scatters the photon in place forever
    pphot->dtp[ip] = pmy_mc->tmax;
    if (pphot->wp[ip] < 0.0)
      pphot->statp[ip] = DESTROYED;
    else
      pphot->statp[ip] = EVOLVING;


    pphot->user[0][ip] = pphot->ep[ip];
    pphot->nscp[ip] = 0;

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

  pphot->user[2][ip] += ppusher->dl;

  // First check radius condition
  Real r = sqrt(SQR(pphot->x1p[ip])+SQR(pphot->x2p[ip])+SQR(pphot->x3p[ip]));
  if (r >= rad0) {
    // Back the photon up to the sphere; the Cartesian pusher advances x0p by the path
    // length, so the correction is a length too
    Real dr = r-rad0;
    pphot->x0p[ip] -= dr;
    pphot->user[2][ip] -= dr;
    pphot->x1p[ip] -= pphot->k1p[ip]*dr;
    pphot->x2p[ip] -= pphot->k2p[ip]*dr;
    pphot->x3p[ip] -= pphot->k3p[ip]*dr;

    pphot->user[1][ip] = pphot->nscp[ip];
    pphot->statp[ip] = ESCAPED;
  }

}

// Used to test photons radial distributions after a fixed travel time
void TimedEscape(MonteCarloBlock *pmcb, Photon *pphot, PhotonPusher *ppusher,
                 int ip) {

  // First check radius condition
  Real r = sqrt(SQR(pphot->x1p[ip])+SQR(pphot->x2p[ip])+SQR(pphot->x3p[ip]));
  if (r >= rad0) {
    Real dr = r-rad0;
    pphot->x0p[ip] -= dr;
    pphot->x1p[ip] -= pphot->k1p[ip]*dr;
    pphot->x2p[ip] -= pphot->k2p[ip]*dr;
    pphot->x3p[ip] -= pphot->k3p[ip]*dr;

    pphot->user[1][ip] = pphot->nscp[ip];
    pphot->statp[ip] = ESCAPED;
  }
  // Then check the path condition (x0p is the path length) so the photon is not
  // carried past time0
  if (pphot->x0p[ip] >= time0) {
    Real dt = pphot->x0p[ip] - time0;
    pphot->x0p[ip] -= dt;
    pphot->x1p[ip] -= pphot->k1p[ip]*dt;
    pphot->x2p[ip] -= pphot->k2p[ip]*dt;
    pphot->x3p[ip] -= pphot->k3p[ip]*dt;

    pphot->user[1][ip] = pphot->nscp[ip];
    pphot->statp[ip] = ESCAPED;
  }
}

} //namespace
