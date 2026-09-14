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

  void FinalPositionSphericalPolar(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot,
                                   int ip);

  // Global variables
  static Real error_sum = 0.;
  static int nerror = 0;

} // namespace

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief monte carlo test problem generator
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  Real rideal = 8.314e7;
  Real rho = 1.;
  Real temp = 1.;
  Real gamma = peos->GetGamma();

  // Set nominal values for grid, unused
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        phydro->u(IDN,k,j,i) = rho;
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        phydro->u(IEN,k,j,i) = rideal*rho*temp/(gamma-1.0);
      }
    }
  }
}

//========================================================================================
//! \fn void MonteCarlo::InitUserMonteCarloData(ParameterInput *pin)
//! \brief Initializes user data specific to MonteCarlo class
//========================================================================================

void MonteCarlo::InitUserMonteCarloData(ParameterInput *pin) {
  nuser_var = 4;
}

//========================================================================================
//! \fn void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype)
//! \brief Initializes Photon packets before integration
//========================================================================================

void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype) {

  // Get meshblock dimensions
  Real nx1 = static_cast<Real>(ie-is+1);
  Real nx2 = static_cast<Real>(je-js+1);
  Real nx3 = static_cast<Real>(ke-ks+1);

  for (int ip=ips; ip<=ipe; ip++) {

    // Randomly assign emission zone
    int i1,i2,i3;
    pphot->i1p[ip] = i1 = static_cast<int>(pran->uniform()*nx1)+is;
    pphot->i2p[ip] = i2 = static_cast<int>(pran->uniform()*nx2)+js;
    pphot->i3p[ip] = i3 = static_cast<int>(pran->uniform()*nx3)+ks;

    pphot->wp[ip] = 1.0;

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
      pphot->sup[ip] = 0.0;
      pphot->sqp[ip] = 0.0;
    }

    // Initialize energy
    pphot->ep[ip] = 1.;

    pphot->dtp[ip] = HUGE_NUMBER;
    // Set status flag
    if (pphot->wp[ip] < 0.0)
      pphot->statp[ip] = DESTROYED;
    else
      pphot->statp[ip] = EVOLVING;

    // Initialize the absorption and scattering extinction coefficients
    pphot->acp[ip] = 0.0;
    pphot->scp[ip] = 0.0;

    // Compute predicted final photon positions for comparison
    FinalPositionSphericalPolar(this,pcoord,pphot,ip);

  } // loop over ip
}

//========================================================================================
//! \fn void MonteCarloBlock::FinalizePhoton(Photon *pphot, ip)
//! \brief Complete work at end of photon packets before integration
//========================================================================================

void MonteCarloBlock::FinalizePhoton(Photon *pphot, int ip) {

  if (pphot->statp[ip] == ESCAPED) {
    Real rf = pphot->user[1][ip];
    Real thf = pphot->user[2][ip];
    Real phf = pphot->user[3][ip];
    Real xp = rf * sin(thf) * cos(phf);
    Real yp = rf * sin(thf) * sin(phf);
    Real zp = rf * cos(thf);
    Real xf = pphot->x1p[ip] * sin(pphot->x2p[ip]) * cos(pphot->x3p[ip]);
    Real yf = pphot->x1p[ip] * sin(pphot->x2p[ip]) * sin(pphot->x3p[ip]);
    Real zf = pphot->x1p[ip] * cos(pphot->x2p[ip]);

    // Save as user variable for photon list
    Real error = sqrt(SQR(xf-xp)+SQR(yf-yp)+SQR(zf-zp))/rf;
    //printf("final: %d %g %g %g %g %g %g\n",ip,xf,xp,yf,yp,zf,zp);
    //pphot->PrintPhoton(ip);
    nerror++;
    error_sum += error;
    pphot->user[0][ip] = error;
    //printf("Mean relative error: %g\n",error_sum/static_cast<Real>(nerror));
  }

}


namespace {
void FinalPositionSphericalPolar(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot,
                                 int ip) {

  Real r = pphot->x1p[ip];
  Real cth = cos(pphot->x2p[ip]);
  Real sth = sin(pphot->x2p[ip]);
  Real cph = cos(pphot->x3p[ip]);
  Real sph = sin(pphot->x3p[ip]);

  Real kr, kth, kph;
  if (pmcb->pmy_mc->general_pusher_flag) {
    kr = pphot->k1p[ip];
    kth = r * pphot->k2p[ip];
    kph = r * sth * pphot->k3p[ip];
  } else {
    kr = pphot->k1p[ip];
    kth = pphot->k2p[ip];
    kph = pphot->k3p[ip];
  }
  // Convert to cartesian
  Real kx = kr * sth * cph + kth * cth*cph - kph * sph;
  Real ky = kr * sth * sph + kth * cth*sph + kph * cph;
  Real kz = kr * cth - kth * sth;

  // Outer boundary is r = rf -- find dlr to this boundary
  Real rf = pco->x1f(pmcb->ie+1);
  Real ndr0 = pphot->x1p[ip] * (sth * (kx * cph + ky * sph) + kz * cth);
  Real det = 1.0 + (SQR(rf) - SQR(pphot->x1p[ip])) / SQR(ndr0);
  Real dlr1 = ndr0 * (sqrt(det) - 1.0);
  Real dlr2 = -ndr0 * (sqrt(det) + 1.0);

  Real dl;
  if (dlr1 > 0.0) {
    if (dlr2 > 0.0) {
      std::cout << "Warning: both roots positive in FinalPositionSphericalPolar: "
                << dlr1 << " " << dlr2 << std::endl;
    } else {
      dl = dlr1;
    }
  } else if (dlr2 > 0.0) {
    dl = dlr2;
  }

  // Compute other boundary positions
  //theta
  Real zf = r * cth + kz * dl;
  Real thf = acos(zf / rf);

  //phi
  Real xf = r * sth * cph + kx * dl;
  Real yf = r * sth * sph + ky * dl;
  Real phf = atan2(yf,xf);
  if (phf < 0.0)
    phf += 2.*PI;

  pphot->user[1][ip] = rf;
  pphot->user[2][ip] = thf;
  pphot->user[3][ip] = phf;

}

} // namespace
