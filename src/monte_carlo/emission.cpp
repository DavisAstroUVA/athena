//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//!  \file emission.cpp
//!  \brief implementation of photon emission functions

// Athena++ headers
#include "montecarlo.hpp"
#include "../defs.hpp"
#include "../mesh/mesh.hpp"
#include "../coordinates/coordinates.hpp"
#include "../hydro/hydro.hpp"
#include "../globals.hpp"

//----------------------------------------------------------------------------------------
//! \fn Real GetEmissionFreefree(MonteCarloBlock *pmcb, int k, int j, int i, int etype)
//! \brief compute free-free emissivity

Real GetEmissionFreeFree(MonteCarloBlock *pmcb, int k, int j, int i, int etype) {

  const Real eta0 = 1.032521e-11;
  const Real gaunt = 1.0; // Gaunt factor

  Real nel = pmcb->species(0,k,j,i);
  Real nion = pmcb->species(1,k,j,i);
  Real temp = pmcb->tgas(k,j,i);

  return eta0 / sqrt(temp) * nel * nion * gaunt;

}

//----------------------------------------------------------------------------------------
//! \fn void PhotonInitFreeFree(MonteCarloBlock *pmcb, Photon *pphot, Real lemin,
//!                             Real lemax, int ip))
//! \brief initialize photon consistent with free-free emission

void PhotonEmitFreeFree(MonteCarloBlock *pmcb, Photon *pphot, Real lemin, Real lemax,
                        int ip)
{
  Real kb_cgs = 1.380649e-16;
  MCRandom *pran = pmcb->pran;

  // Scheme in which packets are drawn from a uniform distribution in log E
  // requires weight = exp(-x) note log(10)=2.30258509299
  Real dev = exp((lemax-lemin)*pran->uniform()+lemin);
  pphot->ep[ip] = dev;
  Real x = dev / (kb_cgs * pmcb->tgas(pphot->i3p[ip],pphot->i2p[ip],pphot->i1p[ip]));

  // Initialize weight
  pphot->wp[ip] *= exp(-x) * (lemax-lemin);

  if (IsPolarized(pmcb->pmy_mc->polarized)) {
    // Initialize Stokes vector
    pphot->sip[ip] = 1.0;
    pphot->sup[ip] = 0.0;
    pphot->sqp[ip] = 0.0;
    pphot->svp[ip] = 0.0;
  }

  // Generate initial angle parameters
  Real phi = 2. * PI * pran->uniform();
  Real cphi = cos(phi);
  Real sphi = sin(phi);
  Real cth = 2. * pran->uniform() - 1.;
  Real sth = std::sqrt(1. - SQR(cth));

  // Initialize wave vector with isotropic distribution.  k0p carries the photon
  // energy, which the emissivity sampler has already placed in ep.
  pphot->k0p[ip] = pphot->ep[ip];
  pphot->k1p[ip] = sth*cphi;
  pphot->k2p[ip] = sth*sphi;
  pphot->k3p[ip] = cth;

}

//----------------------------------------------------------------------------------------
//! \fn Real GetEmissionBlackbody(MonteCarloBlock *pmcb, int k, int j, int i, int etype)
//! \brief compute blackbody emission

Real GetEmissionBlackbody(MonteCarloBlock *pmcb, int k, int j, int i, int etype) {

  Real temp = pmcb->tgas(k,j,i);
  Real c_cgs = 2.99792458e10;
  Real h_cgs = 6.62607015e-27;
  Real kb_cgs = 1.380649e-16;  
  return 4.*PI*1.20206/SQR(c_cgs)*pow(kb_cgs*temp/h_cgs,3); // zeta(3)

}

//----------------------------------------------------------------------------------------
//! \fn void PhotonInitBlackbody(MonteCarloBlock *pmcb, Photon *pphot, BoundaryFace face,
//!       int ip))
//!
//! \brief initialize photon consistent with black body emission

void PhotonEmitBlackbody(MonteCarloBlock *pmcb, Photon *pphot, BoundaryFace face,
                         int ip) {

  // Initialize energy
  MCRandom *pran = pmcb->pran;
  Real temp = pmcb->tgas(pphot->i3p[ip],pphot->i2p[ip],pphot->i1p[ip]);
  pphot->ep[ip] = PlanckDist(temp,pran);

  if (IsPolarized(pmcb->pmy_mc->polarized)) {
    // Initialize Stokes vector
    pphot->sip[ip] = 1.0;
    pphot->sup[ip] = 0.0;
    pphot->sqp[ip] = 0.0;
    pphot->svp[ip] = 0.0;
  }

  // Generate initial angle parameters
  Real phi = 2. * PI * pran->uniform();
  Real cphi = cos(phi);
  Real sphi = sin(phi);
  // isotropic from surface
  Real cth = std::sqrt(pran->uniform()); 
  Real sth = std::sqrt(1. - SQR(cth));

  // Align to appropriate face, assuming emmision
  // points into domain
  Real kx,ky,kz;
  // k0p carries the photon energy, already sampled into ep above.
  pphot->k0p[ip] = pphot->ep[ip];
  switch(face) {
    case BoundaryFace::inner_x1:
      pphot->k1p[ip] = cth;
      pphot->k2p[ip] = sth*cphi;
      pphot->k3p[ip] = sth*sphi;
      break;
    case BoundaryFace::outer_x1:
      pphot->k1p[ip] = -cth;
      pphot->k2p[ip] = sth*cphi;
      pphot->k3p[ip] = sth*sphi;
      break;
    case BoundaryFace::inner_x2:
      pphot->k1p[ip] = sth*sphi;
      pphot->k2p[ip] = cth;
      pphot->k3p[ip] = sth*cphi;
      break;
    case BoundaryFace::outer_x2:
      pphot->k1p[ip] = sth*sphi;
      pphot->k2p[ip] = -cth;
      pphot->k3p[ip] = sth*cphi;
      break;
    case BoundaryFace::inner_x3:
      pphot->k1p[ip] = sth*cphi;
      pphot->k2p[ip] = sth*sphi;
      pphot->k3p[ip] = cth;
      break;
    case BoundaryFace::outer_x3:
      pphot->k1p[ip] = sth*cphi;
      pphot->k2p[ip] = sth*sphi;
      pphot->k3p[ip] = -cth;
      break;
  }

}

//----------------------------------------------------------------------------------------
//! \fn Real PlanckDist(Real temp, MCRandom *pran)
//! \brief returns energy distributed according to Planck function

Real PlanckDist(Real temp, MCRandom *pran)
{
  // Method of choosing the energy of the initial photon which a Planck spectrum
  // distribution. See Pozdnyakov et al. sec 9.4.  Originally, Fleck and Cumming (1971)

  Real x1 = 1.20206 * pran->uniform(); // zeta(3)
  Real x2 = pran->uniform();
  Real x3 = pran->uniform();
  Real x4 = pran->uniform();

  Real sum = 1.0;
  int alpha = 1;
  while (x1 >= sum) {
    alpha++;
    sum +=  1. / (alpha * alpha * alpha);
  }

  Real kb_cgs = 1.380649e-16;
  return -kb_cgs * temp * log(x2 * x3 * x4) / static_cast<Real>(alpha);

}

//----------------------------------------------------------------------------------------
//! \fn void GetZonePositionCartesian(Photon *pphot, MCRandom *pran, MCCoord *pcoord,
//!                                   int ip)
//! \brief choose random position within cartesian cell

void GetZonePositionCartesian(Photon *pphot, MCRandom *pran, MCCoord *pcoord, int ip) {

  Real xl = pcoord->x1f(pphot->i1p[ip]); Real dx = pcoord->x1f(pphot->i1p[ip]+1)-xl;
  Real yl = pcoord->x2f(pphot->i2p[ip]); Real dy = pcoord->x2f(pphot->i2p[ip]+1)-yl;
  Real zl = pcoord->x3f(pphot->i3p[ip]); Real dz = pcoord->x3f(pphot->i3p[ip]+1)-zl;

  pphot->x1p[ip] = xl+pran->uniform()*dx;
  pphot->x2p[ip] = yl+pran->uniform()*dy;
  pphot->x3p[ip] = zl+pran->uniform()*dz;

}

//----------------------------------------------------------------------------------------
//! \fn void GetZonePositionCylindrical(Photon *pphot, MCRandom *pran, MCCoord *pcoord,
//!                                     int ip)
//! \brief choose random position within cylindrical cell

void GetZonePositionCylindrical(Photon *pphot, MCRandom *pran, MCCoord *pcoord,
                                   int ip) {
  Real rl = pcoord->x1f(pphot->i1p[ip]);
  Real rh = pcoord->x1f(pphot->i1p[ip]+1);
  pphot->x1p[ip] = pow(pran->uniform()*(rh*rh-rl*rl)+rl*rl,0.5);

  Real pl = pcoord->x2f(pphot->i2p[ip]);
  Real dp = pcoord->x2f(pphot->i2p[ip]+1)-pl;
  pphot->x2p[ip] = pl+pran->uniform()*dp;

  Real zl = pcoord->x3f(pphot->i3p[ip]);
  Real dz = pcoord->x3f(pphot->i3p[ip]+1)-zl;
  pphot->x3p[ip] = zl+pran->uniform()*dz;
}

//----------------------------------------------------------------------------------------
//! \fn void GetZonePositionSphericalPolar(Photon *pphot, MCRandom *pran, MCCoord *pcoord,
//!                                        int ip)
//! \brief choose random position within spherical-polar cell

void GetZonePositionSphericalPolar(Photon *pphot, MCRandom *pran, MCCoord *pcoord,
                                   int ip) {
  Real rl = pcoord->x1f(pphot->i1p[ip]), rh = pcoord->x1f(pphot->i1p[ip]+1);
  pphot->x1p[ip]  = pow(pran->uniform()*(rh*rh*rh-rl*rl*rl)+rl*rl*rl,1./3.);
  Real cthh = cos(pcoord->x2f(pphot->i2p[ip]));
  Real cthl = cos(pcoord->x2f(pphot->i2p[ip]+1));
  Real cth = cthl + pran->uniform() * (cthh-cthl);
  pphot->x2p[ip] = acos(cth);
  Real pl = pcoord->x3f(pphot->i3p[ip]); Real dp = pcoord->x3f(pphot->i3p[ip]+1)-pl;
  pphot->x3p[ip] = pl+pran->uniform()*dp;

}

//----------------------------------------------------------------------------------------
//! \fn void GetZonePositionCartesianFace(Photon *pphot, MCRandom *pran, MCCoord *pcoord,
//!                                   BoundaryFace face, int ip)
//! \brief choose random position within cartesian cell

void GetZonePositionCartesianFace(Photon *pphot, MCRandom *pran, MCCoord *pcoord,
                                  BoundaryFace face, int ip) {

  Real xl = pcoord->x1f(pphot->i1p[ip]); Real dx = pcoord->x1f(pphot->i1p[ip]+1)-xl;
  Real yl = pcoord->x2f(pphot->i2p[ip]); Real dy = pcoord->x2f(pphot->i2p[ip]+1)-yl;
  Real zl = pcoord->x3f(pphot->i3p[ip]); Real dz = pcoord->x3f(pphot->i3p[ip]+1)-zl;

  switch(face) {
    case BoundaryFace::inner_x1:
      pphot->x1p[ip] = xl;
      pphot->x2p[ip] = yl+pran->uniform()*dy;
      pphot->x3p[ip] = zl+pran->uniform()*dz;
      break;
    case BoundaryFace::outer_x1:
      pphot->x1p[ip] = xl+dx;
      pphot->x2p[ip] = yl+pran->uniform()*dy;
      pphot->x3p[ip] = zl+pran->uniform()*dz;
      break;
    case BoundaryFace::inner_x2:
      pphot->x1p[ip] = xl+pran->uniform()*dx;
      pphot->x2p[ip] = yl;
      pphot->x3p[ip] = zl+pran->uniform()*dz;
      break;
    case BoundaryFace::outer_x2:
      pphot->x1p[ip] = xl+pran->uniform()*dx;
      pphot->x2p[ip] = yl+dy;
      pphot->x3p[ip] = zl+pran->uniform()*dz;
      break;
    case BoundaryFace::inner_x3:
      pphot->x1p[ip] = xl+pran->uniform()*dx;
      pphot->x2p[ip] = yl+pran->uniform()*dy;
      pphot->x3p[ip] = zl;
      break;
    case BoundaryFace::outer_x3:
      pphot->x1p[ip] = xl+pran->uniform()*dx;
      pphot->x2p[ip] = yl+pran->uniform()*dy;
      pphot->x3p[ip] = zl+dz;
      break;
  }

}