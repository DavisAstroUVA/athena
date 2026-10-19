//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//!  \file emission.cpp
//!  \brief implementation of photon emission functions

// C++ headers
#include <algorithm>  // min, max

// Athena++ headers
#include "montecarlo.hpp"
#include "mcutils.hpp"  // mcbisect
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

  // The energy is drawn in ln E and the weight carries e^{-x} times the ln E measure.
  // With an escape table (weights = biased, bias_energy) group l is drawn in proportion
  // to its overlap d_l with [lemin, lemax] times m_l = (1 - xi) + xi p_l/<p>, where p_l
  // is the cell's escape probability in the group, <p> its d-weighted mean and
  // xi = bias_energy_mix, and the measure is divided by m_l.  Samples concentrate at
  // energies that can leave the gas while the expected weight is unchanged, and no
  // sample carries more than 1/(1 - xi) times the flat measure.
  Real y, measure = lemax - lemin;
  const int ng = pmcb->pmy_mc->nescape;
  bool drawn = false;
  if (ng > 0 && pmcb->pmy_mc->bias_energy && pmcb->escape_prob.GetSize() > 0) {
    const AthenaArray<Real> &lne = pmcb->pmy_mc->escape_lne;
    const Real xi = pmcb->pmy_mc->bias_energy_mix;
    const int kt = pphot->i3p[ip]-pmcb->ks, jt = pphot->i2p[ip]-pmcb->js,
              it = pphot->i1p[ip]-pmcb->is;
    Real dsum = 0., psum = 0.;
    for (int l=0; l<ng; ++l) {
      Real d = std::min(lne(l+1),lemax) - std::max(lne(l),lemin);
      if (d > 0.) {
        dsum += d;
        psum += d * pmcb->escape_prob(kt,jt,it,l);
      }
    }
    if (dsum > 0. && psum > 0.) {
      // sum over l of d_l m_l is dsum, so u is drawn on [0, dsum)
      const Real pmean = psum / dsum;
      Real u = dsum * pran->uniform(), cum = 0., lo = lemin, hi = lemax, q = dsum, m = 1.;
      for (int l=0; l<ng; ++l) {
        Real glo = std::max(lne(l),lemin), ghi = std::min(lne(l+1),lemax);
        if (ghi <= glo) continue;
        m = (1. - xi) + xi * pmcb->escape_prob(kt,jt,it,l) / pmean;
        q = (ghi-glo) * m; lo = glo; hi = ghi;
        if (u < cum + q) break;
        cum += q;
      }
      Real frac = std::min(std::max((u-cum)/q, 0.), 1.);
      y = lo + (hi-lo)*frac;
      measure = dsum / m;
      drawn = true;
    }
  }
  if (!drawn) y = lemin + measure*pran->uniform();
  Real dev = exp(y);
  pphot->ep[ip] = dev;
  Real x = dev / (kb_cgs * pmcb->tgas(pphot->i3p[ip],pphot->i2p[ip],pphot->i1p[ip]));

  // Initialize weight
  pphot->wp[ip] *= exp(-x) * measure;

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
//----------------------------------------------------------------------------------------
//! \fn Real GetEmissionTable(MonteCarloBlock *pmcb, int k, int j, int i, int etype)
//! \brief the cell's tabulated emission [photons/cm^3/s], from BuildOpacityTables

Real GetEmissionTable(MonteCarloBlock *pmcb, int k, int j, int i, int etype) {
  return pmcb->tab_emis_tot(k-pmcb->ks, j-pmcb->js, i-pmcb->is);
}

//----------------------------------------------------------------------------------------
//! \fn void PhotonEmitTable(MonteCarloBlock *pmcb, Photon *pphot, int ip)
//! \brief energy from the cell's tabulated emission spectrum, isotropic direction
//
// The group is drawn from the cell's cumulative emission (one uniform) and the energy
// within it by OpacityTable::DrawEnergy, then the direction (two more).  With an escape
// table (weights = biased, bias_energy) the group is drawn in proportion to its photon
// share s_l times m_l = (1 - xi) + xi p_l/<p>, with p_l the cell's escape probability in
// the group, <p> the share-weighted mean and xi = bias_energy_mix, and the weight is
// divided by m_l; the shares sum to one, so the m_l-weighted shares do too, and the
// weight factor is bounded at 1/(1 - xi); see PhotonEmitFreeFree.  That path takes one
// escape group per table group, which the problem generator sets up.

void PhotonEmitTable(MonteCarloBlock *pmcb, Photon *pphot, int ip) {
  MCRandom *pran = pmcb->pran;
  const OpacityTable *tab = pmcb->pmy_mc->optab;
  const int ng = tab->ngroups;
  const int i1 = pphot->i1p[ip], i2 = pphot->i2p[ip], i3 = pphot->i3p[ip];
  const Real temp = pmcb->tgas(i3,i2,i1);

  Real dev = pran->uniform();
  const Real *prob;
  const Real tot = pmcb->TableEmissionCumulative(i3, i2, i1, prob);
  Real energy;
  bool drawn = false;
  if (tot <= 0.) {
    // A non-emitting cell can still be drawn, since cells are picked uniformly and only
    // then weighted.  There is no spectrum to sample and the photon carries zero weight,
    // so take the lowest tabulated energy rather than divide by a zero bin width.
    energy = tab->gedge(0);
    drawn = true;
  } else if (pmcb->pmy_mc->nescape > 0 && pmcb->pmy_mc->bias_energy
             && pmcb->escape_prob.GetSize() > 0) {
    const Real *pe = &(pmcb->escape_prob(i3-pmcb->ks,i2-pmcb->js,i1-pmcb->is,0));
    const Real xi = pmcb->pmy_mc->bias_energy_mix;
    Real pmean = 0.;
    for (int l=0; l<ng; ++l) pmean += (prob[l+1]-prob[l])*pe[l];
    if (pmean > 0.) {
      Real cum = 0., q = 0., m = 1.;
      int l = 0;
      for (l=0; l<ng; ++l) {
        m = (1. - xi) + xi*pe[l]/pmean;
        q = (prob[l+1]-prob[l])*m;
        if (dev < cum + q || l == ng-1) break;
        cum += q;
      }
      Real a = (q > 0.) ? std::min(std::max((dev-cum)/q, 0.), 1.) : 0.;
      pphot->wp[ip] /= m;
      energy = tab->DrawEnergy(l, a, temp);
      drawn = true;
    }
  }
  if (!drawn) {
    const int i = mcbisect(dev, const_cast<Real*>(prob), ng+1);
    const Real a = (dev-prob[i])/(prob[i+1]-prob[i]);
    energy = tab->DrawEnergy(i, a, temp);
  }
  pphot->ep[ip] = energy;

  // isotropic direction; k0p carries the energy
  Real phi = 2. * PI * pran->uniform();
  Real cphi = cos(phi);
  Real sphi = sin(phi);
  Real cth = 2. * pran->uniform() - 1.;
  Real sth = std::sqrt(1. - SQR(cth));
  pphot->k0p[ip] = pphot->ep[ip];
  pphot->k1p[ip] = sth*cphi;
  pphot->k2p[ip] = sth*sphi;
  pphot->k3p[ip] = cth;
}
