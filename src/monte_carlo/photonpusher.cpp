//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file photonpusher.cpp
//! \brief implementation for photon moving functions

// C/C++ headers
#include <algorithm>
#include <cmath>
#include <stdexcept>

// Athena++ headers
#include "photon.hpp"
#include "mrw.hpp"
#include "photonpusher.hpp"
#include "../mesh/mesh.hpp"
#include "../globals.hpp"

//----------------------------------------------------------------------------------------
//! PhotonPusher base class constructor, built from  MonteCarloBlock

PhotonPusher::PhotonPusher(MonteCarloBlock *pmcb) {

  pmy_mcb = pmcb;
  pmy_mc = pmcb->pmy_mc;
  pcoord = NULL;
  UserWorkInMove = pmcb->pmy_mc->UserWorkInMove;
  capmove = pmcb->pmy_mc->capmove;

  // MRW acceleration
  acceleration = pmcb->acceleration;
  boosts = pmcb->boosts;
  resonance = (pmcb->scattering_meth == SCATRES);
  compton = (pmcb->scattering_meth == SCATCOMP);
  //compton = (!pmcb->coherent_scattering) && (!resonance);
  stretch_ = pmy_mc->stretch;
  stretch_lnbound_ = std::log(pmy_mc->stretch_bound);
  stretch_taucell_ = pmy_mc->stretch_taucell;
  stretching_ = (stretch_ != 1.);

  mrw_ = pmy_mc->mrw;
  mrw_s_pmax_ = HUGE_NUMBER;
  if (acceleration && !resonance && mrw_ != nullptr && pmy_mc->accel_pmax > 0.)
    mrw_s_pmax_ = mrw_->SphereFirstPassage(1. - pmy_mc->accel_pmax);
}

//----------------------------------------------------------------------------------------
//! destructor

PhotonPusher::~PhotonPusher() {
}

//----------------------------------------------------------------------------------------
//! \fn void PhotonPusher::Move(Photon *pphot, int ips, int ipe)
//  \brief base class move does nothing

void PhotonPusher::Move(Photon *pphot, int ips, int ipe) {

}

// Time-sampling rejection method functions
// p(t), original function
Real OriginalFunction(Real t, Real decayRate, Real diffusionTime) {
  return exp(-(SQR(diffusionTime / t))) * decayRate * exp(-decayRate * t);
}

// f(t), comparison function
Real ComparisonFunction(Real t, Real decayRate) {
  return decayRate * exp(-decayRate * t);
}


//----------------------------------------------------------------------------------------
//! \fn Real PhotonPusher::SampleEscapeTime(MCRandom *pran, Real decayRate, Real sphereRadius,
//!                                        Real diffusionTime) {
//! \brief Sample a photon escape time from a sphere using the rejection method
Real PhotonPusher::SampleEscapeTime(MCRandom *pran, Real decayRate, Real sphereRadius,
                                   Real diffusionTime) {
  Real c = 2.99792458e10;
  Real lightCrossingTime = sphereRadius / c;
  Real timeSample;

  bool reject = true;
  while (reject) {
    // Sample an area under the comparison function
    Real areaSample = pran->uniform();

    // Find t for which the area under f(t) to the left of t is equal to areaSample
    timeSample = - 1. / decayRate * log(exp(-decayRate * lightCrossingTime) - areaSample);

    // Sample a value between 0 and f(timeSample)
    Real comparisonSample = ComparisonFunction(timeSample, decayRate) * pran->uniform();

    // Reject or accept based on value
    if (comparisonSample <= OriginalFunction(timeSample, decayRate, diffusionTime)) {
      reject = false;
    }
  }
  return timeSample;
}

//----------------------------------------------------------------------------------------
//! \fn bool PhotonPusher::MRWResonanceAcceleration(Photon *pphot, MCRandom *pran, Real dist,
//!                                       Real tauacc, int ip)
//! \brief Accelerate photon diffusion with modified random walk method

void PhotonPusher::MRWResonanceAcceleration(Photon *pphot, MCRandom *pran, Real dist, Real tauacc,
                                           Real &path_length, Real &k1, Real &k2, Real &k3, int ip) {
  MonteCarloBlock *pmcb = pmy_mcb;
  Real r0 = dist;

  // Line constants
  Real melectron = 9.10938215e-28;
  Real charge = 4.80320427e-10;
  Real osc_strength = 0.4164;
  Real nu0 = 2.468e15;
  Real c = 2.99792458e10;
  Real h = 6.62607015e-27;
  Real kb = 1.380649e-16;
  Real mass = 1.660538782e-24;

  if (pmcb->topology == MCTOPO_SPHERICAL) {

    // ********* FREQUENCY REDISTRIBUTION *********
    // Sample outgoing frequency from Dijkstra et al 2006 solution
    int &i1 = pphot->i1p[ip];
    int &i2 = pphot->i2p[ip];
    int &i3 = pphot->i3p[ip];

    // Cell properties
    Real tgas = pmcb->tgas(i3,i2,i1);
    Real rho = pmcb->rho(i3,i2,i1);

    // Derived parameters
    Real vth = sqrt( 2 * kb * tgas / mass);
    Real doppwidth = nu0 * vth / c;
    Real lorwidth = 6.265e8/(4.*PI);
    Real a = lorwidth / doppwidth;
    Real k = (rho/mass) * PI*charge*charge / (melectron*c) * osc_strength;
    Real tau0 = k * r0 / sqrt(PI) / doppwidth;
    //if (a*tau0 > 1.) {
    //  return false;
    //}

    // Sample sigma and convert to frequency
    Real x_s = (pphot->ep[ip] / h - nu0)/doppwidth;
    Real sigma_s = sqrt(2./3.) * PI/a * std::pow(x_s, 3.)/3.;
    Real samp_sigma = tau0 * 2./sqrt(PI) * std::atanh(2.*pran->uniform() - 1.) + sigma_s;
    Real x = std::cbrt(3. * sqrt(3./2.) * a / PI * samp_sigma);
    Real nu = doppwidth * x + nu0;
    pphot->ep[ip] = h * nu;

    // ********* POSITION *********
    // position packet on sphere of radius r0
    Real mu = 2.*pran->uniform()-1.0;

    // Local angles within the sphere of radius r0
    Real lsth = sqrt(1.0-mu*mu);
    Real lphi = 2.*PI*pran->uniform();

    // convert to cartesian
    // Global simulation angles based on photon position
    Real cth = cos(pphot->x2p[ip]);
    Real sth = sqrt(1. - SQR(cth));
    Real cph = cos(pphot->x3p[ip]);
    Real sph = sin(pphot->x3p[ip]);
    Real r = pphot->x1p[ip];

    // Cartesian position before move
    Real x0 = r * sth * cph;
    Real y0 = r * sth * sph;
    Real z0 = r * cth;

    // Cartesian position after move
    Real x1 = x0 + lsth*cos(lphi) * r0;
    Real y1 = y0 + lsth*sin(lphi) * r0;
    Real z1 = z0 + mu * r0;

    // Updated photon position in global spherical polar coordinates
    pphot->x1p[ip] = sqrt(SQR(x1)+SQR(y1)+SQR(z1));
    pphot->x2p[ip] = acos(z1 / pphot->x1p[ip]);
    pphot->x3p[ip] = atan2(y1,x1);
    if (pphot->x3p[ip] < 0.)
      pphot->x3p[ip] += 2.*PI;

    // Updated global coordinate angles after the move onto surf of sphere
    cth = cos(pphot->x2p[ip]);
    sth = sqrt(1. - SQR(cth));
    cph = cos(pphot->x3p[ip]);
    sph = sin(pphot->x3p[ip]);

    // Cartesion displacement direction vectors
    Real disp = sqrt(SQR(x1-x0)+SQR(y1-y0)+SQR(z1-z0));
    Real k1cart = (x1 - x0) / disp;
    Real k2cart = (y1 - y0) / disp;
    Real k3cart = (z1 - z0) / disp;

    // Spherical polar displacement direction vectors - set vars passed by reference
    k1 = k1cart * sth * cph + k2cart * sth * sph + k3cart * cth;
    k2 = k1cart * cth * cph + k2cart * cth * sph - k3cart * sth;
    k3 = -k1cart * sph + k2cart * cph;

    // ********* DIRECTION *********
    // Sample outgoing angles to local normal - zero ingoing flux, so must be outward
    Real sq3 = 2.*sqrt(3.);
    Real xi = pran->uniform();
    Real samp_cth = (2./sq3)*(sqrt(-sq3 * xi - 3.*xi + sq3 + 4.) - 1.);
    Real samp_sth = sqrt(1.0 - samp_cth*samp_cth);
    Real samp_phi = 2.*PI*pran->uniform();

    // Local cos and sin of phi - positions in sphere of radius r0
    Real lcph = cos(lphi);
    Real lsph = sin(lphi);

    // Cartesian direction vectors in local sphere
    // n = er * samp_cth + etheta * samp_sth * cos(samp_phi) + ephi * samp_sth * sin(samp_phi)
    Real nx = lsth * lcph * samp_cth + mu * lcph * samp_sth * cos(samp_phi) - lsph * samp_sth * sin(samp_phi);
    Real ny = lsth * lsph * samp_cth + mu * lsph * samp_sth * cos(samp_phi) + lcph * samp_sth * sin(samp_phi);
    Real nz = mu * samp_cth - lsth * samp_sth * cos(samp_phi);

    // Global sphpol direction vectors from local cartesian on the sphere
    pphot->k1p[ip] = nx * sth * cph + ny * sth * sph + nz * cth;
    pphot->k2p[ip] = nx * cth * cph + ny * cth * sph - nz * sth;
    pphot->k3p[ip] = -nx * sph + ny * cph;

    // Sample an escape time using the rejection method
    Real tcoeff = 1.698161839733523; // from McClellan et al 2022, Fig 8
    Real tdiff = r0 / c * std::pow(a * tau0, 1./3.); // Diffusion timescale
    Real decayRate = 1./(tcoeff * std::pow(a * tau0, 1./3.)); // Fit for the lowest-order eigenfreq
    Real timeSample = SampleEscapeTime(pran, decayRate, r0, tdiff);
    path_length = timeSample * c; // set path length passed by reference

  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [PhotonPusher::MRWAcceleration]"
          <<std::endl<< "Specified coordinate system not implemented for resonance acceleration" <<std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real PhotonPusher::GetOpticalDepth(MCRandom *pran)
//! \brief return exponentially distributed optical depth variable

Real PhotonPusher::GetOpticalDepth(MCRandom *pran) {

  Real dev = pran->uniform();
  while(dev <= 0.)
    dev=pran->uniform();
  //std::cout << dev << std::endl;
  return -log(dev);
}

//----------------------------------------------------------------------------------------
//! \fn Real PhotonPusher::StretchFactor(Photon *pphot, int ip, Real chi, Real dl_seg)
//! \brief extinction multiplier for the next flight segment

Real PhotonPusher::StretchFactor(Photon *pphot, int ip, Real chi, Real dl_seg) {
  if (!stretching_ || chi <= 0. || dl_seg <= 0.) return 1.;
  const Real l_cgs = pmy_mcb->l_cgs;
  if (stretch_taucell_ < HUGE_NUMBER &&
      chi * l_cgs * CellWidth(pphot, ip) > stretch_taucell_) return 1.;
  const Real tau_seg = chi * l_cgs * dl_seg;
  const Real d = pphot->strp[ip];
  if (stretch_ > 1.) return std::min(stretch_, 1. + (stretch_lnbound_ - d) / tau_seg);
  return std::max(stretch_, 1. - (stretch_lnbound_ + d) / tau_seg);
}

//----------------------------------------------------------------------------------------
//! \fn Real PhotonPusher::CellWidth(Photon *pphot, int ip)
//! \brief smallest coordinate width of the photon's cell

Real PhotonPusher::CellWidth(Photon *pphot, int ip) {
  const Real dx1 = pcoord->x1f(pphot->i1p[ip]+1) - pcoord->x1f(pphot->i1p[ip]);
  const Real dx2 = pcoord->x2f(pphot->i2p[ip]+1) - pcoord->x2f(pphot->i2p[ip]);
  const Real dx3 = pcoord->x3f(pphot->i3p[ip]+1) - pcoord->x3f(pphot->i3p[ip]);
  return std::min(dx1, std::min(dx2, dx3));
}

//----------------------------------------------------------------------------------------
//! \fn Real PhotonPusher::GetExtinctionCoefficient(Real ac, Real sc, bool abs_tau)
//! \brief returns total opacity or scattering opacity depending on method

Real PhotonPusher::GetExtinctionCoefficient(Real ac, Real sc, bool abs_tau) {

  Real chi;
  if (abs_tau) {
    chi = sc;
  } else {
    chi = sc + ac;
  }
  return chi;
  //return (chi > TINY_NUMBER) ? chi : TINY_NUMBER;
}

//----------------------------------------------------------------------------------------
//! \fn Real PhotonPusher::ExpTauAbsorption(Real ac, Real dl, bool abs_tau)
//! \brief Computes e^-tau_abs

Real PhotonPusher::ExpTauAbsorption(Real ac, Real dl, bool abs_tau) {

  if (abs_tau) {
    return exp(-ac * dl);
  } else {
    return 1.;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void PhotonPusher::NextFace(Real dx1, Real dx2, Real dx3, int &face, Real &dx)
//! \brief returns flag with next face and distance to next face

void PhotonPusher::NextFace(Real dx1, Real dx2, Real dx3, int &face, Real &dx) {

// face tells which cell coordinates need to be updatde
//   x:   0
//   y:   1
//   z:   2
//   xy:  3
//   yz:  4
//   xz:  5
//   xyz: 6

  // check for positiviity
  /*if (dx1 < 0.) {
    dx1 = HUGE_NUMBER;
    printf("Warning: dx1 < 0\n");
  }
  if (dx2 < 0.) {
    dx2 = HUGE_NUMBER;
    printf("Warning: dx2 < 0\n");
  }
  if (dx3 < 0.) {
    dx3 = HUGE_NUMBER;
    printf("Warning: dx3 < 0\n");
    }*/

  dx = dx1;

  if(dx2 < dx) {
    dx = dx2;
    if(dx3 < dx) {
      dx = dx3;
      face = 2;
      return;
    } else if(dx3 > dx) {
      face = 1;
      return;
    } else {
      face = 4;
      return;
    }
  } else if(dx2 > dx) {
    if(dx3 < dx) {
      dx = dx3;
      face = 2;
      return;
    } else if(dx3 > dx) {
      face = 0;
      return;
    } else {
      face = 5;
      return;
    }
  } else {
    if(dx3 < dx) {
      dx = dx3;
      face = 2;
      return;
    } else if(dx3 > dx) {
      face = 3;
      return;
    } else {
      face = 6;
      return;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void PhotonPusher::MovePhotonToNextZone(Photon *pphot, MCCoord *pco,
//!                           MonteCarloBlock *pmcb, int face, bool ascend[3], int ip))
//! \brief updates photon cell when face is known

void PhotonPusher::MovePhotonToNextZone(Photon *pphot, MCCoord *pco, MonteCarloBlock *pmcb,
                                       int face, bool ascend[3], int ip) {

  // Update face(s) and adjust positions to lie exactly on boundary
  if ((face == 0) || (face == 3) || (face == 5) || (face == 6)) {
    //update x1 face
    if (ascend[0]) {
      pphot->i1p[ip]++;
      pphot->x1p[ip] = pco->x1f(pphot->i1p[ip]);
      if(pphot->i1p[ip] > pmcb->ie)
        pmcb->pbval->BoundaryFunction_[BoundaryFace::outer_x1](pmcb,pco,pphot,ip);
    } else {
      pphot->i1p[ip]--;
      pphot->x1p[ip] = pco->x1f(pphot->i1p[ip]+1);
      if(pphot->i1p[ip] < pmcb->is)
        pmcb->pbval->BoundaryFunction_[BoundaryFace::inner_x1](pmcb,pco,pphot,ip);
    }
  }
  if ((face == 1) || (face == 3) || (face == 4) || (face == 6)) {
    //update x2 face
    if (ascend[1]) {
      pphot->i2p[ip]++;
      pphot->x2p[ip] = pco->x2f(pphot->i2p[ip]);
      if(pphot->i2p[ip] > pmcb->je)
        pmcb->pbval->BoundaryFunction_[BoundaryFace::outer_x2](pmcb,pco,pphot,ip);
    } else {
      pphot->i2p[ip]--;
      pphot->x2p[ip] = pco->x2f(pphot->i2p[ip]+1);
      if(pphot->i2p[ip] < pmcb->js)
        pmcb->pbval->BoundaryFunction_[BoundaryFace::inner_x2](pmcb,pco,pphot,ip);
    }
  }
  if ((face == 2) || (face == 4) || (face == 5) || (face == 6)) {
    //update x3 face
    if (ascend[2]) {
      pphot->i3p[ip]++;
      pphot->x3p[ip] = pco->x3f(pphot->i3p[ip]);
      if(pphot->i3p[ip] > pmcb->ke)
        pmcb->pbval->BoundaryFunction_[BoundaryFace::outer_x3](pmcb,pco,pphot,ip);
    } else {
      pphot->i3p[ip]--;
      pphot->x3p[ip] = pco->x3f(pphot->i3p[ip]+1);
      if(pphot->i3p[ip] < pmcb->ks)
        pmcb->pbval->BoundaryFunction_[BoundaryFace::inner_x3](pmcb,pco,pphot,ip);
    }
  }

  // Update opacities
  if (pphot->statp[ip] == EVOLVING) {
    // Opacities need to be calculated using comoving frame energy and then transformed
    // back to Eulerian frame when Lorentz Transformations are enabled.
    int &i1 = pphot->i1p[ip];
    int &i2 = pphot->i2p[ip];
    int &i3 = pphot->i3p[ip];
    if (pmy_mcb->boosts || false) {
      // Shift photon energy to comoving frame
      //Real shift = pmy_mcb->LorentzTransformFrequencyShift(pphot,ip);
      Real shift = pmy_mcb->FrequencyShiftComoving(pphot,ip);
      pphot->ep[ip] *= shift;
      // compute opacities in comoving frame
      pphot->acp[ip] = pmcb->AbsorptionOpacity(pmcb,pphot,ip);
      pphot->scp[ip] = pmcb->ScatteringOpacity(pmcb,pphot,ip);
      // Shift energy back to Eulerian frame
      pphot->ep[ip] /= shift;
      // Shift opacities to Eulerian frame
      pphot->acp[ip] *= shift;
      pphot->scp[ip] *= shift;
    } else {
      // No distinction between comovinng frame and eulerian frame
      pphot->acp[ip] = pmcb->AbsorptionOpacity(pmcb,pphot,ip);
      pphot->scp[ip] = pmcb->ScatteringOpacity(pmcb,pphot,ip);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn bool PhotonPusher::UpdateZone(photon *pphot, int ip)
//! \brief check/updates photon cell after displacement

bool PhotonPusher::UpdateZone(Photon *pphot, int ip) {

  bool change = false;
  MonteCarloBlock *pmcb = pmy_mcb;
  bool update = false;

  // Check x1 direction
  if (pphot->x1p[ip] >= pcoord->x1f(pphot->i1p[ip]+1)) {
    update = true;
    while (pphot->x1p[ip] >= pcoord->x1f(pphot->i1p[ip]+1)) {
      pphot->i1p[ip]++;
      if(pphot->i1p[ip] > pmcb->ie)
        pmcb->pbval->BoundaryFunction_[BoundaryFace::outer_x1](pmcb,pcoord,pphot,ip);
      if (pphot->statp[ip] != EVOLVING) {
        break;
      }
    }
  } else if (pphot->x1p[ip] < pcoord->x1f(pphot->i1p[ip])) {
    update = true;
    while (pphot->x1p[ip] < pcoord->x1f(pphot->i1p[ip])) {
      pphot->i1p[ip]--;
      if(pphot->i1p[ip] < pmcb->is)
        pmcb->pbval->BoundaryFunction_[BoundaryFace::inner_x1](pmcb,pcoord,pphot,ip);
      if (pphot->statp[ip] != EVOLVING) {
        break;
      }
    }
  }
  // Check x2 direction
  if (pphot->x2p[ip] >= pcoord->x2f(pphot->i2p[ip]+1)) {
    update = true;
    while (pphot->x2p[ip] >= pcoord->x2f(pphot->i2p[ip]+1)) {
      pphot->i2p[ip]++;
      if(pphot->i2p[ip] > pmcb->je)
        pmcb->pbval->BoundaryFunction_[BoundaryFace::outer_x2](pmcb,pcoord,pphot,ip);
      if (pphot->statp[ip] != EVOLVING) {
        break;
      }
    }
  } else if (pphot->x2p[ip] < pcoord->x2f(pphot->i2p[ip])) {
    update = true;
    while (pphot->x2p[ip] < pcoord->x2f(pphot->i2p[ip])) {
      pphot->i2p[ip]--;
      if(pphot->i2p[ip] < pmcb->js)
        pmcb->pbval->BoundaryFunction_[BoundaryFace::inner_x2](pmcb,pcoord,pphot,ip);
      if (pphot->statp[ip] != EVOLVING) {
        break;
      }
    }
  }

  // Check x3 direction
  if (pphot->x3p[ip] >= pcoord->x3f(pphot->i3p[ip]+1)) {
    update = true;
    while (pphot->x3p[ip] >= pcoord->x3f(pphot->i3p[ip]+1)) {
      pphot->i3p[ip]++;
      if(pphot->i3p[ip] > pmcb->ke)
        pmcb->pbval->BoundaryFunction_[BoundaryFace::outer_x3](pmcb,pcoord,pphot,ip);
      if (pphot->statp[ip] != EVOLVING) {
        break;
      }
    }
  } else if (pphot->x3p[ip] < pcoord->x3f(pphot->i3p[ip])) {
    update = true;
    while (pphot->x3p[ip] < pcoord->x3f(pphot->i3p[ip])) {
      pphot->i3p[ip]--;
      if(pphot->i3p[ip] < pmcb->ks)
        pmcb->pbval->BoundaryFunction_[BoundaryFace::inner_x3](pmcb,pcoord,pphot,ip);
      if (pphot->statp[ip] != EVOLVING) {
        break;
      }
    }
  }
  // Returns true if cell changes, false otherwise
  return update;

}

//----------------------------------------------------------------------------------------
//! \fn bool PhotonPusher::IsOnBlock(photon *pphot, int ip)
//! \brief Confirm photon is on block

bool PhotonPusher::IsOnBlock(Photon *pphot, int ip) {

  bool on_block = true;
  if (pphot->i1p[ip] < pmy_mcb->is) {
    on_block = false;
  } else if (pphot->i1p[ip] > pmy_mcb->ie) {
    on_block = false;
  } else if (pphot->i2p[ip] < pmy_mcb->js) {
    on_block = false;
  } else if (pphot->i2p[ip] > pmy_mcb->je) {
    on_block = false;
  } else if (pphot->i3p[ip] < pmy_mcb->ks) {
    on_block = false;
  } else if (pphot->i3p[ip] > pmy_mcb->ke) {
    on_block = false;
  }
  if (!on_block) {
    pphot->statp[ip] = DESTROYED;
    pphot->PrintPhoton("Warning: [CheckZone], Photon not on block, destroyed",ip);
  }
  return on_block;
}

