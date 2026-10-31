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
  mrw_ = pmcb->mrw;
  ddmc_ = pmy_mc->ddmc;
  boosts = pmcb->boosts;
  resonance = (pmcb->scattering_meth == SCATRES);
  compton = (pmcb->scattering_meth == SCATCOMP);
  //compton = (!pmcb->coherent_scattering) && (!resonance);
  stretch_ = pmy_mc->stretch;
  stretch_lnbound_ = std::log(pmy_mc->stretch_bound);
  stretch_taucell_ = pmy_mc->stretch_taucell;
  stretching_ = (stretch_ != 1.);

  mrw_tables_ = pmy_mc->mrw_tables;
  mrw_s_pmax_ = HUGE_NUMBER;
  if (mrw_ && !resonance && mrw_tables_ != nullptr && pmy_mc->mrw_pmax > 0.)
    mrw_s_pmax_ = mrw_tables_->SphereFirstPassage(1. - pmy_mc->mrw_pmax);
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
    // two_band_decorrelate: the new cell's lines unrelated to the old cell's with this
    // probability, so the band rank is redrawn before the absorption is read
    if (pmy_mcb->table_decorrelate > 0.
        && pmy_mcb->pran->uniform() < pmy_mcb->table_decorrelate)
      pphot->ubp[ip] = pmy_mcb->pran->uniform();
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
  // two_band_decorrelate: the new cell's lines unrelated to the old cell's with this
  // probability (the general pusher refreshes the absorption at its next step)
  if (update && pmy_mcb->table_decorrelate > 0.
      && pmy_mcb->pran->uniform() < pmy_mcb->table_decorrelate)
    pphot->ubp[ip] = pmy_mcb->pran->uniform();
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

