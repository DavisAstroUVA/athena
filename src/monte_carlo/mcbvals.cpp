//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file montecarloblock.cpp
//! \brief implementation of functions in class MCBoundaryValues

#include <stdexcept>  // runtime_error

// Athena++ headers
#include "mcbvals.hpp"
#include "photon.hpp"

//----------------------------------------------------------------------------------------
//! MCBoundaryValues class constructor, built from ParameterInput and MonteCarloBlock
// SWD: ParameterInput not needed
MCBoundaryValues::MCBoundaryValues(MonteCarloBlock *pmcb, ParameterInput *pin) {

  pmy_mcb = pmcb;

// Set BC functions for each of the 6 boundaries -----------------------------------------
  // Inner x1
  switch(pmcb->mcb_bcs[BoundaryFace::inner_x1]) {
    case MC_PERIODIC_BNDRY:
      BoundaryFunction_[BoundaryFace::inner_x1] = Periodic;
      break;
    case MC_ESCAPE_BNDRY:
      BoundaryFunction_[BoundaryFace::inner_x1] = Escape;
      break;
    case MC_ABSORB_BNDRY:
      BoundaryFunction_[BoundaryFace::inner_x1] = Absorb;
      break;
    case MC_DESTROY_BNDRY:
      BoundaryFunction_[BoundaryFace::inner_x1] = Destroy;
      break;
    case MC_POLAR_BNDRY:
      BoundaryFunction_[BoundaryFace::inner_x1] = Polar;
      break;
    case MC_REFLECT_BNDRY:
      BoundaryFunction_[BoundaryFace::inner_x1] = ReflectMCInnerX1;
      break;
    case MC_BLOCK_BNDRY:
      BoundaryFunction_[BoundaryFace::inner_x1] = Block;
      break;
    case MC_USER_BNDRY:
      BoundaryFunction_[BoundaryFace::inner_x1] =
        pmcb->pmy_mc->BoundaryFunction_[BoundaryFace::inner_x1];
      break;
    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in MCBoundaryValues constructor" << std::endl
          << "Flag ix1_mc_bc=" << pmcb->mcb_bcs[BoundaryFace::inner_x1] << " not valid"
          << std::endl;
      throw std::runtime_error(msg.str().c_str());
  }
  // outer x1
  switch(pmcb->mcb_bcs[BoundaryFace::outer_x1]) {
    case MC_PERIODIC_BNDRY:
      BoundaryFunction_[BoundaryFace::outer_x1] = Periodic;
      break;
    case MC_ESCAPE_BNDRY:
      BoundaryFunction_[BoundaryFace::outer_x1] = Escape;
      break;
    case MC_ABSORB_BNDRY:
      BoundaryFunction_[BoundaryFace::outer_x1] = Absorb;
      break;
    case MC_DESTROY_BNDRY:
      BoundaryFunction_[BoundaryFace::outer_x1] = Destroy;
      break;
    case MC_POLAR_BNDRY:
      BoundaryFunction_[BoundaryFace::outer_x1] = Polar;
      break;
    case MC_REFLECT_BNDRY:
      BoundaryFunction_[BoundaryFace::outer_x1] = ReflectMCOuterX1;
      break;
    case MC_BLOCK_BNDRY:
      BoundaryFunction_[BoundaryFace::outer_x1] = Block;
      break;
    case MC_USER_BNDRY:
      BoundaryFunction_[BoundaryFace::outer_x1] =
        pmcb->pmy_mc->BoundaryFunction_[BoundaryFace::outer_x1];
      break;
    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in MCBoundaryValues constructor" << std::endl
          << "Flag ix1_mc_bc=" << pmcb->mcb_bcs[BoundaryFace::outer_x1] << " not valid"
          << std::endl;
      throw std::runtime_error(msg.str().c_str());
  }
  // Inner x2
  switch(pmcb->mcb_bcs[BoundaryFace::inner_x2]) {
    case MC_PERIODIC_BNDRY:
      BoundaryFunction_[BoundaryFace::inner_x2] = Periodic;
      break;
    case MC_ESCAPE_BNDRY:
      BoundaryFunction_[BoundaryFace::inner_x2] = Escape;
      break;
    case MC_ABSORB_BNDRY:
      BoundaryFunction_[BoundaryFace::inner_x2] = Absorb;
      break;
    case MC_DESTROY_BNDRY:
      BoundaryFunction_[BoundaryFace::inner_x2] = Destroy;
      break;
    case MC_POLAR_BNDRY:
      BoundaryFunction_[BoundaryFace::inner_x2] = Polar;
      break;
    case MC_REFLECT_BNDRY:
      BoundaryFunction_[BoundaryFace::inner_x2] = ReflectMCInnerX2;
      break;
    case MC_BLOCK_BNDRY:
      BoundaryFunction_[BoundaryFace::inner_x2] = Block;
      break;
    case MC_USER_BNDRY:
      BoundaryFunction_[BoundaryFace::inner_x2] =
        pmcb->pmy_mc->BoundaryFunction_[BoundaryFace::inner_x2];
      break;
    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in MCBoundaryValues constructor" << std::endl
          << "Flag ix1_mc_bc=" << pmcb->mcb_bcs[BoundaryFace::inner_x2] << " not valid"
          << std::endl;
      throw std::runtime_error(msg.str().c_str());
  }
  // outer x2
  switch(pmcb->mcb_bcs[BoundaryFace::outer_x2]) {
    case MC_PERIODIC_BNDRY:
      BoundaryFunction_[BoundaryFace::outer_x2] = Periodic;
      break;
    case MC_ESCAPE_BNDRY:
      BoundaryFunction_[BoundaryFace::outer_x2] = Escape;
      break;
    case MC_ABSORB_BNDRY:
      BoundaryFunction_[BoundaryFace::outer_x2] = Absorb;
      break;
    case MC_DESTROY_BNDRY:
      BoundaryFunction_[BoundaryFace::outer_x2] = Destroy;
      break;
    case MC_POLAR_BNDRY:
      BoundaryFunction_[BoundaryFace::outer_x2] = Polar;
      break;
    case MC_REFLECT_BNDRY:
      BoundaryFunction_[BoundaryFace::outer_x2] = ReflectMCOuterX2;
      break;
    case MC_BLOCK_BNDRY:
      BoundaryFunction_[BoundaryFace::outer_x2] = Block;
      break;
    case MC_USER_BNDRY:
      BoundaryFunction_[BoundaryFace::outer_x2] =
        pmcb->pmy_mc->BoundaryFunction_[BoundaryFace::outer_x2];
      break;
    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in MCBoundaryValues constructor" << std::endl
          << "Flag ix1_mc_bc=" << pmcb->mcb_bcs[BoundaryFace::outer_x2] << " not valid"
          << std::endl;
      throw std::runtime_error(msg.str().c_str());
  }
  // Inner x3
  switch(pmcb->mcb_bcs[BoundaryFace::inner_x3]) {
    case MC_PERIODIC_BNDRY:
      BoundaryFunction_[BoundaryFace::inner_x3] = Periodic;
      break;
    case MC_ESCAPE_BNDRY:
      BoundaryFunction_[BoundaryFace::inner_x3] = Escape;
      break;
    case MC_ABSORB_BNDRY:
      BoundaryFunction_[BoundaryFace::inner_x3] = Absorb;
      break;
    case MC_DESTROY_BNDRY:
      BoundaryFunction_[BoundaryFace::inner_x3] = Destroy;
      break;
    case MC_POLAR_BNDRY:
      BoundaryFunction_[BoundaryFace::inner_x3] = Polar;
      break;
    case MC_REFLECT_BNDRY:
      BoundaryFunction_[BoundaryFace::inner_x3] = ReflectMCInnerX3;
      break;
    case MC_BLOCK_BNDRY:
      BoundaryFunction_[BoundaryFace::inner_x3] = Block;
      break;
    case MC_USER_BNDRY:
      BoundaryFunction_[BoundaryFace::outer_x3] =
        pmcb->pmy_mc->BoundaryFunction_[BoundaryFace::outer_x3];
      break;
    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in MCBoundaryValues constructor" << std::endl
          << "Flag ix1_mc_bc=" << pmcb->mcb_bcs[BoundaryFace::inner_x3] << " not valid"
          << std::endl;
      throw std::runtime_error(msg.str().c_str());
  }
  // outer x3
  switch(pmcb->mcb_bcs[BoundaryFace::outer_x3]) {
    case MC_PERIODIC_BNDRY:
      BoundaryFunction_[BoundaryFace::outer_x3] = Periodic;
      break;
    case MC_ESCAPE_BNDRY:
      BoundaryFunction_[BoundaryFace::outer_x3] = Escape;
      break;
    case MC_ABSORB_BNDRY:
      BoundaryFunction_[BoundaryFace::outer_x3] = Absorb;
      break;
    case MC_DESTROY_BNDRY:
      BoundaryFunction_[BoundaryFace::outer_x3] = Destroy;
      break;
    case MC_POLAR_BNDRY:
      BoundaryFunction_[BoundaryFace::outer_x3] = Polar;
      break;
    case MC_REFLECT_BNDRY:
      BoundaryFunction_[BoundaryFace::outer_x3] = ReflectMCOuterX3;
      break;
    case MC_BLOCK_BNDRY:
      BoundaryFunction_[BoundaryFace::outer_x3] = Block;
      break;
    case MC_USER_BNDRY:
      BoundaryFunction_[BoundaryFace::outer_x3] =
        pmcb->pmy_mc->BoundaryFunction_[BoundaryFace::outer_x3];
      break;
    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in MCBoundaryValues constructor" << std::endl
          << "Flag ix1_mc_bc=" << pmcb->mcb_bcs[BoundaryFace::outer_x3] << " not valid"
          << std::endl;
      throw std::runtime_error(msg.str().c_str());
  }

}

//----------------------------------------------------------------------------------------
//!  destructor

MCBoundaryValues::~MCBoundaryValues() {

}

// SWD: Currently assumes single block -- ***will need to be changed***

//----------------------------------------------------------------------------------------
//! \fn void PeriodicInnerX1(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip)
//! \brief periodic boundary conditions, inner x1 boundary

void PeriodicInnerX1(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip) {

  pphot->i1p[ip] = pmcb->ie;
  pphot->x1p[ip] = pco->x1f(pphot->i1p[ip]+1);

}

//----------------------------------------------------------------------------------------
//! \fn void PeriodicOuterX1(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip)
//! \brief periodic boundary conditions, outer x1 boundary

void PeriodicOuterX1(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip) {

  pphot->i1p[ip] = pmcb->is;
  pphot->x1p[ip] = pco->x1f(pphot->i1p[ip]);

}

//----------------------------------------------------------------------------------------
//! \fn void PeriodicInnerX2(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip)
//! \brief periodic boundary conditions, inner x2 boundary

void PeriodicInnerX2(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip) {

  pphot->i2p[ip] = pmcb->je;
  pphot->x2p[ip] = pco->x2f(pphot->i2p[ip]+1);

}

//----------------------------------------------------------------------------------------
//! \fn void PeriodicOuterX2(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip)
//! \brief periodic boundary conditions, outer x2 boundary

void PeriodicOuterX2(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip) {

  pphot->i2p[ip] = pmcb->js;
  pphot->x2p[ip] = pco->x2f(pphot->i2p[ip]);

}

//----------------------------------------------------------------------------------------
//! \fn void PeriodicInnerX3(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip)
//! \brief periodic boundary conditions, inner x3 boundary

void PeriodicInnerX3(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip) {

  pphot->i3p[ip] = pmcb->ke;
  pphot->x3p[ip] = pco->x3f(pphot->i3p[ip]+1);

}

//----------------------------------------------------------------------------------------
//! \fn void PeriodicOuterX3(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip)
//! \brief periodic boundary conditions, outer x3 boundary

void PeriodicOuterX3(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip) {

  pphot->i3p[ip] = pmcb->ks;
  pphot->x3p[ip] = pco->x3f(pphot->i3p[ip]);

}


//----------------------------------------------------------------------------------------
//! \fn void ReflectMCInnerX1(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip)
//! \brief reflecting boundary conditions, inner x1 boundary

void ReflectMCInnerX1(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip) {

  const Real frac = 1.e-7;
  pphot->k1p[ip] *= -1.;
  pphot->i1p[ip] = pmcb->is;
  pphot->x1p[ip] = pco->x1f(pphot->i1p[ip])*(1.-frac) + pco->x1f(pphot->i1p[ip]+1)*frac;

}

//----------------------------------------------------------------------------------------
//! \fn void ReflectMCOuterX1(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip)
//! \brief reflecting boundary conditions, outer x1 boundary

void ReflectMCOuterX1(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip) {

  const Real frac = 1.e-7;
  pphot->k1p[ip] *= -1.;
  pphot->i1p[ip] = pmcb->ie;
  pphot->x1p[ip] = pco->x1f(pphot->i1p[ip]+1)*(1.-frac) + pco->x1f(pphot->i1p[ip])*frac;

}

//----------------------------------------------------------------------------------------
//! \fn void ReflectMCInnerX2(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip)
//! \brief reflecting boundary conditions, inner x2 boundary

void ReflectMCInnerX2(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip) {

  const Real frac = 1.e-8;
  pphot->k2p[ip] *= -1.;
  pphot->i2p[ip] = pmcb->js;
  pphot->x2p[ip] = pco->x2f(pphot->i2p[ip])*(1.-frac) + pco->x2f(pphot->i2p[ip]+1)*frac;

}

//----------------------------------------------------------------------------------------
//! \fn void ReflectMCOuterX2(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip)
//! \brief periodic boundary conditions, outer x2 boundary

void ReflectMCOuterX2(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip) {

  const Real frac = 1.e-8;
  pphot->k2p[ip] *= -1.;
  pphot->i2p[ip] = pmcb->je;
  pphot->x2p[ip] = pco->x1f(pphot->i2p[ip]+1)*(1.-frac) + pco->x2f(pphot->i2p[ip])*frac;

}

//----------------------------------------------------------------------------------------
//! \fn void ReflectMCInnerX3(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip)
//! \brief periodic boundary conditions, inner x3 boundary

void ReflectMCInnerX3(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip) {

  const Real frac = 1.e-8;
  pphot->k3p[ip] *= -1.;
  pphot->i3p[ip] = pmcb->ks;
  pphot->x3p[ip] = pco->x3f(pphot->i3p[ip])*(1.-frac) + pco->x3f(pphot->i3p[ip]+1)*frac;

}

//----------------------------------------------------------------------------------------
//! \fn void ReflectMCOuterX3(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip)
//! \brief periodic boundary conditions, outer x3 boundary

void ReflectMCOuterX3(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip) {

  const Real frac = 1.e-8;
  pphot->k3p[ip] *= -1.;
  pphot->i3p[ip] = pmcb->ke;
  pphot->x3p[ip] = pco->x3f(pphot->i3p[ip]+1)*(1.-frac) + pco->x3f(pphot->i3p[ip])*frac;

}

//----------------------------------------------------------------------------------------
//! \fn void Escape(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip)
//! \brief mark photon as escaped

void Escape(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip) {

  pphot->statp[ip] = ESCAPED;
}

//----------------------------------------------------------------------------------------
//! \fn void Absorb(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip)
//! \brief mark photon as absorbed

void Absorb(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip) {

  pphot->statp[ip] = ABSORBED;

}

//----------------------------------------------------------------------------------------
//! \fn void Destroy(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip)
//! \brief mark photon as destroyed

void Destroy(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip) {

  pphot->statp[ip] = DESTROYED;

}

//----------------------------------------------------------------------------------------
//! \fn void Polar(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip)
//! \brief mark photon as destroyed and print warning message

void Polar(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip) {

  pphot->PrintPhoton("Warning: photon moving through polar boundary, destroyed",ip);
  pphot->statp[ip] = DESTROYED;

}

//----------------------------------------------------------------------------------------
//! \fn void Periodic(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip)
//! \brief mark photon as being out of meshblock

void Periodic(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip) {

  // Do not send photon to another block if escaped, etc.
  if (pphot->statp[ip] == EVOLVING)
    pphot->statp[ip] = BUFFERED;

}
//----------------------------------------------------------------------------------------
//! \fn void Block(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip)
//! \brief mark photon as being out of meshblock

void Block(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip) {

  /* SWD: for debugging
  RegionSize& mesh_size = pphot->pmy_mcb->pmy_mc->pmy_mesh->mesh_size;
  Real x1 = pphot->x1p[ip];
  Real x2 = pphot->x2p[ip];
  Real x3 = pphot->x3p[ip];
  Real l1cgs, l2cgs = 1., l3cgs = 1.;
  l1cgs = pmcb->l_cgs;
  if ( (COORDINATE_SYSTEM == "cartesian") || (COORDINATE_SYSTEM == "minkowski") ) {
    l2cgs *= pmcb->l_cgs;
    l3cgs *= pmcb->l_cgs;
  }
  if (x1 < mesh_size.x1min * l1cgs) {
    pphot->PrintPhoton("bc x1 < x1min",ip);
  } else if (x1 > mesh_size.x1max * l1cgs) {
    pphot->PrintPhoton("bc x1 > x1max",ip);
  }

  if (x2 < mesh_size.x2min * l2cgs) {
    pphot->PrintPhoton("bc x2 < x2min",ip);
  } else if (x2 > mesh_size.x2max * l2cgs) {
    pphot->PrintPhoton("bc x2 > x2max",ip);
  }

  if (x3 < mesh_size.x3min * l3cgs) {
    pphot->PrintPhoton("bc x3 < x3min",ip);
  } else if (x3 > mesh_size.x3max * l3cgs) {
    pphot->PrintPhoton("bc x3 > x3max",ip);
  }
  */
  if (pphot->statp[ip] == EVOLVING)
    pphot->statp[ip] = BUFFERED;

}
