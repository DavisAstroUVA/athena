#ifndef MONTE_CARLO_PHOTON_FRAMES_HPP_
#define MONTE_CARLO_PHOTON_FRAMES_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file photon_frames.hpp
//! \brief the frames a photon can be expressed in, and the projection into them.
//!
//! This is the layer that applies functions in tetrad.hpp to Monte Carloquantities.
//! Nothing in tetrad.hpp knows about photons or mesh blocks; everything here
//! does.  Implemented in photon_frames.cpp.
//!
//! MCFrame and PhotonFrameState are part of the user-facing interface: a user moment
//! function declares the frame it wants at enrollment and receives a PhotonFrameState
//! already projected into it.  PhotonFrames itself is internal so UpdateMoments builds one
//! per call and the caching allows several user moments to sharing a frame

// C++ headers
#include <cmath>    // isnan, isinf

// Athena++ headers
#include "../athena.hpp"

class MonteCarloBlock;
class Photon;

//----------------------------------------------------------------------------------------
//! \brief the comoving orthonormal frame at a photon: the one definition
//!
//! Used in TransformToComoving/TransformToCoordinate to tranform photon to/from the
//! scattering basis. Built by Gram-Schmidt from the fluid four-velocity against the
//! coordinate metric. In GR the four-velocity comes from FluidFourVelocity, in
//! flat spacetime it is vel carried onto coordinate components by the static tetrad. 
//!
//! econ and ecov are indexed [tetrad leg][coordinate component], for CoordinateToTetrad,
//! TetradToCoordinate, PolarizationToTetrad and PolarizationToCoord.
void ComovingFrame(MonteCarloBlock *pmcb, Photon *pphot, int ip,
                   Real econ[4][4], Real ecov[4][4]);

//! \brief the frames radiation moments can be reported in.
//!
//! MCFRAME_LAB and MCFRAME_COMOVING are orthonormal: in general relativity they are the
//! tetrads of the normal (Eulerian) observer and of the frame vel is built on, and in flat
//! spacetime they are the Eulerian frame and its Lorentz boost.  MCFRAME_COORD is the raw
//! coordinate basis, which is not orthonormal -- its components are dimensionally
//! inhomogeneous by construction, which is what makes them the right form to hand to the
//! GRMHD conservative variables.
enum MCFrame {MCFRAME_LAB = 0, MCFRAME_COMOVING = 1, MCFRAME_COORD = 2, MCFRAME_N = 3};

//----------------------------------------------------------------------------------------
//! \struct PhotonFrameState
//! \brief a photon's energy, propagation direction and path length in one frame
//!
//! Photon pushers pass UpdateMoments a coordinate path length, and the length in any other
//! frame differs by that frame's energy over ep. n is a unit vector in the orthonormal frames;
//  in MCFRAME_COORD it holds k^i/k^0 and is not normalized.

struct PhotonFrameState {
  Real e;     // photon energy in this frame
  Real n[3];  // propagation direction in this frame
  Real dl;    // path length in this frame

  //! guard against a photon whose projection has gone bad
  bool Finite(Real wp) const {
    Real weight = wp * e * dl;
    return !(std::isinf(weight) || std::isnan(weight)
             || std::isnan(n[0]) || std::isinf(n[0])
             || std::isnan(n[1]) || std::isinf(n[1])
             || std::isnan(n[2]) || std::isinf(n[2]));
  }
};


//----------------------------------------------------------------------------------------
//! \class PhotonFrames
//! \brief projects one photon into whichever frames are asked for, at most once each.
//!
//! Caching per call also means a run with several comoving user moments pays for one
//! projection rather than one per function.

class PhotonFrames {
 public:
  PhotonFrames(MonteCarloBlock *pmcb, Photon *pphot, int ip, Real dl);

  // coordinate basis needs a coordinate four-vector, which only the general pusher
  // stores; the legacy pushers keep a unit direction in the local orthonormal basis.
  bool Available(MCFrame f) const { return (f != MCFRAME_COORD) || general_; }
  bool GRTetrad() const { return gr_tetrad_; }
  const Real *Coordinate4Vector() const { return kco_; }

  const PhotonFrameState &Get(MCFrame f) {
    if (!done_[f]) { Fill(f); done_[f] = true; }
    return st_[f];
  }

 private:
  void Fill(MCFrame f);

  // Re-express a unit direction, given in the local orthonormal basis at the photon, in
  // the orthonormal basis at the photon's cell center, the basis where the moments are
  // accumulated. A no-op for a Cartesian topology, where the basis does not depend on
  // position.
  void ToCellCenterBasis(Real n[3]) const;

  MonteCarloBlock *pmcb_;
  Photon *pphot_;
  int ip_;
  Real dl_;        // coordinate path length, as handed in by the pusher
  Real kco_[4];    // coordinate four-vector; general pusher only
  bool general_, gr_tetrad_;
  bool done_[MCFRAME_N];
  PhotonFrameState st_[MCFRAME_N];
};

#endif // MONTE_CARLO_PHOTON_FRAMES_HPP_
