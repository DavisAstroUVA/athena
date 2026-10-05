#ifndef PHOTONPUSHER_HPP
#define PHOTONPUSHER_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file photonpusher.hpp
//! \brief defines abstract base derived classes for moving photons

// C++ headers
#include <cmath>
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../mesh/mesh.hpp"
#include "../coordinates/coordinates.hpp"
#include "photon.hpp"
#include "montecarlo.hpp"
#include "mcutils.hpp"

class MeshBlock;
class ParameterInput;
class MonteCarloBlock;
class MRWTables;

//! \struct MRWCellFrame
//! \brief the photon's cell as the random-walk step sees it: a box in the lab
//!        (Eulerian) frame, with the fluid velocity relative to that frame
struct MRWCellFrame {
  Real W[3];         // widths along the box axes, code length
  Real x[3];         // the photon's offsets from the lower faces, code length
  Real beta[3];      // fluid velocity along the box axes, units of c
  Real beta_tet[3];  // the same on the lab tetrad legs (equal for the legacy pushers)
  Real gam;          // its Lorentz factor
  Real ehat[3][3];   // box axis i on the lab tetrad legs a: ehat[i][a]
  Real econ[4][4];   // lab tetrad legs in coordinate components (general pusher)
  bool general;      // built by the general pusher: displacements go through econ
};
class Photon;
class Coordinate;

//---------------------- prototypes for photon moving ------------------------------------
Real GetOpticalDepth(MCRandom *pran);

//----------------------------------------------------------------------------------------
//! \class PhotonPusher
//! \brief abstract base class for all derived classes

class PhotonPusher {
public:
  PhotonPusher(MonteCarloBlock *pmcb);
  ~PhotonPusher();
  // data

  Real dl; // current displacement
  //! bound on the number of steps in a single free flight, accumulated in Photon::nmvp
  int capmove;

  MonteCarlo *pmy_mc;
  MonteCarloBlock *pmy_mcb;
  MCCoord *pcoord;

  // function pointers
  UserMoveFunc_t UserWorkInMove;

  const MRWTables *mrw_; // MRW tables (see mrw.hpp)
  Real mrw_s_pmax_; // sphere time at MRW fraction is accel_pmax

  bool acceleration;
  bool boosts;
  bool resonance;
  bool compton;

  // Path stretching (see MonteCarlo::stretch); stretching_ is false when it is off
  bool stretching_;
  Real stretch_, stretch_lnbound_, stretch_taucell_;

  // functions
  virtual void Move(Photon *pphot, int ips, int ipe);
  virtual Real GetOpticalDepth(MCRandom *pran);
  //! compute or apply strech factor
  Real StretchFactor(Photon *pphot, int ip, Real chi, Real dl_seg);
  void ApplyStretch(Photon *pphot, int ip, Real alpha, Real tau) {
    if (alpha != 1.) {
      const Real d = (alpha - 1.) * tau;
      pphot->strp[ip] += d;
      pphot->wp[ip] *= std::exp(d);
    }
  }
  //! smallest width of the photon's current cell, in code length
  virtual Real CellWidth(Photon *pphot, int ip);
  virtual Real GetExtinctionCoefficient(Real ac, Real sc, bool abs_tau);
  virtual Real ExpTauAbsorption(Real ac, Real dl, bool abs_tau);
  virtual void NextFace(Real dx1, Real dx2, Real dx3, int &face, Real &dx);
  virtual void MovePhotonToNextZone(Photon *pphot, MCCoord *pco,
               MonteCarloBlock *pmcb, int face, bool ascend[3], int ip);
  virtual bool UpdateZone(Photon *pphot, int ip);
  virtual bool IsOnBlock(Photon *pphot, int ip);
//  virtual bool UpdateSingleZone(Photon *pphot, int ip, bool *multizone);
  // Modified random walk (mrw.cpp)
  //! the photon's cell as a box in the local orthonormal basis: widths and the photon's
  //! distances from the lower faces, code length
  void CellGeometry(Photon *pphot, int ip, Real W[3], Real x[3]);
  //! distance to the nearest face of the photon's cell, in code length
  Real FaceDistance(Photon *pphot, int ip);
  //! whether a step is to be tried here; chi the lab extinction in cm^-1
  bool MRWTrigger(Photon *pphot, int ip, Real chi);
  //! the cell's box and the fluid velocity in the lab frame; false when there is no
  //! usable frame (inside a horizon, say)
  virtual bool MRWFrame(Photon *pphot, int ip, MRWCellFrame &fr);
  //! one step; false when it declines and leaves the photon untouched
  bool MRWStep(Photon *pphot, MCRandom *pran, int ip);
  // Resonant-scattering acceleration (general pusher)
  virtual Real SampleEscapeTime(MCRandom *pran, Real decayRate, Real sphereRadius,
                                Real diffusionTime);
  virtual void MRWResonanceAcceleration(Photon *pphot, MCRandom *pran, Real dist,
                               Real tauacc, Real &path_length, Real &k1, Real &k2,
                               Real &k3, int ip);

};

//----------------------------------------------------------------------------------------
//! \class CartesianPusher
//! \brief derived class for moving in Cartesian coordinates

class CartesianPusher : public PhotonPusher {
public:
  CartesianPusher(MonteCarloBlock *pmcb);
  ~CartesianPusher();

  // functions
  void Move(Photon *pphot, int ips, int ipe);

};

//----------------------------------------------------------------------------------------
//! \class SphericalPolarPusher
//! \brief derived class for moving in spherical-polar coordinates

class SphericalPolarPusher : public PhotonPusher {
public:
  SphericalPolarPusher(MonteCarloBlock *pmcb);
  ~SphericalPolarPusher();

  // functions
  void Move(Photon *pphot, int ips, int ipe);
  Real CellWidth(Photon *pphot, int ip);

};

//----------------------------------------------------------------------------------------
//! \class GeneralPusher
//! \brief derived class for moving in general coordinates

class GeneralPusher : public PhotonPusher {
public:
  GeneralPusher(MonteCarloBlock *pmcb);
  ~GeneralPusher();

  Real step_par;
  Real gamma[NCOORD][NCOORD][NCOORD];

  // functions
  void Move(Photon *pphot, int ips, int ipe);
  void UpdateOpacities(Photon *pphot, MonteCarloBlock *pmcb, int ip);
  //! the cell in the normal observer's tetrad at the photon, with the fluid's velocity
  //! relative to that observer (mrw.cpp)
  bool MRWFrame(Photon *pphot, int ip, MRWCellFrame &fr);
#if MC_VERLET_DK
  void VerletStep(Photon *pphot, Real step, int ip);
#endif
  void RK4Step(Photon *pphot, Real step, int ip);
  void SubStep(Real xcon[4], Real kcov[4], Real dl[8]);
  void AdvanceStep(Photon *pphot, Real step, int ip);
  void ConnectionContraction(Photon *pphot, int ip, Real acon[4][4]);
  void ApplyPolarizationRate(const Real acon[4][4], const std::complex<Real> nin[4][4],
                             std::complex<Real> dndl[4][4]);

  // A^i_k = Gamma^i_kl k^l carried from one step's corrector into the next step's
  // predictor, and a flag saying whether the photon is still where it was computed.
  Real acon[4][4];
  bool acon_valid;
  Real StepSize(Photon *pphot, int ip);

  // The metric pair at the point the last RK4 step ended
  Real metric_x[4];
  Real metric_gcov[4][4];
  Real metric_gcon[4][4];
  bool metric_valid;
  //! g_{mu nu} and g^{mu nu} at x, from the cache when x is the cached point
  void MetricPairAt(Real x[4], Real gcov[4][4], Real gcon[4][4]);
  //! whether the coherency tensor is transported; constant for the run, read every step
  bool polarized_;

};

#endif // PHOTONPUSHER_HPP
