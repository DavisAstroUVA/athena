#ifndef PHOTONPUSHER_HPP
#define PHOTONPUSHER_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file photonpusher.hpp
//! \brief defines abstract base derived classes for moving photons

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
  //! across every Move call and every block the photon crosses between scatterings.  It
  //! catches the flight that never ends -- a near-horizontal photon in an optically thin
  //! domain with periodic sides -- while leaving a photon that simply scatters often
  //! alone.  0 disables the cap.
  int capmove;

  MonteCarlo *pmy_mc;
  MonteCarloBlock *pmy_mcb;
  MCCoord *pcoord;

  // function pointers
  UserMoveFunc_t UserWorkInMove;
  UserEscapeDistanceFunc_t UserEscapeDistance;

  // Arrays for MRW acceleration
  AthenaArray<Real> mrwprob;
  AthenaArray<Real> mrwdev;
  AthenaArray<Real> mrwxf;
  AthenaArray<Real> mrwt;
  AthenaArray<Real> mrwxi;
  AthenaArray<Real> mrwp;
  AthenaArray<Real> mrwrt;
  AthenaArray<Real> mrwrp;
  AthenaArray<Real> mrwrr;
  AthenaArray<Real> mrwta;
  AthenaArray<Real> mrwtp;
  AthenaArray<Real> mrwtt;

  int nmax,nxi,np,nt; //used for acceleration arrays

  bool acceleration;
  bool boosts;
  bool resonance;
  bool compton;
  bool time_acc;

  // functions
  virtual void Move(Photon *pphot, int ips, int ipe);
  virtual Real GetOpticalDepth(MCRandom *pran);
  virtual Real GetExtinctionCoefficient(Real ac, Real sc, bool abs_tau);
  virtual Real ExpTauAbsorption(Real ac, Real dl, bool abs_tau);
  virtual void NextFace(Real dx1, Real dx2, Real dx3, int &face, Real &dx);
  virtual void MovePhotonToNextZone(Photon *pphot, MCCoord *pco,
               MonteCarloBlock *pmcb, int face, bool ascend[3], int ip);
  virtual bool UpdateZone(Photon *pphot, int ip);
  virtual bool IsOnBlock(Photon *pphot, int ip);
//  virtual bool UpdateSingleZone(Photon *pphot, int ip, bool *multizone);
  virtual void InitializeMRWDist(void);
  // Acceleration methods
  virtual Real SampleEscapeTime(MCRandom *pran, Real decayRate, Real sphereRadius,
                                Real diffusionTime);
  virtual bool MRWAcceleration(Photon *pphot, MCRandom *pran, Real dist, Real tauacc,
                               int ip);
  virtual void MRWResonanceAcceleration(Photon *pphot, MCRandom *pran, Real dist,
                               Real tauacc, Real &path_length, Real &k1, Real &k2,
                               Real &k3, int ip);
  virtual Real MRWDist(MCRandom *pran);
  virtual void ReadComptonGreensFunction(void);
  virtual Real InterpComptonEnergy(Real x0, Real time, Real prob);
  virtual void ReadRadiusDistribution(void);
  virtual void ReadTimeDistribution(void);
  virtual Real InterpPathTime(Real tau, Real prob);

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
