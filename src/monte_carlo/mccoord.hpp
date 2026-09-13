#ifndef MCCOORD_HPP
#define MCCOORD_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mccoord.hpp
//! \brief definitions for MCCoord base class and derived classes

// Athena++ classes headers
#include "../athena.hpp"

#define NCOORD 4

// The enums below are defined ahead of the montecarlo.hpp include deliberately.  That
// header includes this one and declares members of these types, so they have to be
// complete before it is processed.

//----------------------------------------------------------------------------------------
//! \brief which metric the Monte Carlo module is integrating on.
//!
//! COORDINATE_SYSTEM is not fully determinant. e.g. "gr_user" does not uniquely specify
//! the metric. MCCoordSystem names the metric outright; see
//! MonteCarlo::SetCoordinateSystem for how it is resolved from the configure-time
//! coordinate system plus <montecarlo>/mc_coord.

enum MCCoordSystem {
  MCCOORD_CARTESIAN = 0,
  MCCOORD_CYLINDRICAL = 1,
  MCCOORD_SPHERICAL_POLAR = 2,
  MCCOORD_MINKOWSKI = 3,
  MCCOORD_KERR_SCHILD = 4,
  MCCOORD_BOYER_LINDQUIST = 5,
  MCCOORD_KERR_SCHILD_CARTESIAN = 6,
  MCCOORD_SNAKE = 7
};

//----------------------------------------------------------------------------------------
//! \brief shape of the coordinate triple (x1,x2,x3), independent of the metric.
//!
//! The metric and the grid topology are separate concepts.
//! Kerr-Schild in Cartesian form is a curved metric on an (x,y,z) grid, so code that
//! orthonormalizes directions, converts wavevectors for output or bins escape angles has
//! to branch on this rather than on MCCoordSystem.

enum MCTopology {
  MCTOPO_CARTESIAN = 0,
  MCTOPO_CYLINDRICAL = 1,
  MCTOPO_SPHERICAL = 2
};

//! \brief grid topology implied by a given metric
MCTopology GetMCTopology(MCCoordSystem c);
//! \brief true when the metric is not flat in the coordinates being integrated
bool IsMCMetricCurved(MCCoordSystem c);
//! \brief human-readable name, used in error messages and for <montecarlo>/mc_coord
const char *GetMCCoordSystemName(MCCoordSystem c);
//! \brief true when the run integrates geodesics in a relativistic spacetime
bool IsMCRelativistic(MCCoordSystem c);
//! \brief true when GeneralPusher is used regardless of <montecarlo>/general_pusher
bool IsMCPusherAlwaysGeneral(MCCoordSystem c);
//! \brief true when the flat scale factors orthonormalize the coordinate basis
bool HasFlatOrthonormalBasis(MCCoordSystem c);

#include "montecarlo.hpp"

//----------------------------------------------------------------------------------------
//! \class MCCoord
//! \brief monte carlo specific coordinate value, base class

class MCCoord {
public:
  MCCoord(Coordinates *pcoord, MonteCarloBlock *pmcb);
  MCCoord(int ncells1, int ncells2, int ncells3, bool acc);
  ~MCCoord();

  bool computedmin;

  AthenaArray<Real> x1f, x2f, x3f; // face  positions
  AthenaArray<Real> vol;
  AthenaArray<Real> dmin;

  // For a general-relativistic run on the general pusher:
  //
  //   Metric                   LIVE.  RK4Step converts k^mu -> k_mu and back and
  //                            renormalizes to k.k = 0; also ComovingFrameMatrix and
  //                            PhotonFrames.
  //   InverseMetric            LIVE.  RK4Step, for dx^mu/dl = g^{mu nu} k_nu.
  //   InverseMetricDerivative  LIVE.  RK4Step, for dk_mu/dl = -1/2 d_mu g^{ab} k_a k_b.
  //                            This is the geodesic right-hand side.
  //   MetricDerivative         Not called by anything, in any coordinate system.
  //   Connect                  LIVE whenever polarization is tracked.  The geodesic itself
  //                            is Hamiltonian and uses metric derivatives, so this is
  //                            reached only through GeneralPusher::ConnectionContraction,
  //                            which builds A^i_k = Gamma^i_kl k^l for the coherency
  //                            tensor transport.  An unpolarized run never calls it.
  //                            Also used in VerletStep, whose call sites are commented
  //                            out in favor of RK4Step.
  //   Tetrad, InverseTetrad    Only the non-GR branches of TransformToComoving and
  //                            TransformToCoordinate, and the flat branch of
  //                            PhotonFrames::Fill (gr_tetrad_ is false there).  A GR run
  //                            goes through boost_lab/boost_cmv, built by ConstructTetrad
  //                            from the metric, and never reaches these.
  //
  // Derived classes should still implement them correctly for now. VerletStep may be revived
  // and the non-GR paths are live for cartesian and spherical_polar.

  virtual void Metric(Real x[4],Real gcov[4][4]);
  virtual void MetricDerivative(Real x[4],Real dgcov[4][4][4]);
  virtual void InverseMetric(Real x[4],Real gcon[4][4]);
  virtual void InverseMetricDerivative(Real x[4],Real dgcon[4][4][4]);
  virtual void Connect(Real x[4],Real gamma[4][4][4]);
  virtual void Tetrad(Real x[4], Real tetrad[4][4]);
  virtual void InverseTetrad(Real x[4], Real invtet[4][4]);

  Real GetMass() const {return bh_mass_;}
  Real GetSpin() const {return bh_spin_;}
  void SetMass(Real m) {bh_mass_ = m;}
  void SetSpin(Real a) {bh_spin_ = a;}

protected:
 // GR-specific variables
  Real bh_mass_;
  Real bh_spin_;

};

//----------------------------------------------------------------------------------------
//! \class MCCartesian
//! \brief derived class for Cartesian coordinates

class MCCartesian : public MCCoord {
public:
  MCCartesian(Coordinates *pcoord, MonteCarloBlock *pmcb);
  MCCartesian(int ncells1, int ncells2, int ncells3, bool acc);
  ~MCCartesian();
};

//----------------------------------------------------------------------------------------
//! \class MCSphericalPolar
//! \brief derived class for Spherical coordinates

class MCSphericalPolar : public MCCoord {
public:
  MCSphericalPolar(Coordinates *pcoord, MonteCarloBlock *pmcb);
  MCSphericalPolar(int ncells1, int ncells2, int ncells3, bool acc);
  ~MCSphericalPolar();

  // functions
  void Metric(Real x[4], Real gcov[4][4]);
  void InverseMetric(Real x[4], Real gcon[4][4]);
  void Connect(Real x[4], Real gamma[4][4][4]);
  void InverseMetricDerivative(Real x[4], Real dgcon[4][4][4]);
  void Tetrad(Real x[4], Real tetrad[4][4]);
  void InverseTetrad(Real x[4], Real invtet[4][4]);

};

//----------------------------------------------------------------------------------------
//! \class MCCylindrical
//! \brief derived class for Cylindrical coordinates

class MCCylindrical : public MCCoord {
public:
  MCCylindrical(Coordinates *pcoord, MonteCarloBlock *pmcb);
  MCCylindrical(int ncells1, int ncells2, int ncells3, bool acc);
  ~MCCylindrical();

  // functions
  void Metric(Real x[4], Real gcov[4][4]);
  void Connect(Real x[4], Real gamma[4][4][4]);

};

//----------------------------------------------------------------------------------------
//! \class MCKerrSchild
//! \brief derived class for Kerr-Schild coordinates

class MCKerrSchild: public MCCoord {
public:
  MCKerrSchild(Coordinates *pcoord, MonteCarloBlock *pmcb);
  MCKerrSchild(int ncells1, int ncells2, int ncells3, bool acc);
  ~MCKerrSchild();

  // functions
  void Metric(Real x[4], Real gcov[4][4]);
  void InverseMetric(Real x[4], Real gcov[4][4]);
  void Connect(Real x[4], Real gamma[4][4][4]);
  void InverseMetricDerivative(Real x[4], Real dgcon[4][4][4]);

};

//----------------------------------------------------------------------------------------
//! \class MCKerrSchildCartesian
//! \brief derived class for cartesian Kerr-Schild coordinates

class MCKerrSchildCartesian: public MCCoord {
public:
  MCKerrSchildCartesian(Coordinates *pcoord, MonteCarloBlock *pmcb);
  MCKerrSchildCartesian(int ncells1, int ncells2, int ncells3, bool acc);
  ~MCKerrSchildCartesian();

  // functions
  void Metric(Real x[4], Real gcov[4][4]);
  void InverseMetric(Real x[4], Real gcov[4][4]);
  void Connect(Real x[4], Real gamma[4][4][4]);
  void InverseMetricDerivative(Real x[4], Real dgcon[4][4][4]);

};

//----------------------------------------------------------------------------------------
//! \class MCBoyerLindquist
//! \brief derived class for Boyer-Lindquist coordinates

class MCBoyerLindquist : public MCCoord {
public:
  MCBoyerLindquist(Coordinates *pcoord, MonteCarloBlock *pmcb);
  MCBoyerLindquist(int ncells1, int ncells2, int ncells3, bool acc);
  ~MCBoyerLindquist();

  // functions
  void Metric(Real x[4], Real gcov[4][4]);
  void InverseMetric(Real x[4], Real gcov[4][4]);
  void Connect(Real x[4], Real gamma[4][4][4]);

};

//----------------------------------------------------------------------------------------
//! \class MCSnake
//! \brief derived class for sinusoidal ("snake") coordinates
//!
//! Flat spacetime written in the sheared coordinates of White, Stone & Gammie (2016),
//! ApJS 225, 22: y = y_M + a sin(k x_M) with the other three unchanged.  Writing
//! beta = a k cos(k x),
//!
//!   g_tt = -1, g_xx = 1 + beta^2, g_xy = g_yx = -beta, g_yy = g_zz = 1
//!
//! Three properties make it a good test geometry.  sqrt(-g) = 1 exactly, so cell volumes
//! match the Cartesian ones and volume normalization drops out of any comparison.  The
//! lapse is unity, so the normal observer is the coordinate-time observer and the lab and
//! coordinate moment bases must agree exactly.  And exactly one Christoffel symbol is
//! non-zero, Gamma^y_xx = a k^2 sin(k x), with geodesics that are straight lines in the
//! underlying Minkowski coordinates -- a closed-form answer to check the integrator
//! against, which Kerr-Schild cannot provide.
//!
//! It is also the only supported metric whose coordinate basis is not orthogonal
//! (g_xy != 0), so it is the first thing to exercise the off-diagonal Tetrad path.

class MCSnake : public MCCoord {
public:
  MCSnake(Coordinates *pcoord, MonteCarloBlock *pmcb);
  MCSnake(int ncells1, int ncells2, int ncells3, bool acc);
  ~MCSnake();

  // functions
  void Metric(Real x[4], Real gcov[4][4]);
  void InverseMetric(Real x[4], Real gcon[4][4]);
  void MetricDerivative(Real x[4], Real dgcov[4][4][4]);
  void InverseMetricDerivative(Real x[4], Real dgcon[4][4][4]);
  void Connect(Real x[4], Real gamma[4][4][4]);
  void Tetrad(Real x[4], Real tetrad[4][4]);
  void InverseTetrad(Real x[4], Real invtet[4][4]);

  //! amplitude and wavenumber of the shear; <coord>/snake_a and <coord>/snake_k.
  //! Deliberately not <coord>/a: gr_user already requires that name for the black hole
  //! spin, which GRUser reads unconditionally.
  void SetSnakeParams(Real a, Real k) {snake_a_ = a; snake_k_ = k;}
  Real GetSnakeAmplitude() const {return snake_a_;}
  Real GetSnakeWavenumber() const {return snake_k_;}

protected:
  Real snake_a_;
  Real snake_k_;

};

//----------------------------------------------------------------------------------------
//! \class MCMinkowski
//! \brief derived class for Minkowski coordinates

class MCMinkowski : public MCCoord {
public:
  MCMinkowski(Coordinates *pcoord, MonteCarloBlock *pmcb);
  MCMinkowski(int ncells1, int ncells2, int ncells3, bool acc);
  ~MCMinkowski();
};

#endif // MCCOORD_HPP
