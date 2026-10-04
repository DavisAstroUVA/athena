#ifndef MRW_HPP
#define MRW_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mrw.hpp
//! \brief the tables of the modified random walk, built once per rank and shared by the
//!        pushers of every block (mrw.cpp)
//
// Two diffusion domains, both dimensionless so that one table serves every cell:
//
//   sphere   a point source at the centre of an absorbing sphere of radius R_0, in the
//            time s = pi^2 D c t / R_0^2: the first-passage CDF F(s) = 1 - P(s),
//            P(s) = 2 sum (-1)^(n+1) exp(-n^2 s), and the radius of a walker still
//            inside at s, p(rho | s) ~ rho sum n sin(n pi rho) exp(-n^2 s).
//
//   box      one dimension of a rectangular cell with absorbing walls at 0 and W, start
//            xi = x_0 / W, in the time sigma = pi^2 D c t / W^2: the survival
//            S(xi, sigma) = sum_{odd n} (4 / n pi) sin(n pi xi) exp(-n^2 sigma), the
//            hazard g = -d ln S / d sigma, the probability that the exit, when it comes,
//            is through the wall at 0, and the quantiles of the position of a walker
//            still inside.  A cell is the product of three of these.
//
// Below the sigma grid the image solution in erf is used, above it the single surviving
// mode; the sphere's radius below its grid is the free-space Gaussian.

// C++ headers
#include <vector>

// Athena++ headers
#include "../athena.hpp"

class MCRandom;

class MRWTables {
 public:
  MRWTables();

  // --- sphere
  //! fraction of walkers still inside the sphere at s
  static Real SphereInside(Real s);
  //! first-passage time s at CDF value u
  Real SphereFirstPassage(Real u) const;
  //! r / R_0 of a walker still inside at s
  Real SphereRadius(Real s, MCRandom *pran) const;

  // --- box, one dimension; xi may be anywhere in [0, 1]
  //! ln of the survival
  Real BoxLnSurvival(Real xi, Real sigma) const;
  //! hazard -d ln S / d sigma
  Real BoxHazard(Real xi, Real sigma) const;
  //! probability that the exit is through the wall at 0
  Real BoxLeftExit(Real xi, Real sigma) const;
  //! position / W of a walker still inside at sigma, at cumulative probability p
  Real BoxPosition(Real xi, Real sigma, Real p) const;

  Real sigma_min, sigma_max;

 private:
  // sphere
  std::vector<Real> fp_lns_, fp_cdf_;
  std::vector<Real> rad_lns_, rad_q_;
  int nrad_s_, nrad_q_;
  // box: grids in xi (0 to 1/2, the rest by symmetry) and ln sigma
  std::vector<Real> xi_, lnsig_;
  std::vector<Real> lns_, haz_, pleft_;   // (nxi, nsig)
  std::vector<float> pos_;                // (nxi, nsig, nlev), position / W at the levels
  int nxi_, nsig_, nlev_;
  Real dxi_, dlnsig_;

  void BuildSphere();
  void BuildBox();
  //! bilinear weights on the (xi, ln sigma) grid, xi already folded to [0, 1/2]
  void BoxWeights(Real xi, Real sigma, int &i, Real &a, int &j, Real &b) const;
  Real Lns(int i, int j) const { return lns_[i*nsig_ + j]; }
  Real Haz(int i, int j) const { return haz_[i*nsig_ + j]; }
  Real Pleft(int i, int j) const { return pleft_[i*nsig_ + j]; }
  Real Pos(int i, int j, int k) const { return pos_[(i*nsig_ + j)*nlev_ + k]; }
};

#endif // MRW_HPP
