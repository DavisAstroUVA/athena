#ifndef KGREENS_HPP
#define KGREENS_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file kgreens.hpp
//! \brief quantile table of the Kompaneets Green's function, for the energy change and
//!        the free-free survival of a photon over a modified random walk step
//
// The table is written by vis/python/montecarlo/problem_specific/sphere_compton/
// kgreens_export.py from the npz that kompaneets_greens.py generates. For a photon of energy
// x_i = h nu / k T that accumulates Compton parameter y = theta tau_path in a medium
// whose free-free absorption parameter is lam = kappa_ff(x = 1) / (kappa_es theta),
// Quantile() draws the final energy x_f at cumulative probability p of the survivors
// and Survival() the fraction S that was not absorbed.

// C++ headers
#include <string>
#include <vector>

// Athena++ headers
#include "../athena.hpp"

class KompaneetsTable {
 public:
  KompaneetsTable() : nl(0), ny(0), nx(0), nq(0) {}

  //! read the flat binary; throws std::runtime_error on a bad file
  void Read(const std::string &filename);
  bool Loaded() const { return nq > 0; }

  //! final energy at cumulative probability p of the surviving photons
  Real Quantile(Real xi, Real y, Real lam, Real p) const;
  //! fraction not absorbed after y
  Real Survival(Real xi, Real y, Real lam) const;
  //! free-free absorption shape (1 - e^-x)/x^3, normalized to 1 at x = 1
  static Real FreeFreeShape(Real x) {
    return -std::expm1(-x)/(x*x*x)/(1. - std::exp(-1.));
  }
  //! inverse of the standard normal CDF
  static Real InverseNormal(Real p);

  // ranges, for the run report and the caller's guards
  Real LamMin() const { return nl > 1 ? lam[1] : 0.; }
  Real LamMax() const { return lam[nl-1]; }
  Real YMin() const { return y[0]; }
  Real YMax() const { return y[ny-1]; }
  Real XiMin() const { return xi[0]; }
  Real XiMax() const { return xi[nx-1]; }
  int nl, ny, nx, nq;

 private:
  std::vector<Real> lam, y, xi, levels;
  std::vector<Real> lnlam;   // ln lam[1:], the absorption axis
  std::vector<Real> lny, lnxi;
  std::vector<float> uq;      // (nl, ny, nx, nq) scaled quantiles ln(xf/xi)/sqrt(2y)
  std::vector<Real> lnabar;   // (nl, ny, nx) ln of -ln S / (lam y)
  std::vector<unsigned char> dead; // (nl, ny, nx) S < 1e-12

  Real UQ(int l, int j, int i, int k) const { return uq[((l*ny + j)*nx + i)*nq + k]; }
  Real LnAbar(int l, int j, int i) const { return lnabar[(l*ny + j)*nx + i]; }
  bool Dead(int l, int j, int i) const { return dead[(l*ny + j)*nx + i] != 0; }

  Real Abar(Real xi, Real y, Real lam) const;
  //! lower index and linear weight of v on a sorted grid, clamped
  static void AxisWeights(const std::vector<Real> &grid, Real v, int &i, Real &t);
  //! indices and 4-point Lagrange weights (2-point linear in the end intervals)
  static int CubicWeights(const std::vector<Real> &grid, Real v, int idx[4], Real w[4]);
};

#endif // KGREENS_HPP
