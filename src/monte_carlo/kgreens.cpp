//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file kgreens.cpp
//! \brief the Kompaneets Green's function quantile table: reader and sampler

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Athena++ headers
#include "kgreens.hpp"

namespace {

template <typename T>
void ReadArray(std::ifstream &f, std::vector<T> &v, std::size_t n, const char *name,
               const std::string &filename) {
  v.resize(n);
  f.read(reinterpret_cast<char *>(v.data()), static_cast<std::streamsize>(n*sizeof(T)));
  if (!f) {
    std::stringstream msg;
    msg << "### FATAL ERROR in KompaneetsTable::Read" << std::endl
        << filename << ": short read in " << name << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn void KompaneetsTable::Read(const std::string &filename)
//! \brief read the flat little-endian binary written by kgreens_export.py

void KompaneetsTable::Read(const std::string &filename) {
  std::ifstream f(filename.c_str(), std::ios::binary);
  if (!f) {
    std::stringstream msg;
    msg << "### FATAL ERROR in KompaneetsTable::Read" << std::endl
        << "cannot open " << filename << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  char magic[8];
  f.read(magic, 8);
  if (!f || std::memcmp(magic, "KGREENS1", 8) != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in KompaneetsTable::Read" << std::endl
        << filename << " is not a KGREENS1 table" << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  std::int32_t dims[4];
  f.read(reinterpret_cast<char *>(dims), 16);
  if (!f || dims[0] < 1 || dims[1] < 2 || dims[2] < 2 || dims[3] < 2) {
    std::stringstream msg;
    msg << "### FATAL ERROR in KompaneetsTable::Read" << std::endl
        << filename << ": bad dimensions" << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  nl = dims[0]; ny = dims[1]; nx = dims[2]; nq = dims[3];
  // the file holds float64, which is Real in this build
  std::vector<double> d;
  ReadArray(f, d, nl, "lam", filename);  lam.assign(d.begin(), d.end());
  ReadArray(f, d, ny, "y", filename);    y.assign(d.begin(), d.end());
  ReadArray(f, d, nx, "xi", filename);   xi.assign(d.begin(), d.end());
  ReadArray(f, d, nq, "levels", filename); levels.assign(d.begin(), d.end());
  ReadArray(f, uq, static_cast<std::size_t>(nl)*ny*nx*nq, "uq", filename);
  ReadArray(f, d, static_cast<std::size_t>(nl)*ny*nx, "abar", filename);
  lnabar.resize(d.size());
  for (std::size_t n=0; n<d.size(); ++n) lnabar[n] = std::log(std::max(d[n], 1.e-300));
  ReadArray(f, dead, static_cast<std::size_t>(nl)*ny*nx, "dead", filename);
  char extra;
  if (f.read(&extra, 1)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in KompaneetsTable::Read" << std::endl
        << filename << ": trailing bytes; dimensions and contents disagree" << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  lny.resize(ny);
  for (int j=0; j<ny; ++j) lny[j] = std::log(y[j]);
  lnxi.resize(nx);
  for (int i=0; i<nx; ++i) lnxi[i] = std::log(xi[i]);
  lnlam.clear();
  for (int l=1; l<nl; ++l) lnlam.push_back(std::log(lam[l]));
}

//----------------------------------------------------------------------------------------
//! \fn void KompaneetsTable::AxisWeights(...)
//! \brief lower index and linear weight of v on a sorted grid, clamped to the grid

void KompaneetsTable::AxisWeights(const std::vector<Real> &grid, Real v, int &i, Real &t) {
  const int n = static_cast<int>(grid.size());
  v = std::min(std::max(v, grid[0]), grid[n-1]);
  // first index with grid[k] > v, minus one, clamped to [0, n-2]
  int k = static_cast<int>(std::upper_bound(grid.begin(), grid.end(), v) - grid.begin()) - 1;
  i = std::min(std::max(k, 0), n-2);
  t = (v - grid[i])/(grid[i+1] - grid[i]);
}

//----------------------------------------------------------------------------------------
//! \fn int KompaneetsTable::CubicWeights(...)
//! \brief 4-point Lagrange weights on the nodes around v, linear in the end intervals

int KompaneetsTable::CubicWeights(const std::vector<Real> &grid, Real v, int idx[4],
                                  Real w[4]) {
  const int n = static_cast<int>(grid.size());
  int i;
  Real t;
  AxisWeights(grid, v, i, t);
  if (i == 0 || i >= n-2) {
    idx[0] = i; idx[1] = i+1;
    w[0] = 1.-t; w[1] = t;
    return 2;
  }
  v = std::min(std::max(v, grid[0]), grid[n-1]);
  for (int k=0; k<4; ++k) {
    idx[k] = i-1+k;
    Real wk = 1.;
    for (int m=0; m<4; ++m) {
      if (m != k) wk *= (v - grid[i-1+m])/(grid[i-1+k] - grid[i-1+m]);
    }
    w[k] = wk;
  }
  return 4;
}

//----------------------------------------------------------------------------------------
//! \fn Real KompaneetsTable::InverseNormal(Real p)
//! \brief inverse of the standard normal CDF: Acklam's rational approximation refined by
//!        one Halley step on erfc, good to double precision

Real KompaneetsTable::InverseNormal(Real p) {
  static const Real a[6] = {-3.969683028665376e+01, 2.209460984245205e+02,
                            -2.759285104469687e+02, 1.383577518672690e+02,
                            -3.066479806614716e+01, 2.506628277459239e+00};
  static const Real b[5] = {-5.447609879822406e+01, 1.615858368580409e+02,
                            -1.556989798598866e+02, 6.680131188771972e+01,
                            -1.328068155288572e+01};
  static const Real c[6] = {-7.784894002430293e-03, -3.223964580411365e-01,
                            -2.400758277161838e+00, -2.549732539343734e+00,
                            4.374664141464968e+00, 2.938163982698783e+00};
  static const Real d[4] = {7.784695709041462e-03, 3.224671290700398e-01,
                            2.445134137142996e+00, 3.754408661907416e+00};
  p = std::min(std::max(p, 1.e-300), 1. - 1.e-16);
  const Real plow = 0.02425, phigh = 1. - plow;
  Real x;
  if (p < plow) {
    Real q = std::sqrt(-2.*std::log(p));
    x = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        / ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.);
  } else if (p <= phigh) {
    Real q = p - 0.5, r = q*q;
    x = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q
        / (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.);
  } else {
    Real q = std::sqrt(-2.*std::log(1. - p));
    x = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        / ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.);
  }
  // one Halley refinement
  const Real e = 0.5*std::erfc(-x/std::sqrt(2.)) - p;
  const Real u = e*std::sqrt(2.*PI)*std::exp(0.5*x*x);
  x -= u/(1. + 0.5*x*u);
  return x;
}

//----------------------------------------------------------------------------------------
//! \fn Real KompaneetsTable::Quantile(Real xi_in, Real y_in, Real lam_in, Real p)
//! \brief final energy at cumulative probability p (format document, Section 4.2)

Real KompaneetsTable::Quantile(Real xi_in, Real y_in, Real lam_in, Real p) const {
  // Case A: below the table, the small-y Gaussian in ln x, drift 3 - x, variance 2y
  if (y_in < y[0]) {
    return xi_in*std::exp((3. - xi_in)*y_in + std::sqrt(2.*y_in)*InverseNormal(p));
  }
  // Case B; above the table the y_max shape is reused, and ye goes into the rescaling
  const Real ye = std::min(y_in, y[ny-1]);
  int iy, ix;
  Real a, b;
  AxisWeights(lny, std::log(ye), iy, a);
  AxisWeights(lnxi, std::log(xi_in), ix, b);
  // level interval
  p = std::min(std::max(p, 0.), 1.);
  int k = static_cast<int>(std::upper_bound(levels.begin(), levels.end(), p)
                           - levels.begin()) - 1;
  k = std::min(std::max(k, 0), nq-2);
  const Real c = (p - levels[k])/(levels[k+1] - levels[k]);
  // absorption slices
  int ls[2];
  Real ws[2];
  int nls;
  if (lam_in <= 0. || nl == 1) {
    nls = 1; ls[0] = 0; ws[0] = 1.;
  } else if (lam_in <= lam[1]) {
    nls = 2; ls[0] = 0; ls[1] = 1;
    ws[1] = lam_in/lam[1]; ws[0] = 1. - ws[1];
  } else {
    int kl;
    Real bl;
    AxisWeights(lnlam, std::log(lam_in), kl, bl);
    nls = 2; ls[0] = kl+1; ls[1] = kl+2; ws[0] = 1. - bl; ws[1] = bl;
  }
  const Real wc[4] = {(1.-a)*(1.-b), a*(1.-b), (1.-a)*b, a*b};
  const int jy[4] = {iy, iy+1, iy, iy+1};
  const int jx[4] = {ix, ix, ix+1, ix+1};
  Real u = 0.;
  for (int s=0; s<nls; ++s) {
    for (int m=0; m<4; ++m) {
      u += ws[s]*wc[m]*((1.-c)*UQ(ls[s], jy[m], jx[m], k) + c*UQ(ls[s], jy[m], jx[m], k+1));
    }
  }
  return xi_in*std::exp(u*std::sqrt(2.*ye));
}

//----------------------------------------------------------------------------------------
//! \fn Real KompaneetsTable::Abar(Real xi_in, Real y_in, Real lam_in)
//! \brief mean absorption -ln S/(lam y): cubic in ln lam and ln xi, linear in ln y, with
//!        the linear fallback next to fully absorbed cells (format document, 4.3)

Real KompaneetsTable::Abar(Real xi_in, Real y_in, Real lam_in) const {
  const Real ll = std::log(std::max(lam_in, lam[1]));
  const Real lx = std::log(xi_in);
  int il[4], ix[4];
  Real wl[4], wx[4];
  int nll = CubicWeights(lnlam, ll, il, wl);
  int nxx = CubicWeights(lnxi, lx, ix, wx);
  int iy;
  Real a;
  AxisWeights(lny, std::log(y_in), iy, a);
  bool anydead = false;
  for (int s=0; s<nll && !anydead; ++s)
    for (int m=0; m<nxx && !anydead; ++m)
      anydead = Dead(il[s]+1, iy, ix[m]) || Dead(il[s]+1, iy+1, ix[m]);
  if (anydead) {
    Real t;
    AxisWeights(lnlam, ll, il[0], t); il[1] = il[0]+1; wl[0] = 1.-t; wl[1] = t; nll = 2;
    AxisWeights(lnxi, lx, ix[0], t);  ix[1] = ix[0]+1; wx[0] = 1.-t; wx[1] = t; nxx = 2;
  }
  Real out = 0.;
  for (int s=0; s<nll; ++s) {
    for (int m=0; m<nxx; ++m) {
      out += wl[s]*wx[m]*((1.-a)*LnAbar(il[s]+1, iy, ix[m]) + a*LnAbar(il[s]+1, iy+1, ix[m]));
    }
  }
  return std::exp(out);
}

//----------------------------------------------------------------------------------------
//! \fn Real KompaneetsTable::Survival(Real xi_in, Real y_in, Real lam_in)
//! \brief fraction of photons not absorbed after y (format document, Section 4.3)

Real KompaneetsTable::Survival(Real xi_in, Real y_in, Real lam_in) const {
  if (lam_in <= 0. || nl == 1) return 1.;
  if (y_in < y[0]) return std::exp(-lam_in*FreeFreeShape(xi_in)*y_in);
  if (y_in > y[ny-1]) {
    // ln S is linear in y beyond the table: a single decaying mode
    const Real y1 = y[ny-2], y2 = y[ny-1];
    const Real l1 = -lam_in*y1*Abar(xi_in, y1, lam_in);
    const Real l2 = -lam_in*y2*Abar(xi_in, y2, lam_in);
    return std::exp(l2 + (l2 - l1)/(y2 - y1)*(y_in - y2));
  }
  return std::exp(-lam_in*y_in*Abar(xi_in, y_in, lam_in));
}
