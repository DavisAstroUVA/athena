//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file opacity_table.cpp
//! \brief the multigroup opacity table, the photon Planck integral it emits with, and
//! the per-block tables built from it

// C headers
#include <stdio.h>

// C++ headers
#include <algorithm>  // min, max
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cstdlib>  // atof
#include <cstring>  // strstr
#include <string>
#include <vector>

// Athena++ headers
#include "opacity_table.hpp"
#include "montecarlo.hpp"
#include "../globals.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

namespace {
// Constants, as literals so that the table path's arithmetic is self-contained
constexpr Real h_cgs = 6.62607015e-27;
constexpr Real c_cgs = 2.99792458e10;
constexpr Real kb_cgs = 1.380649e-16;
constexpr Real keverg = 1.602176634e-9;
constexpr Real mp_cgs = 1.67262192369e-24;
constexpr Real mu_cgs = 1.66053906660e-24;  // atomic mass unit
constexpr Real ffnrm = 3.692146e8;
constexpr Real zeta3x2 = 2.4041138063191885;  // 2 zeta(3), int_0^inf x^2/(e^x-1) dx

// Eight-point Gauss-Legendre rule on [-1, 1], for the free-free group means
constexpr int NGL = 8;
constexpr Real gl_x[NGL] = {-0.9602898564975363, -0.7966664774136267,
                            -0.5255324099163290, -0.1834346424956498,
                             0.1834346424956498,  0.5255324099163290,
                             0.7966664774136267,  0.9602898564975363};
constexpr Real gl_w[NGL] = {0.1012285362903763, 0.2223810344533745,
                            0.3137066458778873, 0.3626837833783620,
                            0.3626837833783620, 0.3137066458778873,
                            0.2223810344533745, 0.1012285362903763};

//----------------------------------------------------------------------------------------
// The cumulative photon Planck function P(x) = int_0^x t^2/(e^t - 1) dt / (2 zeta(3)),
// the fraction of a blackbody's photons below x = h nu / k T, tabulated in ln P against
// ln x on first use.  ln P is close to linear in ln x (P ~ x^2/2 at small x and smooth
// beyond), so the linear interpolation is good to a few 1e-8.  Below the table P comes
// from the series, above it from one minus the leading term of the Wien tail.

constexpr int NPL = 4001;
constexpr Real pl_xmin = 1.e-5, pl_xmax = 100.;
std::vector<Real> pl_lnx, pl_lnp;

void BuildPlanckTable() {
  if (!pl_lnx.empty()) return;
  pl_lnx.resize(NPL);
  pl_lnp.resize(NPL);
  const Real dl = std::log(pl_xmax/pl_xmin)/static_cast<Real>(NPL-1);
  // integrand in ln t: t^3/(e^t - 1), with expm1 for small t
  auto integrand = [](Real t) { return t*t*t/std::expm1(t); };
  // start of the table from the series, then Simpson's rule on each interval with its
  // midpoint, which keeps the cumulative sums consistent to the rule's accuracy
  Real x0 = pl_xmin;
  Real p = (x0*x0/2. - x0*x0*x0/6. + x0*x0*x0*x0/48.)/zeta3x2;
  pl_lnx[0] = std::log(x0);
  pl_lnp[0] = std::log(p);
  for (int i=1; i<NPL; ++i) {
    const Real la = pl_lnx[0] + dl*(i-1), lb = la + dl, lm = 0.5*(la + lb);
    p += (integrand(std::exp(la)) + 4.*integrand(std::exp(lm))
          + integrand(std::exp(lb)))*dl/6./zeta3x2;
    pl_lnx[i] = lb;
    pl_lnp[i] = std::log(std::min(p, 1.));
  }
}
}  // namespace

Real PlanckPhotonCumulative(Real x) {
  if (x <= 0.) return 0.;
  if (x < pl_xmin) return (x*x/2. - x*x*x/6.)/zeta3x2;
  if (x >= pl_xmax) return 1. - std::exp(-x)*(x*x + 2.*x + 2.)/zeta3x2;
  BuildPlanckTable();
  const Real lx = std::log(x);
  const Real s = (lx - pl_lnx[0])/(pl_lnx[1] - pl_lnx[0]);
  int i = static_cast<int>(s);
  i = std::min(std::max(i, 0), NPL-2);
  const Real a = (lx - pl_lnx[i])/(pl_lnx[i+1] - pl_lnx[i]);
  return std::exp((1.-a)*pl_lnp[i] + a*pl_lnp[i+1]);
}

Real PlanckPhotonInverse(Real prob) {
  BuildPlanckTable();
  if (prob <= 0.) return 0.;
  const Real lp = std::log(prob);
  if (lp <= pl_lnp[0]) return std::sqrt(2.*prob*zeta3x2);
  if (lp >= pl_lnp[NPL-1]) return pl_xmax;
  int lo = 0, hi = NPL-1;
  while (hi - lo > 1) {
    const int mid = (lo + hi)/2;
    if (pl_lnp[mid] <= lp)
      lo = mid;
    else
      hi = mid;
  }
  const Real a = (lp - pl_lnp[lo])/(pl_lnp[hi] - pl_lnp[lo]);
  return std::exp((1.-a)*pl_lnx[lo] + a*pl_lnx[hi]);
}

OpacityTable::OpacityTable() :
    nfre(0), ntem(0), nrho(0), uniform_t(false), uniform_rho(false),
    groups(TABLE_GROUPS_MEAN), offgrid(TABLE_OFFGRID_FREEFREE),
    above(TABLE_ABOVE_FREEFREE), emax(0.), cell_storage(true), gray_to_freefree(false),
    fill_freefree(false), heabund(0.09), ngray_rows(0), ngroups(0), nabs(0), abar(1.),
    have_nfree(false), table_ne(false) {}

//----------------------------------------------------------------------------------------
//! \fn void OpacityTable::Read(ParameterInput *pin)
//! \brief read <montecarlo>/opacity_file and the options

void OpacityTable::Read(ParameterInput *pin) {
  // <problem>/opacity_filename is accepted as an alias
  std::string fname = pin->GetOrAddString("montecarlo", "opacity_file", "");
  if (fname.empty() && pin->DoesParameterExist("problem", "opacity_filename")) {
    fname = pin->GetString("problem", "opacity_filename");
    if (Globals::my_rank == 0)
      std::cout << "Opacity table: <problem>/opacity_filename is read as "
                << "<montecarlo>/opacity_file; use the latter" << std::endl;
  }
  if (fname.empty()) {
    std::stringstream msg;
    msg << "### FATAL ERROR in OpacityTable::Read" << std::endl
        << "emission = table or absorption = table needs <montecarlo>/opacity_file"
        << std::endl;
    ATHENA_ERROR(msg);
  }
  {
    std::string mode = pin->GetOrAddString("montecarlo", "table_groups", "mean");
    if (mode == "mean") {
      groups = TABLE_GROUPS_MEAN;
    } else if (mode == "edge") {
      groups = TABLE_GROUPS_EDGE;
    } else {
      std::stringstream msg;
      msg << "### FATAL ERROR in OpacityTable::Read" << std::endl
          << "<montecarlo>/table_groups = " << mode << "; use mean or edge" << std::endl;
      ATHENA_ERROR(msg);
    }
    mode = pin->GetOrAddString("montecarlo", "table_offgrid", "freefree");
    if (mode == "freefree") {
      offgrid = TABLE_OFFGRID_FREEFREE;
    } else if (mode == "clamp") {
      offgrid = TABLE_OFFGRID_CLAMP;
    } else {
      std::stringstream msg;
      msg << "### FATAL ERROR in OpacityTable::Read" << std::endl
          << "<montecarlo>/table_offgrid = " << mode << "; use freefree or clamp"
          << std::endl;
      ATHENA_ERROR(msg);
    }
    // above the top edge: free-free under the mean reading, the top value under edge
    mode = pin->GetOrAddString("montecarlo", "table_above",
                            (groups == TABLE_GROUPS_EDGE) ? "clamp" : "freefree");
    if (mode == "clamp") {
      above = TABLE_ABOVE_CLAMP;
    } else if (mode == "freefree") {
      above = TABLE_ABOVE_FREEFREE;
    } else if (mode == "none") {
      above = TABLE_ABOVE_NONE;
    } else {
      std::stringstream msg;
      msg << "### FATAL ERROR in OpacityTable::Read" << std::endl
          << "<montecarlo>/table_above = " << mode << "; use freefree, clamp or none"
          << std::endl;
      ATHENA_ERROR(msg);
    }
    mode = pin->GetOrAddString("montecarlo", "table_storage", "cell");
    if (mode == "cell") {
      cell_storage = true;
    } else if (mode == "global") {
      cell_storage = false;
    } else {
      std::stringstream msg;
      msg << "### FATAL ERROR in OpacityTable::Read" << std::endl
          << "<montecarlo>/table_storage = " << mode << "; use cell or global"
          << std::endl;
      ATHENA_ERROR(msg);
    }
    mode = pin->GetOrAddString("montecarlo", "table_gray", "keep");
    if (mode == "keep") {
      gray_to_freefree = false;
    } else if (mode == "freefree") {
      gray_to_freefree = true;
    } else {
      std::stringstream msg;
      msg << "### FATAL ERROR in OpacityTable::Read" << std::endl
          << "<montecarlo>/table_gray = " << mode << "; use keep or freefree"
          << std::endl;
      ATHENA_ERROR(msg);
    }
  }
  emax = pin->GetOrAddReal("montecarlo", "table_emax", 0.) * keverg;  // keV in the file
  fill_freefree = pin->GetOrAddBoolean("montecarlo", "table_fill_freefree", false);
  heabund = pin->GetOrAddReal("problem", "heabund", 0.09);
  table_ne = pin->GetOrAddBoolean("montecarlo", "table_ne", false);
  const std::string nfree_name = pin->GetOrAddString("montecarlo", "nfree_file", "");
  if (table_ne && nfree_name.empty()) {
    std::stringstream msg;
    msg << "### FATAL ERROR in OpacityTable::Read" << std::endl
        << "table_ne needs <montecarlo>/nfree_file" << std::endl;
    ATHENA_ERROR(msg);
  }

  FILE *file = fopen(fname.c_str(), "r");
  if (file == NULL) {
    std::stringstream msg;
    msg << "### FATAL ERROR in OpacityTable::Read" << std::endl
        << "cannot open <montecarlo>/opacity_file = " << fname << std::endl;
    ATHENA_ERROR(msg);
  }
  int nread = 0;
  nread += fscanf(file, "%d", &nfre);
  nread += fscanf(file, "%d", &ntem);
  nread += fscanf(file, "%d", &nrho);
  if (nread != 3 || nfre < 2 || ntem < 2 || nrho < 2) {
    std::stringstream msg;
    msg << "### FATAL ERROR in OpacityTable::Read" << std::endl
        << fname << ": bad header (nfre, ntem, nrho = " << nfre << ", " << ntem << ", "
        << nrho << ")" << std::endl;
    ATHENA_ERROR(msg);
  }
  egrid.NewAthenaArray(nfre);
  tgrid.NewAthenaArray(ntem);
  rgrid.NewAthenaArray(nrho);
  gray_ross.NewAthenaArray(ntem, nrho);
  gray_plan.NewAthenaArray(ntem, nrho);
  ross.NewAthenaArray(nfre, ntem, nrho);
  plan.NewAthenaArray(nfre, ntem, nrho);
  lplan.NewAthenaArray(nfre, ntem, nrho);

  // group lower bounds in keV, stored in erg
  for (int i=0; i<nfre; ++i) nread += fscanf(file, "%lf", &(egrid(i)));
  for (int i=0; i<nfre; ++i) egrid(i) *= keverg;
  lmine = std::log10(egrid(0));
  lmaxe = std::log10(egrid(nfre-1));
  dle = (lmaxe-lmine)/static_cast<Real>(nfre-1);
  // temperatures in keV, stored in K
  for (int i=0; i<ntem; ++i) nread += fscanf(file, "%lf", &(tgrid(i)));
  for (int i=0; i<ntem; ++i) tgrid(i) *= keverg/kb_cgs;
  lmint = std::log10(tgrid(0));
  lmaxt = std::log10(tgrid(ntem-1));
  dlt = (lmaxt-lmint)/static_cast<Real>(ntem-1);
  // densities in g/cm^3
  for (int i=0; i<nrho; ++i) nread += fscanf(file, "%lf", &(rgrid(i)));
  lmind = std::log10(rgrid(0));
  lmaxd = std::log10(rgrid(nrho-1));
  dld = (lmaxd-lmind)/static_cast<Real>(nrho-1);

  for (int j=0; j<ntem; ++j)
    for (int i=0; i<nrho; ++i) nread += fscanf(file, "%lf", &(gray_ross(j,i)));
  for (int j=0; j<ntem; ++j)
    for (int i=0; i<nrho; ++i) nread += fscanf(file, "%lf", &(gray_plan(j,i)));
  for (int k=0; k<nfre; ++k)
    for (int j=0; j<ntem; ++j)
      for (int i=0; i<nrho; ++i) nread += fscanf(file, "%lf", &(ross(k,j,i)));
  for (int k=0; k<nfre; ++k)
    for (int j=0; j<ntem; ++j)
      for (int i=0; i<nrho; ++i) nread += fscanf(file, "%lf", &(plan(k,j,i)));
  fclose(file);
  const int nexpect = 3 + nfre + ntem + nrho + 2*ntem*nrho + 2*nfre*ntem*nrho;
  if (nread != nexpect) {
    std::stringstream msg;
    msg << "### FATAL ERROR in OpacityTable::Read" << std::endl
        << fname << ": read " << nread << " values, expected " << nexpect << std::endl;
    ATHENA_ERROR(msg);
  }

  // The energy axis has to be uniform in log (the group of an energy comes from a
  // formula); the other two may be anything monotonic, the uniform ones are reported.
  // Tolerances allow for the five significant digits of the file.
  auto uniform = [](const AthenaArray<Real> &g, int n, Real d) {
    for (int i=0; i<n-1; ++i)
      if (std::abs(std::log10(g(i+1)/g(i)) - d) > 1.e-3*d) return false;
    return true;
  };
  if (!uniform(egrid, nfre, dle)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in OpacityTable::Read" << std::endl
        << fname << ": the group lower bounds are not uniform in log E" << std::endl;
    ATHENA_ERROR(msg);
  }
  uniform_t = uniform(tgrid, ntem, dlt);
  uniform_rho = uniform(rgrid, nrho, dld);

  if (fill_freefree) {
    // every Planck value replaced by the analytic free-free opacity per gram
    for (int k=0; k<nfre; ++k)
      for (int j=0; j<ntem; ++j)
        for (int i=0; i<nrho; ++i)
          plan(k,j,i) = FreeFree(tgrid(j), rgrid(i), egrid(k)) / rgrid(i);
  }
  // Rows that hold one value in every group are TOPS'mode clamped edge cells; with
  // table_gray = freefree they are replaced by free-free.  A row holding a zero cannot
  // be tested by the ratio and is left alone.
  ngray_rows = 0;
  for (int j=0; j<ntem; ++j) {
    for (int i=0; i<nrho; ++i) {
      Real min = 1.e40, max = 1.e-40;
      for (int k=0; k<nfre; ++k) {
        min = (min > plan(k,j,i)) ? plan(k,j,i) : min;
        max = (max < plan(k,j,i)) ? plan(k,j,i) : max;
      }
      if ((min > 0.) && (max/min < 1.1)) {
        ++ngray_rows;
        if (gray_to_freefree) {
          for (int k=0; k<nfre; ++k)
            plan(k,j,i) = FreeFree(tgrid(j), rgrid(i), egrid(k)) / rgrid(i);
        }
      }
    }
  }
  for (int k=0; k<nfre; ++k)
    for (int j=0; j<ntem; ++j)
      for (int i=0; i<nrho; ++i)
        lplan(k,j,i) = (plan(k,j,i) > 0.) ? std::log(plan(k,j,i)) : 0.;

  // the group structure, see the header
  if (groups == TABLE_GROUPS_MEAN) {
    int next = 0;
    const Real top = std::pow(10., lmine + nfre*dle);
    if (above == TABLE_ABOVE_FREEFREE && emax > top)
      next = static_cast<int>(std::ceil(std::log10(emax/top)/dle));
    ngroups = nfre + next;
    nabs = ngroups;
    gedge.NewAthenaArray(ngroups+1);
    for (int g=0; g<nfre; ++g) gedge(g) = egrid(g);
    for (int g=nfre; g<=ngroups; ++g) gedge(g) = std::pow(10., lmine + g*dle);
  } else {
    ngroups = nfre-1;
    nabs = nfre;
    gedge.NewAthenaArray(nfre);
    for (int g=0; g<nfre; ++g) gedge(g) = egrid(g);
  }

  if (!nfree_name.empty()) ReadNFree(nfree_name);

  if (Globals::my_rank == 0) {
    std::cout << "Opacity table " << fname << ": " << nfre << " groups "
              << egrid(0)/keverg << " to " << gedge(ngroups)/keverg << " keV"
              << ((ngroups > nfre) ? " with the free-free extension" : "") << ", " << ntem
              << " temperatures " << tgrid(0) << " to " << tgrid(ntem-1) << " K"
              << (uniform_t ? " (uniform)" : "") << ", " << nrho << " densities "
              << rgrid(0) << " to " << rgrid(nrho-1) << " g/cm^3"
              << (uniform_rho ? " (uniform)" : "") << "; " << ngray_rows << " of "
              << ntem*nrho << " (T,rho) rows gray"
              << (gray_to_freefree ? ", replaced by free-free" : ", kept")
              << "; groups read as "
              << ((groups == TABLE_GROUPS_MEAN) ? "means" : "edges") << ", storage "
              << (cell_storage ? "cell" : "global") << std::endl;
    if (have_nfree)
      std::cout << "  free electrons per atom from " << nfree_name << ": Abar = " << abar
                << ", N_free " << nfree_min_ << " to " << nfree_max_
                << (table_ne ? "; sets the electron density" : "") << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void OpacityTable::ReadNFree(const std::string &fname)
//! \brief read the free electrons per atom on the main table's (T, rho) grid

void OpacityTable::ReadNFree(const std::string &fname) {
  FILE *file = fopen(fname.c_str(), "r");
  if (file == NULL) {
    std::stringstream msg;
    msg << "### FATAL ERROR in OpacityTable::ReadNFree" << std::endl
        << "cannot open <montecarlo>/nfree_file = " << fname << std::endl;
    ATHENA_ERROR(msg);
  }
  // Comment lines first; the one with "Abar =" carries the mean atomic weight.
  abar = 0.;
  char line[1024];
  int ch;
  while ((ch = fgetc(file)) == '#') {
    if (fgets(line, sizeof(line), file) == NULL) break;
    const char *pos = std::strstr(line, "Abar =");
    if (pos != NULL) abar = std::atof(pos + 6);
  }
  if (ch != EOF) ungetc(ch, file);
  if (abar <= 0.) {
    std::stringstream msg;
    msg << "### FATAL ERROR in OpacityTable::ReadNFree" << std::endl
        << fname << ": no \"# Abar = <value>\" line in the header" << std::endl;
    ATHENA_ERROR(msg);
  }
  int nt = 0, nr = 0, nread = 0;
  nread += fscanf(file, "%d", &nt);
  nread += fscanf(file, "%d", &nr);
  if (nread != 2 || nt != ntem || nr != nrho) {
    std::stringstream msg;
    msg << "### FATAL ERROR in OpacityTable::ReadNFree" << std::endl
        << fname << ": " << nt << " temperatures and " << nr << " densities, the opacity "
        << "table has " << ntem << " and " << nrho << std::endl;
    ATHENA_ERROR(msg);
  }
  // the grids, which must be the main table's to the file's five digits
  for (int j=0; j<ntem; ++j) {
    Real tk;
    nread += fscanf(file, "%lf", &tk);
    if (std::abs(tk*keverg/kb_cgs/tgrid(j) - 1.) > 1.e-3) {
      std::stringstream msg;
      msg << "### FATAL ERROR in OpacityTable::ReadNFree" << std::endl
          << fname << ": temperature " << j << " is " << tk*keverg/kb_cgs << " K, the "
          << "opacity table's is " << tgrid(j) << std::endl;
      ATHENA_ERROR(msg);
    }
  }
  for (int i=0; i<nrho; ++i) {
    Real r;
    nread += fscanf(file, "%lf", &r);
    if (std::abs(r/rgrid(i) - 1.) > 1.e-3) {
      std::stringstream msg;
      msg << "### FATAL ERROR in OpacityTable::ReadNFree" << std::endl
          << fname << ": density " << i << " is " << r << ", the opacity table's is "
          << rgrid(i) << std::endl;
      ATHENA_ERROR(msg);
    }
  }
  nfree.NewAthenaArray(ntem, nrho);
  nfree_min_ = 1.e99;
  nfree_max_ = 0.;
  for (int j=0; j<ntem; ++j) {
    for (int i=0; i<nrho; ++i) {
      nread += fscanf(file, "%lf", &(nfree(j,i)));
      nfree_min_ = std::min(nfree_min_, nfree(j,i));
      nfree_max_ = std::max(nfree_max_, nfree(j,i));
    }
  }
  fclose(file);
  if (nread != 2 + ntem + nrho + ntem*nrho) {
    std::stringstream msg;
    msg << "### FATAL ERROR in OpacityTable::ReadNFree" << std::endl
        << fname << ": read " << nread << " values, expected "
        << 2 + ntem + nrho + ntem*nrho << std::endl;
    ATHENA_ERROR(msg);
  }
  have_nfree = true;
}

//----------------------------------------------------------------------------------------
//! \fn Real OpacityTable::NFree(const TableCell &cell) const
//! \brief free electrons per atom at a cell's bracket: bilinear in (log T, log rho), the
//! weights clamped to the grid for a cell off it

Real OpacityTable::NFree(const TableCell &cell) const {
  const int ii = cell.irho, jj = cell.item;
  const Real xi = std::min(std::max(cell.xi, 0.), 1.);
  const Real xj = std::min(std::max(cell.xj, 0.), 1.);
  return (1.-xi)*((1.-xj)*nfree(jj,ii) + xj*nfree(jj+1,ii))
         + xi*((1.-xj)*nfree(jj,ii+1) + xj*nfree(jj+1,ii+1));
}

//----------------------------------------------------------------------------------------
//! \fn Real OpacityTable::ElectronDensity(Real temp, Real rho) const
//! \brief n_e = rho N_free / (Abar m_u) at (temp, rho)

Real OpacityTable::ElectronDensity(Real temp, Real rho) const {
  TableCell cell;
  bool off_rho, off_temp;
  Locate(temp, rho, cell, off_rho, off_temp);
  return rho*NFree(cell)/(abar*mu_cgs);
}

//----------------------------------------------------------------------------------------
//! \fn void OpacityTable::Locate(Real temp, Real rho, TableCell &cell, bool &off_rho,
//!                               bool &off_temp) const
//! \brief the cell's bracket on the (T, rho) grid

void OpacityTable::Locate(Real temp, Real rho, TableCell &cell, bool &off_rho,
                          bool &off_temp) const {
  off_rho = off_temp = false;
  const Real ld = std::log10(rho);
  const Real lt = std::log10(temp);
  Real xi = (ld - lmind) / dld;
  int ii = std::floor(xi);
  if (ii < 0) {
    ii = 0;
    off_rho = true;
  } else if (ii > nrho-2) {
    ii = nrho-2;
    off_rho = true;
  }
  xi -= static_cast<Real>(ii);
  Real xj = (lt - lmint) / dlt;
  int jj = std::floor(xj);
  if (jj < 0) jj = 0;
  if (jj > ntem-2) jj = ntem-2;
  // the temperature axis need not be uniform: walk from the formula's guess
  while ((jj < ntem-2) && (tgrid(jj+1) < temp)) jj++;
  while ((jj > 0) && (tgrid(jj) > temp)) jj--;
  // Fractional position in log T, to match xi in log rho.  A cell off either end of the
  // grid shows up as a weight outside [0,1]; jj itself was clamped above.
  xj = std::log(temp/tgrid(jj)) / std::log(tgrid(jj+1)/tgrid(jj));
  if ((xj < 0.) || (xj > 1.)) off_temp = true;
  cell.irho = ii;
  cell.item = jj;
  cell.xi = xi;
  cell.xj = xj;
  const bool on_grid = !off_rho && !off_temp && (xi >= 0.) && (xi <= 1.);
  if (on_grid || offgrid == TABLE_OFFGRID_CLAMP) {
    if (!on_grid) {
      // clamp the weights to the edge of the grid
      cell.xi = std::min(std::max(cell.xi, 0.), 1.);
      cell.xj = std::min(std::max(cell.xj, 0.), 1.);
    }
    cell.source = TABLE_SRC_TABLE;
  } else {
    cell.source = TABLE_SRC_FREEFREE;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real OpacityTable::PlanckNode(int l, const TableCell &cell) const
//! \brief Planck value l for a cell on the grid [cm^2/g]
//
// Log-log in (rho, T).  These opacities are power laws over most of the plane, so
// interpolating the logarithm is far closer to the truth than interpolating the value:
// reconstructing dropped points from the 64-group table gives a median error of 2.5%
// this way against 7.8% linearly in temperature, and 0.0% against 3.3% in density.  Any
// non-positive corner drops back to linear, where the logarithm is not defined.

Real OpacityTable::PlanckNode(int l, const TableCell &cell) const {
  const int ii = cell.irho, jj = cell.item;
  const Real xi = cell.xi, xj = cell.xj;
  const Real k00 = plan(l,jj  ,ii  ), k10 = plan(l,jj+1,ii  );
  const Real k01 = plan(l,jj  ,ii+1), k11 = plan(l,jj+1,ii+1);
  if ((k00 > 0.) && (k10 > 0.) && (k01 > 0.) && (k11 > 0.)) {
    return std::exp((1.-xi)*((1.-xj)*lplan(l,jj,ii) + xj*lplan(l,jj+1,ii))
                    + xi *((1.-xj)*lplan(l,jj,ii+1) + xj*lplan(l,jj+1,ii+1)));
  }
  return (1.-xi)*((1.-xj)*k00 + xj*k10) + xi*((1.-xj)*k01 + xj*k11);
}

//----------------------------------------------------------------------------------------
//! \fn Real OpacityTable::FreeFree(Real temp, Real rho, Real energy) const
//! \brief free-free extinction [1/cm] of fully ionized hydrogen and helium

Real OpacityTable::FreeFree(Real temp, Real rho, Real energy) const {
  const Real nh = rho / (mp_cgs*(1.+4.*heabund));
  const Real nhe = nh*heabund;
  const Real ne = nh + 2.*nhe;
  const Real nu = energy / h_cgs;
  const Real ehnu = std::exp(-h_cgs*nu / (kb_cgs * temp));
  const Real aff = ffnrm/std::sqrt(temp)/std::pow(nu,3);
  return ne * (nh + 4. * nhe) * aff * (1. - ehnu);
}

//----------------------------------------------------------------------------------------
//! \fn Real OpacityTable::FreeFreeGroupMean(Real elo, Real ehi, Real temp,
//!                                          Real rho) const
//! \brief Planck mean of the free-free extinction over [elo, ehi] [1/cm]
//
// Gauss-Legendre in ln nu for both the numerator and the denominator, so the ratio is
// exact for an opacity constant across the group and good to 1e-7 for a power law over
// the groups' factor of 1.33.  Where B_nu underflows across the whole group (deep in the
// Wien tail) the value at the geometric centre is returned; nothing is emitted there.

Real OpacityTable::FreeFreeGroupMean(Real elo, Real ehi, Real temp, Real rho) const {
  const Real la = std::log(elo), lb = std::log(ehi);
  const Real mid = 0.5*(la + lb), half = 0.5*(lb - la);
  Real num = 0., den = 0.;
  for (int q=0; q<NGL; ++q) {
    const Real energy = std::exp(mid + half*gl_x[q]);
    const Real weight = gl_w[q]*energy*PlanckNu(temp, energy/h_cgs);  // nu B_nu d ln nu
    num += weight*FreeFree(temp, rho, energy);
    den += weight;
  }
  if (den <= 0.) return FreeFree(temp, rho, std::exp(mid));
  return num/den;
}

//----------------------------------------------------------------------------------------
//! \fn Real OpacityTable::PlanckNu(Real temp, Real nu)
//! \brief B_nu(T)

Real OpacityTable::PlanckNu(Real temp, Real nu) {
  return 2.*h_cgs/c_cgs/c_cgs*std::pow(nu,3)/(std::exp(h_cgs*nu/kb_cgs/temp)-1.);
}

//----------------------------------------------------------------------------------------
//! \fn Real OpacityTable::PlanckPhotonBand(Real temp, Real elo, Real ehi)
//! \brief int B_nu / (h nu) dnu over [elo, ehi]: (2/cell^2) (kT/h)^3 2 zeta(3)
//! [P(x_hi) - P(x_lo)], photons/cm^2/s/sr

Real OpacityTable::PlanckPhotonBand(Real temp, Real elo, Real ehi) {
  const Real kt = kb_cgs*temp;
  const Real nt = kt/h_cgs;
  return 2./c_cgs/c_cgs*nt*nt*nt*zeta3x2
         *(PlanckPhotonCumulative(ehi/kt) - PlanckPhotonCumulative(elo/kt));
}

//----------------------------------------------------------------------------------------
//! \fn Real OpacityTable::Extinction(int l, const TableCell &cell, Real temp,
//!                                   Real rho) const
//! \brief extinction l of a cell [1/cm]: group l (mean) or node l (edge)

Real OpacityTable::Extinction(int l, const TableCell &cell, Real temp, Real rho) const {
  if (cell.source == TABLE_SRC_CUT) return TABLE_CUT_VALUE;
  if (groups == TABLE_GROUPS_EDGE) {
    if (cell.source == TABLE_SRC_TABLE) return PlanckNode(l, cell) * rho;
    return FreeFree(temp, rho, egrid(l));
  }
  // the table is per gram; the cell's own density makes the extinction
  if (l < nfre && cell.source == TABLE_SRC_TABLE) return PlanckNode(l, cell) * rho;
  // off the grid, or an extension group above the table
  return FreeFreeGroupMean(gedge(l), gedge(l+1), temp, rho);
}

//----------------------------------------------------------------------------------------
//! \fn Real OpacityTable::EmissionCumulative(const Real *chi, Real temp,
//!                                           Real *cum) const
//! \brief normalized cumulative emission over the groups and the total, in photons per
//! cm^3 per second, which is what the emission array and the sample weights carry
//
// mean: 4 pi chi_g times the photon Planck integral over group g; edge: B_nu chi_nu /
// (h nu) at the nodes, the trapezoid rule in ln nu between them.  A cell with no
// emission leaves the cumulative array at zero
// rather than dividing by it: cells are drawn uniformly in SetEmissionCellWeight and
// only then weighted by the emission array, so a non-emitting cell is still handed to
// the sampler.

Real OpacityTable::EmissionCumulative(const Real *chi, Real temp, Real *cum) const {
  cum[0] = 0.;
  if (groups == TABLE_GROUPS_MEAN) {
    for (int g=0; g<ngroups; ++g)
      cum[g+1] = cum[g] + 4.*PI*chi[g]*PlanckPhotonBand(temp, gedge(g), gedge(g+1));
  } else {
    Real eta_prev = PlanckNu(temp, egrid(0)/h_cgs) * chi[0];
    for (int l=1; l<nfre; ++l) {
      const Real nup = egrid(l)/h_cgs;
      const Real num = egrid(l-1)/h_cgs;
      const Real dlnu = std::log(nup/num);
      const Real eta = PlanckNu(temp, nup) * chi[l];
      const Real eta_ave = 0.5*(eta + eta_prev);
      cum[l] = cum[l-1] + 4.*PI/h_cgs*eta_ave*dlnu;
      eta_prev = eta;
    }
  }
  const Real tot = cum[ngroups];
  if (tot > 0.) {
    for (int l=1; l<=ngroups; ++l) cum[l] /= tot;
  }
  return tot;
}

//----------------------------------------------------------------------------------------
//! \fn Real OpacityTable::DrawEnergy(int g, Real a, Real temp) const
//! \brief energy within group g at fraction a of its emission
//
// mean: the inverse of the photon Planck integral between the group's edges, so the
// energies within a group follow B_nu / (h nu); edge: log-linear in the node interval
// (the node grid is log-spaced, so a draw linear in energy would sit systematically
// high).

Real OpacityTable::DrawEnergy(int g, Real a, Real temp) const {
  if (groups == TABLE_GROUPS_EDGE)
    return std::exp(a*std::log(gedge(g+1)) + (1.-a)*std::log(gedge(g)));
  const Real kt = kb_cgs*temp;
  const Real plo = PlanckPhotonCumulative(gedge(g)/kt);
  const Real phi = PlanckPhotonCumulative(gedge(g+1)/kt);
  if (phi <= plo) return gedge(g);
  const Real x = PlanckPhotonInverse(plo + a*(phi - plo));
  return std::min(std::max(x*kt, gedge(g)), gedge(g+1));
}

//----------------------------------------------------------------------------------------
//! \fn void OpacityTable::EnergyNode(Real energy, int &k, Real &xk) const
//! \brief edge reading: the node bracket of an energy and the weight of node k+1,
//! clamped to the grid

void OpacityTable::EnergyNode(Real energy, int &k, Real &xk) const {
  Real le = std::log10(energy);
  le = (le < lmine) ? lmine : le;
  le = (le > lmaxe) ? lmaxe : le;
  xk = (le - lmine) / dle;
  k = std::floor(xk);
  xk -= static_cast<Real>(k);
  if (k < 0) {
    k = 0;
    xk = 0.;
  } else if (k >= nfre-1) {
    k = nfre-2;
    xk = 1.;
  }
}

//----------------------------------------------------------------------------------------
//! \fn int OpacityTable::GroupIndex(Real energy) const
//! \brief mean reading: the group of an energy; -1 below the bottom edge, -2 above the
//! top one (extension groups included)

int OpacityTable::GroupIndex(Real energy) const {
  const int g = static_cast<int>(std::floor((std::log10(energy) - lmine)/dle));
  if (g < 0) return -1;
  if (g >= ngroups) return -2;
  return g;
}

//========================================================================================
// MonteCarloBlock: the per-block tables
//========================================================================================

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::BuildOpacityTables()
//! \brief locate every active cell on the table and, in cell storage, fill its
//! extinctions and cumulative emission
//
// Called from MonteCarlo::SetupBlockFromFluid after the number densities, so a block
// built on arrival after a redistribution, a restart and every transport of a dynamic run
// carry tables that match the fluid they transport through.  Active cells only: every
// read comes from a photon in an active cell, and the ghost layer would triple the
// footprint on a large mesh.

void MonteCarloBlock::BuildOpacityTables() {
  const OpacityTable *tab = pmy_mc->optab;
  if (tab == nullptr) return;
  for (int n=0; n<NTABCOUNT; ++n) tab_count[n] = 0;
  Real *chi = &tab_scratch_[0];
  Real *cum = &tab_scratch_[tab->nabs];
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        const int kt = k-ks, jt = j-js, it = i-is;
        TableCell &cell = tab_cell[TableIndex(kt,jt,it)];
        const Real temp = tgas(k,j,i), rhoc = rho(k,j,i);
        if (tab_mask(k,j,i) == 0) {
          // masked by the problem generator: negligible extinction and emission
          cell.irho = cell.item = 0;
          cell.xi = cell.xj = 0.;
          cell.source = TABLE_SRC_CUT;
          ++tab_count[TABCOUNT_CUT];
        } else {
          bool off_rho, off_temp;
          tab->Locate(temp, rhoc, cell, off_rho, off_temp);
          if (off_rho) ++tab_count[TABCOUNT_OFF_RHO];
          if (off_temp) ++tab_count[TABCOUNT_OFF_TEMP];
          if (cell.source == TABLE_SRC_TABLE)
            ++tab_count[TABCOUNT_TABLE];
          else
            ++tab_count[TABCOUNT_FREEFREE];
        }
        Real *chi_c = tab->cell_storage ? &tab_abs(kt,jt,it,0) : chi;
        Real *cum_c = tab->cell_storage ? &tab_emis_cum(kt,jt,it,0) : cum;
        for (int l=0; l<tab->nabs; ++l) chi_c[l] = tab->Extinction(l, cell, temp, rhoc);
        Real tot = tab->EmissionCumulative(chi_c, temp, cum_c);
        // A masked cell emits TABLE_CUT_VALUE outright, whatever the Planck function at
        // its temperature made of the cut extinction; its cumulative array, normalized
        // from that shape, still gives the sampler a valid distribution for the
        // negligible weight it carries.
        if (cell.source == TABLE_SRC_CUT) tot = TABLE_CUT_VALUE;
        tab_emis_tot(kt,jt,it) = tot;
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real MonteCarloBlock::TableEmissionCumulative(int i3, int i2, int i1,
//!                                                   const Real *&cum)
//! \brief the cell's normalized cumulative emission over its groups; returns the total
//
// Cell storage hands back the stored array; global storage recomputes it into the
// block's scratch, which the caller must consume before the next call.

Real MonteCarloBlock::TableEmissionCumulative(int i3, int i2, int i1, const Real *&cum) {
  const OpacityTable *tab = pmy_mc->optab;
  const int kt = i3-ks, jt = i2-js, it = i1-is;
  if (tab->cell_storage) {
    cum = &tab_emis_cum(kt,jt,it,0);
    return tab_emis_tot(kt,jt,it);
  }
  Real *chi = &tab_scratch_[0];
  Real *cum_s = &tab_scratch_[tab->nabs];
  const TableCell &cell = tab_cell[TableIndex(kt,jt,it)];
  const Real temp = tgas(i3,i2,i1), rhoc = rho(i3,i2,i1);
  for (int l=0; l<tab->nabs; ++l) chi[l] = tab->Extinction(l, cell, temp, rhoc);
  Real tot = tab->EmissionCumulative(chi, temp, cum_s);
  if (cell.source == TABLE_SRC_CUT) tot = TABLE_CUT_VALUE;
  cum = cum_s;
  return tot;
}
