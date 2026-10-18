//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file opacity_table.cpp
//! \brief the multigroup opacity table and the per-block tables built from it

// C headers
#include <stdio.h>

// C++ headers
#include <algorithm>  // min, max
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

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
constexpr Real ffnrm = 3.692146e8;
}  // namespace

OpacityTable::OpacityTable() :
    nfre(0), ntem(0), nrho(0), groups(TABLE_GROUPS_EDGE), offgrid(TABLE_OFFGRID_FREEFREE),
    above(TABLE_ABOVE_CLAMP), cell_storage(true), gray_to_freefree(false),
    fill_freefree(false), heabund(0.09), ngray_rows(0) {}

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
    std::string s = pin->GetOrAddString("montecarlo", "table_groups", "edge");
    if (s == "edge") {
      groups = TABLE_GROUPS_EDGE;
    } else {
      std::stringstream msg;
      msg << "### FATAL ERROR in OpacityTable::Read" << std::endl
          << "<montecarlo>/table_groups = " << s << "; use edge" << std::endl;
      ATHENA_ERROR(msg);
    }
    s = pin->GetOrAddString("montecarlo", "table_offgrid", "freefree");
    if (s == "freefree") {
      offgrid = TABLE_OFFGRID_FREEFREE;
    } else if (s == "clamp") {
      offgrid = TABLE_OFFGRID_CLAMP;
    } else {
      std::stringstream msg;
      msg << "### FATAL ERROR in OpacityTable::Read" << std::endl
          << "<montecarlo>/table_offgrid = " << s << "; use freefree or clamp"
          << std::endl;
      ATHENA_ERROR(msg);
    }
    s = pin->GetOrAddString("montecarlo", "table_above", "clamp");
    if (s == "clamp") {
      above = TABLE_ABOVE_CLAMP;
    } else {
      std::stringstream msg;
      msg << "### FATAL ERROR in OpacityTable::Read" << std::endl
          << "<montecarlo>/table_above = " << s << "; use clamp" << std::endl;
      ATHENA_ERROR(msg);
    }
    s = pin->GetOrAddString("montecarlo", "table_storage", "cell");
    if (s == "cell") {
      cell_storage = true;
    } else if (s == "global") {
      cell_storage = false;
    } else {
      std::stringstream msg;
      msg << "### FATAL ERROR in OpacityTable::Read" << std::endl
          << "<montecarlo>/table_storage = " << s << "; use cell or global" << std::endl;
      ATHENA_ERROR(msg);
    }
    s = pin->GetOrAddString("montecarlo", "table_gray", "keep");
    if (s == "keep") {
      gray_to_freefree = false;
    } else if (s == "freefree") {
      gray_to_freefree = true;
    } else {
      std::stringstream msg;
      msg << "### FATAL ERROR in OpacityTable::Read" << std::endl
          << "<montecarlo>/table_gray = " << s << "; use keep or freefree" << std::endl;
      ATHENA_ERROR(msg);
    }
  }
  fill_freefree = pin->GetOrAddBoolean("montecarlo", "table_fill_freefree", false);
  heabund = pin->GetOrAddReal("problem", "heabund", 0.09);

  FILE *f = fopen(fname.c_str(), "r");
  if (f == NULL) {
    std::stringstream msg;
    msg << "### FATAL ERROR in OpacityTable::Read" << std::endl
        << "cannot open <montecarlo>/opacity_file = " << fname << std::endl;
    ATHENA_ERROR(msg);
  }
  int nread = 0;
  nread += fscanf(f, "%d", &nfre);
  nread += fscanf(f, "%d", &ntem);
  nread += fscanf(f, "%d", &nrho);
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

  // Group lower bounds in keV, stored in erg.  The energy axis is taken as uniform in
  // log, which the TOPS grids are.
  for (int i=0; i<nfre; ++i) nread += fscanf(f, "%lf", &(egrid(i)));
  for (int i=0; i<nfre; ++i) egrid(i) *= keverg;
  lmine = std::log10(egrid(0));
  lmaxe = std::log10(egrid(nfre-1));
  dle = (lmaxe-lmine)/static_cast<Real>(nfre-1);
  // temperatures in keV, stored in K
  for (int i=0; i<ntem; ++i) nread += fscanf(f, "%lf", &(tgrid(i)));
  for (int i=0; i<ntem; ++i) tgrid(i) *= keverg/kb_cgs;
  lmint = std::log10(tgrid(0));
  lmaxt = std::log10(tgrid(ntem-1));
  dlt = (lmaxt-lmint)/static_cast<Real>(ntem-1);
  // densities in g/cm^3
  for (int i=0; i<nrho; ++i) nread += fscanf(f, "%lf", &(rgrid(i)));
  lmind = std::log10(rgrid(0));
  lmaxd = std::log10(rgrid(nrho-1));
  dld = (lmaxd-lmind)/static_cast<Real>(nrho-1);

  for (int j=0; j<ntem; ++j)
    for (int i=0; i<nrho; ++i) nread += fscanf(f, "%lf", &(gray_ross(j,i)));
  for (int j=0; j<ntem; ++j)
    for (int i=0; i<nrho; ++i) nread += fscanf(f, "%lf", &(gray_plan(j,i)));
  for (int k=0; k<nfre; ++k)
    for (int j=0; j<ntem; ++j)
      for (int i=0; i<nrho; ++i) nread += fscanf(f, "%lf", &(ross(k,j,i)));
  for (int k=0; k<nfre; ++k)
    for (int j=0; j<ntem; ++j)
      for (int i=0; i<nrho; ++i) nread += fscanf(f, "%lf", &(plan(k,j,i)));
  fclose(f);
  const int nexpect = 3 + nfre + ntem + nrho + 2*ntem*nrho + 2*nfre*ntem*nrho;
  if (nread != nexpect) {
    std::stringstream msg;
    msg << "### FATAL ERROR in OpacityTable::Read" << std::endl
        << fname << ": read " << nread << " values, expected " << nexpect << std::endl;
    ATHENA_ERROR(msg);
  }

  if (fill_freefree) {
    // every Planck value replaced by the analytic free-free opacity per gram
    for (int k=0; k<nfre; ++k)
      for (int j=0; j<ntem; ++j)
        for (int i=0; i<nrho; ++i)
          plan(k,j,i) = FreeFree(tgrid(j), rgrid(i), egrid(k)) / rgrid(i);
  }
  // Rows that hold one value in every group are TOPS's clamped edge cells; with
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

  if (Globals::my_rank == 0) {
    std::cout << "Opacity table " << fname << ": " << nfre << " groups "
              << egrid(0)/keverg << " to " << egrid(nfre-1)/keverg << " keV, " << ntem
              << " temperatures " << tgrid(0) << " to " << tgrid(ntem-1) << " K, " << nrho
              << " densities " << rgrid(0) << " to " << rgrid(nrho-1) << " g/cm^3; "
              << ngray_rows << " of " << ntem*nrho << " (T,rho) rows gray"
              << (gray_to_freefree ? ", replaced by free-free" : ", kept")
              << "; storage " << (cell_storage ? "cell" : "global") << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void OpacityTable::Locate(Real temp, Real rho, TableCell &c, bool &off_rho,
//!                               bool &off_temp) const
//! \brief the cell's bracket on the (T, rho) grid

void OpacityTable::Locate(Real temp, Real rho, TableCell &c, bool &off_rho,
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
  c.irho = ii;
  c.item = jj;
  c.xi = xi;
  c.xj = xj;
  const bool on_grid = !off_rho && !off_temp && (xi >= 0.) && (xi <= 1.);
  if (on_grid || offgrid == TABLE_OFFGRID_CLAMP) {
    if (!on_grid) {
      // clamp the weights to the edge of the grid
      c.xi = std::min(std::max(c.xi, 0.), 1.);
      c.xj = std::min(std::max(c.xj, 0.), 1.);
    }
    c.source = TABLE_SRC_TABLE;
  } else {
    c.source = TABLE_SRC_FREEFREE;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real OpacityTable::PlanckNode(int l, const TableCell &c) const
//! \brief Planck value at node l for a cell on the grid [cm^2/g]
//
// Log-log in (rho, T).  These opacities are power laws over most of the plane, so
// interpolating the logarithm is far closer to the truth than interpolating the value:
// reconstructing dropped points from the 64-group table gives a median error of 2.5%
// this way against 7.8% linearly in temperature, and 0.0% against 3.3% in density.  Any
// non-positive corner drops back to linear, where the logarithm is not defined.

Real OpacityTable::PlanckNode(int l, const TableCell &c) const {
  const int ii = c.irho, jj = c.item;
  const Real xi = c.xi, xj = c.xj;
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
//! \fn Real OpacityTable::PlanckNu(Real temp, Real nu)
//! \brief B_nu(T)

Real OpacityTable::PlanckNu(Real temp, Real nu) {
  return 2.*h_cgs/c_cgs/c_cgs*std::pow(nu,3)/(std::exp(h_cgs*nu/kb_cgs/temp)-1.);
}

//----------------------------------------------------------------------------------------
//! \fn Real OpacityTable::NodeExtinction(int l, const TableCell &c, Real temp,
//!                                       Real rho) const
//! \brief extinction at node l [1/cm] of a cell

Real OpacityTable::NodeExtinction(int l, const TableCell &c, Real temp, Real rho) const {
  switch (c.source) {
    case TABLE_SRC_TABLE:
      // the table is per gram; the cell's own density makes the extinction
      return PlanckNode(l, c) * rho;
    case TABLE_SRC_FREEFREE:
      return FreeFree(temp, rho, egrid(l));
    default:
      return TABLE_CUT_VALUE;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real OpacityTable::EmissionCumulative(const Real *chi, Real temp,
//!                                           Real *cum) const
//! \brief normalized cumulative emission over the nodes and the total [erg/cm^3/s]
//
// Edge reading: B_nu kappa_nu at the nodes, the trapezoid rule in ln nu between them.
// A cell with no emission leaves the cumulative array at zero rather than dividing by
// it: cells are drawn uniformly in SetEmissionCellWeight and only then weighted by the
// emission array, so a non-emitting cell is still handed to the sampler.

Real OpacityTable::EmissionCumulative(const Real *chi, Real temp, Real *cum) const {
  Real eta_prev = PlanckNu(temp, egrid(0)/h_cgs) * chi[0];
  cum[0] = 0.;
  for (int l=1; l<nfre; ++l) {
    const Real nup = egrid(l)/h_cgs;
    const Real num = egrid(l-1)/h_cgs;
    const Real dlnu = std::log(nup/num);
    const Real eta = PlanckNu(temp, nup) * chi[l];
    const Real eta_ave = 0.5*(eta + eta_prev);
    cum[l] = cum[l-1] + 4.*PI/h_cgs*eta_ave*dlnu;
    eta_prev = eta;
  }
  const Real tot = cum[nfre-1];
  if (tot > 0.) {
    for (int l=1; l<nfre; ++l) cum[l] /= tot;
  }
  return tot;
}

//----------------------------------------------------------------------------------------
//! \fn void OpacityTable::EnergyNode(Real energy, int &k, Real &xk) const
//! \brief the node bracket of an energy and the weight of node k+1, clamped to the grid

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

//========================================================================================
// MonteCarloBlock: the per-block tables
//========================================================================================

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::BuildOpacityTables()
//! \brief locate every active cell on the table and, in cell storage, fill its
//! extinction nodes and cumulative emission
//
// Called from MonteCarlo::SetupBlockFromFluid after the number densities, so a block
// built on arrival after a redistribution, a restart and every transport of a dynamic run
// carry tables that match the fluid they transport through.  Active cells only.

void MonteCarloBlock::BuildOpacityTables() {
  const OpacityTable *t = pmy_mc->optab;
  if (t == nullptr) return;
  for (int n=0; n<NTABCOUNT; ++n) tab_count[n] = 0;
  const int nf = t->nfre;
  Real *chi = &tab_scratch_[0];
  Real *cum = &tab_scratch_[nf];
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        const int kt = k-ks, jt = j-js, it = i-is;
        TableCell &c = tab_cell[TableIndex(kt,jt,it)];
        const Real temp = tgas(k,j,i), rhoc = rho(k,j,i);
        if (tab_mask(k,j,i) == 0) {
          // masked by the problem generator: negligible extinction and emission
          c.irho = c.item = 0;
          c.xi = c.xj = 0.;
          c.source = TABLE_SRC_CUT;
          ++tab_count[TABCOUNT_CUT];
        } else {
          bool off_rho, off_temp;
          t->Locate(temp, rhoc, c, off_rho, off_temp);
          if (off_rho) ++tab_count[TABCOUNT_OFF_RHO];
          if (off_temp) ++tab_count[TABCOUNT_OFF_TEMP];
          if (c.source == TABLE_SRC_TABLE)
            ++tab_count[TABCOUNT_TABLE];
          else
            ++tab_count[TABCOUNT_FREEFREE];
        }
        Real *chi_c = t->cell_storage ? &tab_abs(kt,jt,it,0) : chi;
        Real *cum_c = t->cell_storage ? &tab_emis_cum(kt,jt,it,0) : cum;
        for (int l=0; l<nf; ++l) chi_c[l] = t->NodeExtinction(l, c, temp, rhoc);
        Real tot = t->EmissionCumulative(chi_c, temp, cum_c);
        // A masked cell emits TABLE_CUT_VALUE outright, whatever the Planck function at
        // its temperature made of the cut extinction; its cumulative array, normalized
        // from that shape, still gives the sampler a valid distribution for the
        // negligible weight it carries.
        if (c.source == TABLE_SRC_CUT) tot = TABLE_CUT_VALUE;
        tab_emis_tot(kt,jt,it) = tot;
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real MonteCarloBlock::TableEmissionCumulative(int i3, int i2, int i1,
//!                                                   const Real *&cum)
//! \brief the cell's normalized cumulative emission over the nodes; returns the total
//
// Cell storage hands back the stored array; global storage recomputes it into the
// block's scratch, which the caller must consume before the next call.

Real MonteCarloBlock::TableEmissionCumulative(int i3, int i2, int i1, const Real *&cum) {
  const OpacityTable *t = pmy_mc->optab;
  const int kt = i3-ks, jt = i2-js, it = i1-is;
  if (t->cell_storage) {
    cum = &tab_emis_cum(kt,jt,it,0);
    return tab_emis_tot(kt,jt,it);
  }
  const int nf = t->nfre;
  Real *chi = &tab_scratch_[0];
  Real *cum_s = &tab_scratch_[nf];
  const TableCell &c = tab_cell[TableIndex(kt,jt,it)];
  const Real temp = tgas(i3,i2,i1), rhoc = rho(i3,i2,i1);
  for (int l=0; l<nf; ++l) chi[l] = t->NodeExtinction(l, c, temp, rhoc);
  Real tot = t->EmissionCumulative(chi, temp, cum_s);
  if (c.source == TABLE_SRC_CUT) tot = TABLE_CUT_VALUE;
  cum = cum_s;
  return tot;
}
