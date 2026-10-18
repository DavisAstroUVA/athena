#ifndef MONTE_CARLO_OPACITY_TABLE_HPP_
#define MONTE_CARLO_OPACITY_TABLE_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file opacity_table.hpp
//! \brief tabulated multigroup opacities (emission = table, absorption = table)
//
// One OpacityTable per rank, owned by MonteCarlo: the (group, temperature, density)
// tables of the file named by <montecarlo>/opacity_file, read once.  Each
// MonteCarloBlock caches where its cells sit on the (T, rho) grid and, in cell storage
// mode, the interpolated extinction per group and the cumulative emission spectrum of
// every cell; in global storage mode those are recomputed from this table at every
// call through the same functions (see MonteCarloBlock::TableExtinction).
//
// File layout (text, whitespace separated; what tops_regrid.py writes):
//   nfre, ntem, nrho
//   group lower bounds [keV] (nfre), temperatures [keV] (ntem), densities [g/cm^3] (nrho)
//   grey Rosseland (ntem rows of nrho), grey Planck (same)
//   Rosseland group means, nfre blocks of (ntem rows of nrho), including scattering
//   Planck group means, nfre blocks, absorption only; all in cm^2/g
//
// Group reading (<montecarlo>/table_groups):
//   edge  the k-th value is taken as the opacity at the k-th lower bound and the
//         extinction is interpolated linearly in log E between consecutive bounds; the
//         emission spectrum is the trapezoid in ln nu of B_nu kappa at those nodes.  It
//         never emits in the top group, and it reads a Planck group mean, which is a mean
//         over [E_k, E_k+1), as the value at E_k: for a nu^-3 opacity that is about 30%
//         low.

// C++ headers
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"

class ParameterInput;

enum TableGroupReading {TABLE_GROUPS_EDGE = 0};
enum TableOffGrid {TABLE_OFFGRID_FREEFREE = 0, TABLE_OFFGRID_CLAMP = 1};
enum TableAbove {TABLE_ABOVE_CLAMP = 0};

//! \struct TableCell
//! \brief where a cell sits on the (T, rho) grid: the lower bracket indices, the
//! fractional positions in log rho (xi) and the table's temperature coordinate (xj),
//! and how the cell is served
struct TableCell {
  int irho, item;
  Real xi, xj;
  int source;  // TABLE_SRC_*
};
enum TableSource {TABLE_SRC_TABLE = 0, TABLE_SRC_FREEFREE = 1, TABLE_SRC_CUT = 2};

//! value given to the extinction of every group and to the emission of a cell a problem
//! generator has masked out (MonteCarloBlock::tab_mask = 0)
constexpr Real TABLE_CUT_VALUE = 1.e-20;

class OpacityTable {
 public:
  OpacityTable();
  ~OpacityTable() = default;

  //! read the file; heabund is the helium fraction of the free-free fallback
  void Read(ParameterInput *pin);

  // grids
  int nfre, ntem, nrho;
  AthenaArray<Real> egrid;  // group lower bounds [erg]
  AthenaArray<Real> tgrid;  // [K]
  AthenaArray<Real> rgrid;  // [g/cm^3]
  // log10 bounds and spacings; the energy and density axes are taken as uniform in log,
  // the temperature axis is searched
  Real lmine, lmaxe, dle, lmint, lmaxt, dlt, lmind, lmaxd, dld;
  // tables, kappa in cm^2/g: (group, item, irho); the log of a positive Planck value
  AthenaArray<Real> plan, lplan, ross, gray_ross, gray_plan;

  // options
  TableGroupReading groups;
  TableOffGrid offgrid;
  TableAbove above;
  bool cell_storage;     // <montecarlo>/table_storage = cell (true) or global
  bool gray_to_freefree; // rows the same in every group are replaced by free-free
  bool fill_freefree;    // the whole table is replaced by free-free (a test knob)
  Real heabund;          // of the free-free fallback
  int ngray_rows;        // rows replaced at load

  //! the cell's bracket and weights on the grid, and whether it is on the grid: density
  //! by formula, temperature by formula then a walk, both clamped; off the grid is a
  //! weight outside [0,1] on either axis.
  void Locate(Real temp, Real rho, TableCell &c, bool &off_rho, bool &off_temp) const;
  //! Planck group value at node l for a cell on the grid [cm^2/g]: log-log bilinear,
  //! linear where a corner is not positive
  Real PlanckNode(int l, const TableCell &c) const;
  //! free-free extinction [1/cm] of the fallback, fully ionized H + He
  Real FreeFree(Real temp, Real rho, Real energy) const;
  //! B_nu(T) [cgs]
  static Real PlanckNu(Real temp, Real nu);

  //! the extinction at node l [1/cm] of a cell with the given state
  Real NodeExtinction(int l, const TableCell &c, Real temp, Real rho) const;
  //! the cell's normalized cumulative emission over the nodes (nfre values, cum[0] = 0)
  //! and its total [erg/cm^3/s]; the arrays are the caller's.  The extinction nodes are
  //! supplied so that cell storage can pass its stored values.
  Real EmissionCumulative(const Real *chi, Real temp, Real *cum) const;
  //! the node bracket of an energy under the edge reading, and the weight of node k+1
  void EnergyNode(Real energy, int &k, Real &xk) const;
};

#endif  // MONTE_CARLO_OPACITY_TABLE_HPP_
