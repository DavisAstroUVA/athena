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
// mode, the extinction per group and the cumulative emission spectrum of every cell; in
// global storage mode those are recomputed from this table at every call through the
// same functions (see MonteCarloBlock::TableExtinction).
//
// File layout (text, whitespace separated; what tops_regrid.py writes):
//   nfre, ntem, nrho
//   group lower bounds [keV] (nfre), temperatures [keV] (ntem), densities [g/cm^3] (nrho)
//   grey Rosseland (ntem rows of nrho), grey Planck (same)
//   Rosseland group means, nfre blocks of (ntem rows of nrho), including scattering
//   Planck group means, nfre blocks, absorption only; all in cm^2/g
// The free electrons per atom, <montecarlo>/nfree_file (tops_regrid.py's _nfree.txt):
//   comment lines (one of them "# Abar = <A>  Z_full = <Z>"), nT, nrho, T [keV], rho,
//   nT rows of nrho values of N_free; the grids must be the main table's.  With
//   table_ne the electron density of every cell is rho N_free / (Abar m_u).
// The energy axis must be uniform in log; the temperature and density axes may be
// uniform or not (a cell's bracket comes from the uniform-spacing formula corrected by a
// walk, which costs nothing on a uniform axis).
//
// Group reading (<montecarlo>/table_groups):
//   mean  the k-th Planck value is the opacity throughout group k, [E_k, E_k+1), the top
//         edge continuing the geometric spacing.  Emission in a group is kappa_k times
//         the photon Planck integral over the group, and the energy within a group is
//         drawn from B_nu / (h nu).  With table_above = freefree and table_emax above
//         the top edge, further groups of the same spacing carry the Planck group means
//         of free-free, so hot gas keeps its emission above the table.
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

enum TableGroupReading {TABLE_GROUPS_MEAN = 0, TABLE_GROUPS_EDGE = 1};
enum TableOffGrid {TABLE_OFFGRID_FREEFREE = 0, TABLE_OFFGRID_CLAMP = 1};
enum TableAbove {TABLE_ABOVE_CLAMP = 0, TABLE_ABOVE_FREEFREE = 1, TABLE_ABOVE_NONE = 2};

//! \struct TableCell
//! \brief where a cell sits on the (T, rho) grid: the lower bracket indices, the
//! fractional positions in log rho (xi) and log T (xj), and how the cell is served
struct TableCell {
  int irho, item;
  Real xi, xj;
  int source;  // TABLE_SRC_*
};
enum TableSource {TABLE_SRC_TABLE = 0, TABLE_SRC_FREEFREE = 1, TABLE_SRC_CUT = 2};

//! value given to the extinction of every group and to the emission of a cell a problem
//! generator has masked out (MonteCarloBlock::tab_mask = 0)
constexpr Real TABLE_CUT_VALUE = 1.e-20;

//! \fn Real PlanckPhotonCumulative(Real x)
//! \brief fraction of a blackbody's photons below x = h nu / k T, int_0^x t^2/(e^t - 1)
//! dt / (2 zeta(3)), from a table built on first use
Real PlanckPhotonCumulative(Real x);
//! \fn Real PlanckPhotonInverse(Real prob)
//! \brief x at which PlanckPhotonCumulative reaches prob
Real PlanckPhotonInverse(Real prob);

class OpacityTable {
 public:
  OpacityTable();
  ~OpacityTable() = default;

  //! read the file and the options
  void Read(ParameterInput *pin);
  //! read the free electrons per atom (called by Read when nfree_file is given)
  void ReadNFree(const std::string &fname);

  // grids
  int nfre, ntem, nrho;
  AthenaArray<Real> egrid;  // group lower bounds [erg]
  AthenaArray<Real> tgrid;  // [K]
  AthenaArray<Real> rgrid;  // [g/cm^3]
  // log10 bounds and spacings
  Real lmine, lmaxe, dle, lmint, lmaxt, dlt, lmind, lmaxd, dld;
  bool uniform_t, uniform_rho;  // axes uniform in log (to 1e-6 of the spacing)
  // tables, kappa in cm^2/g: (group, item, irho); the log of a positive Planck value
  AthenaArray<Real> plan, lplan, ross, gray_ross, gray_plan;
  // free electrons per atom (item, irho) and the mean atomic weight [m_u] of the file
  AthenaArray<Real> nfree;
  Real abar;
  bool have_nfree;       // an nfree_file was read
  bool table_ne;         // <montecarlo>/table_ne: electron density from N_free
  Real nfree_min_, nfree_max_;  // range of the file, for the report

  // options
  TableGroupReading groups;
  TableOffGrid offgrid;
  TableAbove above;
  Real emax;             // [erg] emission extends to here with table_above = freefree
  bool cell_storage;     // <montecarlo>/table_storage = cell (true) or global
  bool gray_to_freefree; // rows the same in every group are replaced by free-free
  bool fill_freefree;    // the whole table is replaced by free-free (a test knob)
  Real heabund;          // of the free-free fallback
  int ngray_rows;        // rows replaced at load

  // The group structure the block tables are built on.  mean: ngroups = nfre plus the
  // free-free extension, gedge the ngroups+1 edges, nabs = ngroups extinctions per cell;
  // edge: ngroups = nfre-1 node intervals, gedge the nfre nodes, nabs = nfre node
  // extinctions.  A cell's cumulative emission has ngroups+1 entries either way.
  int ngroups, nabs;
  AthenaArray<Real> gedge;  // [erg]

  //! the cell's bracket and weights on the grid, and whether it is on the grid: density
  //! and temperature by formula, the latter corrected by a walk; off the grid is a
  //! weight outside [0,1] on either axis.
  void Locate(Real temp, Real rho, TableCell &cell, bool &off_rho, bool &off_temp) const;
  //! Planck group value l for a cell on the grid [cm^2/g]: log-log bilinear, linear
  //! where a corner is not positive
  Real PlanckNode(int l, const TableCell &cell) const;
  //! free-free extinction [1/cm] of fully ionized H + He at one energy
  Real FreeFree(Real temp, Real rho, Real energy) const;
  //! its Planck mean over [elo, ehi] [1/cm]
  Real FreeFreeGroupMean(Real elo, Real ehi, Real temp, Real rho) const;
  //! free electrons per atom of a cell: bilinear in the cell's bracket, which is the
  //! main table's, clamped at the grid's edges
  Real NFree(const TableCell &cell) const;
  //! the electron density [1/cm^3] of gas at (temp, rho) from N_free
  Real ElectronDensity(Real temp, Real rho) const;
  //! B_nu(T) [cgs]
  static Real PlanckNu(Real temp, Real nu);
  //! B_nu / (h nu) integrated over [elo, ehi] [photons/cm^2/s/sr]
  static Real PlanckPhotonBand(Real temp, Real elo, Real ehi);

  //! extinction l of a cell [1/cm]: group l (mean) or node l (edge)
  Real Extinction(int l, const TableCell &cell, Real temp, Real rho) const;
  //! the cell's normalized cumulative emission over its ngroups+1 edges and its total
  //! [photons/cm^3/s]; chi holds the cell's nabs extinctions, the arrays are the caller's
  Real EmissionCumulative(const Real *chi, Real temp, Real *cum) const;
  //! energy drawn within group (or node interval) g at fraction a of its emission
  Real DrawEnergy(int g, Real a, Real temp) const;
  //! edge reading: the node bracket of an energy and the weight of node k+1
  void EnergyNode(Real energy, int &k, Real &xk) const;
  //! mean reading: the group of an energy; -1 below the bottom edge, -2 above the top
  int GroupIndex(Real energy) const;
};

#endif  // MONTE_CARLO_OPACITY_TABLE_HPP_
