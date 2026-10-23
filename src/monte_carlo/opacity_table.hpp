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
// tables of the file named by <opacitytable>/opacity_file, read once.  Each
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
//   optionally the Rosseland absorption, nfre blocks: the Rosseland group means without
//   the scattering of the table's own model (tops_regrid.py --rossabs)
// The free electrons per atom, <opacitytable>/nfree_file (tops_regrid.py's _nfree.txt):
//   comment lines (one of them "# Abar = <A>  Z_full = <Z>"), nT, nrho, T [keV], rho,
//   nT rows of nrho values of N_free; the grids must be the main table's.  With
//   table_ne the electron density of every cell is rho N_free / (Abar m_u).
// The energy axis must be uniform in log; the temperature and density axes may be
// uniform or not (a cell's bracket comes from the uniform-spacing formula corrected by a
// walk, which costs nothing on a uniform axis).
//
// Two-band split (<opacitytable>/two_band, mean reading with table_ne): each table group
// is split into a high band, frequency fraction w with absorption kappa_h, and a low
// band, fraction 1-w with kappa_l, with the cell's scattering kappa_s in both, such that
//   w kappa_h + (1-w) kappa_l = kappa_P,
//   [w/(kappa_h+kappa_s) + (1-w)/(kappa_l+kappa_s)]^-1 = kappa_R,
// so that emission uses the Planck mean and diffusion the Rosseland one with Kirchhoff's
// law holding in each band.  The closure is w = two_band_w, falling back to kappa_l = 0
// where that would need kappa_l < 0 (two_band_closure = gap0 takes kappa_l = 0 always),
// and the Rosseland absorption is floored at two_band_floor times kappa_s; a group whose
// Rosseland absorption reaches its Planck mean stays one band.  A photon carries a band
// rank u (Photon::ubp): u < w(cell) puts it in the high band of its group, the same
// frequencies in every cell; u is redrawn after every electron scattering and whenever
// the photon's group changes.  Where the high band is optically thick across the cell,
// rho kappa_h W > two_band_thick (0 = never), it is in equilibrium with the gas and is
// not transported: its emission is dropped and the low band receives instead what the
// high band would have passed to it by scattering, w kappa_h p_s (1-w)/(1 - w p_s) with
// p_s = kappa_s/(kappa_h + kappa_s), and a photon in the high band there is absorbed on
// the spot.
// two_band_decorrelate redraws the rank at a cell crossing with that probability, to
// measure what the correlated-k assumption carries: 0 and 1 bracket the truth.
//
// Group reading (<opacitytable>/table_groups):
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
enum TableClosure {TABLE_CLOSURE_W = 0, TABLE_CLOSURE_GAP0 = 1};

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
  // tables, kappa in cm^2/g: (group, item, irho); the logs of positive values
  AthenaArray<Real> plan, lplan, ross, lross, gray_ross, gray_plan;
  // the Rosseland absorption block, when the file has one
  bool have_rossabs;
  AthenaArray<Real> rossabs, lrossabs;
  // free electrons per atom (item, irho) and the mean atomic weight [m_u] of the file
  AthenaArray<Real> nfree;
  Real abar;
  bool have_nfree;       // an nfree_file was read
  bool table_ne;         // <opacitytable>/table_ne: electron density from N_free
  Real nfree_min_, nfree_max_;  // range of the file, for the report
  // the two-band split
  bool two_band;
  Real tb_w0, tb_floor, tb_thick;
  TableClosure tb_closure;
  //! probability of redrawing the band rank at a cell crossing (two_band_decorrelate):
  //! 0 keeps a band the same frequencies in every cell (correlated k), 1 makes every
  //! cell's lines unrelated to its neighbours', which returns the Planck-only transport
  Real tb_decorrelate;

  // options
  TableGroupReading groups;
  TableOffGrid offgrid;
  TableAbove above;
  Real emax;             // [erg] emission extends to here with table_above = freefree
  bool cell_storage;     // <opacitytable>/table_storage = cell (true) or global
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
  //! the same for the Rosseland group mean (with the file's scattering) and for the
  //! Rosseland absorption block
  Real RosselandNode(int l, const TableCell &cell) const;
  Real RossAbsNode(int l, const TableCell &cell) const;
  //! the two-band split of one group from its Planck mean kP, Rosseland absorption kRa
  //! and the scattering kappa_s to be used with it, all per gram
  void Split(Real kP, Real kRa, Real kappa_s, Real &w, Real &kh, Real &kl) const;
  //! the bands of extinction l of a cell [1/cm]: the Planck value chi, the high and low
  //! band values and the high-band fraction (1 where the group is one band); kappa_s is
  //! the cell's scattering per gram for the group
  void Bands(int l, const TableCell &cell, Real temp, Real rho, Real kappa_s, Real &chi,
             Real &chi_h, Real &chi_l, Real &w) const;
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
