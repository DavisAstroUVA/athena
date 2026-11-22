//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file ddmc_line.cpp
//! \brief the frequency lattice of a resonance line for discrete diffusion (<ddmc> line):
//!        the grid of frequency bins, the bin-averaged line extinction of each cell, the
//!        leakage rates between bins, and the hops and emergences through frequency faces
//
// The method is that of Smith, Tsang, Bromm & Milosavljevic (2018): in a medium thick
// in a resonance line the frequency redistribution of the scattering is, in the
// Fokker-Planck limit of the wings, a diffusion in frequency, d/dx (k_x / 2 dJ/dx) with
// k_x the extinction at frequency x, which a finite-volume discretization in bins turns
// into
// leakage rates between neighbouring bins exactly as the spatial diffusion is turned into
// leakage rates between cells.  A node of the lattice is a (cell, bin); the walker dwells
// at the node's total rate, the six spatial rates of ddmc.cpp formed with the
// bin-averaged extinction k_ij in place of the Thomson one, and the two frequency rates
// (their equation 11)
//
//   k_freq = 1 / (dx_j (dx_j / k_ij + dx_j' / k_ij'))   per unit comoving path,
//
// the harmonic mean of the two bins' extinctions, with dx in the cell's own Doppler
// units.
//
// The grid is one grid of comoving frequencies for the whole mesh, its bin edges fixed in
// the Doppler units of the reference temperature line_temp (a core bin of half-width
// core_width about the line centre, wing bins of width dx out to xmax, and a virtual bin
// beyond each end that is never a node but gives the edge bins their outer rate), so
// that a spatial hop keeps the walker's energy and only its cell's temperature changes
// the extinction it sees.  Each cell sees the grid through its own Doppler width: its
// bin-averaged extinction is (their equations 13 and 14)
//
//   k_ij = n_H sigma_0 / (sqrt(pi) dnu_D) (Hbar(a, x+) - Hbar(a, x-)) / (x+ - x-),
//   Hbar(a, x) = int_0^x H(a, y) dy
//              = sqrt(pi)/2 erf(x) - 2 a / sqrt(pi) F(x) + a^2 x exp(-x^2) + O(a^3),
//
// F the Dawson integral, x+- the bin's edges in the cell's units.  A bin is a node of a
// cell when k_ij times the cell's smallest width is at least <ddmc> tau; since k_ij falls
// away from the line centre, the nodes of a cell are a band of bins about the core,
// stored as the outermost node bin (ddmc_line_edge), and a cell with any node is a
// diffusion cell (its core bin is one).
//
// The walker's bin is the lattice's state: a spatial hop keeps the comoving energy, a
// frequency hop draws the energy uniformly in the bin it enters, and an emergence
// through a spatial face draws it uniformly in the walker's own bin.  A walker leaves
// the lattice in frequency by leaking into a bin that is not a node of its cell (the
// thin wings, or beyond the grid): it emerges where it is, isotropic, at the frequency
// of the edge it crossed plus the outward overshoot of a scattering step, a uniform
// fraction of a Rayleigh step of unit variance in the cell's Doppler units, which is
// where a kinetic crossing from a uniform population lands.  A transport photon enters
// the lattice in frequency by scattering:
// the analog kernel redistributes it and, if its new frequency lies in a node of its
// cell, DDMCBeforeFlight converts it at the start of its next flight.  An entry from a
// bin that is not a node into a wing node is admitted with the probability that
// balances the lattice's leakage out of that bin against the kernel's crossings of its
// edge, P = sqrt(2 pi) / (2 dx) with dx the bin's width in the cell's Doppler units,
// and is otherwise reflected about the edge (DDMCLineEntry); an entry into the core
// bin, whose kernel redistributes over the core at the next scattering, is always
// admitted.  In a moving medium the velocity divergence shifts the
// walker's energy over a dwell as it does on the grey lattice, and a shift across a bin
// edge is a crossing of that frequency face with nothing deposited.
//
// Lyman alpha only, with the module's constants and species(0) as the scatterer
// density, whichever scattering method the run names for its transport photons.

// C++ headers
#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../globals.hpp"
#include "../parameter_input.hpp"
#include "mcutils.hpp"
#include "montecarlo.hpp"
#include "photon.hpp"
#include "photonpusher.hpp"

namespace {

//! the Dawson integral F(x) = exp(-x^2) int_0^x exp(t^2) dt.  Within |x| <= 6 from the
//! plasma dispersion function of ZetaVoigt at a real argument, F = -Re Z / 2, whose
//! absolute accuracy of 1e-7 is ample there; beyond, the asymptotic series, since the
//! bin averages of the far wing are differences of nearly equal values of Hbar that a
//! fixed absolute error would swamp
Real Dawson(Real x) {
  if (std::abs(x) <= 6.) {
    std::complex<Real> z(x, 0.);
    return -0.5*ZetaVoigt(z).real();
  }
  // F(x) = sum_n (2n-1)!! / (2^(n+1) x^(2n+1)); eight terms at |x| >= 6 are within 1e-9
  const Real x2 = x*x;
  Real term = 0.5/x, sum = term;
  for (int n=1; n<=8; ++n) {
    term *= (2*n - 1)/(2.*x2);
    sum += term;
  }
  return sum;
}

//! Hbar(a, x), the integral of the Voigt function from 0 to x, to second order in a
Real VoigtIntegral(Real a, Real x) {
  return 0.5*std::sqrt(PI)*std::erf(x) - 2.*a/std::sqrt(PI)*Dawson(x)
         + a*a*x*std::exp(-x*x);
}

//! the Doppler width of the line at temperature T, in Hz
Real DopplerWidth(Real T) {
  const Real vth = std::sqrt(2.*MCConstants::kb_cgs*T/MCConstants::mH_cgs);
  return MCConstants::nu_lya*vth/MCConstants::c_cgs;
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn Real LineCoreVisit(Real a, Real xc, MCRandom *pran, std::vector<Real> &over)
//! \brief the mean number of scatterings of a visit to the Doppler core |x| < xc at
//!        damping parameter a, and the overshoot |x| - xc of each visit's exit.  One
//!        photon is scattered in frequency alone (no transport, no escape) with the
//!        kernel of ScatterResonanceLine; a visit opens at a scattering that carries it
//!        into the core and closes at the first that carries it out.  Entries are then
//!        drawn from the stationary distribution, which is what the lattice admits with
//!        probability 1, and the mean visit is the inverse of the exit probability per
//!        scattering that balances them.  The chain runs to kVisitMin visits and
//!        kVisitScatter scatterings, or kVisitMax visits, so that short visits are
//!        sampled enough to resolve the tail of their exits and a long one costs no
//!        more than its own scatterings

const int kVisitMin = 200, kVisitMax = 20000;
const Real kVisitScatter = 1.e6;

Real LineCoreVisit(Real a, Real xc, MCRandom *pran, std::vector<Real> &over) {
  auto scatter = [&](Real x) {
    Real cgam;
    do {
      cgam = 2.*pran->uniform() - 1.;
    } while (2.*pran->uniform() > 1. + SQR(cgam));
    const Real sgam = std::sqrt(1. - SQR(cgam));
    const Real uperp = std::sqrt(-std::log(std::max(pran->uniform(), 1.e-300)))
                       *std::cos(2.*PI*pran->uniform());
    const Real upar = SampleVelocityParallel(a, x, pran);
    return x + (cgam - 1.)*upar + sgam*uperp;
  };
  over.clear();
  Real nsum = 0.;
  Real x = 0.;
  int64_t nsc = 0;
  bool first = true;  // the visit the chain starts inside is not a whole one
  while (static_cast<int>(over.size()) < kVisitMax
         && (static_cast<int>(over.size()) < kVisitMin || nsum < kVisitScatter)) {
    const Real xn = scatter(x);
    const bool was_in = std::abs(x) < xc, now_in = std::abs(xn) < xc;
    if (was_in) ++nsc;
    if (was_in && !now_in) {
      if (!first) {
        over.push_back(std::abs(xn) - xc);
        nsum += static_cast<Real>(nsc);
      }
      first = false;
      nsc = 0;
    }
    x = xn;
  }
  return nsum/static_cast<Real>(over.size());
}

//! the extent of the x_loc grid
const Real kVisitStep = 0.25;

//! the quantile levels of the exit overshoots: NVQ levels u = 1 - (1 - t)^2 on t uniform
//! in [0, 1], denser toward 1 so that the tail, which reaches the thin frequencies,
//! is resolved
Real VisitQuantileLevel(int q, int nq) {
  const Real t = static_cast<Real>(q)/(nq - 1);
  return 1. - SQR(1. - t);
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::BuildDDMCLineVisitTable()
//! \brief the two-dimensional visit table over the core node's local half-width x_loc and
//!        log10 of the Doppler scale s (the same seed on every rank, so every rank holds
//!        the same table), each point one chain of LineCoreVisit.  At each s the rows up to the first grid point at or above
//!        the widest node reachable there, core_width s, are filled; the others are left
//!        at zero.  The exit overshoots of each point are kept as NVQ quantiles

void MonteCarlo::BuildDDMCLineVisitTable() {
  MCRandom ran(20261031);
  const Real a_line = MCConstants::lorwidth_lya/ddmc_line_dnu;
  const int ns = static_cast<int>(ddmc_line_vlogs.size());
  const int nx = static_cast<int>(ddmc_line_vx.size());
  ddmc_line_visit.assign(nx*ns, 0.);
  ddmc_line_over.assign(nx*ns*NVQ, 0.);
  std::vector<Real> over;
  for (int j=0; j<ns; ++j) {
    const Real s = std::pow(10., ddmc_line_vlogs[j]);
    const Real xmax = ddmc_line_core*s;
    for (int i=0; i<nx; ++i) {
      if (i > 0 && ddmc_line_vx[i-1] >= xmax) break;
      const int n = i*ns + j;
      ddmc_line_visit[n] = LineCoreVisit(a_line*s, ddmc_line_vx[i], &ran, over);
      std::sort(over.begin(), over.end());
      const int nv = static_cast<int>(over.size());
      for (int q=0; q<NVQ; ++q) {
        const Real pos = VisitQuantileLevel(q, NVQ)*(nv - 1);
        const int lo = std::min(static_cast<int>(pos), nv - 2);
        const Real f = pos - lo;
        ddmc_line_over[n*NVQ + q] = (1. - f)*over[lo] + f*over[lo+1];
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn bool MonteCarlo::ReadDDMCLineVisitTable()
//! \brief the visit table from ddmc_line_visit_file, if the file exists and its header
//!        (line temperature, core width and minimum, visits per point, the grids) is the
//!        table this run would build

bool MonteCarlo::ReadDDMCLineVisitTable() {
  if (ddmc_line_visit_file.empty()) return false;
  std::ifstream in(ddmc_line_visit_file.c_str());
  if (!in) return false;
  std::string tag;
  Real temp, core, cmin, step;
  int nvisit, nq, ns, nx;
  in >> tag >> temp >> core >> cmin >> nvisit >> nq >> step >> ns >> nx;
  if (!in || tag != "ddmc_line_visit_table" || temp != ddmc_line_temp
      || core != ddmc_line_core || cmin != ddmc_line_core_min || nvisit != kVisitMax
      || nq != NVQ || step != kVisitStep || ns != static_cast<int>(ddmc_line_vlogs.size())
      || nx != static_cast<int>(ddmc_line_vx.size()))
    return false;
  for (int j=0; j<ns; ++j) {
    Real v;
    in >> v;
    if (!in || std::abs(v - ddmc_line_vlogs[j]) > 1.e-12) return false;
  }
  for (int i=0; i<nx; ++i) {
    Real v;
    in >> v;
    if (!in || std::abs(v - ddmc_line_vx[i]) > 1.e-12) return false;
  }
  ddmc_line_visit.assign(nx*ns, 0.);
  ddmc_line_over.assign(nx*ns*NVQ, 0.);
  for (int n=0; n<nx*ns; ++n) {
    in >> ddmc_line_visit[n];
    for (int q=0; q<NVQ; ++q) in >> ddmc_line_over[n*NVQ + q];
  }
  return static_cast<bool>(in);
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::WriteDDMCLineVisitTable() const
//! \brief the visit table to ddmc_line_visit_file (rank 0)

void MonteCarlo::WriteDDMCLineVisitTable() const {
  if (ddmc_line_visit_file.empty() || Globals::my_rank != 0) return;
  std::ofstream out(ddmc_line_visit_file.c_str());
  if (!out) {
    std::cout << "frequency lattice: could not write the visit table to "
              << ddmc_line_visit_file << std::endl;
    return;
  }
  const int ns = static_cast<int>(ddmc_line_vlogs.size());
  const int nx = static_cast<int>(ddmc_line_vx.size());
  out.precision(17);
  out << "ddmc_line_visit_table " << ddmc_line_temp << " " << ddmc_line_core << " "
      << ddmc_line_core_min << " " << kVisitMax << " " << NVQ << " " << kVisitStep
      << " " << ns << " " << nx << "\n";
  for (int j=0; j<ns; ++j) out << ddmc_line_vlogs[j] << (j < ns-1 ? " " : "\n");
  for (int i=0; i<nx; ++i) out << ddmc_line_vx[i] << (i < nx-1 ? " " : "\n");
  for (int n=0; n<nx*ns; ++n) {
    out << ddmc_line_visit[n];
    for (int q=0; q<NVQ; ++q) out << " " << ddmc_line_over[n*NVQ + q];
    out << "\n";
  }
}

namespace {

//! the bilinear weights of (xloc, log10 s) on the visit table: the s interval j, j+1
//! with fraction fs, and at each of the two rows the x interval and fraction, x clamped
//! to the row's filled range
struct VisitWeights {
  int j, i[2];
  Real fs, fx[2];
};

VisitWeights LocateVisit(const std::vector<Real> &vlogs, const std::vector<Real> &vx,
                         const std::vector<Real> &visit, Real xloc, Real s) {
  VisitWeights w;
  const int ns = static_cast<int>(vlogs.size()), nx = static_cast<int>(vx.size());
  const Real logs = std::min(std::max(std::log10(s), vlogs[0]), vlogs[ns-1]);
  w.j = 0;
  while (w.j < ns - 2 && vlogs[w.j+1] <= logs) ++w.j;
  w.fs = (logs - vlogs[w.j])/(vlogs[w.j+1] - vlogs[w.j]);
  for (int r=0; r<2; ++r) {
    const int jj = w.j + r;
    // the row's filled range
    int ifill = 0;
    while (ifill < nx - 1 && visit[(ifill+1)*ns + jj] > 0.) ++ifill;
    const Real x = std::min(std::max(xloc, vx[0]), vx[ifill]);
    int i = 0;
    while (i < ifill - 1 && vx[i+1] <= x) ++i;
    w.i[r] = i;
    w.fx[r] = (ifill > 0) ? (x - vx[i])/(vx[i+1] - vx[i]) : 0.;
    if (ifill == 0) w.fx[r] = 0.;
  }
  return w;
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn Real MonteCarlo::DDMCLineCoreVisit(Real xloc, Real s) const
//! \brief the scatterings of a kinetic visit to a core node of local half-width xloc at
//!        Doppler scale s, bilinear in log of the table, clamped to its range

Real MonteCarlo::DDMCLineCoreVisit(Real xloc, Real s) const {
  const VisitWeights w = LocateVisit(ddmc_line_vlogs, ddmc_line_vx, ddmc_line_visit,
                                     xloc, s);
  const int ns = static_cast<int>(ddmc_line_vlogs.size());
  Real lv[2];
  for (int r=0; r<2; ++r) {
    const int jj = w.j + r, i = w.i[r];
    const Real v0 = ddmc_line_visit[i*ns + jj];
    const Real v1 = (w.fx[r] > 0.) ? ddmc_line_visit[(i+1)*ns + jj] : v0;
    lv[r] = (1. - w.fx[r])*std::log(std::max(v0, 1.)) + w.fx[r]*std::log(std::max(v1, 1.));
  }
  return std::exp((1. - w.fs)*lv[0] + w.fs*lv[1]);
}

//----------------------------------------------------------------------------------------
//! \fn Real MonteCarlo::DDMCLineCoreExit(Real xloc, Real s, MCRandom *pran) const
//! \brief the overshoot beyond the edge of a core node of local half-width xloc at scale
//!        s of the scattering that leaves it, local Doppler units: one quantile level
//!        drawn, its value interpolated between the table's quantiles and bilinearly in
//!        (xloc, log s)

Real MonteCarlo::DDMCLineCoreExit(Real xloc, Real s, MCRandom *pran) const {
  const VisitWeights w = LocateVisit(ddmc_line_vlogs, ddmc_line_vx, ddmc_line_visit,
                                     xloc, s);
  const int ns = static_cast<int>(ddmc_line_vlogs.size());
  // the level u uniform, its position among the quantile levels u = 1 - (1 - t)^2:
  // t = 1 - sqrt(1 - u)
  const Real pos = (1. - std::sqrt(std::max(1. - pran->uniform(), 0.)))*(NVQ - 1);
  const int q = std::min(static_cast<int>(pos), NVQ - 2);
  const Real fq = pos - q;
  auto at = [&](int i, int jj) {
    const Real *o = &ddmc_line_over[(i*ns + jj)*NVQ];
    return (1. - fq)*o[q] + fq*o[q+1];
  };
  Real d[2];
  for (int r=0; r<2; ++r) {
    const int jj = w.j + r, i = w.i[r];
    const Real d0 = at(i, jj);
    const Real d1 = (w.fx[r] > 0.) ? at(i+1, jj) : d0;
    d[r] = (1. - w.fx[r])*d0 + w.fx[r]*d1;
  }
  return std::max((1. - w.fs)*d[0] + w.fs*d[1], 0.);
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::SetupDDMCLine(ParameterInput *pin)
//! \brief read the <ddmc> line keys and lay out the frequency grid

void MonteCarlo::SetupDDMCLine(ParameterInput *pin) {
  const std::string name = pin->GetOrAddString("ddmc","line","none");
  ddmc_line = DDMC_LINE_NONE;
  ddmc_line_nwing = 0;
  ddmc_line_nbin = 0;
  ddmc_line_temp = ddmc_line_xmax = ddmc_line_dx = ddmc_line_core = ddmc_line_dnu = 0.;
  ddmc_line_core_min = 0.;
  if (name == "none") return;
  std::stringstream msg;
  if (name != "lya") {
    msg << "### FATAL ERROR in MonteCarlo constructor" << std::endl
        << "<ddmc>/line must be none or lya, not " << name << std::endl;
    ATHENA_ERROR(msg);
  }
  if (!ddmc && !ddmc_report) {
    msg << "### FATAL ERROR in MonteCarlo constructor" << std::endl
        << "<ddmc> line = lya needs enable = true or report = true" << std::endl;
    ATHENA_ERROR(msg);
  }
  if (scattering_meth != SCATRES && scattering_meth != SCATUSER) {
    msg << "### FATAL ERROR in MonteCarlo constructor" << std::endl
        << "<ddmc> line = lya needs <montecarlo> scattering = resonance or user"
        << std::endl;
    ATHENA_ERROR(msg);
  }
  ddmc_line = DDMC_LINE_LYA;
  ddmc_line_temp = pin->GetReal("ddmc","line_temp");
  ddmc_line_xmax = pin->GetOrAddReal("ddmc","xmax",40.);
  ddmc_line_dx = pin->GetOrAddReal("ddmc","dx",1.);
  ddmc_line_core = pin->GetOrAddReal("ddmc","core_width",3.3);
  ddmc_line_core_min = pin->GetOrAddReal("ddmc","core_min",2.);
  ddmc_line_visit_file = pin->GetOrAddString("ddmc","visit_file","");
  if (!(ddmc_line_temp > 0.) || !(ddmc_line_dx > 0.) || !(ddmc_line_core > 0.)
      || !(ddmc_line_xmax > ddmc_line_core)) {
    msg << "### FATAL ERROR in MonteCarlo constructor" << std::endl
        << "<ddmc> line_temp, dx and core_width must be positive and xmax above "
        << "core_width" << std::endl;
    ATHENA_ERROR(msg);
  }
  if (!(ddmc_line_core_min > 0.)) {
    msg << "### FATAL ERROR in MonteCarlo constructor" << std::endl
        << "<ddmc> core_min must be positive (the smallest core node, in the cell's "
        << "Doppler widths)" << std::endl;
    ATHENA_ERROR(msg);
  }
  // wing bins of width dx from the core's edge out to at least xmax
  ddmc_line_nwing = static_cast<int>(
      std::ceil((ddmc_line_xmax - ddmc_line_core)/ddmc_line_dx - 1.e-9));
  ddmc_line_nbin = 2*ddmc_line_nwing + 1;
  ddmc_line_dnu = DopplerWidth(ddmc_line_temp);
  // the edges of the extended grid, the virtual bins included: nbin + 3 of them
  const int nedge = ddmc_line_nbin + 3;
  ddmc_line_xedge.assign(nedge, 0.);
  ddmc_line_eedge.assign(nedge, 0.);
  for (int b=0; b<nedge; ++b) {
    // b = nwing + 1 is the core's lower edge, nwing + 2 its upper edge
    Real x;
    if (b <= ddmc_line_nwing + 1)
      x = -(ddmc_line_core + (ddmc_line_nwing + 1 - b)*ddmc_line_dx);
    else
      x = ddmc_line_core + (b - ddmc_line_nwing - 2)*ddmc_line_dx;
    ddmc_line_xedge[b] = x;
    ddmc_line_eedge[b] = MCConstants::h_cgs*(MCConstants::nu_lya + x*ddmc_line_dnu);
  }
  // the core node's exit rate is the kinetic one: the scatterings of a core visit and
  // its exit overshoot, tabulated over the node's local half-width and the cell's
  // Doppler scale
  {
    const int ns = 12;
    ddmc_line_vlogs.assign(ns, 0.);
    for (int n=0; n<ns; ++n) ddmc_line_vlogs[n] = -0.7 + 0.1*n;
    const Real xtop = ddmc_line_core*std::pow(10., ddmc_line_vlogs[ns-1]);
    const int nx = static_cast<int>(std::ceil((xtop - ddmc_line_core_min)/kVisitStep
                                              - 1.e-9)) + 2;
    ddmc_line_vx.assign(nx, 0.);
    for (int i=0; i<nx; ++i) ddmc_line_vx[i] = ddmc_line_core_min + i*kVisitStep;
    const bool read = ReadDDMCLineVisitTable();
    if (!read) {
      BuildDDMCLineVisitTable();
      WriteDDMCLineVisitTable();
    }
    if (Globals::my_rank == 0)
      std::cout << "frequency lattice: Lyman alpha, " << ddmc_line_nbin
                << " bins, core |x| <= " << ddmc_line_core << ", " << ddmc_line_nwing
                << " wing bins of " << ddmc_line_dx << " on each side to |x| = "
                << -ddmc_line_xedge[1] << ", Doppler width " << ddmc_line_dnu << " Hz at "
                << ddmc_line_temp << " K; the core node of a cell reaches at least "
                << ddmc_line_core_min << " of its own Doppler widths; a kinetic visit "
                << "to the full core at " << ddmc_line_temp << " K is "
                << DDMCLineCoreVisit(ddmc_line_core, 1.) << " scatterings"
                << (read ? " (table read from " : " (table built")
                << (read ? ddmc_line_visit_file + ")" : std::string(")")) << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn int MonteCarlo::DDMCLineBin(Real e) const
//! \brief the bin of the extended grid holding comoving energy e: the core at nwing + 1,
//!        the wing bins on either side, the virtual bins at 0 and nbin + 1; -1 beyond
//!        them

int MonteCarlo::DDMCLineBin(Real e) const {
  const Real x = (e/MCConstants::h_cgs - MCConstants::nu_lya)/ddmc_line_dnu;
  const Real ax = std::abs(x);
  const int core = ddmc_line_nwing + 1;
  if (ax <= ddmc_line_core) return core;
  const int m = static_cast<int>(std::floor((ax - ddmc_line_core)/ddmc_line_dx)) + 1;
  if (m > ddmc_line_nwing + 1) return -1;
  return (x < 0.) ? core - m : core + m;
}

//----------------------------------------------------------------------------------------
//! \fn Real MonteCarlo::DDMCLineDraw(int b, MCRandom *pran) const
//! \brief an energy uniform in bin b

Real MonteCarlo::DDMCLineDraw(int b, MCRandom *pran) const {
  return ddmc_line_eedge[b] + pran->uniform()*(ddmc_line_eedge[b+1] - ddmc_line_eedge[b]);
}

//----------------------------------------------------------------------------------------
//! \fn Real MonteCarloBlock::DDMCLineScale(int i3, int i2, int i1) const
//! \brief the cell's Doppler scale s = Doppler width at line_temp over the cell's own,
//!        which turns a reference offset x into the cell's x s

Real MonteCarloBlock::DDMCLineScale(int i3, int i2, int i1) const {
  return pmy_mc->ddmc_line_dnu/DopplerWidth(tgas(i3,i2,i1));
}

//----------------------------------------------------------------------------------------
//! \fn Real MonteCarloBlock::DDMCLineKx(int i3, int i2, int i1, Real x) const
//! \brief the cell's Voigt extinction at reference Doppler offset x, per cm

Real MonteCarloBlock::DDMCLineKx(int i3, int i2, int i1, Real x) const {
  const Real T = tgas(i3,i2,i1), nh = species(0,i3,i2,i1);
  if (!(T > 0.) || !(nh > 0.)) return 0.;
  const Real dnu = DopplerWidth(T);
  const Real a = MCConstants::lorwidth_lya/dnu;
  const Real pre = nh*MCConstants::res_osc*MCConstants::oscf_lya/(std::sqrt(PI)*dnu);
  const Real xloc = x*pmy_mc->ddmc_line_dnu/dnu;
  const std::complex<double> z = ZetaVoigt(std::complex<double>(xloc, a));
  return pre*z.imag()/std::sqrt(PI);
}

//----------------------------------------------------------------------------------------
//! \fn int MonteCarloBlock::DDMCLineCell(int i3, int i2, int i1, Real wmin)
//! \brief the bin-averaged extinctions of cell (i3,i2,i1) over the extended grid, from
//!        its temperature and species(0), its core edge and its outermost node bin on
//!        each side of the core: the last bin whose extinction times the cell's smallest
//!        width wmin (cm) reaches <ddmc> tau, the core bin by the extinction at its
//!        edge; where the core bin is thin at its edge but thick at the line centre the
//!        core node is the band |x| < x_c at which the extinction across the cell is
//!        <ddmc> tau, if that band is at least core_min of the cell's Doppler widths
//!        wide.  The core entry of ddmc_line_k is the mean over the node.  Returns the
//!        edge, -1 when the cell has no node

int MonteCarloBlock::DDMCLineCell(int i3, int i2, int i1, Real wmin) {
  const MonteCarlo *pmc = pmy_mc;
  const int nb = pmc->ddmc_line_nbin + 2, core = pmc->ddmc_line_nwing + 1;
  const Real T = tgas(i3,i2,i1), nh = species(0,i3,i2,i1);
  int edge = -1;
  ddmc_line_xc(i3,i2,i1) = 0.;
  if (!(T > 0.) || !(nh > 0.)) {
    for (int b=0; b<nb; ++b) ddmc_line_k(b,i3,i2,i1) = 0.;
    ddmc_line_edge(i3,i2,i1) = edge;
    return edge;
  }
  const Real dnu = DopplerWidth(T);
  const Real a = MCConstants::lorwidth_lya/dnu;
  const Real pre = nh*MCConstants::res_osc*MCConstants::oscf_lya/(std::sqrt(PI)*dnu);
  // the grid's edges in this cell's Doppler units
  const Real scale = pmc->ddmc_line_dnu/dnu;
  Real xlo = pmc->ddmc_line_xedge[0]*scale;
  Real hlo = VoigtIntegral(a, xlo);
  for (int b=0; b<nb; ++b) {
    const Real xhi = pmc->ddmc_line_xedge[b+1]*scale;
    const Real hhi = VoigtIntegral(a, xhi);
    ddmc_line_k(b,i3,i2,i1) = pre*(hhi - hlo)/(xhi - xlo);
    xlo = xhi;
    hlo = hhi;
  }
  // the extinction at a reference offset x, and the band of nodes about the core, the
  // extinction falling away from it.  A wing bin's extinction varies little across it
  // and its mean decides; the core bin's falls by exp(-x_c^2) from the centre to its
  // edge, so the whole core bin is a node only where the cell is thick at the edge, and
  // elsewhere the node is the part of the core at which the cell is thick
  auto kx = [&](Real x) {
    const std::complex<double> z = ZetaVoigt(std::complex<double>(x*scale, a));
    return pre*z.imag()/std::sqrt(PI);
  };
  const Real tau = pmc->ddmc_tau;
  Real xc = 0.;
  if (kx(pmc->ddmc_line_core)*wmin >= tau) {
    xc = pmc->ddmc_line_core;
    edge = 0;
    for (int m=1; m<=pmc->ddmc_line_nwing; ++m) {
      if (ddmc_line_k(core+m,i3,i2,i1)*wmin >= tau)
        edge = m;
      else
        break;
    }
  } else if (kx(0.)*wmin >= tau) {
    // the edge inside the core, by bisection on the monotone extinction
    Real lo = 0., hi = pmc->ddmc_line_core;
    for (int it=0; it<40; ++it) {
      const Real mid = 0.5*(lo + hi);
      if (kx(mid)*wmin >= tau)
        lo = mid;
      else
        hi = mid;
      if (hi - lo < 1.e-6*pmc->ddmc_line_core) break;
    }
    if (lo*scale >= pmc->ddmc_line_core_min) {
      xc = lo;
      edge = 0;
    }
  }
  if (edge >= 0) {
    // the mean extinction over the node, for the scatterings a dwell stands for
    const Real xl = xc*scale;
    ddmc_line_k(core,i3,i2,i1) = pre*(VoigtIntegral(a, xl) - VoigtIntegral(a, -xl))/(2.*xl);
  }
  ddmc_line_xc(i3,i2,i1) = xc;
  ddmc_line_edge(i3,i2,i1) = edge;
  return edge;
}

//----------------------------------------------------------------------------------------
//! \fn bool MonteCarloBlock::DDMCLineNodeBin(int i3, int i2, int i1, int b) const
//! \brief whether bin b holds nodes of cell (i3,i2,i1): a wing bin within the band, or
//!        the core bin of a diffusion cell (whose node may be only part of the bin,
//!        which DDMCLineNode tests by the energy)

bool MonteCarloBlock::DDMCLineNodeBin(int i3, int i2, int i1, int b) const {
  if (b < 0) return false;
  const int edge = ddmc_line_edge(i3,i2,i1);
  return edge >= 0 && std::abs(b - (pmy_mc->ddmc_line_nwing + 1)) <= edge;
}

//----------------------------------------------------------------------------------------
//! \fn bool MonteCarloBlock::DDMCLineNode(int i3, int i2, int i1, Real e) const
//! \brief whether comoving energy e lies at a node of cell (i3,i2,i1): in a wing bin of
//!        the band, or within the cell's core edge

bool MonteCarloBlock::DDMCLineNode(int i3, int i2, int i1, Real e) const {
  const int b = pmy_mc->DDMCLineBin(e);
  if (!DDMCLineNodeBin(i3, i2, i1, b)) return false;
  if (b != pmy_mc->ddmc_line_nwing + 1) return true;
  const Real x = (e/MCConstants::h_cgs - MCConstants::nu_lya)/pmy_mc->ddmc_line_dnu;
  return std::abs(x) < ddmc_line_xc(i3,i2,i1);
}

//----------------------------------------------------------------------------------------
//! \fn Real MonteCarloBlock::DDMCLineK(int i3, int i2, int i1, Real e) const
//! \brief the walker's extinction in the bin holding e in cell (i3,i2,i1), per cm of
//!        comoving path: the bin's mean, over the cell's core node for the core bin;
//!        zero beyond the extended grid

Real MonteCarloBlock::DDMCLineK(int i3, int i2, int i1, Real e) const {
  const int b = pmy_mc->DDMCLineBin(e);
  return (b < 0) ? 0. : ddmc_line_k(b,i3,i2,i1);
}

//----------------------------------------------------------------------------------------
//! \fn Real MonteCarloBlock::DDMCLineKPhoton(int i3, int i2, int i1, Real e) const
//! \brief the extinction at the frequency of e itself in cell (i3,i2,i1): the Voigt
//!        value within the core bin, where it varies by e^(x^2) across the bin, the bin
//!        mean in a wing bin

Real MonteCarloBlock::DDMCLineKPhoton(int i3, int i2, int i1, Real e) const {
  const int b = pmy_mc->DDMCLineBin(e);
  if (b < 0) return 0.;
  if (b != pmy_mc->ddmc_line_nwing + 1) return ddmc_line_k(b,i3,i2,i1);
  const Real x = (e/MCConstants::h_cgs - MCConstants::nu_lya)/pmy_mc->ddmc_line_dnu;
  return DDMCLineKx(i3, i2, i1, x);
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::DDMCLineEntry(Photon *pphot, int ip, Real e0)
//! \brief the frequency interface, for a transport photon that has just scattered in a
//!        diffusion cell, its comoving energy e0 before the scattering and ep after it.
//!        An entry from outside the cell's nodes into a wing node is admitted
//!        with the probability P = sqrt(2 pi) / (2 dx), dx the node bin's width in the
//!        cell's Doppler units, which balances the lattice's leakage out of the bin, J
//!        k / (2 dx) for a uniform population J, against the kernel's crossings of the
//!        edge, J k / sqrt(2 pi) for its unit-variance steps; otherwise the photon is
//!        reflected about the edge it crossed.  An entry into the core node, whose
//!        kernel redistributes over the core at the next scattering and whose exit rate
//!        is the kinetic one, is always admitted, as is a photon already at a node.
//!        DDMCBeforeFlight converts the admitted photon at the start of its next flight.

void MonteCarloBlock::DDMCLineEntry(Photon *pphot, int ip, Real e0) {
  const MonteCarlo *pmc = pmy_mc;
  const int i1 = pphot->i1p[ip], i2 = pphot->i2p[ip], i3 = pphot->i3p[ip];
  const int b1 = pmc->DDMCLineBin(pphot->ep[ip]);
  if (b1 < 0 || b1 == pmc->ddmc_line_nwing + 1 || !DDMCLineNodeBin(i3, i2, i1, b1))
    return;
  if (DDMCLineNode(i3, i2, i1, e0)) return;
  const Real dnu = DopplerWidth(tgas(i3,i2,i1));
  const Real dx = (pmc->ddmc_line_eedge[b1+1] - pmc->ddmc_line_eedge[b1])
                  /(MCConstants::h_cgs*dnu);
  const Real prob = std::sqrt(2.*PI)/(2.*dx);
  if (prob >= 1. || pran->uniform() < prob) return;
  const Real eedge = (e0 < pphot->ep[ip]) ? pmc->ddmc_line_eedge[b1]
                                          : pmc->ddmc_line_eedge[b1+1];
  pphot->ep[ip] = 2.*eedge - pphot->ep[ip];
  nddmc[DDMC_RETURN_FREQ]++;
}

//----------------------------------------------------------------------------------------
//! \fn Real PhotonPusher::DDMCLineRates(Photon *pphot, int ip, Real rate[2])
//! \brief the walker's leakage rates into the bin below (rate[0]) and above (rate[1]) its
//!        own, per cm of comoving path, and their sum: the harmonic-mean form of
//!        Smith et al. (2018) with the bin widths in the cell's own Doppler units.  The
//!        neighbouring bin need not be a node: the rate into a thin bin is the walker's
//!        emergence rate in frequency, which out of the core node is the kinetic one,
//!        the node's mean extinction over the scatterings of a kinetic visit to a core
//!        of the node's half-width (the harmonic form would make it depend on the width
//!        of the thin bin)

Real PhotonPusher::DDMCLineRates(Photon *pphot, int ip, Real rate[2]) {
  MonteCarloBlock *pmcb = pmy_mcb;
  const MonteCarlo *pmc = pmy_mc;
  rate[0] = rate[1] = 0.;
  const int i1 = pphot->i1p[ip], i2 = pphot->i2p[ip], i3 = pphot->i3p[ip];
  const int b = pmc->DDMCLineBin(pphot->ep[ip]);
  if (b < 1 || b > pmc->ddmc_line_nbin) return 0.;
  const Real kj = pmcb->ddmc_line_k(b,i3,i2,i1);
  if (!(kj > 0.)) return 0.;
  const Real scale = pmcb->DDMCLineScale(i3, i2, i1);
  const Real dxj = (pmc->ddmc_line_xedge[b+1] - pmc->ddmc_line_xedge[b])*scale;
  const bool core = (b == pmc->ddmc_line_nwing + 1);
  for (int side=0; side<2; ++side) {
    const int nb = side ? b + 1 : b - 1;
    if (core && !pmcb->DDMCLineNodeBin(i3, i2, i1, nb)) {
      rate[side] = 0.5*kj/pmc->DDMCLineCoreVisit(pmcb->ddmc_line_xc(i3,i2,i1)*scale,
                                                 scale);
      continue;
    }
    const Real kn = pmcb->ddmc_line_k(nb,i3,i2,i1);
    if (!(kn > 0.)) continue;
    const Real dxn = (pmc->ddmc_line_xedge[nb+1] - pmc->ddmc_line_xedge[nb])*scale;
    rate[side] = 1./(dxj*(dxj/kj + dxn/kn));
  }
  return rate[0] + rate[1];
}

//----------------------------------------------------------------------------------------
//! \fn void PhotonPusher::DDMCLineHop(Photon *pphot, MCRandom *pran, int ip, int side)
//! \brief A hop through the frequency face below (side 0) or above (1) the walker's bin:
//!        into a node of its cell, with the energy drawn uniformly in that bin and the
//         extinction of that bin

void PhotonPusher::DDMCLineHop(Photon *pphot, MCRandom *pran, int ip, int side) {
  MonteCarloBlock *pmcb = pmy_mcb;
  const int i1 = pphot->i1p[ip], i2 = pphot->i2p[ip], i3 = pphot->i3p[ip];
  const int b = pmy_mc->DDMCLineBin(pphot->ep[ip]);
  const int nb = side ? b + 1 : b - 1;
  if (!pmcb->DDMCLineNodeBin(i3, i2, i1, nb)) {
    DDMCLineEmerge(pphot, pran, ip, nb, true);
    return;
  }
  pphot->ep[ip] = pmy_mc->DDMCLineDraw(nb, pran);
  pphot->scp[ip] = pmcb->ddmc_line_k(nb,i3,i2,i1);
  pmcb->nddmc[DDMC_HOP]++;
  pmcb->nddmc[DDMC_HOP_FREQ]++;
  pmcb->lb_nstep++;
}

//----------------------------------------------------------------------------------------
//! \fn void PhotonPusher::DDMCLineEmerge(Photon *pphot, MCRandom *pran, int ip, int b,
//!                                       bool draw)
//! \brief the walker leaves the lattice through a frequency face into bin b, the one
//!        above (side 1) or below (0) its own: a transport photon where it is, isotropic
//!        in the comoving frame, at the energy of the edge it crossed plus the outward
//!        overshoot of a scattering step (draw), or at the energy it has (a drift across
//!        the bin edge), and nothing deposited.  Out of a wing bin the overshoot is a
//!        uniform fraction of a Rayleigh step of unit variance in the cell's Doppler
//!        units, the landing distribution of a kinetic crossing from a uniform
//!        population, which keeps the thin side in balance with the entry test of
//!        DDMCLineEntry; out of the core node the edge is the cell's own and the
//!        overshoot is drawn from the exits of the kinetic visits of the table.

void PhotonPusher::DDMCLineEmerge(Photon *pphot, MCRandom *pran, int ip, int b,
                                  bool draw) {
  if (draw) {
    MonteCarloBlock *pmcb = pmy_mcb;
    const MonteCarlo *pmc = pmy_mc;
    const int i1 = pphot->i1p[ip], i2 = pphot->i2p[ip], i3 = pphot->i3p[ip];
    const int b0 = pmc->DDMCLineBin(pphot->ep[ip]);
    const bool up = (b > b0);
    const Real dnu = DopplerWidth(pmcb->tgas(i3,i2,i1));
    if (b0 == pmc->ddmc_line_nwing + 1) {
      const Real scale = pmc->ddmc_line_dnu/dnu;
      const Real xc = pmcb->ddmc_line_xc(i3,i2,i1);
      const Real over = pmc->DDMCLineCoreExit(xc*scale, scale, pran)/scale;
      const Real x = up ? xc + over : -(xc + over);
      pphot->ep[ip] = MCConstants::h_cgs*(MCConstants::nu_lya + x*pmc->ddmc_line_dnu);
    } else {
      const Real eedge = pmc->ddmc_line_eedge[up ? b : b + 1];
      const Real step = pran->uniform()*std::sqrt(-2.*std::log(pran->uniform()));
      const Real de = step*MCConstants::h_cgs*dnu;
      pphot->ep[ip] = up ? eedge + de : eedge - de;
    }
  }
  DDMCLeave(pphot, pran, ip);
  pmy_mcb->nddmc[DDMC_EMERGED]++;
  pmy_mcb->nddmc[DDMC_EMERGED_FREQ]++;
}

//----------------------------------------------------------------------------------------
//! \fn Real PhotonPusher::DDMCLineFaceRate(int i3, int i2, int i1, int face, Real x,
//!                                         Real kcx, Real kabsn)
//! \brief the leakage rate of face f of cell (i3,i2,i1) for a photon at reference offset
//!        x whose extinction in the cell is kcx (scattering plus absorption), per cm,
//!        without the face's geometric factor: the interior form with the neighbour's
//!        extinction at x (its Voigt value plus kabsn) where the neighbour's core node
//!        holds x, else the boundary form

Real PhotonPusher::DDMCLineFaceRate(int i3, int i2, int i1, int face, Real x, Real kcx,
                                    Real kabsn) {
  MonteCarloBlock *pmcb = pmy_mcb;
  const int axis = face/2;
  const bool up = (face & 1) != 0;
  const Real wc = pmcb->ddmc_rate(MonteCarloBlock::DDMC_RATE_W + axis, i3, i2, i1);
  const int n1 = i1 + ((axis == 0) ? (up ? 1 : -1) : 0);
  const int n2 = i2 + ((axis == 1) ? (up ? 1 : -1) : 0);
  const int n3 = i3 + ((axis == 2) ? (up ? 1 : -1) : 0);
  const AthenaArray<int> &cell = pmcb->ddmc_cell;
  const bool inside = (n1 >= 0 && n1 < cell.GetDim1() && n2 >= 0 && n2 < cell.GetDim2()
                       && n3 >= 0 && n3 < cell.GetDim3());
  if (inside && cell(n3,n2,n1) != 0 && std::abs(x) < pmcb->ddmc_line_xc(n3,n2,n1)) {
    Real Wn[3];
    pmcb->DDMCCellWidths(n3, n2, n1, Wn);
    const Real kn = pmcb->DDMCLineKx(n3, n2, n1, x) + kabsn;
    return 2./(3.*wc*(kcx*wc + kn*Wn[axis]*pmcb->l_cgs));
  }
  return 2./(3.*wc*(kcx*wc + 2.*kDDMCLambda));
}

namespace {
//! the Simpson rule over theta in [0, pi/2] with x = x_c cos(theta): the average of a
//! function over the node (1 / x_c) int_0^x_c f dx = int f(x_c cos theta) sin theta
//! d theta, on kCoreQuad intervals, which resolves the rise by e^(x^2) toward the edge
//! that a grid in x would not
const int kCoreQuad = 64;
}

//----------------------------------------------------------------------------------------
//! \fn Real PhotonPusher::DDMCLineCoreRates(Photon *pphot, int ip, Real rate[6])
//! \brief the six face rates of a core walker and their sum: the per-frequency rate of
//!        each face averaged over the cell's core node, over which the photon's
//!        frequency is uniform (a flat J within a thick core and a scattering that
//!        redistributes it), the neighbour's node or the boundary form by frequency

Real PhotonPusher::DDMCLineCoreRates(Photon *pphot, int ip, Real rate[6]) {
  MonteCarloBlock *pmcb = pmy_mcb;
  const int i1 = pphot->i1p[ip], i2 = pphot->i2p[ip], i3 = pphot->i3p[ip];
  const Real xc = pmcb->ddmc_line_xc(i3,i2,i1);
  const Real kabs = pphot->acp[ip];
  Real kc[kCoreQuad+1], sn[kCoreQuad+1], sw[kCoreQuad+1];
  const Real dth = 0.5*PI/kCoreQuad;
  for (int q=0; q<=kCoreQuad; ++q) {
    const Real th = q*dth;
    kc[q] = pmcb->DDMCLineKx(i3, i2, i1, xc*std::cos(th)) + kabs;
    sn[q] = std::sin(th);
    sw[q] = (q == 0 || q == kCoreQuad) ? 1. : (q % 2) ? 4. : 2.;
    sw[q] *= dth/3.;
  }
  Real G[6];
  pmcb->DDMCFaceFactors(i3, i2, i1, G);
  Real total = 0.;
  for (int f=0; f<6; ++f) {
    rate[f] = 0.;
    if (pmcb->ddmc_rate(f, i3, i2, i1) <= 0.) continue;  // unresolved, or on the axis
    const int axis = f/2;
    const bool up = (f & 1) != 0;
    const int n1 = i1 + ((axis == 0) ? (up ? 1 : -1) : 0);
    const int n2 = i2 + ((axis == 1) ? (up ? 1 : -1) : 0);
    const int n3 = i3 + ((axis == 2) ? (up ? 1 : -1) : 0);
    // the neighbour's absorption at the walker's energy, once per face
    Real kabsn = 0.;
    if (kabs > 0.) kabsn = DDMCNeighbourExtinction(pphot, ip, n3, n2, n1) - pmcb->DDMCLineK(n3, n2, n1, pphot->ep[ip]);
    Real sum = 0.;
    for (int q=0; q<=kCoreQuad; ++q)
      sum += sw[q]*sn[q]*DDMCLineFaceRate(i3, i2, i1, f, xc*std::cos(q*dth), kc[q], kabsn);
    rate[f] = sum*G[f];
    total += rate[f];
  }
  return total;
}

//----------------------------------------------------------------------------------------
//! \fn Real PhotonPusher::DDMCLineCoreDraw(Photon *pphot, MCRandom *pran, int ip,
//!                                         int face)
//! \brief the comoving energy at which a core walker crosses face f: a frequency of the
//!        node in proportion to that face's per-frequency rate (the quadrature interval
//!        by its weight, uniform in theta within it, either sign); face < 0 draws
//!        uniformly over the node, the flat population an advective crossing carries

Real PhotonPusher::DDMCLineCoreDraw(Photon *pphot, MCRandom *pran, int ip, int face) {
  MonteCarloBlock *pmcb = pmy_mcb;
  const int i1 = pphot->i1p[ip], i2 = pphot->i2p[ip], i3 = pphot->i3p[ip];
  const Real xc = pmcb->ddmc_line_xc(i3,i2,i1);
  Real x;
  if (face < 0) {
    x = xc*pran->uniform();
  } else {
    const Real kabs = pphot->acp[ip];
    const int axis = face/2;
    const bool up = (face & 1) != 0;
    const int n1 = i1 + ((axis == 0) ? (up ? 1 : -1) : 0);
    const int n2 = i2 + ((axis == 1) ? (up ? 1 : -1) : 0);
    const int n3 = i3 + ((axis == 2) ? (up ? 1 : -1) : 0);
    Real kabsn = 0.;
    if (kabs > 0.) kabsn = DDMCNeighbourExtinction(pphot, ip, n3, n2, n1) - pmcb->DDMCLineK(n3, n2, n1, pphot->ep[ip]);
    const Real dth = 0.5*PI/kCoreQuad;
    Real w[kCoreQuad+1], cum = 0.;
    for (int q=0; q<=kCoreQuad; ++q) {
      const Real th = q*dth;
      const Real kcx = pmcb->DDMCLineKx(i3, i2, i1, xc*std::cos(th)) + kabs;
      Real sw = (q == 0 || q == kCoreQuad) ? 1. : (q % 2) ? 4. : 2.;
      w[q] = sw*std::sin(th)*DDMCLineFaceRate(i3, i2, i1, face, xc*std::cos(th), kcx, kabsn);
      cum += w[q];
    }
    Real v = pran->uniform()*cum;
    int q = 0;
    while (q < kCoreQuad && v >= w[q]) {
      v -= w[q];
      ++q;
    }
    const Real th = std::min(std::max((q + pran->uniform() - 0.5)*dth, 0.), 0.5*PI);
    x = xc*std::cos(th);
  }
  if (pran->uniform() < 0.5) x = -x;
  return MCConstants::h_cgs*(MCConstants::nu_lya + x*pmy_mc->ddmc_line_dnu);
}
