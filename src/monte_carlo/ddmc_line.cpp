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
//! \fn void MonteCarlo::SetupDDMCLine(ParameterInput *pin)
//! \brief read the <ddmc> line keys and lay out the frequency grid

void MonteCarlo::SetupDDMCLine(ParameterInput *pin) {
  const std::string name = pin->GetOrAddString("ddmc","line","none");
  ddmc_line = DDMC_LINE_NONE;
  ddmc_line_nwing = 0;
  ddmc_line_nbin = 0;
  ddmc_line_temp = ddmc_line_xmax = ddmc_line_dx = ddmc_line_core = ddmc_line_dnu = 0.;
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
  if (!(ddmc_line_temp > 0.) || !(ddmc_line_dx > 0.) || !(ddmc_line_core > 0.)
      || !(ddmc_line_xmax > ddmc_line_core)) {
    msg << "### FATAL ERROR in MonteCarlo constructor" << std::endl
        << "<ddmc> line_temp, dx and core_width must be positive and xmax above "
        << "core_width" << std::endl;
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
  if (Globals::my_rank == 0)
    std::cout << "frequency lattice: Lyman alpha, " << ddmc_line_nbin
              << " bins, core |x| <= " << ddmc_line_core << ", " << ddmc_line_nwing
              << " wing bins of " << ddmc_line_dx << " on each side to |x| = "
              << -ddmc_line_xedge[1] << ", Doppler width " << ddmc_line_dnu << " Hz at "
              << ddmc_line_temp << " K" << std::endl;
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
//! \fn int MonteCarloBlock::DDMCLineCell(int i3, int i2, int i1, Real wmin)
//! \brief the bin-averaged extinctions of cell (i3,i2,i1) over the extended grid, from
//!        its temperature and species(0), and its outermost node bin on each side of
//!        the core, the last bin whose extinction times the cell's smallest width wmin
//!        (cm) reaches <ddmc> tau; -1 when not even the core does

int MonteCarloBlock::DDMCLineCell(int i3, int i2, int i1, Real wmin) {
  const MonteCarlo *pmc = pmy_mc;
  const int nb = pmc->ddmc_line_nbin + 2, core = pmc->ddmc_line_nwing + 1;
  const Real T = tgas(i3,i2,i1), nh = species(0,i3,i2,i1);
  int edge = -1;
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
  // the band of nodes about the core, the extinction falling away from it
  for (int m=0; m<=pmc->ddmc_line_nwing; ++m) {
    if (ddmc_line_k(core+m,i3,i2,i1)*wmin >= pmc->ddmc_tau)
      edge = m;
    else
      break;
  }
  ddmc_line_edge(i3,i2,i1) = edge;
  return edge;
}

//----------------------------------------------------------------------------------------
//! \fn bool MonteCarloBlock::DDMCLineNodeBin(int i3, int i2, int i1, int b) const
//! \brief whether bin b is a node of cell (i3,i2,i1)

bool MonteCarloBlock::DDMCLineNodeBin(int i3, int i2, int i1, int b) const {
  if (b < 0) return false;
  const int edge = ddmc_line_edge(i3,i2,i1);
  return edge >= 0 && std::abs(b - (pmy_mc->ddmc_line_nwing + 1)) <= edge;
}

//----------------------------------------------------------------------------------------
//! \fn bool MonteCarloBlock::DDMCLineNode(int i3, int i2, int i1, Real e) const
//! \brief whether the bin holding comoving energy e is a node of cell (i3,i2,i1)

bool MonteCarloBlock::DDMCLineNode(int i3, int i2, int i1, Real e) const {
  return DDMCLineNodeBin(i3, i2, i1, pmy_mc->DDMCLineBin(e));
}

//----------------------------------------------------------------------------------------
//! \fn Real MonteCarloBlock::DDMCLineK(int i3, int i2, int i1, Real e) const
//! \brief the bin-averaged extinction of the bin holding e in cell (i3,i2,i1), per cm of
//!        comoving path; zero beyond the extended grid

Real MonteCarloBlock::DDMCLineK(int i3, int i2, int i1, Real e) const {
  const int b = pmy_mc->DDMCLineBin(e);
  return (b < 0) ? 0. : ddmc_line_k(b,i3,i2,i1);
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::DDMCLineEntry(Photon *pphot, int ip, Real e0)
//! \brief the frequency interface, for a transport photon that has just scattered in a
//!        diffusion cell, its comoving energy e0 before the scattering and ep after it.
//!        An entry from a bin that is not a node of the cell into a wing node is admitted
//!        with the probability P = sqrt(2 pi) / (2 dx), dx the node bin's width in the
//!        cell's Doppler units, which balances the lattice's leakage out of the bin, J
//!        k / (2 dx) for a uniform population J, against the kernel's crossings of the
//!        edge, J k / sqrt(2 pi) for its unit-variance steps; otherwise the photon is
//!        reflected about the edge it crossed.  An entry into the core bin, whose kernel
//!        redistributes over the core at the next scattering, is always admitted, as is
//!        a photon already at a node.  DDMCBeforeFlight converts the admitted photon
//!        at the start of its next flight.

void MonteCarloBlock::DDMCLineEntry(Photon *pphot, int ip, Real e0) {
  const MonteCarlo *pmc = pmy_mc;
  const int i1 = pphot->i1p[ip], i2 = pphot->i2p[ip], i3 = pphot->i3p[ip];
  const int b1 = pmc->DDMCLineBin(pphot->ep[ip]);
  if (b1 < 0 || b1 == pmc->ddmc_line_nwing + 1 || !DDMCLineNodeBin(i3, i2, i1, b1))
    return;
  const int b0 = pmc->DDMCLineBin(e0);
  if (b0 >= 0 && DDMCLineNodeBin(i3, i2, i1, b0)) return;
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
//!        emergence rate in frequency

Real PhotonPusher::DDMCLineRates(Photon *pphot, int ip, Real rate[2]) {
  MonteCarloBlock *pmcb = pmy_mcb;
  const MonteCarlo *pmc = pmy_mc;
  rate[0] = rate[1] = 0.;
  const int i1 = pphot->i1p[ip], i2 = pphot->i2p[ip], i3 = pphot->i3p[ip];
  const int b = pmc->DDMCLineBin(pphot->ep[ip]);
  if (b < 1 || b > pmc->ddmc_line_nbin) return 0.;
  const Real kj = pmcb->ddmc_line_k(b,i3,i2,i1);
  if (!(kj > 0.)) return 0.;
  const Real scale = pmc->ddmc_line_dnu/DopplerWidth(pmcb->tgas(i3,i2,i1));
  const Real dxj = (pmc->ddmc_line_xedge[b+1] - pmc->ddmc_line_xedge[b])*scale;
  for (int side=0; side<2; ++side) {
    const int nb = side ? b + 1 : b - 1;
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
//!        the bin edge), and nothing deposited.  The overshoot is a uniform fraction of a
//!        Rayleigh step of unit variance in the cell's Doppler units, the landing
//!        distribution of a kinetic crossing from a uniform population; it keeps the
//!        thin side in balance with the entry test of DDMCLineEntry.

void PhotonPusher::DDMCLineEmerge(Photon *pphot, MCRandom *pran, int ip, int b,
                                  bool draw) {
  if (draw) {
    MonteCarloBlock *pmcb = pmy_mcb;
    const int i1 = pphot->i1p[ip], i2 = pphot->i2p[ip], i3 = pphot->i3p[ip];
    const Real dnu = DopplerWidth(pmcb->tgas(i3,i2,i1));
    const Real eedge = pmy_mc->ddmc_line_eedge[(b > pmy_mc->DDMCLineBin(pphot->ep[ip]))
                                               ? b : b + 1];
    const Real step = pran->uniform()*std::sqrt(-2.*std::log(pran->uniform()));
    const Real de = step*MCConstants::h_cgs*dnu;
    pphot->ep[ip] = (eedge > pphot->ep[ip]) ? eedge + de : eedge - de;
  }
  DDMCLeave(pphot, pran, ip);
  pmy_mcb->nddmc[DDMC_EMERGED]++;
  pmy_mcb->nddmc[DDMC_EMERGED_FREQ]++;
}
