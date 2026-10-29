//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file ddmc.cpp
//! \brief discrete diffusion Monte Carlo (<ddmc>): the lattice of leakage rates, the
//!        walk, and the interface with transport photons
//
// A cell whose Thomson optical depth across its smallest width is above <ddmc> tau is a
// diffusion cell.  A photon in one is a walker (Photon::ddmcp set): it sits
// at the cell's centre, carries no direction, and instead of flying it dwells for an
// exponential time at the cell's total leakage rate and hops through a face chosen in
// proportion to the face rates.  A hop into another diffusion cell is a move to that
// cell's centre; a hop through any other face (a transport cell, a mesh boundary, a
// block at another refinement level) is its emergence as a transport photon on that face.
// A transport photon that reaches the face of a diffusion cell is converted with the
// probability of the asymptotic diffusion-limit boundary condition, and otherwise sent
// back the way it came.  Absorption acts over the dwell; in this build the scattering is
// grey and elastic (Thomson), so the energy is unchanged by a dwell.
//
// Everything the lattice needs is rebuilt from the fluid in BuildDDMCRates, so restarts
// and redistributions carry nothing but the walkers.  The table holds the rates of pure
// Thomson scattering; a walker that is also absorbed diffuses with the total extinction
// at its energy, so its rates are formed per dwell (DDMCRates).  The formulas and the
// derivation are in doc/monte_carlo/ddmc_guide.tex; the phases in compton_ddmc_plan.md.
//
// The moments of a diffusion cell: the energy density by the time-spent estimator over
// each dwell (UpdateMomentsMRW, isotropic in the comoving frame with the diffusion
// closure P = E/3), the flux by counting the energy carried through each face of the
// cell (a hop deposits s w E Delta_i / 2 into the flux along i of both cells sharing the
// face, which after the normalization by volume and time is the face-average flux), and
// the radiation force from the same crossings, chi F / c.

// C++ headers
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../mesh/mesh.hpp"
#include "mccoord.hpp"
#include "montecarlo.hpp"
#include "photon.hpp"
#include "photonpusher.hpp"

namespace {

//! the extrapolation distance of the asymptotic diffusion-limit boundary condition in
//! mean free paths (Habetler & Matkowsky 1975)
const Real kLambda = 0.7104;
//! how far inside a cell a photon is placed when it is put on a face, as a fraction of
//! the cell's width in that coordinate
const Real kFaceNudge = 1.e-7;

//! a unit direction with component sign*mu along axis and the rest at azimuth phi
void FaceDirection(int axis, Real sign, Real mu, Real phi, Real k[3]) {
  const Real smu = std::sqrt(std::max(1. - mu*mu, 0.));
  const int a1 = (axis + 1) % 3, a2 = (axis + 2) % 3;
  k[axis] = sign*mu;
  k[a1] = smu*std::cos(phi);
  k[a2] = smu*std::sin(phi);
}

//! mu drawn from W(mu) = mu + 3/2 mu^2 on (0, 1), the asymptotic emergent distribution:
//! the root of mu^2/2 + mu^3/2 = xi, by Newton from the small-mu limit
Real SampleAsymptoticMu(Real xi) {
  Real mu = std::min(std::sqrt(2.*xi), 1.);
  for (int it=0; it<8; ++it) {
    const Real f = 0.5*mu*mu + 0.5*mu*mu*mu - xi;
    const Real fp = mu + 1.5*mu*mu;
    if (fp <= 0.) break;
    mu -= f/fp;
    mu = std::min(std::max(mu, 1.e-12), 1.);
  }
  return mu;
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::DDMCCellWidths(int i3, int i2, int i1, Real W[3]) const
//! \brief the widths of a cell along its three coordinate directions at its centre, in
//!        code length; a spherical-polar cell is the box dr x r dtheta x r sin(theta)
//!        dphi

void MonteCarloBlock::DDMCCellWidths(int i3, int i2, int i1, Real W[3]) const {
  W[0] = pcoord->x1f(i1+1) - pcoord->x1f(i1);
  W[1] = pcoord->x2f(i2+1) - pcoord->x2f(i2);
  W[2] = pcoord->x3f(i3+1) - pcoord->x3f(i3);
  if (topology == MCTOPO_SPHERICAL) {
    const Real r = 0.5*(pcoord->x1f(i1) + pcoord->x1f(i1+1));
    const Real th = 0.5*(pcoord->x2f(i2) + pcoord->x2f(i2+1));
    W[1] *= r;
    W[2] *= r*std::sin(th);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::BuildDDMCRates()
//! \brief which cells are diffusion cells and the leakage rates of each
//
// ddmc_cell, the flag that marks the diffusion cells (1) among the transport cells (0),
// covers the ghost cells too: a ghost cell across a block face whose neighbour is at the
// same refinement level is tested like an active cell, on the neighbour's values that
// the mesh put in the ghost zone, so a walker can hop into it (and be buffered to that
// block) and a transport photon can convert at that face.  A ghost cell beyond any
// other kind of face (a mesh boundary of any sort, periodic included, or a block at
// another level) is never a diffusion cell, which makes that face a boundary face of
// the lattice.  A cell that a problem generator's accel_mask excludes is a transport
// cell, as it is for the random walk.
//
// The rates, per cm of comoving path, for the active diffusion cells only:
//   interior face   2 / (3 Delta_c (chi_c Delta_c + chi_n Delta_n))
//   boundary face   2 / (3 Delta_c (chi_c Delta_c + 2 lambda))
// with chi the Thomson extinction and Delta the width along the face's axis, in cm; a
// direction the mesh does not resolve (one cell) has no rate.  Slots 0..5 are the
// faces (-x1, +x1, -x2, +x2, -x3, +x3), 6 the sum, 7..9 the widths in cm.

void MonteCarloBlock::BuildDDMCRates() {
  if (ddmc_cell.GetSize() == 0) return;
  const Real sigt = MCConstants::sigmat;
  const int nc3 = ddmc_cell.GetDim3(), nc2 = ddmc_cell.GetDim2();
  const int nc1 = ddmc_cell.GetDim1();
  const int level = pmy_block->loc.level;
  const BoundaryValues *pbv = pmy_block->pbval;
  const bool use_mask = (accel_mask.GetSize() > 0);
  const bool active2 = (nx2 > 1), active3 = (nx3 > 1);
  int64_t ncell = 0;

  for (int k=0; k<nc3; ++k) {
    const int ox3 = (k < ks) ? -1 : (k > ke) ? 1 : 0;
    for (int j=0; j<nc2; ++j) {
      const int ox2 = (j < js) ? -1 : (j > je) ? 1 : 0;
      for (int i=0; i<nc1; ++i) {
        const int ox1 = (i < is) ? -1 : (i > ie) ? 1 : 0;
        bool real = true;
        if (ox1 != 0 || ox2 != 0 || ox3 != 0) {
          // a ghost cell: real only across a same-level block face
          const BoundaryFace f1 = (ox1 < 0) ? BoundaryFace::inner_x1
                                            : BoundaryFace::outer_x1;
          const BoundaryFace f2 = (ox2 < 0) ? BoundaryFace::inner_x2
                                            : BoundaryFace::outer_x2;
          const BoundaryFace f3 = (ox3 < 0) ? BoundaryFace::inner_x3
                                            : BoundaryFace::outer_x3;
          if (ox1 != 0 && pbv->block_bcs[f1] != BoundaryFlag::block) real = false;
          if (ox2 != 0 && pbv->block_bcs[f2] != BoundaryFlag::block) real = false;
          if (ox3 != 0 && pbv->block_bcs[f3] != BoundaryFlag::block) real = false;
          if (real && pbv->nblevel[ox3+1][ox2+1][ox1+1] != level) real = false;
        }
        bool diffusion = real;
        if (diffusion && use_mask && accel_mask(k,j,i) == 0) diffusion = false;
        if (diffusion) {
          const Real chi = species(0,k,j,i)*sigt;
          Real W[3];
          DDMCCellWidths(k, j, i, W);
          Real wmin = W[0];
          if (active2) wmin = std::min(wmin, W[1]);
          if (active3) wmin = std::min(wmin, W[2]);
          diffusion = (chi > 0.) && (chi*wmin*l_cgs >= pmy_mc->ddmc_tau)
                      && (tgas(k,j,i) > 0.);
        }
        ddmc_cell(k,j,i) = diffusion ? 1 : 0;
      }
    }
  }

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        for (int n=0; n<DDMC_NRATE; ++n) ddmc_rate(n,k,j,i) = 0.;
        if (ddmc_cell(k,j,i) == 0) continue;
        ++ncell;
        const Real chi_c = species(0,k,j,i)*sigt;
        Real W[3];
        DDMCCellWidths(k, j, i, W);
        Real total = 0.;
        for (int axis=0; axis<3; ++axis) {
          const Real wc = W[axis]*l_cgs;
          ddmc_rate(DDMC_RATE_W + axis, k, j, i) = wc;
          if ((axis == 1 && !active2) || (axis == 2 && !active3)) continue;
          for (int side=0; side<2; ++side) {
            const int di = (axis == 0) ? (side ? 1 : -1) : 0;
            const int dj = (axis == 1) ? (side ? 1 : -1) : 0;
            const int dk = (axis == 2) ? (side ? 1 : -1) : 0;
            Real rate;
            if (ddmc_cell(k+dk, j+dj, i+di) != 0) {
              const Real chi_n = species(0,k+dk,j+dj,i+di)*sigt;
              Real Wn[3];
              DDMCCellWidths(k+dk, j+dj, i+di, Wn);
              rate = 2./(3.*wc*(chi_c*wc + chi_n*Wn[axis]*l_cgs));
            } else {
              rate = 2./(3.*wc*(chi_c*wc + 2.*kLambda));
            }
            ddmc_rate(2*axis + side, k, j, i) = rate;
            total += rate;
          }
        }
        ddmc_rate(DDMC_RATE_TOT, k, j, i) = total;
      }
    }
  }
  nddmc_cells = ncell;
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::DDMCHopDeposit(int type, int i3, int i2, int i1, int axis,
//!                                          Real sign, Real we)
//! \brief the moments and source terms a crossing of a face of cell (i3,i2,i1) leaves in
//!        that cell: energy we = w E carried in the direction sign along axis
//
// The flux along axis, after the normalization by the cell's volume and the integration
// time, is the average of the energy crossing the cell's two faces per unit area and
// time; the deposit is therefore sign * we * Delta_axis / 2 with Delta the width.  The
// radiation force on the gas of a diffusion cell is chi F / c, so the same crossing
// deposits sign * we * chi * Delta / (2c) into the momentum source terms, both the
// total and the flux force (they coincide on the lattice), chi the total extinction the
// walker sees in that cell.  The flux is comoving; with
// the cell moving it is carried to the lab moments to first order in the velocity, the
// isotropic part having been deposited by the dwell already.  The held-back force
// contribution of sourceterms_error follows the photon's: a crossing into the other
// cell flushes the pending one, so the error of a cell counts a walker's visits to
// it separately, which overstates the independence somewhat.

void MonteCarloBlock::DDMCHopDeposit(int type, int i3, int i2, int i1, int axis,
                                     Real sign, Real we, Real chi) {
  const Real width = ddmc_rate(DDMC_RATE_W + axis, i3, i2, i1);
  const Real fx = 0.5*sign*we*width;
  if (call_moments && mom_flag_lab) {
    moments(type, MCIFR1 + axis, i3, i2, i1) += fx;
    if (boosts && vel.GetSize() > 0) {
      const Real gam = vel(i3,i2,i1,0);
      const Real beta[3] = {vel(i3,i2,i1,1)/gam, vel(i3,i2,i1,2)/gam,
                            vel(i3,i2,i1,3)/gam};
      const Real c = MCConstants::c_cgs;
      moments(type, MCIER, i3, i2, i1) += 2.*beta[axis]*fx/c;
      for (int b=0; b<3; ++b) {
        const int a = std::min(axis, b), d = std::max(axis, b);
        const Real term = (a == d) ? 2.*beta[b]*fx/c : beta[b]*fx/c;
        moments(type, MomentSlot[a+1][d+1], i3, i2, i1) += term;
      }
    }
  }
  if (call_moments && mom_flag_com && accumulate_com)
    moments_com(type, MCIFR1 + axis, i3, i2, i1) += fx;
  if (call_srcterms) {
    const Real f = fx*chi/MCConstants::c_cgs;
    sourceterms(MCRS1 + axis, i3, i2, i1) += f;
    sourceterms(MCRF1 + axis, i3, i2, i1) += f;
    if (i1 != force_pend_i1_ || i2 != force_pend_i2_ || i3 != force_pend_i3_) {
      FlushForceError();
      force_pend_i1_ = i1;
      force_pend_i2_ = i2;
      force_pend_i3_ = i3;
    }
    force_pend_sum_[axis] += f;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void PhotonPusher::DDMCPlaceAtCentre(Photon *pphot, int ip, int i3, int i2,
//!                                         int i1)
//! \brief the walker's position: the centre of cell (i3,i2,i1) in coordinates; the
//!        indices are left to the caller

void PhotonPusher::DDMCPlaceAtCentre(Photon *pphot, int ip, int i3, int i2, int i1) {
  pphot->x1p[ip] = 0.5*(pcoord->x1f(i1) + pcoord->x1f(i1+1));
  pphot->x2p[ip] = 0.5*(pcoord->x2f(i2) + pcoord->x2f(i2+1));
  pphot->x3p[ip] = 0.5*(pcoord->x3f(i3) + pcoord->x3f(i3+1));
}

//----------------------------------------------------------------------------------------
//! \fn void PhotonPusher::DDMCSetWalker(Photon *pphot, int ip)
//! \brief the columns of a walker resident in an active cell: the energy comoving (the
//!        caller has put the lab energy and direction in place), no direction, no
//!        polarization, no flight, the opacities those of the cell at that energy

void PhotonPusher::DDMCSetWalker(Photon *pphot, int ip) {
  MonteCarloBlock *pmcb = pmy_mcb;
  if (boosts) pmcb->TransformToComoving(pphot, ip, ip);
  pphot->k1p[ip] = pphot->k2p[ip] = pphot->k3p[ip] = 0.;
  if (IsPolarized(pmy_mc->polarized)) {
    pphot->sqp[ip] = 0.;
    pphot->sup[ip] = 0.;
    pphot->svp[ip] = 0.;
  }
  pphot->ddmcp[ip] = 1;
  pphot->taup[ip] = -1.;
  pphot->strp[ip] = 0.;
  pphot->nmvp[ip] = 0;
  pphot->acp[ip] = pmcb->AbsorptionOpacity(pmcb, pphot, ip);
  pphot->scp[ip] = pmcb->ScatteringOpacity(pmcb, pphot, ip);
}

//----------------------------------------------------------------------------------------
//! \fn void PhotonPusher::DDMCConvert(Photon *pphot, int ip)
//! \brief a transport photon starting a flight in a diffusion cell (emitted or just
//!        scattered there) becomes a walker at the cell's centre

void PhotonPusher::DDMCConvert(Photon *pphot, int ip) {
  DDMCPlaceAtCentre(pphot, ip, pphot->i3p[ip], pphot->i2p[ip], pphot->i1p[ip]);
  DDMCSetWalker(pphot, ip);
  pmy_mcb->nddmc[DDMC_CONVERT_BIRTH]++;
}

//----------------------------------------------------------------------------------------
//! \fn void PhotonPusher::DDMCRelease(Photon *pphot, MCRandom *pran, int ip)
//! \brief a walker in a cell that is not a diffusion cell (the lattice was rebuilt
//!        under it, or a block disagreed with its neighbour's ghost mask) goes back to
//!        transport at the cell's centre with an isotropic direction

void PhotonPusher::DDMCRelease(Photon *pphot, MCRandom *pran, int ip) {
  MonteCarloBlock *pmcb = pmy_mcb;
  const Real mu = 2.*pran->uniform() - 1.;
  const Real smu = std::sqrt(std::max(1. - mu*mu, 0.));
  const Real phi = 2.*PI*pran->uniform();
  pphot->k1p[ip] = smu*std::cos(phi);
  pphot->k2p[ip] = smu*std::sin(phi);
  pphot->k3p[ip] = mu;
  pphot->ddmcp[ip] = 0;
  pphot->taup[ip] = -1.;
  pphot->strp[ip] = 0.;
  pphot->nmvp[ip] = 0;
  pphot->acp[ip] = pmcb->AbsorptionOpacity(pmcb, pphot, ip);
  pphot->scp[ip] = pmcb->ScatteringOpacity(pmcb, pphot, ip);
  if (boosts) pmcb->TransformToCoordinate(pphot, ip, ip);
  pmcb->nddmc[DDMC_RELEASE]++;
}

//----------------------------------------------------------------------------------------
//! \fn void PhotonPusher::DDMCEmerge(Photon *pphot, MCRandom *pran, int ip, int face)
//! \brief the walker leaves the lattice through face (0..5) of its cell: a transport
//!        photon at a uniform point of that face, a hair inside the cell, moving outward
//!        with mu from W(mu) (emergence = asymptotic) or uniform (isotropic),
//!        unpolarized, with a fresh flight ahead of it.  The crossing is the pusher's
//!        first step, so a mesh boundary is met by its boundary function as usual.

void PhotonPusher::DDMCEmerge(Photon *pphot, MCRandom *pran, int ip, int face) {
  MonteCarloBlock *pmcb = pmy_mcb;
  const int axis = face/2;
  const bool up = (face & 1) != 0;
  const int i1 = pphot->i1p[ip], i2 = pphot->i2p[ip], i3 = pphot->i3p[ip];
  // the crossing deposits the flux and force of the leaving side
  pmcb->DDMCHopDeposit(pphot->type[ip], i3, i2, i1, axis, up ? 1. : -1.,
                       pphot->wp[ip]*pphot->ep[ip], pphot->scp[ip] + pphot->acp[ip]);
  // position
  Real lo[3] = {pcoord->x1f(i1), pcoord->x2f(i2), pcoord->x3f(i3)};
  Real hi[3] = {pcoord->x1f(i1+1), pcoord->x2f(i2+1), pcoord->x3f(i3+1)};
  Real x[3];
  for (int a=0; a<3; ++a) {
    if (a == axis) {
      x[a] = up ? hi[a] - kFaceNudge*(hi[a] - lo[a])
                : lo[a] + kFaceNudge*(hi[a] - lo[a]);
    } else {
      x[a] = lo[a] + pran->uniform()*(hi[a] - lo[a]);
    }
  }
  pphot->x1p[ip] = x[0];
  pphot->x2p[ip] = x[1];
  pphot->x3p[ip] = x[2];
  // direction, in the local orthonormal basis of the legacy pushers
  const Real xi = pran->uniform();
  const Real mu = (pmy_mc->ddmc_emergence == DDMC_EMERGE_ASYMPTOTIC)
                  ? SampleAsymptoticMu(xi) : std::max(xi, 1.e-12);
  Real k[3];
  FaceDirection(axis, up ? 1. : -1., mu, 2.*PI*pran->uniform(), k);
  pphot->k1p[ip] = k[0];
  pphot->k2p[ip] = k[1];
  pphot->k3p[ip] = k[2];
  pphot->ddmcp[ip] = 0;
  pphot->taup[ip] = -1.;
  pphot->strp[ip] = 0.;
  pphot->nmvp[ip] = 0;
  // the opacities at the comoving energy, which the transform to the lab then shifts
  pphot->acp[ip] = pmcb->AbsorptionOpacity(pmcb, pphot, ip);
  pphot->scp[ip] = pmcb->ScatteringOpacity(pmcb, pphot, ip);
  if (boosts) pmcb->TransformToCoordinate(pphot, ip, ip);
  pmcb->nddmc[DDMC_EMERGED]++;
}

//----------------------------------------------------------------------------------------
//! \fn void PhotonPusher::DDMCArrive(Photon *pphot, int ip)
//! \brief a walker that has just come through a block face (ddmcp = 2 + face of its
//!        new cell): its energy into this cell's frame, the entering side's deposit

void PhotonPusher::DDMCArrive(Photon *pphot, int ip) {
  MonteCarloBlock *pmcb = pmy_mcb;
  const int face = pphot->ddmcp[ip] - 2;
  const int axis = face/2;
  // in through the lower face means moving along +axis
  const Real sign = (face & 1) ? -1. : 1.;
  DDMCSetWalker(pphot, ip);
  if (pmcb->ddmc_cell(pphot->i3p[ip], pphot->i2p[ip], pphot->i1p[ip]) != 0)
    pmcb->DDMCHopDeposit(pphot->type[ip], pphot->i3p[ip], pphot->i2p[ip],
                         pphot->i1p[ip], axis, sign, pphot->wp[ip]*pphot->ep[ip],
                         pphot->scp[ip] + pphot->acp[ip]);
}

//----------------------------------------------------------------------------------------
//! \fn void PhotonPusher::DDMCHop(Photon *pphot, MCRandom *pran, int ip, int face)
//! \brief an interior hop through face (0..5): the walker to the neighbour's centre
//!        with its energy in the neighbour's frame; a neighbour in the ghost zone means
//!        the walker leaves the block, carrying the face it enters through

void PhotonPusher::DDMCHop(Photon *pphot, MCRandom *pran, int ip, int face) {
  MonteCarloBlock *pmcb = pmy_mcb;
  const int axis = face/2;
  const bool up = (face & 1) != 0;
  const Real sign = up ? 1. : -1.;
  const int type = pphot->type[ip];
  const int i1 = pphot->i1p[ip], i2 = pphot->i2p[ip], i3 = pphot->i3p[ip];
  const int n1 = i1 + ((axis == 0) ? (up ? 1 : -1) : 0);
  const int n2 = i2 + ((axis == 1) ? (up ? 1 : -1) : 0);
  const int n3 = i3 + ((axis == 2) ? (up ? 1 : -1) : 0);
  pmcb->DDMCHopDeposit(type, i3, i2, i1, axis, sign, pphot->wp[ip]*pphot->ep[ip],
                       pphot->scp[ip] + pphot->acp[ip]);
  // The energy changes frame through a direction drawn on the face: comoving in this
  // cell to the lab here, and in the new cell back to comoving; the direction is then
  // dropped.  Nothing to do when the medium is at rest.
  if (boosts) {
    Real k[3];
    FaceDirection(axis, sign, SampleAsymptoticMu(pran->uniform()),
                  2.*PI*pran->uniform(), k);
    pphot->k1p[ip] = k[0];
    pphot->k2p[ip] = k[1];
    pphot->k3p[ip] = k[2];
    pmcb->TransformToCoordinate(pphot, ip, ip);
  }
  const bool ghost = (n1 < pmcb->is || n1 > pmcb->ie || n2 < pmcb->js || n2 > pmcb->je
                      || n3 < pmcb->ks || n3 > pmcb->ke);
  DDMCPlaceAtCentre(pphot, ip, n3, n2, n1);
  pmcb->nddmc[DDMC_HOP]++;
  pmcb->lb_nstep++;
  if (!ghost) {
    pphot->i1p[ip] = n1;
    pphot->i2p[ip] = n2;
    pphot->i3p[ip] = n3;
    DDMCSetWalker(pphot, ip);
    pmcb->DDMCHopDeposit(type, n3, n2, n1, axis, sign, pphot->wp[ip]*pphot->ep[ip],
                         pphot->scp[ip] + pphot->acp[ip]);
    return;
  }
  // out of the block: the lab energy and direction ride along, the face of the new cell
  // the walker enters through is encoded in the flag, and the cell update hands the
  // walker to the neighbour block (the indices still name this cell, so the update
  // sees the crossing)
  pphot->ddmcp[ip] = 2 + 2*axis + (up ? 0 : 1);
  pmcb->nddmc[DDMC_HOP_BLOCK]++;
  UpdateZone(pphot, ip);
  if (pphot->statp[ip] == EVOLVING) {
    // the face was not a block face after all; BuildDDMCRates should have made it a
    // boundary face
    pphot->PrintPhoton("Warning: walker hopped out of its block through a "
                       "face that is not a block face, destroyed", ip);
    pphot->statp[ip] = DESTROYED;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real PhotonPusher::DDMCNeighbourExtinction(Photon *pphot, int ip, int n3, int n2,
//!                                                int n1)
//! \brief the total extinction, Thomson plus absorption at the photon's energy, of cell
//!        (n3,n2,n1), in cm^-1 comoving; the absorption function reads the photon's cell,
//!        so the indices are lent to it

Real PhotonPusher::DDMCNeighbourExtinction(Photon *pphot, int ip, int n3, int n2,
                                           int n1) {
  MonteCarloBlock *pmcb = pmy_mcb;
  const int i1 = pphot->i1p[ip], i2 = pphot->i2p[ip], i3 = pphot->i3p[ip];
  pphot->i1p[ip] = n1;
  pphot->i2p[ip] = n2;
  pphot->i3p[ip] = n3;
  const Real kabs = pmcb->AbsorptionOpacity(pmcb, pphot, ip);
  pphot->i1p[ip] = i1;
  pphot->i2p[ip] = i2;
  pphot->i3p[ip] = i3;
  return pmcb->species(0,n3,n2,n1)*MCConstants::sigmat + kabs;
}

//----------------------------------------------------------------------------------------
//! \fn Real PhotonPusher::DDMCRates(Photon *pphot, int ip, Real rate[6])
//! \brief the six face leakage rates of the walker's cell and their sum.  The table
//!        holds them for pure Thomson scattering; a walker that is absorbed as well
//!        diffuses with the total extinction, so its rates are formed here from the
//!        Thomson extinction plus the absorption at its energy, in its cell and in each
//!        neighbouring diffusion cell (the formulas of BuildDDMCRates)

Real PhotonPusher::DDMCRates(Photon *pphot, int ip, Real rate[6]) {
  MonteCarloBlock *pmcb = pmy_mcb;
  const int i1 = pphot->i1p[ip], i2 = pphot->i2p[ip], i3 = pphot->i3p[ip];
  if (!(pphot->acp[ip] > 0.)) {
    for (int f=0; f<6; ++f) rate[f] = pmcb->ddmc_rate(f, i3, i2, i1);
    return pmcb->ddmc_rate(MonteCarloBlock::DDMC_RATE_TOT, i3, i2, i1);
  }
  const Real chi_c = pphot->scp[ip] + pphot->acp[ip];
  Real total = 0.;
  for (int axis=0; axis<3; ++axis) {
    const Real wc = pmcb->ddmc_rate(MonteCarloBlock::DDMC_RATE_W + axis, i3, i2, i1);
    for (int side=0; side<2; ++side) {
      const int f = 2*axis + side;
      if (pmcb->ddmc_rate(f, i3, i2, i1) <= 0.) {
        rate[f] = 0.;  // a direction the mesh does not resolve
        continue;
      }
      const int n1 = i1 + ((axis == 0) ? (side ? 1 : -1) : 0);
      const int n2 = i2 + ((axis == 1) ? (side ? 1 : -1) : 0);
      const int n3 = i3 + ((axis == 2) ? (side ? 1 : -1) : 0);
      if (pmcb->ddmc_cell(n3,n2,n1) != 0) {
        Real Wn[3];
        pmcb->DDMCCellWidths(n3, n2, n1, Wn);
        const Real chi_n = DDMCNeighbourExtinction(pphot, ip, n3, n2, n1);
        rate[f] = 2./(3.*wc*(chi_c*wc + chi_n*Wn[axis]*pmcb->l_cgs));
      } else {
        rate[f] = 2./(3.*wc*(chi_c*wc + 2.*kLambda));
      }
      total += rate[f];
    }
  }
  return total;
}

//----------------------------------------------------------------------------------------
//! \fn void PhotonPusher::DDMCWalk(Photon *pphot, MCRandom *pran, int ip)
//! \brief the walk: dwell, deposit, hop, until it emerges as a transport
//!        photon, leaves the block, is absorbed, or runs out of the cycle's budget

void PhotonPusher::DDMCWalk(Photon *pphot, MCRandom *pran, int ip) {
  MonteCarloBlock *pmcb = pmy_mcb;
  const Real l_cgs = pmcb->l_cgs;
  const Real c_code = MCConstants::c_cgs/pmcb->vel_cgs;
  const int type = pphot->type[ip];
  const enum AbsorptionMethodFlag absm = pmy_mc->absorption_method[type];

  if (pphot->ddmcp[ip] >= 2) DDMCArrive(pphot, ip);

  while (pphot->statp[ip] == EVOLVING && pphot->ddmcp[ip] == 1) {
    const int i1 = pphot->i1p[ip], i2 = pphot->i2p[ip], i3 = pphot->i3p[ip];
    if (pmcb->ddmc_cell(i3,i2,i1) == 0) {
      DDMCRelease(pphot, pran, ip);
      break;
    }
    Real rate[6];
    const Real ktot = DDMCRates(pphot, ip, rate);
    if (!(ktot > 0.)) {
      DDMCRelease(pphot, pran, ip);
      break;
    }
    // the fluid's velocity on the lab legs, for the moments and the time
    Real beta[3] = {0., 0., 0.};
    Real gam = 1.;
    if (boosts && pmcb->vel.GetSize() > 0) {
      gam = pmcb->vel(i3,i2,i1,0);
      for (int m=0; m<3; ++m) beta[m] = pmcb->vel(i3,i2,i1,m+1)/gam;
    }
    // the dwell: comoving path c t, capped by the cycle's remaining lab time
    Real ct_b = HUGE_NUMBER;
    if (pphot->dtp[ip] < 0.5*HUGE_NUMBER) ct_b = pphot->dtp[ip]*c_code*l_cgs/gam;
    Real ct = -std::log(std::max(1. - pran->uniform(), 1.e-300))/ktot;
    bool stop = false;
    if (ct >= ct_b) {
      ct = std::max(ct_b, 0.);
      stop = true;
    }
    // absorption over the dwell: the survival, and the mean weight along it for the
    // time-spent estimator
    const Real w0 = pphot->wp[ip], e0 = pphot->ep[ip];
    const Real kabs = pphot->acp[ip];
    Real surv = 1., wmean = w0;
    if (kabs > 0.) {
      const Real tau = kabs*ct;
      surv = std::exp(-tau);
      if (tau > 1.e-6) wmean *= (1. - surv)/tau;
    }
    if (pmcb->call_moments) {
      pphot->wp[ip] = wmean;
      pmcb->UpdateMomentsMRW(pphot, ct, e0, beta, gam, ip);
      pphot->wp[ip] = w0;
    }
    if (absm == ABSPROB) {
      if (pran->uniform() > surv) {
        pphot->wp[ip] = 0.;
        pphot->statp[ip] = ABSORBED;
      }
    } else {
      pphot->wp[ip] *= surv;
      if (pphot->wp[ip] <= pmcb->minweight) pmcb->RouletteOrAbsorb(pphot, ip);
    }
    if (pmcb->call_srcterms && pphot->wp[ip] != w0)
      pmcb->UpdateSourceTerms(pphot, e0, w0, 0., 0., 0., ip);
    // time (lab), the scatterings the dwell stands for, and what a user hook reads
    const Real dlab = gam*ct/l_cgs;
    pphot->x0p[ip] += dlab;
    pphot->dtp[ip] -= dlab/c_code;
    // the scatterings the dwell stood for, up to the absorption in expectation
    const Real ct_eff = (kabs > 0. && kabs*ct > 1.e-6) ? (1. - surv)/kabs : ct;
    pphot->nscp[ip] += static_cast<int>(std::lround(pphot->scp[ip]*ct_eff));
    pmcb->nddmc_scat += pphot->scp[ip]*ct_eff;
    pmcb->nddmc[DDMC_DWELL]++;
    pmcb->lb_nstep++;
    dl = dlab;
    if (UserWorkInMove != NULL) UserWorkInMove(pmcb, pphot, this, ip);
    if (pphot->statp[ip] != EVOLVING) break;
    if (stop) {
      pmcb->nddmc[DDMC_STOP]++;
      break;
    }
    // the face, in proportion to the rates
    Real v = pran->uniform()*ktot;
    int face = 0;
    while (face < 5 && v >= rate[face]) {
      v -= rate[face];
      ++face;
    }
    const int axis = face/2;
    const bool up = (face & 1) != 0;
    const int n1 = i1 + ((axis == 0) ? (up ? 1 : -1) : 0);
    const int n2 = i2 + ((axis == 1) ? (up ? 1 : -1) : 0);
    const int n3 = i3 + ((axis == 2) ? (up ? 1 : -1) : 0);
    if (pmcb->ddmc_cell(n3,n2,n1) != 0) {
      DDMCHop(pphot, pran, ip, face);
    } else {
      DDMCEmerge(pphot, pran, ip, face);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn bool PhotonPusher::DDMCBeforeFlight(Photon *pphot, MCRandom *pran, int ip,
//!                                         bool first)
//! \brief the lattice's turn before a transport flight.  On the first pass of a Move
//!        call a transport photon starting a fresh flight after a scattering in a
//!        diffusion cell is converted; a walker then walks.  Returns true when the
//!        caller should fly the photon: it is a transport photon and evolving.

bool PhotonPusher::DDMCBeforeFlight(Photon *pphot, MCRandom *pran, int ip, bool first) {
  if (pphot->statp[ip] != EVOLVING) return false;
  // A photon that has just scattered in a diffusion cell is converted; one just emitted
  // there flies its first flight first, so that a photon born in the top layer of the
  // lattice can escape directly from the position it was born at instead of from the
  // cell's centre
  if (first && pphot->ddmcp[ip] == 0 && pphot->taup[ip] < 0. && pphot->nscp[ip] > 0
      && pmy_mcb->ddmc_cell(pphot->i3p[ip], pphot->i2p[ip], pphot->i1p[ip]) != 0)
    DDMCConvert(pphot, ip);
  if (pphot->ddmcp[ip] != 0) DDMCWalk(pphot, pran, ip);
  return pphot->statp[ip] == EVOLVING && pphot->ddmcp[ip] == 0;
}

//----------------------------------------------------------------------------------------
//! \fn bool PhotonPusher::DDMCFaceTest(Photon *pphot, MCRandom *pran, int ip, int face,
//!                                     const bool ascend[3])
//! \brief a transport photon at a face of its cell, about to cross: when the cell beyond
//!        is a diffusion cell the photon is converted with the probability of the
//!        asymptotic boundary condition, P(mu) = 4 (1 + 3 mu / 2) / (3 chi Delta + 6
//!        lambda), or else sent back into its cell with an isotropic inward direction,
//!        keeping its flight.  Returns true when it did either (the caller does not
//!        cross), false when the face is not an interface.  A crossing through an edge
//!        or corner (face codes 3 to 6) is left alone; the loop-start conversion
//!        catches the photon at its next scattering.

bool PhotonPusher::DDMCFaceTest(Photon *pphot, MCRandom *pran, int ip, int face,
                                const bool ascend[3]) {
  if (face < 0 || face > 2) return false;
  MonteCarloBlock *pmcb = pmy_mcb;
  const int axis = face;
  const bool up = ascend[axis];
  const int i1 = pphot->i1p[ip], i2 = pphot->i2p[ip], i3 = pphot->i3p[ip];
  const int n1 = i1 + ((axis == 0) ? (up ? 1 : -1) : 0);
  const int n2 = i2 + ((axis == 1) ? (up ? 1 : -1) : 0);
  const int n3 = i3 + ((axis == 2) ? (up ? 1 : -1) : 0);
  if (pmcb->ddmc_cell(n3,n2,n1) == 0) return false;

  Real k[3] = {pphot->k1p[ip], pphot->k2p[ip], pphot->k3p[ip]};
  const Real knorm = std::sqrt(SQR(k[0]) + SQR(k[1]) + SQR(k[2]));
  const Real mu = (knorm > 0.) ? std::abs(k[axis])/knorm : 1.;
  // the cell beyond, with its absorption at the photon's energy (lab energy here, the
  // comoving one to first order)
  const Real chi_n = DDMCNeighbourExtinction(pphot, ip, n3, n2, n1);
  Real Wn[3];
  pmcb->DDMCCellWidths(n3, n2, n1, Wn);
  // the W-weighted incident intensity of the boundary condition is a mean over the
  // sphere while the incident flux is over the hemisphere, hence the 4
  Real prob = 4.*(1. + 1.5*mu)/(3.*chi_n*Wn[axis]*pmcb->l_cgs + 6.*kLambda);
  if (prob > 1.) {
    prob = 1.;
    pmcb->nddmc[DDMC_PCLAMP]++;
  }
  const Real sign = up ? 1. : -1.;
  if (pran->uniform() < prob) {
    pmcb->nddmc[DDMC_CONVERT_FACE]++;
    const bool ghost = (n1 < pmcb->is || n1 > pmcb->ie || n2 < pmcb->js || n2 > pmcb->je
                        || n3 < pmcb->ks || n3 > pmcb->ke);
    DDMCPlaceAtCentre(pphot, ip, n3, n2, n1);
    if (IsPolarized(pmy_mc->polarized)) {
      pphot->sqp[ip] = 0.;
      pphot->sup[ip] = 0.;
      pphot->svp[ip] = 0.;
    }
    if (!ghost) {
      pphot->i1p[ip] = n1;
      pphot->i2p[ip] = n2;
      pphot->i3p[ip] = n3;
      DDMCSetWalker(pphot, ip);
      pmcb->DDMCHopDeposit(pphot->type[ip], n3, n2, n1, axis, sign,
                           pphot->wp[ip]*pphot->ep[ip], pphot->scp[ip] + pphot->acp[ip]);
      return true;
    }
    // into the neighbour block as a walker owed its entering deposit there; the lab
    // energy and direction ride along for the frame change on arrival
    pphot->ddmcp[ip] = 2 + 2*axis + (up ? 0 : 1);
    pphot->taup[ip] = -1.;
    pphot->strp[ip] = 0.;
    pphot->nmvp[ip] = 0;
    pmcb->nddmc[DDMC_HOP_BLOCK]++;
    UpdateZone(pphot, ip);
    if (pphot->statp[ip] == EVOLVING) {
      pphot->PrintPhoton("Warning: photon converted into a ghost cell that is not across "
                         "a block face, destroyed", ip);
      pphot->statp[ip] = DESTROYED;
    }
    return true;
  }
  // back into the cell it came from: an isotropic inward direction, a hair inside
  pmcb->nddmc[DDMC_RETURN]++;
  FaceDirection(axis, -sign, std::max(pran->uniform(), 1.e-12),
                2.*PI*pran->uniform(), k);
  pphot->k1p[ip] = k[0];
  pphot->k2p[ip] = k[1];
  pphot->k3p[ip] = k[2];
  const Real lo = (axis == 0) ? pcoord->x1f(i1)
                  : (axis == 1) ? pcoord->x2f(i2) : pcoord->x3f(i3);
  const Real hi = (axis == 0) ? pcoord->x1f(i1+1) : (axis == 1) ? pcoord->x2f(i2+1)
                                                  : pcoord->x3f(i3+1);
  const Real x = up ? hi - kFaceNudge*(hi - lo) : lo + kFaceNudge*(hi - lo);
  if (axis == 0) pphot->x1p[ip] = x;
  else if (axis == 1) pphot->x2p[ip] = x;
  else pphot->x3p[ip] = x;
  return true;
}
