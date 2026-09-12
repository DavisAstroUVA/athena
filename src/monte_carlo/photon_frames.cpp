//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file photon_frames.cpp
//! \brief projecting photons and radiation moments between reference frames.
//!
//! MCFRAME_LAB and MCFRAME_COMOVING are orthonormal: in general relativity the tetrads of
//! the normal (Eulerian) observer and of the frame vel is built on, and in flat spacetime
//! the Eulerian frame and its Lorentz boost.  MCFRAME_COORD is the coordinate basis.
//!
//! Two operations live here and are easy to conflate.  PhotonFrames projects a single
//! photon, per crossing; ComovingFrameMatrix and DeriveComovingMoments transform an
//! accumulated tensor, once per cell at output.

// C headers

// C++ headers
#include <cmath>
#include <cstring>   // strcmp

// Athena++ headers
#include "photon_frames.hpp"
#include "montecarlo.hpp"
#include "tetrad.hpp"
#include "photon.hpp"
#include "mccoord.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../coordinates/coordinates.hpp"

//----------------------------------------------------------------------------------------
//! \fn PhotonFrames::PhotonFrames(...)
//! \brief bind to one photon; frames are projected lazily as they are asked for

PhotonFrames::PhotonFrames(MonteCarloBlock *pmcb, Photon *pphot, int ip, Real dl)
    : pmcb_(pmcb), pphot_(pphot), ip_(ip), dl_(dl) {
  general_ = pmcb->pmy_mc->general_pusher_flag;
  gr_tetrad_ = general_ && GENERAL_RELATIVITY;
  if (general_) pphot->GetFourVector(ip, false, kco_);
  for (int f=0; f<MCFRAME_N; ++f) done_[f] = false;
}

//----------------------------------------------------------------------------------------
//! \fn void PhotonFrames::Fill(MCFrame f)
//! \brief project the photon into one frame

void PhotonFrames::Fill(MCFrame f) {
  PhotonFrameState &s = st_[f];
  const Real ep = pphot_->ep[ip_];
  const int i1 = pphot_->i1p[ip_], i2 = pphot_->i2p[ip_], i3 = pphot_->i3p[ip_];

  if (f == MCFRAME_COORD) {
    // Not orthonormal: n holds k^i/k^0 and is deliberately not normalized.
    s.e = kco_[IMC0];
    for (int i=0; i<3; ++i) s.n[i] = kco_[IMC1+i]/ep;
    s.dl = dl_;
    return;
  }

  if (gr_tetrad_) {
    // One matrix multiply carries the coordinate four-vector to orthonormal components.
    const AthenaArray<Real> &m = (f == MCFRAME_LAB) ? pmcb_->boost_lab : pmcb_->boost_cmv;
    Real p[4];
    for (int a=0; a<4; ++a) {
      p[a] = 0.;
      for (int b=0; b<4; ++b) p[a] += m(i3,i2,i1,a,b) * kco_[b];
    }
    // k is null at the photon to machine precision, but the tetrad is orthonormal with
    // respect to the metric at the cell center, so the projection is not exactly null.
    // Normalzing by the spatial magnitude keeps n a genuine direction and the moment
    // tensor exactly traceless; the energy comes from the time component.
    Real mag = std::sqrt(SQR(p[IMC1]) + SQR(p[IMC2]) + SQR(p[IMC3]));
    s.e = p[IMC0];
    for (int i=0; i<3; ++i) s.n[i] = p[IMC1+i]/mag;
    s.dl = dl_ * s.e / ep;
    return;
  }

  // ---- flat spacetime ----
  if (f == MCFRAME_LAB) {
    if (general_) {
      Real x[4] = {pphot_->x0p[ip_], pphot_->x1p[ip_], pphot_->x2p[ip_], pphot_->x3p[ip_]};
      Real invtet[4][4], kf[4];
      pmcb_->pcoord->InverseTetrad(x, invtet);
      for (int a=0; a<4; ++a) {
        kf[a] = 0.;
        for (int b=0; b<4; ++b) kf[a] += invtet[a][b] * kco_[b];
      }
      // The tetrad time leg is unity for every flat coordinate system, so kf[0] == ep and
      // dividing the spatial parts by ep is the same unit direction the legacy pushers
      // store.  Kept in this form rather than normalized by the spatial magnitude so the
      // arithmetic is unchanged from before this was factored out.
      s.e = kf[IMC0];
      for (int i=0; i<3; ++i) s.n[i] = kf[IMC1+i]/ep;
      // InverseTetrad is evaluated at the photon, so what comes back is the direction in
      // the orthonormal basis there.  The same rotation the legacy pushers apply carries
      // it to the cell center, which is the basis the moments are summed in; without it
      // this branch and the legacy one disagree about the same photon.
      ToCellCenterBasis(s.n);
    } else {
      s.e = ep;
      s.n[0] = pphot_->k1p[ip_];
      s.n[1] = pphot_->k2p[ip_];
      s.n[2] = pphot_->k3p[ip_];
      ToCellCenterBasis(s.n);
    }
    s.dl = dl_;
    return;
  }

  // comoving in flat spacetime: boost the lab direction, which is where boost_cmv acts
  const PhotonFrameState &lab = Get(MCFRAME_LAB);
  Real ki[4] = {1., lab.n[0], lab.n[1], lab.n[2]};
  Real kc[4];
  for (int a=0; a<4; ++a) {
    kc[a] = 0.;
    for (int b=0; b<4; ++b) kc[a] += pmcb_->boost_cmv(i3,i2,i1,a,b) * ki[b];
  }
  Real shift = kc[IMC0];
  s.e = ep * shift;
  for (int i=0; i<3; ++i) s.n[i] = kc[IMC1+i]/kc[IMC0];
  s.dl = dl_ * shift;
}

//----------------------------------------------------------------------------------------
//! \fn void PhotonFrames::ToCellCenterBasis(Real n[3]) const
//! \brief carry a unit direction from the orthonormal basis at the photon to the one at
//!        its cell center
//
// The moments are summed over the photons crossing a cell, so they need every
// contribution referred to one basis.  On a curvilinear grid the orthonormal legs turn
// with position, by the full angular width of a cell between one face and the other, so
// the choice of basis may not be small. In flat spacetime this is an exact rotation.

//----------------------------------------------------------------------------------------
//! \fn void ComovingFrame(MonteCarloBlock *pmcb, Photon *pphot, int ip,
//!                        Real econ[4][4], Real ecov[4][4])
//! \brief the comoving orthonormal frame at a photon; see photon_frames.hpp
//!
//! econ and ecov are indexed [tetrad leg][coordinate component], for CoordinateToTetrad,
//! TetradToCoordinate, PolarizationToTetrad and PolarizationToCoord.

void ComovingFrame(MonteCarloBlock *pmcb, Photon *pphot, int ip,
                   Real econ[4][4], Real ecov[4][4]) {

  Real x[4];
  x[IMC0] = pphot->x0p[ip];
  x[IMC1] = pphot->x1p[ip];
  x[IMC2] = pphot->x2p[ip];
  x[IMC3] = pphot->x3p[ip];
  const int i1 = pphot->i1p[ip], i2 = pphot->i2p[ip], i3 = pphot->i3p[ip];

  // The metric at the photon rather than at the cell center, so the leg is a unit
  // timelike vector where it is used.
  Real gcov[4][4];
  pmcb->pcoord->Metric(x, gcov);

  Real ucon[4];
  if (GENERAL_RELATIVITY) {
    pmcb->FluidFourVelocity(x, i3, i2, i1, ucon);
  } else {
    // vel is (gamma, gamma*beta^i) on the local orthonormal legs.  Carry it onto
    // coordinate components with the static tetrad, whose legs are those of the
    // orthonormal frame at rest; the boost then happens inside ConstructTetrad, by growing
    // the frame from the moving four-velocity.  Tetrad maps orthonormal to coordinate
    // components as ucon[j] = tet[j][i] vel[i], the convention TransformToCoordinate uses.
    Real tet[4][4];
    pmcb->pcoord->Tetrad(x, tet);
    for (int j = 0; j < 4; ++j) {
      ucon[j] = 0.0;
      for (int i = 0; i < 4; ++i) ucon[j] += tet[j][i] * pmcb->vel(i3, i2, i1, i);
    }
  }
  ConstructTetrad(ucon, gcov, econ, ecov);
}

//----------------------------------------------------------------------------------------

void PhotonFrames::ToCellCenterBasis(Real n[3]) const {
  Coordinates *pco = pmcb_->pmy_block->pcoord;

  if (pmcb_->topology == MCTOPO_SPHERICAL) {
    const int i2 = pphot_->i2p[ip_], i3 = pphot_->i3p[ip_];
    Real sth = std::sin(pphot_->x2p[ip_]), cth = std::cos(pphot_->x2p[ip_]);
    Real sph = std::sin(pphot_->x3p[ip_]), cph = std::cos(pphot_->x3p[ip_]);
    Real nx = sth*cph*n[0] + cth*cph*n[1] - sph*n[2];
    Real ny = sth*sph*n[0] + cth*sph*n[1] + cph*n[2];
    Real nz = cth*n[0] - sth*n[1];
    sth = std::sin(pco->x2v(i2));
    cth = std::cos(pco->x2v(i2));
    sph = std::sin(pco->x3v(i3));
    cph = std::cos(pco->x3v(i3));
    n[0] = sth*cph*nx + sth*sph*ny + cth*nz;
    n[1] = cth*cph*nx + cth*sph*ny - sth*nz;
    n[2] = -sph*nx + cph*ny;
  } else if (pmcb_->topology == MCTOPO_CYLINDRICAL) {
    // (R, phi, z) with the azimuth in x2.  Only R-hat and phi-hat turn, by the azimuthal
    // offset, so this is a rotation in the first two components and z is untouched.
    Real d = pphot_->x2p[ip_] - pco->x2v(pphot_->i2p[ip_]);
    Real cd = std::cos(d), sd = std::sin(d);
    Real nr = n[0]*cd - n[1]*sd;
    Real np = n[0]*sd + n[1]*cd;
    n[0] = nr;
    n[1] = np;
  }
  // Cartesian topology: the basis does not depend on position, so there is nothing to do.
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::ComovingFrameMatrix(...)
//! \brief the lab to comoving transformation for one cell
//!
//! Both frames are orthonormal at the same event, so this is a Lorentz transformation.
//! In general relativity it must be composed from the two tetrads rather than rebuilt as
//! a pure boost from beta: the Gram-Schmidt legs are grown from the same coordinate trial
//! vectors against different timelike directions, so the map is a boost composed with a
//! rotation whenever the fluid velocity has all three spatial components.  A pure boost
//! would leave Er correct, being a scalar, while silently rotating flux and pressure.

void MonteCarloBlock::ComovingFrameMatrix(int k, int j, int i, const AthenaArray<Real> &g,
                                          const AthenaArray<Real> &gi, Real lam[4][4]) {
  if (!GENERAL_RELATIVITY) {
    // boost_cmv already maps lab-frame components to the comoving frame.
    for (int a=0; a<4; ++a)
      for (int b=0; b<4; ++b) lam[a][b] = boost_cmv(k,j,i,a,b);
    return;
  }

  Real x[4];
  x[IMC0] = 0.;
  x[IMC1] = pmy_block->pcoord->x1v(i);
  x[IMC2] = pmy_block->pcoord->x2v(j);
  x[IMC3] = pmy_block->pcoord->x3v(k);
  Real gcov[4][4];
  pcoord->Metric(x, gcov);

  Real alpha = 1.0/std::sqrt(-gi(I00,i));
  Real ncon[4];
  ncon[IMC0] = -alpha*gi(I00,i);
  ncon[IMC1] = -alpha*gi(I01,i);
  ncon[IMC2] = -alpha*gi(I02,i);
  ncon[IMC3] = -alpha*gi(I03,i);
  Real econL[4][4], ecovL[4][4];
  ConstructTetrad(ncon, gcov, econL, ecovL);

  // Rebuilt at the cell center, which is the right point here: this matrix is per cell
  // and transforms the accumulated moments, which are cell averages.
  Real ucon[4];
  FluidFourVelocity(x, k, j, i, ucon);
  Real econF[4][4], ecovF[4][4];
  ConstructTetrad(ucon, gcov, econF, ecovF);

  for (int a=0; a<4; ++a) {
    for (int b=0; b<4; ++b) {
      lam[a][b] = 0.;
      for (int m=0; m<4; ++m) lam[a][b] += ecovF[a][m] * econL[b][m];
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::DeriveComovingMoments()
//! \brief fill moments_com by transforming the accumulated lab moments
//!
//! The moments are a rank two tensor and the transformation is the same matrix for every
//! photon in the cell, so sum Lp Lp == L (sum p p) L exactly.  Deriving is therefore not
//! an approximation of accumulating, and it removes the second per-photon projection --
//! measured at 5% of runtime on the general pusher and 12% on the legacy pusher.

void MonteCarloBlock::DeriveComovingMoments() {
  const Real c_cgs = MCConstants::c_cgs;
  AthenaArray<Real> g, gi;
  if (GENERAL_RELATIVITY) {
    g.NewAthenaArray(NMETRIC,ie+1);
    gi.NewAthenaArray(NMETRIC,ie+1);
  }
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      if (GENERAL_RELATIVITY) pmy_block->pcoord->CellMetric(k,j,is,ie,g,gi);
      for (int i=is; i<=ie; ++i) {
        Real lam[4][4];
        ComovingFrameMatrix(k,j,i,g,gi,lam);
        for (int m=0; m<pmy_mc->ntype; ++m) {
          // MomentSlot carries the stored convention, shared with AccumulateMoments
          Real T[4][4];
          for (int a=0; a<4; ++a)
            for (int b=0; b<4; ++b) {
              T[a][b] = moments(m,MomentSlot[a][b],k,j,i);
              if (MomentNeedsC(a,b)) T[a][b] /= c_cgs;
            }

          Real T1[4][4], U[4][4];
          for (int a=0; a<4; ++a)
            for (int b=0; b<4; ++b) {
              T1[a][b] = 0.;
              for (int c=0; c<4; ++c) T1[a][b] += lam[a][c]*T[c][b];
            }
          for (int a=0; a<4; ++a)
            for (int b=0; b<4; ++b) {
              U[a][b] = 0.;
              for (int c=0; c<4; ++c) U[a][b] += T1[a][c]*lam[b][c];
            }

          for (int a=0; a<4; ++a)
            for (int b=a; b<4; ++b) {
              Real v = U[a][b];
              if (MomentNeedsC(a,b)) v *= c_cgs;
              moments_com(m,MomentSlot[a][b],k,j,i) = v;
            }
        }
      }
    }
  }
  if (GENERAL_RELATIVITY) { g.DeleteAthenaArray(); gi.DeleteAthenaArray(); }
}
