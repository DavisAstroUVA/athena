//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file polarization.cpp
//! \brief supports moving a photon into and out of the polarized scattering basis
//
// See polarization.hpp for the convention these two functions implement.

// C++ headers
#include <cmath>
#include <complex>

// Athena++ headers
#include "../athena.hpp"
#include "polarization.hpp"
#include "montecarlo.hpp"
#include "photon.hpp"
#include "tetrad.hpp"
#include "mccoord.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ToScatteringBasis(MonteCarloBlock *pmcb, Photon *pphot, int ip)
//! \brief rotate the photon into the basis the polarized scattering routines assume
//
// On a spherical grid the wavevector is stored in the local orthonormal spherical basis,
// which rotates along the ray; the scattering routines want cartesian components.  Q and
// U are deliberately untouched: they are already referenced to the global cartesian
// meridian, which does not move. Used by non GR coordinates.


static bool MeridianPair(const Real n[3], const Real zref[3],
                         Real lhat[3], Real rhat[3]);
static void WriteMeridianStokes(Photon *pphot, int ip, Real ecov[4][4],
                                const Real lhat[3], const Real rhat[3]);

void ToScatteringBasis(MonteCarloBlock *pmcb, Photon *pphot, int ip) {

  if (pmcb->topology != MCTOPO_SPHERICAL) return;

  Real cth = std::cos(pphot->x2p[ip]);
  Real sth = std::sin(pphot->x2p[ip]);
  Real cph = std::cos(pphot->x3p[ip]);
  Real sph = std::sin(pphot->x3p[ip]);

  Real k3[3];
  k3[0] = pphot->k1p[ip];
  k3[1] = pphot->k2p[ip];
  k3[2] = pphot->k3p[ip];

  pphot->k1p[ip] = k3[0]*sth*cph + k3[1]*cth*cph - k3[2]*sph;
  pphot->k2p[ip] = k3[0]*sth*sph + k3[1]*cth*sph + k3[2]*cph;
  pphot->k3p[ip] = k3[0]*cth     - k3[1]*sth;

  Real knorm = std::sqrt(SQR(pphot->k1p[ip]) + SQR(pphot->k2p[ip]) + SQR(pphot->k3p[ip]));
  pphot->k1p[ip] /= knorm;
  pphot->k2p[ip] /= knorm;
  pphot->k3p[ip] /= knorm;
}

//----------------------------------------------------------------------------------------
//! \fn void FromScatteringBasis(MonteCarloBlock *pmcb, Photon *pphot, int ip)
//! \brief rotate the photon back out of the scattering basis
//
// Exact inverse of ToScatteringBasis, evaluated at the same position, so the round trip
// is the identity to round-off. Used by non GR coordinates.

void FromScatteringBasis(MonteCarloBlock *pmcb, Photon *pphot, int ip) {

  if (pmcb->topology != MCTOPO_SPHERICAL) return;

  Real cth = std::cos(pphot->x2p[ip]);
  Real sth = std::sin(pphot->x2p[ip]);
  Real cph = std::cos(pphot->x3p[ip]);
  Real sph = std::sin(pphot->x3p[ip]);

  Real k3[3];
  k3[0] = pphot->k1p[ip];
  k3[1] = pphot->k2p[ip];
  k3[2] = pphot->k3p[ip];

  pphot->k1p[ip] =  k3[0]*sth*cph + k3[1]*sth*sph + k3[2]*cth;
  pphot->k2p[ip] =  k3[0]*cth*cph + k3[1]*cth*sph - k3[2]*sth;
  pphot->k3p[ip] = -k3[0]*sph     + k3[1]*cph;

  Real knorm = std::sqrt(SQR(pphot->k1p[ip]) + SQR(pphot->k2p[ip]) + SQR(pphot->k3p[ip]));
  pphot->k1p[ip] /= knorm;
  pphot->k2p[ip] /= knorm;
  pphot->k3p[ip] /= knorm;
}

//----------------------------------------------------------------------------------------
//! \fn static bool MeridianBasis(...)
//! \brief the scattering basis in the comoving orthonormal frame
//
// Builds the comoving tetrad E -- the same construction TransformToComoving uses, so the
// photon's stored k is expressed in it -- and returns the pair of unit three-vectors the
// scattering routines reference Q and U to:
//
//   n = k-hat in E,   l = normalize(z - (z.n) n),   r = n x l,
//
// with z the third spatial leg of E.  That is the meridian plane containing the frame's
// z axis and the photon direction, which is the convention ScatterThomsonPolarized reads
// off the stored components (see polarization.hpp).  l goes into e_(1) because
// StokesToTensor puts I+Q there, i.e. Q > 0 means polarization along the meridian.
//
// Returns false when n is parallel to z, where the meridian is undefined; the caller
// then leaves the Stokes parameters alone

static bool MeridianBasis(MonteCarloBlock *pmcb, Photon *pphot, int ip,
                          Real econ[4][4], Real ecov[4][4], Real lhat[3], Real rhat[3]) {

  int i1 = pphot->i1p[ip], i2 = pphot->i2p[ip], i3 = pphot->i3p[ip];

  Real x[4];
  x[IMC0] = pphot->x0p[ip];
  x[IMC1] = pphot->x1p[ip];
  x[IMC2] = pphot->x2p[ip];
  x[IMC3] = pphot->x3p[ip];
  Real gcov[4][4];
  pmcb->pcoord->Metric(x, gcov);

  // The frame k is expressed in at a scattering.  The MonteCarloBlock constructor
  // guarantees vel is allocated whenever polarized is set, holding the fluid velocity,
  // the normal observer or the coordinate observer as appropriate, so no test is needed
  // here.
  //
  // In general relativity the frame is rebuilt at the photon instead, so that it is a
  // unit timelike vector where gcov above is evaluated.  uprim only exists in GR, which
  // is why the flat path still reads vel; there the metric is constant across a cell and
  // the two agree anyway.
  Real ucon[4];
  if (GENERAL_RELATIVITY) {
    pmcb->FluidFourVelocity(x, i3, i2, i1, ucon);
  } else {
    for (int m = 0; m < 4; ++m) ucon[m] = pmcb->vel(i3, i2, i1, m);
  }
  ConstructTetrad(ucon, gcov, econ, ecov);

  // k as stored is already in this frame and is a unit direction there
  Real n[3];
  n[0] = pphot->k1p[ip];
  n[1] = pphot->k2p[ip];
  n[2] = pphot->k3p[ip];
  Real nmag = std::sqrt(SQR(n[0]) + SQR(n[1]) + SQR(n[2]));
  if (nmag <= TINY_NUMBER) return false;
  for (int i = 0; i < 3; ++i) n[i] /= nmag;

  // the scattering routines read Q and U against the comoving frame's own third leg
  const Real zleg[3] = {0.0, 0.0, 1.0};
  return MeridianPair(n, zleg, lhat, rhat);
}

//----------------------------------------------------------------------------------------
//! \fn static bool MeridianPair(const Real n[3], const Real zref[3], ...)
//! \brief the meridian pair for a unit direction in some orthonormal frame
//
//   l = normalize(zref - (zref.n) n),   r = n x l,
//
// Split out so that both the frame and the axis within it can be chosen by the caller: the
// scattering routines want the comoving frame's own third leg, while the outputs want a
// single physical direction shared by every photon.  zref is given in the same tetrad
// components as n and must be a unit vector.  Returns false when n is parallel to zref,
// where the meridian is undefined.

static bool MeridianPair(const Real n[3], const Real zref[3],
                         Real lhat[3], Real rhat[3]) {

  Real zdn = zref[0]*n[0] + zref[1]*n[1] + zref[2]*n[2];
  Real l[3] = {zref[0] - zdn*n[0], zref[1] - zdn*n[1], zref[2] - zdn*n[2]};
  Real lmag = std::sqrt(SQR(l[0]) + SQR(l[1]) + SQR(l[2]));
  if (lmag <= TINY_NUMBER) return false;          // photon along the frame z axis
  for (int i = 0; i < 3; ++i) lhat[i] = l[i]/lmag;

  // r = n x l, completing a right-handed triad (l, r, n)
  rhat[0] = n[1]*lhat[2] - n[2]*lhat[1];
  rhat[1] = n[2]*lhat[0] - n[0]*lhat[2];
  rhat[2] = n[0]*lhat[1] - n[1]*lhat[0];

  return true;
}

//----------------------------------------------------------------------------------------
//! \fn void CoherencyToScatteringStokes(MonteCarloBlock *pmcb, Photon *pphot, int ip)
//! \brief coherency tensor -> Stokes, in the scattering basis

void CoherencyToScatteringStokes(MonteCarloBlock *pmcb, Photon *pphot, int ip) {

  if (!pmcb->pmy_mc->general_pusher_flag) return;

  Real econ[4][4], ecov[4][4], lhat[3], rhat[3];
  if (!MeridianBasis(pmcb, pphot, ip, econ, ecov, lhat, rhat)) return;

  WriteMeridianStokes(pphot, ip, ecov, lhat, rhat);
}

//----------------------------------------------------------------------------------------
//! \fn static void WriteMeridianStokes(...)
//! \brief project the coherency tensor onto a meridian pair and store the Stokes
//!        parameters the scattering routines and the outputs both read

static void WriteMeridianStokes(Photon *pphot, int ip, Real ecov[4][4],
                                const Real lhat[3], const Real rhat[3]) {

  std::complex<Real> ntet[4][4];
  pphot->PolarizationToTetrad(ntet, ecov, ip);

  // Contract the spatial block onto the meridian pair
  std::complex<Real> nll(0., 0.), nrr(0., 0.), nlr(0., 0.), nrl(0., 0.);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const std::complex<Real> &nij = ntet[i+1][j+1];
      nll += lhat[i]*lhat[j]*nij;
      nrr += rhat[i]*rhat[j]*nij;
      nlr += lhat[i]*rhat[j]*nij;
      nrl += rhat[i]*lhat[j]*nij;
    }
  }

  // Pass the meridian pair to TensorToStokes the(l,r) contraction is the (1,2) block it
  // reads, with l and r standing in for e_(1) and e_(2).
  Real inten = 0.5*(nll + nrr).real();
  if (std::fabs(inten) <= TINY_NUMBER) return;   // TensorToStokes divides by I

  std::complex<Real> nmer[4][4];
  for (int a = 0; a < 4; ++a)
    for (int b = 0; b < 4; ++b) nmer[a][b] = std::complex<Real>(0., 0.);
  nmer[IMC1][IMC1] = nll;
  nmer[IMC1][IMC2] = nlr;
  nmer[IMC2][IMC1] = nrl;
  nmer[IMC2][IMC2] = nrr;

  Real stokes[4];
  TensorToStokes(nmer, stokes);  // returns I normalized to one

  pphot->sip[ip] = stokes[0];
  pphot->sqp[ip] = stokes[1];
  pphot->sup[ip] = stokes[2];
  pphot->svp[ip] = stokes[3];
}

//----------------------------------------------------------------------------------------
//! \fn bool NormalFrameWavevector(MonteCarloBlock *pmcb, Photon *pphot, int ip,
//!                                Real econ[4][4], Real ecov[4][4], Real ktet[4])
//! \brief the normal observer's orthonormal frame at the photon, and the photon's
//!        wavevector expressed in it
//
// The wavevector written to a photon list, the direction a spectrum bins its angles on,
// and the plane the Stokes parameters are referenced to are here Anything that
// needs the output frame should call this rather than reconstruct it
//
// The photon is in the coordinate frame whenever this is called, so the wavevector is
// projected here rather than being read off the photon.  Because the frame is orthonormal
// and the photon is null, the components that come back satisfy |k_spatial| = k^(0); a
// reader can rely on that, and on the frame's third spatial leg being (0,0,1), to rebuild
// the meridian basis from the stored components alone.
//
// Returns false when there is no normal observer, which NormalObserver reports when
// g^{tt} is not negative.  econ, ecov and ktet are left untouched in that case.

bool NormalFrameWavevector(MonteCarloBlock *pmcb, Photon *pphot, int ip,
                           Real econ[4][4], Real ecov[4][4], Real ktet[4]) {

  Real x[4];
  x[IMC0] = pphot->x0p[ip];
  x[IMC1] = pphot->x1p[ip];
  x[IMC2] = pphot->x2p[ip];
  x[IMC3] = pphot->x3p[ip];
  Real gcov[4][4], gcon[4][4];
  pmcb->pcoord->Metric(x, gcov);
  pmcb->pcoord->InverseMetric(x, gcon);

  Real ncon[4];
  if (!NormalObserver(gcon, ncon)) return false;
  ConstructTetrad(ncon, gcov, econ, ecov);

  // The photon is in the coordinate frame at this point, so the wavevector has to be
  // projected before anything can be measured against the observer's axes.
  Real kcoord[4];
  pphot->GetFourVector(ip, false, kcoord);
  CoordinateToTetrad(kcoord, ktet, ecov);
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn void CoherencyToObserverStokes(MonteCarloBlock *pmcb, Photon *pphot, int ip)
//! \brief refresh the stored Stokes parameters from the coherency tensor, in the frame
//!        the outputs are referenced to
//
// Sets sqp/sup/svp to the Stokes parameters referenced to the meridian containing the
// normal observer's z axis and the photon's direction, which is what the outputs write.
// The frame is the normal (Eulerian) observer at the photon's position, n^mu = -alpha
// g^{mu t}, rather than the fluid. Q is positive along the meridian containing the
// frame's third spatial leg

void CoherencyToObserverStokes(MonteCarloBlock *pmcb, Photon *pphot, int ip) {

  // The legacy pushers reference the Stokes parameters to a globally constant frame and
  // do not evolve them between scatterings, so they are already current.
  if (!pmcb->pmy_mc->general_pusher_flag) return;

  Real econ[4][4], ecov[4][4], ktet[4];
  if (!NormalFrameWavevector(pmcb, pphot, ip, econ, ecov, ktet)) return;

  Real nmag = std::sqrt(SQR(ktet[IMC1]) + SQR(ktet[IMC2]) + SQR(ktet[IMC3]));
  if (nmag <= TINY_NUMBER) return;
  Real n[3] = {ktet[IMC1]/nmag, ktet[IMC2]/nmag, ktet[IMC3]/nmag};

  // Reference axis, in tetrad components.  (0,0,1) is the tetrad's third spatial leg,
  // which is what ConstructTetrad grows from the third coordinate direction: e_z in a
  // cartesian or cylindrical chart, and so already the global z axis there.  In spherical
  // polar it is e_phi, which rotates with azimuth -- Q and U would then be referenced to a
  // different physical plane for every photon, they could not be combined across the grid,
  // and they would not vanish for a photon leaving along the polar axis.  Project the
  // global z direction into the tetrad instead, so the stored Stokes parameters mean the
  // same thing everywhere and agree with a spectrum binned on polar_axis.
  Real zref[3] = {0.0, 0.0, 1.0};
  if (pmcb->topology == MCTOPO_SPHERICAL) {
    // z = r cos(theta), so d/dz = cos(theta) d/dr - (sin(theta)/r) d/dtheta
    Real sth = std::sin(pphot->x2p[ip]);
    Real cth = std::cos(pphot->x2p[ip]);
    Real rad = pphot->x1p[ip];
    if (rad <= TINY_NUMBER) return;
    Real zcoord[4], ztet[4];
    zcoord[IMC0] = 0.0;
    zcoord[IMC1] = cth;
    zcoord[IMC2] = -sth/rad;
    zcoord[IMC3] = 0.0;
    CoordinateToTetrad(zcoord, ztet, ecov);
    Real zmag = std::sqrt(SQR(ztet[IMC1]) + SQR(ztet[IMC2]) + SQR(ztet[IMC3]));
    if (zmag <= TINY_NUMBER) return;
    zref[0] = ztet[IMC1]/zmag;
    zref[1] = ztet[IMC2]/zmag;
    zref[2] = ztet[IMC3]/zmag;
  }

  Real lhat[3], rhat[3];
  if (!MeridianPair(n, zref, lhat, rhat)) return;   // photon along the reference axis

  WriteMeridianStokes(pphot, ip, ecov, lhat, rhat);
}

//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
//! \fn void ScatteringStokesToCoherency(MonteCarloBlock *pmcb, Photon *pphot, int ip)
//! \brief Stokes -> coherency tensor, undoing CoherencyToScatteringStokes
//
// Rebuilds the basis at the photon's current direction, which after a scattering is the
// outgoing one, since the outgoing Stokes parameters are referenced to the outgoing
// meridian.

void ScatteringStokesToCoherency(MonteCarloBlock *pmcb, Photon *pphot, int ip) {

  if (!pmcb->pmy_mc->general_pusher_flag) return;

  Real econ[4][4], ecov[4][4], lhat[3], rhat[3];
  if (!MeridianBasis(pmcb, pphot, ip, econ, ecov, lhat, rhat)) return;

  Real inten = pphot->sip[ip];
  if (std::fabs(inten) <= TINY_NUMBER) return;
  Real q = pphot->sqp[ip]/inten;
  Real u = pphot->sup[ip]/inten;
  Real v = pphot->svp[ip]/inten;

  // StokesToTensor retruns the four block entries, then spread them over the meridian
  // pair: N = N11 l l + N12 l r + N21 r l + N22 r r.
  Real stokes[4] = {1.0, q, u, v};
  std::complex<Real> blk[4][4];
  StokesToTensor(stokes, blk);

  std::complex<Real> ntet[4][4];
  for (int a = 0; a < 4; ++a)
    for (int b = 0; b < 4; ++b) ntet[a][b] = std::complex<Real>(0., 0.);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      ntet[i+1][j+1] = blk[IMC1][IMC1]*lhat[i]*lhat[j] + blk[IMC1][IMC2]*lhat[i]*rhat[j]
                     + blk[IMC2][IMC1]*rhat[i]*lhat[j] + blk[IMC2][IMC2]*rhat[i]*rhat[j];
    }
  }

  pphot->PolarizationToCoord(ntet, econ, ip);
}
