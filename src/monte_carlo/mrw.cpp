//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mrw.cpp
//! \brief the modified random walk: one step moves a photon that is deep inside an
//!        optically thick cell through many scatterings at once
//
// Fleck & Canfield (1984), Min et al. (2009), Robitaille (2010), with the energy change
// of Compton scattering drawn from the Kompaneets Green's function (kgreens.hpp).  The
// plan and the reasoning are in doc/monte_carlo/compton_mrw_plan.md, Section 6.
//
// The diffusion domain is the cell itself if accel_domain = cell: a
// rectangular box with absorbing walls, whose Green's function separates into three
// one-dimensional problems, so a photon anywhere in a thick cell can step, including
// one near a face, which steps out through it.  The sphere of the references, the
// largest that fits around the photon, is kept as accel_domain = sphere for comparison
//
// The step is a fixed-time step.  A comoving times limit is chosen first (the remaining
// time of the cycle, the room advection leaves in a moving medium, or none). With the
// survival P of the domain at the times limit, the photon either reached a wall or is
// still inside, at a position drawn from the fixed-time solution.  The new direction is
// assumed isotropic.  At rest with no time limit the sphere version is the fixed-radius
// step of the references.
//
// Energy and weight: for Compton scattering y = theta n_e sigma_T c t and the table gives
// x_f and the free-free survival S; for coherent scattering the energy is unchanged and
// S = exp(-alpha_abs c t).  The path-length estimator gets the whole path, isotropic in
// the comoving frame, at the mean of the two energies.
//
// In a moving medium (boosts, flat spacetime), the walk is done in the comoving frame,
// entered and left with the block loop's own transforms; the fluid element advects by
// beta gamma c t_c in the lab over comoving time t_c, so the domain is shrunk on the
// downstream side by the advection the time limit allows.

// C++ headers
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "kgreens.hpp"
#include "mccoord.hpp"
#include "montecarlo.hpp"
#include "mrw.hpp"
#include "photon.hpp"
#include "photonpusher.hpp"

namespace {

// sphere: first-passage CDF on ln s, and radius quantiles on (ln s, level); below
// kSRadMin the free-space Gaussian is exact to exp(-pi^2/(4 s))
const int kNFirstPassage = 4000;
const Real kSMin = 1.e-3, kSMax = 40.;
const int kNRadS = 160, kNRadQ = 129, kNRadGrid = 2000;
const Real kSRadMin = 0.03;

// box: xi from 1/(2 nxi) to 1/2, ln sigma from kSigMin to kSigMax; positions are
// tabulated from kSigPosMin up and drawn from the image solution below it
const int kNXi = 65, kNSig = 225, kNLev = 65, kNPosGrid = 801;
const Real kSigMin = 1.e-4, kSigMax = 40., kSigPosMin = 0.01;
const Real kEps = 1.e-9;

//! the one-dimensional series at (xi, sigma): survival, -dS/dsigma, and the wall fluxes
//! at 0 and W up to a common factor
void BoxSeries(Real xi, Real sigma, Real &surv, Real &dsurv, Real &flux0, Real &fluxw) {
  surv = dsurv = flux0 = fluxw = 0.;
  const int nmax = std::min(20000, static_cast<int>(std::sqrt(46./sigma)) + 2);
  for (int n=1; n<=nmax; ++n) {
    const Real en = std::exp(-static_cast<Real>(n)*n*sigma);
    const Real sn = std::sin(n*PI*xi);
    flux0 += n*sn*en;
    fluxw += ((n % 2 == 1) ? n*sn*en : -n*sn*en);
    if (n % 2 == 1) {
      surv += 4./(n*PI)*sn*en;
      dsurv += 4.*n/PI*sn*en;
    }
  }
}

} // namespace

//----------------------------------------------------------------------------------------
//! MRWTables constructor: both domains' tables

MRWTables::MRWTables() {
  sigma_min = kSigMin;
  sigma_max = kSigMax;
  BuildSphere();
  BuildBox();
}

//----------------------------------------------------------------------------------------
//! \fn Real MRWTables::SphereInside(Real s)
//! \brief fraction of walkers still inside the sphere at s

Real MRWTables::SphereInside(Real s) {
  if (s <= 0.) return 1.;
  Real sum = 0.;
  for (int n=1; n<=400; ++n) {
    const Real term = std::exp(-static_cast<Real>(n)*n*s);
    sum += (n % 2 == 1) ? term : -term;
    if (term < 1.e-17) break;
  }
  return std::min(std::max(2.*sum, 0.), 1.);
}

//----------------------------------------------------------------------------------------
//! \fn void MRWTables::BuildSphere()

void MRWTables::BuildSphere() {
  fp_lns_.resize(kNFirstPassage);
  fp_cdf_.resize(kNFirstPassage);
  const Real dls = std::log(kSMax/kSMin)/(kNFirstPassage-1);
  for (int i=0; i<kNFirstPassage; ++i) {
    fp_lns_[i] = std::log(kSMin) + i*dls;
    fp_cdf_[i] = 1. - SphereInside(std::exp(fp_lns_[i]));
  }
  for (int i=1; i<kNFirstPassage; ++i) fp_cdf_[i] = std::max(fp_cdf_[i], fp_cdf_[i-1]);

  nrad_s_ = kNRadS;
  nrad_q_ = kNRadQ;
  rad_lns_.resize(kNRadS);
  rad_q_.resize(static_cast<std::size_t>(kNRadS)*kNRadQ);
  const Real dlr = std::log(kSMax/kSRadMin)/(kNRadS-1);
  std::vector<Real> rho(kNRadGrid), pdf(kNRadGrid), cdf(kNRadGrid);
  for (int i=0; i<kNRadGrid; ++i) rho[i] = static_cast<Real>(i)/(kNRadGrid-1);
  for (int j=0; j<kNRadS; ++j) {
    rad_lns_[j] = std::log(kSRadMin) + j*dlr;
    const Real s = std::exp(rad_lns_[j]);
    for (int i=0; i<kNRadGrid; ++i) {
      Real sum = 0.;
      for (int n=1; n<=400; ++n) {
        const Real en = std::exp(-static_cast<Real>(n)*n*s);
        if (en < 1.e-17) break;
        sum += n*std::sin(n*PI*rho[i])*en;
      }
      pdf[i] = std::max(rho[i]*sum, 0.);
    }
    cdf[0] = 0.;
    for (int i=1; i<kNRadGrid; ++i)
      cdf[i] = cdf[i-1] + 0.5*(pdf[i]+pdf[i-1])*(rho[i]-rho[i-1]);
    for (int k=0; k<kNRadQ; ++k) {
      const Real target = cdf[kNRadGrid-1]*static_cast<Real>(k)/(kNRadQ-1);
      int i = static_cast<int>(std::lower_bound(cdf.begin(), cdf.end(), target) - cdf.begin());
      i = std::min(std::max(i, 1), kNRadGrid-1);
      const Real dc = cdf[i] - cdf[i-1];
      const Real t = (dc > 0.) ? (target - cdf[i-1])/dc : 0.;
      rad_q_[j*kNRadQ + k] = rho[i-1] + t*(rho[i]-rho[i-1]);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real MRWTables::SphereFirstPassage(Real u)

Real MRWTables::SphereFirstPassage(Real u) const {
  if (u <= fp_cdf_[0]) return kSMin;
  if (u >= fp_cdf_[kNFirstPassage-1]) return kSMax;
  int i = static_cast<int>(std::upper_bound(fp_cdf_.begin(), fp_cdf_.end(), u)
                           - fp_cdf_.begin());
  i = std::min(std::max(i, 1), kNFirstPassage-1);
  const Real dc = fp_cdf_[i] - fp_cdf_[i-1];
  const Real t = (dc > 0.) ? (u - fp_cdf_[i-1])/dc : 0.;
  return std::exp(fp_lns_[i-1] + t*(fp_lns_[i]-fp_lns_[i-1]));
}

//----------------------------------------------------------------------------------------
//! \fn Real MRWTables::SphereRadius(Real s, MCRandom *pran)

Real MRWTables::SphereRadius(Real s, MCRandom *pran) const {
  if (s < kSRadMin) {
    // free-space Gaussian: r^2 = 2 D c t chi^2_3, with 2 D c t = 2 s R_0^2 / pi^2
    Real chi2 = 0.;
    for (int m=0; m<2; ++m) {
      Real u1 = pran->uniform();
      while (u1 <= 0.) u1 = pran->uniform();
      const Real u2 = pran->uniform();
      const Real r = std::sqrt(-2.*std::log(u1));
      const Real n1 = r*std::cos(2.*PI*u2), n2 = r*std::sin(2.*PI*u2);
      chi2 += n1*n1 + ((m == 0) ? n2*n2 : 0.);
    }
    return std::min(std::sqrt(2.*s*chi2)/PI, 1.);
  }
  const Real ls = std::log(std::min(s, kSMax));
  const Real dlr = (rad_lns_[nrad_s_-1] - rad_lns_[0])/(nrad_s_-1);
  int j = static_cast<int>((ls - rad_lns_[0])/dlr);
  j = std::min(std::max(j, 0), nrad_s_-2);
  const Real a = std::min(std::max((ls - rad_lns_[j])/dlr, 0.), 1.);
  const Real u = pran->uniform()*(nrad_q_-1);
  int k = static_cast<int>(u);
  k = std::min(std::max(k, 0), nrad_q_-2);
  const Real b = u - k;
  const Real r0 = (1.-b)*rad_q_[j*nrad_q_+k] + b*rad_q_[j*nrad_q_+k+1];
  const Real r1 = (1.-b)*rad_q_[(j+1)*nrad_q_+k] + b*rad_q_[(j+1)*nrad_q_+k+1];
  return (1.-a)*r0 + a*r1;
}

//----------------------------------------------------------------------------------------
//! \fn void MRWTables::BuildBox()
//! \brief the one-dimensional tables on (xi, ln sigma): ln S, hazard, left-exit
//!        probability, and the in-box position quantiles

void MRWTables::BuildBox() {
  nxi_ = kNXi;
  nsig_ = kNSig;
  nlev_ = kNLev;
  xi_.resize(nxi_);
  for (int i=0; i<nxi_; ++i) xi_[i] = 0.5*(i+1)/nxi_;
  dxi_ = 0.5/nxi_;
  lnsig_.resize(nsig_);
  dlnsig_ = std::log(kSigMax/kSigMin)/(nsig_-1);
  for (int j=0; j<nsig_; ++j) lnsig_[j] = std::log(kSigMin) + j*dlnsig_;
  lns_.assign(static_cast<std::size_t>(nxi_)*nsig_, 0.);
  haz_.assign(lns_.size(), 0.);
  pleft_.assign(lns_.size(), 0.5);
  pos_.assign(lns_.size()*nlev_, 0.f);

  std::vector<Real> x(kNPosGrid), cdf(kNPosGrid);
  for (int m=0; m<kNPosGrid; ++m) x[m] = static_cast<Real>(m)/(kNPosGrid-1);

  for (int i=0; i<nxi_; ++i) {
    const Real xi = xi_[i];
    for (int j=0; j<nsig_; ++j) {
      const Real sigma = std::exp(lnsig_[j]);
      Real surv, dsurv, flux0, fluxw;
      BoxSeries(xi, sigma, surv, dsurv, flux0, fluxw);
      surv = std::max(surv, 1.e-300);
      lns_[i*nsig_+j] = std::log(surv);
      haz_[i*nsig_+j] = std::max(dsurv/surv, 0.);
      if (flux0 + fluxw > 1.e-280) {
        pleft_[i*nsig_+j] = std::min(std::max(flux0/(flux0 + fluxw), 0.), 1.);
      } else {
        // both fluxes underflow at small sigma: the image solution's ratio
        const Real e = std::exp(-PI*PI*((1.-xi)*(1.-xi) - xi*xi)/(4.*sigma));
        pleft_[i*nsig_+j] = 1./(1. + (1.-xi)/xi*e);
      }
      if (sigma < kSigPosMin) continue; // drawn from the image solution instead
      // conditional CDF of the position of a walker still inside
      const int nmax = std::min(20000, static_cast<int>(std::sqrt(46./sigma)) + 2);
      for (int m=0; m<kNPosGrid; ++m) {
        Real c = 0.;
        for (int n=1; n<=nmax; ++n) {
          c += 2./(n*PI)*std::sin(n*PI*xi)*(1. - std::cos(n*PI*x[m]))
               *std::exp(-static_cast<Real>(n)*n*sigma);
        }
        cdf[m] = c/surv;
      }
      cdf[0] = 0.;
      for (int m=1; m<kNPosGrid; ++m) cdf[m] = std::max(cdf[m], cdf[m-1]);
      const Real ctop = std::max(cdf[kNPosGrid-1], 1.e-300);
      for (int m=0; m<kNPosGrid; ++m) cdf[m] /= ctop;
      for (int k=0; k<nlev_; ++k) {
        const Real p = static_cast<Real>(k)/(nlev_-1);
        int m = static_cast<int>(std::lower_bound(cdf.begin(), cdf.end(), p) - cdf.begin());
        m = std::min(std::max(m, 1), kNPosGrid-1);
        const Real dc = cdf[m] - cdf[m-1];
        const Real t = (dc > 0.) ? (p - cdf[m-1])/dc : 0.;
        pos_[(static_cast<std::size_t>(i)*nsig_+j)*nlev_+k] =
            static_cast<float>(x[m-1] + t*(x[m]-x[m-1]));
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MRWTables::BoxWeights(...)
//! \brief bilinear weights on the (xi, ln sigma) grid; xi folded to [0, 1/2] already

void MRWTables::BoxWeights(Real xi, Real sigma, int &i, Real &a, int &j, Real &b) const {
  Real fi = (xi - xi_[0])/dxi_;
  fi = std::min(std::max(fi, 0.), static_cast<Real>(nxi_-1));
  i = std::min(static_cast<int>(fi), nxi_-2);
  a = fi - i;
  Real fj = (std::log(sigma) - lnsig_[0])/dlnsig_;
  fj = std::min(std::max(fj, 0.), static_cast<Real>(nsig_-1));
  j = std::min(static_cast<int>(fj), nsig_-2);
  b = fj - j;
}

//----------------------------------------------------------------------------------------
//! \fn Real MRWTables::BoxLnSurvival(Real xi, Real sigma)

Real MRWTables::BoxLnSurvival(Real xi, Real sigma) const {
  if (xi > 0.5) xi = 1. - xi;
  xi = std::max(xi, kEps);
  if (sigma <= 0.) return 0.;
  if (sigma < kSigMin) {
    const Real q = 0.5*PI/std::sqrt(sigma);
    const Real s = std::erf(q*xi) + std::erf(q*(1.-xi)) - 1.;
    return std::log(std::max(s, 1.e-300));
  }
  if (sigma > kSigMax) return BoxLnSurvival(xi, kSigMax) - (sigma - kSigMax);
  int i, j;
  Real a, b;
  BoxWeights(xi, sigma, i, a, j, b);
  return (1.-a)*(1.-b)*Lns(i,j) + a*(1.-b)*Lns(i+1,j) + (1.-a)*b*Lns(i,j+1)
         + a*b*Lns(i+1,j+1);
}

//----------------------------------------------------------------------------------------
//! \fn Real MRWTables::BoxHazard(Real xi, Real sigma)

Real MRWTables::BoxHazard(Real xi, Real sigma) const {
  if (xi > 0.5) xi = 1. - xi;
  xi = std::max(xi, kEps);
  if (sigma < kSigMin) {
    // d/dsigma of the image survival erf(a/sqrt(sigma)) + erf(b/sqrt(sigma)) - 1
    const Real a = 0.5*PI*xi, b = 0.5*PI*(1.-xi);
    const Real s32 = std::pow(sigma, -1.5);
    const Real ds = (a*std::exp(-a*a/sigma) + b*std::exp(-b*b/sigma))*s32/std::sqrt(PI);
    const Real q = 0.5*PI/std::sqrt(sigma);
    const Real s = std::max(std::erf(q*xi) + std::erf(q*(1.-xi)) - 1., 1.e-300);
    return ds/s;
  }
  if (sigma > kSigMax) return 1.;
  int i, j;
  Real a, b;
  BoxWeights(xi, sigma, i, a, j, b);
  return (1.-a)*(1.-b)*Haz(i,j) + a*(1.-b)*Haz(i+1,j) + (1.-a)*b*Haz(i,j+1)
         + a*b*Haz(i+1,j+1);
}

//----------------------------------------------------------------------------------------
//! \fn Real MRWTables::BoxLeftExit(Real xi, Real sigma)

Real MRWTables::BoxLeftExit(Real xi, Real sigma) const {
  bool folded = false;
  if (xi > 0.5) { xi = 1. - xi; folded = true; }
  xi = std::max(xi, kEps);
  Real p;
  if (sigma < kSigMin) {
    const Real e = std::exp(-PI*PI*((1.-xi)*(1.-xi) - xi*xi)/(4.*sigma));
    p = 1./(1. + (1.-xi)/xi*e);
  } else if (sigma > kSigMax) {
    p = 0.5;
  } else {
    int i, j;
    Real a, b;
    BoxWeights(xi, sigma, i, a, j, b);
    p = (1.-a)*(1.-b)*Pleft(i,j) + a*(1.-b)*Pleft(i+1,j) + (1.-a)*b*Pleft(i,j+1)
        + a*b*Pleft(i+1,j+1);
  }
  return folded ? 1. - p : p;
}

//----------------------------------------------------------------------------------------
//! \fn Real MRWTables::BoxPosition(Real xi, Real sigma, Real p)
//! \brief position / W of a walker still inside at sigma, at cumulative probability p.
//!        Below the tabulated sigma the two-image solution is sampled by thinning a
//!        Gaussian, the extra uniforms derived from p so that the caller supplies one.

Real MRWTables::BoxPosition(Real xi, Real sigma, Real p) const {
  bool folded = false;
  if (xi > 0.5) { xi = 1. - xi; folded = true; p = 1. - p; }
  xi = std::max(xi, kEps);
  Real x;
  if (sigma < kSigPosMin) {
    // density ~ G(x - xi) - G(x + xi) - G(x - 2 + xi), G a Gaussian of width
    // w = sqrt(2 sigma)/pi: the Gaussian inverse CDF at the uniform, accepted with the
    // image correction 1 - exp(-2 x xi/w^2) - exp(-2 (1-x)(1-xi)/w^2)
    const Real w = std::sqrt(2.*sigma)/PI;
    Real u = p;
    x = xi;
    for (int it=0; it<64; ++it) {
      x = xi + w*KompaneetsTable::InverseNormal(std::min(std::max(u, 1.e-12), 1.-1.e-12));
      Real acc = 0.;
      if (x > 0. && x < 1.)
        acc = 1. - std::exp(-2.*x*xi/(w*w)) - std::exp(-2.*(1.-x)*(1.-xi)/(w*w));
      Real v = u*7919.; v -= std::floor(v);
      if (v < acc) break;
      u = u*104729. + 0.5; u -= std::floor(u);
      x = xi;
    }
    x = std::min(std::max(x, kEps), 1.-kEps);
    return folded ? 1. - x : x;
  }
  int i, j;
  Real a, b;
  BoxWeights(xi, std::min(sigma, kSigMax), i, a, j, b);
  // the position table starts at kSigPosMin
  const int jmin = static_cast<int>(std::ceil((std::log(kSigPosMin) - lnsig_[0])/dlnsig_));
  if (j < jmin) { j = jmin; b = 0.; }
  const Real fk = p*(nlev_-1);
  int k = std::min(std::max(static_cast<int>(fk), 0), nlev_-2);
  const Real c = fk - k;
  auto P = [&](int ii, int jj) {
    return (1.-c)*Pos(ii,jj,k) + c*Pos(ii,jj,k+1);
  };
  x = (1.-a)*(1.-b)*P(i,j) + a*(1.-b)*P(i+1,j) + (1.-a)*b*P(i,j+1) + a*b*P(i+1,j+1);
  x = std::min(std::max(x, kEps), 1.-kEps);
  return folded ? 1. - x : x;
}

//----------------------------------------------------------------------------------------
//! \fn void PhotonPusher::CellGeometry(Photon *pphot, int ip, Real W[3], Real x[3])
//! \brief widths of the photon's cell and the photon's distances from its lower faces,
//!        in code length, in the pusher's local orthonormal basis (a spherical-polar
//!        cell is treated as the box dr x r dtheta x r sin(theta) dphi at the photon)

void PhotonPusher::CellGeometry(Photon *pphot, int ip, Real W[3], Real x[3]) {
  MCCoord *pco = pcoord;
  const int i1 = pphot->i1p[ip], i2 = pphot->i2p[ip], i3 = pphot->i3p[ip];
  if (pmy_mcb->topology == MCTOPO_SPHERICAL) {
    const Real r = pphot->x1p[ip], th = pphot->x2p[ip], ph = pphot->x3p[ip];
    const Real rs = r*std::sin(th);
    W[0] = pco->x1f(i1+1) - pco->x1f(i1);
    W[1] = r*(pco->x2f(i2+1) - pco->x2f(i2));
    W[2] = rs*(pco->x3f(i3+1) - pco->x3f(i3));
    x[0] = r - pco->x1f(i1);
    x[1] = r*(th - pco->x2f(i2));
    x[2] = rs*(ph - pco->x3f(i3));
  } else {
    W[0] = pco->x1f(i1+1) - pco->x1f(i1);
    W[1] = pco->x2f(i2+1) - pco->x2f(i2);
    W[2] = pco->x3f(i3+1) - pco->x3f(i3);
    x[0] = pphot->x1p[ip] - pco->x1f(i1);
    x[1] = pphot->x2p[ip] - pco->x2f(i2);
    x[2] = pphot->x3p[ip] - pco->x3f(i3);
  }
  for (int m=0; m<3; ++m) x[m] = std::min(std::max(x[m], 0.), W[m]);
}

//----------------------------------------------------------------------------------------
//! \fn Real PhotonPusher::FaceDistance(Photon *pphot, int ip)
//! \brief distance from the photon to the nearest face of its cell, in code length

Real PhotonPusher::FaceDistance(Photon *pphot, int ip) {
  Real W[3], x[3];
  CellGeometry(pphot, ip, W, x);
  Real d = HUGE_NUMBER;
  for (int m=0; m<3; ++m) d = std::min(d, std::min(x[m], W[m] - x[m]));
  return std::max(d, 0.);
}

//----------------------------------------------------------------------------------------
//! \fn bool PhotonPusher::MRWTrigger(Photon *pphot, int ip, Real chi)
//! \brief whether a step is to be tried: the cell unmasked and thick enough.  chi is the
//!        lab extinction in cm^-1.  cell domain: the cell's optical half-width above
//!        accel_tau and the photon more than accel_face_tau from every face; sphere:
//!        the face distance above accel_tau.

bool PhotonPusher::MRWTrigger(Photon *pphot, int ip, Real chi) {
  MonteCarloBlock *pmcb = pmy_mcb;
  if (pmcb->accel_mask.GetSize() > 0
      && pmcb->accel_mask(pphot->i3p[ip],pphot->i2p[ip],pphot->i1p[ip]) == 0) return false;
  Real W[3], x[3];
  CellGeometry(pphot, ip, W, x);
  Real dmin = HUGE_NUMBER, wmin = HUGE_NUMBER;
  for (int m=0; m<3; ++m) {
    dmin = std::min(dmin, std::min(x[m], W[m] - x[m]));
    wmin = std::min(wmin, W[m]);
  }
  const Real l = pmcb->l_cgs;
  if (pmy_mc->accel_domain == MRW_DOMAIN_SPHERE) return chi*dmin*l > pmy_mc->accel_tau;
  return (chi*0.5*wmin*l > pmy_mc->accel_tau) && (chi*dmin*l > pmy_mc->accel_face_tau);
}

//----------------------------------------------------------------------------------------
//! \fn bool PhotonPusher::MRWStep(Photon *pphot, MCRandom *pran, int ip)
//! \brief the step.  Returns false, with the photon untouched, when it declines; true
//!        when it was taken, after which the photon is at a new position with a new
//!        isotropic direction and the caller starts a new free flight (or handles a
//!        status the cell update may have set).

bool PhotonPusher::MRWStep(Photon *pphot, MCRandom *pran, int ip) {
  MonteCarloBlock *pmcb = pmy_mcb;
  const MRWTables *tab = mrw_;
  const int i1 = pphot->i1p[ip], i2 = pphot->i2p[ip], i3 = pphot->i3p[ip];

  if (boosts && GENERAL_RELATIVITY) { // the tetrad step is not built yet (plan, 6.4)
    pmcb->nmrw_decline++;
    return false;
  }

  const Real l_cgs = pmcb->l_cgs;
  const Real c_code = MCConstants::c_cgs / pmcb->vel_cgs;
  const bool box = (pmy_mc->accel_domain == MRW_DOMAIN_CELL);

  // A moving medium: the walk is done in the comoving frame, where the medium is at rest.
  // vel holds (gamma, gamma beta^i) in the pusher's orthonormal basis.
  Real beta[3] = {0., 0., 0.}, gam = 1., bmag = 0., nufact = 1.;
  const bool moving = boosts && pmcb->vel.GetSize() > 0;
  if (moving) {
    gam = pmcb->vel(i3,i2,i1,0);
    for (int m=0; m<3; ++m) beta[m] = pmcb->vel(i3,i2,i1,m+1)/gam;
    bmag = std::sqrt(SQR(beta[0]) + SQR(beta[1]) + SQR(beta[2]));
    if (bmag > 0.) {
      const Real e_lab = pphot->ep[ip];
      pmcb->TransformToComoving(pphot, ip, ip);
      nufact = pphot->ep[ip]/e_lab;
    }
  }
  const bool transformed = moving && bmag > 0.;
  auto decline = [&]() {
    if (transformed) pmcb->TransformToCoordinate(pphot, ip, ip);
    pmcb->nmrw_decline++;
    return false;
  };

  const Real chi = pphot->scp[ip] + pphot->acp[ip]; // cm^-1, comoving
  if (chi <= 0.) return decline();
  const Real D = 1./(3.*chi);

  // the cell around the photon, in code length
  Real W[3], x[3];
  CellGeometry(pphot, ip, W, x);
  Real dmin = HUGE_NUMBER;
  for (int m=0; m<3; ++m) dmin = std::min(dmin, std::min(x[m], W[m] - x[m]));
  const Real dmin_cm = dmin*l_cgs;

  // Advection: the fluid element moves beta gamma c t_c in the lab over comoving time t_c,
  // and the domain plus that displacement has to stay inside the cell.  A fraction f of
  // the nearest-face distance goes to advection, set so that the expected diffusion time
  // across that distance, d^2 chi / 2, carries the centre just that far; small for a
  // slow flow, and the step declines when advection would cross the cell before the
  // walk could leave even a small domain.
  Real f = 0.;
  if (bmag > 0.) {
    const Real a = 0.5*bmag*gam*chi*dmin_cm;
    f = std::min(((2.*a + 1.) - std::sqrt(4.*a + 1.))/(2.*a), 0.5);
  }
  const Real adv_room = f*dmin; // code length, along beta

  // domain geometry and the thickness tests with the comoving chi
  Real R0 = 0.;              // sphere radius, cm
  Real Wp[3], xp[3];         // box after the downstream shrink, code length
  if (box) {
    for (int m=0; m<3; ++m) {
      const Real am = (bmag > 0.) ? adv_room*std::abs(beta[m])/bmag : 0.;
      Wp[m] = W[m] - am;
      xp[m] = (beta[m] < 0.) ? x[m] - am : x[m];
      if (xp[m] <= 0. || xp[m] >= Wp[m] || Wp[m] <= 0.) return decline();
    }
    Real dpmin = HUGE_NUMBER, wpmin = HUGE_NUMBER;
    for (int m=0; m<3; ++m) {
      dpmin = std::min(dpmin, std::min(xp[m], Wp[m] - xp[m]));
      wpmin = std::min(wpmin, Wp[m]);
    }
    if (chi*0.5*wpmin*l_cgs <= pmy_mc->accel_tau
        || chi*dpmin*l_cgs <= pmy_mc->accel_face_tau) return decline();
  } else {
    R0 = (1. - f)*dmin_cm*(1. - kEps);
    if (chi*R0 <= pmy_mc->accel_tau) return decline();
  }

  // Compton: temperature, energy and absorption parameter of the cell, and the guards
  // of the Kompaneets description.  The Thomson coefficient that counts y is scaled to
  // the comoving frame the way the opacities were.
  Real theta = 0., xi = 0., lam = 0., kT = 0., nsig = 0.;
  if (compton) {
    const Real T = pmcb->tgas(i3,i2,i1);
    kT = MCConstants::kb_cgs*T;
    theta = MCConstants::kmec2*T;
    xi = pphot->ep[ip]/kT;
    nsig = pmcb->species(0,i3,i2,i1)*MCConstants::sigmat/nufact;
    const KompaneetsTable *kg = pmy_mc->kgreens;
    if (pmcb->absorption_opac == ABSFF) {
      const Real nu1 = kT/MCConstants::h_cgs;
      const Real alpha1 = pmcb->ff_cell(0,i3,i2,i1)*(1. - std::exp(-1.))/(nu1*nu1*nu1)
                          /nufact;
      lam = alpha1/(nsig*theta);
    }
    if (theta > 0.03 || xi*theta > 0.05 || xi < kg->XiMin() || xi > kg->XiMax()
        || theta*lam*KompaneetsTable::FreeFreeShape(xi) > 0.1 || nsig <= 0.)
      return decline();
  }

  // comoving path budget c t_b in cm: the advection room, the remaining time of the
  // cycle (lab time, so divided by gamma), and accel_pmax
  Real ct_b = HUGE_NUMBER;
  if (bmag > 0.) ct_b = adv_room*l_cgs/(bmag*gam);
  if (pphot->dtp[ip] < 0.5*HUGE_NUMBER)
    ct_b = std::min(ct_b, pphot->dtp[ip]*c_code*l_cgs/gam);
  if (ct_b <= 0.) return decline();

  // the draw: path c t (cm) and the comoving displacement dpos (code length)
  Real ct, dpos[3];
  if (box) {
    Real rate[3], xif[3];
    for (int m=0; m<3; ++m) {
      rate[m] = PI*PI*D/SQR(Wp[m]*l_cgs); // sigma per cm of path
      xif[m] = xp[m]/Wp[m];
    }
    auto lnsurv = [&](Real c) {
      return tab->BoxLnSurvival(xif[0], rate[0]*c) + tab->BoxLnSurvival(xif[1], rate[1]*c)
             + tab->BoxLnSurvival(xif[2], rate[2]*c);
    };
    // the path at which the box survival has fallen to exp(target), by bisection in ln c
    auto invert = [&](Real target, Real c_hi) {
      Real lo = 1.e-6/std::max(std::max(rate[0], rate[1]), rate[2]);
      Real hi = c_hi;
      if (c_hi >= 0.5*HUGE_NUMBER) {
        hi = lo;
        while (lnsurv(hi) > target && hi < 1.e30) hi *= 2.;
      }
      Real llo = std::log(lo), lhi = std::log(hi);
      for (int it=0; it<60; ++it) {
        const Real mid = 0.5*(llo + lhi);
        if (lnsurv(std::exp(mid)) > target) llo = mid; else lhi = mid;
      }
      return std::exp(0.5*(llo + lhi));
    };
    if (pmy_mc->accel_pmax > 0.)
      ct_b = std::min(ct_b, invert(std::log(pmy_mc->accel_pmax), ct_b));
    const Real lns_b = (ct_b < 0.5*HUGE_NUMBER) ? lnsurv(ct_b) : -HUGE_NUMBER;
    const Real u = pran->uniform();
    const Real lnu = std::log(std::max(1. - u, 1.e-300));
    Real newx[3];
    if (lnu > lns_b) {
      // out through a wall before the budget: the time, the dimension, the side
      ct = invert(lnu, ct_b);
      Real h[3], hsum = 0.;
      for (int m=0; m<3; ++m) {
        h[m] = rate[m]*tab->BoxHazard(xif[m], rate[m]*ct);
        hsum += h[m];
      }
      Real v = pran->uniform()*hsum;
      int mexit = 0;
      while (mexit < 2 && v >= h[mexit]) { v -= h[mexit]; ++mexit; }
      const bool left = pran->uniform() < tab->BoxLeftExit(xif[mexit], rate[mexit]*ct);
      for (int m=0; m<3; ++m) {
        if (m == mexit) {
          newx[m] = left ? kEps*Wp[m] : (1. - kEps)*Wp[m];
        } else {
          newx[m] = Wp[m]*tab->BoxPosition(xif[m], rate[m]*ct, pran->uniform());
        }
      }
    } else {
      ct = ct_b;
      for (int m=0; m<3; ++m)
        newx[m] = Wp[m]*tab->BoxPosition(xif[m], rate[m]*ct, pran->uniform());
    }
    for (int m=0; m<3; ++m) dpos[m] = newx[m] - xp[m];
  } else {
    const Real s_per_ct = PI*PI*D/(R0*R0);
    Real s_b = (ct_b < 0.5*HUGE_NUMBER) ? s_per_ct*ct_b : HUGE_NUMBER;
    if (pmy_mc->accel_pmax > 0.) s_b = std::min(s_b, mrw_s_pmax_);
    const Real f_b = 1. - MRWTables::SphereInside(s_b);
    const Real v = pran->uniform();
    Real s_e, rfac;
    if (v < f_b) {
      s_e = std::min(tab->SphereFirstPassage(v), s_b);
      rfac = 1.;
    } else {
      s_e = s_b;
      rfac = tab->SphereRadius(s_b, pran);
    }
    ct = s_e/s_per_ct;
    const Real mu = 2.*pran->uniform() - 1.;
    const Real sth = std::sqrt(1. - mu*mu);
    const Real phi = 2.*PI*pran->uniform();
    const Real rd = rfac*R0/l_cgs;
    dpos[0] = sth*std::cos(phi)*rd;
    dpos[1] = sth*std::sin(phi)*rd;
    dpos[2] = mu*rd;
  }

  // energy and survival
  const Real e_old = pphot->ep[ip];
  Real surv = 1.;
  if (compton) {
    const Real y = theta*nsig*ct;
    const Real xf = pmy_mc->kgreens->Quantile(xi, y, lam, pran->uniform());
    surv = pmy_mc->kgreens->Survival(xi, y, lam);
    if (pmcb->absorption_opac != ABSFF && pphot->acp[ip] > 0.)
      surv *= std::exp(-pphot->acp[ip]*ct); // grey absorption that is not free-free
    pphot->ep[ip] = xf*kT;
    pmcb->nmrw_scat += nsig*ct;
  } else {
    if (pphot->acp[ip] > 0.) surv = std::exp(-pphot->acp[ip]*ct);
    pmcb->nmrw_scat += pphot->scp[ip]*ct;
  }
  if (pmcb->call_moments) pmcb->UpdateMomentsMRW(pphot, ct, 0.5*(e_old + pphot->ep[ip]), ip);
  pphot->wp[ip] *= surv;

  // lab displacement: the comoving one plus the advection over the lab time gamma t_c,
  // in the pusher's local orthonormal basis
  const Real adv = (bmag > 0.) ? gam*ct/l_cgs : 0.;
  const Real dx = dpos[0] + beta[0]*adv, dy = dpos[1] + beta[1]*adv, dz = dpos[2] + beta[2]*adv;
  if (pmcb->topology == MCTOPO_SPHERICAL) {
    const Real r = pphot->x1p[ip];
    const Real cth = std::cos(pphot->x2p[ip]), snt = std::sin(pphot->x2p[ip]);
    const Real cph = std::cos(pphot->x3p[ip]), sph = std::sin(pphot->x3p[ip]);
    const Real ex = dx*snt*cph + dy*cth*cph - dz*sph;
    const Real ey = dx*snt*sph + dy*cth*sph + dz*cph;
    const Real ez = dx*cth - dy*snt;
    const Real xx = r*snt*cph + ex, yy = r*snt*sph + ey, zz = r*cth + ez;
    pphot->x1p[ip] = std::sqrt(xx*xx + yy*yy + zz*zz);
    pphot->x2p[ip] = std::acos(std::min(std::max(zz/pphot->x1p[ip], -1.), 1.));
    pphot->x3p[ip] = std::atan2(yy, xx);
    if (pphot->x3p[ip] < 0.) pphot->x3p[ip] += 2.*PI;
  } else {
    pphot->x1p[ip] += dx;
    pphot->x2p[ip] += dy;
    pphot->x3p[ip] += dz;
  }

  // new direction, isotropic in the comoving frame; assumed unpolarized
  const Real mu = 2.*pran->uniform() - 1.;
  const Real sth = std::sqrt(1. - mu*mu);
  const Real phi = 2.*PI*pran->uniform();
  pphot->k1p[ip] = sth*std::cos(phi);
  pphot->k2p[ip] = sth*std::sin(phi);
  pphot->k3p[ip] = mu;
  if (IsPolarized(pmy_mc->polarized)) {
    pphot->sqp[ip] = 0.;
    pphot->sup[ip] = 0.;
    pphot->svp[ip] = 0.;
  }

  // time (lab), path and the scatterings the path implies
  dl = gam*ct/l_cgs;
  pphot->x0p[ip] += dl;
  pphot->dtp[ip] -= dl/c_code;
  pphot->nscp[ip] += static_cast<int>(std::lround(pphot->scp[ip]*ct));
  pmcb->nmrw++;

  // the cell, which the domain should not have left. The boundary functions run if the
  // rounding put it across a block face.  Opacities at the new energy, in the comoving
  // frame like the block loop after a scattering, and everything else in lab frame
  const bool newzone = UpdateZone(pphot, ip);
  if (pphot->statp[ip] != EVOLVING) return true;
  if (newzone || compton) {
    pphot->acp[ip] = pmcb->AbsorptionOpacity(pmcb, pphot, ip);
    pphot->scp[ip] = pmcb->ScatteringOpacity(pmcb, pphot, ip);
  }
  if (transformed) pmcb->TransformToCoordinate(pphot, ip, ip);
  return true;
}
