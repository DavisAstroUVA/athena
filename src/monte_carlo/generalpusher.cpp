//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file grpusher.cpp
//! \brief implementation for moving photons via integration with metric and connection

// Athena++ headers
#include <complex>

#include "montecarlo.hpp"
#include "photon.hpp"
#include "photonpusher.hpp"
#include "../mesh/mesh.hpp"

// SWD: remove all of these
#define NCOORD 4

// GR headers
#define tolerance 1.e-5
#define max_iteration 2

//----------------------------------------------------------------------------------------
//! GeneralPusher class constructor, derived from PhotonPusher base class

GeneralPusher::GeneralPusher(MonteCarloBlock *pmcb)
  : PhotonPusher(pmcb) {

  step_par = pmy_mcb->stepsize;
  acon_valid = false;
  // A coordinate system whose Connect leaves components untouched would otherwise have
  // them read as whatever was on the stack, which is silent and looks like a physics bug.
  for (int i = 0; i < NCOORD; i++)
    for (int j = 0; j < NCOORD; j++)
      for (int k = 0; k < NCOORD; k++) gamma[i][j][k] = 0.0;

}

//----------------------------------------------------------------------------------------
//! destructor

GeneralPusher::~GeneralPusher() {

}

//----------------------------------------------------------------------------------------
//! \fn void GeneralPusher::Move(Photon *pphot, int ips, int ipe)
//! \brief Moves photon using geodesic integration

void GeneralPusher::Move(Photon *pphot, int ips, int ipe) {

  MonteCarloBlock *pmcb = pmy_mcb;
  MCRandom *pran = pmy_mcb->pran;
  PhotonTrajectoryList *ptraj = pmy_mcb->ptraj;
  Real l_cgs = pmcb->l_cgs;
  Real c_cgs = 2.99792458e10;
  Real c_code = c_cgs / pmcb->vel_cgs;

    // Check if using continuous absorption (all photon use same method)
  bool abs_tau = (pmy_mc->absorption_method[pphot->type[ips]] == ABSTAU);

  for (int ip=ips; ip<=ipe; ip++) {
    // get number of mean free paths photon will travel
    Real tauremaining = GetOpticalDepth(pran);
    if (!IsOnBlock(pphot,ip)) continue;
    Real step = StepSize(pphot,ip);
    //printf("step %g\n",step);
    Real path_length;
    Real k1, k2, k3;
    // Steps already taken in this free flight, plus the ones taken here.  Counted in a
    // local so it stays in a register across a body that calls out to AdvanceStep,
    // UpdateMoments and UserWorkInMove.
    const int nmv0 = pphot->nmvp[ip];
    int iter = 0;

    // set total extinction coefficient
    Real chi = abs_tau ? pphot->scp[ip] : (pphot->scp[ip] + pphot->acp[ip]);
    // nothing cached carries over from the previous photon
    acon_valid = false;
    // Cell the opacities were last computed for.  Starts invalid so that the first step
    // always refreshes; see the shift_unity comment below.
    int oi1 = -1, oi2 = -1, oi3 = -1;
    while ( (pphot->statp[ip] == EVOLVING) && (tauremaining > TINY_NUMBER) &&
            (pphot->dtp[ip] > 0.) ) {
      // Tested before the step is counted, so nmvp only ever counts steps actually taken.
      if (capmove > 0 && nmv0 + iter >= capmove) {
        pphot->statp[ip] = REMOVED;
        break;
      }
      iter++;

      // Update opacities at the beginning of each step.  Two regimes:
      //
      //   shift_unity false -- the comoving shift moves as the photon does, through the
      //     lapse and through the Doppler term, so the opacity has to be recomputed every
      //     step.  Any curved metric, or any moving fluid, lands here.
      //   shift_unity true  -- flat metric and fluid at rest, so the shift is identically
      //     one and the opacity depends on the cell alone.  Refreshing inside a cell is
      //     then provably a no-op; it measured exactly 0 over 1.2e7 same-cell steps in
      //     mc_snake_atm before this shortcut existed.  Only a cell change needs one.
      //
      // The first step of a Move always refreshes either way, because oi1 starts invalid.
      // Whatever left the opacities behind -- emission, or a coherent scatter that turned
      // the photon without recomputing them -- they cannot be assumed current on entry.
      if (!pmcb->shift_unity) {
        UpdateOpacities(pphot,pmcb,ip);
        chi = abs_tau ? pphot->scp[ip] : (pphot->scp[ip] + pphot->acp[ip]);
      } else if (pphot->i1p[ip] != oi1 || pphot->i2p[ip] != oi2
                 || pphot->i3p[ip] != oi3) {
        UpdateOpacities(pphot,pmcb,ip);
        chi = abs_tau ? pphot->scp[ip] : (pphot->scp[ip] + pphot->acp[ip]);
        oi1 = pphot->i1p[ip]; oi2 = pphot->i2p[ip]; oi3 = pphot->i3p[ip];
      }
      bool accel_success = false;
      if ((acceleration) && (resonance)) {
        // Get distance from photon to closest cell face
        Real dl;
        Real dw3, dw2, dw1;
        Real dx3f = fabs(pcoord->x3f(pphot->i3p[ip]) - pcoord->x3f(pphot->i3p[ip] + 1));
        Real dx2f = fabs(pcoord->x2f(pphot->i2p[ip]) - pcoord->x2f(pphot->i2p[ip] + 1));
        Real dx1f = fabs(pcoord->x1f(pphot->i1p[ip]) - pcoord->x1f(pphot->i1p[ip] + 1));
        Real x1v = (pcoord->x1f(pphot->i1p[ip]) + pcoord->x1f(pphot->i1p[ip] + 1))/2.;
        Real x2v = (pcoord->x2f(pphot->i2p[ip]) + pcoord->x2f(pphot->i2p[ip] + 1))/2.;
        dw3 = dx3f * x1v * sin(x2v);
        dw2 = dx2f * x1v;
        Real dmin0 = std::min(dx1f, dw2);
        dl = std::min(dmin0, dw3); // Distance to nearest face

        Real tauacc = 1000.; //BCM: make this an input parameter
        // Try to perform MRW acceleration if optical depth is large enough
        if (dl*l_cgs*chi > tauacc) {
          MRWResonanceAcceleration(pphot,pran,dl,tauacc,path_length,k1,k2,k3,ip);
          accel_success = true;
          // the photon has been relocated, so the cached connection no longer applies
          acon_valid = false;
        }
      }
      if (!accel_success) {// Acceleration not triggered - take standard step
        Real tau_step = chi * l_cgs * step * pphot->ep[ip];
        if (tauremaining > tau_step) { // Photon hasn't yet reached tauremaining
          // advance photon position, momentum, and polarization
          AdvanceStep(pphot,step,ip);
          tauremaining -= tau_step; // uses ep, chi at step start
        } else { // Photon has reached end of tauremaining - step to make it 0
          step = tauremaining / (chi * l_cgs * pphot->ep[ip]);
          AdvanceStep(pphot,step,ip);
          tauremaining = 0.;
        }
        pphot->dtp[ip] -= pphot->ep[ip] * step / c_code;
        // Update moments
        if (pmcb->call_moments) {
          Real dl_cgs = step * pphot->ep[ip] * l_cgs;
          if (abs_tau) {
            Real etaua = std::exp(-pphot->acp[ip] * dl_cgs);
            pmcb->UpdateMoments(pphot,dl_cgs,etaua,ip);
            pphot->wp[ip] *= etaua;
          } else {
            pmcb->UpdateMoments(pphot,dl_cgs,ip);
          }
        }
      } else {
        // Photon has been given a new position on sphere of radius dl
        // Set exit parameters and continue the loop over photons
        step = dl;
        tauremaining = 0.;
        if (pmcb->call_moments) {
          // SWD needs to use correct dl here
          pmcb->UpdateMomentsAcceleration(pphot,step,path_length,k1,k2,k3,1.,ip);
        }
      }

      // Check if photon changed cells.  The opacities are no longer refreshed here --
      // the top of the loop does it every step, which subsumes this case -- but the
      // cached connection still has to be dropped, because crossing a cell can remap the
      // position (e.g. a periodic boundary).
      if (UpdateZone(pphot,ip)) {
        acon_valid = false;
      }

      if (pphot->IsNanPhoton(ip)) {
        pphot->statp[ip] = DESTROYED;
        pphot->PrintPhoton("Photon returned Nan in general pusher",ip);
        break;
      }
      step = StepSize(pphot,ip);

      // Perform any user work
      if (UserWorkInMove != NULL) UserWorkInMove(pmcb,pphot,this,ip);
      // SWD: put here for now, may need additional flag
      if (ptraj != NULL) ptraj->AddToTrajectory(pphot,ip);

    } // end of photon integration
    //printf("it: %d\n",iter);
    /*if (pphot->statp[ip] == ESCAPED) {
      pphot->ep[ip] *= pphot->k0p[ip];
      //pphot->PrintPhoton(ip);
      }*/

    // Retire a photon whose free flight has run past the cap.  nmvp carries across Move
    // calls and blocks, which is what it takes to bound a flight; it is reset at each
    // scattering, in TransferPhotonsOnBlock.  Here the unit is an integration substep
    // rather than a cell crossing, so a meaningful cap is larger than for the two
    // coordinate pushers.
    //
    // This repeats the loop's test because the loop cannot make it on the step that
    // matters: a photon leaving the block ends that step BUFFERED, so the loop exits on
    // its own condition and never sees the final count.  Without this the photon goes
    // on to the next block, possibly through an MPI message, only to be retired on its
    // first step there.  Any other status is already terminal, REMOVED included, so this
    // cannot fire twice.
    pphot->nmvp[ip] = nmv0 + iter;
    if (capmove > 0 && pphot->nmvp[ip] >= capmove &&
        (pphot->statp[ip] == EVOLVING || pphot->statp[ip] == BUFFERED))
      pphot->statp[ip] = REMOVED;
  } // end loop over ip
}

//----------------------------------------------------------------------------------------
//! \fn void GeneralPusher::UpdateOpacities(Photon *pphot, MonteCarloBlock *pmcb, int ip)
//! \brief update opacities after a photon has changed cells

void GeneralPusher::UpdateOpacities(Photon *pphot, MonteCarloBlock *pmcb, int ip) {

  pmy_mcb = pmcb;

    if (pphot->statp[ip] == EVOLVING) {
    // Opacities need to be calculated using comoving frame energy and then transformed
    // back to Eulerian frame when Lorentz Transformations are enabled.
    Real shift;
    int i1 = pphot->i1p[ip];
    int i2 = pphot->i2p[ip];
    int i3 = pphot->i3p[ip];
    if (pmcb->boosts || pmcb->tetrads) {
      // Shift photon energy to comoving frame
      shift = pmy_mcb->FrequencyShiftComoving(pphot,ip);
      pphot->ep[ip] *= shift;
      // compute opacities in comoving frame
      pphot->acp[ip] = pmcb->AbsorptionOpacity(pmcb,pphot,ip);
      pphot->scp[ip] = pmcb->ScatteringOpacity(pmcb,pphot,ip);
      // Shift energy back to Eulerian frame
      pphot->ep[ip] /= shift;
      // Shift opaciteis to Eulerian frame
      pphot->acp[ip] *= shift;
      pphot->scp[ip] *= shift;
    } else {
      // No distinction between comovinng frame and eulerian frame
      pphot->acp[ip] = pmcb->AbsorptionOpacity(pmcb,pphot,ip);
      pphot->scp[ip] = pmcb->ScatteringOpacity(pmcb,pphot,ip);
    }

  }
}

//----------------------------------------------------------------------------------------
//! \fn void GeneralPusher::VerletStep(Photon *pphot, Real step, int ip)
//! \brief performs a single verlet integration step

void GeneralPusher::VerletStep(Photon *pphot, Real step, int ip) {

  Real k_n1[NCOORD],k_n1_copy[NCOORD];
  Real dk_n1[NCOORD];
  Real error;
  Real x[NCOORD], k0[NCOORD], dk0[NCOORD];

  // SWD: Need to think about how to do this better without invoking recurssion
  // SWD: Need to rename variables and clean this up with new scheme

  x[IMC0] = pphot->x0p[ip] += (pphot->k0p[ip])*step + 0.5*(pphot->dk0p[ip])*SQR(step);
  x[IMC1] = pphot->x1p[ip] += (pphot->k1p[ip])*step + 0.5*(pphot->dk1p[ip])*SQR(step);
  x[IMC2] = pphot->x2p[ip] += (pphot->k2p[ip])*step + 0.5*(pphot->dk2p[ip])*SQR(step);
  x[IMC3] = pphot->x3p[ip] += (pphot->k3p[ip])*step + 0.5*(pphot->dk3p[ip])*SQR(step);

  k_n1[IMC0] = (pphot->k0p[ip]) + 0.5*(pphot->dk0p[ip]) * step;
  k_n1[IMC1] = (pphot->k1p[ip]) + 0.5*(pphot->dk1p[ip]) * step;
  k_n1[IMC2] = (pphot->k2p[ip]) + 0.5*(pphot->dk2p[ip]) * step;
  k_n1[IMC3] = (pphot->k3p[ip]) + 0.5*(pphot->dk3p[ip]) * step;

  k0[IMC0] = pphot->k0p[ip];
  k0[IMC1] = pphot->k1p[ip];
  k0[IMC2] = pphot->k2p[ip];
  k0[IMC3] = pphot->k3p[ip];

  dk0[IMC0] = pphot->dk0p[ip];
  dk0[IMC1] = pphot->dk1p[ip];
  dk0[IMC2] = pphot->dk2p[ip];
  dk0[IMC3] = pphot->dk3p[ip];


  // Update gamma for current location
  pcoord->Connect(x, gamma);
  int n_iteration = 0;

  // SWD: not clear this while loops is accomplishing anything
  do {
    n_iteration += 1;
    error = 0.;
    for (int i=0;i<NCOORD;i++) {
      k_n1_copy[i] = k_n1[i];
    }

    for (int k=0;k<NCOORD;k++) {
      // off diagonal elements
      dk_n1[k] =
        -2. * (k_n1_copy[IMC0] *
               (gamma[k][IMC0][IMC1] * k_n1_copy[IMC1] +
                gamma[k][IMC0][IMC2] * k_n1_copy[IMC2] +
                gamma[k][IMC0][IMC3] * k_n1_copy[IMC3])
               +
               k_n1_copy[IMC1] * (gamma[k][IMC1][IMC2] * k_n1_copy[IMC2] +
                                  gamma[k][IMC1][IMC3] * k_n1_copy[IMC3]) +
               k_n1_copy[IMC2] * gamma[k][IMC2][IMC3] * k_n1_copy[IMC3]);
      // diagonal elements
      dk_n1[k] -=
        (gamma[k][IMC0][IMC0] * k_n1_copy[IMC0] * k_n1_copy[IMC0] +
         gamma[k][IMC1][IMC1] * k_n1_copy[IMC1] * k_n1_copy[IMC1] +
         gamma[k][IMC2][IMC2] * k_n1_copy[IMC2] * k_n1_copy[IMC2] +
         gamma[k][IMC3][IMC3] * k_n1_copy[IMC3] * k_n1_copy[IMC3]);

      k_n1[k] = k0[k] + 0.5 * (dk0[k] + dk_n1[k]) * step;

      error += fabs(k_n1_copy[k] - k_n1[k]) / (k_n1[k]);
    }
  } while ((error > tolerance) && (n_iteration < max_iteration));

  // SWD probably should not do this here
  // update photon energy due to evolving k_t (coordinate frame)
  //pphot->ep[ip] *= k_n1[IMC0]/(pphot->k[IMC0]);

  pphot->k0p[ip] = k_n1[IMC0];
  pphot->k1p[ip] = k_n1[IMC1];
  pphot->k2p[ip] = k_n1[IMC2];
  pphot->k3p[ip] = k_n1[IMC3];

  pphot->dk0p[ip] = dk_n1[IMC0];
  pphot->dk1p[ip] = dk_n1[IMC1];
  pphot->dk2p[ip] = dk_n1[IMC2];
  pphot->dk3p[ip] = dk_n1[IMC3];

}

//----------------------------------------------------------------------------------------
//! \fn void GeneralPusher::RK4Step(Photon *pphot, Real step, int ip)
//! \brief performs a single RK4 integration step of Hamiltonian formalism

void GeneralPusher::RK4Step(Photon *pphot, Real step, int ip) {

  Real x0[4];
  x0[IMC0] = pphot->x0p[ip];
  x0[IMC1] = pphot->x1p[ip];
  x0[IMC2] = pphot->x2p[ip];
  x0[IMC3] = pphot->x3p[ip];

  Real kcon[4];
  kcon[IMC0] = pphot->k0p[ip];
  kcon[IMC1] = pphot->k1p[ip];
  kcon[IMC2] = pphot->k2p[ip];
  kcon[IMC3] = pphot->k3p[ip];

  Real k0[4], gcov[4][4];
  pcoord->Metric(x0, gcov);
  for (int j = 0; j < 4; j++) {
    k0[j] = 0.;
    for (int i = 0; i < 4; i++)
      k0[j] += gcov[j][i]*kcon[i];
  }
  Real dl[8], x[4], k[4], xs[4], ks[4];
  // first stubstep
  SubStep(x0,k0,dl);
  for (int i = 0; i < 4; i++) {
    x[i] = x0[i] + step / 6. * dl[i];
    k[i] = k0[i] + step / 6. * dl[i+4];
  }
  // second substep
  for (int i = 0; i < 4; i++) {
    xs[i] = x0[i] + 0.5 * step * dl[i];
    ks[i] = k0[i] + 0.5 * step * dl[i+4];
  }
  SubStep(xs,ks,dl);
  for (int i = 0; i < 4; i++) {
    x[i] += step / 3. * dl[i];
    k[i] += step / 3. * dl[i+4];
  }
  // third substep
  for (int i = 0; i < 4; i++) {
    xs[i] = x0[i] + 0.5 * step * dl[i];
    ks[i] = k0[i] + 0.5 * step * dl[i+4];
  }
  SubStep(xs,ks,dl);
  for (int i = 0; i < 4; i++) {
    x[i] += step / 3. * dl[i];
    k[i] += step / 3. * dl[i+4];
  }
  // fourth substep
  for (int i = 0; i < 4; i++) {
    xs[i] = x0[i] + step * dl[i];
    ks[i] = k0[i] + step * dl[i+4];
  }
  SubStep(xs,ks,dl);
  for (int i = 0; i < 4; i++) {
    x[i] += step / 6. * dl[i];
    k[i] += step / 6. * dl[i+4];
  }

  Real gcon[4][4];
  pcoord->InverseMetric(x, gcon);
  for (int j = 0; j < 4; j++) {
    kcon[j] = 0.;
    for (int i = 0; i < 4; i++) {
      kcon[j] += gcon[j][i]*k[i];
    }
  }
  // Renormalize space components to keep k on shell
  pcoord->Metric(x, gcov);
  Real a = 0.;
  for (int j = 1; j < 4; j++)
    for (int i = 1; i < 4; i++)
      a += gcov[j][i] * kcon[j] * kcon[i];
  Real b = 0.;
  for (int i = 1; i < 4; i++)
    b += 2. * gcov[0][i] * kcon[0] * kcon[i];
  Real c = gcov[0][0] * kcon[0] * kcon[0];
  Real d = std::sqrt(SQR(b) - 4.*a*c);
  Real factor = b < 0. ? (d-b) / (2.*a) : -2.*c / (b+d);

  // Update photon parameters
  pphot->x0p[ip] = x[IMC0];
  pphot->x1p[ip] = x[IMC1];
  pphot->x2p[ip] = x[IMC2];
  pphot->x3p[ip] = x[IMC3];
  // k0p is the photon energy
  pphot->k0p[ip] = kcon[IMC0];
  pphot->k1p[ip] = kcon[IMC1] * factor;
  pphot->k2p[ip] = kcon[IMC2] * factor;
  pphot->k3p[ip] = kcon[IMC3] * factor;
}

//----------------------------------------------------------------------------------------
//! \fn void GeneralPusher::SubStep(Real xcon[4], Real kcov[4], Real dl[9])
//! \brief performs a single verlet integration step

void GeneralPusher::SubStep(Real xcon[4], Real kcov[4], Real dl[8]) {

  for (int i = 0; i < 8; i++)
    dl[i] = 0.0;

  Real gcon[4][4];
  pcoord->InverseMetric(xcon, gcon);
  for (int j = 0; j < 4; j++)
    for (int i = 0; i < 4; i++)
      dl[j] += gcon[j][i] * kcov[i];

  Real dgcon[4][4][4];
  pcoord->InverseMetricDerivative(xcon, dgcon);
  for (int k = 0; k < 4; k++)
    for (int j = 0; j < 4; j++)
      for (int i = 0; i < 4; i++)
        dl[k+4] -= 0.5 * dgcon[k][j][i] * kcov[j] * kcov[i];
  // proper distance
  //Real gcov[4][4];
  //pcoord->Metric(xcon, gcov);
  /*Real ku[4] = {};
  for (int j = 1; j < 4; j++)
    for (int i = 0; i < 4; i++)
      ku[j] += (gcon[j][i] - gcon[0][j] * gcon[0][i] / gcon[0][0]) * kcov[i];
  for (int j = 1; j < 4; j++)
    for (int i = 1; i < 4; i++)
      dl[8] += gcov[j][i] * ku[j] * ku[i];
      dl[8] = -std::sqrt(dl[8]);*/

}

//----------------------------------------------------------------------------------------
//! \fn void GeneralPusher::ConnectionContraction(Photon *pphot, int ip, Real acon[4][4])
//! \brief A^i_k = Gamma^i_kl k^l at the photon's current position and wavevector
//
// Contract k with connection. This is the part of the polarization transport that depends
// on where the photon is rather so AdvanceStep saves for next step

void GeneralPusher::ConnectionContraction(Photon *pphot, int ip, Real acon[4][4]) {

  Real xpol[4];
  xpol[IMC0] = pphot->x0p[ip];
  xpol[IMC1] = pphot->x1p[ip];
  xpol[IMC2] = pphot->x2p[ip];
  xpol[IMC3] = pphot->x3p[ip];
  pcoord->Connect(xpol, gamma);

  Real kp[4];
  kp[IMC0] = pphot->k0p[ip];
  kp[IMC1] = pphot->k1p[ip];
  kp[IMC2] = pphot->k2p[ip];
  kp[IMC3] = pphot->k3p[ip];

  for (int i = 0; i < 4; i++) {
    for (int k = 0; k < 4; k++) {
      Real sum = 0.;
      for (int l = 0; l < 4; l++) sum += gamma[i][k][l] * kp[l];
      acon[i][k] = sum;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void GeneralPusher::ApplyPolarizationRate(...)
//! \brief dN/dlambda for a given coherency tensor, from a precomputed A^i_k
//
// eq. 16 of Moscibrodzka & Gammie in vacuum, written as
//
//     dN^ij/dl = -(A^i_k N^kj + A^j_k N^ik),
//
// which is the same contraction with the sum over l pulled out into A.  Done inline it
// recomputes the same sixteen numbers once for every (i,j) pair.

void GeneralPusher::ApplyPolarizationRate(const Real acon[4][4],
                                          const std::complex<Real> nin[4][4],
                                          std::complex<Real> dndl[4][4]) {

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      std::complex<Real> sum(0., 0.);
      for (int k = 0; k < 4; k++) {
        sum -= acon[i][k] * nin[k][j] + acon[j][k] * nin[i][k];
      }
      dndl[i][j] = sum;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void GeneralPusher::AdvanceStep(Photon *pphot, Real step, int ip)
//! \brief advance the geodesic and, when it is tracked, the coherency tensor
//
// The coherency tensor is transported with the midpoint rule: a half step to estimate the
// tensor midway through the interval, then a full step from the initial value using the
// rate evaluated there.  Writing the transport as dN/dl = -A(l) N, that is
//
//   N_half = N_n + (h/2) f(A_n, N_n),    N_(n+1) = N_n + h f(A_mid, N_half),
//
// with A_mid the average of the connection contractions at the two ends of the step.
//
// ipole (ipolarray.c, push_polar) has a true midpoint to hand because its geodesic tracer
// stores Xhalf and Kconhalf for every step: it runs the geodesic to completion first and
// integrates the radiation along the stored ray.  Monte Carlo cannot work that way, since
// the path is not known ahead of time so the midpoint is approximated as in blacklight
// (polarized.cpp), by averaging the endpoints.  Averaging the contracted A rather than
// averaging Gamma and k separately differs at O(h^2) in the operator and O(h^3) in N,
// which is the scheme's own truncation error.
//
// Leaving RK4Step a single full step matters twice: the geodesic is untouched, so
// unpolarized results stay bit-identical, and the endpoint contraction is still the last
// one computed, so acon carries into the next step's predictor as it did before.

void GeneralPusher::AdvanceStep(Photon *pphot, Real step, int ip) {

  if (!IsPolarized(pmy_mcb->pmy_mc->polarized)) {
    RK4Step(pphot, step, ip);
    return;
  }

  std::complex<Real> n0[4][4], d1[4][4];
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) n0[i][j] = pphot->polten[i*4+j][ip];

  // Rate at the step start.  The previous step evaluated the connection at exactly this
  // position and wavevector -- only the tensor it was applied to differs -- so reuse it
  // whenever nothing has moved the photon since.
  if (!acon_valid) ConnectionContraction(pphot, ip, acon);
  Real acon_start[4][4];
  for (int i = 0; i < 4; i++)
    for (int k = 0; k < 4; k++) acon_start[i][k] = acon[i][k];
  ApplyPolarizationRate(acon_start, n0, d1);

  // Half step, giving the tensor midway through the interval
  std::complex<Real> nhalf[4][4];
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      nhalf[i][j] = n0[i][j] + d1[i][j]*(0.5*step);

  RK4Step(pphot, step, ip);

  // Contraction at the step end, which is also what the next step's predictor reuses
  ConnectionContraction(pphot, ip, acon);
  acon_valid = true;

  // Rate at the midpoint, from the averaged operator applied to the half-step tensor
  Real acon_mid[4][4];
  for (int i = 0; i < 4; i++)
    for (int k = 0; k < 4; k++) acon_mid[i][k] = 0.5*(acon_start[i][k] + acon[i][k]);
  std::complex<Real> d2[4][4];
  ApplyPolarizationRate(acon_mid, nhalf, d2);

  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      pphot->polten[i*4+j][ip] = n0[i][j] + d2[i][j]*step;
}

//----------------------------------------------------------------------------------------
//! \fn Real GeneralPusher::StepSize(Photon *pphot, int ip)
//! \brief computes stepsize based on size of current cell

// SWD: Requires updates
// return the stepsize based on the current cell and k-vector
// this should be updated with every iteration since k continuously changes
Real GeneralPusher::StepSize(Photon *pphot, int ip) {

  if (!pmy_mcb->varystep_flag) {
    return step_par; // keep step constant
  }

  Real small = 1.e-30;
  Real kx1 = (fabs(pphot->k1p[ip]) > small) ? fabs(pphot->k1p[ip]) : small;
  Real kx2 = (fabs(pphot->k2p[ip]) > small) ? fabs(pphot->k2p[ip]) : small;
  Real kx3 = (fabs(pphot->k3p[ip]) > small) ? fabs(pphot->k3p[ip]) : small;

  // SWD: May want to store as dx1, etc.
  Real stepx1 = ((pcoord->x1f(pphot->i1p[ip]+1)-pcoord->x1f(pphot->i1p[ip]))/kx1);
  Real stepx2 = ((pcoord->x2f(pphot->i2p[ip]+1)-pcoord->x2f(pphot->i2p[ip]))/kx2);
  Real stepx3 = ((pcoord->x3f(pphot->i3p[ip]+1)-pcoord->x3f(pphot->i3p[ip]))/kx3);

  Real step = (stepx1 < stepx2) ? stepx1 : stepx2;
  step = (step < stepx3) ? step : stepx3;
  Real tol = 1.e-4;
  /*
  if ( (fabs(pcoord->x2f(pphot->i2p[ip]+1) - PI) < tol) || (fabs(pcoord->x2f(pphot->i2p[ip])) < tol) ) {
    if (step == stepx3) {
      step *= 0.01;
    }
  }*/

  //  printf("step: %g %g %g %g\n", step, stepx1, stepx2, stepx3);
  //if (pcoord->x2f(pphot->i2p[ip]) < 0.0000001)
  //  printf("step: %g %g %g %g\n", step, stepx1, stepx2, stepx3);
  return step * step_par;
}
