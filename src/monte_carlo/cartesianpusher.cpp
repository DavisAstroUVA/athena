//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cartesianpusher.cpp
//  \brief implementation for moving photons through cartesian grid

// Athena++ headers
#include "photon.hpp"
#include "photonpusher.hpp"
#include "../mesh/mesh.hpp"

// function prototypes
Real DistanceToNearestFace(MCCoord *pco, Photon *pphot, int ip);

//----------------------------------------------------------------------------------------
//! CartesianPusher class constructor, derived from PhotonPusher base class

CartesianPusher::CartesianPusher(MonteCarloBlock *pmcb)
  : PhotonPusher(pmcb) {

}

//----------------------------------------------------------------------------------------
//! destructor

CartesianPusher::~CartesianPusher() {

}

//----------------------------------------------------------------------------------------
//! \fn void CartesianPusher::Move(Photon *pphot, int ips, int ipe)
//! \brief Moves photon using cell-by-cell approach through spherical polar grid

void CartesianPusher::Move(Photon *pphot, int ips, int ipe) {

  MonteCarloBlock *pmcb = pmy_mcb;
  MCRandom *pran = pmy_mcb->pran;
  MCCoord *pco = pmy_mcb->pcoord;
  Real l_cgs = pmcb->l_cgs;
  Real c_cgs = 2.99792458e10;
  Real c_code = c_cgs / pmcb->vel_cgs;

  // Check if using continuous absorption (all photon use same method)
  bool abs_tau = (pmy_mc->absorption_method[pphot->type[ips]] == ABSTAU);

  for (int ip=ips; ip<=ipe; ip++) {

    // get number of mean free paths photon will travel
    Real tauremaining = GetOpticalDepth(pran);

    Real& kx = pphot->k1p[ip];
    Real& ky = pphot->k2p[ip];
    Real& kz = pphot->k3p[ip];

    // Steps already taken in this free flight, plus the ones taken here.  Counted in a
    // local so it stays in a register across a body that calls out to functions the
    // compiler must assume could alias the photon arrays.
    const int nmv0 = pphot->nmvp[ip];
    int iter = 0;

    while( (tauremaining > 0.) && (pphot->statp[ip] == EVOLVING) &&
           (pphot->dtp[ip] > 0.) ) {
      // Tested before the step is counted, so nmvp only ever counts steps actually taken.
      if (capmove > 0 && nmv0 + iter >= capmove) {
        pphot->statp[ip] = REMOVED;
        break;
      }
      iter++;

      // Compute distance to all faces
      Real dlx, dly, dlz;
      bool ascend[3];
      if(kx > 0.0) {
        dlx = (pco->x1f(pphot->i1p[ip]+1) - pphot->x1p[ip]) / kx;
        ascend[0] = true;
      } else if(kx < 0.0) {
        dlx = (pco->x1f(pphot->i1p[ip]) - pphot->x1p[ip]) / kx;
        ascend[0] = false;
      } else {
        dlx = HUGE_NUMBER;
        ascend[0] = false;
      }

      if(ky > 0.0) {
        dly = (pco->x2f(pphot->i2p[ip]+1)  - pphot->x2p[ip]) / ky;
        ascend[1] = true;
      } else if(ky < 0.0) {
        dly = (pco->x2f(pphot->i2p[ip]) - pphot->x2p[ip]) / ky;
        ascend[1] = false;
      } else {
        dly = HUGE_NUMBER;
        ascend[1] = false;
      }

      if(kz > 0.0) {
        dlz = (pco->x3f(pphot->i3p[ip]+1) - pphot->x3p[ip]) / kz;
        ascend[2] = true;
      } else if(kz < 0.0) {
        dlz = (pco->x3f(pphot->i3p[ip]) - pphot->x3p[ip]) / kz;
        ascend[2] = false;
      } else {
        dlz = HUGE_NUMBER;
        ascend[2] = false;
      }

      int face;
      NextFace(dlx,dly,dlz,face,dl);
      // set total extinction coefficient
      Real chi = abs_tau ? pphot->scp[ip] : (pphot->scp[ip] + pphot->acp[ip]);

      if ((chi > 0.) && (dl*l_cgs > tauremaining / chi)) { // Photon remains in cell
        bool accel_success = false;
        if (acceleration) {
          Real dist;
          dist = DistanceToNearestFace(pco,pphot,ip);

          // Try/perform MRW acceleration if optical depth is large enough
          if (pmcb->coherent_scattering) {
            Real tauacc = 10.;
            if ((pphot->acp[ip]+pphot->scp[ip]) * dist > tauacc)
              accel_success = MRWAcceleration(pphot,pran,dist,tauacc,ip);
          } else {
            Real tauacc = 10.;
            if (pmcb->planck_inv_opacity(pphot->i3p[ip],pphot->i2p[ip],pphot->i1p[ip])
                * dist > tauacc)
              accel_success = MRWAcceleration(pphot,pran,dist,tauacc,ip);
          }
        }

        // Perform standard displacement if acceleration not atempted or unsuccsessful
        if (!accel_success) {
          if (pphot->statp[ip] != EVOLVING)
            break;
          // compute distance remaining in cell
          dl = tauremaining / chi / l_cgs;
          pphot->dtp[ip] -= dl / c_code; // SWD: set with k0p instead

  
          if (pmcb->call_moments) {
            Real dl_cgs = dl * l_cgs;
            if (abs_tau) {
              Real etaua = std::exp(-pphot->acp[ip] * dl_cgs);
              pmcb->UpdateMoments(pphot,dl_cgs,etaua,ip);
              pphot->wp[ip] *= etaua;
            } else {
              pmcb->UpdateMoments(pphot,dl_cgs,ip);
            }
          }
          //pphot->wp[ip] *= etaua;
          // update position
          // k0p is the photon energy, not a unit time component: the coordinate time
          // advance over a path length dl is just dl (in units where c = 1).
          pphot->x0p[ip] += dl;
          pphot->x1p[ip] += pphot->k1p[ip] * dl;
          pphot->x2p[ip] += pphot->k2p[ip] * dl;
          pphot->x3p[ip] += pphot->k3p[ip] * dl;
        }
        // Perform any user work
        if (UserWorkInMove != NULL) UserWorkInMove(pmcb,pphot,this,ip);
        break;

      } else { // Photon moves to next cell and reduce tauremaining
        // Account for absorption (if needed) and update moments
        if (pmcb->call_moments) {
          Real dl_cgs = dl * l_cgs;
          if (abs_tau) {
            Real etaua = std::exp(-pphot->acp[ip] * dl_cgs);
            pmcb->UpdateMoments(pphot,dl_cgs,etaua,ip);
            pphot->wp[ip] *= etaua;
          } else {
            pmcb->UpdateMoments(pphot,dl_cgs,ip);
          }
        }

        // update position
        // k0p is the photon energy, not a unit time component: the coordinate time
          // advance over a path length dl is just dl (in units where c = 1).
          pphot->x0p[ip] += dl;
        pphot->x1p[ip] += pphot->k1p[ip] * dl;
        pphot->x2p[ip] += pphot->k2p[ip] * dl;
        pphot->x3p[ip] += pphot->k3p[ip] * dl;

        tauremaining -= chi * l_cgs * dl;
        pphot->dtp[ip] -= dl / c_code;

        // Perform any user work
        if (UserWorkInMove != NULL) UserWorkInMove(pmcb,pphot,this,ip);
        MovePhotonToNextZone(pphot,pco,pmcb,face,ascend,ip);
      }
    }

    // Retire a photon whose free flight has run past the cap.  nmvp carries across Move
    // calls and blocks, which is what it takes to bound a flight; it is reset at each
    // scattering, in TransferPhotonsOnBlock.
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

  } // loop over photons

}

//----------------------------------------------------------------------------------------
//! \fn Real DistanceToNearestFace(MCCoord *pco, Photon *pphot, int ip)
//! \brief Computes distance to nearest face along current trajectory

Real DistanceToNearestFace(MCCoord *pco, Photon *pphot, int ip) {

  int i1 = pphot->i1p[ip];
  int i2 = pphot->i2p[ip];
  int i3 = pphot->i3p[ip];

  Real dx1p = pco->x1f(i1+1) - pphot->x1p[ip];
  Real dx1m = pphot->x1p[ip] - pco->x1f(i1);
  Real dx2p = pco->x2f(i2+1) - pphot->x2p[ip];
  Real dx2m = pphot->x2p[ip] - pco->x2f(i2);
  Real dx3p = pco->x3f(i3+1) - pphot->x3p[ip];
  Real dx3m = pphot->x3p[ip] - pco->x3f(i3);
  dx1p = (dx1p < dx1m) ? dx1p : dx1m;
  dx2p = (dx2p < dx2m) ? dx2p : dx2m;
  dx3p = (dx3p < dx3m) ? dx3p : dx3m;
  Real dist = (dx1p < dx2p) ? dx1p : dx2p;
  dist = (dist < dx3p) ? dist : dx3p;
  dist = (dist > 0.) ? dist : 0.;
  return dist;
}
