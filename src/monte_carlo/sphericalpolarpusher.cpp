//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file sphericalpolarpusher.cpp
//  \brief implementation for moving photons through sphericalpolar grid

// Athena++ headers
#include "photon.hpp"
#include "photonpusher.hpp"
#include "../mesh/mesh.hpp"


//#define DEBUG_SM
//#define NBUFFER 50

//----------------------------------------------------------------------------------------
//! SphericalPolarPusher class constructor, derived from PhotonPusher base class

SphericalPolarPusher::SphericalPolarPusher(MonteCarloBlock *pmcb)
  : PhotonPusher(pmcb) {

}

//----------------------------------------------------------------------------------------
//! destructor

SphericalPolarPusher::~SphericalPolarPusher() {

}

//----------------------------------------------------------------------------------------
//! \fn void SphericalPolarPusher::Move(Photon *pphot, int ips, int ipe)
//! \brief Moves photon using cell-by-cell approach through spherical polar grid

void SphericalPolarPusher::Move(Photon *pphot, int ips, int ipe) {

  MonteCarloBlock *pmcb = pmy_mcb;
  MCRandom *pran = pmy_mcb->pran;
  MCCoord *pco = pmy_mcb->pcoord;
  PhotonTrajectoryList *ptraj = pmy_mcb->ptraj;
  Real l_cgs = pmcb->l_cgs;
  Real c_cgs = 2.99792458e10;
  Real c_code = c_cgs / pmcb->vel_cgs;

  // Check if using continuous absorption (all photon use same method)
  bool abs_tau = (pmy_mc->absorption_method[pphot->type[ips]] == ABSTAU);

  for (int ip=ips; ip<=ipe; ip++) {

    // get number of mean free paths photon will travel
    Real tauremaining = GetOpticalDepth(pran);
    // References for momentum vectors
    Real& kr  = pphot->k1p[ip];
    Real& kth = pphot->k2p[ip];
    Real& kph = pphot->k3p[ip];
    bool thface = false;

#ifdef DEBUG_SM
    typedef struct {
      Real dlr, dlt, dlp;
      Real cth, sth, cph, sph;
      Real x,y,z;
      int i,j,k;
      bool ascend[3];
      Real l_ext,det,lm,lp;
    } debug_t;
    debug_t db[NBUFFER];
#endif

    Real cth = cos(pphot->x2p[ip]);
    Real sth = sin(pphot->x2p[ip]);
    Real cph = cos(pphot->x3p[ip]);
    Real sph = sin(pphot->x3p[ip]);

    // Make sure kcart is set
    Real kx = kr*sth*cph + kth*cth*cph - kph*sph;
    Real ky = kr*sth*sph + kth*cth*sph + kph*cph;
    Real kz = kr*cth - kth*sth;

    // Steps already taken in this free flight, plus those taken here.  iter has to stay
    // a per-Move count: the DEBUG_SM trace below indexes db[iter-1] into NBUFFER entries,
    // and nmvp carries over from earlier Moves and blocks, so it would run off the end.
    // Nothing would catch that, since DEBUG_SM is off by default.
    const int nmv0 = pphot->nmvp[ip];
    int iter = 0;

    // Move photon until requisite # of mean free paths or escape
    while( (tauremaining > 0.) && (pphot->statp[ip] == EVOLVING) &&
           (pphot->dtp[ip] > 0.) ) {
      // Tested before the step is counted, so nmvp only ever counts steps actually taken.
      if (capmove > 0 && nmv0 + iter >= capmove) {
        pphot->statp[ip] = REMOVED;
        break;
      }
      iter++;

      // Compute cartesian positions
      Real r0 = pphot->x1p[ip];
      Real x0 = r0 * sth * cph;
      Real y0 = r0 * sth * sph;
      Real z0 = r0 * cth;

      // Compute distance to all faces
      Real dlr, dlt, dlp;
      bool ascend[3];

      // r face
      if (kr > 0.0) { // Can only intersect outer sphere
        Real ri = pco->x1f(pphot->i1p[ip]+1) / r0;
        Real delta = 1. + (SQR(ri) - 1.) / SQR(kr);
        dlr = kr * r0 * (sqrt(delta) - 1.);
        ascend[0] = true;
      } else if (kr < 0.0) { // Can intersect either sphere
        Real ri = pco->x1f(pphot->i1p[ip]) / r0;
        Real delta = 1.0 + (SQR(ri)-1.)/SQR(kr);
        if (delta < 0.) {
          // ray does not intersect inner sphere so must intersect outer
          ri = pco->x1f(pphot->i1p[ip]+1) / r0;
          delta = 1. + (SQR(ri) - 1.) / SQR(kr);
          dlr = -kr * r0 * (sqrt(delta) + 1.); //positive solution
          ascend[0] = true;
        } else {
          // ray intersects inner sphere first
          // one or two solutions -- pick shorter dlr
          dlr = r0 * kr * (sqrt(delta) - 1.);
          ascend[0] = false;
        }
      } else { // kr == 0
        Real ri = pco->x1f(pphot->i1p[ip]+1) / r0;
        if (ri > 1.) {
          dlr = r0 * sqrt(SQR(ri) - 1.);
          ascend[0] = true;
        } else {
          std::cout << "Warning: kr == 0 and ri < r0, absorbing photon" << std::endl;
          pphot->PrintPhoton(ip);
          pphot->statp[ip] = DESTROYED;
          break; // break out of while loop
        }
      }

      // theta face
      Real thi;
      if (kth > 0) {
        thi = pco->x2f(pphot->i2p[ip]+1);
        ascend[1] = true;
      } else if (kth < 0) {
        thi = pco->x2f(pphot->i2p[ip]);
        ascend[1] = false;
      } else {
        if (pphot->x2p[ip] < 0.5*PI) {
          thi = pco->x2f(pphot->i2p[ip]+1);
          ascend[1] = true;
        } else {
          thi = pco->x2f(pphot->i2p[ip]);
          ascend[1] = false;
        }
      }
      Real cthi = cos(thi);

      // impact parameters
      Real b2 = SQR(r0) * (1. - SQR(kr));
      Real bz = r0 * (cth - kr * kz);
      Real th_ninf = acos(-kz);
      Real th_pinf = acos(kz);
      //Real l_ext = (kz * b2 - r0 * kr * bz) / bz; // delta l
      Real l_ext = kz * b2 / bz - r0 * kr; // delta l
      Real cth_ext2 = SQR(kz) + SQR(bz)/b2;
      Real th_ext;
      if (bz < 0) {
        th_ext = acos(-sqrt(cth_ext2));
      } else {
        th_ext = acos(sqrt(cth_ext2));
      }

      // check for turning point
      Real lp, lm, lpa, lma;
      if ((l_ext > 0) && ((thi-th_ninf) * (thi-th_ext)) > 0.) {
        dlt = HUGE_NUMBER;
      } else if ((l_ext <= 0) && ((thi-th_ext) * (thi-th_pinf)) > 0.) {
        dlt = HUGE_NUMBER;
      } else { //No turning point
        //Real delta = b2*SQR(cthi)*(cth_ext2-SQR(cthi)); //delta should be positive
        //lm = (-bz*kz - sqrt(delta)) / (SQR(kz)-SQR(cthi)) - r0*kr;
        //lp = (-bz*kz + sqrt(delta)) / (SQR(kz)-SQR(cthi)) - r0*kr;
        // user muller's method
        Real aa = SQR(kz)-SQR(cthi);
        Real bb = 2.0*bz*kz;
        Real cc = SQR(bz) - b2*SQR(cthi);
        Real delta = std::max(0.0, bb*bb - 4.0*aa*cc);
        Real xx = -0.5 * (bb + std::copysign(std::sqrt(delta), bb));
        if (aa == 0.) {
          lm = cc/xx - r0*kr;
          lp = HUGE_NUMBER;
        } else {
          lm = xx/aa - r0*kr;
          lp = cc/xx - r0*kr;
        }
        if (lm <= 0.) {
          if (lp <= 0.)
            dlt = HUGE_NUMBER;
          else
            dlt = lp;
        } else if (lp <= 0.) {
          dlt = lm;
        } else {
          if (lp >= lm)
            dlt = lm;
          else
            dlt = lp;
        }
      }
 
      // phi face
      Real phii;
      if (kph > 0) {
        phii = pco->x3f(pphot->i3p[ip]+1);
        ascend[2] = true;
      } else if (kph < 0.0) {
        phii = pco->x3f(pphot->i3p[ip]);
        ascend[2] = false;
      }
      Real tphii;
      if (kph != 0.) {
        tphii = tan(phii);
        if (fabs(kx * tphii - ky) < TINY_NUMBER) {
          dlp = HUGE_NUMBER;
        } else {
          dlp = r0 * sth * (sph-tphii*cph) / (kx*tphii-ky);
          if (dlp < 0.) dlp = HUGE_NUMBER;
        }
      } else {
        dlp = HUGE_NUMBER;
      }


      if ((dlr <= dlt) && (dlr <= dlp)) {
        dl = dlr;
      } else if ((dlp <= dlt) && (dlp <= dlr)) {
        dl = dlp;
      } else if ((dlt <= dlr) && (dlt <= dlp)) {
        dl = dlt;
      }

      //checking turning point for theta
      if ((dl > l_ext) && (l_ext > 0.)) {
        if (kth < 0.) {
          cthi = cos(pco->x2f(pphot->i2p[ip]+1));
        } else if (kth > 0.) {
          cthi = cos(pco->x2f(pphot->i2p[ip]));
        }
        b2 = SQR(r0) * (1. - SQR(kr));
        bz = r0 * (cth - kr * kz);
        Real delta = b2 * SQR(cthi) * (cth_ext2 - SQR(cthi)); //delta should be positive
        Real lm = (-bz*kz - sqrt(delta)) / (SQR(kz) - SQR(cthi)) - kr*r0;
        Real lp = (-bz*kz + sqrt(delta)) / (SQR(kz) - SQR(cthi)) - kr*r0;
        /*Real aa = SQR(kz)-SQR(cthi);
        Real bb = 2.0*bz*kz;
        Real cc = SQR(bz) - b2*SQR(cthi);
        Real delta = std::max(0.0, bb*bb - 4.0*aa*cc);
        Real xx = -0.5 * (bb + std::copysign(std::sqrt(delta), bb));
        if (aa == 0.) {
          lm = cc/xx - r0*kr;
          lp = HUGE_NUMBER;
        } else {
          lm = xx/aa - r0*kr;
          lp = cc/xx - r0*kr;
        }*/
#ifdef DEBUG_SM
        if (iter < NBUFFER) {
          db[iter-1].lm = lm;
          db[iter-1].lp = lp;
          db[iter-1].delta = delta;
        }
#endif
        if (lm <= 0) {
          if ( (lp > 0) && !(thface) )
            dlt = lp;
          else
            dlt = HUGE_NUMBER;
        } else {
          if (lp > 0)
            dlt = (lm > lp) ? lm : lp;
          else
            dlt = lm;
        }
        if (dlt <= dl) {
          ascend[1] = !(ascend[1]);
        }
      }

      int face;
      NextFace(dlr,dlt,dlp,face,dl);
#ifdef DEBUG_SM
      if (iter < NBUFFER) {
        db[iter-1].dl = dl; db[iter-1].dlr = dlr;
        db[iter-1].dlt = dlt; db[iter-1].dlp = dlp;
        db[iter-1].cth = cth; db[iter-1].sth = sth;
        db[iter-1].cph = cph; db[iter-1].sph = sph;
        db[iter-1].kr = kr; db[iter-1].kth = kth; db[iter-1].kph = kph;
        db[iter-1].kx = kx; db[iter-1].ky = ky; db[iter-1].kz = kz;
        db[iter-1].x = pphot->x[IMC1];
        db[iter-1].y = pphot->x[IMC2]; db[iter-1].z = pphot->x[IMC3];
        db[iter-1].i = pphot->i1;
        db[iter-1].j = pphot->i2; db[iter-1].k = pphot->i3;
        db[iter-1].ascend[0] = ascend[0]; db[iter-1].ascend[1] = ascend[1];
        db[iter-1].ascend[2] = ascend[2];
        db[iter-1].l_ext = l_ext;
      }
#endif
      // set total extinction coefficient
      Real chi = abs_tau ? pphot->scp[ip] : (pphot->scp[ip] + pphot->acp[ip]);

      bool test = false;
      if ((chi > 0.) && (dl * l_cgs > tauremaining / chi)) { // Photon remains in cell
        bool accel_success = false;
        if (acceleration) {
          Real dist = pco->dmin(pphot->i3p[ip],pphot->i2p[ip],pphot->i1p[ip]);
          // Try/perform MRW acceleration if optical depth is large enough
          if (pmcb->coherent_scattering) {
            Real tauacc = 10.;
            if ((pphot->acp[ip]+pphot->scp[ip]) * dist > tauacc)
              accel_success = MRWAcceleration(pphot,pran,dist,tauacc,ip);
          } else {
            Real tauacc = 10.;
            Real pio = pmcb->planck_inv_opacity(pphot->i3p[ip],pphot->i2p[ip],pphot->i1p[ip]);
            if (pio * dist > tauacc)
              accel_success = MRWAcceleration(pphot,pran,dist,tauacc,ip);
          }
        }

        if (!accel_success) {
          if (pphot->statp[ip] != EVOLVING)
            break; // break out of while loop
          // compute distance remaining in cell
          dl = tauremaining / chi / l_cgs;
          pphot->dtp[ip] -= dl / c_code;;
          // Update moments
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
          // Update postions
          pphot->x1p[ip] = sqrt(SQR(r0) + 2. * dl * kr * r0 + SQR(dl));
          pphot->x2p[ip] = acos((z0 + kz * dl) / pphot->x1p[ip]);
          pphot->x3p[ip] = atan2(y0 + ky * dl,x0 + kx * dl);
          if (pphot->x3p[ip] < 0.)
            pphot->x3p[ip] += 2.*PI;
          // k0p is the photon energy, not a unit time component: the coordinate time
          // advance over a path length dl is just dl (in units where c = 1).
          pphot->x0p[ip] += dl;

          // Update k vector
          cth = cos(pphot->x2p[ip]);
          sth = sin(pphot->x2p[ip]);
          cph = cos(pphot->x3p[ip]);
          sph = sin(pphot->x3p[ip]);
          kr  = kx * sth * cph + ky * sth * sph + kz * cth;
          kth = kx * cth * cph + ky * cth * sph - kz * sth;
          kph = -kx * sph + ky * cph;
        }
        //if (ptraj != NULL) ptraj->AddToTrajectory(pphot,ip);
        if (UserWorkInMove != NULL) UserWorkInMove(pmcb,pphot,this,ip);
        break;
      } else { // Photon moves to next cell and reduce tauremaining
        // Update moments
        pphot->dtp[ip] -= dl / c_code;

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
        // Update positions
        pphot->x1p[ip] = sqrt(SQR(r0) + 2. * dl * kr * r0 + SQR(dl));
        pphot->x2p[ip] = acos((z0 + kz * dl) / pphot->x1p[ip]);
        pphot->x3p[ip] = atan2(y0 + ky * dl,x0 + kx * dl);
        if (pphot->x3p[ip] < 0.)
          pphot->x3p[ip] += 2.*PI;
        // k0p is the photon energy, not a unit time component: the coordinate time
          // advance over a path length dl is just dl (in units where c = 1).
          pphot->x0p[ip] += dl;

        tauremaining -= chi * l_cgs * dl;
        // move photon to next cell and update angular positions
        MovePhotonToNextZone(pphot,pco,pmcb,face,ascend,ip);
        if ((face == 1) || (face == 3) || (face == 4) || (face == 6))
          thface = true;
        cth = cos(pphot->x2p[ip]);
        sth = sin(pphot->x2p[ip]);
        cph = cos(pphot->x3p[ip]);
        sph = sin(pphot->x3p[ip]);
        kr  = kx * sth * cph + ky * sth * sph + kz * cth;
        kth = kx * cth * cph + ky * cth * sph - kz * sth;
        kph = -kx * sph + ky * cph;
        //if (ptraj != NULL) ptraj->AddToTrajectory(pphot,ip);
      }

      if (UserWorkInMove != NULL) UserWorkInMove(pmcb,pphot,this,ip);

    } // end while loop

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

  } // loop over ip

}
