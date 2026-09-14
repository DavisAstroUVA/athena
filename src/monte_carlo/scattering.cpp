//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//!  \file scattering.cp
//!  \brief implementation of scattering functions

// Athena++ headers
#include "montecarlo.hpp"
#include "photonpusher.hpp"

//----------------------------------------------------------------------------------------
//! \fn void NoScatter(MonteCarloBlock *pmcb, Photon *pphot, int ips, int ipe)
//! \brief prints warning, should not be called

void NoScatter(MonteCarloBlock *pmcb, Photon *pphot, int ips, int ipe) {

  for (int ip=ips; ip<=ipe; ip++) {
    std::cout << "Warning: Scatter called with scattering=none." << std::endl;
    pphot->PrintPhoton(ip);
    pphot->statp[ip] = DESTROYED;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ScatterIsotropic(MonteCarloBlock *pmcb, Photon *pphot, int ips, int ipe)
//! \brief isotropic scattering of unpolarized radiation

void ScatterIsotropic(MonteCarloBlock *pmcb, Photon *pphot, int ips, int ipe) {

  MCRandom *pran = pmcb->pran;

  for (int ip=ips; ip<=ipe; ip++) {

    Real phi = 2.*PI * pran->uniform();
    Real cphi = cos(phi);
    Real sphi = sin(phi);

    Real cth = 2. * pran->uniform() - 1.;
    Real sth = sqrt(1. - SQR(cth));

    // calculate new wave vector
    pphot->k1p[ip] = sth * cphi;
    pphot->k2p[ip] = sth * sphi;
    pphot->k3p[ip] = cth;

  }

}

//----------------------------------------------------------------------------------------
//! \fn void ScatterThomsonPolarized(MonteCarloBlock *pmcb, Photon *pphot, int ips,
//!                                  int ipe)
//! \brief Thomson scattering of polarized radiation
//  See Chandrasekhar(1960) Chapter 1 figure 8
//  Rotation matrix (r):  L(\pi-i2)R(\Theta)L(-i1)
//  Here \Theta=smu

// SWD: Needs to be changed to handle non-cartesian k
void ScatterThomsonPolarized(MonteCarloBlock *pmcb, Photon *pphot, int ips, int ipe) {

  MCRandom *pran = pmcb->pran;
  PhotonPusher *ppusher = pmcb->ppusher;

  for (int ip=ips; ip<=ipe; ip++) {

    Real norm = pphot->sip[ip];
    Real stokes[4];
    stokes[0] = 1.;
    stokes[1] = pphot->sqp[ip] / norm;
    stokes[2] = pphot->sup[ip] / norm;
    stokes[3] = pphot->svp[ip] / norm;

    // Polarized scattering must be computed relative to cartesian bases due to
    // definition of stokes vectors
    Real &kx = pphot->k1p[ip];
    Real &ky = pphot->k2p[ip];
    Real &kz = pphot->k3p[ip];

    Real mu = kz;
    Real stheta = sqrt(1. - SQR(mu));
    
    Real phio = 0.;
    if (stheta > TINY_NUMBER) {
      Real carg = kx / stheta;
      carg = (carg > 1.) ? 1. : ((carg < -1.) ? -1. : carg);
      phio = acos(carg);
      if(ky < 0.0)
        phio = 2.*PI - phio;
    }

    Real smu, sstheta, s1, s2, s3, i1;
    Real ci1, si1, ci2, si2, s2i1, c2i1;
    Real intensi;
    do {
      smu = 1. - 2. * pran->uniform();
      sstheta = sqrt(1.0 - SQR(smu));

      s1 = SQR(smu) + 1.0;
      s2 = SQR(smu) - 1.0;
      s3 = 2. * smu;

      i1 = 2. * PI * pran->uniform();
      ci1 = cos(i1);
      si1 = sin(i1);
      s2i1 = 2. * si1 * ci1;
      c2i1 = 2. * ci1 * ci1 - 1.;

      Real r00 =  s1;
      Real r01 =  s2 * c2i1;
      Real r02 = -s2 * s2i1;

      intensi = r00 * stokes[0] + r01 * stokes[1] + r02 * stokes[2];

    } while (pran->uniform()>0.5*intensi);


    Real r10,r11,r12,r20,r21,r22;
    Real mup,sthetap,phip;
    if(i1 < PI) {
      mup = mu * smu + stheta * sstheta * ci1;
      if(mup > 1.0)
        mup = 1.0;
      else if(mup < -1.0)
        mup = -1.0;

      sthetap = sqrt(1. - SQR(mup));

      if( (sthetap != 0.) && (sstheta !=0.)) {
        si2 = si1 * stheta / sthetap;
        ci2 = (mu - mup * smu)/ (sthetap * sstheta);
      } else {
        if (sthetap == 0.0) {
          si2 = 0.;
          ci2 = 1.;
        } else if (sstheta == 0.0) {
          si2 = si1;
          ci2 = -ci1;
        }
      }
      Real s2i2 = 2. * si2 * ci2;
      Real c2i2 = 2. * ci2 * ci2 - 1.;

      Real cdphi = -ci1 * ci2 + si1 * si2 * smu;
      if(cdphi > 1.)
        cdphi = 1.0;
      else if(cdphi < -1.)
        cdphi = -1.;

      phip = phio - acos(cdphi);
      if(phip > 2.*PI)
        phip = phip - 2.*PI;
      else if (phip < 0.)
        phip = phip + 2.*PI;

      Real c1c2 = c2i1 * c2i2;
      Real s1s2 = s2i1 * s2i2;
      Real c1s2 = c2i1 * s2i2;
      Real s1c2 = c2i2 * s2i1;

      r10 =  s2 * c2i2;
      r11 =  s1 * c1c2 - s3 * s1s2;
      r12 = -s1 * s1c2 - s3 * c1s2;
      r20 =  s2 * s2i2;
      r21 =  s1 * c1s2 + s3 * s1c2;
      r22 =  s3 * c1c2 - s1 * s1s2;

    } else {
      si1 = -si1;
      s2i1 = -s2i1;
      mup = mu * smu + stheta * sstheta * ci1;
      if(mup > 1.)
        mup = 1.;
      else if(mup < -1.)
        mup = -1.;

      sthetap = sqrt(1. - SQR(mup));
      if( (sthetap != 0.) && (sstheta != 0.)) {
        si2 = si1 * stheta / sthetap;
        ci2 = (mu - mup * smu)/ (sthetap * sstheta);
      }
      else {
        if (sthetap == 0.0) {
          si2 = 0.0;
          ci2 = 1.0;
        } else if (sstheta == 0.0) {
          si2 = si1;
          ci2 = -ci1;
        }
      }

      Real s2i2 = 2.0*si2*ci2;
      Real c2i2 = 2.0*ci2*ci2-1.0;

      Real cdphi = -ci1 * ci2 + si1 * si2 * smu;
      if(cdphi > 1.)
        cdphi = 1.0;
      else if(cdphi < -1.)
        cdphi = -1.;

      phip = phio + acos(cdphi);
      if(phip > 2.*PI) {
        phip = phip - 2.*PI;
      } else if (phip < 0.) {
        phip = phip + 2.*PI;
      }

      Real c1c2 = c2i1 * c2i2;
      Real s1s2 = s2i1 * s2i2;
      Real c1s2 = c2i1 * s2i2;
      Real s1c2 = c2i2 * s2i1;

      r10 =  s2 * c2i2;
      r11 =  s1 * c1c2 - s3 * s1s2;
      r12 =  s1 * s1c2 + s3 * c1s2;
      r20 = -s2 * s2i2;
      r21 = -s1 * c1s2 - s3 * s1c2;
      r22 =  s3 * c1c2 - s1 * s1s2;
    } // end if (i1 < PI) else

    // Calculate new stokes vectors from rotation matrix.  Note that the
    // value of the overall intensity (stokes[0]) does not change
    pphot->sqp[ip] = norm*(r10*stokes[0]+r11*stokes[1]+r12*stokes[2])/intensi;
    pphot->sup[ip] = norm*(r20*stokes[0]+r21*stokes[1]+r22*stokes[2])/intensi;
    pphot->svp[ip] = norm*(s3*stokes[3])/intensi;

    // Calculate new photon direction
    kx = sthetap * cos(phip);
    ky = sthetap * sin(phip);
    kz = mup;

  } // end loop over ip

}


//----------------------------------------------------------------------------------------
//! \fn void ScatterThomsonPolarized(MonteCarloBlock *pmcb, Photon *pphot, int ips,
//!                                  int ipe)
//! \brief Thomson scattering of unpolarized radiation

void ScatterThomsonUnpolarized(MonteCarloBlock *pmcb, Photon *pphot, int ips, int ipe) {

  MCRandom *pran = pmcb->pran;

  for (int ip=ips; ip<=ipe; ip++) {

    Real &k1 = pphot->k1p[ip];
    Real &k2 = pphot->k2p[ip];
    Real &k3 = pphot->k3p[ip];

    Real k1p,k2p,k3p,cths;
    do {
      Real cthp = 2.0 * pran->uniform() - 1.;
      Real phip = 2.*PI * pran->uniform();
      Real cphip = cos(phip);
      Real sphip = sin(phip);
      Real sthp = sqrt(1. - SQR(cthp));

      k1p = sthp * cphip;
      k2p = sthp * sphip;
      k3p = cthp;

      cths = k1 * k1p + k2 * k2p + k3 * k3p;

    } while(pran->uniform() > 0.5*(1+SQR(cths)));

    k1 = k1p;
    k2 = k2p;
    k3 = k3p;

  } // end loop over ip
}

//----------------------------------------------------------------------------------------
//! \fn void ScatterComptonUnpolarized(MonteCarloBlock *pmcb, Photon *pphot, int ips,
//!                                    int ipe)
//! \brief Unpolarized Compton scattering

void ScatterComptonUnpolarized(MonteCarloBlock *pmcb, Photon *pphot, int ips, int ipe) {

  MCRandom *pran = pmcb->pran;
  Real mec2 = 8.18711e-7;
  Real ethom = 1.602176634e-11;
  for (int ip=ips; ip<=ipe; ip++) {
    if (pphot->ep[ip] < ethom) {
      ScatterThomsonUnpolarized(pmcb,pphot,ip,ip);
      continue;
    }

    Real &k1 = pphot->k1p[ip];
    Real &k2 = pphot->k2p[ip];
    Real &k3 = pphot->k3p[ip];
    int &i1 = pphot->i1p[ip];
    int &i2 = pphot->i2p[ip];
    int &i3 = pphot->i3p[ip];

    Real v1,v2,v3,vdc,gamma,x,onemuvdc;
    do {
      // Pick random direction for velocity of scattering electron
      v3 = 2.0*pran->uniform() - 1.;
      Real sinv = sqrt(1. - SQR(v3));
      Real phiv = 2.*PI * pran->uniform();
      v1 = sinv * cos(phiv);
      v2 = sinv * sin(phiv);

      // Calculate v . k
      Real vdotk = k1 * v1 + k2 * v2 + k3 * v3;

      // Draw momentum from random distribution and calculate gamma,v/c,x
      // sigma^ and test for acceptance
      Real mom = ElectronDist(pmcb->tgas(i3,i2,i1),pran);
      gamma = sqrt(1. + SQR(mom));
      vdc = mom / gamma;
      onemuvdc = 1. - vdotk * vdc;
      x = 2. * gamma * pphot->ep[ip] * onemuvdc / mec2;
    } while(pran->uniform() >= 0.375*SigmaHat(x)*onemuvdc);

    // Calculate rho,mup,phip,cths (cos scattering angle) etc.
    // and test for acceptance
    Real rho = sqrt(SQR(v1) + SQR(v2));
    Real onecthpvdc,k1p,k2p,k3p,xp;
    do {
      Real cthp = 2.0 * pran->uniform() - 1.;
      cthp = (vdc + cthp) / (1. + vdc * cthp);
      Real phip = 2.*PI * pran->uniform();
      Real cphip = cos(phip);
      Real sphip = sin(phip);
      Real sthp = sqrt(1. - SQR(cthp));

      if(rho > 0.0) {
        k1p = cthp * v1 + sthp / rho * (v2 * cphip + v1 * v3 * sphip);
        k2p = cthp * v2 + sthp / rho * (v2 * v3 * sphip - v1 * cphip);
        k3p = cthp * v3 - sthp * rho * sphip;
      } else {
        k1p = sthp * cphip;
        k2p = sthp * sphip;
        k3p = cthp;
      }

      Real cths = k1 * k1p + k2 * k2p + k3 * k3p;
      onecthpvdc = 1. - cthp * vdc;
      xp = x / (1. + pphot->ep[ip] / mec2 * (1.-cths) / (gamma*onecthpvdc));

    } while(2.*pran->uniform() >= Bigy(x,xp));

    // Assign new energy and direction vector
    pphot->ep[ip] = xp * mec2 / (2. * gamma * onecthpvdc);
    k1 = k1p;
    k2 = k2p;
    k3 = k3p;

  } //end loop over ip
}


//------------------------------------------------------------------------------
//! \fn void ScatterComptonPolarized(MonteCarloBlock *pmcb, Photon *pphot,
//!                                  int ips, int ipe)
//  \brief Polarized Compton scattering
//
// Note that this uses the correct R matrix expressions -- not the incorrect
// expression listed in the appendix of Davis et al. (2009) */

void ScatterComptonPolarized(MonteCarloBlock *pmcb, Photon *pphot, int ips,
                             int ipe) {

  MCRandom *pran = pmcb->pran;

  for (int ip=ips; ip<=ipe; ip++) {

    Real norms = pphot->sip[ip];
    Real stokes[4];
    stokes[0] = 1.;
    stokes[1] = pphot->sqp[ip] / norms;
    stokes[2] = pphot->sup[ip] / norms;
    stokes[3] = pphot->svp[ip] / norms;
;
    // Polarized scattering must be computed relative to cartesian basis due
    // to definition of stokes vectors
    Real &kx = pphot->k1p[ip];
    Real &ky = pphot->k2p[ip];
    Real &kz = pphot->k3p[ip];

    int &i1 = pphot->i1p[ip];
    int &i2 = pphot->i2p[ip];
    int &i3 = pphot->i3p[ip];

    Real stheta = sqrt(1. - SQR(kz));
    Real mec2 = 8.18711e-7;
    Real v1,v2,v3,vdc,mom,gamma,bgamma,x,vdotk;
    do {
      // Pick random direction for velocity of scattering electron
      v3 = 2.0*pran->uniform()-1.0;
      Real sinv = sqrt(1.0 - SQR(v3));
      Real phiv = 2.*PI*pran->uniform();
      v1 = sinv * cos(phiv);
      v2 = sinv * sin(phiv);

      // Calculate v . k
      vdotk = kx * v1 + ky * v2 + kz * v3;

      // Draw momentum from random distribution and calculate gamma,v/c,x
      // sigma^ and test for acceptance
      mom = ElectronDist(pmcb->tgas(i3,i2,i1),pran);
      gamma = sqrt(1. + SQR(mom));
      bgamma = gamma - vdotk * mom;
      x = 2.0*pphot->ep[ip] /mec2*bgamma;

    } while(pran->uniform()>=0.375*SigmaHat(x)*bgamma/gamma);

    Real rho = sqrt(v1*v1+v2*v2);
    Real beta = mom/gamma;

    // Calculate scattering angles and scattered photon energies in fluid frame
    // to evaluate probability for scattering
    Real mup,k1p,k2p,k3p,bgammap,mus,cmus,xp,erat,diff,deltac,kcon,s2i1,c2i1;
    Real s2,r00,r01,r02,norm;
    do {

      // Calculate trial scattering angles
      Real mud = 2.0*pran->uniform()-1.0;
      mup = (beta + mud)/(1.0 + beta * mud);

      Real phip = 2.*PI*pran->uniform();
      Real cphip = cos(phip);
      Real sphip = sin(phip);
      Real sthetap = sqrt(1.0-mup*mup);
      bgammap = gamma-mup*mom;

      if(rho > 0.0) {
        k1p = mup * v1 + sthetap / rho * (v2*cphip + v1*v3*sphip);
        k2p = mup * v2 + sthetap / rho * (v2*v3*sphip - v1*cphip);
        k3p = mup * v3 - sthetap * rho * sphip;
      } else {
        k1p = sthetap * cphip;
        k2p = sthetap * sphip;
        k3p = mup;
      }

      mus = kx*k1p + ky*k2p + kz*k3p;
      cmus = 1.0 - mus;

      xp = x / (1.0+pphot->ep[ip]/mec2*cmus/bgammap);
      erat = xp / x;
      diff = (1.0/x-1.0/xp);
      deltac = sqrt(-1.0-1.0/diff);

      // Cacluate angle, i1, which rotates stokes vectors into "internal"
      // scattering basis
      Real bottom = stheta*cmus*deltac;
      kcon = k1p * ky - k2p * kx;
      Real ci1,si1;
      if(fabs(bottom) > TINY_NUMBER) {
        ci1 = (bgamma*(k3p-mus*kz)+mom*cmus*(kz*vdotk-v3))/bottom;
        si1 = (bgamma*kcon-mom*cmus*(v1*ky-v2*kx))/bottom;
      } else { // SWD: Very rare, but this should be improved
        ci1 = 1.0;
        si1 = 0.0;
      }
      s2i1 = 2.0*si1*ci1;
      c2i1 = 2.0*ci1*ci1-1.0;

      s2 = 4.0*diff*(1.0+diff);
      r00 = (erat+1.0/erat+s2);
      r01 = s2*c2i1;
      r02 = s2*s2i1;

      norm = r00 + r01 * stokes[1] + r02 * stokes[2];

      // Scattering probability proportional to cross section
    } while(2*pran->uniform() >= (erat*erat*norm));

    // Calculate other elements of the Compton matrix
    Real s1 = s2+2.0;
    Real s3 = 4.0*diff+2.0;

    // Determine i2 angle which rotates stokes vectors back from scattering plane
    Real bottom = sqrt(1.0-k3p*k3p)*cmus*deltac;
    Real ci2,si2;
    if( fabs(bottom) > TINY_NUMBER) {
      ci2 = (bgammap*(kz-mus*k3p)+mom*cmus*(k3p*mup-v3))/bottom;
      si2 = -(bgammap*kcon+mom*cmus*(v1*k2p-v2*k1p))/bottom;
    } else {
      ci2 = 1.0;
      si2 = 0.0;
    }
    Real s2i2 = 2.0*si2*ci2;
    Real c2i2 = 2.0*ci2*ci2-1.0;

    // Rotate Stokes vectors
    Real c1c2 = c2i1 * c2i2;
    Real s1s2 = s2i1 * s2i2;
    Real c1s2 = c2i1 * s2i2;
    Real s1c2 = c2i2 * s2i1;

    Real r10 = s2 * c2i2;
    Real r11 = s1 * c1c2 + s3 * s1s2;
    Real r12 = s1 * s1c2 - s3 * c1s2;

    Real r20 = s2 * s2i2;
    Real r21 = s1 * c1s2 - s3 * s1c2;
    Real r22 = s3 * c1c2 + s1 * s1s2;

    Real r33 = 0.5*s3*(erat + 1.0/erat);

    // Calculate new stokes vectors from rotation matrix.  Note that the
    // value of the overall intensity (stokes[0]=1) does not change
    pphot->sqp[ip] = norms*(r10+r11*stokes[1]+r12*stokes[2]) / norm;
    pphot->sup[ip] = norms*(r20+r21*stokes[1]+r22*stokes[2]) / norm;
    pphot->svp[ip] = norms*(r33*stokes[3]) / norm;

    if (pphot->IsNanPhoton(ip)) {
      pphot->statp[ip] = DESTROYED;
      pphot->PrintPhoton("Warning: Nan encounterd in ScatterComptonPolarized(),"
                               " photon destroyed",ip);
      printf("stokes %g %g\n",stokes[1],stokes[2]);
    }
    // Calculate scattered energy and update k vector
    pphot->ep[ip] = xp / (2.0 * bgammap) * mec2;
    kx = k1p;
    ky = k2p;
    kz = k3p;

  } // end loop over ip
}

//----------------------------------------------------------------------------------------
//! \fn void ScatterResonanceLine(MonteCarloBlock *pmcb, Photon *pphot, int ips, int ipe)
//! \brief Implements resonance line scattering

void ScatterResonanceLine(MonteCarloBlock *pmcb, Photon *pphot, int ips, int ipe) {

  MCRandom *pran = pmcb->pran;

  for (int ip=ips; ip<=ipe; ip++) {

    Real &kx = pphot->k1p[ip];
    Real &ky = pphot->k2p[ip];
    Real &kz = pphot->k3p[ip];

    int &i1 = pphot->i1p[ip];
    int &i2 = pphot->i2p[ip];
    int &i3 = pphot->i3p[ip];

    // Compute atom thermal velocity
    Real kb = MCConstants::kb_cgs;
    Real mass = MCConstants::mH_cgs;
    Real tgas = pmcb->tgas(i3,i2,i1);
    Real vth = sqrt( 2 * kb * tgas / mass );

    // Compute a and x
    Real nu0 = MCConstants::nu_lya;
    Real c = MCConstants::c_cgs;
    Real doppwidth = nu0 * vth / c;
    Real lorwidth = MCConstants::lorwidth_lya;
    Real a = lorwidth / doppwidth;
    Real h = MCConstants::h_cgs;
    Real nu = pphot->ep[ip] / h;
    Real x = (nu - nu0) / doppwidth;

    // Parallel and perpendicular velocities
    Real vperp = vth * sqrt(-log(pran->uniform())) * cos(2.*PI*pran->uniform());
    Real vpar = vth * SampleVelocityParallel(a, x, pran);

    //Compute incoming angles
    Real sth_in = sqrt(1. - SQR(kz));
    Real phi_in = atan2(ky , kx);

    // Sample outgoing angles
    Real cth,sth,phi,cgam;
    do {
      // sample uniform in cos theta
      cth = 2. * pran->uniform() - 1.;
      sth = sqrt(1. - SQR(cth));

      // sample uniform phi
      phi = 2. * PI * pran->uniform();

      // evaluate scattering angle
      cgam = sth_in * sth * cos(phi_in - phi) + cth * kz;

    } while (pran->uniform()*2. > (1.+SQR(cgam)));

    kx = sth * cos(phi);
    ky = sth * sin(phi);
    kz = cth;

    // Evaluate outgoing photon energy
    Real sgam = sqrt(1. - SQR(cgam));
    pphot->ep[ip] = h * (nu + nu0 * ( (cgam - 1.) * vpar + sgam * vperp ) / c);

    //printf("nu: %g %g %g %g %g %g\n", pphot->ep[i]/h, nu, cgam, vpar, vperp, vth);

  } // end loop over ip
}

//----------------------------------------------------------------------------------------
//! \fn void ScatterDust(MonteCarloBlock *pmcb, Photon *pphot, int ips, int ipe)
//! \brief Implements dust scattering

void ScatterDust(MonteCarloBlock *pmcb, Photon *pphot, int ips, int ipe) {

  MCRandom *pran = pmcb->pran;
  PhotonPusher *ppusher = pmcb->ppusher;

  Real g=0.41; //will make an parameter
  Real g2 = g*g;
  Real pl=0.51; //will make a parameter
  Real pc=0.;
  Real sc = 1.; //asymmentry of circular polariztion ---- idk what value for this
  for (int ip=ips; ip<=ipe; ip++) {

    Real norm = pphot->sip[ip];
    Real stokes[4];
    stokes[0] = 1.;
    stokes[1] = pphot->sqp[ip] / norm;
    stokes[2] = pphot->sup[ip] / norm;
    stokes[3] = pphot->svp[ip] / norm;

    // Polarized scattering must be computed relative to cartesian bases due to
    // definition of stokes vectors
    Real &kx = pphot->k1p[ip];
    Real &ky = pphot->k2p[ip];
    Real &kz = pphot->k3p[ip];

    Real ctho = kz;
    Real stho = sqrt(1. - SQR(ctho));
    //printf("tho: %g %g\n",ctho,stho);
    //Real phio = acos(kx / stho);
    //if(ky < 0.0)
    //  phio = 2.*PI - phio;
    Real phio = atan2(ky,kx);
    if(phio < 0.0)
      phio += 2.*PI;

    Real xi = pran->uniform();
    Real cths = (1.+g2-SQR( (1. - g2) / (1. - g + 2.*g*xi)))/(2.*g);
    //Real cths = (1.+g2-SQR((1.-g2)/(1.-g+2.*g*0.05)))/(2.*g); //SWD
    Real sths = sqrt(1.0 - SQR(cths));

    //changed to p12&3 from equation 32
    Real s1 = (1. - g2) / pow(1. + g2 - 2.*g*cths,1.5);
    Real s2 = -pl * s1 * (1. - SQR(cths)) / (1. + SQR(cths));
    Real s3 = s1 * (2. * cths) / (1. + SQR(cths));
    Real ths = acos(cths); //scattering angle
    Real thf = ths * (1. + (3.13 * sc *exp(-7. * ths / PI)));
    Real s4 = -pc * s1 * (1. - SQR(cos(thf))) / (1. + SQR(cos(thf)));


    //Real i1 = 0.2*PI*2.; //SWD
    //Real i1 = PI * pran->uniform()+PI;//SWD
    Real i1 = 2. * PI * pran->uniform();
    Real ci1 = cos(i1);
    Real si1 = sin(i1);
    Real s2i1 = 2. * si1 * ci1;
    Real c2i1 = 2. * ci1 * ci1 - 1.;

    Real r00 =  s1;
    Real r01 =  s2 * c2i1;
    Real r02 = -s2 * s2i1;
    //printf("\ns %g %g %g %g\n",s1,s2,s3,s4);
    //printf("r %g %g %g\n",r00,r01,r02);
    Real inew = r00 * stokes[0] + r01 * stokes[1] + r02 * stokes[2];
    Real r10, r11, r12, r13, r20, r21, r22, r23, r31, r32, r33;
    Real cthp,sthp,phip;
    if(i1 < PI) { // i1 < pi
      cthp = ctho * cths + stho * sths * ci1;
      if(cthp > 1.0)
        cthp = 1.0;
      else if(cthp < -1.0)
        cthp = -1.0;
      sthp = sqrt(1. - SQR(cthp));

      Real ci2,si2;
      if( (sthp != 0.) && (sths !=0.)) {
        //printf("%g %g %g\n",si1,stho,sthp);
        si2 = si1 * stho / sthp;
        ci2 = (ctho - cthp * cths)/ (sthp * sths);
      } else {
        if (sthp == 0.0) {
          si2 = 0.;
          ci2 = 1.;
        } else if (sths == 0.0) {
          si2 = si1;
          ci2 = -ci1;
        }
      }
      Real s2i2 = 2. * si2 * ci2;
      Real c2i2 = 2. * ci2 * ci2 - 1.;

      Real cdphi = -ci1 * ci2 + si1 * si2 * cths;
      if(cdphi > 1.)
        cdphi = 1.0;
      else if(cdphi < -1.)
        cdphi = -1.;

      phip = phio - acos(cdphi);
      if(phip > 2.*PI)
        phip = phip - 2.*PI;
      else if (phip < 0.)
        phip = phip + 2.*PI;

      Real c1c2 = c2i1 * c2i2;
      Real s1s2 = s2i1 * s2i2;
      Real c1s2 = c2i1 * s2i2;
      Real s1c2 = c2i2 * s2i1;
      //printf("a %g %g %g %g %g %g\n",c1c2,s1s2,c2i1,c2i2,s2i1,s2i2);
      r10 =  s2 * c2i2;
      r11 =  s1 * c1c2 - s3 * s1s2;
      r12 = -s1 * s1c2 - s3 * c1s2;
      r13 =  s4 * s2i2;
      r20 =  s2 * s2i2;
      r21 =  s1 * c1s2 + s3 * s1c2;
      r22 =  s3 * c1c2 - s1 * s1s2;
      r23 = -s4 * c2i2;
      r31 =  s4 * s2i1;
      r32 =  s4 * c2i1;
      r33 =  s3;
    } else { // i1 > pi
      si1 = -si1;
      s2i1 = -s2i1;
      cthp = ctho * cths + stho * sths * ci1;
      if(cthp > 1.)
        cthp = 1.;
      else if(cthp < -1.)
        cthp = -1.;
      sthp = sqrt(1. - SQR(cthp));

      Real ci2, si2;
      if( (sthp != 0.) && (sths != 0.)) {
        si2 = si1 * stho / sthp;
        ci2 = (ctho - cthp * cths)/ (sthp * sths);
      } else {
        if (sths==0.0) {
          si2 = 0.0;
          ci2 = 1.0;
        } else if (sthp==0.0) {
          si2 = si1;
          ci2 = -ci1;
        }
      }

      Real s2i2 = 2.0*si2*ci2;
      Real c2i2 = 2.0*ci2*ci2-1.0;
      Real cdphi = -ci1 * ci2 + si1 * si2 * cths;
      if(cdphi > 1.)
        cdphi = 1.0;
      else if(cdphi < -1.)
        cdphi = -1.;

      phip = phio + acos(cdphi);
      if(phip > 2.*PI) {
        phip = phip - 2.*PI;
      } else if (phip < 0.) {
        phip = phip + 2.*PI;
      }

      Real c1c2 = c2i1 * c2i2;
      Real s1s2 = s2i1 * s2i2;
      Real c1s2 = c2i1 * s2i2;
      Real s1c2 = c2i2 * s2i1;
      //printf("b %g %g\n",c1c2,s1s2);
      // matrix elements
      r10 =  s2 * c2i2;
      r11 =  s1 * c1c2 - s3 * s1s2;
      r12 =  s1 * s1c2 + s3 * c1s2;
      r13 = -s4 * s2i2;
      r20 = -s2 * s2i2;
      r21 = -s1 * c1s2 - s3 * s1c2;
      r22 =  s3 * c1c2 - s1 * s1s2;
      r23 = -s4 * c2i2;
      r31 = -s4 * s2i1;
      r32 =  s4 * c2i1;
      r33 =  s3;

    } // end (i1 < PI)
    //printf("%g %g %g\n",phip,phio,phip-phio);
    //printf("%g %g %g %g\n",r10,r11,r12,r13);
    //printf("%g %g %g %g\n",r20,r21,r22,r23);
    //printf("%g %g %g\n",r31,r32,r33);
    // Calculate new stokes vectors from rotation matrix.  Note that the
    // value of the overall intensity (stokes[0]) does not change
    pphot->sqp[ip] = norm*(r10*stokes[0]+r11*stokes[1]+r12*stokes[2]+r13*stokes[3])/s1;
    pphot->sup[ip] = norm*(r20*stokes[0]+r21*stokes[1]+r22*stokes[2]+r23*stokes[3])/s1;
    pphot->svp[ip] = norm*(r31*stokes[1]+r32*stokes[2]+r33*stokes[3])/s1;

//added stoke 4; svp[ip]


    // Calculate new photon direction
    kx = sthp * cos(phip);
    ky = sthp * sin(phip);
    kz = cthp;

    // SWD: Should be done outside of scattering
    //ppusher->CartesianToCurvalinear(pphot);
  } // end loop over ip
}

//----------------------------------------------------------------------------------------
//! \fn Real Bigy(Real x, Real xp)
//! \brief Helper function for compton scattering functions

Real Bigy(Real x, Real xp)
{
// As defined by Pozdynakov et al. page 324 (see also 201)

  Real r = xp / x;
  Real diff = (1. / x -1. / xp);
  return SQR(r) * (1. / r + r + 4. * diff * (1. + diff));

}

//----------------------------------------------------------------------------------------
//! \fn Real SigmaHat(Real x)
//! \brief Helper function used by Compton scattering routines

Real SigmaHat(Real x) {

  if (x < 0.001) // more accurate at low x
    return 4. * (1. - x * (1. - x * (1.3 - 1.6625 * x))) / 3.;
  else
    return ( (1. - 4. / x * (1. + 2. / x) ) * log(1. + x) +
             0.5 + 8. / x - 0.5 / SQR(1. + x) ) / x;
}

//----------------------------------------------------------------------------------------
//! \fn Real ElectronDistPzdnyakov(Real tgas, MCRandom *pran)
//! \brief Return momentum for electon distribution
//
//  Method is from Pozdnyakov et al. page 317 for low temperaure electrons

Real ElectronDistPozdnyakov(Real tgas, MCRandom *pran) {

  Real kmec2 = 1.68638e-10;
  Real ktgmec2 = kmec2 * tgas;

  if( ktgmec2 <= 0.29) {
    Real xi,nxi,x1,x2,x3;
    do {
      do {
        x1 = pran->uniform();
      } while (x1 == 0.);
      x2 = pran->uniform();
      xi = -1.5 * log(x1);
      nxi = ktgmec2 * xi;
    } while (SQR(x2) >= (0.151 * (1. + nxi) * (1. + nxi) * xi * (2. + nxi) * x1));
    return sqrt(nxi * (2. + nxi));
  } else {
    Real etapp,etap,x1,x2,x3,x4;
    do {
      do {
        x1 = pran->uniform();
        x2 = pran->uniform();
        x3 = pran->uniform();
        x4 = pran->uniform();
      } while((x1*x2*x3*x4) == 0.0);
      etap = -ktgmec2 * log(x1*x2*x3);
      etapp = -ktgmec2 * log(x1*x2*x3*x4);
    } while((SQR(etapp) - SQR(etap)) <= 1.);
    return etap;
  }

}


//----------------------------------------------------------------------------------------
//! \fn Real ElectronDist(Real tgas, MCRandom *pran)
//! \brief Return momentum for electon distribution
//
//  Method is from Canfield et al. 1987, ApJ 323, 565

Real ElectronDist(Real tgas, MCRandom *pran) {

  Real kmec2 = 1.68638e-10;
  Real ktgmec2 = kmec2 * tgas;

  Real pi3 = 0.25 * sqrt(PI);
  Real pi4 = 0.5 * sqrt(0.5 * ktgmec2);
  Real pi5 = 0.375 * sqrt(PI) * ktgmec2;
  Real pi6 = ktgmec2 * sqrt(0.5 * ktgmec2);
  Real S3 = pi3 + pi4 + pi5 + pi6;

  pi3 /= S3;
  pi4 /= S3;
  pi5 /= S3;
  pi6 /= S3;

  Real y, prob;
  do {
    Real dev = pran->uniform();
    Real x;
    if (dev < pi3) {
      x = pran->chisquare(3.);
    } else if (dev < pi3 + pi4) {
      x = pran->chisquare(4.);
    } else if (dev < pi3 + pi4 + pi5) {
      x = pran->chisquare(5.);
    } else {
      x = pran->chisquare(6.);
    }
    // convert x to y
    y = sqrt(0.5 * x);
    prob = sqrt(1. + 0.5 * ktgmec2 * y * y) / (1. + y * sqrt(0.5 * ktgmec2));

  } while (pran->uniform() >= prob);

  // Convert y to dimensionless momentum (beta * gamma)
  Real gamma = 1.+ SQR(y) * ktgmec2;
  Real beta = sqrt(1.-1./SQR(gamma));
  return beta*gamma;

}

//----------------------------------------------------------------------------------------
//! \fn void SampleDipole(Real theta_in, Real phi_in, Real *theta_out, Real *phi_out,
//!                       MCRandom *pran)
//! \brief Sample angular distribution dipole scattering
//
// SWD: ported from Phil's code -- not used and can be removed after testing completed
void SampleDipole(Real theta_in, Real phi_in, Real &theta_out, Real &phi_out,
                  MCRandom *pran){

  // theta_out -> (*theta_out);
  Real f;

  int it = 0;
  do{
    it++;
    // sample theta_out from uniform in mu
    Real x = 2. * pran->uniform() - 1.;
    theta_out = acos( x );

    // sample phi_out from uniform
    phi_out = 2. * PI * pran->uniform();

    // compute distribution
    Real cosgam = sin(theta_in)*sin(theta_out)*cos(phi_in-phi_out) +
                  cos(theta_in)*cos(theta_out);
    f = 1. + SQR(cosgam);

    // test if ok to accept
    if (2. <= f){
      printf("max not big enough\n");
    }
  } while (pran->uniform()*2. > f);

}

//----------------------------------------------------------------------------
//! \fn Real SampleVelocityParallel(Real a, Real x_in, MCRandom *pran)
//! \brief Return parallel velocity for scattering atom

Real SampleVelocityParallel(Real a, Real x_in, MCRandom *pran) {

  Real x = fabs(x_in); // switch sign at end
  Real u0 = 0.96 * x;
  Real th0 = atan((u0-x) / a);
  Real p = (th0 + PI/2.) / (th0 + PI/2. + exp(-SQR(u0)) * (PI/2. - th0));

  Real u,comp;
  do {
    if (pran->uniform()<=p) {
      Real th = -PI/2. + (th0+PI/2.)*pran->uniform();
      u = x + a*tan(th);
      comp = exp(-SQR(u));
    } else {
      Real th = th0 + (PI/2.-th0)*pran->uniform();
      u = x + a*tan(th);
      comp = exp(SQR(u0)-SQR(u));
    }
  } while (pran->uniform() >= comp);

  if (x_in < 0.)
    u = - u;
  return u;
}
