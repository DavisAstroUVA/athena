//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mc_isoth.cpp
//! \brief Problem generator for monte carlo isothermal atmosphere
//
//========================================================================================

// C++ headers
#include <iostream> // SWD: temporary for testing

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../monte_carlo/montecarlo.hpp"
#include "../monte_carlo/photon.hpp"
#include "../monte_carlo/photonpusher.hpp"

#if !MONTE_CARLO_ENABLED
#error "This problem requires monte carlo"
#endif

namespace {
  // Global variables
  bool tnorm;
  Real logemin, logemax;
  Real DensityProfile(Real x, Real xl, Real xh, Real taul, Real tauh, Real kap);
  void JMeanOpacity(MonteCarloBlock *pmcb, Photon *pphot, int ip, int imom,
                  const PhotonFrameState &s);
  void AverageEnergy(MonteCarloBlock *pmcb, Photon *pphot, int ip, int imom,
                  const PhotonFrameState &s);
}


//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief monte carlo test problem generator
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  // Determine density via optical depth or constant density
  bool constdens = pin->GetOrAddBoolean("problem","constdens",false);
  Real rho, tau, taumin, taumax;
  if (constdens) {
    rho = pin->GetOrAddReal("problem","dens",-1.);
    tau = pin->GetOrAddReal("problem","tau",-1.);
  } else {
    taumin = pin->GetReal("problem","taumin");
    taumax = pin->GetReal("problem","taumax");
  }

  Real heabund = 0.09; //hardcode for now
  Real mp = 1.6726e-24;
  Real sigmat = 6.65248e-25;
  Real kappaes = sigmat * (1. + 2.*heabund) / (mp * (1.+4.*heabund) );
  if (tau > 0.) {
    Real length;
    if (COORDINATE_SYSTEM == "cartesian") {
      Real xlow = pin->GetReal("mesh","x3min");
      Real xhigh = pin->GetReal("mesh","x3max");
      length = xhigh-xlow;
    } else {
      length = pin->GetReal("mesh","x1max");
    }
    rho = tau / (kappaes * length);
  }

  Real xlow, xhigh;
  if (!constdens) {
    if (COORDINATE_SYSTEM == "cartesian") {
      xlow = pin->GetReal("mesh","x3min");
      xhigh = pin->GetReal("mesh","x3max");
    } else {
      bool radial = pin->GetOrAddBoolean("problem","radial","true");
      if (radial) {
        xlow = pin->GetReal("mesh","x1min");
        xhigh = pin->GetReal("mesh","x1max");
      } else {
        // Stratified in z = r cos(theta) rather than in r, so the profile is measured from
        // the midplane out to zmax.  The disk is symmetric about z = 0, so only |z| is
        // ever passed to DensityProfile and the lower bound is the midplane itself.
        xlow = 0.0;
        xhigh = pin->GetReal("problem","zmax");
      }
    }
  }

  // Assume constant velocity provided as fraction of speed of light
  Real vel = pin->GetOrAddReal("problem","velocity",0.);
  Real c = 2.99792458e10;
  vel *= c;

  // Assume constant temperature and ideal gas
  Real gamma = peos->GetGamma();
  Real rideal = 8.314e7;
  Real tgas = pin->GetReal("problem","temp");
  // Set initial conditions
  if (COORDINATE_SYSTEM == "cartesian") {
    // density varies in the z direction

    for (int k=ks; k<=ke; k++) {
      Real x1 = pcoord->x3v(k);
      if (!constdens) rho = DensityProfile(x1,xlow,xhigh,taumin,taumax,kappaes);
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          phydro->u(IDN,k,j,i) = rho;
          phydro->u(IM1,k,j,i) = 0.0;
          phydro->u(IM2,k,j,i) = 0.0;
          phydro->u(IM3,k,j,i) = rho*vel;
          phydro->u(IEN,k,j,i) = rideal*rho*tgas/(gamma-1.0);
        }
      }
    }
  } else if  (COORDINATE_SYSTEM == "spherical_polar") {
    bool radial = pin->GetOrAddBoolean("problem","radial","true");

    if (radial) {
      // density varies in the r direction
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
          for (int i=is; i<=ie; i++) {
            Real x1 = pcoord->x1v(i);
            if (!constdens) rho = DensityProfile(x1,xlow,xhigh,taumin,taumax,kappaes);
            phydro->u(IDN,k,j,i) = rho;
            phydro->u(IM1,k,j,i) = rho*vel;
            phydro->u(IM2,k,j,i) = 0.0;
            phydro->u(IM3,k,j,i) = 0.0;
            phydro->u(IEN,k,j,i) = rideal*rho*tgas/(gamma-1.0);
          }
        }
      }
    } else {
      // Vertically stratified disk: the density depends only on z = r cos(theta), so every
      // cylindrical radius carries the same plane-parallel atmosphere and the emergent
      // polarization can be compared against a plane-parallel solution.
      //
      // Above |z| = zmax the exponential simply continues, which is the right continuation
      // for an isothermal atmosphere -- there is nothing up there.  Clamping the argument
      // at zmax instead, so that everything above it carries the full surface density, is
      // wrong in a way that is easy to miss: the atmosphere top is a plane (z = zmax) but
      // the domain top is a cone (constant theta), and the wedge between them grows with
      // radius.  Filled with surface-density material it is optically thin vertically
      // (tau ~ 0.02) but not along a grazing ray (tau ~ 0.26 at mu = 0.06), so it
      // contaminates the most grazing angle bin only, which is also the bin carrying most
      // of the polarization signal.
      Real rhofloor = pin->GetOrAddReal("problem","rhofloor",0.0);
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
          Real cth = std::cos(pcoord->x2v(j));
          for (int i=is; i<=ie; i++) {
            Real zabs = std::fabs(pcoord->x1v(i)*cth);
            if (!constdens) {
              rho = DensityProfile(zabs,xlow,xhigh,taumin,taumax,kappaes);
              if (rho < rhofloor) rho = rhofloor;
            }
            phydro->u(IDN,k,j,i) = rho;
            phydro->u(IM1,k,j,i) = rho*vel;
            phydro->u(IM2,k,j,i) = 0.0;
            phydro->u(IM3,k,j,i) = 0.0;
            phydro->u(IEN,k,j,i) = rideal*rho*tgas/(gamma-1.0);
          }
        }
      }
    }
  }
  // add kinetic energy
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        phydro->u(IEN,k,j,i) += 0.5*SQR(phydro->u(IM1,k,j,i))/phydro->u(IDN,k,j,i);
        phydro->u(IEN,k,j,i) += 0.5*SQR(phydro->u(IM2,k,j,i))/phydro->u(IDN,k,j,i);
        phydro->u(IEN,k,j,i) += 0.5*SQR(phydro->u(IM3,k,j,i))/phydro->u(IDN,k,j,i);
      }}}
}

//========================================================================================
//! \fn void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe)
//! \brief Initializes Photon packets before integration
//========================================================================================

void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype) {

  // Set initial cells and emission weights for all photon samples
  BoundaryFace face;
  if (pmy_mc->emission_flag == EMISFF) {
      SetEmissionCellWeight(pphot,ips,ipe);
  } else if (pmy_mc->emission_flag == EMISBB) {
    face = pmy_mc->emission_face[0];
    SetEmissionCellWeightArea(pphot,face,ips,ipe);
  }

  for (int ip=ips; ip<=ipe; ip++) {

    // Set maximum integration time
    pphot->dtp[ip] = pphot->pmy_mcb->pmy_mc->tmax;

    if (pmy_mc->emission_flag == EMISFF) {
      // Obtain initial position within zone
        GetZonePosition(pphot,pran,pcoord,ip);

      // Obtain intitial energy, polarization, direction and weight
      // Utilize free-free emission function in emission.cpp
      if(tnorm) {
        Real logtg = log(tgas(pphot->i3p[ip],pphot->i2p[ip],pphot->i1p[ip]));
        PhotonEmitFreeFree(this,pphot,logemin+logtg,logemax+logtg,ip);
      } else{
        PhotonEmitFreeFree(this,pphot,logemin,logemax,ip);
      }
    } else if (pmy_mc->emission_flag == EMISBB) {
      // Obtain initial position within zone on surface
       GetZonePositionCartesianFace(pphot,pran,pcoord,face,ip);

      PhotonEmitBlackbody(this,pphot,face,ip);
    }

    // Convert k unit vector to k^\alpha
    /*if (pmy_mc->general_pusher_flag) {
      pphot->k0p[ip] = 1.;
      pphot->k2p[ip] /= pphot->x1p[ip];
      pphot->k3p[ip] /= (pphot->x1p[ip]*sin(pphot->x2p[ip]));
      pphot->dk0p[ip] = 0.;
      pphot->dk1p[ip] = 0.;
      pphot->dk2p[ip] = 0.;
      pphot->dk3p[ip] = 0.;
      }*/

    //printf("%g %g \n",pphot->ep[ip],pphot->wp[ip]);
    // Set status flag
    if (pphot->wp[ip] < 0.0)
      pphot->statp[ip] = DESTROYED;
    else
      pphot->statp[ip] = EVOLVING;

    // initialize scattering number
    pphot->nscp[ip] = 0;

    // Initialize the absorption and scattering extinction coefficients
    // to the values appropriate in the emitted zone
    pphot->acp[ip] = AbsorptionOpacity(this,pphot,ip);
    pphot->scp[ip] = ScatteringOpacity(this,pphot,ip);

    //pphot->PrintPhoton("start",ip);
    //printf("start: %d %g %g %d\n",pphot->i3p[ip],pphot->wp[ip],pphot->ep[ip],pphot->statp[ip]);
  }
  //pphot->nphot++;

}

//========================================================================================
//! \fn void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin)
//! \brief Analogous to problem generator but used in support of InitializePhoton
//========================================================================================

void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin) {


  // Set the energy boundaries for free-free emission
  tnorm = pin->GetOrAddBoolean("problem","tnorm",false);
  if (tnorm) {
    // interpret as xmin/xmax with x=E/(kb*T)
    const Real kb = 1.380649e-16;
    logemin = log(kb*pin->GetReal("problem", "emin"));
    logemax = log(kb*pin->GetReal("problem", "emax"));
  } else {
    // interpret as emin/emax in eV
    const Real everg = 1.6021772e-12;
    logemin = log(everg*pin->GetReal("problem", "emin"));
    logemax = log(everg*pin->GetReal("problem", "emax"));
  }

}

//========================================================================================
//! \fn void MonteCarlo::InitUserMonteCarloData(ParameterInput *pin)
//! \brief Initializes user data specific to MonteCarlo class
//========================================================================================

void MonteCarlo::InitUserMonteCarloData(ParameterInput *pin){

  nuser_var = 1;
  AllocateUserMoments(2);
  EnrollUserMoment(0, JMeanOpacity, "kapJ");
  EnrollUserMoment(1, AverageEnergy, "eave");
}

namespace {
Real DensityProfile(Real x, Real xl, Real xh, Real taul, Real tauh, Real kap) {

  Real l0 = (xh-xl) / log(tauh/taul);
  return taul/l0/kap*exp((xh-x)/l0);
}

void JMeanOpacity(MonteCarloBlock *pmcb, Photon *pphot, int ip, int imom,
                  const PhotonFrameState &s) {

  int i1 = pphot->i1p[ip];
  int i2 = pphot->i2p[ip];
  int i3 = pphot->i3p[ip];

  // energy and path length come from the frame this moment was enrolled with
  Real weight = pphot->wp[ip]*s.e*s.dl/MCConstants::c_cgs;
  pmcb->moments_user(imom,i3,i2,i1) += weight*pphot->acp[ip];

}

void AverageEnergy(MonteCarloBlock *pmcb, Photon *pphot, int ip, int imom,
                  const PhotonFrameState &s) {

  int i1 = pphot->i1p[ip];
  int i2 = pphot->i2p[ip];
  int i3 = pphot->i3p[ip];

  // energy and path length come from the frame this moment was enrolled with
  Real weight = pphot->wp[ip]*s.e*s.dl/MCConstants::c_cgs;
  pmcb->moments_user(imom,i3,i2,i1) += weight*s.e;

}

}
