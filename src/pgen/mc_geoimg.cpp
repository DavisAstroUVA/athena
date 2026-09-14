//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mc_geoimg.cpp
//  \brief Problem generator for creating an image with geodesics in kerr spacetime
//
//========================================================================================

#include <iostream> // temporary for testing

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../monte_carlo/montecarlo.hpp"
#include "../monte_carlo/tetrad.hpp"
#include "../monte_carlo/photon.hpp"
#include "../monte_carlo/photonpusher.hpp"
#include "../globals.hpp"

#if !MONTE_CARLO_ENABLED
#error "This problem requires monte carlo"
#endif

namespace {

  int iphot;
  int nrays;
  int i1start, i2start, i3start;
  int nalpha, nbeta;
  Real *alpha,*beta;
  Real rcam,thcam,phcam;
  Real rh, rdisk;
  bool forward_integration;

  Real polch[21] = {0.11713,0.08979,0.07448,0.06311,0.05410,0.04667,0.04041,0.03502,
                    0.03033,0.02619,0.02252,0.01923,0.01627,0.01358,0.011123,0.008880,
                    0.006818,0.004919,0.003155,0.001522,0};
  Real much[21] = {0.,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,
                   0.75,0.8,0.85,0.9,0.95,1.};

  // User function definitions
  void MidplaneCrossing(MonteCarloBlock *pmcb, Photon *pphot, PhotonPusher *ppusher,int ip);
  void GetDirectionKerrtrans(Photon *pphot, Real alpha, Real beta, int ip);
  void GetDirectionTetrad(Photon *pphot, Real alpha, Real beta, int ip);
  void TransformPhotonAtDisk(MonteCarloBlock *pmcb, Photon *pphot, int ip);
  void TransformPhotonAtGridEdge(MonteCarloBlock *pmcb, Photon *pphot, int ip);
  void BuildImageArrayUniform(int nx, Real xmin, Real xmax, int ny, Real ymin, Real ymax,
                              Real *x, Real *y);

}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief monte carlo test problem generator
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  Real rideal = 8.314e7;
  Real rho = 1.;
  Real temp = 1.;
  Real gamma = peos->GetGamma();


  // Set nominal values for grid, unused
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        phydro->u(IDN,k,j,i) = rho;
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        phydro->u(IEN,k,j,i) = rideal*rho*temp/(gamma-1.0);
      }
    }
  }

}

//========================================================================================
//! \fn void MonteCarlo::InitUserMonteCarloData(ParameterInput *pin)
//! \brief Initializes user data specific to MonteCarlo class
//========================================================================================

void MonteCarlo::InitUserMonteCarloData(ParameterInput *pin){

  // Enroll four user variables
  nuser_var = 6;
  // Enroll function for determining plane crossing
  EnrollUserWorkInMove(MidplaneCrossing);

  // Determine if camera is on the is block
  /*rcam = pin->GetOrAddReal("problem", "rcam", 0.9999*pin->GetReal("mesh","x1max"));
  thcam = pin->GetOrAddReal("problem", "thcam", 45.) * M_PI / 180.;
  phcam = pin->GetOrAddReal("problem", "phcam", 90.) * M_PI / 180.;
  printf("%g %g %g\n",rcam,thcam,phcam);
  int cam = 0;
  printf("nbl: %d\n",nblocal);
  for (int i=0; i<nblocal; i++) {
    MonteCarloBlock *pmcb = my_blocks(i);
    MeshBlock *pmb = pmcb->pmy_block;
    if ( (rcam > pmb->block_size.x1min) && (rcam <= pmb->block_size.x1max) &&
         (thcam > pmb->block_size.x2min) && (thcam <= pmb->block_size.x2max) &&
         (phcam > pmb->block_size.x3min) && (phcam <= pmb->block_size.x3max) ) {
      cam = 1;
      int nx = pin->GetInteger("problem", "nx");
      int ny = pin->GetInteger("problem", "ny");
      nsamp = nx * ny;
      printf("y %d\n",Globals::my_rank);
    } else {
      printf("n %d\n",Globals::my_rank);
      nsamp = 0;
    }
  }

#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE,&cam,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
#endif

  if (cam < 1) {
    std::stringstream msg;
    msg << "### FATAL ERROR in InitUserMonteCarloData" << std::endl
        << "Camera not found on any block." << std::endl;
    ATHENA_ERROR(msg);
  } else if (cam > 1) {
    std::stringstream msg;
    msg << "### FATAL ERROR in InitUserMonteCarloData" << std::endl
        << "Camera found on multiple blocks." << std::endl;
    ATHENA_ERROR(msg);
    }*/

}

//========================================================================================
//! \fn void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin)
//! \brief Analogous to problem generator but used in support of InitializePhoton
//========================================================================================

void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin) {


  // Set rh
  Real abh = pcoord->GetSpin();
  Real mbh = pcoord->GetMass();
  rh = 1.0 + sqrt(1.0 - SQR(abh));

  forward_integration = pin->GetOrAddBoolean("problem","forward",false);

  // set outer disk radius
  rdisk = pin->GetOrAddReal("problem", "rdisk", 1.e20);

  // Determine if camera is on the is block
  rcam = pin->GetOrAddReal("problem", "rcam", 0.9999*pin->GetReal("mesh","x1max"));
  thcam = pin->GetOrAddReal("problem", "thcam", 45.) * M_PI / 180.;
  phcam = pin->GetOrAddReal("problem", "phcam", 90.) * M_PI / 180.;
  MeshBlock *pmb = pmy_block;
  if ( (rcam > pmb->block_size.x1min) && (rcam <= pmb->block_size.x1max) &&
       (thcam > pmb->block_size.x2min) && (thcam <= pmb->block_size.x2max) &&
       (phcam > pmb->block_size.x3min) && (phcam <= pmb->block_size.x3max) ) {
    nalpha = pin->GetInteger("problem", "nx");
    nbeta = pin->GetInteger("problem", "ny");
    // reset nphremain
    nphremain = nalpha * nbeta;
  } else {
    nphremain = 0;
    return;
  }

  Real amin = pin->GetOrAddReal("problem", "alpha_min", -10.);
  Real amax = pin->GetOrAddReal("problem", "alpha_max", 10.);
  Real bmin = pin->GetOrAddReal("problem", "beta_min", -10.);
  Real bmax = pin->GetOrAddReal("problem", "beta_max", 10.);

  alpha = new Real[nalpha];
  beta = new Real[nbeta];
  if (nalpha == 1)
    amax = amin;
  if (nbeta == 1)
    bmax = bmin;
  BuildImageArrayUniform(nalpha,amin,amax,nbeta,bmin,bmax,alpha,beta);

  iphot = 0;

  // set the photon samples 's initial zone indices
  Real r = rcam * 0.999999;
  Real theta = thcam;
  Real phi = phcam + 1.0e-3;
  MCCoord *pco = pcoord;
  i1start = -1;
  for(int i=is; i<=ie; i++) {
    if ((r >= pcoord->x1f(i)) && (r < pcoord->x1f(i+1)))
      i1start = i;
  }
  i2start = -1;
  for(int i=js; i<=je; i++) {
    if ((theta >= pcoord->x2f(i)) && (theta < pcoord->x2f(i+1)))
      i2start = i;
  }
  i3start = -1;
  for(int i=ks; i<=ke; i++) {
    if ((phi >= pcoord->x3f(i)) && (phi < pcoord->x3f(i+1)))
      i3start = i;
  }
  if ((i1start < 0) || (i2start < 0) || (i3start < 0)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in InitUserMonteCarloBlockData" << std::endl
        << "Initial position not found within domain." << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

}

//========================================================================================
//! \fn void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype)
//! \brief Initializes Photon packets before integration
//========================================================================================

void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype) {

  MCCoord *pco = pphot->pmy_mcb->pcoord;

  for (int ip=ips; ip<=ipe; ip++) {
    // Set status flag
    pphot->statp[ip] = EVOLVING;
    pphot->wp[ip] = 1.;

    pphot->dtp[ip] = HUGE_NUMBER;

    // initialize cell coordinates
    pphot->i1p[ip] = i1start;
    pphot->i2p[ip] = i2start;
    pphot->i3p[ip] = i3start;

    // Set alpha, beta by iterating over photons
    int ia,ib;
    ia = iphot / nalpha;
    ib = iphot % nalpha;
    pphot->user[5][ip] = static_cast<Real>(iphot);
    iphot++;

    Real alpha0, beta0;
    alpha0 = alpha[ia];
    beta0 = beta[ib];
    pphot->user[0][ip] = alpha0;
    pphot->user[1][ip] = beta0;

    // Initialize position
    // SWD: remove some of this
    Real r = rcam * 0.999999;
    Real theta = thcam;
    Real phi = phcam + 1.0e-3;

    Real x[NCOORD];
    x[IMC0] = pphot->x0p[ip] = 0.;
    x[IMC1] = pphot->x1p[ip] = r;
    x[IMC2] = pphot->x2p[ip] = theta;
    x[IMC3] = pphot->x3p[ip] = phi;

    // Initialize Stokes vector as unpolarized
    if (IsPolarized(pphot->polarized)) {
      pphot->sip[ip] = 1.0;
      pphot->sqp[ip] = 0.0;
      pphot->sup[ip] = 0.0;
      pphot->svp[ip] = 0.0;
    }

    // Set the initial photon direction using alpha, beta and the position

    GetDirectionTetrad(pphot, alpha0, beta0, ip);

    pphot->ep[ip] = -pphot->k0p[ip];

    // Initialize dk to zero
    pphot->dk0p[ip] = 0;
    pphot->dk1p[ip] = 0;
    pphot->dk2p[ip] = 0;
    pphot->dk3p[ip] = 0;

    // Set plane crossing flag to zero
    pphot->user[4][ip] = 0.;

    // Initialize the absorption and scattering extinction coefficients
    // to the values appropriate in the emitted zone
    pphot->acp[ip] = 0.;
    pphot->scp[ip] = 0.;

  } // loop over ip
}

//========================================================================================
//! \fn void MonteCarloBlock::FinalizePhoton(Photon *pphot, int ip)
//! \brief Complete work at end of photon packets before integration
//========================================================================================

void MonteCarloBlock::FinalizePhoton(Photon *pphot, int ip) {

  if (pphot->statp[ip] == DESTROYED) {
    pphot->statp[ip] = ESCAPED;
    return;
  }

  if (forward_integration) {
    TransformPhotonAtGridEdge(this,pphot,ip);
  } else {
    TransformPhotonAtDisk(this,pphot,ip);
  }

}

namespace {

void TransformPhotonAtDisk(MonteCarloBlock *pmcb, Photon *pphot, int ip) {

  // If r is inside ISCO, do not attempt to transform and instead mark
  // as escaped
  if (pphot->x1p[ip] < rh + 1.0e-5) {
    pphot->statp[ip] = ESCAPED;
    pphot->user[2][ip] = 0.;
    pphot->user[3][ip] = rh;
    return;
  }

  // SWD may not need full tetrad here unless presribed angular dependence
  // to intensity

  // Construct the orthonormal tetrad in comoving frame of circular orbit
  Real gcov[4][4];
  Real x[4];
  x[IMC0] = pphot->x0p[ip];
  x[IMC1] = pphot->x1p[ip];
  x[IMC2] = pphot->x2p[ip];
  x[IMC3] = pphot->x3p[ip];
  pmcb->pcoord->Metric(x, gcov);

  Real abh = pmcb->pcoord->GetSpin();
  Real mbh = pmcb->pcoord->GetMass();
  Real r = x[IMC1];
  Real omega = pow(mbh,0.5)/(pow(r, 3./2.) + abh*pow(mbh,3./2.)); // circular velocity
  Real ucon[4];
  ucon[IMC0] = sqrt(-1.0/(gcov[IMC0][IMC0] + 2.*gcov[IMC0][IMC3]*omega +
                            SQR(omega)*gcov[IMC3][IMC3]));
  ucon[IMC1] = 0.;
  ucon[IMC2] = 0.;
  ucon[IMC3] = (ucon[IMC0])*omega;

  // create tetrad basis
  Real econ[4][4], ecov[4][4];
  ConstructTetrad(ucon, gcov, econ, ecov);

  // Reverse photon direction to get properties of photon that was emitted
  pphot->k0p[ip] *= -1.;
  pphot->k1p[ip] *= -1.;
  pphot->k2p[ip] *= -1.;
  pphot->k3p[ip] *= -1.;

  //  Transform to comoving tetrad
  Real kcopy[4];
  kcopy[IMC0] = pphot->k0p[ip];
  kcopy[IMC1] = pphot->k1p[ip];
  kcopy[IMC2] = pphot->k2p[ip];
  kcopy[IMC3] = pphot->k3p[ip];
  Real k[4];
  CoordinateToTetrad(kcopy, k, ecov);
  k[IMC0] = pphot->k0p[ip];
  k[IMC1] = pphot->k1p[ip];
  k[IMC2] = pphot->k2p[ip];
  k[IMC3] = pphot->k3p[ip];
  pphot->user[2][ip] = pphot->k0p[ip];
  // Get radius at disk crossing
  pphot->user[3][ip] = pphot->x1p[ip];

}

void TransformPhotonAtGridEdge(MonteCarloBlock *pmcb, Photon *pphot, int ip) {

  if (IsPolarized(pphot->polarized)) {
    // Construct the orthonormal tetrad at edge of simulation grid
    Real ucon[4];
    ucon[IMC0] = 1.;
    ucon[IMC1] = 0.;
    ucon[IMC2] = 0.;
    ucon[IMC3] = 0.;

    // create tetrad basis
    Real gcov[4][4], gcon[4][4];
    Real x[4];
    x[IMC0] = pphot->x0p[ip];
    x[IMC1] = pphot->x1p[ip];
    x[IMC2] = pphot->x2p[ip];
    x[IMC3] = pphot->x3p[ip];
    pmcb->pcoord->Metric(x, gcov);
    pmcb->pcoord->InverseMetric(x,gcon);
    Real wcon[4] = {0.,1.,0.,0.};
    Real vcov[4] = {0.,1.,0.,0.};
    //Real wcon[4] = {0,1.,0.,0.}; // Q=1 points along projected BH symmetry axis
    //Real vcov[4] = {1.,0.,0.,1.};// Make image center point away from origin
    Real vcon[4];
    CovToCon(vcov,vcon,gcon);
    Real econ[4][4], ecov[4][4];
    ConstructTetrad(ucon, vcon, wcon, gcov, econ, ecov);

    // Get stokes parameters
    std::complex<Real> tcopy[4][4];
    pphot->PolarizationToTetrad(tcopy,ecov,ip);
    Real stokes[4];

    TensorToStokes(tcopy,stokes);
    pphot->sip[ip] = stokes[0];
    pphot->sqp[ip] = stokes[1];
    pphot->sup[ip] = stokes[2];
    //pphot->PrintPhoton("finalize photon",ip);
  }
}


// Given initial position x^alpha, alpha, beta, determine the initial photon direction
// Uses alpha, beta definitions from Cunningham & Bardeen (1973)
void GetDirectionKerrtrans(Photon *pphot, Real alpha, Real beta, int ip) {

  MonteCarloBlock *pmcb = pphot->pmy_mcb;

  // calculate g_{alpha,beta}, g^{a,b}
  Real x[4];
  x[IMC0] = pphot->x0p[ip];
  x[IMC1] = pphot->x1p[ip];
  x[IMC2] = pphot->x2p[ip];
  x[IMC3] = pphot->x3p[ip];
  Real gcon[4][4], gcov[4][4];
  pmcb->pcoord->Metric(x, gcov);
  pmcb->pcoord->InverseMetric(x, gcon);

  // kcov is set by alpha, beta, but kcon is what is integrated
  // assumes initial radius r_0 >> alpha, beta
  Real kcov[4]; //kcov = k_alpha
  kcov[IMC0] = 1.0; // The goal is to have k^t [coordinate frame] to be 1.0 at emission
  kcov[IMC3] = alpha * kcov[IMC0] * sin(x[IMC2]);
  kcov[IMC2] = beta * kcov[IMC0];

  // use null geodesic equation to determine k_r: k_a * g^(a,b) * k_b = 0
  // solve the quadratic equation k_r^2 * gamma + k_r * zeta + xi = 0
  Real gamma = gcon[IMC1][IMC1];
  Real zeta = kcov[IMC0] * 2. * (gcon[IMC1][IMC0] + gcon[IMC1][IMC2] * beta +
                                 gcon[IMC1][IMC3] * alpha * sin(x[IMC2]));
  Real xi = SQR(kcov[IMC0]) * (gcon[IMC0][IMC0] + 2. * gcon[IMC0][IMC2] * beta +
                               2. * gcon[IMC0][IMC3] * alpha * sin(x[IMC2])
                               + gcon[IMC2][IMC2] * SQR(beta)
                               + gcon[IMC3][IMC3] * SQR(alpha) * SQR(sin(x[IMC2])) +
                               2. * gcon[IMC2][IMC3] * alpha * beta * sin(x[IMC2]));
  Real sqrtdis = sqrt(SQR(zeta) - 4. * gamma * xi);
  Real plus = (-zeta + sqrtdis) / (2. * gamma);
  Real minus = (-zeta - sqrtdis) / (2. * gamma);

  // choose the positive root since dividing by k^t (which is < 0) switches sign
  if ((fabs(plus) < 1.0e-20) && (fabs(minus) < 1.0e-20)) {
    printf("WARNING: both roots of k_r = 0: %g  %g\n", plus, minus);
    pphot->statp[ip] = DESTROYED;
  } else if ((plus < 0) && (minus > 0)) {
    kcov[IMC1] = minus;
  } else if ((plus > 0) && (minus < 0)) {
    kcov[IMC1] = plus;
  } else if ((plus < 0) && (minus < 0)) {
    printf("WARNING: both roots of k_r < 0: %g  %g\n", plus, minus);
    pphot->statp[ip] = DESTROYED;
  } else if ((plus > 0) && (minus > 0)) {
    printf("WARNING: both roots of k_r > 0: %g  %g\n", plus, minus);
    pphot->statp[ip] = DESTROYED;
  }

  // now k_alpha is set, so raise to k^alpha
  Real kcon[4]; // kcon = k^alpha
  CovToCon(kcov, kcon, gcon); // converts k_alpha to k^alpha
  // normalize k^alpha such that k^t = 1
  pphot->k0p[ip] = 1.;
  pphot->k1p[ip] = kcon[IMC1] / kcon[IMC0];
  pphot->k2p[ip] = kcon[IMC2] / kcon[IMC0];
  pphot->k3p[ip] = kcon[IMC3] / kcon[IMC0];

}

// Given initial position x^alpha, alpha, beta, determine the initial photon direction
// Equivalent to Cunningham & Bardeen implementation up to minus signs in alpha, beta
void GetDirectionTetrad(Photon *pphot, Real alpha, Real beta, int ip) {

  // Set metric components
  MCCoord *pcoord = pphot->pmy_mcb->pcoord;

  Real x[4];
  x[IMC0] = pphot->x0p[ip];
  x[IMC1] = pphot->x1p[ip];
  x[IMC2] = pphot->x2p[ip];
  x[IMC3] = pphot->x3p[ip];
  Real gcov[4][4], gcon[4][4];
  pcoord->Metric(x, gcov);
  pcoord->InverseMetric(x,gcon);

  // Set tetrad vector for camera
  Real ucon[4] = {1.,0.,0.,0.}; // Static observer
  Real wcon[4] = {0.,0,-1.,0.}; // Q=1 points along projected BH symmetry axis
  Real vcov[4] = {1.,1.,0.,0.};// Make image center point away from origin
  //Real ucon[4] = {0.,0.,0.,1.}; // Static observer
  //Real wcon[4] = {0,-1.,0.,0.}; // Q=1 points along projected BH symmetry axis
  //Real vcov[4] = {1.,0.,0.,1.};// Make image center point away from origin
  Real vcon[4];
  CovToCon(vcov,vcon,gcon);

  // Construct tetrad
  Real econ[4][4], ecov[4][4];
  ConstructTetrad(ucon, vcon, wcon, gcov, econ, ecov);

  // Construct k in the tetrad
  Real ktet[4];
  Real kx = alpha / x[IMC1];
  Real ky = beta / x[IMC1];
  Real knorm = sqrt(1.+SQR(kx)+SQR(ky));
  ktet[IMC0] = -1.; // Photon is moving backward in time
  ktet[IMC1] = kx / knorm;
  ktet[IMC2] = ky / knorm;
  ktet[IMC3] = -1. / knorm; // points along radial direction

  Real k[4];
  TetradToCoordinate(ktet,k,econ);
  pphot->k0p[ip] = k[IMC0];
  pphot->k1p[ip] = k[IMC1];
  pphot->k2p[ip] = k[IMC2];
  pphot->k3p[ip] = k[IMC3];

}

void MidplaneCrossing(MonteCarloBlock *pmcb, Photon *pphot, PhotonPusher *ppusher, int ip) {

 // check if photon has crossed midplane and whether to terminate or keep integrating
  pphot->PrintPhoton("in move",ip);
  if (pphot->x1p[ip] < rh + 1.0e-5) {
    pphot->statp[ip] = DESTROYED;
    pphot->user[2][ip] = 0.;
    pphot->user[3][ip] = rh;
    if (IsPolarized(pphot->polarized)) {
      pphot->sqp[ip] = 0.;
      pphot->sup[ip] = 0.;
    }
    return;
  }

  bool reverse = false;

  if (pphot->user[4][ip] < 1.) { // photon has not yet crossed the plane
      if (pphot->x2p[ip] >= (M_PI / 2.0)) { // photon has crossed plane for the first time
        if (pphot->x1p[ip] <= rdisk) { // photon is less than outer disk radius
          Real step = -(pphot->x2p[ip] - M_PI/2.0) / pphot->k2p[ip];
          pphot->x0p[ip] += pphot->k0p[ip] * step;
          pphot->x1p[ip] += pphot->k1p[ip] * step;
          pphot->x2p[ip] += pphot->k2p[ip] * step;
          pphot->x3p[ip] += pphot->k3p[ip] * step;
          if (forward_integration) {
            reverse = true;
          } else {
            pphot->statp[ip] = ESCAPED;
          }
        } else {
          // keep integrating
          pphot->user[4][ip] += 1.;
        }
      }
  } else if (static_cast<int>(pphot->user[4][ip]) % 2 == 1) {
    // photon has crossed plane an odd number
    if (pphot->x2p[ip] <= (M_PI / 2.0)) { //new crossing
      if (pphot->x1p[ip] <= rdisk) {
        if (forward_integration)
          reverse = true;
        else
          pphot->statp[ip] = ESCAPED;
      } else {
        // keep integrating
        pphot->user[4][ip] += 1.;
      }
    }
  } else if (static_cast<int>(pphot->user[4][ip]) % 2 == 0) {
    // photon has crossed plane an evern number
    if (pphot->x2p[ip] <= (M_PI / 2.0)) { // new crossing
      if (pphot->x1p[ip] <= rdisk) {
        if (forward_integration)
          reverse = true;
        else
          pphot->statp[ip] = ESCAPED;
      } else {
        // keep integrating
        pphot->user[4][ip] += 1.;
      }
    }
  }


  if (reverse) {

    // Transform from tetrad frame to comoving frame, initialize stokes, and record
    // emission energy of photon in comoving frame

    Real x[4];
    x[IMC0] = pphot->x0p[ip];
    x[IMC1] = pphot->x1p[ip];
    x[IMC2] = pphot->x2p[ip];
    x[IMC3] = pphot->x3p[ip];
    Real gcov[4][4];
    pmcb->pcoord->Metric(x, gcov);

    Real r = x[IMC1];
    Real a = pmcb->pcoord->GetSpin();
    Real mbh = pmcb->pcoord->GetMass();
    Real omega = pow(mbh,0.5)/(pow(r, 3./2.) + a*pow(mbh,3./2.));
    // Initialize ucon and vcon (= z unit vector in symmetry plane)
    Real ucon[4];
    ucon[IMC0] = sqrt(-1.0/(gcov[IMC0][IMC0] + 2.*gcov[IMC0][IMC3]*omega +
                            SQR(omega)*gcov[IMC3][IMC3]));
    ucon[IMC1] = 0.;
    ucon[IMC2] = 0.;
    ucon[IMC3] = (ucon[IMC0])*omega;
    Real vcon[4];
    vcon[IMC0] = 0.;
    vcon[IMC1] = 0.;
    vcon[IMC2] = -1.;
    vcon[IMC3] = 0.;

    // Reverse photon direction
    pphot->k0p[ip] *= -1.;
    pphot->k1p[ip] *= -1.;
    pphot->k2p[ip] *= -1.;
    pphot->k3p[ip] *= -1.;
    pphot->dk0p[ip] *= -1.;
    pphot->dk1p[ip] *= -1.;
    pphot->dk2p[ip] *= -1.;
    pphot->dk3p[ip] *= -1.;

    // create tetrad basis
    Real k[4];
    k[IMC0] = pphot->k0p[ip];
    k[IMC1] = pphot->k1p[ip];
    k[IMC2] = pphot->k2p[ip];
    k[IMC3] = pphot->k3p[ip];
    Real econ[4][4], ecov[4][4];
    ConstructTetrad(ucon, k, vcon, gcov, econ, ecov);

    // Get photon energy in rest frame
    Real kcopy[4];
    CoordinateToTetrad(k, kcopy, ecov);
    pphot->user[2][ip] = kcopy[IMC0];
    // Get radius at disk crossing
    pphot->user[3][ip] = x[IMC1];
    // set plane crossing to zero
    pphot->user[4][ip] = 0.;

    if (IsPolarized(pphot->polarized)) {
      // Initialize and transform Stokes vector
      Real stokes[4];
      int ipol = static_cast<int>(kcopy[IMC2]*20.);
      Real aint = kcopy[IMC3] - much[ipol];
      Real frac = polch[ipol]*(1.-aint) + polch[ipol+1]*aint;
      stokes[0] = 1.0;
      stokes[1] = frac;
      stokes[2] = 0.0;
      stokes[3] = 0.0;
      std::complex<Real> tcopy[4][4];
      StokesToTensor(stokes,tcopy);
      pphot->PolarizationToCoord(tcopy,econ,ip);
    //pphot->PrintPhoton("midlplane crossing",ip);
    }
  } // if (reverse)
}

void BuildImageArrayUniform(int nx, Real xmin, Real xmax, int ny, Real ymin, Real ymax,
                            Real *x , Real *y) {

  // Build Uniform grid in x and y
  Real dx = (xmax-xmin) / static_cast<Real>(nx);
  for (int i = 0; i<nx; i++)
    x[i] = xmin+dx*(0.5+i);
  Real dy = (ymax-ymin) / static_cast<Real>(ny);
  for (int i = 0; i<ny; i++)
    y[i] = ymin+dy*(0.5+i);

}

} // namespace
