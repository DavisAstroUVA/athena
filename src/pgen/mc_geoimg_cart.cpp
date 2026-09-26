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
  Real abh, mbh;
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

  void CartesianKerrSchild(Real x, Real y, Real z, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx,
                         AthenaArray<Real> &dg_dy, AthenaArray<Real> &dg_dz);
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief monte carlo test problem generator
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  // Gas constant of the default hydrogen-helium mixture, so the temperature the
  // Monte Carlo inverts from this pressure is the one written here.
  Real rideal = MonteCarloBlock::GasConstant(pin->GetOrAddReal("problem","heabund",0.09));
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

//----------------------------------------------------------------------------------------
// Function for preparing Mesh
// Inputs:
//   pin: input parameters (unused)
// Outputs: (none)
// Notes:
//   Reads inputs, allocates arrays, and enrolls functions.

void Mesh::InitUserMeshData(ParameterInput *pin) {

  // Check for Kerr-Schild coordinates
  if (std::strcmp(COORDINATE_SYSTEM, "gr_user") != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Problem Generator: Must use \"gr_user\" coordinates."
        << std::endl;
    ATHENA_ERROR(msg);
    return;
  }

  // Enroll user-defined functions
  EnrollUserMetric(CartesianKerrSchild);

  return;
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
  abh = pcoord->GetSpin();
  mbh = pcoord->GetMass();
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
  Real xcam = sin(thcam)*(rcam*cos(phcam) - abh*sin(phcam));
  Real ycam = sin(thcam)*(rcam*sin(phcam) + abh*cos(phcam));
  Real zcam = rcam*cos(thcam);

  MCCoord *pco = pcoord;
  i1start = -1;
  for(int i=is; i<=ie; i++) {
    if ((xcam >= pcoord->x1f(i)) && (xcam < pcoord->x1f(i+1)))
      i1start = i;
  }
  i2start = -1;
  for(int i=js; i<=je; i++) {
    if ((ycam >= pcoord->x2f(i)) && (ycam < pcoord->x2f(i+1)))
      i2start = i;
  }
  i3start = -1;
  for(int i=ks; i<=ke; i++) {
    if ((zcam >= pcoord->x3f(i)) && (zcam < pcoord->x3f(i+1)))
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
    Real xcam = sin(thcam)*(rcam*cos(phcam) - abh*sin(phcam));
    Real ycam = sin(thcam)*(rcam*sin(phcam) + abh*cos(phcam));
    Real zcam = rcam*cos(thcam);

    Real x[NCOORD];
    x[IMC0] = pphot->x0p[ip] = 0.;
    x[IMC1] = pphot->x1p[ip] = xcam;
    x[IMC2] = pphot->x2p[ip] = ycam;
    x[IMC3] = pphot->x3p[ip] = zcam;

    // Initialize Stokes vector as unpolarized
    if (IsPolarized(pphot->polarized)) {
      pphot->sip[ip] = 1.0;
      pphot->sqp[ip] = 0.0;
      pphot->sup[ip] = 0.0;
      pphot->svp[ip] = 0.0;
    }

    // Set the initial photon direction using alpha, beta and the position

    GetDirectionTetrad(pphot, alpha0, beta0, ip);

    pphot->ep[ip] = pphot->k0p[ip];

    pphot->PrintPhoton("init",ip);

    // Initialize dk to zero
    //pphot->dk0p[ip] = 0;
    //pphot->dk1p[ip] = 0;
    //pphot->dk2p[ip] = 0;
    //pphot->dk3p[ip] = 0;

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
  pphot->k0p[ip] = k[IMC0];
  pphot->k1p[ip] = k[IMC1];
  pphot->k2p[ip] = k[IMC2];
  pphot->k3p[ip] = k[IMC3];
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
void GetDirectionTetradOld(Photon *pphot, Real alpha, Real beta, int ip) {

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

void GetDirectionTetrad(Photon *pphot, Real u, Real v, int ip) {

  // Just copy black light for now to get a test working

  Real x[4];
  x[IMC0] = pphot->x0p[ip];
  x[IMC1] = pphot->x1p[ip];
  x[IMC2] = pphot->x2p[ip];
  x[IMC3] = pphot->x3p[ip];

 // Calculate trigonometric quantities
  double sth = std::sin(thcam);
  double cth = std::cos(thcam);
  double sph = std::sin(phcam);
  double cph = std::cos(phcam);

  double z_sign = x[3] >= 0.0 ? 1.0 : -1.0;

  // Calculate metric in spherical coordinates
  double bh_a = abh;
  double bh_m = mbh;
  double a2 = bh_a * bh_a;
  double r2 = rcam * rcam;
  double delta = r2 - 2.0 * bh_m * rcam + a2;
  double sigma = r2 + a2 * cth * cth;
  double g_cov_r_r = 1.0 + 2.0 * bh_m * rcam / sigma;
  double g_cov_r_th = 0.0;
  double g_cov_r_ph = -(1.0 + 2.0 * bh_m * rcam / sigma) * bh_a * sth * sth;
  double g_cov_th_th = sigma;
  double g_cov_th_ph = 0.0;
  double g_cov_ph_ph = (r2 + a2 + 2.0 * bh_m * a2 * rcam / sigma * sth * sth) * sth * sth;
  double g_con_t_t = -(1.0 + 2.0 * bh_m * rcam / sigma);
  double g_con_t_r = 2.0 * bh_m * rcam / sigma;
  double g_con_t_th = 0.0;
  double g_con_t_ph = 0.0;
  double g_con_r_r = delta / sigma;
  double g_con_r_th = 0.0;
  double g_con_r_ph = bh_a / sigma;
  double g_con_th_th = 1.0 / sigma;
  double g_con_th_ph = 0.0;
  double g_con_ph_ph = 1.0 / (sigma * sth * sth);

  // stationary camera
  double camera_urn = 0.;
  double camera_uthn = 0.;
  double camera_uphn = 0.;
  Real u_con[4];
  // Calculate camera velocity in spherical coordinates
  double alpha = 1.0 / std::sqrt(-g_con_t_t);
  double beta_con_r = -g_con_t_r / g_con_t_t;
  double beta_con_th = -g_con_t_th / g_con_t_t;
  double beta_con_ph = -g_con_t_ph / g_con_t_t;
  double utn = std::sqrt(1.0 + g_cov_r_r * camera_urn * camera_urn
      + 2.0 * g_cov_r_th * camera_urn * camera_uthn + 2.0 * g_cov_r_ph * camera_urn * camera_uphn
      + g_cov_th_th * camera_uthn * camera_uthn + 2.0 * g_cov_th_ph * camera_uthn * camera_uphn
      + g_cov_ph_ph * camera_uphn * camera_uphn);
  u_con[0] = utn / alpha;
  double ur = camera_urn - beta_con_r / alpha * utn;
  double uth = camera_uthn - beta_con_th / alpha * utn;
  double uph = camera_uphn - beta_con_ph / alpha * utn;

  // Calculate Jacobian of transformation
  double dx_dr = sth * cph;
  double dy_dr = sth * sph;
  double dz_dr = cth;
  double dx_dth = cth * (rcam * cph - bh_a * sph);
  double dy_dth = cth * (rcam * sph + bh_a * cph);
  double dz_dth = -rcam * sth;
  double dx_dph = sth * (-rcam * sph - bh_a * cph);
  double dy_dph = sth * (rcam * cph - bh_a * sph);
  double dz_dph = 0.0;

  // Calculate camera velocity
  u_con[1] = dx_dr * ur + dx_dth * uth + dx_dph * uph;
  u_con[2] = dy_dr * ur + dy_dth * uth + dy_dph * uph;
  u_con[3] = dz_dr * ur + dz_dth * uth + dz_dph * uph;


  MCCoord *pcoord = pphot->pmy_mcb->pcoord;
  Real g_cov[4][4];
  Real u_cov[4];
  pcoord->Metric(x, g_cov);
  for (int mu = 0; mu < 4; mu++)
  {
    u_cov[mu] = 0.0;
    for (int nu = 0; nu < 4; nu++)
      u_cov[mu] += g_cov[mu][nu] * u_con[nu];
  }

  double camera_k_r = 1.;
  double camera_k_th = 0.;
  double camera_k_ph = 0.;
  // Calculate photon momentum in spherical coordinates
  double g_con_rn_rn = (g_con_t_t * g_con_r_r - g_con_t_r * g_con_t_r) / g_con_t_t;
  double g_con_rn_thn = (g_con_t_t * g_con_r_th - g_con_t_r * g_con_t_th) / g_con_t_t;
  double g_con_rn_phn = (g_con_t_t * g_con_r_ph - g_con_t_r * g_con_t_ph) / g_con_t_t;
  double g_con_thn_thn = (g_con_t_t * g_con_th_th - g_con_t_th * g_con_t_th) / g_con_t_t;
  double g_con_thn_phn = (g_con_t_t * g_con_th_ph - g_con_t_th * g_con_t_ph) / g_con_t_t;
  double g_con_phn_phn = (g_con_t_t * g_con_ph_ph - g_con_t_ph * g_con_t_ph) / g_con_t_t;
  double k_rn = camera_k_r;
  double k_thn = camera_k_th;
  double k_phn = camera_k_ph;
  double k_tn = -std::sqrt(g_con_rn_rn * k_rn * k_rn + 2.0 * g_con_rn_thn * k_rn * k_thn
      + 2.0 * g_con_rn_phn * k_rn * k_phn + g_con_thn_thn * k_thn * k_thn
      + 2.0 * g_con_thn_phn * k_thn * k_phn + g_con_phn_phn * k_phn * k_phn);
  double k_t = alpha * k_tn + (beta_con_r * k_rn + beta_con_th * k_thn + beta_con_ph * k_phn);

  // Calculate Jacobian of transformation
  double rr2 = x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
  double dr_dx = rcam * x[1] / (2.0 * r2 - rr2 + a2);
  double dr_dy = rcam * x[2] / (2.0 * r2 - rr2 + a2);
  double dr_dz = (rcam * x[3] + a2 * x[3] / rcam) / (2.0 * r2 - rr2 + a2);
  double dth_dx = x[3] * dr_dx / (r2 * sth);
  double dth_dy = x[3] * dr_dy / (r2 * sth);
  double dth_dz = (x[3] * dr_dz - rcam) / (r2 * sth);
  double dph_dx =
      -x[2] / (x[1] * x[1] + x[2] * x[2]) + bh_a / (r2 + a2) * dr_dx;
  double dph_dy = x[1] / (x[1] * x[1] + x[2] * x[2]) + bh_a / (r2 + a2) * dr_dy;
  double dph_dz = bh_a / (r2 + a2) * dr_dz;


  // Calculate photon momentum
  double k_x = dr_dx * camera_k_r + dth_dx * camera_k_th + dph_dx * camera_k_ph;
  double k_y = dr_dy * camera_k_r + dth_dy * camera_k_th + dph_dy * camera_k_ph;
  double k_z = dr_dz * camera_k_r + dth_dz * camera_k_th + dph_dz * camera_k_ph;
  double k_tc = u_con[0] * k_t + u_con[1] * k_x + u_con[2] * k_y + u_con[3] * k_z;

  // Calculate contravariant metric in camera frame
  Real g_con[4][4];
  pcoord->InverseMetric(x, g_con);
  double g_con_xc_xc = g_con[1][1] + u_con[1] * u_con[1];
  double g_con_xc_yc = g_con[1][2] + u_con[1] * u_con[2];
  double g_con_xc_zc = g_con[1][3] + u_con[1] * u_con[3];
  double g_con_yc_yc = g_con[2][2] + u_con[2] * u_con[2];
  double g_con_yc_zc = g_con[2][3] + u_con[2] * u_con[3];
  double g_con_zc_zc = g_con[3][3] + u_con[3] * u_con[3];

  // Calculate camera normal direction in camera frame
  double norm_cov_xc = k_x - u_cov[1] / u_cov[0] * k_t;
  double norm_cov_yc = k_y - u_cov[2] / u_cov[0] * k_t;
  double norm_cov_zc = k_z - u_cov[3] / u_cov[0] * k_t;

  double norm_con_c[4];
  norm_con_c[0] = -k_tc;
  norm_con_c[1] = g_con_xc_xc * norm_cov_xc + g_con_xc_yc * norm_cov_yc + g_con_xc_zc * norm_cov_zc;
  norm_con_c[2] = g_con_xc_yc * norm_cov_xc + g_con_yc_yc * norm_cov_yc + g_con_yc_zc * norm_cov_zc;
  norm_con_c[3] = g_con_xc_zc * norm_cov_xc + g_con_yc_zc * norm_cov_yc + g_con_zc_zc * norm_cov_zc;
  double norm_norm = std::sqrt(norm_cov_xc * norm_con_c[1] + norm_cov_yc * norm_con_c[2]
      + norm_cov_zc * norm_con_c[3]);
  norm_cov_xc /= norm_norm;
  norm_cov_yc /= norm_norm;
  norm_cov_zc /= norm_norm;
  norm_con_c[0] /= norm_norm;
  norm_con_c[1] /= norm_norm;
  norm_con_c[2] /= norm_norm;
  norm_con_c[3] /= norm_norm;
  double norm_con[4];
  norm_con[0] = u_con[0] * norm_con_c[0]
      - (u_cov[1] * norm_con_c[1] + u_cov[2] * norm_con_c[2] + u_cov[3] * norm_con_c[3]) / u_cov[0];
  norm_con[1] = norm_con_c[1] + u_con[1] * norm_con_c[0];
  norm_con[2] = norm_con_c[2] + u_con[2] * norm_con_c[0];
  norm_con[3] = norm_con_c[3] + u_con[3] * norm_con_c[0];

  // Define unprojected vertical direction in camera frame
  double up_con_xc = 0.0;
  double up_con_yc = 0.0;
  double up_con_zc = 1.0;

  // Calculate covariant metric in camera frame
  double g_cov_xc_xc = g_cov[1][1] - u_cov[1] / u_cov[0] * g_cov[1][0]
      - u_cov[1] / u_cov[0] * g_cov[1][0]
      + u_cov[1] * u_cov[1] / (u_cov[0] * u_cov[0]) * g_cov[0][0];
  double g_cov_xc_yc = g_cov[1][2] - u_cov[1] / u_cov[0] * g_cov[2][0]
      - u_cov[2] / u_cov[0] * g_cov[1][0]
      + u_cov[1] * u_cov[2] / (u_cov[0] * u_cov[0]) * g_cov[0][0];
  double g_cov_xc_zc = g_cov[1][3] - u_cov[1] / u_cov[0] * g_cov[3][0]
      - u_cov[3] / u_cov[0] * g_cov[1][0]
      + u_cov[1] * u_cov[3] / (u_cov[0] * u_cov[0]) * g_cov[0][0];
  double g_cov_yc_yc = g_cov[2][2] - u_cov[2] / u_cov[0] * g_cov[2][0]
      - u_cov[2] / u_cov[0] * g_cov[2][0]
      + u_cov[2] * u_cov[2] / (u_cov[0] * u_cov[0]) * g_cov[0][0];
  double g_cov_yc_zc = g_cov[2][3] - u_cov[2] / u_cov[0] * g_cov[3][0]
      - u_cov[3] / u_cov[0] * g_cov[2][0]
      + u_cov[2] * u_cov[3] / (u_cov[0] * u_cov[0]) * g_cov[0][0];
  double g_cov_zc_zc = g_cov[3][3] - u_cov[3] / u_cov[0] * g_cov[3][0]
      - u_cov[3] / u_cov[0] * g_cov[3][0]
      + u_cov[3] * u_cov[3] / (u_cov[0] * u_cov[0]) * g_cov[0][0];

  // Calculate camera vertical direction without rotation in camera frame
  double up_norm = up_con_xc * norm_cov_xc + up_con_yc * norm_cov_yc + up_con_zc * norm_cov_zc;
  double vert_con_c[4];
  vert_con_c[0] = 0.0;
  vert_con_c[1] = up_con_xc - up_norm * norm_con_c[1];
  vert_con_c[2] = up_con_yc - up_norm * norm_con_c[2];
  vert_con_c[3] = up_con_zc - up_norm * norm_con_c[3];
  double vert_cov_xc =
      g_cov_xc_xc * vert_con_c[1] + g_cov_xc_yc * vert_con_c[2] + g_cov_xc_zc * vert_con_c[3];
  double vert_cov_yc =
      g_cov_xc_yc * vert_con_c[1] + g_cov_yc_yc * vert_con_c[2] + g_cov_yc_zc * vert_con_c[3];
  double vert_cov_zc =
      g_cov_xc_zc * vert_con_c[1] + g_cov_yc_zc * vert_con_c[2] + g_cov_zc_zc * vert_con_c[3];
  double vert_norm = std::sqrt(vert_cov_xc * vert_con_c[1] + vert_cov_yc * vert_con_c[2]
      + vert_cov_zc * vert_con_c[3]);
  vert_cov_xc /= vert_norm;
  vert_cov_yc /= vert_norm;
  vert_cov_zc /= vert_norm;
  double vert_con[4];
  vert_con_c[1] /= vert_norm;
  vert_con_c[2] /= vert_norm;
  vert_con_c[3] /= vert_norm;

  // Calculate determinant of metric in camera frame
  double det = g_cov_xc_xc * (g_cov_yc_yc * g_cov_zc_zc - g_cov_yc_zc * g_cov_yc_zc)
      + g_cov_xc_yc * (g_cov_yc_zc * g_cov_xc_zc - g_cov_xc_yc * g_cov_zc_zc)
      + g_cov_xc_zc * (g_cov_xc_yc * g_cov_yc_zc - g_cov_yc_yc * g_cov_xc_zc);
  double det_sqrt = std::sqrt(det);

  // Calculate camera horizontal direction without rotation in camera frame
  double hor_con_c[4];
  hor_con_c[0] = 0.0;
  hor_con_c[1] = (vert_cov_yc * norm_cov_zc - vert_cov_zc * norm_cov_yc) / det_sqrt;
  hor_con_c[2] = (vert_cov_zc * norm_cov_xc - vert_cov_xc * norm_cov_zc) / det_sqrt;
  hor_con_c[3] = (vert_cov_xc * norm_cov_yc - vert_cov_yc * norm_cov_xc) / det_sqrt;

  // Calculate pixel direction
  double normalization = std::sqrt(u*u + v*v + rcam*rcam);
  double frac_norm = rcam / normalization;
  double frac_hor = -u / normalization;
  double frac_vert = -v / normalization;
  double dir_con_tc = norm_con_c[0];
  double dir_con_xc =
      frac_norm * norm_con_c[1] + frac_hor * hor_con_c[1] + frac_vert * vert_con_c[1];
  double dir_con_yc =
      frac_norm * norm_con_c[2] + frac_hor * hor_con_c[2] + frac_vert * vert_con_c[2];
  double dir_con_zc =
      frac_norm * norm_con_c[3] + frac_hor * hor_con_c[3] + frac_vert * vert_con_c[3];
  double dir_con_x = dir_con_xc + u_con[1] * dir_con_tc;
  double dir_con_y = dir_con_yc + u_con[2] * dir_con_tc;
  double dir_con_z = dir_con_zc + u_con[3] * dir_con_tc;

  double k[4];
  k[1] = dir_con_x;
  k[2] = dir_con_y;
  k[3] = dir_con_z;

  // Calculate time component of momentum
  Real gcov[4][4];
  pcoord->Metric(x, gcov);
  double temp_a = gcov[0][0];
  double temp_b = 0.0;
  for (int a = 1; a < 4; a++)
    temp_b += 2.0 * gcov[0][a] * k[a];
  double temp_c = 0.0;
  for (int a = 1; a < 4; a++)
    for (int b = 1; b < 4; b++)
      temp_c += gcov[a][b] * k[a] * k[b];
  double temp_d = std::sqrt(std::max(temp_b * temp_b - 4.0 * temp_a * temp_c, 0.0));
  k[0] = temp_a == 0.0 ? -temp_c / (2.0 * temp_b)
      : (temp_b < 0.0 ? 2.0 * temp_c / (temp_d - temp_b) : -(temp_b + temp_d) / (2.0 * temp_a));

  pphot->k0p[ip] = -k[IMC0];
  pphot->k1p[ip] = -k[IMC1];
  pphot->k2p[ip] = -k[IMC2];
  pphot->k3p[ip] = -k[IMC3];

}

void MidplaneCrossing(MonteCarloBlock *pmcb, Photon *pphot, PhotonPusher *ppusher, int ip) {

 // check if photon has crossed midplane and whether to terminate or keep integrating

  pphot->PrintPhoton("in move",ip);
  Real R2 = SQR(pphot->x1p[ip])+SQR(pphot->x2p[ip])+SQR(pphot->x3p[ip]);
  Real a2 = abh*abh;
  Real z2 = pphot->x3p[ip]*pphot->x3p[ip];
  Real r = std::sqrt(0.5*(R2-a2+std::sqrt(SQR(R2+a2)+4*a2*z2)));
  Real tmp1 = 2*r*r-(R2-a2);
  Real drdx = (1+(R2-a2)/tmp1)*pphot->x1p[ip]/2./r;
  Real drdy = (1+(R2-a2)/tmp1)*pphot->x2p[ip]/2./r;
  Real drdz = (1+(R2+a2)/tmp1)*pphot->x3p[ip]/2./r;
  Real kr = drdx*pphot->k1p[ip]+drdy*pphot->k2p[ip]+drdz*pphot->k3p[ip];
  Real x[4];
  x[IMC0] = pphot->x0p[ip];
  x[IMC1] = pphot->x1p[ip];
  x[IMC2] = pphot->x2p[ip];
  x[IMC3] = pphot->x3p[ip];
  Real gcov[4][4];
  pmcb->pcoord->Metric(x, gcov);
  Real k[4];
  k[IMC0] = pphot->k0p[ip];
  k[IMC1] = pphot->k1p[ip];
  k[IMC2] = pphot->k2p[ip];
  k[IMC3] = pphot->k3p[ip];

  Real kt2=0.;
  Real kxi2=0.;
  for (int i=0; i<4; i++) {
    kt2 += gcov[0][i]*k[i]*k[0];
    for (int j=1; j<4; j++) {
      kxi2 += gcov[j][i]*k[i]*k[j];
    }
  }
  printf("k2: %g\n",kt2/kxi2);
  printf("r, kr: %g %g\n",r,kr);
  if (r < rh + 1.0e-5) {
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
      if (pphot->x3p[ip] <= 0) { // photon has crossed plane for the first time
        if (r <= rdisk) { // photon is less than outer disk radius
          Real step = -(pphot->x3p[ip] - 0.) / pphot->k3p[ip];
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
    if (pphot->x3p[ip] <= 0.) { //new crossing
      if (r <= rdisk) {
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
    if (pphot->x2p[ip] <= 0.) { // new crossing
      if (r <= rdisk) {
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
    pphot->PrintPhoton(ip);
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
    Real omega = pow(mbh,0.5)/(pow(r, 3./2.) + abh*pow(mbh,3./2.));
    // Initialize ucon and vcon (= z unit vector in symmetry plane)
    Real ucon[4];


    // first define in spherical KS coords
    R2 = SQR(pphot->x1p[ip])+SQR(pphot->x2p[ip])+SQR(pphot->x3p[ip]);
    z2 = pphot->x3p[ip]*pphot->x3p[ip];
    r = std::sqrt(0.5*(R2-a2+std::sqrt(SQR(R2+a2)+4*a2*z2)));
    Real th = std::acos(pphot->x3p[ip]/r);
    Real r2 = SQR(r);
    Real sth = sin(th);
    Real cth = cos(th);
    Real cth2 = SQR(cth);
    Real sth2 = SQR(sth);
    Real sigma = r2 + a2 * cth2;
    Real delta = r2 - 2 * r + a2;
    Real A = SQR(r2 + a2) - a2 * delta * sth2;
    Real gcov_00 = -1. * (1. - 2. * r / sigma);
    Real gcov_03 = -2. * abh * r * sth2 / sigma;
    Real gcov_33 = A * sth2 / sigma;
    ucon[IMC0] = sqrt(-1.0/(gcov_00 + 2.*gcov_03*omega +
                            SQR(omega)*gcov_33));
    ucon[IMC1] = ucon[IMC0]*omega;
    ucon[IMC2] = ucon[IMC0]*omega;
    ucon[IMC3] = 0.;
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

//----------------------------------------------------------------------------------------
// Function for defining Cartesian Kerr-Schild metric
// Inputs:
//   x, y, z: Cartesian Kerr-Schild coordinates
//   pin: input parameters
// Outputs:
//   g, g_inv: covariant and contravariant metric components set
//   dg_dx, dg_dy, dg_dz: spatial derivatives of covariant metric components set

void CartesianKerrSchild(Real x, Real y, Real z, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx,
    AthenaArray<Real> &dg_dy, AthenaArray<Real> &dg_dz) {

  // Extract inputs
  Real a = pin->GetReal("coord", "a");

  // Calculate scalar quantities
  Real a2 = SQR(a);
  Real z2 = SQR(z);
  Real rr2 = SQR(x) + SQR(y) + z2;
  Real r2 = 0.5 * (rr2 - a2 + std::sqrt(SQR(rr2 - a2) + 4.0 * a2 * z2));
  Real r4 = SQR(r2);
  Real r = std::sqrt(r2);
  Real f = 2.0 * r * r2 / (r4 + a2 * z2);

  // Calculate vector quantities
  Real l_0 = 1.0;
  Real l_1 = (r * x + a * y) / (r2 + a2);
  Real l_2 = (r * y - a * x) / (r2 + a2);
  Real l_3 = z / r;
  Real l0 = -1.0;
  Real l1 = l_1;
  Real l2 = l_2;
  Real l3 = l_3;

  // Calculate scalar derivatives
  Real dr_dx = r * x / (2.0 * r2 - rr2 + a2);
  Real dr_dy = r * y / (2.0 * r2 - rr2 + a2);
  Real dr_dz = (r * z + a2 * z / r) / (2.0 * r2 - rr2 + a2);
  Real df_dx = -(r4 - 3.0 * a2 * z2) * dr_dx / (r * (r4 + a2 * z2)) * f;
  Real df_dy = -(r4 - 3.0 * a2 * z2) * dr_dy / (r * (r4 + a2 * z2)) * f;
  Real df_dz =
      -((r4 - 3.0 * a2 * z2) * dr_dz + 2.0 * a2 * r * z) / (r * (r4 + a2 * z2)) * f;

  // Calculate vector derivatives
  Real dl_0_dx = 0.0;
  Real dl_0_dy = 0.0;
  Real dl_0_dz = 0.0;
  Real dl_1_dx = ((x - 2.0 * r * l_1) * dr_dx + r) / (r2 + a2);
  Real dl_1_dy = ((x - 2.0 * r * l_1) * dr_dy + a) / (r2 + a2);
  Real dl_1_dz = (x - 2.0 * r * l_1) * dr_dz / (r2 + a2);
  Real dl_2_dx = ((y - 2.0 * r * l_2) * dr_dx - a) / (r2 + a2);
  Real dl_2_dy = ((y - 2.0 * r * l_2) * dr_dy + r) / (r2 + a2);
  Real dl_2_dz = (y - 2.0 * r * l_2) * dr_dz / (r2 + a2);
  Real dl_3_dx = -z / r2 * dr_dx;
  Real dl_3_dy = -z / r2 * dr_dy;
  Real dl_3_dz = -z / r2 * dr_dz + 1.0 / r;

  // Calculate covariant components
  g(I00) = f * l_0 * l_0 - 1.0;
  g(I01) = f * l_0 * l_1;
  g(I02) = f * l_0 * l_2;
  g(I03) = f * l_0 * l_3;
  g(I11) = f * l_1 * l_1 + 1.0;
  g(I12) = f * l_1 * l_2;
  g(I13) = f * l_1 * l_3;
  g(I22) = f * l_2 * l_2 + 1.0;
  g(I23) = f * l_2 * l_3;
  g(I33) = f * l_3 * l_3 + 1.0;

  // Calculate contravariant components
  g_inv(I00) = -f * l0 * l0 - 1.0;
  g_inv(I01) = -f * l0 * l1;
  g_inv(I02) = -f * l0 * l2;
  g_inv(I03) = -f * l0 * l3;
  g_inv(I11) = -f * l1 * l1 + 1.0;
  g_inv(I12) = -f * l1 * l2;
  g_inv(I13) = -f * l1 * l3;
  g_inv(I22) = -f * l2 * l2 + 1.0;
  g_inv(I23) = -f * l2 * l3;
  g_inv(I33) = -f * l3 * l3 + 1.0;

  // Calculate covariant x-derivatives
  dg_dx(I00) = df_dx * l_0 * l_0 + f * dl_0_dx * l_0 + f * l_0 * dl_0_dx;
  dg_dx(I01) = df_dx * l_0 * l_1 + f * dl_0_dx * l_1 + f * l_0 * dl_1_dx;
  dg_dx(I02) = df_dx * l_0 * l_2 + f * dl_0_dx * l_2 + f * l_0 * dl_2_dx;
  dg_dx(I03) = df_dx * l_0 * l_3 + f * dl_0_dx * l_3 + f * l_0 * dl_3_dx;
  dg_dx(I11) = df_dx * l_1 * l_1 + f * dl_1_dx * l_1 + f * l_1 * dl_1_dx;
  dg_dx(I12) = df_dx * l_1 * l_2 + f * dl_1_dx * l_2 + f * l_1 * dl_2_dx;
  dg_dx(I13) = df_dx * l_1 * l_3 + f * dl_1_dx * l_3 + f * l_1 * dl_3_dx;
  dg_dx(I22) = df_dx * l_2 * l_2 + f * dl_2_dx * l_2 + f * l_2 * dl_2_dx;
  dg_dx(I23) = df_dx * l_2 * l_3 + f * dl_2_dx * l_3 + f * l_2 * dl_3_dx;
  dg_dx(I33) = df_dx * l_3 * l_3 + f * dl_3_dx * l_3 + f * l_3 * dl_3_dx;

  // Calculate covariant y-derivatives
  dg_dy(I00) = df_dy * l_0 * l_0 + f * dl_0_dy * l_0 + f * l_0 * dl_0_dy;
  dg_dy(I01) = df_dy * l_0 * l_1 + f * dl_0_dy * l_1 + f * l_0 * dl_1_dy;
  dg_dy(I02) = df_dy * l_0 * l_2 + f * dl_0_dy * l_2 + f * l_0 * dl_2_dy;
  dg_dy(I03) = df_dy * l_0 * l_3 + f * dl_0_dy * l_3 + f * l_0 * dl_3_dy;
  dg_dy(I11) = df_dy * l_1 * l_1 + f * dl_1_dy * l_1 + f * l_1 * dl_1_dy;
  dg_dy(I12) = df_dy * l_1 * l_2 + f * dl_1_dy * l_2 + f * l_1 * dl_2_dy;
  dg_dy(I13) = df_dy * l_1 * l_3 + f * dl_1_dy * l_3 + f * l_1 * dl_3_dy;
  dg_dy(I22) = df_dy * l_2 * l_2 + f * dl_2_dy * l_2 + f * l_2 * dl_2_dy;
  dg_dy(I23) = df_dy * l_2 * l_3 + f * dl_2_dy * l_3 + f * l_2 * dl_3_dy;
  dg_dy(I33) = df_dy * l_3 * l_3 + f * dl_3_dy * l_3 + f * l_3 * dl_3_dy;

  // Calculate covariant z-derivatives
  dg_dz(I00) = df_dz * l_0 * l_0 + f * dl_0_dz * l_0 + f * l_0 * dl_0_dz;
  dg_dz(I01) = df_dz * l_0 * l_1 + f * dl_0_dz * l_1 + f * l_0 * dl_1_dz;
  dg_dz(I02) = df_dz * l_0 * l_2 + f * dl_0_dz * l_2 + f * l_0 * dl_2_dz;
  dg_dz(I03) = df_dz * l_0 * l_3 + f * dl_0_dz * l_3 + f * l_0 * dl_3_dz;
  dg_dz(I11) = df_dz * l_1 * l_1 + f * dl_1_dz * l_1 + f * l_1 * dl_1_dz;
  dg_dz(I12) = df_dz * l_1 * l_2 + f * dl_1_dz * l_2 + f * l_1 * dl_2_dz;
  dg_dz(I13) = df_dz * l_1 * l_3 + f * dl_1_dz * l_3 + f * l_1 * dl_3_dz;
  dg_dz(I22) = df_dz * l_2 * l_2 + f * dl_2_dz * l_2 + f * l_2 * dl_2_dz;
  dg_dz(I23) = df_dz * l_2 * l_3 + f * dl_2_dz * l_3 + f * l_2 * dl_3_dz;
  dg_dz(I33) = df_dz * l_3 * l_3 + f * dl_3_dz * l_3 + f * l_3 * dl_3_dz;
  return;
}

} // namespace
