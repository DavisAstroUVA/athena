//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mc_geokerr.cpp
//! \brief Problem generator for testing geodesic integration in kerr spacetime
//
//========================================================================================

#include <iostream> // temporary for testing
#include <stdexcept>

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

  // global variables
  int iphot;
  Real r0,th0,phi0;
  Real gcov0[4][4];
  Real muk,phik;
  Real rprev;
  int i1start,i2start,i3start;
  bool first;

  // user function definitions
  void TurningPointCheck(MonteCarloBlock *pmcb, Photon *pphot, PhotonPusher *ppusher,
                         int ip);

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

  // Request 6 user variables
  nuser_var = 6;

  // Enroll user work in move function
  EnrollUserWorkInMove(TurningPointCheck);
}

//========================================================================================
//! \fn void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin)
//! \brief Analogous to problem generator but used in support of InitializePhoton
//========================================================================================

void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin) {

  Real signa;
  if(pin->GetOrAddBoolean("problem", "corotating",true))
    signa = 1.;
  else
    signa = -1.;

  Real abh = pin->GetReal("coord", "a");
  Real mbh = pin->GetReal("coord","m");
  Real a2 = abh*abh;
  Real z1 = 1.0 + pow(1.0 - a2, 1./3.) * (pow(1. + abh, 1./3.) + pow(1.0-abh,1./3.));
  Real z2 = sqrt(3.*a2 + z1*z1);
  Real risco = 3.0 + z2 - signa * sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2)) + 1.0e-3;
  r0 = pin->GetReal("problem", "radius")*risco;
  th0 = pin->GetOrAddReal("problem", "theta",0.5)*M_PI;
  phi0 = pin->GetOrAddReal("problem", "phi", 0.)*M_PI;
  muk = pin->GetOrAddReal("problem", "muk",-0.8);
  phik = pin->GetOrAddReal("problem", "phik", 0.4);

  // Initialize gcov0
  MCCoord *pcobl = new MCBoyerLindquist(1,1,1,false);
  Real x[4];
  x[IMC0] = 1.; x[IMC1] = r0; x[IMC2] = th0; x[IMC3] = phi0;
  pcobl->SetSpin(abh);
  pcobl->SetMass(mbh);
  pcobl->Metric(x,gcov0);

  // set the photon samples 's initial zone indices
  MCCoord *pco = pcoord;
  i1start = -1;
  for(int i=is; i<=ie; i++) {
    if ((r0 >= pcoord->x1f(i)) && (r0 < pcoord->x1f(i+1)))
      i1start = i;
  }
  i2start = -1;
  for(int i=js; i<=je; i++) {
    if ((th0 >= pcoord->x2f(i)) && (th0 < pcoord->x2f(i+1)))
      i2start = i;
  }
  i3start = -1;
  for(int i=ks; i<=ke; i++) {
    if ((phi0 >= pcoord->x3f(i)) && (phi0 < pcoord->x3f(i+1)))
      i3start = i;
  }
  if ((i1start < 0) || (i2start < 0) || (i3start < 0)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in InitUserMonteCarloBlockData" << std::endl
        << "Initial position not found within domain." << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

#ifdef MPI_PARALLEL
  // Set iphot based on assumption that rays are distributed evenly
  // across active processes
  int rank = Globals::my_rank;
  int ntot = pin->GetInteger("montecarlo", "nphot");
  if (rank > 0) {
    int nranks = Globals::nranks;
    int myn = ntot/(nranks-1);
    int remain = ntot % (nranks-1);
    if (rank <= remain) {
      myn++;
      iphot = (rank-1)*myn;
    } else {
      iphot = remain*(myn+1) + (rank-1-remain)*myn;
    }
    printf("iphot: %d %d %d\n",rank,iphot,myn);
  }
#else
  iphot = 0;
#endif

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

    // initialize cell coordinates
    pphot->i1p[ip] = i1start;
    pphot->i2p[ip] = i2start;
    pphot->i3p[ip] = i3start;

    // Initialize photon position
    Real x[NCOORD];
    x[IMC0] = pphot->x0p[ip] = 0.;
    x[IMC1] = pphot->x1p[ip] = r0;
    x[IMC2] = pphot->x2p[ip] = th0;
    x[IMC3] = pphot->x3p[ip] = phi0;

    //SWDFIX
    rprev = r0; // Used to check for turning point in r

    // Set the initial photon direction assuming "isotropic" emission
    pphot->ep[ip] = 1.0;
    pphot->type[ip] = iphot;
    int ith = iphot / 4;
    int iph = iphot % 4;
    Real cth = muk + 0.2 * static_cast<Real>(ith);
    Real phi = (phik + 0.4 * static_cast<Real>(iph)) * M_PI;
    Real sth = sqrt(1.-cth*cth);

    iphot++;

    Real ktet[NCOORD];
    ktet[IMC0] = pphot->ep[ip];
    ktet[IMC1] = pphot->ep[ip]*sth*sin(phi);
    ktet[IMC2] = pphot->ep[ip]*cth;
    ktet[IMC3] = pphot->ep[ip]*sth*cos(phi);

    // Initialize Stokes vector as unpolarized
    if (IsPolarized(pphot->polarized)) {
      pphot->sip[ip] = 1.0;
      pphot->sqp[ip] = 0.0;
      pphot->sup[ip] = 0.0;
      pphot->svp[ip] = 0.0;
    }

    // Initialize the absorption and scattering extinction coefficients
    // to the values appropriate in the emitted zone
    pphot->acp[ip] = 0.;
    pphot->scp[ip] = 0.;

    // Transform to coordinate frame
    Real gcov[NCOORD][NCOORD];
    pco->Metric(x, gcov);

    Real r = pphot->x1p[ip];
    Real a = pco->GetSpin();
    Real omega = 1.0/(pow(r, 3./2.) + a); // circular velocity
    Real ucon[NCOORD];
    ucon[IMC0] = sqrt(-1.0/(gcov0[IMC0][IMC0] + 2.*gcov0[IMC0][IMC3]*omega +
                            SQR(omega)*gcov0[IMC3][IMC3]));
    ucon[IMC1] = 0.;
    ucon[IMC2] = 0.;
    ucon[IMC3] = (ucon[IMC0])*omega;

    // create tetrad basis
    Real econ[NCOORD][NCOORD], ecov[NCOORD][NCOORD];
    ConstructTetrad(ucon, gcov, econ, ecov);

    // Transform k
    Real k[NCOORD];
    TetradToCoordinate(ktet, k, econ);
    pphot->k0p[ip] = k[IMC0];
    pphot->k1p[ip] = k[IMC1];
    pphot->k2p[ip] = k[IMC2];
    pphot->k3p[ip] = k[IMC3];

    //  Initialize dK
    Real gamma[NCOORD][NCOORD][NCOORD];
    pco->Connect(x, gamma);

    Real dk[NCOORD];
    for (int i = 0; i < 4; i++) {
      dk[i] =
        -2.*(pphot->k[IMC1]*(gamma[i][IMC0][IMC1]*k[IMC1]+gamma[i][IMC0][IMC2]*k[IMC2]+
                        gamma[i][IMC0][IMC3]*k[IMC3])+
           k[IMC1]*(gamma[i][IMC1][IMC2]*k[IMC2]+gamma[i][IMC1][IMC3]*k[IMC3])+
           k[IMC2]*gamma[i][IMC2][IMC3]*k[IMC3])-
        (gamma[i][IMC0][IMC0]*SQR(k[IMC0])+gamma[i][IMC1][IMC1]*SQR(k[IMC1])+
         gamma[i][IMC2][IMC2]*SQR(k[IMC2])+gamma[i][IMC3][IMC3]*SQR(k[IMC3]));
    }
    pphot->dk0p[ip] = dk[IMC0];
    pphot->dk1p[ip] = dk[IMC1];
    pphot->dk2p[ip] = dk[IMC2];
    pphot->dk3p[ip] = dk[IMC3];

  // Compute input variables for geokerr and store as user varibles for photon list
  // Geokerr uses BL coordinates so we first transfer from KS to BL and then compute
  // k_\alpha needed to define alpha, beta for geokerr

    Real alpha,beta;
    if (pphot->pmy_mcb->coord_system != MCCOORD_BOYER_LINDQUIST) {

      Real delta = SQR(x[IMC1]) - 2 * x[IMC1] + SQR(a);
      Real kt0_bl = (k[IMC0] - 2.*x[IMC1]/delta*k[IMC1])*gcov0[IMC0][IMC0]
        + k[IMC1]*gcov0[IMC0][IMC1] + k[IMC2]*gcov0[IMC0][IMC2]
        + (k[IMC3] - a/delta*k[IMC1])*gcov0[IMC0][IMC3];
      Real kth0_bl = (k[IMC0] - 2.*x[IMC1]/delta*k[IMC1])*gcov0[IMC2][IMC0]
        + k[IMC1]*gcov0[IMC2][IMC1] + k[IMC2]*gcov0[IMC2][IMC2]
        + (k[IMC3] - a/delta*k[IMC1])*gcov0[IMC2][IMC3];
      Real kphi0_bl = (k[IMC0] - 2.*x[IMC1]/delta*k[IMC1])*gcov0[IMC3][IMC0]
        + k[IMC1]*gcov0[IMC3][IMC1] + k[IMC2]*gcov0[IMC3][IMC2]
        + (k[IMC3] - a/delta*k[IMC1])*gcov0[IMC3][IMC3];
      alpha = -kphi0_bl / kt0_bl;
      // Assumes with start at theta = pi/2, then beta^2 = q^2
      beta = sqrt(SQR(kth0_bl)/SQR(kt0_bl));
      if (kth0_bl > 0)
        beta = -beta;
    } else {
      Real kt0 = k[IMC0]*gcov0[IMC0][IMC0] + k[IMC1]*gcov0[IMC0][IMC1] +
        k[IMC2]*gcov0[IMC0][IMC2] + k[IMC3]*gcov0[IMC0][IMC3];
      Real kth0 = k[IMC0]*gcov0[IMC2][IMC0] + k[IMC1]*gcov0[IMC2][IMC1] +
        k[IMC2]*gcov0[IMC2][IMC2] + k[IMC3]*gcov0[IMC2][IMC3];
      Real kphi0 = k[IMC0]*gcov0[IMC3][IMC0] + k[IMC1]*gcov0[IMC3][IMC1] +
        k[IMC2]*gcov0[IMC3][IMC2] + k[IMC3]*gcov0[IMC3][IMC3];
      alpha = -kphi0 / kt0;
      // SWD change?
      // Assumes with start at theta = pi/2, then beta^2 = q^2
      beta = sqrt(SQR(kth0)/SQR(kt0));
    }

    // Geokerr initialization parameters
    pphot->user[0][ip] = alpha;
    pphot->user[1][ip] = beta;
    pphot->user[2][ip] = 1./x[IMC1]; // ui for
    if (pphot->k1p[ip] < 0.) {
      pphot->user[3][ip] = 1.;
    } else {
      pphot->user[3][ip] = -1.;
    }
    first = true; // for setting sign of du/dlambda //SWDFIX
    pphot->user[4][ip] = 0.; // tpr
    pphot->user[5][ip] = cos(th0);

    pphot->PrintPhoton(ip);
  } // loop over ip
}

//========================================================================================
//! \fn void MonteCarloBlock::FinalizePhoton(Photon *pphot, int ip)
//! \brief Complete work at end of photon packets before integration
//========================================================================================

void MonteCarloBlock::FinalizePhoton(Photon *pphot, int ip) {

  Real a = pcoord->GetSpin();
  Real r_outer = 1.0 + sqrt(1.0 - SQR(a)) + 1.0e-3;

  // r is outside ISCO, transform into comoving frame tetrad, assuming
  // circular flow velocity
  if (pphot->x1p[ip] < r_outer + 1.0e-5) {
    pphot->statp[ip] = ESCAPED;
    return;
  }

}

namespace {

void TurningPointCheck(MonteCarloBlock *pmcb, Photon *pphot, PhotonPusher *ppusher,
                       int ip) {

  // Check if r is increasing and set sign of du/dlamda accordingly
  // SWDFIX
  /*if (first) {
    if (pphot->x1p[ip] > rprev)
      pphot->user[3][ip] = -1.;
    else
      pphot->user[3][ip] = 1.;
    first = false;
    }*/

  if (pphot->user[3][ip] > 0.)
    if (pphot->x1p[ip] > rprev)
      pphot->user[4][ip] = 1.;
  else
    if (pphot->x1p[ip] < rprev)
      pphot->user[4][ip] = 1.;
  rprev = pphot->x1p[ip];

}

}
