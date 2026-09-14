//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mc_dust.cpp
//  \brief Problem generator for dust scattering monte carlo problem
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
#include "../monte_carlo/photon.hpp"
#include "../monte_carlo/photonpusher.hpp"

#if MAGNETIC_FIELDS_ENABLED
#error "This problem generator does not support magnetic fields"
#endif

namespace {

  int i1s1,i2s1,i3s1,i1s2,i2s2,i3s2;
  Real xs1,ys1,zs1,xs2,ys2,zs2;
  Real lum1,lum2,lfrac;
  // User function definitions
  Real DensityPoints(Real x, Real y, Real z);
}
//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief monte carlo test problem generator
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  //Real rideal = 8.314e7;
  Real rideal = 1.;
  Real temp = pin->GetReal("problem","temp");
  Real gamma = peos->GetGamma();

  for (int k=ks; k<=ke; k++) {
    Real z = pcoord->x3v(k);
    for (int j=js; j<=je; j++) {
      Real y = pcoord->x2v(j);
      for (int i=is; i<=ie; i++) {
        Real x = pcoord->x1v(i);
        Real rho = DensityPoints(x,y,z);
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
//! \fn void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin)
//! \brief Analogous to problem generator but used in support of InitializePhoton
//========================================================================================

void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin) {

  // Define the starting location of two sources, whose position are read in from
  // input file
  xs1 = pin->GetReal("problem","x1");
  ys1 = pin->GetReal("problem","y1");
  xs2 = pin->GetReal("problem","x2");
  ys2 = pin->GetReal("problem","y2");
  zs1 = 0.;
  zs2 = 0.;

  // Read in the luminosity of each source
  lum1 = pin->GetReal("problem","lum1");
  lum2 = pin->GetReal("problem","lum2");
  lfrac = lum1/(lum1+lum2);

  // Deterime cell of initial photon for each source
  i1s1 = -1; i1s2 = -1;
  for(int i=is; i<=ie; i++) {
    if ((xs1 > pcoord->x1f(i)) && (xs1 <= pcoord->x1f(i+1)))
      i1s1 = i;
    if ((xs2 > pcoord->x1f(i)) && (xs2 <= pcoord->x1f(i+1)))
      i1s2 = i;
  }
  i2s1 = -1; i2s2 = -1;
  for(int i=js; i<=je; i++) {
    if ((ys1 > pcoord->x2f(i)) && (ys1 <= pcoord->x2f(i+1)))
      i2s1 = i;
    if ((ys2 > pcoord->x2f(i)) && (ys2 <= pcoord->x2f(i+1)))
      i2s2 = i;
  }
  i3s1 = -1; i3s2 = -1;
  for(int i=ks; i<=ke; i++) {
    if ((zs1 > pcoord->x3f(i)) && (zs1 <= pcoord->x3f(i+1)))
      i3s1 = i;
    if ((zs2 > pcoord->x3f(i)) && (zs2 <= pcoord->x3f(i+1)))
      i3s2 = i;
  }
  if ((i1s1 < 0) || (i2s1 < 0) || (i3s1 < 0) || (i1s2 < 0) || (i2s2 < 0) || (i3s2 < 0)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in MonteCarloProblemGenerator" << std::endl
        << "Origin not found within domain." << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

}

//========================================================================================
//! \fn void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype)
//! \brief Initializes Photon packets before integration
//========================================================================================

void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype) {

  for (int ip=ips; ip<=ipe; ip++) {

    // Set status flag
    pphot->statp[ip] = EVOLVING;

    // choose source randomly based on relative luminosities
    if (pran->uniform() < lfrac) {
      pphot->x1p[ip] = xs1;
      pphot->x2p[ip] = ys1;
      pphot->x3p[ip] = zs1;
      pphot->i1p[ip] = i1s1;
      pphot->i2p[ip] = i2s1;
      pphot->i3p[ip] = i3s1;
    } else {
      pphot->x1p[ip] = xs2;
      pphot->x2p[ip] = ys2;
      pphot->x3p[ip] = zs2;
      pphot->i1p[ip] = i1s2;
      pphot->i2p[ip] = i2s2;
      pphot->i3p[ip] = i3s2;
    }
    int i1 = pphot->i1p[ip];
    int i2 = pphot->i2p[ip];
    int i3 = pphot->i3p[ip];
    // Set weight and energy
    pphot->wp[ip] = 1.0;
    pphot->ep[ip] = 1.602176634e-12;

    // Initialize Stokes vector
    pphot->sip[ip] = 1.0;
    pphot->sup[ip] = 0.0;
    pphot->sqp[ip] = 0.0;
    pphot->svp[ip] = 0.0;

    // Generate initial angle parameters
    Real phi = 2. * PI * pran->uniform();
    //Real phi =0.;
    Real cphi = cos(phi);
    Real sphi = sin(phi);
    Real cth = 2. * pran->uniform() - 1.;
    //Real cth = 0.;
    Real sth = sqrt(1. - SQR(cth));

    // Initialize wave vector with isotropic distribution
    pphot->k1p[ip] = sth*cphi;
    pphot->k2p[ip] = sth*sphi;
    pphot->k3p[ip] = cth;

    // Initialize the absorption and scattering extinction coefficients
    // to the values appropriate in the emitted zone
    pphot->acp[ip] = AbsorptionOpacity(this,pphot,ip);
    pphot->scp[ip] = ScatteringOpacity(this,pphot,ip);

    //pphot->PrintPhoton(ip);
  } //loop over ip

}


namespace {

Real DensityPoints(Real x, Real y, Real z) {

  Real w2 = SQR(x) + SQR(y);
  Real w = sqrt(w2);
  Real r2 = w2 + SQR(z);
  Real r = sqrt(r2);

  Real h,rho, h0 = 7.;
  if ( (r > 200.) && (r < 800.) ) {
    h = h0 * pow((w / 100.),1.25);
    rho = exp(-0.5*SQR(z/h))/w2*2.4e-11;
  } else
    rho = 1.e-30;

  return rho;
}

} //namespace
