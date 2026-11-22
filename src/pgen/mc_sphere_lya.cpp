//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mc_sphere_lya.cpp
//! \brief Monte Carlo through a uniform isothermal medium of neutral hydrogen with Lyman
//!        alpha scattering: a point source at the origin and an escape surface at
//!        <problem> radius.  The medium is the cells whose centres lie inside the radius
//!        (the rest of the mesh is empty to a factor 1e-20), so that every medium cell
//!        can be a cell of the DDMC lattice and the escape surface runs through the
//!        empty cells beyond; with the radius pushed past the mesh the medium fills it
//!        and the escape is at the mesh boundaries instead: a slab when the mesh is one
//!        cell wide in two directions (periodic there, escape in the third).  In
//!        spherical polar coordinates the medium fills the mesh, the source is a shell
//!        at the inner edge x1min, isotropic outward, which a reflecting inner boundary
//!        makes the point source as x1min goes to zero, and the escape is the outer
//!        boundary x1max, the radius.
//========================================================================================

// C/C++ headers
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

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
#include "../globals.hpp"

#if !MONTE_CARLO_ENABLED
#error "This problem requires monte carlo"
#endif

namespace {
  // Global variables
  Real rad0,time0;
  Real energy0;

  // function headers
  void SphericalEscape(MonteCarloBlock *pmcb, Photon *phot, PhotonPusher *ppusher, int ip);
  void TimedEscape(MonteCarloBlock *pmcb, Photon *phot, PhotonPusher *ppusher, int ip);
  void RadialPath(MonteCarloBlock *pmcb, Photon *phot, PhotonPusher *ppusher, int ip);
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief monte carlo test problem generator
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  // Gas constant of the default hydrogen-helium mixture, so the temperature the
  // Monte Carlo inverts from this pressure is the one written here.
  Real rideal = MonteCarloBlock::GasConstant(pin->GetOrAddReal("problem","heabund",0.09));
  Real c = MCConstants::c_cgs;
  Real temp = pin->GetReal("problem","temp");
  Real tau = pin->GetReal("problem","tau");
  Real rad0;
  if (COORDINATE_SYSTEM == "cartesian") {
    rad0 = pin->GetReal("problem","radius");
  } else if (COORDINATE_SYSTEM == "spherical_polar") {
    rad0 = pmy_mesh->mesh_size.x1max;
  }
  Real vel = pin->GetOrAddReal("problem","velocity",0.);
  // uniform: the medium moves along z at velocity; homologous: an expansion,
  // v = velocity c r / radius radially, velocity the value at the surface
  std::string profile = pin->GetOrAddString("problem","velocity_profile","uniform");
  if (profile != "uniform" && profile != "homologous") {
    std::stringstream msg;
    msg << "### FATAL ERROR in ProblemGenerator" << std::endl
        << "velocity_profile = " << profile << " is not uniform or homologous" << std::endl;
    ATHENA_ERROR(msg);
  }
  const bool homologous = (profile == "homologous");
  Real gamma = peos->GetGamma();
  vel *= c;

  // the line-centre optical depth tau over the radius sets the density: the
  // line-centre cross section with H(a, 0) = 1, and species(0) = rho / m_p
  Real mass = MCConstants::mp_cgs;
  Real vth = std::sqrt(2.*MCConstants::kb_cgs*temp/MCConstants::mH_cgs);
  Real dopw = MCConstants::nu_lya*vth/c;
  Real kappa = ResLinePre()/(mass*std::sqrt(PI)*dopw);
  Real rho0 = tau/(kappa*rad0);
  // the cells whose centres lie outside the radius hold 1e-20 of the density, which the
  // floors of the equation of state must admit (the default dfloor is 3.5e-18 in cgs)
  bool empty = false;
  if (COORDINATE_SYSTEM == "cartesian") {
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          if (std::sqrt(SQR(pcoord->x1v(i)) + SQR(pcoord->x2v(j)) + SQR(pcoord->x3v(k)))
              >= rad0)
            empty = true;
        }
      }
    }
  }
  if (empty) {
    const Real rim = 1.e-20*rho0;
    const Real floor0 = std::sqrt(1024*std::numeric_limits<float>::min());
    const Real dfloor = pin->GetOrAddReal("hydro","dfloor",floor0);
    const Real pfloor = pin->GetOrAddReal("hydro","pfloor",floor0);
    if (dfloor >= rim || pfloor >= rideal*rim*temp) {
      std::stringstream msg;
      msg << "### FATAL ERROR in ProblemGenerator" << std::endl
          << "<hydro> dfloor and pfloor must lie below the density " << rim
          << " and pressure " << rideal*rim*temp << " of the cells outside the radius"
          << std::endl;
      ATHENA_ERROR(msg);
    }
  }
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        // the medium is the cells whose centres lie inside the radius
        Real rc = 0.;
        if (COORDINATE_SYSTEM == "cartesian")
          rc = std::sqrt(SQR(pcoord->x1v(i)) + SQR(pcoord->x2v(j)) + SQR(pcoord->x3v(k)));
        const Real rho = (rc < rad0) ? rho0 : 1.e-20*rho0;
        phydro->u(IDN,k,j,i) = rho;
        if (COORDINATE_SYSTEM == "spherical_polar") {
          // the momenta are the physical components along r, theta and phi
          if (homologous) {
            phydro->u(IM1,k,j,i) = rho*vel*pcoord->x1v(i)/rad0;
            phydro->u(IM2,k,j,i) = 0.0;
          } else {
            phydro->u(IM1,k,j,i) = rho*vel*std::cos(pcoord->x2v(j));
            phydro->u(IM2,k,j,i) = -rho*vel*std::sin(pcoord->x2v(j));
          }
          phydro->u(IM3,k,j,i) = 0.0;
        } else if (homologous) {
          const Real f = rho*vel/rad0;
          phydro->u(IM1,k,j,i) = f*pcoord->x1v(i);
          phydro->u(IM2,k,j,i) = f*pcoord->x2v(j);
          phydro->u(IM3,k,j,i) = f*pcoord->x3v(k);
        } else {
          phydro->u(IM1,k,j,i) = 0.0;
          phydro->u(IM2,k,j,i) = 0.0;
          phydro->u(IM3,k,j,i) = rho*vel;
        }
        phydro->u(IEN,k,j,i) = rideal*rho*temp/(gamma-1.0);
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
      }
    }
  }
}

//========================================================================================
//! \fn void MonteCarlo::InitUserMonteCarloData(ParameterInput *pin)
//! \brief Initializes user data specific to MonteCarlo class
//========================================================================================

void MonteCarlo::InitUserMonteCarloData(ParameterInput *pin){

  nuser_var = 3;
  // With <problem> time set the photons stop at that path length (c = 1 in the
  // pushers); otherwise at the sphere
  Real time = pin->GetOrAddReal("problem","time",-1.);
  if (COORDINATE_SYSTEM == "spherical_polar") {
    EnrollUserWorkInMove(RadialPath);
  } else if (time > 0.) {
    EnrollUserWorkInMove(TimedEscape);
  } else {
    EnrollUserWorkInMove(SphericalEscape);
  }
}

//========================================================================================
//! \fn void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin)
//! \brief Analogous to problem generator but used in support of InitializePhoton
//========================================================================================

void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin) {

  // the source energy, x0 Doppler widths from the line centre at the gas temperature
  Real x0 = pin->GetReal("problem","x0");
  Real temp = pin->GetReal("problem","temp");
  Real vth = std::sqrt(2.*MCConstants::kb_cgs*temp/MCConstants::mH_cgs);
  Real dopw = MCConstants::nu_lya*vth/MCConstants::c_cgs;
  energy0 = MCConstants::h_cgs*(MCConstants::nu_lya + dopw*x0);

  time0 = pin->GetOrAddReal("problem","time",-1.);

  // With no emission function the block has to be given its photon count
  const int64_t nphot = pin->GetInteger64("montecarlo","nphot");
  if (COORDINATE_SYSTEM == "cartesian") {
    rad0 = pin->GetReal("problem","radius");
    // Whether this block holds the origin, which is assumed to be a cell centre when
    // more than one cell is specified in a direction; the point source goes to that block
    int i1start = -1, i2start = -1, i3start = -1;
    for (int i=is; i<=ie; i++)
      if ((0. > pcoord->x1f(i)) && (0. <= pcoord->x1f(i+1))) i1start = i;
    for (int j=js; j<=je; j++)
      if ((0. > pcoord->x2f(j)) && (0. <= pcoord->x2f(j+1))) i2start = j;
    for (int k=ks; k<=ke; k++)
      if ((0. > pcoord->x3f(k)) && (0. <= pcoord->x3f(k+1))) i3start = k;
    const bool has_origin = (i1start >= 0) && (i2start >= 0) && (i3start >= 0);
    nphremain = has_origin ? nphot : 0;
  } else {
    // the shell source at the mesh's inner edge, shared among the blocks there in
    // proportion to their solid angle
    const Mesh *pm = pmy_block->pmy_mesh;
    rad0 = pm->mesh_size.x1max;
    const bool inner = (pcoord->x1f(is) <= pm->mesh_size.x1min*(1. + 1.e-12));
    const Real dcos = std::cos(pcoord->x2f(js)) - std::cos(pcoord->x2f(je+1));
    const Real dphi = pcoord->x3f(ke+1) - pcoord->x3f(ks);
    nphremain = inner ? static_cast<int64_t>(std::llround(nphot*dcos*dphi/(4.*PI))) : 0;
  }
  nphrun = 0;

}


//========================================================================================
//! \fn void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype)
//! \brief Initializes Photon packets before integration
//========================================================================================

void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype) {

  // The cell of this block that holds the origin, for the point source.  Found here
  // and not kept from the problem generator, which runs for every block and would
  // leave the last block's values
  int i1origin = -1, i2origin = -1, i3origin = -1;
  if (COORDINATE_SYSTEM == "cartesian") {
    for (int i=is; i<=ie; i++)
      if ((0. > pcoord->x1f(i)) && (0. <= pcoord->x1f(i+1))) i1origin = i;
    for (int j=js; j<=je; j++)
      if ((0. > pcoord->x2f(j)) && (0. <= pcoord->x2f(j+1))) i2origin = j;
    for (int k=ks; k<=ke; k++)
      if ((0. > pcoord->x3f(k)) && (0. <= pcoord->x3f(k+1))) i3origin = k;
    if (ipe >= ips && (i1origin < 0 || i2origin < 0 || i3origin < 0)) {
      std::stringstream msg;
      msg << "### FATAL ERROR in MonteCarloBlock::InitializePhoton" << std::endl
          << "point-source photons on a block that does not hold the origin" << std::endl;
      ATHENA_ERROR(msg);
    }
  }
  for (int ip=ips; ip<=ipe; ip++) {

    // user[0] birth energy (set below), user[1] scattering count at escape,
    // user[2] unweighted path length
    pphot->user[1][ip] = 0.;
    pphot->user[2][ip] = 0.;

    if (COORDINATE_SYSTEM == "cartesian") {
      // the point source at the origin, isotropic
      pphot->i1p[ip] = i1origin;
      pphot->i2p[ip] = i2origin;
      pphot->i3p[ip] = i3origin;
      pphot->x1p[ip] = 0.;
      pphot->x2p[ip] = 0.;
      pphot->x3p[ip] = 0.;
      pphot->x0p[ip] = 0.; //time

      Real phi = 2. * PI * pran->uniform();
      Real cphi = cos(phi);
      Real sphi = sin(phi);
      Real cth = 2. * pran->uniform() - 1.;
      Real sth = sqrt(1. - SQR(cth));
      pphot->k1p[ip] = sth*cphi;
      pphot->k2p[ip] = sth*sphi;
      pphot->k3p[ip] = cth;

    } else if (COORDINATE_SYSTEM == "spherical_polar") {
      // the shell source at the inner edge: uniform in solid angle over the block,
      // isotropic over the outward hemisphere (the reflecting inner boundary turns the
      // rest around on the spot), the wavevector in the local basis (r, theta, phi)
      const Real cthh = std::cos(pcoord->x2f(js)), cthl = std::cos(pcoord->x2f(je+1));
      const Real theta = std::acos(cthl + pran->uniform()*(cthh - cthl));
      const Real phi = pcoord->x3f(ks)
                       + pran->uniform()*(pcoord->x3f(ke+1) - pcoord->x3f(ks));
      pphot->i1p[ip] = is;
      pphot->i2p[ip] = js;
      while (pphot->i2p[ip] < je && theta >= pcoord->x2f(pphot->i2p[ip]+1)) pphot->i2p[ip]++;
      pphot->i3p[ip] = ks;
      while (pphot->i3p[ip] < ke && phi >= pcoord->x3f(pphot->i3p[ip]+1)) pphot->i3p[ip]++;
      pphot->x1p[ip] = pcoord->x1f(is) + 1.e-6*(pcoord->x1f(is+1) - pcoord->x1f(is));
      pphot->x2p[ip] = theta;
      pphot->x3p[ip] = phi;
      pphot->x0p[ip] = 0.; //time
      const Real mu = pran->uniform();
      const Real ph = 2.*PI*pran->uniform();
      pphot->k1p[ip] = mu;
      pphot->k2p[ip] = std::sqrt(1. - SQR(mu))*std::cos(ph);
      pphot->k3p[ip] = std::sqrt(1. - SQR(mu))*std::sin(ph);
    }

    // weight and energy; a luminosity of 1e20 erg/s shared by the samples
    Real target_lum = 1.e20;
    pphot->wp[ip] = target_lum / energy0;
    pphot->ep[ip] = energy0;

    if (IsPolarized(pphot->polarized)) {
      pphot->sip[ip] = 1.0;
      pphot->sqp[ip] = 0.0;
      pphot->sup[ip] = 0.0;
      pphot->svp[ip] = 0.0;
    }

    // Set status flag and the time budget; with no budget the pusher takes no step and
    // the block loop scatters the photon in place forever
    pphot->dtp[ip] = pmy_mc->tmax;
    if (pphot->wp[ip] < 0.0)
      pphot->statp[ip] = DESTROYED;
    else
      pphot->statp[ip] = EVOLVING;

    pphot->user[0][ip] = pphot->ep[ip];
    pphot->nscp[ip] = 0;

    // Initialize the absorption and scattering extinction coefficients
    // to the values appropriate in the emitted zone
    pphot->acp[ip] = AbsorptionOpacity(this,pphot,ip);
    pphot->scp[ip] = ScatteringOpacity(this,pphot,ip);
  } // loop over ip
}



namespace {

//! the escape surface at rad0: a photon beyond it is backed up to the surface along its
//! direction and escapes.  A walker has no direction and sits at the centre of a medium
//! cell, which is inside the surface, so it is left alone
void SphericalEscape(MonteCarloBlock *pmcb, Photon *pphot, PhotonPusher *ppusher,
                     int ip) {

  pphot->user[2][ip] += ppusher->dl;

  Real r = sqrt(SQR(pphot->x1p[ip])+SQR(pphot->x2p[ip])+SQR(pphot->x3p[ip]));
  if (r >= rad0) {
    Real dr = r-rad0;
    Real kn = sqrt(SQR(pphot->k1p[ip])+SQR(pphot->k2p[ip])+SQR(pphot->k3p[ip]));
    if (kn > 0.) {
      pphot->x0p[ip] -= dr;
      pphot->user[2][ip] -= dr;
      pphot->x1p[ip] -= pphot->k1p[ip]/kn*dr;
      pphot->x2p[ip] -= pphot->k2p[ip]/kn*dr;
      pphot->x3p[ip] -= pphot->k3p[ip]/kn*dr;
    }
    pphot->user[1][ip] = pphot->nscp[ip];
    pphot->statp[ip] = ESCAPED;
  }
}

//! in spherical polar the escape is the mesh's outer boundary; the path and the
//! scattering count are kept in the user columns at every move
void RadialPath(MonteCarloBlock *pmcb, Photon *pphot, PhotonPusher *ppusher, int ip) {
  pphot->user[2][ip] += ppusher->dl;
  pphot->user[1][ip] = pphot->nscp[ip];
}

//! the escape surface, and a stop at the path length time0
void TimedEscape(MonteCarloBlock *pmcb, Photon *pphot, PhotonPusher *ppusher,
                 int ip) {

  pphot->user[2][ip] += ppusher->dl;
  Real r = sqrt(SQR(pphot->x1p[ip])+SQR(pphot->x2p[ip])+SQR(pphot->x3p[ip]));
  Real kn = sqrt(SQR(pphot->k1p[ip])+SQR(pphot->k2p[ip])+SQR(pphot->k3p[ip]));
  if (r >= rad0) {
    Real dr = r-rad0;
    if (kn > 0.) {
      pphot->x0p[ip] -= dr;
      pphot->user[2][ip] -= dr;
      pphot->x1p[ip] -= pphot->k1p[ip]/kn*dr;
      pphot->x2p[ip] -= pphot->k2p[ip]/kn*dr;
      pphot->x3p[ip] -= pphot->k3p[ip]/kn*dr;
    }
    pphot->user[1][ip] = pphot->nscp[ip];
    pphot->statp[ip] = ESCAPED;
  }
  // the path condition (x0p is the path length), so the photon is not carried past
  // time0
  if (pphot->statp[ip] == EVOLVING && pphot->x0p[ip] >= time0) {
    Real dt = pphot->x0p[ip] - time0;
    if (kn > 0.) {
      pphot->x0p[ip] -= dt;
      pphot->user[2][ip] -= dt;
      pphot->x1p[ip] -= pphot->k1p[ip]/kn*dt;
      pphot->x2p[ip] -= pphot->k2p[ip]/kn*dt;
      pphot->x3p[ip] -= pphot->k3p[ip]/kn*dt;
    }
    pphot->user[1][ip] = pphot->nscp[ip];
    pphot->statp[ip] = ESCAPED;
  }
}

} //namespace
