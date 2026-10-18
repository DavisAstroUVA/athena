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
#include "../monte_carlo/tetrad.hpp"

#if !MONTE_CARLO_ENABLED
#error "This problem requires monte carlo"
#endif

namespace {
  // Photon list slots for the frame diagnostic.  vel is a four-velocity normalized with
  // the metric at the cell center, but every consumer contracts it with the metric at the
  // photon, where u.u is only -1 to first order in the offset.  IUUDEV carries the
  // running maximum of |u.u + 1| along each path and IUURAD the radius where it peaked,
  // which is what tst/montecarlo/kerr_frames reports.  Snake cannot show this -- it has
  // g_tt = -1 and a static fluid, so u.u is -1 identically -- hence a Kerr-Schild case.
  const int IUUDEV = 0;
  const int IUURAD = 1;
  // Same quantity evaluated at the photon's cell center rather than at the photon.  That
  // is the frame ComputeTransformations and ComovingFrameMatrix build the per-cell moment
  // matrices on, so this slot guards the moments while IUUDEV guards the transport.  Both
  // should sit at roundoff; they are separate slots because they are separate frames, and
  // the split between them is deliberate (see UpdateMoments).
  const int IUUCEN = 2;
  const int NUSER_FRAMES = 3;

  // Global variables
  bool tnorm;
  Real logemin, logemax;
  Real DensityProfile(Real x, Real xl, Real xh, Real taul, Real tauh, Real kap);
  void JMeanOpacity(MonteCarloBlock *pmcb, Photon *pphot, int ip, int imom,
                  const PhotonFrameState &s);
  void AverageEnergy(MonteCarloBlock *pmcb, Photon *pphot, int ip, int imom,
                  const PhotonFrameState &s);

  Real abh, r_hor;
  //functions
  void InsideHorizon(MonteCarloBlock *pmcb, Photon *pphot, PhotonPusher *ppusher,int ip);
  void ComputeMetricAndInverse(Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv);
}


//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief monte carlo test problem generator
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  if (std::strcmp(COORDINATE_SYSTEM, "kerr-schild") != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Problem Generator\n"
        << "GR MC Isoth only supports Kerr-Schild coordinates" << std::endl;
    ATHENA_ERROR(msg);
    return;
  }
  

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
  if (constdens && (tau > 0.)) {
    Real length;
    length = pin->GetReal("mesh","x1max");
    
    rho = tau / (kappaes * length);
  }

  Real xlow, xhigh;
  if (!constdens) {
      bool radial = pin->GetOrAddBoolean("problem","radial","true");
      if (radial) {
        xlow = pin->GetReal("mesh","x1min");
        xhigh = pin->GetReal("mesh","x1max");
      } 
  }

  // Assume constant velocity provided as fraction of speed of light
  Real vel = pin->GetOrAddReal("problem","velocity",0.);
  Real c = 2.99792458e10;
  vel *= c;

  

  // Assume constant temperature and ideal gas
  Real gamma = peos->GetGamma();
  // Gas constant of the default hydrogen-helium mixture, so the temperature the
  // Monte Carlo inverts from this pressure is the one written here.
  Real rideal = MonteCarloBlock::GasConstant(pin->GetOrAddReal("problem","heabund",0.09));
  Real tgas = pin->GetReal("problem","temp");
  // Set initial conditions
  //only accept spherical polar coordinates
  bool radial = pin->GetOrAddBoolean("problem","radial","true");
  Real kB = 1.3806488e-16;


  AthenaArray<Real> b;
  b.NewAthenaArray(3, ncells3, ncells2, ncells1);
  
  
  AthenaArray<Real> glower;
  glower.NewAthenaArray(4,4, ncells1);
  AthenaArray<Real> gupper;
  gupper.NewAthenaArray(4,4, ncells1);

  //if we have ghost cells, include those in initialization 

  int il = is - NGHOST;
  int iu = ie + NGHOST;
  int jl = js;
  int ju = je;

  if (block_size.nx2 > 1) {
    jl -= NGHOST;
    ju += NGHOST;
  }

  int kl = ks;
  int ku = ke;
  if (block_size.nx3 > 1) {
    kl -= NGHOST;
    ku += NGHOST;
  }

  if (radial) {
      // density varies in the r direction
       // Set index bounds
  

      for (int k=kl; k<=ku; k++) {
        for (int j=jl; j<=ju; j++) {
          // Extract metric and inverse
          pcoord->CellMetric(k,j,il,iu,glower,gupper);

          for (int i=il; i<=iu; i++) {
            b(IB1,k,j,i) = 0.0;
            b(IB2,k,j,i) = 0.0;
            b(IB3,k,j,i) = 0.0;

            Real x1 = pcoord->x1v(i);
            Real x2 = pcoord->x2v(j);
            Real x3 = pcoord->x3v(k);
            if (!constdens) rho = DensityProfile(x1,xlow,xhigh,taumin,taumax,kappaes);

            

            //MRM: transform to normal frame, like gr_torus;L406
            //xs: currently use the quadratic form as gr_bondi:L282
            //Real u1 = vel, u2 = 0.0, u3 = 0.0; //zero velocity in the coordinate frame
            //Real tmp = glower(I11,i)*u1*u1 + 2.0*glower(I12,i)*u1*u2 + 2.0*glower(I13,i)*u1*u3
            //    + glower(I22,i)*u2*u2 + 2.0*glower(I23,i)*u2*u3
            //    + glower(I33,i)*u3*u3;
            //Real gammasq = 1.0 + tmp;
            //Real alpha = std::sqrt(-1.0/gupper(I00,i)); 
            //Real b0i = glower(I01,i)*u1 + glower(I02,i)*u2 + glower(I03,i)*u3;
            //Real u0 = sqrt(gammasq)/alpha;
            //(-b0i - sqrt(fmax(SQR(b0i) - glower(I00,i)*gammasq, 0.0)))/glower(I00,i);

            //Real uu1 = u1 - gupper(I01,i)/gupper(I00,i) * u0;
            Real uu1 = 1./std::sqrt(1-2./x1)*2./(x1+2);
            Real uu2 = 0.; //u2 - gupper(I02,i)/gupper(I00,i) * u0;
            Real uu3 = 0.; //u3 - gupper(I03,i)/gupper(I00,i) * u0;

            

            phydro->w(IDN,k,j,i) = rho;
            
            phydro->w(IVX,k,j,i) = uu1;
            phydro->w(IVY,k,j,i) = uu2;
            phydro->w(IVZ,k,j,i) = uu3;
            
            phydro->w(IPR,k,j,i) = rideal*rho*tgas;
          }
        }
      }
  }

 
  // Initialize conserved
  peos->PrimitiveToConserved(phydro->w, b, phydro->u, pcoord, il, iu, jl, ju,
                             kl, ku);
  

}

//========================================================================================
//! \fn void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe)
//! \brief Initializes Photon packets before integration
//========================================================================================

void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype) {

  // Set initial cells and emission weights for all photon samples
  //SetEmissionCellWeight(pphot,ips,ipe);

  for (int ip=ips; ip<=ipe; ip++) {

    // Obtain initial position within cell
    //temp GetZonePosition(pphot,pran,pcoord,ip);

    // Set maximum integration time
    pphot->dtp[ip] = pphot->pmy_mcb->pmy_mc->tmax;

    // Frame diagnostic starts at zero, which is the answer a run in flat spacetime, or
    // one with the velocity rebuilt at the photon, should give.  Zero is a real value
    // here, not a sentinel.
    if (pphot->nuser_var > IUUDEV) pphot->user[IUUDEV][ip] = 0.0;
    if (pphot->nuser_var > IUURAD) pphot->user[IUURAD][ip] = pphot->x1p[ip];
    if (pphot->nuser_var > IUUCEN) pphot->user[IUUCEN][ip] = 0.0;

    if (pmy_mc->emission_flag == EMISFF) {
      // Obtain intitial energy, polarization, direction and weight
      // Utilize free-free emission function in emission.cpp
      if(tnorm) {
        Real logtg = log(tgas(pphot->i3p[ip],pphot->i2p[ip],pphot->i1p[ip]));
        PhotonEmitFreeFree(this,pphot,logemin+logtg,logemax+logtg,ip);
      } else{
        PhotonEmitFreeFree(this,pphot,logemin,logemax,ip);
      }
    }

    pphot->wp[ip] = 1.0;
    // k0p carries the photon energy, so setting ep is what fixes k0p too.
    pphot->ep[ip] = 1.e-10;
    pphot->k1p[ip] = 1.;
    pphot->k2p[ip] = 0.;
    pphot->k3p[ip] = 0.;
    pphot->i1p[ip] = 2;
    pphot->i2p[ip] = 2;
    pphot->i3p[ip] = 2;
    pphot->x0p[ip] = 0;
    pphot->x1p[ip] = 3.000001;
    pphot->x2p[ip] = 0.01;
    pphot->x3p[ip] = 0.01;

    if (pphot->IsNanPhoton(ip))
      pphot->PrintPhoton("initialization: ",ip);

    //TEGAN: understand more of what this is doing!!!
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

    // Set status flag
    if (pphot->wp[ip] < 0.0)
      pphot->statp[ip] = DESTROYED;
    else
      pphot->statp[ip] = EVOLVING;

    // initialize scattering number
    pphot->nscp[ip] = 0;

    // Initialize the absorption and scattering extinction coefficients
    // to the values appropriate in the emitted cell
    pphot->acp[ip] = AbsorptionOpacity(this,pphot,ip);
    pphot->scp[ip] = ScatteringOpacity(this,pphot,ip);
  }
  //pphot->nphot++;

}

//========================================================================================
//! \fn void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin)
//! \brief Analogous to problem generator but used in support of InitializePhoton
//========================================================================================

void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin) {

  nphremain = pin->GetInteger64("montecarlo","nphot");

}

//========================================================================================
//! \fn void MonteCarlo::InitUserMonteCarloData(ParameterInput *pin)
//! \brief Initializes user data specific to MonteCarlo class
//========================================================================================

void MonteCarlo::InitUserMonteCarloData(ParameterInput *pin){

  nuser_var = NUSER_FRAMES;
  AllocateUserMoments(2);
  EnrollUserMoment(0, JMeanOpacity, "kapJ");
  EnrollUserMoment(1, AverageEnergy, "eave");

  abh = pin->GetReal("coord","a");
  //abh = 0.0;
  // assumes mbh = 1 in code units
  r_hor = 1.0 + sqrt(1.0 - SQR(abh));
  EnrollUserWorkInMove(InsideHorizon);

  // The tabulated opacities are the module's (emission = table, absorption = table,
  // <montecarlo>/opacity_file)
  const std::string emis = pin->GetOrAddString("montecarlo","emission","none");
  const std::string absn = pin->GetOrAddString("montecarlo","absorption","none");
  if (emis == "user" || absn == "user") {
    std::stringstream msg;
    msg << "### FATAL ERROR in MonteCarlo::InitUserMonteCarloData" << std::endl
        << "mc_gr_simple has no user emission or absorption; use "
        << "emission = table and absorption = table with <montecarlo>/opacity_file"
        << std::endl;
    ATHENA_ERROR(msg);
  }
}

/*void Mesh::InitUserMeshData(ParameterInput *pin) {

  EnrollUserMetric(SphericalKerrSchild);
}*/

//========================================================================================
//! \fn void MonteCarloBlock::FinalizePhoton(Photon *pphot, int ip)
//! \brief Complete work at end of photon packets before integration
//========================================================================================

void MonteCarloBlock::FinalizePhoton(Photon *pphot, int ip) {

  TransformToComoving(pphot,ip,ip);

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

void InsideHorizon(MonteCarloBlock *pmcb, Photon *pphot, PhotonPusher *ppusher, int ip) {

  //already in SKS units so don't need to adjust r to consider spin
  Real x1 = pphot->x1p[ip];

  if (x1 < r_hor) {
    pphot->statp[ip] = ABSORBED;
    printf("Photon absorbed inside horizon at r=%g\n",x1);
  }

  // Record how far the cell-centred four-velocity misses being a unit vector where it is
  // actually used.  vel was normalized against the metric at the cell center; contracting
  // it with the metric here is exactly what FrequencyShiftComoving and the two transform
  // routines do, so this is the error they inherit, not a synthetic one.
  const int i1 = pphot->i1p[ip], i2 = pphot->i2p[ip], i3 = pphot->i3p[ip];

  const bool active = (i1 >= pmcb->is && i1 <= pmcb->ie &&
                       i2 >= pmcb->js && i2 <= pmcb->je &&
                       i3 >= pmcb->ks && i3 <= pmcb->ke);

  // Ghost cells are skipped, but no longer because vel is zero there -- GetVelocity and
  // the other fillers now cover ghosts.  They are skipped because a photon holding ghost
  // indices sits a long way from that cell's center, at a distance set by the angular
  // grid rather than the radial one, so including them contributes an offset error the
  // radial refinement below cannot resolve and that swamps the O(1e-2) being measured.
  // Sampling them reports a flat 5.2e-1 at every nx1.  Measuring that properly needs a
  // scan in nx2 and nx3, which this test does not do.
  if (pphot->nuser_var > IUUCEN && active) {
    Real x[4] = {pphot->x0p[ip], pphot->x1p[ip], pphot->x2p[ip], pphot->x3p[ip]};
    Real gcov[4][4];
    pmcb->pcoord->Metric(x, gcov);

    // Whatever the frame transformations use.  Before the reconstruction this read vel
    // straight from the cell center and reported O(1e-2); FluidFourVelocity rebuilds it
    // here, and u.u = -1 should then hold to roundoff at every resolution.
    Real ucon[4];
    pmcb->FluidFourVelocity(x, i3, i2, i1, ucon);

    Real dev = std::fabs(DotVec(ucon, ucon, gcov) + 1.0);
    if (dev > pphot->user[IUUDEV][ip]) {
      pphot->user[IUUDEV][ip] = dev;
      pphot->user[IUURAD][ip] = x1;
    }

    // The moment frames are built at the cell center, not here, so check that point too.
    Real xc[4];
    xc[IMC0] = x[IMC0];
    xc[IMC1] = pmcb->pmy_block->pcoord->x1v(i1);
    xc[IMC2] = pmcb->pmy_block->pcoord->x2v(i2);
    xc[IMC3] = pmcb->pmy_block->pcoord->x3v(i3);
    Real gcen[4][4], ucen[4];
    pmcb->pcoord->Metric(xc, gcen);
    pmcb->FluidFourVelocity(xc, i3, i2, i1, ucen);
    Real devc = std::fabs(DotVec(ucen, ucen, gcen) + 1.0);
    if (devc > pphot->user[IUUCEN][ip]) pphot->user[IUUCEN][ip] = devc;
  }
  /*Real keverg = 1.602176634e-9;
  if (pphot->ep[ip] > 2.e3*keverg)
    pphot->statp[ip] = REMOVED;
  */
    
  return;
}

void ComputeMetricAndInverse(Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv){
      Real a = pin->GetReal("coord", "a");

      Real r = x1;
      Real r2 = SQR(r);
      Real th = x2;
      Real sth = sin(th);
      Real cth = cos(th);
      Real cth2 = SQR(cth);
      Real sth2 = SQR(sth);
      Real a2 = SQR(a);
      Real phi = x3;

      Real sigma = r2 + a2 * cth2;
      Real delta = r2 - 2. * r + a2;
      Real A = SQR(r2 + a2) - a2 * delta * sth2;

      // Calculate covariant components
      g(I00) = -1. * (1. - 2. * r / sigma);
      g(I01) = 2. * r / sigma;
      g(I02) = 0.;
      g(I03) = -2. * a * r * sth2 / sigma;

      //g(I10) = g(I01);
      g(I11) = 1. + 2. * r / sigma;
      g(I12) = 0.;
      g(I13) = -a * sth2 * (1. + 2. * r / sigma);

      //g(I20) = g(I02);
      //g(I21) = g(I12);
      g(I22) = sigma;
      g(I23) = 0.;

      //g(I30) = g(I03);
      //g(I31) = g(I13);
      //g(I32) = g(I23);
      g(I33) = A * sth2 / sigma;

      // Calculate contravariant components
      g_inv(I00) = -(1. + (2. * r / sigma));
      g_inv(I01) = 2. * r / sigma;
      g_inv(I02) = 0.0;
      g_inv(I03) = 0.0;

      //g_inv(I10) = g_inv(I01);
      g_inv(I11) = delta / sigma;
      g_inv(I12) = 0.0;
      g_inv(I13) = a / sigma;

      //g_inv(I20) = g_inv(I02);
      //g_inv(I21) = g_inv(I12);
      g_inv(I22) = 1. / sigma;
      g_inv(I23) = 0.0;

      //g_inv(I30) = g_inv(I03);
      //g_inv(I31) = g_inv(I13);
      //g_inv(I32) = g_inv(I23);
      g_inv(I33) = 1. / (sigma * sth2);
      return;
    }

}  // namespace
