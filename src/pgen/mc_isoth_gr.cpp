//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mc_isoth_gr.cpp
//! \brief Isothermal, optically thick shell in Kerr-Schild coordinates.
//!
//! A narrow shell of gas sits between xlow and xhigh (in gravitational radii) with a
//! density profile chosen so that the electron-scattering optical depth runs from taumin
//! at the top of the shell to taumax at its base.  Free-free emission and absorption
//! inside the shell thermalize the radiation towards a Planck function at the local gas
//! temperature; photons that escape through the outer boundary carry the gravitational
//! redshift accumulated between their last interaction and x1max.
//!
//! The gas is at rest with respect to a *static* observer, not the normal (ZAMO)
//! observer, so that the only frequency shift is gravitational.  In Kerr-Schild the
//! normal observer is infalling, so this requires boosts = true: with boosts = false the
//! comoving tetrad is built on the normal observer instead (see
//! MonteCarloBlock::SetNormalObserver) and the spectrum picks up a spurious Doppler
//! shift.  general_pusher = true is also required -- Kerr-Schild always integrates with
//! GeneralPusher, but general_pusher_flag separately selects the four-vector storage
//! convention used by TransformToCoordinate.
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
#include "../monte_carlo/mccoord.hpp"
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
  // lengths passed to DensityProfile are in cm, so that kap [cm^2/g] gives a
  // dimensionless optical depth
  Real DensityProfile(Real x, Real xl, Real xh, Real taul, Real tauh, Real kap);
  Real l_cgs;
  void JMeanOpacity(MonteCarloBlock *pmcb, Photon *pphot, int ip, int imom,
                    const PhotonFrameState &s);
  void AverageEnergy(MonteCarloBlock *pmcb, Photon *pphot, int ip, int imom,
                     const PhotonFrameState &s);

  // frequency table parameters
  std::string emission_type;
  int nfre, nrho, ntem;
  Real lmine, lmaxe, dle, lmint, lmaxt, dlt, lmind, lmaxd, dld;
  Real abh, r_hor;
  AthenaArray<Real> fre_grid;
  AthenaArray<Real> temp_grid;
  AthenaArray<Real> rho_grid;
  AthenaArray<Real> ross_tab;
  AthenaArray<Real> plan_tab;
  AthenaArray<Real> eta_cum_tab;
  AthenaArray<Real> eta_tab;
  AthenaArray<Real> prob_tab;
  //functions
  void InsideHorizon(MonteCarloBlock *pmcb, Photon *pphot, PhotonPusher *ppusher,int ip);
  Real TableOpacity(MonteCarloBlock *pmcb, Photon *pphot, int ip);
  Real IntegrateEmission(Real temp, Real num, Real nup, Real am, Real ap);
  Real Planck(Real temp, Real nu);
  Real TableEmission(MonteCarloBlock *pmcb, int k, int j, int i, int etype);
  Real SampleEmissivity(MonteCarloBlock *pmcb, Photon *pphot, int ip);
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
  

  // Radial extent of the shell.  Defaults to the full domain, which reproduces the
  // atmosphere of the non-relativistic mc_isoth.
  Real x1min = pin->GetReal("mesh","x1min");
  Real x1max = pin->GetReal("mesh","x1max");
  Real xlow = pin->GetOrAddReal("problem","xlow",x1min);
  Real xhigh = pin->GetOrAddReal("problem","xhigh",x1max);

  // Unit conversions.  Coordinates are in gravitational radii, so any optical depth has
  // to be built from lengths multiplied by l_cgs.
  l_cgs = pin->GetOrAddReal("problem","l_cgs",1.);
  Real rho_cgs = pin->GetOrAddReal("problem","rho_cgs",1.);
  Real tgas_cgs = pin->GetOrAddReal("problem","tgas_cgs",-1.);
  Real thick_cgs = (xhigh-xlow) * l_cgs;

  // Determine density via optical depth or constant density
  bool constdens = pin->GetOrAddBoolean("problem","constdens",false);
  Real rho, tau, taumin, taumax;
  Real heabund = 0.09; //hardcode for now
  Real mp = 1.6726e-24;
  Real sigmat = 6.65248e-25;
  Real kappaes = sigmat * (1. + 2.*heabund) / (mp * (1.+4.*heabund) );
  if (constdens) {
    rho = pin->GetOrAddReal("problem","dens",-1.);
    tau = pin->GetOrAddReal("problem","tau",-1.);
    // tau is the scattering optical depth across the shell, not across the domain
    if (tau > 0.)
      rho = tau / (kappaes * thick_cgs);
  } else {
    taumin = pin->GetReal("problem","taumin");
    taumax = pin->GetReal("problem","taumax");
  }

  // Ambient density outside the shell, as a fraction of the density at its base.  This
  // must not be zero: GetTemperature() would form 0/0, and the resulting NaN loses both
  // floor comparisons and lands on tfloor_cgs = 0, at which point GetEmissionFreeFree()
  // returns eta0/sqrt(0) * 0 = NaN and poisons the whole emission array.
  Real ambient = pin->GetOrAddReal("problem","ambient",1.0e-10);
  Real rho_base = constdens ? rho
                : DensityProfile(xlow*l_cgs,xlow*l_cgs,xhigh*l_cgs,taumin,taumax,kappaes);
  Real rho_amb = ambient * rho_base;

  // Assume constant temperature and ideal gas
  Real rideal = 8.314e7;
  Real tgas = pin->GetReal("problem","temp");

  // GetTemperature() recovers T as tconv * w(IPR)/w(IDN), so the pressure has to be
  // written in whatever units tconv inverts.  With tgas_cgs set (= c^2/rideal for the
  // usual GR choice) that ratio is dimensionless; without it the code falls back to
  // 1/rideal and the ratio is cgs.  Writing cgs pressure while tgas_cgs is set
  // overestimates T by c^2/(rideal*T) -- twenty-one orders of magnitude for T = 1e6 K.
  Real tconv = (tgas_cgs > 0.) ? tgas_cgs : 1./rideal;

  AthenaArray<Real> b;
  b.NewAthenaArray(3, ncells3, ncells2, ncells1);

  AthenaArray<Real> glower, gupper;
  glower.NewAthenaArray(NMETRIC, ncells1);
  gupper.NewAthenaArray(NMETRIC, ncells1);

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

  // density varies in the r direction
  for (int k=kl; k<=ku; k++) {
    for (int j=jl; j<=ju; j++) {
      // Extract metric and inverse
      pcoord->CellMetric(k,j,il,iu,glower,gupper);

      for (int i=il; i<=iu; i++) {
        b(IB1,k,j,i) = 0.0;
        b(IB2,k,j,i) = 0.0;
        b(IB3,k,j,i) = 0.0;

        Real x1 = pcoord->x1v(i);
        if (x1 > xlow && x1 < xhigh) {
          if (!constdens)
            rho = DensityProfile(x1*l_cgs,xlow*l_cgs,xhigh*l_cgs,taumin,taumax,kappaes);
        } else {
          rho = rho_amb;
        }

        // Static observer, u^i = 0, expressed in the Athena++ GR primitive velocity
        // uu^i = u^i - g^{i0}/g^{00} u^0 with u^0 = 1/sqrt(-g_00).  Taking this from the
        // metric rather than the a = 0 closed form keeps it correct for a spinning hole.
        if (glower(I00,i) >= 0.) {
          std::stringstream msg;
          msg << "### FATAL ERROR in Problem Generator\n"
              << "cell at r = " << x1 << ", theta = " << pcoord->x2v(j)
              << " lies inside the ergosphere, where no static observer exists.\n"
              << "Move x1min outside r = 1+sqrt(1-a^2 cos^2 theta)." << std::endl;
          ATHENA_ERROR(msg);
        }
        Real u0 = 1./std::sqrt(-glower(I00,i));
        Real uu1 = -gupper(I01,i)/gupper(I00,i) * u0;
        Real uu2 = -gupper(I02,i)/gupper(I00,i) * u0;
        Real uu3 = -gupper(I03,i)/gupper(I00,i) * u0;

        phydro->w(IDN,k,j,i) = phydro->w1(IDN,k,j,i) = rho / rho_cgs;

        phydro->w(IVX,k,j,i) = phydro->w1(IVX,k,j,i) = uu1;
        phydro->w(IVY,k,j,i) = phydro->w1(IVY,k,j,i) = uu2;
        phydro->w(IVZ,k,j,i) = phydro->w1(IVZ,k,j,i) = uu3;

        phydro->w(IPR,k,j,i) = phydro->w1(IPR,k,j,i) =
            phydro->w(IDN,k,j,i) * tgas / tconv;
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
  SetEmissionCellWeight(pphot,ips,ipe);

  for (int ip=ips; ip<=ipe; ip++) {

    // Obtain initial position within zone.  GetZonePosition sets only the spatial
    // components, so the coordinate time has to be zeroed here.
    GetZonePosition(pphot,pran,pcoord,ip);
    pphot->x0p[ip] = 0.0;

    // Set maximum integration time
    pphot->dtp[ip] = pphot->pmy_mcb->pmy_mc->tmax;

    if (emission_type == "freefree") {
      // Obtain intitial energy, polarization, direction and weight
      // Utilize free-free emission function in emission.cpp
      if(tnorm) {
        Real logtg = log(tgas(pphot->i3p[ip],pphot->i2p[ip],pphot->i1p[ip]));
        PhotonEmitFreeFree(this,pphot,logemin+logtg,logemax+logtg,ip);
      } else{
        PhotonEmitFreeFree(this,pphot,logemin,logemax,ip);
      }
    } else {
      pphot->ep[ip] = SampleEmissivity(this,pphot,ip);
      //printf("%g\n",pphot->ep[ip]/1.6e-9);
      // Generate initial angle parameters
      Real phi = 2. * PI * pran->uniform();
      Real cphi = cos(phi);
      Real sphi = sin(phi);
      Real cth = 2. * pran->uniform() - 1.;
      Real sth = sqrt(1. - SQR(cth));
      // Initialize wave vector with isotropic distribution
      pphot->k1p[ip] = sth*cphi;
      pphot->k2p[ip] = sth*sphi;
      pphot->k3p[ip] = cth;
    }

    if (pphot->IsNanPhoton(ip))
      pphot->PrintPhoton("initialization: ",ip);

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
  }

}

//========================================================================================
//! \fn void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin)
//! \brief Analogous to problem generator but used in support of InitializePhoton
//========================================================================================

void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin) {

  if (emission_type == "freefree") {
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
  } else {

    // Create array for emissivity
    AthenaArray<Real> eta_nu_tab;
    eta_nu_tab.NewAthenaArray(nfre,ntem,nrho);
    Real h = 6.62607015e-27;
    for(int k=0; k<nfre; ++k) {
      Real nup = fre_grid(k+1)/h;
      Real num = fre_grid(k)/h;
      for(int j=0; j<ntem; ++j) {
        Real temp = temp_grid(j);
        for(int i=0; i<nrho; ++i) {
          Real opp = plan_tab(k,j,i);
          Real opm;
          if (k == 0)
            opm = 0.;
          else
            opm = plan_tab(k-1,j,i);
          eta_nu_tab(k,j,i) = IntegrateEmission(temp,num,nup,opm,opp);
          //if ((Globals::my_rank == 0)) {
          //    printf("%d %g %g %g %g\n",k,temp_grid(j),rho_grid(i),eta_nu_tab(k,j,i),
          //           plan_tab(k,j,i));
          //}
        }
      }
    }

    eta_tab.NewAthenaArray(ntem,nrho);
    for(int j=0; j<ntem; ++j) {
      Real temp = temp_grid(j);
      for(int i=0; i<nrho; ++i) {
        eta_tab(j,i) = 0.;
        for(int k=0; k<nfre; ++k) {
          eta_tab(j,i) += 4.*PI*eta_nu_tab(k,j,i);
          //printf("%d %d %d %g %g\n",j,i,k,eta_nu_tab(k,j,i),eta_tab(j,i));
        }
      }
    }

    int ncells1 = nx1 + 2*(NGHOST);
    int ncells2 = 1, ncells3 = 1;
    if (nx2 > 1) ncells2 = nx2 + 2*(NGHOST);
    if (nx3 > 1) ncells3 = nx3 + 2*(NGHOST);
    eta_cum_tab.NewAthenaArray(ncells3,ncells2,ncells1,nfre+1);
    Real keverg = 1.602176634e-9;
    for(int k=ks; k<=ke; ++k) {
      for(int j=js; j<=je; ++j) {
        for(int i=is; i<=ie; ++i) {
          Real ld = log10(rho(k,j,i));
          ld = (ld < lmind) ? lmind : ld;
          ld = (ld > lmaxd) ? lmaxd : ld;
          Real temp = tgas(k,j,i);
          Real lt = log10(temp);
          //printf("%d %d %d %g %g %g\n",k,j,i,lt,tgas(k,j,i),rho(k,j,i));
          lt = (lt < lmint) ? lmint : lt;
          lt = (lt > lmaxt) ? lmaxt : lt;
          Real xi = (ld - lmind) / dld;
          int ii = std::floor(xi);
          xi -= static_cast<Real>(ii);
          Real xj = (lt - lmint) / dlt;
          int jj = std::floor(xj);

          while ((jj<ntem-2) && (temp_grid(jj+1) < temp)){
            jj++;
          }
          while ((jj>0) && (temp_grid(jj) > temp)){
            jj--;
          }
          if(jj > ntem-2) {
            jj = ntem-2;
          }
          xj = (temp-temp_grid(jj))/(temp_grid(jj+1)-temp_grid(jj));

          //printf("%d %d %g %g\n",ii,jj,xi,xj);
          eta_cum_tab(k,j,i,0) = 0.;
          for(int l=0; l<nfre; ++l) {
            Real eta = (1.-xi) * ((1.-xj)*eta_nu_tab(l,jj,ii)+xj*eta_nu_tab(l,jj+1,ii))
              + xi * ((1.-xj)*eta_nu_tab(l,jj,ii+1)+xj*eta_nu_tab(l,jj+1,ii+1));
            eta_cum_tab(k,j,i,l+1) =  eta_cum_tab(k,j,i,l)+eta;
            if ((Globals::my_rank == 0) && (k==ks) && (j == js) && (i == is)) {
              printf("%d %g %g %g %g %g %g %g\n",l,fre_grid(l)/keverg,eta_cum_tab(k,j,i,l),
                     eta,eta_nu_tab(l,jj,ii),eta_nu_tab(l,jj+1,ii),eta_nu_tab(l,jj,ii+1),
                     eta_nu_tab(l,jj+1,ii+1));
            }
          }

          for(int l=0; l<nfre+1; ++l) {
            eta_cum_tab(k,j,i,l) /= eta_cum_tab(k,j,i,nfre);

            if ((Globals::my_rank == 0) && (k==ks) && (j == js) && (i == is)) {
              printf("%d %d %g %g\n",l,jj,fre_grid(l)/keverg,eta_cum_tab(k,j,i,l));
            }
          }
        }
      }
    }
    eta_nu_tab.DeleteAthenaArray();
  }

}

//========================================================================================
//! \fn void MonteCarlo::InitUserMonteCarloData(ParameterInput *pin)
//! \brief Initializes user data specific to MonteCarlo class
//========================================================================================

void MonteCarlo::InitUserMonteCarloData(ParameterInput *pin){

  // Kerr-Schild always integrates with GeneralPusher, but general_pusher_flag separately
  // selects the four-vector storage convention: with it false, TransformToCoordinate
  // stores a unit spatial direction plus ep while RK4Step reads k1p..k3p as genuine
  // contravariant components, and the geodesics are integrated on the wrong vector.
  if (!pin->GetOrAddBoolean("montecarlo","general_pusher",false)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in mc_isoth_gr InitUserMonteCarloData" << std::endl
        << "mc_isoth_gr requires general_pusher = true" << std::endl;
    ATHENA_ERROR(msg);
  }
  // The gas is placed at rest relative to a static observer so the only frequency shift
  // is gravitational.  Without boosts that velocity is never read and the comoving tetrad
  // is built on the normal observer, which is infalling in Kerr-Schild.
  if (!pin->GetOrAddBoolean("montecarlo","boosts",false)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in mc_isoth_gr InitUserMonteCarloData" << std::endl
        << "mc_isoth_gr requires boosts = true, otherwise the comoving frame is the"
        << " infalling normal observer and the spectrum picks up a Doppler shift"
        << std::endl;
    ATHENA_ERROR(msg);
  }

  l_cgs = pin->GetOrAddReal("problem","l_cgs",1.);

  nuser_var = 1;
  AllocateUserMoments(2);
  EnrollUserMoment(0, JMeanOpacity, "kapJ");
  EnrollUserMoment(1, AverageEnergy, "eave");

  abh = pin->GetReal("coord","a");
  //abh = 0.0;
  // assumes mbh = 1 in code units
  r_hor = 1.0 + sqrt(1.0 - SQR(abh));
  EnrollUserWorkInMove(InsideHorizon);

  emission_type = pin->GetOrAddString("montecarlo","emission","none");

  if (emission_type == "freefree")
    return;

  // Read in opacity table
  FILE  *opac_file;
  if ( (opac_file=fopen("out_opacity_table_nfreq64.txt","r"))==NULL) {
    std::stringstream msg;
    msg << "FATAL ERROR: Could not open out_opacity_table_nfreq16.txt." << std::endl;
    ATHENA_ERROR(msg);
  }

  fscanf(opac_file,"%d",&(nfre));
  fscanf(opac_file,"%d",&(ntem));
  fscanf(opac_file,"%d",&(nrho));

  // Create arrays for opacity
  fre_grid.NewAthenaArray(nfre+1);
  temp_grid.NewAthenaArray(ntem);
  rho_grid.NewAthenaArray(nrho);
  ross_tab.NewAthenaArray(nfre,ntem,nrho);
  plan_tab.NewAthenaArray(nfre,ntem,nrho);

  if (Globals::my_rank == 0)
    printf("Frequency grid (keV):\n");
  for(int i=1; i<=nfre; ++i){
    fscanf(opac_file,"%lf",&(fre_grid(i)));
    if (Globals::my_rank == 0)
      printf("%d %g\n",i,fre_grid(i));
  }
  fre_grid(0) = fre_grid(1)*fre_grid(1)/fre_grid(2);
  // convert to erg
  Real keverg = 1.602176634e-9;
  for(int i=0; i<=nfre; ++i)
    fre_grid(i) *= keverg;
  lmine = std::log10(fre_grid(1));
  lmaxe = std::log10(fre_grid(nfre-1));
  dle = (lmaxe-lmine)/static_cast<Real>(nfre-2);
  /*if (Globals::my_rank == 0) {
    for(int i=1; i<=nfre; ++i)
      printf("%g \n",fre_grid(i)/keverg);
      }*/
  // temperature grid (keV)
  for(int i=0; i<ntem; ++i){
    fscanf(opac_file,"%lf",&(temp_grid(i)));
  }
  // convert to kelvin
  Real kb = 1.380649e-16;
  if (Globals::my_rank == 0)
    printf("Temperature grid:\n");
  for(int i=0; i<ntem; ++i) {
    temp_grid(i) *= keverg/kb;
    if (Globals::my_rank == 0)
      printf("%d %g\n",i,temp_grid(i));
  }
  lmint = std::log10(temp_grid(0));
  lmaxt = std::log10(temp_grid(ntem-1));
  dlt = (lmaxt-lmint)/static_cast<Real>(ntem-1);
  /*if (Globals::my_rank == 0) {
    printf("temp %g %g %g\n",lmint,lmaxt,dlt);
    for(int i=1; i<ntem; ++i) {
      printf("temp %g\n",temp_grid(i)/temp_grid(i-1));
    }
    }*/

  // density grid (g/cm^3)
  if (Globals::my_rank == 0)
    printf("Density Grid:\n");
  for(int i=0; i<nrho; ++i) {
    fscanf(opac_file,"%lf",&(rho_grid(i)));
    if (Globals::my_rank == 0)
      printf("%d %g\n",i,rho_grid(i));
  }
  lmind = std::log10(rho_grid(0));
  lmaxd = std::log10(rho_grid(nrho-1));
  dld = (lmaxd-lmind)/static_cast<Real>(nrho-1);

  // frequency integrated rosseland mean
  // Read in but not used
  Real buf;
  for(int j=0; j<ntem; ++j) {
    for(int i=0; i<nrho; ++i) {
      fscanf(opac_file,"%lf",&buf);
    }
  }

  // frequency integrated planck mean
  // Read in but not used
  for(int j=0; j<ntem; ++j) {
    for(int i=0; i<nrho; ++i) {
      fscanf(opac_file,"%lf",&buf);
    }
  }

  // ross mean for each frequency group
  for(int k=0; k<nfre; ++k) {
    for(int j=0; j<ntem; ++j) {
      for(int i=0; i<nrho; ++i) {
        fscanf(opac_file,"%lf",&(ross_tab(k,j,i)));
        ross_tab(k,j,i) *= rho_grid(i);
      }
    }
  }
  /*if (Globals::my_rank == 0) {
    for(int j=0; j<ntem; ++j) {
      for(int i=0; i<nrho; ++i) {
        for(int k=0; k<nfre; ++k) {
          printf("%d %d %d %g\n",j,i,k,ross_tab(k,j,i)/rho_grid(i));
      }
    }
    }

  }*/
  // planck mean for each frequency group
  for(int k=0; k<nfre; ++k) {
    for(int j=0; j<ntem; ++j) {
      for(int i=0; i<nrho; ++i) {
        fscanf(opac_file,"%lf",&(plan_tab(k,j,i)));
        plan_tab(k,j,i) *= rho_grid(i);
      }
    }
  }


  bool useff = pin->GetOrAddBoolean("problem","useff",false);
  if (useff) {
    Real dummy;
    for(int k=0; k<nfre; ++k) {
      Real ffnrm = 3.692146e8;
      Real heabund = 0.09; //hardcode for now (should be parameter)
      Real mp = 1.67262192369e-24;
      Real h = 6.62607015e-27;
      Real kb = 1.380649e-16;
      Real nu = fre_grid(k) / h;
      for(int j=0; j<ntem; ++j) {
        Real tgas = temp_grid(j);
        Real ehnu = exp(-h*nu / (kb * tgas) );
        for(int i=0; i<nrho; ++i) {
          Real nh = rho_grid(i) / (mp*(1.+4.*heabund));
          Real nhe = nh*heabund;
          Real ne = nh + 2.*nhe;
          fscanf(opac_file,"%lf",&(dummy));
          Real aff = ffnrm/sqrt(tgas)/pow(nu,3);
          Real opac = ne * (nh + 4. * nhe) * aff * (1. - ehnu);
          plan_tab(k,j,i) = opac;
        }
      }
    }
  }


  // planck mean for each frequency group
  for(int j=0; j<ntem; ++j) {
    for(int i=0; i<nrho; ++i) {
      Real min = 1.e40;
      Real max = 1.e-40;
      for(int k=0; k<nfre; ++k) {
        min = (min > plan_tab(k,j,i)) ? plan_tab(k,j,i) : min;
        max = (max < plan_tab(k,j,i)) ? plan_tab(k,j,i) : max;
      }
      if (max/min < 1.2) {
        for(int k=0; k<nfre; ++k) {
          //if ((j == 15) || (j == 16)) {
          //  printf("%g %g\n",temp_grid(j),rho_grid(i));
          //}
          plan_tab(k,j,i) = 1.e-60;
        }
      }
    }
  }

  fclose(opac_file);

  EnrollUserEmissionFunction(TableEmission);
  EnrollUserOpacityFunction(TableOpacity,true);
}

/*void Mesh::InitUserMeshData(ParameterInput *pin) {

  EnrollUserMetric(SphericalKerrSchild);
}*/

//========================================================================================
//! \fn void MonteCarloBlock::FinalizePhoton(Photon *pphot, int ip)
//! \brief Complete work at end of photon packets before integration
//========================================================================================

void MonteCarloBlock::FinalizePhoton(Photon *pphot, int ip) {

}

namespace {
//----------------------------------------------------------------------------------------
//! \fn Real DensityProfile(Real x, Real xl, Real xh, Real taul, Real tauh, Real kap)
//! \brief exponential atmosphere with scattering optical depth taul at xh, tauh at xl
//!
//! All lengths are in cm so that kap [cm^2/g] yields a dimensionless optical depth; the
//! caller multiplies coordinates by l_cgs.  Integrating kap*rho inward from xh gives
//! tau(x) = taul*(exp((xh-x)/l0) - 1), so tau(xl) = tauh - taul.

Real DensityProfile(Real x, Real xl, Real xh, Real taul, Real tauh, Real kap) {

  Real l0 = (xh-xl) / log(tauh/taul);
  return taul/l0/kap*exp((xh-x)/l0);
}

//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
//! \fn void JMeanOpacity(...)
//! \brief opacity-weighted mean intensity
//!
//! s carries the photon energy and path length in the frame this moment was enrolled
//! with, so the affine-parameter conversion no longer has to be rebuilt here.  The
//! PathWeight() helper this replaced was a coordinate-frame estimator that omitted the
//! tetrad time component, and was documented as a diagnostic rather than a
//! frame-consistent energy density.

void JMeanOpacity(MonteCarloBlock *pmcb, Photon *pphot, int ip, int imom,
                  const PhotonFrameState &s) {

  int i1 = pphot->i1p[ip];
  int i2 = pphot->i2p[ip];
  int i3 = pphot->i3p[ip];

  Real weight = pphot->wp[ip]*s.e*s.dl/MCConstants::c_cgs;
  pmcb->moments_user(imom,i3,i2,i1) += weight*pphot->acp[ip];

}

//----------------------------------------------------------------------------------------
//! \fn void AverageEnergy(...)
//! \brief energy-weighted mean photon energy

void AverageEnergy(MonteCarloBlock *pmcb, Photon *pphot, int ip, int imom,
                   const PhotonFrameState &s) {

  int i1 = pphot->i1p[ip];
  int i2 = pphot->i2p[ip];
  int i3 = pphot->i3p[ip];

  Real weight = pphot->wp[ip]*s.e*s.dl/MCConstants::c_cgs;
  pmcb->moments_user(imom,i3,i2,i1) += weight*s.e;

}

void InsideHorizon(MonteCarloBlock *pmcb, Photon *pphot, PhotonPusher *ppusher, int ip) {

  //already in SKS units so don't need to adjust r to consider spin
  Real x1 = pphot->x1p[ip];

  if (x1 < r_hor) {
    pphot->statp[ip] = ABSORBED;
  }
  /*Real keverg = 1.602176634e-9;
  if (pphot->ep[ip] > 2.e3*keverg)
    pphot->statp[ip] = REMOVED;
  */
    
  return;
}

Real TableOpacity(MonteCarloBlock *pmcb, Photon *pphot, int ip) {

  // Sets energy, temp, dens to table minimum if outside bounds
  Real le = log10(pphot->ep[ip]);
  le = (le < lmine) ? lmine : le;
  le = (le > lmaxe) ? lmaxe : le;
  int i1 = pphot->i1p[ip];
  int i2 = pphot->i2p[ip];
  int i3 = pphot->i3p[ip];
  Real ld = log10(pmcb->rho(i3,i2,i1));
  ld = (ld < lmind) ? lmind : ld;
  ld = (ld > lmaxd) ? lmaxd : ld;
  Real temp = pmcb->tgas(i3,i2,i1);
  Real lt = log10(temp);
  lt = (lt < lmint) ? lmint : lt;
  lt = (lt > lmaxt) ? lmaxt : lt;
  Real xi = (ld - lmind) / dld;
  Real xj = (lt - lmint) / dlt;
  Real xk = (le - lmine) / dle;
  int i = std::floor(xi);
  int k = std::floor(xk);
  xi -= static_cast<Real>(i);
  xk -= static_cast<Real>(k);

  int j = std::floor(xj);
  while ((j<ntem-2) && (temp_grid(j+1) < temp)){
    j++;
  }
  while ((j>0) && (temp_grid(j) > temp)){
    j--;
  }
  if(j > ntem-2) {
    j = ntem-2;
  }
  xj = (temp-temp_grid(j))/(temp_grid(j+1)-temp_grid(j));
  Real opacl = (1.-xi) * ((1.-xj) * plan_tab(k,j,i) + xj * plan_tab(k,j+1,i))
                 + xi  * ((1.-xj) * plan_tab(k,j,i+1) + xj * plan_tab(k,j+1,i+1));
  Real opach = (1.-xi) * ((1.-xj) * plan_tab(k+1,j,i) + xj * plan_tab(k+1,j+1,i))
                 + xi  * ((1.-xj) * plan_tab(k+1,j,i+1) + xj * plan_tab(k+1,j+1,i+1));

  Real opac = (1.-xk) * opacl + xk * opach;
  //printf("%g %g\n",pphot->ep[ip]/1.6e-9,opac);
  return opac;

}

Real IntegrateEmission(Real temp, Real num, Real nup, Real am, Real ap) {

  int n = 20;
  if (num < 1.)
    num = nup/10;
  Real h = 6.62607015e-27;
  Real dlnu = std::log(nup/num)/static_cast<Real>(n);
  Real dadnu = (ap-am)/(nup-num);
  Real lnu = std::log(num);
  Real sum = Planck(temp,num)*am*dlnu/h/2.;
  // Interior nodes run to n-1: the composite trapezoid rule over n intervals weights
  // nodes 1..n-1 fully and the two endpoints by a half.
  for(int i=1; i<n; ++i) {
    lnu += dlnu;
    Real nu = std::exp(lnu);
    Real alpha = dadnu*(nu-num)+am;
    sum += Planck(temp,nu)*alpha*dlnu/h;
  }
  sum += Planck(temp,nup)*ap*dlnu/h/2.;
  return sum;
}

Real Planck(Real temp, Real nu) {

  Real h = 6.62607015e-27;
  Real c_cgs = 2.99792458e10;
  Real kb = 1.380649e-16;

  return 2.*h/c_cgs/c_cgs*pow(nu,3)/(std::exp(h*nu/kb/temp)-1.);

}

Real TableEmission(MonteCarloBlock *pmcb, int i3, int i2, int i1, int etype) {

  Real ld = log10(pmcb->rho(i3,i2,i1));
  ld = (ld < lmind) ? lmind : ld;
  ld = (ld > lmaxd) ? lmaxd : ld;
  Real temp = pmcb->tgas(i3,i2,i1);
  Real lt = log10(temp);
  lt = (lt < lmint) ? lmint : lt;
  lt = (lt > lmaxt) ? lmaxt : lt;
  Real xi = (ld - lmind) / dld;
  Real xj = (lt - lmint) / dlt;

  int i = std::floor(xi);
  int j = std::floor(xj);

  xi -= static_cast<Real>(i);
  while ((j<ntem-2) && (temp_grid(j+1) < temp)){
    j++;
  }
  while ((j>0) && (temp_grid(j) > temp)){
    j--;
  }
  if(j > ntem-2) {
    j = ntem-2;
  }
  xj = (temp-temp_grid(j))/(temp_grid(j+1)-temp_grid(j));

  Real eta = (1.-xi) * ((1.-xj) * eta_tab(j,i) + xj * eta_tab(j+1,i))
               + xi  * ((1.-xj) * eta_tab(j,i+1) + xj * eta_tab(j+1,i+1));
  return eta;
}


Real SampleEmissivity(MonteCarloBlock *pmcb, Photon *pphot, int ip) {

  Real dev = pmcb->pran->uniform();
  int i1 = pphot->i1p[ip];
  int i2 = pphot->i2p[ip];
  int i3 = pphot->i3p[ip];

  Real *prob = &(eta_cum_tab(i3,i2,i1,0));
  int i = mcbisect(dev,prob,nfre+1);
  Real a = (dev-prob[i])/(prob[i+1]-prob[i]);
  Real a1 = 1.-a;

  return a*fre_grid(i+1)+a1*fre_grid(i);
}

}
