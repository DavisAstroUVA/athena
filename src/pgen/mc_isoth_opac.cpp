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

  // frequency table parameters
  std::string emission_type;
  int nfre, nrho, ntem;
  Real lmine, lmaxe, dle, lmint, lmaxt, dlt, lmind, lmaxd, dld;
  AthenaArray<Real> fre_grid;
  AthenaArray<Real> temp_grid;
  AthenaArray<Real> rho_grid;
  AthenaArray<Real> ross_tab;
  AthenaArray<Real> plan_tab;
  AthenaArray<Real> eta_cum_tab;
  AthenaArray<Real> prob_tab;
  AthenaArray<Real> emisst;
  AthenaArray<Real> opact;

  //functions
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
      }
    }
  }

  // Assume constant velocity provided as fraction of speed of light
  Real c_cgs = 2.99792458e10;
  Real vel1 = pin->GetOrAddReal("problem","vel1",0.)*c_cgs;
  Real vel2 = pin->GetOrAddReal("problem","vel2",0.)*c_cgs;
  Real vel3 = pin->GetOrAddReal("problem","vel3",0.)*c_cgs;

  // Assume constant temperature and ideal gas
  Real gamma = peos->GetGamma();
  // Gas constant of the default hydrogen-helium mixture, so the temperature the
  // Monte Carlo inverts from this pressure is the one written here.
  Real rideal = MonteCarloBlock::GasConstant(pin->GetOrAddReal("problem","heabund",0.09));
  Real tgas = pin->GetReal("problem","temp");
  // Set initial conditions
  if (COORDINATE_SYSTEM == "cartesian") {
    // density varies in the z direction

    for (int k=ks; k<=ke; k++) {
      Real x1 = pcoord->x3v(k);
      if (!constdens) rho = DensityProfile(x1,xlow,xhigh,taumin,taumax,kappaes);
      if ((k == ks) || (k == ke)) printf("min/max dens: %g\n",rho);
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          phydro->u(IDN,k,j,i) = rho;
          phydro->u(IM1,k,j,i) = rho*vel1;
          phydro->u(IM2,k,j,i) = rho*vel2;
          phydro->u(IM3,k,j,i) = rho*vel3;
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
            Real x1 = pcoord->x1v(k);
            if (!constdens) rho = DensityProfile(x1,xlow,xhigh,taumin,taumax,kappaes);
            phydro->u(IDN,k,j,i) = rho;
            phydro->u(IM1,k,j,i) = rho*vel1;
            phydro->u(IM2,k,j,i) = rho*vel2;
            phydro->u(IM3,k,j,i) = rho*vel3;
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
//! \fn void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype)
//! \brief Initializes Photon packets before integration
//========================================================================================

void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype) {

  // Set initial cells and emission weights for all photon samples
  SetEmissionCellWeight(pphot,ips,ipe);

  for (int ip=ips; ip<=ipe; ip++) {

    // Obtain initial position within zone
    GetZonePosition(pphot,pran,pcoord,ip);

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
      // Generate initial angle parameters
      Real phi = 2. * PI * pran->uniform();
      Real cphi = cos(phi);
      Real sphi = sin(phi);
      Real cth = 2. * pran->uniform() - 1.;
      Real sth = sqrt(1. - SQR(cth));
      // Initialize wave vector with isotropic distribution.  k0p carries the photon
      // energy, already sampled into ep above, so it must not be reset here.
      pphot->k1p[ip] = sth*cphi;
      pphot->k2p[ip] = sth*sphi;
      pphot->k3p[ip] = cth;
    }

    if (pphot->IsNanPhoton(ip))
      pphot->PrintPhoton("initialization: ",ip);

    // Convert k unit vector to k^\alpha
    if (pmy_mc->general_pusher_flag) {
      // k0p is the photon energy; only the spatial components are rescaled here.
      pphot->k2p[ip] /= pphot->x1p[ip];
      pphot->k3p[ip] /= (pphot->x1p[ip]*sin(pphot->x2p[ip]));
      pphot->dk0p[ip] = 0.;
      pphot->dk1p[ip] = 0.;
      pphot->dk2p[ip] = 0.;
      pphot->dk3p[ip] = 0.;
    }
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

    //pphot->PrintPhoton("initialization: ",ip);
    //printf("start: %d %g %g %d\n",pphot->i3p[ip],pphot->wp[ip],pphot->ep[ip],pphot->statp[ip]);
  }
  //pphot->nphot++;

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

    int ncells1 = nx1 + 2*(NGHOST);
    int ncells2 = 1, ncells3 = 1;
    if (nx2 > 1) ncells2 = nx2 + 2*(NGHOST);
    if (nx3 > 1) ncells3 = nx3 + 2*(NGHOST);

    // Compute opacity table corresponding to each cell and frequency
    opact.NewAthenaArray(ncells3,ncells2,ncells1,nfre+1);
    for(int k=ks; k<=ke; ++k) {
      for(int j=js; j<=je; ++j) {
        for(int i=is; i<=ie; ++i) {
          Real ld = log10(rho(k,j,i));
          ld = (ld < lmind) ? lmind : ld;
          ld = (ld > lmaxd) ? lmaxd : ld;
          Real temp = tgas(k,j,i);
          Real lt = log10(temp);
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
          for(int l=0; l<nfre; ++l) {
            opact(k,j,i,l) = (1.-xi)*( (1.-xj)*plan_tab(l,jj,ii)+xj*plan_tab(l,jj+1,ii) )
              + xi*( (1.-xj)* plan_tab(l,jj,ii+1)+xj*plan_tab(l,jj+1,ii+1) );
          }
        }
      }
    }

    // Compute emissivity table for each cell and frequncy
    AthenaArray<Real> eta_nu_tab;
    eta_nu_tab.NewAthenaArray(ncells3,ncells2,ncells1,nfre);
    Real h = 6.62607015e-27;
    for(int l=0; l<nfre; ++l) {
      Real nup = fre_grid(l+1)/h;
      Real num = fre_grid(l)/h;
      for(int k=ks; k<=ke; ++k) {
        for(int j=js; j<=je; ++j) {
          for(int i=is; i<=ie; ++i) {
            Real temp = tgas(k,j,i);
            Real opp = opact(k,j,i,l);
            Real opm;
            if (l == 0)
              opm = 0.;
            else
              opm = opact(k,j,i,l-1);
            eta_nu_tab(k,j,i,l) = IntegrateEmission(temp,num,nup,opm,opp);
          }
        }
      }
    }

    // Compute integratred emission table for each cell
    emisst.NewAthenaArray(ncells3,ncells2,ncells1);
    for(int k=ks; k<=ke; ++k) {
      for(int j=js; j<=je; ++j) {
        for(int i=is; i<=ie; ++i) {
          emisst(k,j,i) = 0.;
          for(int l=0; l<nfre; ++l) {
            emisst(k,j,i) += 4.*PI*eta_nu_tab(k,j,i,l);
          }
        }
      }
    }

    // Compute cumulative emission table for each cell
    eta_cum_tab.NewAthenaArray(ncells3,ncells2,ncells1,nfre+1);
    Real keverg = 1.602176634e-9;
    for(int k=ks; k<=ke; ++k) {
      for(int j=js; j<=je; ++j) {
        for(int i=is; i<=ie; ++i) {
          eta_cum_tab(k,j,i,0) = 0.;
          for(int l=0; l<nfre; ++l) {
            eta_cum_tab(k,j,i,l+1) = eta_cum_tab(k,j,i,l)+eta_nu_tab(k,j,i,l);
            //if ((Globals::my_rank == 0) && (k==ks) && (j == js) && (i == is)) {
            //}
          }
          for(int l=0; l<nfre+1; ++l) {
            eta_cum_tab(k,j,i,l) /= eta_cum_tab(k,j,i,nfre);
            //if ((Globals::my_rank == 0) && (k==ks) && (j == js) && (i == is)) {
            //  printf("%d %g %g\n",l,fre_grid(l)/keverg,eta_cum_tab(k,j,i,l));
            //}
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

  nuser_var = 1;
  AllocateUserMoments(2);
  EnrollUserMoment(0, JMeanOpacity, "kapJ");
  EnrollUserMoment(1, AverageEnergy, "eave");

  emission_type = pin->GetOrAddString("montecarlo","emission","none");

  if (emission_type == "freefree")
    return;

  // Read in opacity table
  FILE  *opac_file;
  std::string opacity_filename = pin->GetString("problem", "opacity_filename");
  if ( (opac_file=fopen(opacity_filename.c_str(),"r"))==NULL) {
  //if ( (opac_file=fopen("out_opacity_table_nfreq32.txt","r"))==NULL) {
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

  //if (Globals::my_rank == 0)
  //  printf("Frequency grid (keV):\n");
  for(int i=1; i<=nfre; ++i){
    fscanf(opac_file,"%lf",&(fre_grid(i)));
    //if (Globals::my_rank == 0)
    //  printf("%d %g\n",i,fre_grid(i));
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
  //if (Globals::my_rank == 0)
  //  printf("Temperature grid:\n");
  for(int i=0; i<ntem; ++i) {
    temp_grid(i) *= keverg/kb;
    //if (Globals::my_rank == 0)
    //  printf("%d %g\n",i,temp_grid(i));
  }
  lmint = std::log10(temp_grid(0));
  lmaxt = std::log10(temp_grid(ntem-1));
  dlt = (lmaxt-lmint)/static_cast<Real>(ntem-1);

  // density grid (g/cm^3)
  //if (Globals::my_rank == 0)
  //  printf("Density Grid:\n");
  for(int i=0; i<nrho; ++i) {
    fscanf(opac_file,"%lf",&(rho_grid(i)));
    //if (Globals::my_rank == 0)
    //  printf("%d %g\n",i,rho_grid(i));
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

Real TableOpacity(MonteCarloBlock *pmcb, Photon *pphot, int ip) {

  // Sets energy, temp, dens to table minimum if outside bounds
  Real le = log10(pphot->ep[ip]);
  le = (le < lmine) ? lmine : le;
  le = (le > lmaxe) ? lmaxe : le;
  int i1 = pphot->i1p[ip];
  int i2 = pphot->i2p[ip];
  int i3 = pphot->i3p[ip];
  Real xk = (le - lmine) / dle;
  int k = std::floor(xk);
  xk -= static_cast<Real>(k);
  Real opac = (1.-xk) * opact(i3,i2,i1,k) + xk * opact(i3,i2,i1,k+1);
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

  return emisst(i3,i2,i1);
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
  //if (std::isinf(a)) {
  ///printf("%d %d %d\n",i3,i2,i1);

  //for (int j=0; j< nfre+1; ++j)
  //  printf("%d %e\n",j,prob[j]);
  //printf("%d %g %g %g %g\n",i,dev,fre_grid(i),a,a1);
  //}
  //printf("%d %g %g %g %g\n",i,dev,prob[i],prob[i+1],(a*fre_grid(i+1)+a1*fre_grid(i))/1.6e-9);
  return a*fre_grid(i+1)+a1*fre_grid(i);
}

}
