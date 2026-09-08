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
  Real rideal = 8.314e7;
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

    if (emission_type == "freefree") {
      // Obtain intitial energy, polarization, direction and weight
      // Utilize free-free emission function in emission.cpp
      if(tnorm) {
        Real logtg = log(tgas(pphot->i3p[ip],pphot->i2p[ip],pphot->i1p[ip]));
        PhotonEmitFreeFree(this,pphot,logemin+logtg,logemax+logtg,ip);
      } else{
        PhotonEmitFreeFree(this,pphot,logemin,logemax,ip);
      }
    }/* else {
      pphot->ep[ip] = SampleEmissivity(this,pphot,ip);
      //printf("%g\n",pphot->ep[ip]/1.6e-9);
      // Generate initial angle parameters
      Real phi = 2. * PI * pran->uniform();
      Real cphi = cos(phi);
      Real sphi = sin(phi);
      Real cth = 2. * pran->uniform() - 1.;
      Real sth = sqrt(1. - SQR(cth));
      // Initialize wave vector with isotropic distribution
      pphot->k0p[ip] = 1.;
      pphot->k1p[ip] = sth*cphi;
      pphot->k2p[ip] = sth*sphi;
      pphot->k3p[ip] = cth;
    }*/

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

  nphremain = pin->GetInteger("montecarlo","nphot");
  return;
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

  nuser_var = NUSER_FRAMES;
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
