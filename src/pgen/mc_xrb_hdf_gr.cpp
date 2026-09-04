//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file from_array.cpp//! \brief Problem generator for initializing with preexisting array from HDF5 input

// C headers

// C++ headers
#include <algorithm>  // max()
#include <string>     // c_str(), string

// Athena++ headers
#include "../athena.hpp"              // Real
#include "../athena_arrays.hpp"       // AthenaArray
#include "../field/field.hpp"         // Field
#include "../globals.hpp"             // Globals
#include "../hydro/hydro.hpp"         // Hydro
#include "../eos/eos.hpp"                  // EquationOfState
#include "../inputs/hdf5_reader.hpp"  // HDF5ReadRealArray()
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"     // ParameterInput
#include "../monte_carlo/mcgrid.hpp"
#include "../monte_carlo/mcsnapshot.hpp"
#include "../monte_carlo/montecarlo.hpp"
#include "../monte_carlo/photon.hpp"
#include "../monte_carlo/mcutils.hpp"

namespace {
  // Global variables
  bool tnorm;
  Real logemin, logemax;
  Real abh, r_hor;
  Real dcut;
  std::string emission_type;
  // frequency table parameters
  int nfre, nrho, ntem;
  Real lmine, lmaxe, dle, lmint, lmaxt, dlt, lmind, lmaxd, dld;
  AthenaArray<Real> fre_grid;
  AthenaArray<Real> temp_grid;
  AthenaArray<Real> rho_grid;
  AthenaArray<Real> ross_tab;
  AthenaArray<Real> ross_gray_tab;
  AthenaArray<Real> plan_tab;
  AthenaArray<Real> emis_cum;
  AthenaArray<Real> emis_tot;
  AthenaArray<Real> opact;

  //functions
  void InsideHorizon(MonteCarloBlock *pmcb, Photon *pphot, PhotonPusher *ppusher,int ip);
  Real TableOpacity(MonteCarloBlock *pmcb, Photon *pphot, int ip);
  Real IntegrateEmission(Real temp, Real num, Real nup, Real am, Real ap);
  Real Planck(Real temp, Real nu);
  Real TableEmission(MonteCarloBlock *pmcb, int k, int j, int i, int etype);
  Real SampleEmissivity(MonteCarloBlock *pmcb, Photon *pphot, int ip);
  Real FreeFreeOpacity(Real tgas, Real rho, Real energy);
  void GetNel(MonteCarloBlock *pmcb);
  void GetNelFloor(MonteCarloBlock *pmcb);
  Real UserScatteringOpacity(MonteCarloBlock *pmcb, Photon *pphot, int ip);
  void CartesianKerrSchild(Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3);
}

std::vector<float> x1coord;
std::vector<float> x2coord;
std::vector<float> x3coord;

int getindex(std::vector<float> vec, float val){
  std::vector<float>::iterator it = std::find(vec.begin(), vec.end(), val);
  int index = std::distance(vec.begin(), it);
  return index;
}

void MonteCarlo::InitUserMonteCarloData(ParameterInput *pin) {

  nuser_var = 3;

  abh = pin->GetReal("coord","a");
  // assumes mbh = 1 in code units
  r_hor = 1.0 + sqrt(1.0 - SQR(abh));
  EnrollUserWorkInMove(InsideHorizon);
  
  emission_type = pin->GetOrAddString("montecarlo","emission","none");
  if (emission_type == "freefree") {
    EnrollUserGetNumberDensity(GetNelFloor);
    return;
  }
  // Read in opacity table
  FILE  *opac_file;
  std::string opacity_filename = pin->GetString("problem", "opacity_filename");
  if ( (opac_file=fopen(opacity_filename.c_str(),"r"))==NULL) {
    std::stringstream msg;
    msg << "FATAL ERROR: Could not open out_opacity_table_nfreq16.txt." << std::endl;
    ATHENA_ERROR(msg);
  }

  fscanf(opac_file,"%d",&(nfre));
  fscanf(opac_file,"%d",&(ntem));
  fscanf(opac_file,"%d",&(nrho));

  // Create arrays for opacity
  fre_grid.NewAthenaArray(nfre);
  temp_grid.NewAthenaArray(ntem);
  rho_grid.NewAthenaArray(nrho);
  ross_gray_tab.NewAthenaArray(ntem,nrho);
  ross_tab.NewAthenaArray(nfre,ntem,nrho);
  plan_tab.NewAthenaArray(nfre,ntem,nrho);

  for(int i=0; i<nfre; ++i){
    fscanf(opac_file,"%lf",&(fre_grid(i)));
  }
  //fre_grid(0) = fre_grid(1)*fre_grid(1)/fre_grid(2);
  // convert to erg
  Real keverg = 1.602176634e-9;
  for(int i=0; i<nfre; ++i)
    fre_grid(i) *= keverg;
  lmine = std::log10(fre_grid(0));
  lmaxe = std::log10(fre_grid(nfre-1));
  dle = (lmaxe-lmine)/static_cast<Real>(nfre-1);
  // temperature grid (keV)
  for(int i=0; i<ntem; ++i){
    fscanf(opac_file,"%lf",&(temp_grid(i)));
  }
  // convert to kelvin
  Real kb_cgs = 1.380649e-16;
  for(int i=0; i<ntem; ++i)
    temp_grid(i) *= keverg/kb_cgs;
  lmint = std::log10(temp_grid(0));
  lmaxt = std::log10(temp_grid(ntem-1));
  dlt = (lmaxt-lmint)/static_cast<Real>(ntem-1);
  // note temperature grid not uniform in log

  // density grid (g/cm^3)
  for(int i=0; i<nrho; ++i) {
    fscanf(opac_file,"%lf",&(rho_grid(i)));
  }
  lmind = std::log10(rho_grid(0));
  lmaxd = std::log10(rho_grid(nrho-1));
  dld = (lmaxd-lmind)/static_cast<Real>(nrho-1);

  if (Globals::my_rank == 0) {
    printf("Max/min/num energies (keV) in table: %g %g %d\n",
           fre_grid(0)/keverg,fre_grid(nfre-1)/keverg,nfre);
    printf("Max/min/num temperatures in table: %g %g %d\n",
           temp_grid(0),temp_grid(ntem-1),ntem);
    printf("Max/min/num densities in table: %g %g %d\n",
           rho_grid(0),rho_grid(nrho-1),nrho);
  }
  // frequency integrated rosseland mean
  // Read in but not used
  Real buf;
  for(int j=0; j<ntem; ++j) {
    for(int i=0; i<nrho; ++i) {
      fscanf(opac_file,"%lf",&(ross_gray_tab(j,i)));
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

  // planck mean for each frequency group
  for(int k=0; k<nfre; ++k) {
    for(int j=0; j<ntem; ++j) {
      for(int i=0; i<nrho; ++i) {
        fscanf(opac_file,"%lf",&(plan_tab(k,j,i)));
        plan_tab(k,j,i) *= rho_grid(i);
      }
    }
  }

  bool user_ff = pin->GetOrAddBoolean("problem", "userff", false);
  if (user_ff) {
    // Replaces plan_tab with free-free values (for testing purposes)
    Real dummy;
    for(int k=0; k<nfre; ++k) {
      for(int j=0; j<ntem; ++j) {
        for(int i=0; i<nrho; ++i) {
          plan_tab(k,j,i) = FreeFreeOpacity(temp_grid(j),rho_grid(i),fre_grid(k));
        }
      }
    }
  }

  // search for case where planck mean ise used for each frequency group and replace
  for(int j=0; j<ntem; ++j) {
    for(int i=0; i<nrho; ++i) {
      Real min = 1.e40;
      Real max = 1.e-40;
      for(int k=0; k<nfre; ++k) {
        min = (min > plan_tab(k,j,i)) ? plan_tab(k,j,i) : min;
        max = (max < plan_tab(k,j,i)) ? plan_tab(k,j,i) : max;
      }
      // Identify table values using gray opacity and replace with free-free
      if (max/min < 1.1) {
        for(int k=0; k<nfre; ++k) {
          plan_tab(k,j,i) = FreeFreeOpacity(temp_grid(j),rho_grid(i),fre_grid(k));
        }
      }
    }
  }
  fclose(opac_file);


  EnrollUserEmissionFunction(TableEmission);
  EnrollUserOpacityFunction(TableOpacity,true);

  EnrollUserGetNumberDensity(GetNel);

  int nx1 = pin->GetInteger("meshblock", "nx1");
  int nx2 = pin->GetInteger("meshblock", "nx2");
  int nx3 = pin->GetInteger("meshblock", "nx3");
  int ncells1 = nx1 + 2*(NGHOST);
  int ncells2 = 1, ncells3 = 1;
  if (nx2 > 1) ncells2 = nx2 + 2*(NGHOST);
  if (nx3 > 1) ncells3 = nx3 + 2*(NGHOST);
  int nblocal =  pmy_mesh->nblocal;
  //printf("blocks: %d %d\n",Globals::my_rank,nblocal);
  opact.NewAthenaArray(nblocal,ncells3,ncells2,ncells1,nfre);
  emis_tot.NewAthenaArray(nblocal,ncells3,ncells2,ncells1);
  emis_cum.NewAthenaArray(nblocal,ncells3,ncells2,ncells1,nfre);

}

void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin) {

  dcut = pin->GetOrAddReal("problem", "dcut",1.e-20);
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
    int lid = pmy_block->lid;
    // Compute opacity table corresponding to each cell and frequency
    //opact.NewAthenaArray(ncells3,ncells2,ncells1,nfre);
    for(int k=ks; k<=ke; ++k) {
      for(int j=js; j<=je; ++j) {
        for(int i=is; i<=ie; ++i) {
          bool on_grid = true;
          Real ld = log10(rho(k,j,i));
          //ld = (ld < lmind) ? lmind : ld;
          //ld = (ld > lmaxd) ? lmaxd : ld;
          Real temp = tgas(k,j,i);
          Real lt = log10(temp);
          //lt = (lt < lmint) ? lmint : lt;
          //lt = (lt > lmaxt) ? lmaxt : lt;
          Real xi = (ld - lmind) / dld;
          int ii = std::floor(xi);
          if (ii < 0) {
            ii = 0;
            printf("Warning: %g is less than the lowest density in grid: %g.",
                   rho(k,j,i),rho_grid(0));
            on_grid = false;
          } else if (ii > nrho-2) {
            ii = nrho-2;
            printf("Warning: %g exceeds the largest density in grid: %g.",
                   rho(k,j,i),rho_grid(nrho-1));
            on_grid = false;
          }
          xi -= static_cast<Real>(ii);
          Real xj = (lt - lmint) / dlt;
          int jj = std::floor(xj);
          if (jj < 0)
            jj = 0;
          if (jj > ntem-2)
            jj = ntem-2;
          while ((jj<ntem-2) && (temp_grid(jj+1) < temp)){
            jj++;
          }
          while ((jj>0) && (temp_grid(jj) > temp)){
            jj--;
          }
          if(jj > ntem-2) {
            jj = ntem-2;
            printf("Warning: %g exceeds largest temp in grid: %g.",
                   temp,temp_grid(ntem-1));
            on_grid = false;
          }
          if(jj < 0) {
            jj = 00;
            printf("Warning: %g is less than smallest temp in grid: %g.",
                   temp,temp_grid(0));
            on_grid = false;
          }
          xj = (temp-temp_grid(jj))/(temp_grid(jj+1)-temp_grid(jj));
          if (xj > 1.)
            xj = 1.;
          if (on_grid) {
            for(int l=0; l<nfre; ++l) {
              opact(lid,k,j,i,l) = (1.-xi)*( (1.-xj)*plan_tab(l,jj,ii)
                +xj*plan_tab(l,jj+1,ii) ) + xi*( (1.-xj)* plan_tab(l,jj,ii+1)
                +xj*plan_tab(l,jj+1,ii+1) );
            }
          } else {
              printf(" Using free-free opacity\n");
              for(int l=0; l<nfre; ++l) {
                opact(lid,k,j,i,l) = FreeFreeOpacity(temp,rho(k,j,i),fre_grid(l));
              }
          }
        }
      }
    }

    // Compute emissivity table for each cell and frequncy
    AthenaArray<Real> eta_nu_tab;
    eta_nu_tab.NewAthenaArray(ncells3,ncells2,ncells1,nfre);
    Real h_cgs = 6.62607015e-27;
    for(int l=0; l<nfre; ++l) {
      Real nu = fre_grid(l)/h_cgs;
      for(int k=ks; k<=ke; ++k) {
        for(int j=js; j<=je; ++j) {
          for(int i=is; i<=ie; ++i) {
            Real temp = tgas(k,j,i);
            eta_nu_tab(k,j,i,l) = Planck(temp,nu) * opact(lid,k,j,i,l);
          }
        }
      }
    }

    // Compute integratred emission tables (total and cumulative) for each cell
    for(int k=ks; k<=ke; ++k) {
      for(int j=js; j<=je; ++j) {
        for(int i=is; i<=ie; ++i) {
          emis_cum(k,j,i,0) = 0.;
          for(int l=1; l<nfre; ++l) {
            Real nup = fre_grid(l)/h_cgs;
            Real num = fre_grid(l-1)/h_cgs;
            Real dlnu = std::log(nup/num);
            Real eta_ave = 0.5*(eta_nu_tab(k,j,i,l)+eta_nu_tab(k,j,i,l-1));
            emis_cum(lid,k,j,i,l) = emis_cum(lid,k,j,i,l-1) + 4.*PI/h_cgs*eta_ave*dlnu;
          }
          emis_tot(lid,k,j,i) = emis_cum(lid,k,j,i,nfre-1);
          for(int l=1; l<nfre; ++l) {
            emis_cum(lid,k,j,i,l) /= emis_tot(lid,k,j,i);
          }
        }
      }
    }
    eta_nu_tab.DeleteAthenaArray();
  }

}


void Mesh::InitUserMeshData(ParameterInput *pin) {

  EnrollUserMetric(CartesianKerrSchild);

  bool resampled = pin->GetOrAddBoolean("problem","resampled",false);
  bool collective = pin->GetOrAddBoolean("problem","collective",false);

  if (resampled) {
    // Read in hdf5 file to initialize pgen
    std::string input_filename = pin->GetString("problem", "input_filename");
    int mesh_nx1 = pin->GetInteger("mesh", "nx1");
    Real mesh_x1min = pin->GetReal("mesh", "x1min");
    Real mesh_x1max = pin->GetReal("mesh", "x1max");
    int mesh_nx2 = pin->GetInteger("mesh", "nx2");
    Real mesh_x2min = pin->GetReal("mesh", "x2min");
    Real mesh_x2max = pin->GetReal("mesh", "x2max");
    int mesh_nx3 = pin->GetInteger("mesh", "nx3");
    Real mesh_x3min = pin->GetReal("mesh", "x3min");
    Real mesh_x3max = pin->GetReal("mesh", "x3max");
    Real x1ratio = pin->GetReal("mesh", "x1rat");

    //load data file
    int start_file[3] = {0,0,0};
    int count_file[3] = {mesh_nx3, mesh_nx2, mesh_nx1};
    int start_mem[3] = {0,0,0};
    int count_mem[3] = {mesh_nx3, mesh_nx2, mesh_nx1};

    //load data to user mesh data for later use
    AllocateRealUserMeshDataField(5);
    ruser_mesh_data[0].NewAthenaArray(mesh_nx3, mesh_nx2, mesh_nx1);
    ruser_mesh_data[1].NewAthenaArray(mesh_nx3, mesh_nx2, mesh_nx1);
    ruser_mesh_data[2].NewAthenaArray(mesh_nx3, mesh_nx2, mesh_nx1);
    ruser_mesh_data[3].NewAthenaArray(mesh_nx3, mesh_nx2, mesh_nx1);
    ruser_mesh_data[4].NewAthenaArray(mesh_nx3, mesh_nx2, mesh_nx1);
    HDF5ReadRealArray(input_filename.c_str(), "prim/rho", 3, start_file, count_file,
                      3, start_mem, count_mem, ruser_mesh_data[0], collective);
    HDF5ReadRealArray(input_filename.c_str(), "prim/vel1", 3, start_file, count_file,
                      3, start_mem, count_mem, ruser_mesh_data[1], collective);
    HDF5ReadRealArray(input_filename.c_str(), "prim/vel2", 3, start_file, count_file,
                      3, start_mem, count_mem, ruser_mesh_data[2], collective);
    HDF5ReadRealArray(input_filename.c_str(), "prim/vel3", 3, start_file, count_file,
                      3, start_mem, count_mem, ruser_mesh_data[3], collective);
    HDF5ReadRealArray(input_filename.c_str(), "prim/press", 3, start_file, count_file,
                      3, start_mem, count_mem, ruser_mesh_data[4], collective);

    //Real dx1 = (mesh_x1max - mesh_x1min)/mesh_nx1;
    Real dx2 = (mesh_x2max - mesh_x2min)/mesh_nx2;
    Real dx3 = (mesh_x3max - mesh_x3min)/mesh_nx3;

    //prepare three vectors for index finding of x1 x2 x3 coordinates
    //the vector are equivalent to pcoord->x1v, x2v, x3v
    for(int i=0; i<mesh_nx1; i++){
      Real x1coord_now = (pow(x1ratio, i)-1.0)/(pow(x1ratio, mesh_nx1)-1.0) *
        (mesh_x1max - mesh_x1min) + mesh_x1min;
      x1coord.push_back(x1coord_now);
    }
    for(int j=0; j<mesh_nx2; j++){
      x2coord.push_back(mesh_x2min+j*dx2);
    }
    for(int k=0; k<mesh_nx3; k++){
      x3coord.push_back(mesh_x3min+k*dx3);
    }
  } //end if (resampled)
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief monte carlo test problem generator
//! Inputs:
//! - pin: parameters
//! Outputs: (none)
//! Notes:
//! - reads in array using a slightly modified verions of from_array.cpp
//!   - NHYDRO
//!   - total number of MeshBlocks
//!   - MeshBlock/nx3
//!   - MeshBlock/nx2
//!   - MeshBlock/nx1

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  // All of the hyperslab bookkeeping now lives in MCReadSnapshotBlock, which locates the
  // variables by name from the file itself.  That covers both an Athena++ dump (rho,
  // press, vel1) and one converted from an AthenaK run (dens, eint, velx), so the
  // <problem>/dataset_cons, index_* and athenak_input keys are no longer needed: the
  // internal-energy-to-pressure conversion is driven by what the file actually holds.
  // Mesh::nblist is private to everything but MeshBlock, so the collective-read padding
  // count is gathered here and handed over.
  int max_blocks_per_rank = 0;
  for (int r = 0; r < Globals::nranks; ++r)
    max_blocks_per_rank = std::max(max_blocks_per_rank, pmy_mesh->nblist[r]);
  MCReadSnapshotBlock(this, pin, max_blocks_per_rank);

  // Set index bounds
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

  // Initialize conserved
  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, il, iu, jl, ju,
                             kl, ku);
}

//========================================================================================
//! \fn void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin)
//! \brief Analogous to problem generator but used in support of InitializePhoton
//========================================================================================


//========================================================================================
//! \fn void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe)
//! \brief Initializes Photon packets before integration
//========================================================================================

void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype) {

  // Set initial cells and emission weights for all photon samples
  SetEmissionCellWeight(pphot,ips,ipe);

  for (int ip=ips; ip<=ipe; ip++) {
    if (pphot->IsNanPhoton(ip)) {
      pphot->PrintPhoton("init",ip);
    }
    // Obtain initial position within zone
    GetZonePosition(pphot,pran,pcoord,ip);

    // Set maximum integration time
    pphot->dtp[ip] = pphot->pmy_mcb->pmy_mc->tmax;

    //xs: store gas temperature
    pphot->user[0][ip] = tgas(pphot->i3p[ip],pphot->i2p[ip],pphot->i1p[ip]);

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
      if (pphot->IsNanPhoton(ip))
        pphot->PrintPhoton("initialization: ",ip);
      //printf("en: %g\n",pphot->ep[ip]);
      //pphot->PrintPhoton("initialization: ",ip)

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

    if (IsPolarized(pmy_mc->polarized)) {
      // Initialize Stokes vector
      pphot->sip[ip] = 1.0;
      pphot->sup[ip] = 0.0;
      pphot->sqp[ip] = 0.0;
      pphot->svp[ip] = 0.0;
    }

    //xs: store photon energy
    pphot->user[1][ip] = pphot->ep[ip];

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

    //pphot->PrintPhoton("init",ip);
    //pphot->statp[ip] = ESCAPED;
  } // loop over ip

}

//========================================================================================
//! \fn void MonteCarloBlock::FinalizePhoton(Photon *pphot, int ip)
//! \brief Complete work at end of photon packets before integration
//========================================================================================

void MonteCarloBlock::FinalizePhoton(Photon *pphot, int ip) {

  //xs: store scatter number
  pphot->user[2][ip] = pphot->nscp[ip];

}

namespace {


void InsideHorizon(MonteCarloBlock *pmcb, Photon *pphot, PhotonPusher *ppusher, int ip) {

  Real x1 = pphot->x1p[ip];
  Real x2 = pphot->x2p[ip];
  Real x3 = pphot->x3p[ip];

  Real rad = std::sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  Real r = sqrt((SQR(rad)-SQR(abh)+sqrt(SQR(SQR(rad)-SQR(abh))+4.0*SQR(abh)*SQR(x3)))/2.);

  if (r < r_hor) {
    pphot->statp[ip] = ABSORBED;
    //printf("Photon absorbed inside horizon at r=%g\n",r);
  }
  Real keverg = 1.602176634e-9;
  if (pphot->ep[ip] > 2.e3*keverg)
    pphot->statp[ip] = REMOVED;
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
  Real xk = (le - lmine) / dle;
  int k = std::floor(xk);
  xk -= static_cast<Real>(k);
  if (k < 0) {
    //printf("o: %d %g\n",k,le,lmine,lmaxe);
    k = 0;
    xk = 0.;
  } else if (k >= nfre-1) {
    //printf("o: %d %g\n",k,le,lmine,lmaxe);
    k = nfre-2;
    xk = 1.;
  }
  Real lid = pmcb->pmy_block->lid;
  return (1.-xk) * opact(lid,i3,i2,i1,k) + xk * opact(lid,i3,i2,i1,k+1);

}

Real IntegrateEmission(Real temp, Real num, Real nup, Real am, Real ap) {

  int n = 20;
  Real h_cgs = 6.62607015e-27;
  Real dlnu = std::log(nup/num)/static_cast<Real>(n);
  Real dadnu = (ap-am)/(nup-num);
  Real lnu = std::log(num);
  Real sum = Planck(temp,num)*am*dlnu/h_cgs/2.;
  for(int i=1; i<n-1; ++i) {
    lnu += dlnu;
    Real nu = std::exp(lnu);
    Real alpha = dadnu*(nu-num)+am;
    sum += Planck(temp,nu)*alpha*dlnu/h_cgs;
  }
  sum += Planck(temp,nup)*ap*dlnu/h_cgs/2.;
  //if (sum < 0)
  //  printf("sum: %g %g %g %g\n",num,nup,am,ap);
  return sum;
}

Real Planck(Real temp, Real nu) {

  Real h_cgs = 6.62607015e-27;
  Real c_cgs = 2.99792458e10;
  Real kb_cgs = 1.380649e-16;

  return 2.*h_cgs/c_cgs/c_cgs*pow(nu,3)/(std::exp(h_cgs*nu/kb_cgs/temp)-1.);

}

Real TableEmission(MonteCarloBlock *pmcb, int i3, int i2, int i1, int etype) {

  //Real comp =GetEmissionFreeFree(pmcb,i3,i2,i1);
  int lid = pmcb->pmy_block->lid;
  //Real ratio = emis_tot(lid,i3,i2,i1)/comp;
  //if ((ratio > 5.9) || (ratio < 5.7)) {
  //  printf("%d %d %d %d %g %g %g\n",Globals::my_rank,
  //         i3,i2,i1,ratio,pmcb->tgas(i3,i2,i1),pmcb->rho(i3,i2,i1));
  //}
  //return GetEmissionFreeFree(pmcb,i3,i2,i1);
  //int lid = pmcb->pmy_block->lid;
  return emis_tot(lid,i3,i2,i1);
}


Real SampleEmissivity(MonteCarloBlock *pmcb, Photon *pphot, int ip) {

  Real dev = pmcb->pran->uniform();
  int i1 = pphot->i1p[ip];
  int i2 = pphot->i2p[ip];
  int i3 = pphot->i3p[ip];
  int lid = pmcb->pmy_block->lid;

  Real *prob = &(emis_cum(lid,i3,i2,i1,0));
  int i = mcbisect(dev,prob,nfre);
  Real a = (dev-prob[i])/(prob[i+1]-prob[i]);
  Real a1 = 1.-a;
  //printf("%d %g %g\n",i,a,a1);
  if ((a < 0.) || (a > 1.)) {
    printf("%d %d %d\n",i3,i2,i1);
    for (int j=0; j< nfre+1; ++j)
      printf("%d %e\n",j,1-prob[j]);
    printf("%d %g %g %g %g\n",i,dev,fre_grid(i),a,a1);
  }
  //Real nu = std::exp(a*std::log(fre_grid(i+1))+a1*std::log(fre_grid(i)));
  Real nu = a*fre_grid(i+1)+a1*fre_grid(i);
  return nu;
}

Real FreeFreeOpacity(Real tgas, Real rho, Real energy) {
  Real ffnrm = 3.692146e8;
  Real heabund = 0.09; //hardcode for now (should be parameter)
  Real mp = 1.67262192369e-24;
  Real h = 6.62607015e-27;
  Real kb = 1.380649e-16;
  Real nh = rho / (mp*(1.+4.*heabund));
  Real nhe = nh*heabund;
  Real ne = nh + 2.*nhe;
  Real nu = energy / h;
  Real ehnu = exp(-h*nu / (kb * tgas) );
  Real aff = ffnrm/sqrt(tgas)/pow(nu,3);
  Real opac = ne * (nh + 4. * nhe) * aff * (1. - ehnu);

  return opac;
}

void GetNelFloor(MonteCarloBlock *pmcb) {

  Real heabund = 0.09; //hardcode for now (should be parameter)
  Real mp = 1.67262192369e-24;
  Real dmin = dcut*pmcb->rho_cgs; // dfloor
  
  for (int k=pmcb->ks; k<=pmcb->ke; ++k) {
    for (int j=pmcb->js; j<=pmcb->je; ++j) {
      for (int i=pmcb->is; i<=pmcb->ie; ++i) {
        Real rho = pmcb->rho(k,j,i);
	
	if (rho < dmin) {
	  //printf("rho: %d %d %d %d %g\n",pmcb->pmy_block->gid,k,j,i,rho);
	  rho = 1.e-30;
	}
        Real nh = rho / (mp*(1.+4.*heabund));
        Real nhe = nh*heabund;
	pmcb->species(1,k,j,i) = nh + 4. * nhe;
        pmcb->species(0,k,j,i) = nh + 2. * nhe;
      }
    }
  }
}
  
void GetNel(MonteCarloBlock *pmcb) {

  Real heabund = 0.09; //hardcode for now (should be parameter)
  Real mp = 1.67262192369e-24;

  for (int k=pmcb->ks; k<=pmcb->ke; ++k) {
    for (int j=pmcb->js; j<=pmcb->je; ++j) {
      for (int i=pmcb->is; i<=pmcb->ie; ++i) {
        Real rho = pmcb->rho(k,j,i);
        Real nh = rho / (mp*(1.+4.*heabund));
        Real nhe = nh*heabund;
        //nion(k,j,i) = nh + 4. * nhe;
        pmcb->species(0,k,j,i) = nh + 2. * nhe;

        Real tgas = pmcb->tgas(k,j,i);
        Real ld = log10(rho);
        Real lt = log10(tgas);
        Real xi = (ld - lmind) / dld;
        int ii = std::floor(xi);
        if (ii < 0) {
          ii = 0;
        } else if (ii > nrho-2) {
          ii = nrho-2;
        }
        xi -= static_cast<Real>(ii);
        Real xj = (lt - lmint) / dlt;
        int jj = std::floor(xj);
        if (jj < 0)
          jj = 0;
        if (jj > ntem-2)
          jj = ntem-2;
        while ((jj<ntem-2) && (temp_grid(jj+1) < tgas)){
          jj++;
        }
        while ((jj>0) && (temp_grid(jj) > tgas)){
          jj--;
        }
        if(jj > ntem-2) {
          jj = ntem-2; // above T grid, assume ionized
          continue;
        }
        if(jj < 0) {
          jj = 00;
          pmcb->species(0,k,j,i) = 0.; //below T grid, assume neutral
          continue;
        }
        xj = (tgas-temp_grid(jj))/(temp_grid(jj+1)-temp_grid(jj));
        Real ross = (1.-xi)*( (1.-xj)*ross_gray_tab(jj,ii)
                +xj*ross_gray_tab(jj+1,ii) ) + xi*( (1.-xj)*ross_gray_tab(jj,ii+1)
                +xj*ross_gray_tab(jj+1,ii+1) );
        if ((tgas < 1.e5) && (ross < 0.34)) {
          pmcb->species(0,k,j,i) = 0.; // assume neutral
          //printf("%d %d %g %g %g %g\n",jj,ii,tgas,rho,ross,ross_gray_tab(jj,ii));
        }
      } // loop over i
    }
  }
}

Real UserGetDensity(MonteCarloBlock *pmcb) {

  Real l_cgs = pmcb->l_cgs;
  Real rho_cgs = pmcb->rho_cgs;
  Real kappa_s = 0.39/(rho_cgs*l_cgs);
  Real dfloor_op = 1.e-14;
  Real tau_trunc = 1.e-4;
  Real dtrunc_max = 1.e-5;
  Real sigmoid_res = 1.e-2;
  Real dfloor = 1.e-8;
  for (int k=pmcb->ks; k<=pmcb->ke; ++k) {
    for (int j=pmcb->js; j<=pmcb->je; ++j) {
      for (int i=pmcb->is; i<=pmcb->ie; ++i) {
	Real wdn = pmcb->pmy_block->phydro->u(IDN,k,j,i);

	Real sigma_cold = 0.;
	// Match Lizhong's scattering reduction
	Real wdn_opacity = fmax(wdn-dfloor, dfloor_op);
	
	Real dx1 = pmcb->pmy_block->pcoord->dx1f(i);
	Real dx2 = pmcb->pmy_block->pcoord->dx2f(k);
	Real dx3 = pmcb->pmy_block->pcoord->dx3f(j);
	Real delta_l = fmax(fmax(dx1, dx2), dx3);
	Real dtrunc = fmax(0.0, sigma_cold)*tau_trunc / (kappa_s*delta_l);
	dtrunc = fmin(dtrunc_max, fmax(dfloor, dtrunc)); // dfloor <= dtrunc <= dtrunc_max
	Real fac_trunc = dtrunc / dfloor;
	Real wid_trunc = 0.5*std::log10(fac_trunc) / log(1./sigmoid_res - 1.);
	Real wdn_real = fmax(wdn-dfloor, dfloor_op);
	Real del_reduce = std::log10(dfloor) - std::log10(dfloor_op);

	Real fac_inv = 1.0;
	if (fabs(fac_trunc-1) > 1e-12) {
	  fac_inv = 1.0 + exp( -1./wid_trunc * (std::log10(wdn_real) - (std::log10(dfloor) + 0.5*std::log10(fac_trunc)) ) );
	}

	Real lg_rho_op = std::log10(wdn_real) - (1.-1./fac_inv) * del_reduce;
	wdn_opacity = pow(10.0, lg_rho_op);

	pmcb->rho(k,j,i) = wdn_opacity;
      }
    }
  }
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
  g(I23) = f * l_3 * l_3;
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
  g_inv(I23) = -f * l3 * l3;
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

}
