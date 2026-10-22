//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mc_readhdf.cpp
//! \brief Monte Carlo problem generator initialized from an athdf snapshot, either block
//! aligned with the snapshot's own grid or resampled onto a uniform mesh.

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
  // Helium nuclei per hydrogen nucleus, <problem>/heabund; the same key sets
  // MonteCarloBlock::heabund, so the number densities here, the free-free opacity, and
  // the library's default temperature inversion all describe one mixture.
  Real heabund = 0.09;

  //functions
  void GetNel(MonteCarloBlock *pmcb);
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
  heabund = pin->GetOrAddReal("problem", "heabund", heabund);
  // The tabulated opacities are the module's (emission = table, absorption = table,
  // <opacitytable>/opacity_file); this generator sets the electron density from them
  const std::string emis = pin->GetOrAddString("montecarlo","emission","none");
  const std::string absn = pin->GetOrAddString("montecarlo","absorption","none");
  if (emis == "user" || absn == "user") {
    std::stringstream msg;
    msg << "### FATAL ERROR in MonteCarlo::InitUserMonteCarloData" << std::endl
        << "mc_readhdf has no user emission or absorption; use "
        << "emission = table and absorption = table with <opacitytable>/opacity_file"
        << std::endl;
    ATHENA_ERROR(msg);
  }
  if (optab != nullptr) EnrollUserGetNumberDensity(GetNel);
}

void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin) {

  if (pmy_mc->emission_flag == EMISFF) {
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
  }
}

void Mesh::InitUserMeshData(ParameterInput *pin) {

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
//! - with <problem>/resampled the primitives are interpolated from the mesh-sized arrays
//!   InitUserMeshData filled; otherwise MCReadSnapshotBlock reads this block's slab from
//!   the snapshot named by <problem>/input_filename, or by <montecarlo>/grid_from_file
//!   when the grid was built from that snapshot.  Variables are located by name, so the
//!   layout does not have to be described here.

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  // <problem>/input_filename is not read here: the resampled branch works from
  // ruser_mesh_data, which InitUserMeshData already filled, and the block-aligned branch
  // lets MCReadSnapshotBlock resolve the file (falling back to <montecarlo>/grid_from_file
  // when no <problem>/input_filename is given).  Requiring it here made a run that names
  // its snapshot only in <montecarlo> fail with a confusing missing-parameter error.
  bool resampled = pin->GetOrAddBoolean("problem","resampled",false);

  if (resampled) {
    for (int k=ks; k<=ke; ++k) {
      Real z_now = pcoord->x3f(k);
      int index_znow = getindex(x3coord, z_now);
      for (int j=js; j<=je; ++j) {
        Real y_now = pcoord->x2f(j);
        int index_ynow = getindex(x2coord, y_now);
        for (int i=is; i<=ie; ++i) {
          Real x_now = pcoord->x1f(i);
          int index_xnow = getindex(x1coord, x_now);

          phydro->w(IDN,k,j,i) = pmy_mesh->ruser_mesh_data[0](index_znow, index_ynow,
                                                              index_xnow);
          phydro->w(IVX,k,j,i) = pmy_mesh->ruser_mesh_data[1](index_znow, index_ynow,
                                                              index_xnow);
          phydro->w(IVY,k,j,i) = pmy_mesh->ruser_mesh_data[2](index_znow, index_ynow,
                                                              index_xnow);
          phydro->w(IVZ,k,j,i) = pmy_mesh->ruser_mesh_data[3](index_znow, index_ynow,
                                                              index_xnow);
          phydro->w(IPR,k,j,i) = pmy_mesh->ruser_mesh_data[4](index_znow, index_ynow,
                                                              index_xnow);

        }// end i
      }//end j
    }// end k
  } else {
    // The hyperslab bookkeeping lives in MCReadSnapshotBlock, which locates variables by
    // name from the file, so an Athena++ dump (rho, press, vel1) and one converted from an
    // AthenaK run (dens, eint, velx) are both read without describing the layout here.
    // Mesh::nblist is private to everything but MeshBlock, so the collective-read padding
    // count is gathered here and handed over.
    int max_blocks_per_rank = 0;
    for (int r = 0; r < Globals::nranks; ++r)
      max_blocks_per_rank = std::max(max_blocks_per_rank, pmy_mesh->nblist[r]);
    MCReadSnapshotBlock(this, pin, max_blocks_per_rank);
  } // end if (resampled) else

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
  // for testing
  /*Real rho_const = pin->GetOrAddReal("problem", "rho_const", 0.);
    if (rho_const > 0.) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
    for (int i=is; i<=ie; ++i) {
    //if (phydro->w(IDN,k,j,i) > rho_const)
    phydro->w(IDN,k,j,i) = rho_const;
    }
    }
    }
    }
    Real temp_const = pin->GetOrAddReal("problem", "temp_const", 0.);
    if (temp_const > 0.) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
    for (int i=is; i<=ie; ++i) {
    //if (phydro->w(IPR,k,j,i)/phydro->w(IDN,k,j,i) > temp_const)
    phydro->w(IPR,k,j,i) = phydro->w(IDN,k,j,i) * temp_const;
    }
    }
    }
    }*/

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

    if (pmy_mc->emission_flag == EMISFF) {
      // Obtain intitial energy, polarization, direction and weight
      // Utilize free-free emission function in emission.cpp
      if(tnorm) {
        Real logtg = log(tgas(pphot->i3p[ip],pphot->i2p[ip],pphot->i1p[ip]));
        PhotonEmitFreeFree(this,pphot,logemin+logtg,logemax+logtg,ip);
      } else{
        PhotonEmitFreeFree(this,pphot,logemin,logemax,ip);
      }
    } else {
      // energy from the cell's tabulated emission spectrum, isotropic direction
      PhotonEmitTable(this,pphot,ip);
      if (pphot->IsNanPhoton(ip))
        pphot->PrintPhoton("initialization: ",ip);
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

// Electron and ion number densities.  The ion density is what the free-free emission and
// opacity read as species(1).  With table_ne the electron density is the table's; else
// the gas is taken as neutral where the table's grey Rosseland mean falls below the
// Thomson value of ionized gas and T < 1e5 K.
void GetNel(MonteCarloBlock *pmcb) {

  Real mp = 1.67262192369e-24;
  const OpacityTable *tab = pmcb->pmy_mc->optab;

  for (int k=pmcb->ks; k<=pmcb->ke; ++k) {
    for (int j=pmcb->js; j<=pmcb->je; ++j) {
      for (int i=pmcb->is; i<=pmcb->ie; ++i) {
        Real rho = pmcb->rho(k,j,i);
        Real nh = rho / (mp*(1.+4.*heabund));
        Real nhe = nh*heabund;
        pmcb->species(1,k,j,i) = nh + 4. * nhe;
        pmcb->species(0,k,j,i) = nh + 2. * nhe;

        Real tgas = pmcb->tgas(k,j,i);
        if (tab->table_ne) {
          // the free electrons per atom of the table
          pmcb->species(0,k,j,i) = tab->ElectronDensity(tgas, rho);
          continue;
        }
        Real ld = log10(rho);
        Real lt = log10(tgas);
        Real xi = (ld - tab->lmind) / tab->dld;
        int ii = std::floor(xi);
        if (ii < 0) {
          ii = 0;
        } else if (ii > tab->nrho-2) {
          ii = tab->nrho-2;
        }
        xi -= static_cast<Real>(ii);
        Real xj = (lt - tab->lmint) / tab->dlt;
        int jj = std::floor(xj);
        if (jj < 0)
          jj = 0;
        if (jj > tab->ntem-2)
          jj = tab->ntem-2;
        while ((jj<tab->ntem-2) && (tab->tgrid(jj+1) < tgas)){
          jj++;
        }
        while ((jj>0) && (tab->tgrid(jj) > tgas)){
          jj--;
        }
        if(jj < 0) {
          pmcb->species(0,k,j,i) = 0.; //below T grid, assume neutral
          continue;
        }
        xj = (tgas-tab->tgrid(jj))/(tab->tgrid(jj+1)-tab->tgrid(jj));
        Real ross = (1.-xi)*( (1.-xj)*tab->gray_ross(jj,ii)
                +xj*tab->gray_ross(jj+1,ii) ) + xi*( (1.-xj)*tab->gray_ross(jj,ii+1)
                +xj*tab->gray_ross(jj+1,ii+1) );
        if ((tgas < 1.e5) && (ross < 0.34)) {
          pmcb->species(0,k,j,i) = 0.; // assume neutral
        }
      } // loop over i
    }
  }
}

}
