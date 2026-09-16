//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file monte_carlo.cpp
//! \brief implementation of functions in class MonteCarlo, MCRandom

// C++ headers
#include <algorithm>  // min, max
#include <cmath>      // floor
#include <limits>     // numeric_limits
#include <cstdio>
#include <fstream>
#include <chrono>      // steady_clock, for the transport wall time
#include <ctime>       // clock(), CLOCKS_PER_SEC
#include <iostream>
#include <random>
#include <stdexcept>  // runtime_error
// C++ headers
#include <cstring>  // strcmp
#include <string>
#include <vector>

#ifdef MPI_PARALLEL
#include <sched.h>  // sched_yield, for the idle wait in TransportAsync
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif
#endif

// Athena++ headers
#include "mcexchange.hpp"
#include "mcpartition.hpp"
#include "montecarlo.hpp"
#include "../globals.hpp"
#include "../parameter_input.hpp"
#include "../mesh/mesh.hpp"
#include "../hydro/hydro.hpp"

// GSL library
#if GSL
#include <gsl/gsl_randist.h>
#endif

//----------------------------------------------------------------------------------------
//! MonteCarlo constructor, builds monte carlo using parameters in input file

MonteCarlo::MonteCarlo(ParameterInput *pin, Mesh *pmesh) {

  std::stringstream msg;

  pmy_mesh = pmesh;

  UserWorkInMove=nullptr;
  GetEmission=nullptr;
  UserGetDensity=nullptr;
  UserGetTemperature=nullptr;
  UserGetNumberDensity=nullptr;
  UserScattering=nullptr;
  UserScatteringOpacity=nullptr;
  user_moment_names=nullptr;
  user_moment_func=nullptr;
  user_moment_frame=nullptr;
  UserAbsorptionOpacity=nullptr;
  UserSourcetermFunc=nullptr;


  // read bc flags for each of the 6 physical boundaries.
  mc_bcs[BoundaryFace::inner_x1] = GetMCBoundaryFlag(pin->GetString("mesh","ix1_mc_bc"));
  mc_bcs[BoundaryFace::outer_x1] = GetMCBoundaryFlag(pin->GetString("mesh","ox1_mc_bc"));
  mc_bcs[BoundaryFace::inner_x2] = GetMCBoundaryFlag(pin->GetString("mesh","ix2_mc_bc"));
  mc_bcs[BoundaryFace::outer_x2] = GetMCBoundaryFlag(pin->GetString("mesh","ox2_mc_bc"));
  mc_bcs[BoundaryFace::inner_x3] = GetMCBoundaryFlag(pin->GetString("mesh","ix3_mc_bc"));
  mc_bcs[BoundaryFace::outer_x3] = GetMCBoundaryFlag(pin->GetString("mesh","ox3_mc_bc"));

  // intitialize boundary functions
  for (int dir=0; dir<6; dir++)
    BoundaryFunction_[dir]=nullptr;

  // SWD: replace dynamic/coupled with single method flag?
  using_bfield = pin->GetOrAddBoolean("montecarlo","bfields",false);
  dynamic = pin->GetOrAddBoolean("montecarlo","dynamic",false);
  coupled = pin->GetOrAddBoolean("montecarlo","coupled",false);
  boosts = pin->GetOrAddBoolean("montecarlo","boosts",false);
  polarized = GetMCPolarizationFlag(pin->GetOrAddString("montecarlo","polarized","none"));
  acceleration = pin->GetOrAddBoolean("montecarlo","acceleration",false);
  time_acc = pin->GetOrAddBoolean("montecarlo","time_acc",false);
  verbose = pin->GetOrAddBoolean("montecarlo", "verbose", true);
  lb_report = pin->GetOrAddBoolean("montecarlo", "lb_report", false);
  raytrace_flag = pin->GetOrAddBoolean("montecarlo", "raytrace", false);
  if (raytrace_flag)
    general_pusher_flag = true;
  else
    general_pusher_flag = pin->GetOrAddBoolean("montecarlo","general_pusher",false);
  scattering_meth = GetScatteringFlag(pin->GetOrAddString("montecarlo","scattering",
                                                          "none"));
  // Which metric the module integrates on.  Must come before SetGeometryTag and before
  // any MonteCarloBlock is constructed.
  SetCoordinateSystem(pin);
  // Canonical tag describing the geometry and wavevector convention of the outputs.
  // Must come after general_pusher_flag is known.
  SetGeometryTag(pin);
  // free parameters of that metric, for the output headers
  SetMetricParams(pin);
  // The frame the outputs measure directions and polarization in
  frame_tag = general_pusher_flag ? "normal" : "lab";
  nuser_var = 0; // photon user variables to zero
  nuser_mom = 0; // user moments

  // Set mininmum weight if using weighting for absorption
  weightratio = pin->GetOrAddReal("montecarlo","minweight",1.0e-20);

  // Number of outputs for static monte carlo
  nout = pin->GetOrAddInteger("montecarlo","nout",1);

  // Initialize Emmision parameters and methods
  InitializeEmission(pin);

  // Set default size parameters
  max_phots_init = pin->GetOrAddInteger("montecarlo","max_phots_init",10000);
  list_size_init = pin->GetOrAddInteger("montecarlo","list_size_init",10000);
  checkscat = pin->GetOrAddInteger("montecarlo","checkscat",10000);
  capmove = pin->GetOrAddInteger("montecarlo","capmove",0);

  // Initialize user MonteCarlo data before initializing MonteCarloBlocks
  // Should be caleld before Output constuctor
  InitUserMonteCarloData(pin);

  // Initialize output
  pmcout = new MCOutput(this,pin);

  // Create and intitialize randon number generator
  iseed = pin->GetInteger("montecarlo","iseed");

  // Initialize arrays sizes for photon instances
  Photon::Initialize(this,pin);

  // Initialize ncells and broadcast
  if (Globals::my_rank == 0) {
    ncells = pmesh->GetTotalCells();
  }
#ifdef MPI_PARALLEL
  // then broadcasts it
  MPI_Bcast(&ncells, sizeof(int64_t), MPI_BYTE, 0, MPI_COMM_WORLD);
#endif

  // Create MonteCarloBlock for each MeshBlock for this process
  nblocal = pmy_mesh->nblocal;
  nbtotal = pmy_mesh->nbtotal;
  int root_level = pmy_mesh->root_level;
  my_blocks.NewAthenaArray(nblocal);

  for (int i=0; i<nblocal; i++) {
    MeshBlock *pmb = pmy_mesh->my_blocks(i);
    my_blocks(i) = new MonteCarloBlock(pmb, NULL, this, pin);
    pmb->pmy_mcb = my_blocks(i);
    // Set neighbors for photon class
    // SWD: ideally moved to mesh constructor, but awkward
    int nrbx1 = pmy_mesh->mesh_size.nx1/pmb->block_size.nx1;
    int nrbx2 = pmy_mesh->mesh_size.nx2/pmb->block_size.nx2;
    int nrbx3 = pmy_mesh->mesh_size.nx3/pmb->block_size.nx3;
    my_blocks(i)->pphot->LinkNeighbors(pmy_mesh->tree, nrbx1, nrbx2, nrbx3, root_level);
    // Which blocks straddle a rank boundary is fixed once the neighbors are linked, and
    // the transfer round needs it every time, so record it here rather than rediscover
    // it each round.
    my_blocks(i)->pphot->SetOffRankNeighborFlag();
  }

  // Photons crossing a rank boundary move a rank at a time rather than a block-neighbor
  // at a time.  Off by setting <montecarlo>/rank_exchange = false, which restores the
  // per-block protocol; the two give identical results, the difference is message count.
  // The mesh calls back into the module when it redistributes blocks; see the hooks in
  // amr_loadbalance.cpp and PackDeparting / RebuildAfterRedistribution below.
  pmy_mesh->pmc = this;
  lb_epoch = 0;
  lb_rank_time = 0.0;
  lb_transport_wall = 0.0;
  lb_idle_time = lb_exchange_time = 0.0;
  lb_t_complete = lb_t_drain_in = lb_t_drain_arr = lb_t_local = lb_t_send = 0.0;
  lb_passes = 0;
  per_block_cap_ = 10000;
  photon_budget_ = 1000000;
#ifdef MPI_PARALLEL
  MPI_Comm_dup(MPI_COMM_WORLD, &lb_comm_);
#endif
  {
    const std::string mode = pin->GetOrAddString("montecarlo", "lb_test_repack", "none");
    if (mode == "none") {
      lb_test_repack = LBTEST_NONE;
    } else if (mode == "local") {
      lb_test_repack = LBTEST_LOCAL;
    } else if (mode == "mesh") {
      lb_test_repack = LBTEST_MESH;
    } else {
      msg << "### FATAL ERROR in MonteCarlo constructor" << std::endl
          << "<montecarlo>/lb_test_repack = " << mode << "; use none, local or mesh"
          << std::endl;
      ATHENA_ERROR(msg);
    }
    lb_cost_file = pin->GetOrAddString("loadbalancing", "cost_file", "");
    lb_costs_loaded = false;
    lb_min_gain = pin->GetOrAddReal("montecarlo", "lb_min_gain", 0.05);
    lb_check_interval = pin->GetOrAddInteger("montecarlo", "lb_check_interval", 0);
    lb_max_per_transport = pin->GetOrAddInteger("montecarlo", "lb_max_per_transport", 4);
    lb_min_window = pin->GetOrAddInteger("montecarlo", "lb_min_window", lb_check_interval);
    lb_check_fraction = pin->GetOrAddReal("montecarlo", "lb_check_fraction", 0.1);
    lb_cost_decay = pin->GetOrAddReal("montecarlo", "lb_cost_decay", 1.0);
    const std::string part = pin->GetOrAddString("montecarlo", "lb_partition", "optimal");
    if (part == "optimal") {
      lb_partition_optimal = true;
    } else if (part == "greedy") {
      lb_partition_optimal = false;
    } else {
      msg << "### FATAL ERROR in MonteCarlo constructor" << std::endl
          << "<montecarlo>/lb_partition = " << part << "; use optimal or greedy" << std::endl;
      ATHENA_ERROR(msg);
    }
    const std::string init = pin->GetOrAddString("montecarlo", "lb_initial", "none");
    if (init == "none") {
      lb_initial_photons = false;
    } else if (init == "photons") {
      lb_initial_photons = true;
    } else {
      msg << "### FATAL ERROR in MonteCarlo constructor" << std::endl
          << "<montecarlo>/lb_initial = " << init << "; use none or photons" << std::endl;
      ATHENA_ERROR(msg);
    }
    const std::string cmode = pin->GetOrAddString("montecarlo", "lb_test_costs", "measured");
    if (cmode == "measured") {
      lb_test_costs = LBCOST_MEASURED;
    } else if (cmode == "alternate") {
      lb_test_costs = LBCOST_ALTERNATE;
    } else {
      msg << "### FATAL ERROR in MonteCarlo constructor" << std::endl
          << "<montecarlo>/lb_test_costs = " << cmode << "; use measured or alternate"
          << std::endl;
      ATHENA_ERROR(msg);
    }
  }

  local_max_sweeps = pin->GetOrAddInteger("montecarlo", "local_max_sweeps", 1000);
  async_term = pin->GetOrAddBoolean("montecarlo", "async_term", true);
  pexch = nullptr;
  if (pin->GetOrAddBoolean("montecarlo", "rank_exchange", true)) {
    pexch = new MCRankExchange(this);
    pexch->BuildPeerList();
    if (Globals::my_rank == 0 && pexch->Active() && verbose) {
      std::cout << "Monte Carlo photon exchange: aggregated by rank, "
                << pexch->NumPeers() << " peer(s) on rank 0" << std::endl;
    }
  }
}

//----------------------------------------------------------------------------------------
//! destructor

MonteCarlo::~MonteCarlo() {

  delete pmcout;
  delete pexch;
#ifdef MPI_PARALLEL
  MPI_Comm_free(&lb_comm_);
#endif
  for (int i=0; i<nblocal; i++)
    delete my_blocks(i);
}

//----------------------------------------------------------------------------------------
//! \fn enum AbsorptionOpacityFlag GetAbsorptionOpacityFlag(std::string input_string)
//! \brief set absorption opacity flag

enum AbsorptionOpacityFlag GetAbsorptionOpacityFlag(std::string input_string) {
  if (input_string == "user") {
    return ABSUSER;
  } else if (input_string == "none") {
    return ABSNONE;
  } else if (input_string == "freefree") {
    return ABSFF;
  } else if (input_string == "dust") {
    return ABSDUST;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in GetAbsorptionOpacityFlag" << std::endl
        << "Input string=" << input_string << " not valid absorption opacity"
        << std::endl;
    ATHENA_ERROR(msg);
  }
}

//----------------------------------------------------------------------------------------
//! \fn enum AbsorptionMethodFlag GetAbsorptionMethodFlag(std::string input_string)
//! \brief set absorption method flag

enum AbsorptionMethodFlag GetAbsorptionMethodFlag(std::string input_string) {
  if (input_string == "weight") {
    return ABSWEIGHT;
  } else if (input_string == "prob") {
    return ABSPROB;
  } else if (input_string == "tau") {
    return ABSTAU;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in GetAbsorptionMethodFlag" << std::endl
        << "Input string=" << input_string << " not valid absorption method" << std::endl;
    ATHENA_ERROR(msg);
  }

}

//----------------------------------------------------------------------------------------
//! \fn enum MCPolarization GetMCPolarizationFlag(std::string input_string)
//! \brief set polarization tracking flag
//
// Accepts the legacy booleans as well as the named modes, so existing input files keep
// working: false means none, true means linear, which is what "true" has always done.

enum MCPolarization GetMCPolarizationFlag(std::string input_string) {
  if (input_string == "none" || input_string == "false"
      || input_string == "0" || input_string == "False") {
    return MCPOL_NONE;
  } else if (input_string == "linear" || input_string == "true"
             || input_string == "1" || input_string == "True") {
    return MCPOL_LINEAR;
  } else if (input_string == "circular") {
    return MCPOL_CIRCULAR;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in GetMCPolarizationFlag" << std::endl
        << "Input string=" << input_string << " not a valid polarization mode."
        << std::endl
        << "Use none, linear or circular (true and false are still accepted, "
        << "as linear and none)." << std::endl;
    ATHENA_ERROR(msg);
  }
}

//----------------------------------------------------------------------------------------
//! \fn enum ScatteringFlag GetScatteringFlag(std::string input_string)
//! \brief set scattering flag

enum ScatteringFlag GetScatteringFlag(std::string input_string) {

  if (input_string == "user") {
    return SCATUSER;
  } else if (input_string == "none") {
    return SCATNONE;
  } else if (input_string == "isotropic") {
    return SCATISO;
  } else if (input_string == "thomson") {
    return SCATTHOM;
  } else if (input_string == "compton") {
    return SCATCOMP;
  } else if (input_string == "resonance") {
    return SCATRES;
  } else if (input_string == "dust") {
    return SCATDUST;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in GetScatteringFlag" << std::endl
        << "Input string=" << input_string << " not valid scattering type" << std::endl;
    ATHENA_ERROR(msg);
  }

}

//----------------------------------------------------------------------------------------
//! \fn enum EmissionFlag GetEmissionFlag(std::string input_string)
//! \brief set emission flag

enum EmissionFlag GetEmissionFlag(std::string input_string) {

  if (input_string == "none") {
    return EMISNONE;
  } else if (input_string == "user") {
    return EMISUSER;
  } else if (input_string == "freefree") {
    return EMISFF;
  } else if (input_string == "blackbody") {
    return EMISBB;
  } else if (input_string == "multi") {
    return MULTI;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in GetEmissionFlag" << std::endl
        << "Input string=" << input_string << " not valid emission type" << std::endl;
    ATHENA_ERROR(msg);
  }

}

//----------------------------------------------------------------------------------------
//! \fn enum EmissionGeometery GetEmissionGeometry(std::string input_string)
//! \brief set emission flag

enum EmissionGeometry GetEmissionGeometry(std::string input_string) {

  if (input_string == "volume") {
    return EMISVOL;
  } else if (input_string == "area") {
    return EMISAREA;
  } else if (input_string == "none") {
    return EMISGNONE;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in GetEmissionGeometry" << std::endl
        << "Input string=" << input_string << " not valid emission geometry" << std::endl;
    ATHENA_ERROR(msg);
  }

}

//----------------------------------------------------------------------------------------
//! \fn enum BoundaryFace SetEmissionSurface(std::string input_face)
//! \brief set emission surface

enum BoundaryFace SetEmissionSurface(std::string input_face) {

  if (input_face == "inner_x1") {
    return BoundaryFace::inner_x1;
  } else if (input_face == "outer_x1") {
    return BoundaryFace::outer_x1;
  } else if (input_face == "inner_x2") {
    return BoundaryFace::inner_x2;
  } else if (input_face == "outer_x2") {
    return BoundaryFace::outer_x2;
  } else if (input_face == "inner_x3") {
    return BoundaryFace::inner_x3;
  } else if (input_face == "outer_x3") {
    return BoundaryFace::outer_x3;
  } else if (input_face == "none") {
    return BoundaryFace::undef;
  } else {
    std::stringstream msg;
      msg << "### FATAL ERROR in function [SetEmissionSurface]" << std::endl
          << "Face not recognized in input." << std::endl;
      throw std::runtime_error(msg.str().c_str());
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::EnrollUserMCBoundaryFunction(enum BoundaryFace dir,
//!       BValHydro_t my_bc)
//!  \brief Enroll a user-defined monte carlo boundary function

void MonteCarlo::EnrollUserMCBoundaryFunction(enum BoundaryFace dir, MCBValFunc_t my_bc) {
  std::stringstream msg;
  if (dir<0 || dir>5) {
    msg << "### FATAL ERROR in EnrollMCBoundaryCondition function" << std::endl
        << "dirName = " << dir << " not valid" << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  if (mc_bcs[dir]!=MC_USER_BNDRY) {
    msg << "### FATAL ERROR in EnrollUserMCBoundaryFunction" << std::endl
        << "The boundary condition flag must be set to the string 'user' in the "
        << " <mesh> block in the input file to use user-enrolled BCs" << std::endl;
    ATHENA_ERROR(msg);
  }
  BoundaryFunction_[dir]=my_bc;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::EnrollUserEmissionFunction(EmisFunc_t emissfunc)
//! \brief Enroll a user-defined function for computing emission array

void MonteCarlo::EnrollUserEmissionFunction(EmisFunc_t emissfunc) {

  EnrollUserEmissionFunction(emissfunc,0);
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::EnrollUserEmissionFunction(EmisFunc_t emissfunc, int etype)
//! \brief Enroll a user-defined function for computing emission array

void MonteCarlo::EnrollUserEmissionFunction(EmisFunc_t emissfunc, int etype) {

  GetEmission[etype] = emissfunc;
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::EnrollUserGetDensity(DensFunc_t densfunc)
//! \brief Enroll a user-defined function for computing density

void MonteCarlo::EnrollUserGetDensity(DensFunc_t densfunc) {

  UserGetDensity = densfunc;
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::EnrollUserGetTemperature(TempFunc_t tempfunc)
//! \brief Enroll a user-defined function for computing temperature

void MonteCarlo::EnrollUserGetTemperature(TempFunc_t tempfunc) {

  UserGetTemperature = tempfunc;
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::EnrollUserGetNumberDensity(NumbFunc_t numbfunc)
//! \brief Enroll a user-defined function for computing number densities

void MonteCarlo::EnrollUserGetNumberDensity(NumbFunc_t numbfunc) {

  UserGetNumberDensity = numbfunc;
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::EnrollUserWorkInMove(UserMoveFunc_t userfunc)
//! \brief Enroll a user-defined condition to be called during photon moves

void MonteCarlo::EnrollUserWorkInMove(UserMoveFunc_t userfunc) {

  UserWorkInMove = userfunc;
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::EnrollUserScatteringFunction(ScatFunc_t scatfunc)
//! \brief Enroll a user-defined scattering function

void MonteCarlo::EnrollUserScatteringFunction(ScatFunc_t scatfunc) {

  UserScattering = scatfunc;
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::EnrollUserOpacityFunction(OpacFunc_t opacfunc, bool abs)
//! \brief Enroll a user-defined opacity function

void MonteCarlo::EnrollUserOpacityFunction(OpacFunc_t opacfunc, bool abs) {

  if (abs)
    UserAbsorptionOpacity = opacfunc;
  else
    UserScatteringOpacity = opacfunc;
}

//----------------------------------------------------------------------------------------
//! \fn void void MonteCarlo::AllocateUserMoments(int n)
//! \brief allocate user moments

void MonteCarlo::AllocateUserMoments(int n) {

  nuser_mom = n;
  user_moment_names = new std::string[n];
  user_moment_func = new UserMomentFunc_t[n];
  user_moment_frame = new MCFrame[n];
  for (int i=0; i<n; ++i) user_moment_frame[i] = MCFRAME_LAB;
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::EnrollUserMoment(int i, UserMomentFunc_t my_func,
//                                        const char *name)
//! \brief Enroll a user-defined history output function and set its name

void MonteCarlo::EnrollUserMoment(int i, UserMomentFunc_t my_func, const char *name,
                                  MCFrame frame) {

  std::stringstream msg;
  if (i >= nuser_mom) {
    msg << "### FATAL ERROR in EnrollUserMoment function" << std::endl
        << "The number of the user-defined moment (" << i << ") "
        << "exceeds the declared number (" << nuser_mom << ")." << std::endl;
    ATHENA_ERROR(msg);
  }
  user_moment_names[i] = name;
  user_moment_func[i] = my_func;
  user_moment_frame[i] = frame;

}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::EnrollUserSourcetermUpdate(UserSourctermFunc_t my_func)
//! \brief Enroll a user-defined source term function

void MonteCarlo::EnrollUserSourcetermUpdate(UserSourcetermFunc_t my_func) {

  UserSourcetermFunc = my_func;

}
//----------------------------------------------------------------------------------------
//! \fn enum MCBoundaryFlag GetMCBoundaryFlag(std::string input_string)
//! \brief set boundary flag

enum MCBoundaryFlag GetMCBoundaryFlag(std::string input_string) {

  if (input_string == "periodic") {
    return MC_PERIODIC_BNDRY;
  } else if (input_string == "escape") {
    return MC_ESCAPE_BNDRY;
  } else if (input_string == "absorb") {
    return MC_ABSORB_BNDRY;
  } else if (input_string == "destroy") {
    return MC_DESTROY_BNDRY;
  } else if (input_string == "polar") {
    return MC_POLAR_BNDRY;
  } else if (input_string == "reflecting") {
    return MC_REFLECT_BNDRY;
  } else if (input_string == "user") {
    return MC_USER_BNDRY;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in GetMCBoundaryFlag" << std::endl
        << "Input string=" << input_string << " not valid boundary type" << std::endl;
    ATHENA_ERROR(msg);
  }

}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::Initialize(ParameterInput *pinput)
//! \brief initialize grid data in each monte carlo block

void MonteCarlo::Initialize(ParameterInput *pin) {

  if (dynamic) {
    tmax = pin->GetOrAddReal("montecarlo","tmax",-1.);
    if (tmax < 0.)
      tmax = pmy_mesh->dt;
    tint = pmy_mesh->dt;
  } else {
    // initialize timing parameters if static calculation
    tint = pin->GetOrAddReal("montecarlo","tint",1.);
    tmax = pin->GetOrAddReal("montecarlo","tmax",HUGE_NUMBER);
  }
  // convert to cgs units
  Real vel_cgs = pin->GetOrAddReal("problem","vel_cgs",1.);
  Real l_cgs = pin->GetOrAddReal("problem","l_cgs",1.);
  Real time_cgs = l_cgs/vel_cgs;
  tint *= time_cgs;
  tmax *= time_cgs;

  // Initialize monte carlo blocks
 
  // loop_max_size caps the photons resident on ONE block, so the memory it authorizes is
  // multiplied by however many blocks land on a rank. max_resident_photons bounds that
  // product instead, which is the quantity that has to fit in memory. The per-block cap
  // still applies on top.
  const int per_block_cap = pin->GetOrAddInteger("montecarlo","loop_max_size",10000);
  const int photon_budget =
      pin->GetOrAddInteger("montecarlo","max_resident_photons",1000000);
  per_block_cap_ = per_block_cap;
  photon_budget_ = photon_budget;
  const int loop_max = ComputeLoopMax();

  // Report only when the budget actually binds; otherwise this is noise.  The range is
  // taken across ranks because an unbalanced mesh gives them different block counts and so
  // different shares.
  {
    int lo = loop_max, hi = loop_max, nbmax = nblocal;
#ifdef MPI_PARALLEL
    MPI_Allreduce(MPI_IN_PLACE, &lo, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &hi, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &nbmax, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
    if (Globals::my_rank == 0 && lo < per_block_cap) {
      std::cout << "Monte Carlo: loop_max_size cut from " << per_block_cap << " to ";
      if (lo == hi) std::cout << lo;
      else std::cout << lo << "-" << hi;
      std::cout << " by <montecarlo>/max_resident_photons = " << photon_budget
                << " (up to " << nbmax << " blocks per rank)" << std::endl;
    }
  }

  for (int i=0; i<nblocal; i++) {
    MonteCarloBlock *pmcb = my_blocks(i);
    // Initialize variables over all blocks
    SetupBlockFromFluid(pmcb);

    // initialize counters to zero
    pmcb->nscat = pmcb->nesc = pmcb->nabs = pmcb->ndes = pmcb->nrem = 0;
    pmcb->loop_max_size = loop_max;

    // Call problem generators for Monte Carlo
    pmcb->MonteCarloProblemGenerator(pin);
  }

  // Costs from an earlier run on this mesh, so the first transport is already balanced
  if (!dynamic && !lb_cost_file.empty()) lb_costs_loaded = ReadCostFile();
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::SetCoordinateSystem(ParameterInput *pin)
//! \brief resolve the metric the Monte Carlo module will integrate on
//
// COORDINATE_SYSTEM fixes the choice where it can, and <montecarlo>/mc_coord
// supplies it where it cannot.  mc_coord is required for gr_user and optional elsewhere,
// where it is checked for consistency rather than obeyed -- naming a metric that
// contradicts the build is a mistake worth reporting, not a request.
//
// Must be called before SetGeometryTag and before any MonteCarloBlock is constructed.

void MonteCarlo::SetCoordinateSystem(ParameterInput *pin) {

  // std::strcmp, not ==: COORDINATE_SYSTEM is a bare string literal, so == compares
  // addresses.  This is the one place in the module that has to look at it.
  bool implied_known = true;
  MCCoordSystem implied;
  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
    implied = MCCOORD_CARTESIAN;
  } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    implied = MCCOORD_CYLINDRICAL;
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    implied = MCCOORD_SPHERICAL_POLAR;
  } else if (std::strcmp(COORDINATE_SYSTEM, "minkowski") == 0) {
    implied = MCCOORD_MINKOWSKI;
  } else if (std::strcmp(COORDINATE_SYSTEM, "kerr-schild") == 0) {
    // Retained rather than folded into mc_coord so existing input files keep working.
    implied = pin->GetOrAddBoolean("montecarlo","boyerlindquist",false)
              ? MCCOORD_BOYER_LINDQUIST : MCCOORD_KERR_SCHILD;
  } else {
    // gr_user, or a coordinate system the module does not support at all
    implied_known = false;
    implied = MCCOORD_KERR_SCHILD_CARTESIAN;
  }

  std::string name = pin->GetOrAddString("montecarlo","mc_coord","");

  if (name.empty()) {
    if (!implied_known) {
      std::stringstream msg;
      msg << "### FATAL ERROR in MonteCarlo::SetCoordinateSystem" << std::endl
          << "Coordinate system '" << COORDINATE_SYSTEM << "' does not determine the "
          << "Monte Carlo metric." << std::endl
          << "Set <montecarlo>/mc_coord to one of: kerr_schild_cartesian, snake."
          << std::endl
          << "Note that this previously defaulted to kerr_schild_cartesian; set that "
          << "explicitly to reproduce the old behaviour." << std::endl;
      ATHENA_ERROR(msg);
    }
    coord_system = implied;
  } else {
    bool matched = false;
    for (int c = MCCOORD_CARTESIAN; c <= MCCOORD_SNAKE; ++c) {
      if (name == GetMCCoordSystemName(static_cast<MCCoordSystem>(c))) {
        coord_system = static_cast<MCCoordSystem>(c);
        matched = true;
        break;
      }
    }
    if (!matched) {
      std::stringstream msg;
      msg << "### FATAL ERROR in MonteCarlo::SetCoordinateSystem" << std::endl
          << "Unrecognized <montecarlo>/mc_coord = '" << name << "'." << std::endl
          << "Valid values are:";
      for (int c = MCCOORD_CARTESIAN; c <= MCCOORD_SNAKE; ++c)
        msg << " " << GetMCCoordSystemName(static_cast<MCCoordSystem>(c));
      msg << std::endl;
      ATHENA_ERROR(msg);
    }
    if (implied_known && coord_system != implied) {
      std::stringstream msg;
      msg << "### FATAL ERROR in MonteCarlo::SetCoordinateSystem" << std::endl
          << "<montecarlo>/mc_coord = '" << name << "' contradicts the configured "
          << "coordinate system '" << COORDINATE_SYSTEM << "', which implies '"
          << GetMCCoordSystemName(implied) << "'." << std::endl;
      ATHENA_ERROR(msg);
    }
  }

  // The pusher is chosen from the coordinate system, not from general_pusher_flag: the
  // MonteCarloBlock constructor honours the flag only for Cartesian and spherical-polar
  // and builds a GeneralPusher unconditionally for everything else.  The flag still
  // selects the four-vector storage convention and gates the polarization and frame
  // machinery, so the two must agree.  Reject rather than silently correct: the flag also
  // changes what the outputs mean, so flipping it under the user would misdescribe files
  // they are about to write.
  if (IsMCPusherAlwaysGeneral(coord_system) && !general_pusher_flag) {
    std::stringstream msg;
    msg << "### FATAL ERROR in MonteCarlo::SetCoordinateSystem" << std::endl
        << "Coordinate system '" << GetMCCoordSystemName(coord_system)
        << "' is always integrated with GeneralPusher, but <montecarlo>/general_pusher"
        << std::endl
        << "is false.  Set <montecarlo>/general_pusher = true." << std::endl
        << std::endl
        << "general_pusher does not select the pusher.  It selects the four-vector "
        << "storage" << std::endl
        << "convention -- with it false, TransformToCoordinate stores a unit spatial "
        << "direction" << std::endl
        << "plus ep while RK4Step reads k1p..k3p as contravariant components, so the "
        << "geodesics" << std::endl
        << "are integrated on the wrong vector.";
    if (IsPolarized(polarized)) {
      msg << "  With polarized = "
          << GetMCPolarizationName(polarized) << " it also silently disables the"
          << std::endl
          << "Stokes/coherency conversions in polarization.cpp and drops the "
          << "polarization" << std::endl
          << "tensor whenever a photon crosses a MeshBlock boundary.";
    }
    msg << std::endl;
    ATHENA_ERROR(msg);
  }

  topology = GetMCTopology(coord_system);
  curved_metric = IsMCMetricCurved(coord_system);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::SetMetricParams(ParameterInput *pin)
//! \brief record the metric's free parameters for the output headers
//
// Parameters of special metrics, such as spin and mass in Kerr.  Written as
// "key=value" pairs so readers can parse it generically and so a metric added
// later needs no reader change.

void MonteCarlo::SetMetricParams(ParameterInput *pin) {

  char buf[256];
  switch (coord_system) {
    case MCCOORD_KERR_SCHILD:
    case MCCOORD_BOYER_LINDQUIST:
    case MCCOORD_KERR_SCHILD_CARTESIAN:
      std::snprintf(buf, sizeof(buf), "m=%.17g,a=%.17g",
                    pin->GetOrAddReal("coord", "m", 1.0),
                    pin->GetOrAddReal("coord", "a", 0.0));
      metric_params = buf;
      break;
    case MCCOORD_SNAKE:
      std::snprintf(buf, sizeof(buf), "snake_a=%.17g,snake_k=%.17g",
                    pin->GetOrAddReal("coord", "snake_a", 0.0),
                    pin->GetOrAddReal("coord", "snake_k", 0.0));
      metric_params = buf;
      break;
    default:
      // flat metrics in their own coordinates have no free parameters
      metric_params = "";
      break;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::SetGeometryTag(ParameterInput *pin)
//! \brief build the canonical geometry tag written into photon list/trajectory headers
//
// coord_system fixes the metric but still does not say enough to interpret an output
// list: cartesian and spherical_polar select GeneralPusher or the legacy pusher
// depending on general_pusher, and the two store the wavevector differently.  The legacy
// pushers keep a unit orthonormal three-vector alongside the energy, while GeneralPusher
// keeps genuine contravariant components 
//
//! Post-processing has to know both, so the tag encodes them.  It doubles as a format
//! discriminator: the tags below are new strings, so a list written before this change
//! (coord=kerr-schild, gr_user, minkowski, ...) is recognizably the old layout, in which
//! the energy column held k^t rather than the conserved -k_t.
//
// Must be called after SetCoordinateSystem and after general_pusher_flag is known.

void MonteCarlo::SetGeometryTag(ParameterInput *pin) {

  switch (coord_system) {
    case MCCOORD_CARTESIAN:
      geometry_tag = general_pusher_flag ? "cartesian_gp" : "cartesian";
      break;
    case MCCOORD_SPHERICAL_POLAR:
      geometry_tag = general_pusher_flag ? "spherical_gp" : "spherical_polar";
      break;
    case MCCOORD_CYLINDRICAL:
      geometry_tag = "cylindrical_gp";
      break;
    case MCCOORD_MINKOWSKI:
      geometry_tag = "minkowski_cart";
      break;
    case MCCOORD_KERR_SCHILD:
      geometry_tag = "ks_spherical";
      break;
    case MCCOORD_BOYER_LINDQUIST:
      geometry_tag = "bl_spherical";
      break;
    case MCCOORD_KERR_SCHILD_CARTESIAN:
      geometry_tag = "ks_cartesian";
      break;
    case MCCOORD_SNAKE:
      geometry_tag = "snake_cart";
      break;
  }

  // Flat-but-relativistic metrics (Minkowski, snake) carry -k_t in the list just as the
  // curved ones do, so this asks IsMCRelativistic rather than curved_metric.
  relativistic_output = IsMCRelativistic(coord_system);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void void MonteCarlo::InitializeEmission(ParameterInput *pin)
//! \brief initialize flags that control emission, called for single emission type

void MonteCarlo::InitializeEmission(ParameterInput *pin) {

  // Options currently include none, user, and free-free
  // "none" means that no arrays are set up to assist photon initialization
  // "user" means that meshblock arrays will be set up but the user will provide the
  //        emissivity function
  // "freefree" mean that the default freefree emissivity function will be used for
  //            the emission array
  // "multi" means that there are multiple emission methods to be considered

  emission_flag = GetEmissionFlag(pin->GetOrAddString("montecarlo","emission","none"));

  if (emission_flag != MULTI) {
    ntype = 1;
  } else {
    ntype = pin->GetInteger("montecarlo","ntype"); // must be set for multi
  }
  nsamptype = new int64_t[ntype];
  emission_eqwt = new bool[ntype];
  initialize_comoving = new bool[ntype];
  std::string abs_def = pin->GetOrAddString("montecarlo","abs_method","weight");
  absorption_method = new AbsorptionMethodFlag[ntype];
  GetEmission = new EmisFunc_t[ntype];
  emission_geometry = new int[ntype];
  emission_face = new BoundaryFace[ntype];
  // Set emmisivity functions and flag for determining emission array
  if (emission_flag == EMISNONE) {
    GetEmission[0] = nullptr; // left unset
    nsamptype[0] = nsamp = pin->GetInteger64("montecarlo","nphot");
    emission_eqwt[0] = pin->GetOrAddBoolean("montecarlo","equal_weight",false);
    initialize_comoving[0] = pin->GetOrAddBoolean("montecarlo","initialize_comoving",true);
    absorption_method[0] = GetAbsorptionMethodFlag(abs_def);
    emission_geometry[0] = GetEmissionGeometry(pin->GetOrAddString("montecarlo","emission_geometry","none"));
    emission_array = false; // do not allocate memory for array;
  } else if (emission_flag ==  EMISUSER) {
    GetEmission[0] = nullptr; // must be set in InitUserMonteCarloData
    nsamptype[0] = nsamp = pin->GetInteger64("montecarlo","nphot");
    emission_eqwt[0] = pin->GetOrAddBoolean("montecarlo","equal_weight",false);
    initialize_comoving[0] = pin->GetOrAddBoolean("montecarlo","initialize_comoving",true);
    absorption_method[0] = GetAbsorptionMethodFlag(abs_def);
    emission_geometry[0] = GetEmissionGeometry(pin->GetOrAddString("montecarlo","emission_geometry","volume"));
    if (emission_geometry[0] == EMISAREA)
      emission_face[0] = SetEmissionSurface(pin->GetOrAddString("montecarlo","emission_face","none"));
    emission_array = true; // allocate memory for array
  } else if (emission_flag ==  EMISFF) {
    GetEmission[0] = GetEmissionFreeFree;
    nsamptype[0] = nsamp = pin->GetInteger64("montecarlo","nphot");
    emission_eqwt[0] = pin->GetOrAddBoolean("montecarlo","equal_weight",false);
    initialize_comoving[0] = pin->GetOrAddBoolean("montecarlo","initialize_comoving",true);
    absorption_method[0] = GetAbsorptionMethodFlag(abs_def);
    emission_geometry[0] = EMISVOL; // Must be volumetric
    emission_array = true; // allocate memory for array
  } else if (emission_flag ==  EMISBB) {
    GetEmission[0] = GetEmissionBlackbody;
    nsamptype[0] = nsamp = pin->GetInteger64("montecarlo","nphot");
    emission_eqwt[0] = pin->GetOrAddBoolean("montecarlo","equal_weight",false);
    initialize_comoving[0] = pin->GetOrAddBoolean("montecarlo","initialize_comoving",true);
    absorption_method[0] = GetAbsorptionMethodFlag(abs_def);
    emission_geometry[0] = EMISAREA; // Must be areal
    emission_face[0] = SetEmissionSurface(pin->GetString("montecarlo","emission_face"));
    emission_array = true; // allocate memory for array
  } else if (emission_flag ==  MULTI) {
    // GetEmission and other arrays must be set in InitUserMonteCarloData
    emission_array = pin->GetOrAddBoolean("montecarlo","emission_array",true);
  }


}


//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::DistributeSamples(int etype)
//! \brief Distribute the samples across blocks based on emission functions

void MonteCarlo::DistributeSamples(int etype) {

  if (emission_flag == EMISNONE) {
    // Do nothing.  nphremain for each block needs to be set in the
    // problem generator based on the user's criteria
    return;
  } else if (GetEmission == nullptr) {
    std::stringstream msg;
    msg << "### FATAL ERROR in DistributeSamples" << std::endl
        << "emission method is not none, but GetEmission is not set."
        << std::endl;
    ATHENA_ERROR(msg);
  }
  // Set methods, numbers for this emission type. 64-bit throughout
  const int64_t ntot = nsamptype[etype];
  if (ntot < 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in DistributeSamples" << std::endl
        << "photon count for emission type " << etype << " is " << ntot
        << "; a count above 2^31-1 read through a 32-bit path wraps to a value like this"
        << std::endl;
    ATHENA_ERROR(msg);
  }
  bool equal_weight = emission_eqwt[etype];
  
  // compute emission properties over all blocks on this process
  Real em_min = SQR(HUGE_NUMBER), em_max = -HUGE_NUMBER, em_tot = 0.;
  Real *tot_block = new Real[nblocal];

  for (int nb=0; nb<nblocal; nb++) {
    Real min_block, max_block;
    my_blocks(nb)->ComputeEmissionArray(etype,min_block,max_block,tot_block[nb]);
    em_tot += tot_block[nb];
    em_min = (em_min < min_block) ? em_min : min_block;
    em_max = (em_max > max_block) ? em_max : max_block;
  }
  //printf("em: %d %g %g %g\n",Globals::my_rank,em_min,em_max,em_tot);
  Real em_proc = em_tot;

  // Compute emmision properties over all processes
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE,&em_min,1,MPI_ATHENA_REAL,MPI_MIN,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&em_max,1,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&em_tot,1,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
#endif

  if (equal_weight) {
    // emmision weights are all equal 

    // First, each process sends its its own block totals to rank 0
    std::vector<Real> emiss_proc(Globals::nranks, 0.0);
#ifdef MPI_PARALLEL
    MPI_Gather(&em_proc,1,MPI_ATHENA_REAL,emiss_proc.data(),1,MPI_ATHENA_REAL,0,
               MPI_COMM_WORLD);
#else
    emiss_proc[0] = em_proc;
#endif
    std::vector<int64_t> count(Globals::nranks, 0);
    // Rank 0 compute the distribution of photons accross all processes and brodcasts
    if (Globals::my_rank == 0) {
      std::vector<Real> prob(Globals::nranks);
      for (int irank=0; irank<Globals::nranks; irank++)
        prob[irank] = emiss_proc[irank]/em_tot;
      my_blocks(0)->pran->SampleMultinomial(ntot,Globals::nranks,prob.data(),count.data());
    }
    int64_t my_count;
#ifdef MPI_PARALLEL
    MPI_Scatter(count.data(),1,MPI_INT64_T,&my_count,1,MPI_INT64_T,0,MPI_COMM_WORLD);
#else
    my_count = count[0];
#endif
    // Now distribute the photons across all blocks on this process
    std::vector<int64_t> count_b(nblocal, 0);
    std::vector<Real> prob_b(nblocal);
    for (int nb=0; nb<nblocal; nb++)
      prob_b[nb] = tot_block[nb]/em_proc;
    my_blocks(0)->pran->SampleMultinomial(my_count,nblocal,prob_b.data(),count_b.data());
    Real ave_weight = em_tot/static_cast<Real>(ntot);


    for (int nb=0; nb<nblocal; nb++) {
      my_blocks(nb)->nphremain = count_b[nb];
      my_blocks(nb)->nphrun = 0;
      my_blocks(nb)->minweight = weightratio * ave_weight;
      my_blocks(nb)->emiss_to_weight = ave_weight;
      // distribute photons within each block
      my_blocks(nb)->ComputeEmissionSampleArray();
    }

  } else {
    // emission weights will just be proportional to emmisivity
    // Determine number of photons per block per step assuming each block is equal

    // First determine the active number of cells. A meshblock is inactive if it has
    // zero emission
    int nb_active = 0;
    for (int nb=0; nb<nblocal; nb++) {
      if (tot_block[nb] > 0.) {
        nb_active++;
      }
    }
    int active_proc[Globals::nranks];
#ifdef MPI_PARALLEL
    MPI_Gather(&nb_active,1,MPI_INT,active_proc,1,MPI_INT,0,MPI_COMM_WORLD);
#else
    active_proc[0] = nb_active;
#endif
    // Rank 0 compute the distribution of photons accross all processes and brodcasts
    // to all processes
    std::vector<int64_t> count(Globals::nranks, 0);
    int nb_active_total = 0;
    if (Globals::my_rank == 0) {
      std::vector<Real> prob(Globals::nranks);
      for (int irank=0; irank<Globals::nranks; irank++) {
        nb_active_total += active_proc[irank];
      }
      for (int irank=0; irank<Globals::nranks; irank++) {
        prob[irank] = static_cast<Real>(active_proc[irank])/static_cast<Real>(nb_active_total);
      }
      my_blocks(0)->pran->SampleMultinomial(ntot,Globals::nranks,prob.data(),count.data());
    }
    int64_t my_count;
#ifdef MPI_PARALLEL
    MPI_Scatter(count.data(),1,MPI_INT64_T,&my_count,1,MPI_INT64_T,0,MPI_COMM_WORLD);
    MPI_Bcast(&nb_active_total, 1, MPI_INT, 0, MPI_COMM_WORLD);
#else
    my_count = count[0];
#endif
    // Now distribute the photons across all active blocks on this process
    std::vector<int64_t> count_b(nblocal, 0);
    std::vector<Real> prob_b(nblocal);
    for (int nb=0; nb<nblocal; nb++)
      if (tot_block[nb] > 0.)
        prob_b[nb] = 1./static_cast<Real>(nb_active);
      else
        prob_b[nb] = 0.;
    my_blocks(0)->pran->SampleMultinomial(my_count,nblocal,prob_b.data(),count_b.data());
    // Acount for inactive blocks in weighting
    const int64_t block_size = ncells / pmy_mesh->nbtotal;
    const int64_t ncells_active = static_cast<int64_t>(nb_active_total) * block_size;
    for (int nb=0; nb<nblocal; nb++) {
      my_blocks(nb)->nphremain = count_b[nb];
      my_blocks(nb)->nphrun = 0;
      my_blocks(nb)->minweight = weightratio * em_max;
      // following assumes all cells on block emit. Correct for volumetric but needs
      // to be corrected for areal
      my_blocks(nb)->emiss_to_weight = static_cast<Real>(ncells_active)/static_cast<Real>(ntot);
    }
  }
  // Report emissivity ranges
  if (Globals::my_rank == 0) {
    std::cout << "Emission array range (min, max), total: " << em_min << " "
              << em_max << " " << em_tot << std::endl;
    std::cout << "Minimum weight: " << my_blocks(0)->minweight << std::endl;
  }
  delete[] tot_block;

}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::RunMonteCarlo(Outputs *pouts, Mesh *pmesh,
//!                                          ParameterInput *pinput)
//! \brief Finish Initialization of MonteCarloBlocks and run steady-state MC calculation

void MonteCarlo::RunMonteCarlo(Outputs *pouts, Mesh *pmesh,
                                     ParameterInput *pinput) {

  if (dynamic) {
    tmax = pinput->GetOrAddReal("montecarlo","tmax",-1.);
    if (tmax < 0.)
      tmax = pmy_mesh->dt;
    tint = pmy_mesh->dt;
    // convert to cgs units
    Real vel_cgs = pinput->GetOrAddReal("problem","vel_cgs",1.);
    Real l_cgs = pinput->GetOrAddReal("problem","l_cgs",1.);
    Real time_cgs = l_cgs/vel_cgs;
    tint *= time_cgs;
    tmax *= time_cgs;
  }

  // Transfer test knobs (see the LbTestMode declaration): rebuild every block in place
  // with no mesh call, or run the mesh's redistribution with unchanged costs.
  if (lb_test_repack == LBTEST_LOCAL) {
    RepackAllLocal(pinput);
  } else if (lb_test_repack == LBTEST_MESH) {
    pmy_mesh->RedistributeAndRefineMeshBlocks(pinput, pmy_mesh->nbtotal);
  }

  // A static run balances here, between transports, on what the previous one measured.
  if (!dynamic) BalanceStatic(pinput);

  // Update MC blocks if needed
  for (int nb=0; nb<nblocal; nb++) {
    MonteCarloBlock *pmcb = my_blocks(nb);
    // dynamic MC: (re)initialize the fluid-derived arrays for this cycle
    if (dynamic) SetupBlockFromFluid(pmcb);
    // Reset the photon boundary state for this transport, in both modes.
    pmcb->pphot->ClearBoundary();
    // reset counters
    pmcb->nscat = pmcb->nesc = pmcb->nabs = pmcb->ndes = pmcb-> nrem = 0;
  }

   // reset moments/sourcterms for start of new timestep
  if (dynamic) {
    for(int nb=0; nb<nblocal; ++nb) {
      MonteCarloBlock *pmcb = my_blocks(nb);
      if (pmcb->call_moments)
        pmcb->ResetMoments();
      if (pmcb->call_srcterms)
        pmcb->ResetSourceTerms();
    }
  }

  for (int etype=0; etype < ntype; etype++) {

      // reset counters
    for (int nb=0; nb<nblocal; nb++) {
      MonteCarloBlock *pmcb = my_blocks(nb);
      pmcb->nscat = pmcb->nesc = pmcb->nabs = pmcb->ndes = 0;
      pmcb->lb_time = 0.0;
      pmcb->lb_nstep = 0;
      pmcb->lb_pending = 0.0;
    }
    lb_rank_time = 0.0;
    lb_idle_time = lb_exchange_time = 0.0;
    lb_t_complete = lb_t_drain_in = lb_t_drain_arr = lb_t_local = lb_t_send = 0.0;
    lb_passes = 0;
    const std::chrono::steady_clock::time_point wall0 = std::chrono::steady_clock::now();

    // Distribute samples to all blocks based on emission properties
    // Sets nphremain and parameters for determining initial photon weights
    DistributeSamples(etype);
    emission_method = etype;

    // First transport of a static run with no measured or filed costs: balance on each
    // block's share of the photons about to be emitted, if asked to.  The shares travel
    // with their blocks, so this is safe before a photon exists.
    if (!dynamic && lb_initial_photons && pmy_mesh->ncycle == 0 && etype == 0
        && !lb_costs_loaded && lb_epoch == 0
        && (pmy_mesh->lb_automatic_ || pmy_mesh->lb_manual_)) {
      for (int nb=0; nb<nblocal; ++nb) {
        MonteCarloBlock *pmcb = my_blocks(nb);
        pmcb->pmy_block->cost_ = static_cast<double>(pmcb->nphremain);
      }
      BalanceNow(pinput);
    }

    // Run Monte Carlo until all photons have escaped/been absorbed
    const bool mid_transport_lb = !dynamic && lb_check_interval > 0
        && (pmy_mesh->lb_automatic_ || pmy_mesh->lb_manual_)
        && (Globals::nranks == 1 || (pexch != nullptr && pexch->Active()));
    int rounds_since_check = 0, rounds_since_lb = 0, nlb_transport = 0;
    if (async_term && pexch != nullptr && pexch->Active()) {
      TransportAsync(etype, pinput);
    } else {
      bool photons_remain = true; // True if photons on any process
      while(photons_remain) {
        // Keep transporting while photons are only moving between blocks on this rank.
        // Every pass of the outer loop costs a barrier on every rank, so a photon that
        // has to cross many blocks should not be buying one per crossing; see
        // ExchangeLocal.  Capped so that a pair of blocks passing a photon back and forth
        // cannot hold the other ranks at the barrier indefinitely.
        const bool local_loop = (pexch != nullptr && pexch->Active());
        if (local_loop) pexch->Reset();
        int sweeps = 0;
        bool local_progress = true;
        while (local_progress) {
          for(int nb=0; nb<nblocal; ++nb)
            TransportBlock(nb, etype);
          local_progress = ExchangeLocal() && local_loop
                           && (++sweeps < local_max_sweeps);
        }
        photons_remain = FinishRound();

        // Mid-transport balance point.  FinishRound has just completed every transfer
        // and flushed every receive buffer behind the same collective on every rank, so
        // each photon sits in exactly one block and the round counters agree everywhere.
        // Only with the rank exchange: the per-block protocol leaves receives posted
        // across rounds that rebuilding the blocks would orphan.
        ++rounds_since_check;
        ++rounds_since_lb;
        if (photons_remain && mid_transport_lb && rounds_since_check >= lb_check_interval
            && rounds_since_lb >= lb_min_window && nlb_transport < lb_max_per_transport) {
          rounds_since_check = 0;
          if (BalanceNow(pinput)) {
            ++nlb_transport;
            rounds_since_lb = 0;
          }
        }
      }
    }

    // Report diagnostic results from all blocks.  All six are reduced as MPI_INT64_T
    // below, so all six have to be 64 bits wide: reducing into a 32-bit ntot writes four
    // bytes past it.
    int64_t ntot = 0;
    int64_t nesc = 0, nabs = 0, ndes = 0, nscat = 0, nrem = 0;
    for(int nb=0; nb<nblocal; ++nb) {
      MonteCarloBlock *pmcb = my_blocks(nb);
      nesc += pmcb->nesc;
      nabs += pmcb->nabs;
      ndes += pmcb->ndes;
      nrem += pmcb->nrem;
      nscat += pmcb->nscat;
      ntot += pmcb->nphrun;
    }
    // Local count only: this runs before the reduction, so it is what this rank ran.
    pmcout->UpdateOutputCount(ntot);

  #ifdef MPI_PARALLEL
    MPI_Allreduce(MPI_IN_PLACE,&nesc,1,MPI_INT64_T,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&nabs,1,MPI_INT64_T,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&ndes,1,MPI_INT64_T,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&nscat,1,MPI_INT64_T,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&ntot,1,MPI_INT64_T,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&nrem,1,MPI_INT64_T,MPI_SUM,MPI_COMM_WORLD);
  #endif
    if (Globals::my_rank == 0) {
      std::cout  << "ntot: " << ntot
                << " nesc: " << nesc
                << " nabs: " << nabs
                << " ndes: " << ndes;
      if (nrem > 0)
        std::cout << " nrem: " << nrem;    
      if (ntot > 0)
        std::cout << " nscat/ntot: "
                  << static_cast<Real>(nscat)/static_cast<Real>(ntot) << std::endl;
      else
          std::cout << std::endl;
    }

    lb_transport_wall = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall0).count();
    if (lb_report) ReportLoadBalance(etype);
    if (!dynamic && !lb_cost_file.empty()) WriteCostFile();

    for(int nb=0; nb<nblocal; ++nb) {
      my_blocks(nb)->UserWorkAfterTransfer(etype);
    }
  } // end loop over ntype

  // A dynamic run is balanced by the mesh after this returns (main.cpp), on the hydro
  // time plus what TransportBlock added.  The forced-cost test writes its costs now so
  // that call sees them.  Otherwise apply the same prediction guard as the static
  // entry point: when the partition the mesh would choose is no better, hold its
  // cycle counter one short of the interval, so that its check waits another cycle.
  if (dynamic && (pmy_mesh->lb_automatic_ || pmy_mesh->lb_manual_)) {
    if (lb_test_costs != LBCOST_MEASURED) {
      AssignTestCosts();
      if (pmy_mesh->lb_manual_) pmy_mesh->lb_flag_ = true;
    } else if (pmy_mesh->step_since_lb + 1 >= pmy_mesh->lb_interval_
               && !RedistributionWorthwhile()) {
      pmy_mesh->step_since_lb = pmy_mesh->lb_interval_ - 2;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::BalanceStatic(ParameterInput *pin)
//! \brief the static run's balance point
//!
//! Between two transports nothing is in flight and every block's cost for the last
//! transport sits in its MeshBlock (TransportBlock), so this simply hands control to the
//! mesh's load balancer after measuring costs.

void MonteCarlo::BalanceStatic(ParameterInput *pin) {
  Mesh *pm = pmy_mesh;
  if (!(pm->lb_automatic_ || pm->lb_manual_)) return;
  if (pm->ncycle == 0) {
    // Nothing measured yet, unless costs were read from a file: then balance on those
    // now, before the first transport.
    if (!lb_costs_loaded) return;
    lb_costs_loaded = false;
  }
  BalanceNow(pin);
}

//----------------------------------------------------------------------------------------
//! \fn bool MonteCarlo::BalanceNow(ParameterInput *pin)
//! \brief test costs if requested, predict, and if worthwhile let the mesh redistribute

bool MonteCarlo::BalanceNow(ParameterInput *pin) {
  Mesh *pm = pmy_mesh;
  if (lb_test_costs != LBCOST_MEASURED) AssignTestCosts();
  if (!RedistributionWorthwhile()) {
    DecayCosts();
    return false;
  }
  // The mesh counts cycles since its last balance; inside a transport, or right after
  // reading a cost file, tell it the interval has passed.
  pm->step_since_lb = pm->lb_interval_;
  // manual costs are taken up only when the flag says they changed
  if (pm->lb_manual_) pm->lb_flag_ = true;
  const int epoch = lb_epoch;
  const double t0 = LoadBalanceClock();
  pm->LoadBalancingAndAdaptiveMeshRefinement(pin);
  const bool moved = (lb_epoch != epoch);
  if (moved && Globals::my_rank == 0 && verbose) {
    const std::ios::fmtflags flags = std::cout.flags();
    const std::streamsize prec = std::cout.precision();
    std::cout.unsetf(std::ios::floatfield);
    std::cout.precision(3);
    std::cout << "Monte Carlo load balance: redistribution " << lb_epoch << " took "
              << LoadBalanceSeconds(LoadBalanceClock() - t0) << " s on rank 0" << std::endl;
    std::cout.flags(flags);
    std::cout.precision(prec);
  }
  if (!moved) DecayCosts();
  return moved;
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::Partition(...)
//! \brief the partition the mesh will use; see the declaration

void MonteCarlo::Partition(double *cost, int nb, int *rlist, int *slist,
                           int *nlist) const {
  if (lb_partition_optimal) {
    OptimalContiguousPartition(cost, nb, Globals::nranks, rlist, slist, nlist);
  } else {
    pmy_mesh->CalculateLoadBalance(cost, rlist, slist, nlist, nb);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::DecayCosts()
//! \brief scale the accumulated block costs after a check that kept the layout

void MonteCarlo::DecayCosts() {
  if (lb_cost_decay >= 1.0) return;
  for (int nb=0; nb<nblocal; ++nb) my_blocks(nb)->pmy_block->cost_ *= lb_cost_decay;
}

//----------------------------------------------------------------------------------------
//! \fn bool MonteCarlo::RedistributionWorthwhile() const
//! \brief predict before committing
//!
//! The mesh's check asks only whether the current layout is imbalanced; its greedy
//! contiguous partition of the same costs can be worse than what it replaces when there
//! are few blocks per rank, and a partition identical to the current one still rebuilds
//! every block.  So run the partitioner on a copy and approve only if the busiest rank
//! improves by lb_min_gain.

bool MonteCarlo::RedistributionWorthwhile() {
  std::vector<double> cost;
  GatherBalancerCosts(cost);
  return WorthwhileFromCosts(cost);
}

bool MonteCarlo::WorthwhileFromCosts(const std::vector<double> &cost) const {
  Mesh *pm = pmy_mesh;
  const int nr = Globals::nranks;
  double cur_max = 0.0;
  for (int r=0; r<nr; ++r) {
    double rc = 0.0;
    for (int n=pm->nslist[r]; n<pm->nslist[r] + pm->nblist[r]; ++n) rc += cost[n];
    cur_max = std::max(cur_max, rc);
  }
  std::vector<double> ccopy(cost);
  std::vector<int> rank_new(pm->nbtotal), ns_new(nr), nb_new(nr);
  Partition(ccopy.data(), pm->nbtotal, rank_new.data(), ns_new.data(), nb_new.data());
  double new_max = 0.0;
  for (int r=0; r<nr; ++r) {
    double rc = 0.0;
    for (int n=ns_new[r]; n<ns_new[r] + nb_new[r]; ++n) rc += cost[n];
    new_max = std::max(new_max, rc);
  }
  if (new_max <= (1.0 - lb_min_gain)*cur_max) return true;
  if (Globals::my_rank == 0 && verbose) {
    const std::ios::fmtflags flags = std::cout.flags();
    const std::streamsize prec = std::cout.precision();
    std::cout.unsetf(std::ios::floatfield);
    std::cout.precision(3);
    std::cout << "Monte Carlo load balance: layout kept; the partition the balancer"
              << " would choose has busiest rank " << LoadBalanceSeconds(new_max)
              << " s against " << LoadBalanceSeconds(cur_max) << " s now ("
              << (cur_max > 0 ? new_max/cur_max : 1.0) << ", threshold "
              << 1.0 - lb_min_gain << ")" << std::endl;
    std::cout.flags(flags);
    std::cout.precision(prec);
  }
  return false;
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::GatherBalancerCosts(std::vector<double> &cost) const
//! \brief Every block's cost as Mesh::UpdateCostList is about to compute it 
//!
//! The aged list plus the MeshBlock's accumulated time under automatic, the MeshBlock's
//! value under manual.  Gathered the way the mesh gathers its cost list.

void MonteCarlo::FoldPendingCosts() {
  for (int nb=0; nb<nblocal; ++nb) {
    MonteCarloBlock *pmcb = my_blocks(nb);
    if (pmcb->lb_pending == 0.0) continue;
    pmcb->lb_time += pmcb->lb_pending;
    lb_rank_time += pmcb->lb_pending;
    pmcb->pmy_block->cost_ += pmcb->lb_pending;
    pmcb->lb_pending = 0.0;
  }
}

void MonteCarlo::FillLocalBalancerCosts(std::vector<double> &cost) {
  Mesh *pm = pmy_mesh;
  FoldPendingCosts();
  cost.assign(pm->nbtotal, 0.0);
  const double w = pm->lb_automatic_
      ? static_cast<double>(pm->lb_interval_ - 1)/static_cast<double>(pm->lb_interval_)
      : 0.0;
  for (int nb=0; nb<nblocal; ++nb) {
    MeshBlock *pmb = my_blocks(nb)->pmy_block;
    cost[pmb->gid] = pm->lb_automatic_ ? pm->costlist[pmb->gid]*w + pmb->cost_
                                       : pmb->cost_;
  }
}

void MonteCarlo::GatherBalancerCosts(std::vector<double> &cost) {
  FillLocalBalancerCosts(cost);
#ifdef MPI_PARALLEL
  Mesh *pm = pmy_mesh;
  MPI_Allgatherv(MPI_IN_PLACE, pm->nblist[Globals::my_rank], MPI_DOUBLE, cost.data(),
                 pm->nblist, pm->nslist, MPI_DOUBLE, MPI_COMM_WORLD);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::WriteCostFile() const
//! \brief Gather every block's MeshBlock cost and write to cost_file

void MonteCarlo::WriteCostFile() {
  Mesh *pm = pmy_mesh;
  FoldPendingCosts();
  std::vector<double> cost(pm->nbtotal, 0.0);
  for (int nb=0; nb<nblocal; ++nb) {
    MeshBlock *pmb = my_blocks(nb)->pmy_block;
    cost[pmb->gid] = pmb->cost_;
  }
#ifdef MPI_PARALLEL
  MPI_Allgatherv(MPI_IN_PLACE, pm->nblist[Globals::my_rank], MPI_DOUBLE, cost.data(),
                 pm->nblist, pm->nslist, MPI_DOUBLE, MPI_COMM_WORLD);
#endif
  if (Globals::my_rank != 0) return;
  std::ofstream f(lb_cost_file.c_str());
  if (!f) {
    std::cout << "### Warning in MonteCarlo::WriteCostFile: could not open "
              << lb_cost_file << std::endl;
    return;
  }
  f.precision(17);
  f << pm->nbtotal << std::endl;
  for (int n=0; n<pm->nbtotal; ++n) f << n << " " << cost[n] << std::endl;
}

//----------------------------------------------------------------------------------------
//! \fn bool MonteCarlo::ReadCostFile()
//! \brief load cost_file into the local MeshBlocks' costs, return true if used
//!
//! Every rank reads the whole file; it is one line per block.  A missing file is not an
//! error and a file for another block count is refused.

bool MonteCarlo::ReadCostFile() {
  Mesh *pm = pmy_mesh;
  std::ifstream f(lb_cost_file.c_str());
  if (!f) {
    if (Globals::my_rank == 0 && verbose)
      std::cout << "Monte Carlo: no cost file " << lb_cost_file
                << " yet; the first transport runs on the initial layout" << std::endl;
    return false;
  }
  int nb_file = -1;
  f >> nb_file;
  if (nb_file != pm->nbtotal) {
    std::stringstream msg;
    msg << "### FATAL ERROR in MonteCarlo::ReadCostFile" << std::endl
        << lb_cost_file << " describes " << nb_file << " blocks; this mesh has "
        << pm->nbtotal << std::endl;
    ATHENA_ERROR(msg);
  }
  std::vector<double> cost(pm->nbtotal, 0.0);
  for (int n=0; n<pm->nbtotal; ++n) {
    int gid;
    double c;
    if (!(f >> gid >> c) || gid < 0 || gid >= pm->nbtotal) {
      std::stringstream msg;
      msg << "### FATAL ERROR in MonteCarlo::ReadCostFile" << std::endl
          << lb_cost_file << " is malformed at line " << n + 2 << std::endl;
      ATHENA_ERROR(msg);
    }
    cost[gid] = c;
  }
  for (int nb=0; nb<nblocal; ++nb) {
    MeshBlock *pmb = my_blocks(nb)->pmy_block;
    pmb->cost_ = cost[pmb->gid];
  }
  if (Globals::my_rank == 0 && verbose)
    std::cout << "Monte Carlo: block costs read from " << lb_cost_file << std::endl;
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::AssignTestCosts()
//! \brief synthetic block costs for the forced-move test; see LbCostMode

void MonteCarlo::AssignTestCosts() {
  const int half = pmy_mesh->nbtotal/2;
  const bool first_half_heavy = (lb_epoch % 2 == 0);
  for (int nb=0; nb<nblocal; ++nb) {
    MeshBlock *pmb = my_blocks(nb)->pmy_block;
    const bool first_half = (pmb->gid < half);
    pmb->cost_ = (first_half == first_half_heavy) ? 3.0 : 1.0;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::SetupBlockFromFluid(MonteCarloBlock *pmcb)
//! \brief the block's fluid-derived arrays from its MeshBlock's primitives

void MonteCarlo::SetupBlockFromFluid(MonteCarloBlock *pmcb) {
  pmcb->GetDensity();
  pmcb->GetTemperature();
  pmcb->GetNumberDensity();
  pmcb->ComputeFreeFreePrefactor();
  if (boosts) {
    pmcb->GetVelocity();
    pmcb->ComputeTransformations();
  } else if (GENERAL_RELATIVITY) {
    // no fluid velocity, but the GR tetrad still needs a frame to be built on
    pmcb->SetNormalObserver();
    pmcb->ComputeTransformations();
  }
  if (NSCALARS > 0) pmcb->GetScalars();
  if (using_bfield) pmcb->GetBField();
}

//----------------------------------------------------------------------------------------
//! \fn int MonteCarlo::ComputeLoopMax() const
//! \brief per-block resident photon cap: loop_max_size, cut so that nblocal blocks stay
//!        within max_resident_photons.  Recomputed whenever nblocal changes.

int MonteCarlo::ComputeLoopMax() const {
  int loop_max = per_block_cap_;
  if (photon_budget_ > 0 && nblocal > 0)
    loop_max = std::min(per_block_cap_, std::max(1, photon_budget_/nblocal));
  return loop_max;
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::PackDeparting(...)
//! \brief first redistribution hook: record the new layout and pack what leaves
//!
//! Called by Mesh::RedistributeAndRefineMeshBlocks after it has computed the new rank of
//! every block (so nslist and nblist already describe the new layout) and before it
//! deletes the MeshBlocks that leave this rank.  ranklist is still the old one here, and
//! that is what the second hook needs to know where each arrival comes from, so it is
//! read now.  Blocks that stay are kept as they are; the mesh reuses their MeshBlock
//! objects too.  Departing payloads are sent on the module's own communicator, so
//! nothing here can be confused with the mesh's own transfer.

void MonteCarlo::PackDeparting(const int *newrank, const int *newtoold,
                               const int *oldtonew, int ntot_new) {
  Mesh *pm = pmy_mesh;
  std::stringstream msg;
  if (ntot_new != pm->nbtotal) {
    msg << "### FATAL ERROR in MonteCarlo::PackDeparting" << std::endl
        << "the block tree changed (" << pm->nbtotal << " -> " << ntot_new << " blocks);"
        << " the Monte Carlo module follows redistribution, not refinement" << std::endl;
    ATHENA_ERROR(msg);
  }
  const int me = Globals::my_rank;
  const int nbs = pm->nslist[me];
  const int nnew = pm->nblist[me];
  const bool repack_all = (lb_test_repack == LBTEST_MESH);

  lb_kept_.assign(nnew, nullptr);
  lb_src_.assign(nnew, -1);
  lb_local_.clear();
  if (repack_all) lb_local_.resize(nnew);
  lb_departed_.clear();
  lb_send_.clear();
  lb_send_.reserve(nblocal);   // no reallocation while sends point into it
#ifdef MPI_PARALLEL
  lb_req_.clear();
#endif

  // where each of this rank's new blocks comes from
  for (int n=0; n<nnew; ++n) {
    const int on = newtoold[nbs + n];
    if (pm->ranklist[on] != me) lb_src_[n] = pm->ranklist[on];
  }

  // what each of this rank's current blocks does
  for (int nb=0; nb<nblocal; ++nb) {
    MonteCarloBlock *pmcb = my_blocks(nb);
    const int on = pmcb->pmy_block->gid;
    const int nn = oldtonew[on];
    const int dest = newrank[nn];
    const int lid_new = nn - pm->nslist[dest];
    if (dest == me && !repack_all) {
      lb_kept_[lid_new] = pmcb;
      continue;
    }
    lb_departed_.push_back(pmcb);
    if (dest == me) {   // test mode: travels through memory
      BlockPayload &p = lb_local_[lid_new];
      pmcb->PackForTransfer(p.ib, p.rb, p.sb);
      continue;
    }
#ifdef MPI_PARALLEL
    if (3*lid_new + 2 > 32767) {
      msg << "### FATAL ERROR in MonteCarlo::PackDeparting" << std::endl
          << "destination lid " << lid_new << " does not fit the MPI tag range" << std::endl;
      ATHENA_ERROR(msg);
    }
    lb_send_.emplace_back();
    BlockPayload &p = lb_send_.back();
    pmcb->PackForTransfer(p.ib, p.rb, p.sb);
    MPI_Request r;
    MPI_Isend(p.ib.data(), static_cast<int>(p.ib.size()), MPI_INT, dest, 3*lid_new,
              lb_comm_, &r);
    lb_req_.push_back(r);
    MPI_Isend(p.rb.data(), static_cast<int>(p.rb.size()), MPI_ATHENA_REAL, dest,
              3*lid_new + 1, lb_comm_, &r);
    lb_req_.push_back(r);
    MPI_Isend(p.sb.data(), static_cast<int>(p.sb.size()), MPI_CHAR, dest, 3*lid_new + 2,
              lb_comm_, &r);
    lb_req_.push_back(r);
#endif
  }
}

//----------------------------------------------------------------------------------------
//! \fn MonteCarloBlock *MonteCarlo::RebuildArrival(...)
//! \brief a block that has just landed on this rank: construct it on its MeshBlock,
//!        derive its fluid arrays, run the problem generator, then restore its payload

MonteCarloBlock *MonteCarlo::RebuildArrival(MeshBlock *pmb, ParameterInput *pin,
                                            const BlockPayload &payload) {
  MonteCarloBlock *pmcb = new MonteCarloBlock(pmb, nullptr, this, pin);
  SetupBlockFromFluid(pmcb);
  pmcb->nscat = pmcb->nesc = pmcb->nabs = pmcb->ndes = pmcb->nrem = 0;
  pmcb->loop_max_size = ComputeLoopMax();
  // The problem generator runs before the payload is unpacked, exactly as at startup:
  // it supplies per-block state that is neither fluid-derived nor carried (photon
  // budgets of emission = none decks, image geometry, per-block tables), and whatever
  // it sets that the payload also carries is then overwritten by the block's real
  // state.  The contract for a generator is therefore that this hook is repeatable for
  // a block and touches no state shared across blocks or indexed by lid; one that does
  // must rebuild it in UserWorkAfterRebalance or refuse balancing, as the readers with
  // per-lid opacity tables do.
  pmcb->MonteCarloProblemGenerator(pin);
  pmcb->UnpackFromTransfer(payload.ib, payload.rb, payload.sb);
  return pmcb;
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::RebuildAfterRedistribution(ParameterInput *pin)
//! \brief second redistribution hook: rebuild the block list on the mesh's new one
//!
//! Called after Initialize(2), so every new MeshBlock has its conserved variables, its
//! ghost zones and its primitives.  Kept blocks are reattached in their new lid order;
//! arrivals are built and unpacked; departed blocks are deleted (their MeshBlocks are
//! already gone, and nothing here touches them); every block is relinked to its new
//! neighbors and the rank exchange rebuilt.

void MonteCarlo::RebuildAfterRedistribution(ParameterInput *pin) {
  Mesh *pm = pmy_mesh;
  const int nnew = pm->nblocal;
  std::stringstream msg;
  if (static_cast<int>(lb_kept_.size()) != nnew) {
    msg << "### FATAL ERROR in MonteCarlo::RebuildAfterRedistribution" << std::endl
        << "called without PackDeparting, or the layout changed in between" << std::endl;
    ATHENA_ERROR(msg);
  }

  // Departed blocks first: their state is already in the send payloads and their
  // MeshBlocks are gone, and freeing them before the arrivals are built keeps the peak
  // memory of a large move near the payload size rather than twice the block size.
  for (std::size_t i=0; i<lb_departed_.size(); ++i) delete lb_departed_[i];
  lb_departed_.clear();

  AthenaArray<MonteCarloBlock*> newlist;
  newlist.NewAthenaArray(nnew);
  int64_t narrive = 0;
  for (int lid=0; lid<nnew; ++lid) {
    MeshBlock *pmb = pm->my_blocks(lid);
    if (lb_kept_[lid] != nullptr) {
      pmb->pmy_mcb = lb_kept_[lid];
      newlist(lid) = lb_kept_[lid];
      continue;
    }
    BlockPayload recv;
    const BlockPayload *pp = nullptr;
    if (lb_src_[lid] < 0) {
      pp = &lb_local_.at(lid);
    } else {
#ifdef MPI_PARALLEL
      const int src = lb_src_[lid];
      MPI_Status st;
      int n;
      MPI_Probe(src, 3*lid, lb_comm_, &st);
      MPI_Get_count(&st, MPI_INT, &n);
      recv.ib.resize(n);
      MPI_Recv(recv.ib.data(), n, MPI_INT, src, 3*lid, lb_comm_, MPI_STATUS_IGNORE);
      MPI_Probe(src, 3*lid + 1, lb_comm_, &st);
      MPI_Get_count(&st, MPI_ATHENA_REAL, &n);
      recv.rb.resize(n);
      MPI_Recv(recv.rb.data(), n, MPI_ATHENA_REAL, src, 3*lid + 1, lb_comm_,
               MPI_STATUS_IGNORE);
      MPI_Probe(src, 3*lid + 2, lb_comm_, &st);
      MPI_Get_count(&st, MPI_CHAR, &n);
      recv.sb.resize(n);
      MPI_Recv(recv.sb.data(), n, MPI_CHAR, src, 3*lid + 2, lb_comm_, MPI_STATUS_IGNORE);
#endif
      pp = &recv;
    }
    newlist(lid) = RebuildArrival(pmb, pin, *pp);
    ++narrive;
  }

#ifdef MPI_PARALLEL
  if (!lb_req_.empty())
    MPI_Waitall(static_cast<int>(lb_req_.size()), lb_req_.data(), MPI_STATUSES_IGNORE);
  lb_req_.clear();
#endif
  lb_send_.clear();
  lb_local_.clear();
  lb_kept_.clear();
  lb_src_.clear();

  my_blocks.ExchangeAthenaArray(newlist);
  nblocal = nnew;
  for (int nb=0; nb<nblocal; ++nb) my_blocks(nb)->loop_max_size = ComputeLoopMax();
  RelinkAll();
  UserWorkAfterRebalance(pin);
  ++lb_epoch;

#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &narrive, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (Globals::my_rank == 0 && verbose) {
    std::cout << "Monte Carlo blocks rebuilt after redistribution " << lb_epoch << ": "
              << narrive << " block(s) arrived on a new rank" << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::RepackAllLocal(ParameterInput *pin)
//! \brief test mode: pack, destroy and rebuild every block in place, no mesh call

void MonteCarlo::RepackAllLocal(ParameterInput *pin) {
  for (int nb=0; nb<nblocal; ++nb) {
    MonteCarloBlock *old = my_blocks(nb);
    MeshBlock *pmb = old->pmy_block;
    BlockPayload p;
    old->PackForTransfer(p.ib, p.rb, p.sb);
    my_blocks(nb) = RebuildArrival(pmb, pin, p);
    delete old;
  }
  RelinkAll();
  UserWorkAfterRebalance(pin);
  ++lb_epoch;
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::RelinkAll()
//! \brief refresh every block's neighbor links, boundary state and the peer list, which
//!        all depend on which rank and lid each neighbor now has

void MonteCarlo::RelinkAll() {
  Mesh *pm = pmy_mesh;
  for (int nb=0; nb<nblocal; ++nb) {
    MonteCarloBlock *pmcb = my_blocks(nb);
    MeshBlock *pmb = pmcb->pmy_block;
    Photon *pp = pmcb->pphot;
    pp->ClearNeighbors();
    const int nrbx1 = pm->mesh_size.nx1/pmb->block_size.nx1;
    const int nrbx2 = pm->mesh_size.nx2/pmb->block_size.nx2;
    const int nrbx3 = pm->mesh_size.nx3/pmb->block_size.nx3;
    pp->LinkNeighbors(pm->tree, nrbx1, nrbx2, nrbx3, pm->root_level);
    pp->SetOffRankNeighborFlag();
    pp->ClearBoundary();
  }
  if (pexch != nullptr) {
    pexch->BuildPeerList();
    pexch->Reset();
    pexch->ResetCounters();
  }
  send_list_.clear();
  recv_list_.clear();
}

//----------------------------------------------------------------------------------------
//! \fn double MonteCarlo::LoadBalanceClock()
//! \brief the clock MeshBlock::StartTimeMeasurement reads, so transport and hydro costs
//!        can be added in a dynamic run: omp_get_wtime with OpenMP, clock() otherwise

double MonteCarlo::LoadBalanceClock() {
#ifdef OPENMP_PARALLEL
  return omp_get_wtime();
#else
  return static_cast<double>(clock());
#endif
}

double MonteCarlo::LoadBalanceSeconds(double clock_units) {
#ifdef OPENMP_PARALLEL
  return clock_units;
#else
  return clock_units/static_cast<double>(CLOCKS_PER_SEC);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::TransportBlock(int nb, int etype)
//! \brief one block's sweep, timed into the block's window cost and its MeshBlock cost

void MonteCarlo::TransportBlock(int nb, int etype) {
  MonteCarloBlock *pmcb = my_blocks(nb);
  const double t0 = LoadBalanceClock();
  if (raytrace_flag)
    pmcb->RayTracePhotonsOnBlock(etype);
  else
    pmcb->TransferPhotonsOnBlock(etype);
  const double dt = LoadBalanceClock() - t0 + pmcb->lb_pending;
  pmcb->lb_pending = 0.0;
  pmcb->lb_time += dt;
  lb_rank_time += dt;
  pmcb->pmy_block->cost_ += dt;
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::ReportLoadBalance(int etype)
//! \brief print how the transport cost of this emission type was spread over ranks and
//!        blocks: the imbalance a balancer could remove, and the granularity ceiling it
//!        cannot
//!
//! The per-block costs are gathered into a gid-indexed list the same way the mesh
//! gathers its own cost list (blocks of a rank are contiguous in gid, so every rank's
//! contribution sits at its nslist offset).  Two ratios matter: the busiest rank over the
//! mean, which is what balancing can fix, and the costliest single block over a rank's
//! fair share, which it cannot -- a block is never split, so no partition beats that.

void MonteCarlo::ReportLoadBalance(int etype) {
  FoldPendingCosts();
  const int nbtot = pmy_mesh->nbtotal;
  const int nr = Globals::nranks;
  std::vector<double> cost(nbtot, 0.0);
  std::vector<int64_t> nstep(nbtot, 0), nscat_b(nbtot, 0), nemit(nbtot, 0);
  for (int nb=0; nb<nblocal; ++nb) {
    MonteCarloBlock *pmcb = my_blocks(nb);
    const int gid = pmcb->pmy_block->gid;
    cost[gid] = LoadBalanceSeconds(pmcb->lb_time);
    nstep[gid] = pmcb->lb_nstep;
    nscat_b[gid] = pmcb->nscat;
    nemit[gid] = pmcb->nphrun;
  }
#ifdef MPI_PARALLEL
  int *nblist = pmy_mesh->nblist, *nslist = pmy_mesh->nslist;
  MPI_Allgatherv(MPI_IN_PLACE, nblist[Globals::my_rank], MPI_DOUBLE, cost.data(),
                 nblist, nslist, MPI_DOUBLE, MPI_COMM_WORLD);
  MPI_Allgatherv(MPI_IN_PLACE, nblist[Globals::my_rank], MPI_INT64_T, nstep.data(),
                 nblist, nslist, MPI_INT64_T, MPI_COMM_WORLD);
  MPI_Allgatherv(MPI_IN_PLACE, nblist[Globals::my_rank], MPI_INT64_T, nscat_b.data(),
                 nblist, nslist, MPI_INT64_T, MPI_COMM_WORLD);
  MPI_Allgatherv(MPI_IN_PLACE, nblist[Globals::my_rank], MPI_INT64_T, nemit.data(),
                 nblist, nslist, MPI_INT64_T, MPI_COMM_WORLD);
#endif
  // time each rank used before move
  std::vector<double> rank_time(nr, LoadBalanceSeconds(lb_rank_time));
  std::vector<double> idle(nr, lb_idle_time), exch(nr, lb_exchange_time);
  const double split_local[5] = {lb_t_complete, lb_t_drain_in, lb_t_drain_arr, lb_t_local,
                                 lb_t_send};
  std::vector<double> split(5*nr, 0.0);
  for (int k=0; k<5; ++k) split[5*Globals::my_rank + k] = split_local[k];
#ifdef MPI_PARALLEL
  MPI_Allgather(MPI_IN_PLACE, 1, MPI_DOUBLE, rank_time.data(), 1, MPI_DOUBLE,
                MPI_COMM_WORLD);
  MPI_Allgather(MPI_IN_PLACE, 1, MPI_DOUBLE, idle.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD);
  MPI_Allgather(MPI_IN_PLACE, 1, MPI_DOUBLE, exch.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD);
  MPI_Allgather(MPI_IN_PLACE, 5, MPI_DOUBLE, split.data(), 5, MPI_DOUBLE, MPI_COMM_WORLD);
#endif
  if (Globals::my_rank != 0) return;

  double total = 0.0, rank_max = 0.0;
  int rank_arg = 0;
  int64_t steps = 0, scats = 0, emits = 0;
  for (int r=0; r<nr; ++r) {
    total += rank_time[r];
    if (rank_time[r] > rank_max) { rank_max = rank_time[r]; rank_arg = r; }
  }
  for (int n=0; n<nbtot; ++n) { steps += nstep[n]; scats += nscat_b[n]; emits += nemit[n]; }
  const double fair = total/nr;
  std::vector<int> order(nbtot);
  for (int n=0; n<nbtot; ++n) order[n] = n;
  std::partial_sort(order.begin(), order.begin() + std::min(5, nbtot), order.end(),
                    [&cost](int a, int b) { return cost[a] > cost[b]; });
  const int top = order[0];
  // main.cpp leaves cout in 17-digit scientific notation; three figures read better here
  const std::ios::fmtflags flags = std::cout.flags();
  const std::streamsize prec = std::cout.precision();
  std::cout.unsetf(std::ios::floatfield);
  std::cout.precision(3);
  std::cout << "Monte Carlo load balance, type " << etype << ": transport time "
            << total << " s over " << nr << " rank(s); busiest rank " << rank_arg
            << " = " << rank_max << " s = " << (fair > 0 ? rank_max/fair : 0.0)
            << " x fair share; transport wall " << lb_transport_wall << " s" << std::endl
            << "  costliest block gid " << top << " = " << cost[top] << " s = "
            << (fair > 0 ? cost[top]/fair : 0.0) << " x fair share (the ceiling no"
            << " partition beats); top " << std::min(5, nbtot) << " by gid:cost";
  for (int i=0; i<std::min(5, nbtot); ++i)
    std::cout << " " << order[i] << ":" << cost[order[i]];
  {
    double idle_min = idle[0], idle_max = idle[0], exch_min = exch[0], exch_max = exch[0];
    for (int r=1; r<nr; ++r) {
      idle_min = std::min(idle_min, idle[r]); idle_max = std::max(idle_max, idle[r]);
      exch_min = std::min(exch_min, exch[r]); exch_max = std::max(exch_max, exch[r]);
    }
    std::cout << std::endl
              << "  per rank: idle passes " << idle_min << " to " << idle_max
              << " s, exchange " << exch_min << " to " << exch_max << " s, "
              << lb_passes << " passes on rank 0";
    const char *names[5] = {"complete sends", "drain incoming", "flush arrivals",
                            "local hand-off", "post sends"};
    std::cout << std::endl << "  exchange split, min to max over ranks:";
    for (int k=0; k<5; ++k) {
      double mn = split[k], mx = split[k];
      for (int r=1; r<nr; ++r) {
        mn = std::min(mn, split[5*r + k]); mx = std::max(mx, split[5*r + k]);
      }
      std::cout << " " << names[k] << " " << mn << "-" << mx << " s;";
    }
  }
  std::cout << std::endl
            << "  steps " << steps << ", scatterings " << scats << ", emitted " << emits;
  if (steps > 0) std::cout << ", " << 1.0e6*total/static_cast<double>(steps) << " us/step";
  if (emits > 0) std::cout << ", " << 1.0e6*total/static_cast<double>(emits) << " us/photon";
  std::cout << std::endl;
  std::cout.flags(flags);
  std::cout.precision(prec);
}

//----------------------------------------------------------------------------------------
//! \fn bool MonteCarlo::CheckAndBroadCastPhotonsRemaining() 
//! \brief Checks if photons remain on any process and sends message to rank 0
//!        process if none remaining.

bool MonteCarlo::CheckAndBroadCastPhotonsRemaining() {
  if (pexch != nullptr && pexch->Active()) pexch->Reset();
  ExchangeLocal();
  return FinishRound();
}

//----------------------------------------------------------------------------------------
//! \fn bool MonteCarlo::ExchangeLocal()
//! \brief hand photons to blocks on this rank, and set the rest aside for later
//!
//! Returns true when something landed here, meaning there is more transport to do without
//! talking to anyone else.  Photons bound elsewhere are staged in MCRankExchange and go
//! out once, in FinishRound.
//!
//! Separating this from the global step is what lets a rank keep working instead of
//! taking a barrier for every block a photon crosses.  A nearly horizontal photon in a
//! domain with periodic sides crosses blocks essentially without end, and measured on a
//! 32-rank test 99.6% of all rounds were transporting a single such photon while every
//! rank went through the full round for it.
//!
//! Only blocks that can take part are swept.  One with no photons and nothing delivered
//! cannot send, receive or dirty any boundary state.  Sweeping all of them regardless
//! costs O(nblocal * nneighbor) whatever the photon count, which is what made a mesh of
//! many small blocks slow: cost grew as nblocal^1.3 at fixed photon number.

bool MonteCarlo::ExchangeLocal() {
  const bool rank_exchange = (pexch != nullptr && pexch->Active());

  // A block can only hand photons over if it holds some.  Without the rank exchange one
  // with an off-rank neighbor takes part regardless, because the receiving rank is
  // waiting on a per-link count message from it whether or not there is anything to
  // report; with the exchange running, silence is the signal and it can be skipped.
  send_list_.clear();
  for (int nb=0; nb<nblocal; ++nb) {
    Photon *pp = my_blocks(nb)->pphot;
    if (pp->nphot > 0 || (!rank_exchange && pp->has_offrank_neighbor_))
      send_list_.push_back(nb);
  }
  for (std::size_t n=0; n<send_list_.size(); ++n)
    my_blocks(send_list_[n])->pphot->SendToNeighbors();

  return DrainArrivals();
}

//----------------------------------------------------------------------------------------
//! \fn bool MonteCarlo::DrainArrivals()
//! \brief flush whatever is sitting in receive buffers into the blocks that own them

bool MonteCarlo::DrainArrivals() {
  const bool rank_exchange = (pexch != nullptr && pexch->Active());

  recv_list_.clear();
  for (int nb=0; nb<nblocal; ++nb) {
    Photon *pp = my_blocks(nb)->pphot;
    if (pp->has_incoming_ || (!rank_exchange && pp->has_offrank_neighbor_))
      recv_list_.push_back(nb);
  }
  const bool landed = !recv_list_.empty();

  // Blocks that report themselves finished drop out rather than being rescanned on every
  // later pass.  With the rank exchange the first pass finishes all of them: a same-rank
  // hand-off already happened during the send sweep, and anything from another rank was
  // delivered before this was called, so there is nothing to poll for.
  std::vector<int> pending(recv_list_);
  while (!pending.empty()) {
    std::size_t keep = 0;
    for (std::size_t n=0; n<pending.size(); ++n) {
      if (!my_blocks(pending[n])->pphot->ReceiveFromNeighbors())
        pending[keep++] = pending[n];
    }
    pending.resize(keep);
  }

  // Only the receivers need clearing: a same-rank send writes into the target's buffer
  // and leaves the sender's own boundary state untouched.
  for (std::size_t n=0; n<recv_list_.size(); ++n)
    my_blocks(recv_list_[n])->pphot->ClearBoundary();

  return landed;
}

//----------------------------------------------------------------------------------------
//! \fn bool MonteCarlo::FinishRound()
//! \brief send what was staged for other ranks, take delivery, and test for completion

bool MonteCarlo::FinishRound() {
  // Collective over the peer set, so it runs on every rank whether or not this one had
  // anything to contribute.
  if (pexch != nullptr && pexch->Active()) {
    pexch->ExchangeAndDeliver();
    DrainArrivals();
  }

  // Check if photons have completed.  nremain is a sum of 64-bit per-block counts and
  // can exceed an int on a rank that still has most of a large emission to do.
  int64_t nremain=0,nprop=0;
  for(int nb=0; nb<nblocal; ++nb){
    MonteCarloBlock *pmcb = my_blocks(nb);
    nremain += pmcb->nphremain;
    nprop += pmcb->pphot->nphot;
  }
#ifdef MPI_PARALLEL
  // One reduction, blocking over all ranks
  int64_t counts[2] = {nprop, nremain};
  MPI_Allreduce(MPI_IN_PLACE, counts, 2, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
  nprop = counts[0];
  nremain = counts[1];
#endif

  bool active;
  if ((nremain > 0) || (nprop > 0)) {
    active = true;
  } else {
    active = false;
  }
  return active;
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::TransportAsync(int etype)
//! \brief transport to completion, deciding the end from photon counters, not a barrier
//!
//! FinishRound settles every round with a blocking Allreduce, so the whole run advances
//! in lockstep at the pace of whichever rank still has a photon.
//!
//! So no rank waits on any other here.  A rank transports what it holds, posts what is
//! leaving without waiting for it to be taken, and picks up whatever has arrived.  The
//! only question left is when everyone is finished, and that is answered from two running
//! counters: photons handed to other ranks, and photons taken from them.  Their global
//! sums differ by exactly the number of photons in flight, so the mesh is done when no
//! rank holds a photon and the two sums agree.
//!
//! That test is taken with a nonblocking reduction, and a single one of them is not
//! enough to trust.  Its inputs are read at different instants on different ranks, so a
//! photon can be counted as received before the rank that sent it counted it as sent, and
//! a lone reduction can report a quiet mesh that is not quiet.  Two consecutive ones are
//! enough: this rank posts the second only after the first has completed, so if both
//! report no photons anywhere and the identical pair of totals, then nothing was sent
//! anywhere between them and nothing was in flight.

void MonteCarlo::TransportAsync(int etype, ParameterInput *pin) {
#ifdef MPI_PARALLEL
  // The fourth entry carries the draining flag for the mid-transport balance below, the
  // fifth the photons finished so far, which is that balance's clock.
  int64_t sbuf[5] = {0, 0, 0, 0, 0}, rbuf[5] = {0, 0, 0, 0, 0};
  int64_t prev_sent = -1, prev_recv = -1;
  int quiet_streak = 0;
  bool pending = false;
  MPI_Request treq = MPI_REQUEST_NULL;

  // Without rounds, the sequence of completed reductions is assesses the work done. If
  // lb_check_fraction of the transport's photons have finished since the last check, a
  // nonblocking gather of the block costs is started on the next completion.  Each rank
  // judges the same list with the same predicate and, if a better partition exists, enters
  // draining. Once a completed reduction shows every rank draining and the sent and
  // received totals equal and unchanged, and with all photos in blocks, all ranks make the
  // same blocking balance call.
  Mesh *pm = pmy_mesh;
  const bool lb_active = !dynamic && lb_check_interval > 0
                         && (pm->lb_automatic_ || pm->lb_manual_);
  const int64_t check_photons = std::max<int64_t>(1,
      static_cast<int64_t>(lb_check_fraction*static_cast<Real>(nsamptype[etype])));
  int64_t finished_at_check = 0, finished_at_lb = 0;
  int nlb = 0;
  bool draining = false, gather_pending = false;
  int drain_streak = 0;
  int64_t drain_prev_sent = -1, drain_prev_recv = -1;
  MPI_Request greq = MPI_REQUEST_NULL;
  std::vector<double> gcost;
  // Passes spent with neither work nor an arrival, and how many to spin through before
  // handing the core back.  Small enough that a rank stays responsive to a peer, large
  // enough that a busy rank alternating between work and short waits never yields.
  int idle_passes = 0;
  const int kSpinBeforeYield = 64;

  while (true) {
    const std::chrono::steady_clock::time_point pass0 = std::chrono::steady_clock::now();
    ++lb_passes;
    // Free the buffers of whatever sends have completed, without waiting for the rest:
    // each message set owns its buffers, so nothing here can overwrite one still in
    // flight, and a rank never stalls on a peer that has not reached its drain yet.
    // What landed is read from the counter across the two calls below, so that no
    // delivery can be missed by the flush that follows.
    const int64_t recv_before = pexch->NumRecv();
    std::chrono::steady_clock::time_point tt = std::chrono::steady_clock::now();
    pexch->RetireSends();
    std::chrono::steady_clock::time_point tt2 = std::chrono::steady_clock::now();
    lb_t_complete += std::chrono::duration<double>(tt2 - tt).count();
    pexch->DrainIncoming();
    tt = std::chrono::steady_clock::now();
    lb_t_drain_in += std::chrono::duration<double>(tt - tt2).count();
    const bool landed = (pexch->NumRecv() != recv_before);

    // Flushing the receive buffers is only needed when something actually landed: the
    // sub-loop below already drains what it hands between blocks on this rank, and
    // ClearBoundary leaves has_incoming_ false behind it, so with nothing delivered here
    // there is nothing waiting anywhere.  This matters because it is the whole idle path:
    // a rank waiting on a straggler goes round this loop as fast as it can, and a scan of
    // every block on each pass would cost more than the barrier it replaced.
    if (landed) {
      tt = std::chrono::steady_clock::now();
      DrainArrivals();
      lb_t_drain_arr += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - tt).count();
    }

    // Anything to push forward?  A rank with nothing skips the transport sweep entirely
    // rather than walking every block to find that out.
    bool have_work = false;
    for (int nb = 0; nb < nblocal; ++nb) {
      MonteCarloBlock *pmcb = my_blocks(nb);
      if (pmcb->pphot->nphot > 0 || pmcb->nphremain > 0) { have_work = true; break; }
    }

    // Time so far this pass is exchange (delivery and send completion)
    const std::chrono::steady_clock::time_point pass1 = std::chrono::steady_clock::now();
    const double sweep_before = lb_rank_time;
    if (have_work && !draining) {
      pexch->Reset();
      // local_max_sweeps bounds a same-rank ping-pong here.  Its other job in the
      // synchronous loop -- keeping two blocks trading a photon from holding every other
      // rank at the barrier -- does not apply, because there is no barrier to hold them
      // at.  What it still buys is a return to the drain above, so photons arriving from
      // other ranks are not left waiting behind an unbounded local sweep.
      int sweeps = 0;
      bool local_progress = true;
      while (local_progress) {
        for (int nb = 0; nb < nblocal; ++nb)
          TransportBlock(nb, etype);
        tt = std::chrono::steady_clock::now();
        local_progress = ExchangeLocal() && (++sweeps < local_max_sweeps);
        lb_t_local += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - tt).count();
      }
      tt = std::chrono::steady_clock::now();
      pexch->SendStaged();
      lb_t_send += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - tt).count();
      idle_passes = 0;
      // The sweep time is already counted per block; the rest of this working pass,
      // hand-offs between blocks and staging, is exchange.
      FoldPendingCosts();
      const double pass_wall = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - pass0).count();
      lb_exchange_time += pass_wall - LoadBalanceSeconds(lb_rank_time - sweep_before);
    } else if (!landed) {
      lb_idle_time += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - pass0).count();
      // Nothing to do and nothing arrived.  This rank is waiting on someone else's
      // straggler and will go round this loop as fast as the probe returns, holding a core
      // at full tilt for as long as that takes -- which is exactly the cycles the rank
      // still working needs, once a node is oversubscribed.  Yield after a short spin, so
      // a peer that is about to send still gets an immediate response but a long wait
      // costs the scheduler rather than the run.
      if (++idle_passes > kSpinBeforeYield) sched_yield();
    }

    // Local contribution to the termination test.  has_incoming_ has to count: a photon
    // that has been delivered into a block's receive buffer but not yet flushed into the
    // block is neither in flight nor visible in nphot, and would otherwise vanish from
    // both sides of the test at once.  A rank that neither had work nor took delivery has
    // none of the three and does not need to look.
    // Counted whenever draining, too: a draining rank holds photons it is not moving,
    // and leaving them out would let the termination test see an empty mesh.
    int64_t act = 0;
    if (have_work || landed || draining) {
      for (int nb = 0; nb < nblocal; ++nb) {
        MonteCarloBlock *pmcb = my_blocks(nb);
        act += pmcb->pphot->nphot;
        act += pmcb->nphremain;
        if (pmcb->pphot->has_incoming_) ++act;
      }
    }

    if (!pending) {
      sbuf[0] = act;
      sbuf[1] = pexch->NumSent();
      sbuf[2] = pexch->NumRecv();
      sbuf[3] = draining ? 1 : 0;
      sbuf[4] = 0;
      if (lb_active) {
        for (int nb = 0; nb < nblocal; ++nb) {
          MonteCarloBlock *pmcb = my_blocks(nb);
          sbuf[4] += pmcb->nesc + pmcb->nabs + pmcb->ndes + pmcb->nrem;
        }
      }
      MPI_Iallreduce(sbuf, rbuf, 5, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD, &treq);
      pending = true;
    } else {
      int done = 0;
      MPI_Test(&treq, &done, MPI_STATUS_IGNORE);
      if (done) {
        pending = false;
        const bool quiet = (rbuf[0] == 0 && rbuf[1] == rbuf[2]);
        if (quiet && rbuf[1] == prev_sent && rbuf[2] == prev_recv) {
          ++quiet_streak;
        } else {
          quiet_streak = quiet ? 1 : 0;
        }
        prev_sent = rbuf[1];
        prev_recv = rbuf[2];
        if (quiet_streak >= 2) break;

        if (lb_active && draining) {
          // Every rank draining, and the wire empty for two consecutive reductions
          const bool wire_quiet = (rbuf[3] == Globals::nranks && rbuf[1] == rbuf[2]);
          if (wire_quiet && rbuf[1] == drain_prev_sent && rbuf[2] == drain_prev_recv) {
            ++drain_streak;
          } else {
            drain_streak = wire_quiet ? 1 : 0;
          }
          drain_prev_sent = rbuf[1];
          drain_prev_recv = rbuf[2];
          if (drain_streak >= 2) {
            // Quiet means every send has been taken; release the handles before the
            // blocks and the peer list are rebuilt.
            pexch->CompleteSends();
            if (BalanceNow(pin)) ++nlb;
            // The exchange counters were zeroed by the relink if blocks moved; either
            // way start the termination history afresh so no stale pair can match.
            draining = false;
            drain_streak = 0;
            finished_at_lb = rbuf[4];
            finished_at_check = rbuf[4];
            prev_sent = prev_recv = -1;
            quiet_streak = 0;
          }
        } else if (lb_active && !gather_pending && nlb < lb_max_per_transport
                   && rbuf[4] - finished_at_lb >= check_photons
                   && rbuf[4] - finished_at_check >= check_photons) {
          finished_at_check = rbuf[4];
          if (lb_test_costs != LBCOST_MEASURED) AssignTestCosts();
          FillLocalBalancerCosts(gcost);
          MPI_Iallgatherv(MPI_IN_PLACE, pm->nblist[Globals::my_rank], MPI_DOUBLE,
                          gcost.data(), pm->nblist, pm->nslist, MPI_DOUBLE,
                          MPI_COMM_WORLD, &greq);
          gather_pending = true;
        }
      }
    }

    if (gather_pending) {
      int done = 0;
      MPI_Test(&greq, &done, MPI_STATUS_IGNORE);
      if (done) {
        gather_pending = false;
        if (WorthwhileFromCosts(gcost)) {
          draining = true;
          drain_streak = 0;
          drain_prev_sent = drain_prev_recv = -1;
        } else {
          DecayCosts();
        }
      }
    }
  }
  if (gather_pending) MPI_Wait(&greq, MPI_STATUS_IGNORE);

  if (pending) MPI_Wait(&treq, MPI_STATUS_IGNORE);
  // Termination means every photon this rank sent has been taken, so nothing is left to
  // wait on here; this just releases the request handles.
  pexch->CompleteSends();
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarlo::NormalizeDomainOutputs(bool nomralize)
//! \brief Normalize or unnormalize domain outputs

void MonteCarlo::NormalizeDomainOutputs(bool normalize) {

  if (normalize) {
    // normalize moments for output
    for(int nb=0; nb<nblocal; ++nb) {
      MonteCarloBlock *pmcb = my_blocks(nb);
      if (pmcb->mom_flag_lab || pmcb->mom_flag_com ||
          pmcb->mom_flag_scat || pmcb->mom_flag_usr)
        pmcb->NormalizeMoments(true);
      // SWD: Not sure this is needed
      if (pmcb->call_srcterms)
        pmcb->NormalizeSourceTerms(true);
    }
  } else {
    // unnormalize moments after output
    for(int nb=0; nb<nblocal; ++nb) {
      MonteCarloBlock *pmcb = my_blocks(nb);
      if (pmcb->mom_flag_lab || pmcb->mom_flag_com ||
          pmcb->mom_flag_scat || pmcb->mom_flag_usr)
        pmcb->NormalizeMoments(false);
      if (pmcb->call_srcterms)
        pmcb->NormalizeSourceTerms(false);
    }
  }
}

//----------------------------------------------------------------------------------------
//! MCRandom constructor, builds Athena++ random number generator
//  current implementation is wrapper for gsl function

MCRandom::MCRandom(int iseed)
#if GSL
  {
  dev = gsl_rng_alloc(gsl_rng_mt19937);
  gsl_rng_set(dev, iseed);
#else
  : gen(iseed), uniform_dist(0.0, 1.0)
  {
#endif

}

//----------------------------------------------------------------------------------------
//! destructor

MCRandom::~MCRandom() {
#if GSL
  gsl_rng_free(dev);
#endif
}

Real MCRandom::uniform() {

#if GSL
  return static_cast<Real>(gsl_rng_uniform(dev));
#else
  return uniform_dist(gen);
#endif
}

Real MCRandom::chisquare(Real nu) {
#if GSL
  return static_cast<Real>(gsl_ran_chisq(dev, nu));
#else
  std::chi_squared_distribution<Real> chi_dist(nu);
  return chi_dist(gen);
#endif
}

int MCRandom::binomial(unsigned int n, Real p) {
#if GSL
  return static_<Real>(gsl_ran_binomial(dev, p, n));
#else
  std::binomial_distribution<int> binomial(n, p);
  return binomial(gen);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn std::string MCRandom::SaveState() const
//! \brief the generator's state as bytes, so a moved block continues its own stream

std::string MCRandom::SaveState() const {
#if GSL
  return std::string(static_cast<const char*>(gsl_rng_state(dev)), gsl_rng_size(dev));
#else
  std::ostringstream os;
  os << gen;
  return os.str();
#endif
}

void MCRandom::RestoreState(const std::string &state) {
  std::stringstream msg;
#if GSL
  if (state.size() != gsl_rng_size(dev)) {
    msg << "### FATAL ERROR in MCRandom::RestoreState" << std::endl
        << "state of " << state.size() << " bytes for a generator of "
        << gsl_rng_size(dev) << std::endl;
    ATHENA_ERROR(msg);
  }
  std::memcpy(gsl_rng_state(dev), state.data(), state.size());
#else
  std::istringstream is(state);
  is >> gen;
  if (!is) {
    msg << "### FATAL ERROR in MCRandom::RestoreState" << std::endl
        << "could not parse a saved generator state" << std::endl;
    ATHENA_ERROR(msg);
  }
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void MCRandom::SampleMultinomial(std::int64_t n, int m, const Real *prob,
//!                                     std::int64_t *counts)
//! \brief multinomial draw for a population that may not fit in an int
//!
//! Both random-number backends take a 32-bit population in their binomial, so the chain
//! of binomials below cannot be used past 2^31-1.  Above that the expected count of each
//! bin, n*p_i rounded down, is assigned outright and only the remainder -- fewer than m
//! photons plus rounding slack -- is drawn with the int sampler.  The mean is exact and
//! the variance is below the multinomial's by an amount that does not matter at these
//! populations: the multinomial's own scatter in a bin is sqrt(n p_i), a part in
//! sqrt(n p_i) of the count, and the photon noise downstream is the same size.  Below
//! the threshold the int sampler is used unchanged, so every run that fit before draws
//! exactly what it always did.

void MCRandom::SampleMultinomial(std::int64_t n, int m, const Real *prob,
                                 std::int64_t *counts) {
  const std::int64_t int_max = std::numeric_limits<int>::max();
  std::vector<Real> p(prob, prob + m);
  std::vector<int> c(m, 0);
  if (n <= int_max) {
    SampleMultinomial(static_cast<int>(n), m, p.data(), c.data());
    for (int i=0; i<m; ++i) counts[i] = c[i];
    return;
  }
  std::int64_t assigned = 0;
  for (int i=0; i<m; ++i) {
    Real expect = static_cast<Real>(n) * prob[i];
    counts[i] = (expect > 0.0) ? static_cast<std::int64_t>(std::floor(expect)) : 0;
    assigned += counts[i];
  }
  // Rounding in the products can leave the floors summing past n; take the excess back
  // from the fullest bins.
  while (assigned > n) {
    int imax = 0;
    for (int i=1; i<m; ++i) if (counts[i] > counts[imax]) imax = i;
    --counts[imax];
    --assigned;
  }
  const std::int64_t remainder = n - assigned;
  if (remainder > 0) {
    if (remainder > int_max) {
      std::stringstream msg;
      msg << "### FATAL ERROR in MCRandom::SampleMultinomial" << std::endl
          << "remainder " << remainder << " after assigning expected counts; the"
          << " probabilities do not sum to one" << std::endl;
      ATHENA_ERROR(msg);
    }
    SampleMultinomial(static_cast<int>(remainder), m, p.data(), c.data());
    for (int i=0; i<m; ++i) counts[i] += c[i];
  }
}

void MCRandom::SampleMultinomial(int n, int m, Real *prob, int *counts) {
   
  int remain = n;
  Real prob_sum = 1.;
  // initialize counts to zero
  for (int i=0; i < m; i++) {
    counts[i] = 0;
  }
  for (int i=0; i < m-1; i++) {
    Real p = prob[i] / prob_sum;
    counts[i] = binomial(remain,p);
    remain -= counts[i];
    prob_sum -= prob[i];
    if (remain == 0) break;
  }
  counts[m-1] = remain;
  
}
