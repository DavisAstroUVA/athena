//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file monte_carlo.cpp
//! \brief implementation of functions in class MonteCarlo, MCRandom

// C++ headers
#include <cstdio>
#include <random>
#include <stdexcept>  // runtime_error
// C++ headers
#include <cstring>  // strcmp
#include <string>

// Athena++ headers
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
  checkmove = pin->GetOrAddInteger("montecarlo","checkmove",10000);

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
  }
}

//----------------------------------------------------------------------------------------
//! destructor

MonteCarlo::~MonteCarlo() {

  delete pmcout;
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
  for (int i=0; i<nblocal; i++) {
    MonteCarloBlock *pmcb = my_blocks(i);
    // Initialize variables over all blocks
    pmcb->GetDensity();
    pmcb->GetTemperature();
    pmcb->GetNumberDensity();
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

    // initialize counters to zero
    pmcb->nscat = pmcb->nesc = pmcb->nabs = pmcb->ndes = pmcb->nrem = 0;
    pmcb->loop_max_size = pin->GetOrAddInteger("montecarlo","loop_max_size",10000);

    // Call problem generators for Monte Carlo
    pmcb->MonteCarloProblemGenerator(pin);
  }

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
    nsamptype[0] = nsamp = pin->GetInteger("montecarlo","nphot");
    emission_eqwt[0] = pin->GetOrAddBoolean("montecarlo","equal_weight",false);
    initialize_comoving[0] = pin->GetOrAddBoolean("montecarlo","initialize_comoving",true);
    absorption_method[0] = GetAbsorptionMethodFlag(abs_def);
    emission_geometry[0] = GetEmissionGeometry(pin->GetOrAddString("montecarlo","emission_geometry","none"));
    emission_array = false; // do not allocate memory for array;
  } else if (emission_flag ==  EMISUSER) {
    GetEmission[0] = nullptr; // must be set in InitUserMonteCarloData
    nsamptype[0] = nsamp = pin->GetInteger("montecarlo","nphot");
    emission_eqwt[0] = pin->GetOrAddBoolean("montecarlo","equal_weight",false);
    initialize_comoving[0] = pin->GetOrAddBoolean("montecarlo","initialize_comoving",true);
    absorption_method[0] = GetAbsorptionMethodFlag(abs_def);
    emission_geometry[0] = GetEmissionGeometry(pin->GetOrAddString("montecarlo","emission_geometry","volume"));
    if (emission_geometry[0] == EMISAREA)
      emission_face[0] = SetEmissionSurface(pin->GetOrAddString("montecarlo","emission_face","none"));
    emission_array = true; // allocate memory for array
  } else if (emission_flag ==  EMISFF) {
    GetEmission[0] = GetEmissionFreeFree;
    nsamptype[0] = nsamp = pin->GetInteger("montecarlo","nphot");
    emission_eqwt[0] = pin->GetOrAddBoolean("montecarlo","equal_weight",false);
    initialize_comoving[0] = pin->GetOrAddBoolean("montecarlo","initialize_comoving",true);
    absorption_method[0] = GetAbsorptionMethodFlag(abs_def);
    emission_geometry[0] = EMISVOL; // Must be volumetric
    emission_array = true; // allocate memory for array
  } else if (emission_flag ==  EMISBB) {
    GetEmission[0] = GetEmissionBlackbody;
    nsamptype[0] = nsamp = pin->GetInteger("montecarlo","nphot");
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
  // Set methods, numbers for this emission type
  int ntot = nsamptype[etype];
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
    Real emiss_proc[Globals::nranks];
#ifdef MPI_PARALLEL
    MPI_Gather(&em_proc,1,MPI_ATHENA_REAL,emiss_proc,1,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
#else
    emiss_proc[0] = em_proc;
#endif
    int count[Globals::nranks];
    // Rank 0 compute the distribution of photons accross all processes and brodcasts
    if (Globals::my_rank == 0) {
      Real prob[Globals::nranks];
      for (int irank=0; irank<Globals::nranks; irank++) 
        prob[irank] = emiss_proc[irank]/em_tot;
      my_blocks(0)->pran->SampleMultinomial(ntot,Globals::nranks,prob,count);
    }
    int my_count;
#ifdef MPI_PARALLEL
    MPI_Scatter(count,1,MPI_INT,&my_count,1,MPI_INT,0,MPI_COMM_WORLD);
#else
    my_count = count[0];
#endif
    // Now distribute the photons across all blocks on this process
    int count_b[nblocal];
    Real prob_b[nblocal];
    for (int nb=0; nb<nblocal; nb++)
      prob_b[nb] = tot_block[nb]/em_proc;
    my_blocks(0)->pran->SampleMultinomial(my_count,nblocal,prob_b,count_b);
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
    int count[Globals::nranks];
    int nb_active_total = 0;
    if (Globals::my_rank == 0) {
      Real prob[Globals::nranks];
      for (int irank=0; irank<Globals::nranks; irank++) {
        nb_active_total += active_proc[irank];
      }
      for (int irank=0; irank<Globals::nranks; irank++) {
        prob[irank] = static_cast<Real>(active_proc[irank])/static_cast<Real>(nb_active_total);
      }
      my_blocks(0)->pran->SampleMultinomial(ntot,Globals::nranks,prob,count);
    }
    int my_count;
#ifdef MPI_PARALLEL
    MPI_Scatter(count,1,MPI_INT,&my_count,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&nb_active_total, 1, MPI_INT, 0, MPI_COMM_WORLD);
#else
    my_count = count[0];
#endif
    // Now distribute the photons across all active blocks on this process
    int count_b[nblocal];
    Real prob_b[nblocal];
    for (int nb=0; nb<nblocal; nb++)
      if (tot_block[nb] > 0.)
        prob_b[nb] = 1./static_cast<Real>(nb_active);
      else
        prob_b[nb] = 0.;
    my_blocks(0)->pran->SampleMultinomial(my_count,nblocal,prob_b,count_b);
    //printf("Rank %d: nb_active=%d, nb_active_total=%d\n", Globals::my_rank, nb_active, nb_active_total);  
    // Acount for inactive blocks in weighting
    int block_size = ncells / pmy_mesh->nbtotal;
    int ncells_active = nb_active_total * block_size;
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

  // Update MC blocks if needed
  for (int nb=0; nb<nblocal; nb++) {
    MonteCarloBlock *pmcb = my_blocks(nb);
    if (dynamic) { // dynamic MC
      // (Re)Initialize/update variables over all blocks
      pmcb->GetDensity();
      pmcb->GetTemperature();
      pmcb->GetNumberDensity();
      if (boosts) {
        pmcb->GetVelocity();
        pmcb->ComputeTransformations();
      } else if (GENERAL_RELATIVITY) {
        // no fluid velocity, but the GR tetrad still needs a frame to be built on
        pmcb->SetNormalObserver();
        pmcb->ComputeTransformations();
      }
      if (NSCALARS > 0) pmcb->GetScalars(); //scalars
      if (using_bfield) pmcb->GetBField();
    } else { // static MC
      // Clear Boundary buffers for photons
      pmcb->pphot->ClearBoundary();
    }
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
    }
    // Distribute samples to all blocks based on emission properties
    // Sets nphremain and parameters for determining initial photon weights
    DistributeSamples(etype);
    emission_method = etype;

    // Run Monte Carlo until all photons have escaped/been absorbed
    bool photons_remain = true; // True if photons on any process
    while(photons_remain) {
      for(int nb=0; nb<nblocal; ++nb){
        if (raytrace_flag)
          my_blocks(nb)->RayTracePhotonsOnBlock(etype);
        else
          my_blocks(nb)->TransferPhotonsOnBlock(etype);
      }
      photons_remain = CheckAndBroadCastPhotonsRemaining();
    }

    // Report diagnostic results from all blocks
    int ntot = 0;
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
    pmcout->UpdateOutputCount(ntot);

  #ifdef MPI_PARALLEL
    MPI_Allreduce(MPI_IN_PLACE,&nesc,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&nabs,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&ndes,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&nscat,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&ntot,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&nrem,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
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

    for(int nb=0; nb<nblocal; ++nb) {
      my_blocks(nb)->UserWorkAfterTransfer(etype);
    }
  } // end loop over ntype

}

//----------------------------------------------------------------------------------------
//! \fn bool MonteCarlo::CheckAndBroadCastPhotonsRemaining() 
//! \brief Checks if photons remain on any process and sends message to rank 0
//!        process if none remaining.

bool MonteCarlo::CheckAndBroadCastPhotonsRemaining() {

  // Send photons from all blocks
  for(int nb=0; nb<nblocal; ++nb){
    my_blocks(nb)->pphot->SendToNeighbors();
  }

  // Receive photons from all blocks
  bool complete = false;
  //int count = 0;
  while(!complete) {
    complete = true;
    for(int nb=0; nb<nblocal; ++nb) {
      bool success = my_blocks(nb)->pphot->ReceiveFromNeighbors();
      if (!success)
        complete = false;
    }
    //count++;
    //if (count % 100000 == 0)
    //printf("here %d %d \n",count, Globals::my_rank);
  }

  // Clear Boundaries
  for(int nb=0; nb<nblocal; ++nb)
    my_blocks(nb)->pphot->ClearBoundary();

  // Check if photons have completed
  int nremain=0,nprop=0;
  for(int nb=0; nb<nblocal; ++nb){
    MonteCarloBlock *pmcb = my_blocks(nb);
    nremain += pmcb->nphremain;
    nprop += pmcb->pphot->nphot;
    //if (pmcb->nphremain > 0)
    //  printf("rem: %d %d \n",pmcb->pmy_block->gid,pmcb->nphremain);
    //if (pmcb->pphot->nphot > 0)
    //  printf("prop: %d %d \n",pmcb->pmy_block->gid,pmcb->pphot->nphot);
  }
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE,&nprop,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&nremain,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
#endif

  bool active;
  if ((nremain > 0) || (nprop > 0)) {
    active = true;
  } else {
    active = false;
  }
  //if (Globals::my_rank == 0) {
  //  printf("nremain: %d nprop: %d active: %d\n",nremain,nprop,active);
  //}
  return active;
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
