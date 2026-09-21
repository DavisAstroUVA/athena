//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file mc_hotjupiter.cpp
//! \brief Plane-parallel source of radiation from a star incident on the atomic H layer
//! of an exoplanet atmosphere, including photoelectric heating, recombination, and
//! resonance scattering of Lya

// C headers

// C++ headers
#include <algorithm> // upper_bound()
#include <cmath>
#include <cstdio>  // fopen(), fprintf(), freopen()
#include <cstring> // strcmp()
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../monte_carlo/mcsnapshot.hpp"
#include "../monte_carlo/montecarlo.hpp"
#include "../monte_carlo/photon.hpp"
#include "../monte_carlo/photonpusher.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"

#if NSCALARS < 1
#error "This problem generator requires scalars to track neutral hydrogen fraction"
#endif

namespace {

// ***** VARIABLES *****

// Flags to determine which processe etype corresponds to
int ion_str = -1, lya_str = -1, lya_rec = -1;

// parameters for hydro integration /grid initialization
Real n_cgs; // code units for number density
Real P_cgs;
Real rin, rout; // inner and outer limits of coordinate system
Real gm_planet, gm_star, sep, psi, rstar;
Real temp0, nbase, a2;
Real mdot;
Real pfloor, dfloor;
// Ghost zones arrays
Real rho_gz[2];
Real P_gz[2];
Real nfrac_gz[2];
Real user_dt;
bool flag_static_inner;
int  flag_tidal_gravity;
bool flag_wind = false;
bool flag_init_neutral;
bool flag_update_nh;

// parameters for MC radiation transfer
Real energy_lya;
Real lya_flux;
Real ion_flux;
Real linewidth_cutoff_energy, linewidth;
Real numin, mean_nu, nuexp, numinpow, numaxpow;
Real chromo_temp, sigmamin;
bool flag_incident_from_z;
bool flag_incident_from_r;
bool flag_finite_star;
bool flag_pow_law;
bool flag_lya_abs;
bool flag_core_skipping;
bool flag_init_escaped;
// initialization for timing photons
const auto global_start_time = std::chrono::system_clock::now();

// sampling from finite stellar spectrum
bool flag_sample_lya;
Real spectrum_lya_itot; // wavelength-integrated intensity [erg/cm^2/s]
Real spectrum_lya_energy; // mean energy from spectrum [erg]
std::vector<Real> spectrum_lya_wl; // lya wavelengths [cm]
std::vector<Real> spectrum_lya_cdf; // lya CDF per wl bin

// Flag for performance-saving random deviate sampling using Box-Muller method
//  which generates two deviates at a time, used for stellar lyman alpha
int iset = 0;
Real gset; // saves extra deviate

enum EmissionType {LYA=0, IONIZING=1};
enum EmissionGeometryHJ {VOLUME=0, SURFACE=1};

// ***** FUNCTION PROTOTYPES *****

// Box-Muller sampling uses for stellar lyman alpha
void gasdev(MeshBlock *pmb, Real mean, Real sigma, Real &samp);

// Read a two-column stellar spectrum and build the CDF used to sample from it
void ReadSpectrumToCDF(const std::string& filename, std::vector<Real>& wl,
                       std::vector<Real>& cdf, Real& itot, Real& emean);

// Invert the spectrum CDF: linear interpolation of wl(cdf) at cdf = u
Real SampleSpectrumWavelength(const std::vector<Real>& wl, const std::vector<Real>& cdf,
                              Real u);

// velocity initial isothermal wind profile
Real GetIsowindVelocity(Real x);

// Functions to calculate the emission for various prcosesses
Real VolumeEmissivityLya(MonteCarloBlock *pmcb, int k, int j, int i, int etype);
Real SurfaceEmissivityLya(MonteCarloBlock *pmcb, int k, int j, int i, int etype);
Real SurfaceEmissivityIonizing(MonteCarloBlock *pmcb, int k, int j, int i, int etype);
Real GetProjectedArea(MonteCarloBlock *pmcb, int k, int j, int i);
Real GetProjectedAreaOmegaFiniteStar(MonteCarloBlock *pmcb, int k, int j, int i);

// user gravity source term functions
void GetTidalAcceleration(Real r, Real th, Real ph, int order, Real &a_r, Real &a_th, Real &a_ph);
void TwoPointMass(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);
void HillTidalGravity(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);
void ThirdOrderTidalGravity(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);
Real HydrostaticIsothermalDensity(Real r, Real th, Real ph, Real r0, Real a2, int tidal_order);
Real HydrostaticAdiabaticDensity(Real r, Real th, Real ph, Real r0, Real a2, Real invgm1, int tidal_order);
Real HydrostaticIonization(Real n_a, Real n_c);

// user functions for defining scattering and opacities
void ResonantScattering(MonteCarloBlock *pmcb, Photon *pphot, int ips, int ipe);
Real ResonantScatteringOpacity(MonteCarloBlock *pmcb, Photon *pphot, int ip);
Real BoundFreeAbsorptionOpacity(MonteCarloBlock *pmcb, Photon *pphot, int ip);
Real ChooseAbsorptionOpacity(MonteCarloBlock *pmcb, Photon *pphot, int ip);

// user timestep SWD: not sure why this is used
Real ConstantTimestep(MeshBlock *pmb);

// user functions for converting temperature and number density to cgs
void GetIonizationTemperature(MonteCarloBlock *pmcb);
void GetNH(MonteCarloBlock *pmcb);

// user function for photon acceleration
void CoreSkipping(MonteCarloBlock *pmcb, Photon *pphot, PhotonPusher *ppusher, int ip);

// user function to update source terms for coupling to hydro; handles different types
void UpdateSourceTerms(MonteCarloBlock *pmcb, Photon *pphot,Real energy0, Real weight0,
                  Real k1p0, Real k2p0, Real k3p0, int ip);

// Boundary Conditions
void InflowInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void OutflowOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void StaticInflowInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);

} // end namespace



void Mesh::InitUserMeshData(ParameterInput *pin) {

  // Initialize file-scope variables with values read in from the input file
  // Enroll the user gravity source term

  psi = pin->GetOrAddReal("problem", "psi", 0.0)*PI; // radians, defaults to star in -x direction

  Real l_cgs = pin->GetOrAddReal("problem","l_cgs",1.);
  Real vel_cgs = pin->GetOrAddReal("problem","vel_cgs",1.);
  Real rho_cgs = pin->GetOrAddReal("problem","rho_cgs",1.);
  n_cgs = rho_cgs / MCConstants::mp_cgs;
  P_cgs = rho_cgs * SQR(vel_cgs);
  Real time_cgs = l_cgs / vel_cgs;

  // Gravity source term configuration
  gm_planet = pin->GetReal("problem", "GM");
  gm_star = pin->GetReal("problem", "gm_star");
  sep = pin->GetReal("problem", "orbit_sep")*1.49597871e+13/l_cgs;
  flag_tidal_gravity = pin->GetOrAddInteger("problem", "tidal_gravity", 0);
  if (flag_tidal_gravity == 1) {
    EnrollUserExplicitSourceFunction(HillTidalGravity);
  }
  if (flag_tidal_gravity == 2) {
    EnrollUserExplicitSourceFunction(ThirdOrderTidalGravity);
  }

  // Parameters to set initial atmospheric conditions
  temp0 = pin->GetOrAddReal("problem", "isothermal_temp", 1e4);
  nbase = pin->GetOrAddReal("problem", "nbase", 1.0e10)/n_cgs; // base hydrogen nH density in cm^-3 at rin
  flag_init_neutral = pin->GetOrAddBoolean("problem", "init_neutral", false);
  flag_wind = pin->GetOrAddBoolean("problem", "wind", true);

  // Time evolution
  // Set a constant timestep (best for photoionization equilibrium convergence tests)
  //user_dt = pin->GetReal("problem", "user_dt");
  //EnrollUserTimeStepFunction(ConstantTimestep);

  // Update ionization fraction
  flag_update_nh = pin->GetOrAddBoolean("problem", "update_nh", true);

  // Boundary conditions
  flag_static_inner = pin->GetOrAddBoolean("problem", "static_inner", true);
  if (flag_static_inner) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, StaticInflowInnerX1);
  } else {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, InflowInnerX1);
  }
  EnrollUserBoundaryFunction(BoundaryFace::outer_x1, OutflowOuterX1);

  return;
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {

  AllocateRealUserMeshBlockDataField(1);
  // Array for each component of the tidal gravity acceleration.
  ruser_meshblock_data[0].NewAthenaArray(3,ncells3,ncells2,ncells1);

  AllocateUserOutputVariables(7);
  SetUserOutputVariableName(0, "vol_emis");
  SetUserOutputVariableName(1, "surf_emis");
  SetUserOutputVariableName(2, "gravsrc_r");
  SetUserOutputVariableName(3, "gravsrc_th");
  SetUserOutputVariableName(4, "gravsrc_ph");
  SetUserOutputVariableName(5, "collisional_cooling");
  SetUserOutputVariableName(6, "gradP");
  return;
}

// set user output variables to corresponding meshblock arrays
void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  AthenaArray<Real> &uov = user_out_var;
  for (int k = ks; k <= ke; k++) {
    for (int j = js; j <= je; j++) {
      for (int i = is; i <= ie; i++) {
        // arrays to store tidal gravity components
        uov(2,k,j,i) = ruser_meshblock_data[0](0,k,j,i);
        uov(3,k,j,i) = ruser_meshblock_data[0](1,k,j,i);
        uov(4,k,j,i) = ruser_meshblock_data[0](2,k,j,i);

        Real r_m = pcoord->x1f(i);
        Real r_p = pcoord->x1f(i+1);
        Real pres_i = phydro->w(IPR,k,j,i);
        Real pres_im1 = phydro->w(IPR,k,j,i-1);
        Real pres_ip1 = phydro->w(IPR,k,j,i+1);
        Real pres_m = 0.5*( pres_im1 + pres_i );
        Real pres_p = 0.5*( pres_ip1 + pres_i );
        uov(6,k,j,i) = (pres_p - pres_m) / (r_p - r_m);
      }
    }
  }
}

// Read in input params that define the attributes of the Monte Carlo photons
void MonteCarlo::InitUserMonteCarloData(ParameterInput *pin) {

  bool report_setup = pin->GetOrAddBoolean("problem","report_setup",false);

  int nion = pin->GetOrAddInteger("problem","nion",0);
  int nlyastr = pin->GetOrAddInteger("problem","nlyastr",0);
  int nlyarec = pin->GetOrAddInteger("problem","nlyarec",0);
  int ecount = 0;
  if (nion > 0) ecount++;
  if (nlyastr > 0) ecount++;
  if (nlyarec > 0) ecount++;
  if (ecount != ntype) {
    std::stringstream msg;
    msg << "### FATAL ERROR in MonteCarlo::InitUserMonteCarloData" << std::endl
        << "Number of emission types (ntype) does not match number of emission flags set "
        << "to true. Please check that ntype is consistent with the number of true values "
        << " among ion_surface_emis, lya_surface_emis, or lya_volume_emis."
        << std::endl;
    throw std::runtime_error(msg.str());
  }
  // set up emission methods.  nsamp accumulates over the types that are turned on, so it
  // starts from zero here as well as in the MULTI branch of InitializeEmission: assigning
  // it in the first block only worked while the first block was the one that ran.
  int etype = 0;
  nsamp = 0;
  if (nion > 0) {
    ion_str = etype;
    nsamp += nsamptype[etype] = nion;
    emission_eqwt[etype] = pin->GetOrAddBoolean("problem", "ion_eqwt",true);
    initialize_comoving[etype] = false; // ions initialized in lab frame
    std::string abs_meth = pin->GetOrAddString("problem","ion_str_abs","tau");
    absorption_method[etype] = GetAbsorptionMethodFlag(abs_meth);
    EnrollUserEmissionFunction(SurfaceEmissivityIonizing,etype);
    emission_geometry[etype] = EMISAREA;
    emission_face[etype] = SetEmissionSurface("outer_x1");
    etype++;
    if (report_setup && (Globals::my_rank == 0)) {
      std::cout << "Computing stellar ionizing photons emitted from surface."
                << std::endl;
      std::cout << "nion: " << nion << std::endl;
      std::cout << "absortion method: " << abs_meth << std::endl;
      std::cout << "equal weight: " << emission_eqwt[etype-1] << std::endl;
      std::cout << "etype: " << etype-1 << std::endl << std::endl;
    }
  }
  if (nlyastr > 0) {
    lya_str = etype;
    nsamp += nsamptype[etype] = nlyastr;
    emission_eqwt[etype] = pin->GetOrAddBoolean("problem", "lys_eqwt",true);
    initialize_comoving[etype] = false; // stellar  initialized in lab frame
    std::string abs_meth = pin->GetOrAddString("problem","lya_str_abs","weight");
    absorption_method[etype] = GetAbsorptionMethodFlag(abs_meth);
    emission_geometry[etype] = EMISAREA;
    EnrollUserEmissionFunction(SurfaceEmissivityLya,etype);
    emission_face[etype] = SetEmissionSurface("outer_x1");
    etype++;
    if (report_setup && (Globals::my_rank == 0)) {
      std::cout << "Computing stellar lyman alpha photons emitted from surface."
                << std::endl;
      std::cout << "nlyastr: " << nlyastr << std::endl;
      std::cout << "absortion method: " << abs_meth << std::endl;
      std::cout << "equal weight: " << emission_eqwt[etype-1] << std::endl;
      std::cout << "etype: " << etype-1 << std::endl << std::endl;
    }
  }
  if (nlyarec > 0) {
    lya_rec = etype;
    nsamp += nsamptype[etype] = nlyarec;
    emission_eqwt[etype] = pin->GetOrAddBoolean("problem", "lyr_eqwt",false);
    initialize_comoving[etype] = true; // recomb. lyman alpha initialized in comoving frame
    std::string abs_meth = pin->GetOrAddString("problem","lya_rec_abs","weight");
    absorption_method[etype] = GetAbsorptionMethodFlag(abs_meth);
    emission_geometry[etype] = EMISVOL;
    EnrollUserEmissionFunction(VolumeEmissivityLya,etype);
    emission_face[etype] = SetEmissionSurface("none");
    if (report_setup && (Globals::my_rank == 0)) {
      std::cout << "Computing recombination lyman alpha photons emitted throughout volume."
                << std::endl;
      std::cout << "nlyarec: " << nlyarec << std::endl;
      std::cout << "absortion method: " << abs_meth << std::endl;
      std::cout << "equal weight: " << emission_eqwt[etype] << std::endl;
      std::cout << "etype: " << etype << std::endl << std::endl;
    }
  }

  EnrollUserSourcetermUpdate(UpdateSourceTerms);
  // Photon user variables for incident and outgoing direction vector and position
  nuser_var = 2;

  // Enroll user functions for handling MC transport
  EnrollUserScatteringFunction(ResonantScattering);
  EnrollUserOpacityFunction(ResonantScatteringOpacity, false);
  flag_lya_abs = pin->GetOrAddBoolean("problem", "lya_abs", false);
  if (!flag_lya_abs) {
    EnrollUserOpacityFunction(BoundFreeAbsorptionOpacity, true);
  } else {
    EnrollUserOpacityFunction(ChooseAbsorptionOpacity, true);
  }

  EnrollUserGetTemperature(GetIonizationTemperature);
  EnrollUserGetNumberDensity(GetNH);

  flag_core_skipping = pin->GetOrAddBoolean("problem", "core_skip", false);
  if (flag_core_skipping) {
    // CoreSkipping compares the cell's comoving line optical depth against a threshold,
    // so it reads MCCoord::dmin.  That array is only built under MRW acceleration or
    // <montecarlo>/compute_dmin
    if (!acceleration && !compute_dmin) {
      std::stringstream msg;
      msg << "### FATAL ERROR in MonteCarlo::InitUserMonteCarloData" << std::endl
          << "<problem>/core_skip needs the smallest cell width, MCCoord::dmin."
          << std::endl
          << "Set <montecarlo>/compute_dmin = true (or acceleration = true, which also "
          << "turns on" << std::endl << "the MRW random walk)." << std::endl;
      throw std::runtime_error(msg.str());
    }
    EnrollUserWorkInMove(CoreSkipping);
  }

  flag_init_escaped = pin->GetOrAddBoolean("problem", "init_escaped", false);

  // sampling from an input stellar spectrum
  flag_sample_lya = pin->GetOrAddBoolean("problem", "sample_lya", false);
  if (flag_sample_lya) {
    std::string input_spectrum = pin->GetString("problem", "lya_filename");
    ReadSpectrumToCDF(input_spectrum, spectrum_lya_wl, spectrum_lya_cdf, spectrum_lya_itot, spectrum_lya_energy);
    if (report_setup && (Globals::my_rank == 0)) {
      std::cout << "Sampling from finite star:" << std::endl;
      std::cout << "itot: " << spectrum_lya_itot << " [erg/cm^2/s]" << std::endl;
      std::cout << "emean: " << spectrum_lya_energy << " [erg], emean / linecenter: " << spectrum_lya_energy/MCConstants::h_cgs/MCConstants::nu_lya << std::endl;
    }
  }


}


void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  // Set initial conditions before the main loop starts. Note that any variables
  // or arrays initialized here are *not* saved in restart files and are
  // inaccessible except at initialization

  // If initializing passive scalar from an existing file, load it here
  bool flag_initialize_scalar_from_file = pin->GetBoolean("problem", "initialize_scalar_from_file");
  bool flag_initialize_pressure_from_file = pin->GetBoolean("problem", "initialize_pressure_from_file");
  bool flag_initialize_velocity_from_file = pin->GetBoolean("problem", "initialize_velocity_from_file");
  bool flag_initialize_density_from_file = pin->GetBoolean("problem", "initialize_density_from_file");

  // BEGIN INITIALIZATION FILE I/O
  //
  // Ugilizd MCReadSnapshotBlock for parsing the athdf inputfile
  const bool read_from_file = flag_initialize_scalar_from_file
                              || flag_initialize_pressure_from_file
                              || flag_initialize_density_from_file
                              || flag_initialize_velocity_from_file;
  if (read_from_file) {
    // Mesh::nblist is private to everything but MeshBlock, so the collective-read padding
    // count is gathered here and handed over.
    int max_blocks_per_rank = 0;
    for (int r = 0; r < Globals::nranks; ++r)
      max_blocks_per_rank = std::max(max_blocks_per_rank, pmy_mesh->nblist[r]);
    MCReadSnapshotBlock(this, pin, max_blocks_per_rank);
  }
  // END INITIALIZATION FILE I/O

  Real l_cgs = pin->GetOrAddReal("problem","l_cgs",1.);
  Real vel_cgs = pin->GetOrAddReal("problem","vel_cgs",1.);
  Real rho_cgs = pin->GetOrAddReal("problem","rho_cgs",1.);

  rin = pmy_mesh->mesh_size.x1min;       // base radius in cgs
  rout = pmy_mesh->mesh_size.x1max;  // outer radius code units

  Real kb_cgs = MCConstants::kb_cgs;
  Real mp_cgs = MCConstants::mp_cgs;
  Real mmw0 = 1.0; // mean molecular weight at base
  a2 = kb_cgs * temp0 / mmw0 / mp_cgs / SQR(vel_cgs);
  Real a = std::sqrt(a2);  // base sound speed

  Real lambda = gm_planet / (rin * a2);
  Real rsonic = gm_planet / (2.0 * a2);

  Real gamma = peos->GetGamma();
  Real invgm1 = 1.0/(gamma - 1.0);

  // optically thin photoionization rate approximation
  // assumes power-law flux (-2) and cross section (-3)
  // normalized to solar flux (1.1e3 erg cm^-2 s^-1)
  ion_flux = pin->GetReal("problem", "ion_flux");
  Real Gamma0 = 6.3e-5 * ion_flux / 1.1e3; // s^-1

  // recombination rate coefficient, se Trammell 2011
  Real alpha = 2.6e-13 * std::pow(temp0*1.e-4, -0.8); // cm^3 s^-1

  // critical density for ionization equilibrium
  Real neq0 = Gamma0 / alpha / n_cgs; // dimensionless

  Real sigmapi = 6.e-18; // cm^2, cross section at the ionization edge
  sigmapi /= (mp_cgs/rho_cgs/l_cgs); // convert to code units

  Real vbase = 0.0;
  Real vel1 = 0.0;
  Real vel2 = 0.0;
  Real vel3 = 0.0;
  if (flag_wind) {
    vbase = a * GetIsowindVelocity(2.0/lambda);
    mdot = 4.*PI * SQR(rin) * nbase * vbase;
  }

  // density and pressure floors
  dfloor = pin->GetOrAddReal("hydro", "dfloor", 1.e-8);
  pfloor = pin->GetOrAddReal("hydro", "pfloor", 1.e-8);

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is-NGHOST; i<=ie+NGHOST; i++) {
        // dimensionless variables
        Real r = pcoord->x1v(i);
        Real th = pcoord->x2v(j);
        Real ph = pcoord->x3v(k);

        // Initialize density to hydrostatic profile, with tidal correction
        Real hydrostatic_profile = HydrostaticIsothermalDensity(r, th, ph, rin, a2, flag_tidal_gravity+1);
        //Real hydrostatic_profile = HydrostaticAdiabaticDensity(r, th, ph, rin, a2, invgm1, flag_tidal_gravity+1);
        Real rho = nbase * hydrostatic_profile;
        Real P = a2 * nbase / gamma * std::pow(hydrostatic_profile, gamma);

        // adiabatic solution has critical point where density -> zero; correct for this
        if (rho <= dfloor) {
          rho = dfloor;
        }
        if (P <= pfloor) {
          P = pfloor;
        }

        if (flag_wind) {
          // isothermal Parker wind approximation
          vel1 = a * std::exp(1.5*(1 - rsonic/r)) * std::sqrt(rsonic/r);
          rho = mdot / (4*PI * SQR(r) * vel1);
          P = a2 * rho;
        } else {
          vel1 = 0;
        }
        vel2 = 0;
        vel3 = 0;

        // Ionization fractions
        Real neutral_frac = 1;
        Real mmw = 1;
        Real nH = rho;
        Real np = 0;
        if (!flag_init_neutral) {
          nH = HydrostaticIonization(rho, neq0);
          // approximation for spherically attenuated ionization
          //nH = nbase * std::exp(lambda * (rin/r - 1.));
          //Real H = rin / lambda * (SQR(r) / SQR(rin));
          //Real ion_rate_atten = 1. / (1.0 + std::pow(nH*sigmapi*H, 1.5));
          //np = std::sqrt(ion_rate_atten * nH * neq0);
          //neutral_frac = nH / (nH + 2.*np);
          //mmw = (1. + neutral_frac)/2.;
        }

       // if (flag_wind) {
       //   vel1 = a * GetIsowindVelocity(r/rsonic);
       //   rho = mdot / (4.0 * PI * r * r * vel1);
       //   // Re-assign neutral and ion densities based on new density profile
       //   // Doing it this way so that nH + np = rho
       //   // If we kept the nH from HSE, we may have places where nH > new rho
       //   nH = neutral_frac * rho / mmw;
       //   np = (1.-mmw) * rho / mmw;
       // } else {
       //   vel1 = 0.0;
       //   rho = (nH + np);
       // }
        // SWD: not sure if this need recomputing if !flag_wind
        //mmw = (nH + np)/(2.*np + nH);
        //Real a2_mmw = kb_cgs * temp0 / mmw / mp_cgs;
        //Real P = rho * a2_mmw / (vel_cgs*vel_cgs);

        // CMF: need a better solution for tidal boundary that depends on theta/phi
        // Set global ghost zone constant values for inner boundary
        //if ((r <= rin) && (i < NGHOST)) {
        //  rho_gz[i] = nbase * hydrostatic_profile;
        //  P_gz[i] = a2 * nbase / gamma * std::pow(hydrostatic_profile, gamma);

        // A snapshot covers the active cells only, so the x1 ghosts this loop also visits
        // keep the analytic hydrostatic values above.
        const bool from_file = (i >= is && i <= ie);

        if (flag_initialize_pressure_from_file && from_file) {
          P = phydro->w(IPR,k,j,i);
        }

        if (flag_initialize_density_from_file && from_file) {
          rho = phydro->w(IDN,k,j,i);
        }

        if (flag_initialize_velocity_from_file && from_file) {
          vel1 = phydro->w(IVX,k,j,i);
          vel2 = phydro->w(IVY,k,j,i);
          vel3 = phydro->w(IVZ,k,j,i);
        }

        // Set corresponding conserved fluid variables for the above
        phydro->u(IDN,k,j,i) = rho;        // Density
        phydro->u(IM1,k,j,i) = rho * vel1; // X1 Momentum
        phydro->u(IM2,k,j,i) = rho * vel2; // X2 Momentum
        phydro->u(IM3,k,j,i) = rho * vel3; // X3 Momentum
        phydro->u(IEN,k,j,i) = P * invgm1;   // Internal energy
        phydro->u(IEN,k,j,i) += 0.5 * SQR(phydro->u(IM1,k,j,i)) / phydro->u(IDN,k,j,i);
        phydro->u(IEN,k,j,i) += 0.5 * SQR(phydro->u(IM2,k,j,i)) / phydro->u(IDN,k,j,i);
        phydro->u(IEN,k,j,i) += 0.5 * SQR(phydro->u(IM3,k,j,i)) / phydro->u(IDN,k,j,i);

        // Ionization state of the gas.  The file carries the concentration r0 = s0/rho,
        // which MCReadSnapshotBlock left in pscalars->r.
        if (flag_initialize_scalar_from_file && from_file) {
          pscalars->s(0,k,j,i) = pscalars->r(0,k,j,i)*rho;
        } else {
          pscalars->s(0,k,j,i) = nH;
        }

        GetTidalAcceleration(r, th, ph, flag_tidal_gravity+1,
          ruser_meshblock_data[0](0,k,j,i),
          ruser_meshblock_data[0](1,k,j,i),
          ruser_meshblock_data[0](2,k,j,i));
      }
    }
  }

}


void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin) {

  // Parameters for Lya emission
  Real nu_lya = MCConstants::nu_lya;
  Real h_cgs = MCConstants::h_cgs;
  energy_lya = h_cgs * nu_lya;

  // Sources of photons
  flag_incident_from_z = pin->GetOrAddBoolean("problem","incident_from_z",false);
  flag_incident_from_r = pin->GetOrAddBoolean("problem","incident_from_r",false);
  flag_finite_star = pin->GetOrAddBoolean("problem","finite_star",false);
  rstar = pin->GetOrAddReal("problem","rstar",7.e10)/l_cgs;
  lya_flux = pin->GetReal("problem","lya_flux");
  ion_flux = pin->GetReal("problem","ion_flux");

  // Set-up parmeters for ionizing photons
  numin = pin->GetReal("problem", "numin");
  flag_pow_law = pin->GetOrAddBoolean("problem","pow_law",false);
  // SWD: WARNING: mean_nu is used in SurfaceEmissivityIonizing even if flag_pow_law is false
  if (flag_pow_law) {
    Real numax = pin->GetReal("problem", "numax");
    Real powa = pin->GetReal("problem", "powa");
	  if (powa <= 2.) {
		  mean_nu = std::log(numax/numin) / (std::pow(numax, 1-powa) - std::pow(numin, 1-powa)) * (1-powa);
	  } else {
		  mean_nu = (std::pow(numax, 2-powa) - std::pow(numin, 2-powa)) * (1-powa)
			  / (std::pow(numax, 1-powa) - std::pow(numin, 1-powa)) / (2-powa);
	  }
    numinpow = std::pow(numin, 1.0-powa);
    numaxpow = std::pow(numax, 1.0-powa);
    nuexp = 1.0 / (1.0 - powa);
  }
	chromo_temp = pin->GetReal("problem", "chromo_temp");

  // Normal distribution for sampling initial photon frequencies
  // Using a linewdith (sigma) of 67 km/s
  linewidth = pin->GetReal("problem","linewidth");
  Real c_cgs = MCConstants::c_cgs;
  linewidth *= 1.0e5 / c_cgs * energy_lya;
  Real linewidth_cutoff = pin->GetReal("problem","linewidth_cutoff");
  linewidth_cutoff_energy = linewidth_cutoff * 1.0e5 / c_cgs * energy_lya;

  sigmamin = 6.3e-18; // absorption edge cross section

}

void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe, int etype) {
  // Function called each time a photon is initialized

  Real h_cgs = MCConstants::h_cgs;
  Real kb_cgs = MCConstants::kb_cgs;
  Real c_cgs = MCConstants::c_cgs;

  //printf("InitializePhoton: etype=%d\n", etype);
  EmissionType phot_type;
  EmissionGeometryHJ emis_geometry;
  // Set emission parameters based on etype
  if (etype == ion_str) {
    emis_geometry = SURFACE;
    phot_type = IONIZING;
    SetEmissionCellWeightArea(pphot,pmy_mc->emission_face[etype],ips,ipe);
  } else if (etype == lya_str) {
    emis_geometry = SURFACE;
    phot_type = LYA;
    SetEmissionCellWeightArea(pphot,pmy_mc->emission_face[etype],ips,ipe);
  } else if (etype == lya_rec) {
    emis_geometry = VOLUME;
    phot_type = LYA;
    SetEmissionCellWeight(pphot,ips,ipe);
  }

  // Loop over photon index
  for (int ip=ips; ip<=ipe; ip++) {
    pphot->type[ip] = etype;
    pphot->dtp[ip] = pphot->pmy_mcb->pmy_mc->tmax;

    int i = pphot->i1p[ip];
    int j = pphot->i2p[ip];
    int k = pphot->i3p[ip];

    // Choose photon position and direction based on the emission geometry
    switch (emis_geometry) {

      case VOLUME: {
        // Obtain initial position within zone
        GetZonePosition(pphot,pran,pcoord,ip);

        // Sample isotropic angle in orthonormal basis
        Real mu = 2.*pran->uniform()-1.0;
        Real stheta = std::sqrt(1.0-mu*mu);
        Real phi = 2.*PI*pran->uniform();
        pphot->k1p[ip] = stheta * std::cos(phi);
        pphot->k2p[ip] = stheta * std::sin(phi);
        pphot->k3p[ip] = mu;

        break;
      } // END CASE VOLUME

      case SURFACE: {

        // Set initial position on the outer radial surface of this zone
        // Use a rejection method for theta and phi within the zone's bounds

        Real phmax = pmy_block->pcoord->x3f(k+1);
        Real phmin = pmy_block->pcoord->x3f(k);
        Real thmax = pmy_block->pcoord->x2f(j+1);
        Real thmin = pmy_block->pcoord->x2f(j);
        Real sinphmax = std::sin(phmax);
        Real sinphmin = std::sin(phmin);
        Real sinthmax = std::sin(thmax);
        Real sinthmin = std::sin(thmin);

        if (flag_incident_from_z) {

          Real sthp = std::sin(thmax);
          Real sthm = std::sin(thmin);
          Real sth_samp = std::sqrt(pran->uniform()*(SQR(sthp) - SQR(sthm)) + SQR(sthm));
          Real th = std::asin(sth_samp);
          if (th < 0.0) {
            printf("WARNING: arcsin domain - sampled theta was %g\n", th);
            th += PI;
          }
          // initialize position
          pphot->x0p[ip] = 0.0;
          pphot->x1p[ip] = pmy_block->pmy_mesh->mesh_size.x1max;
          pphot->x2p[ip] = th;
          pphot->x3p[ip] = pran->uniform() * (phmax - phmin) + phmin;

          // check for nightside emission
          if (pphot->x2p[ip] > 0.5*PI) {
            pphot->PrintPhoton("Warning: Photon on nightside",ip);
            pphot->statp[ip] = DESTROYED;
          }

          // Set direction vector - parallel to star-planet separation vector.
          // k0p is the photon energy and is set from ep further below.
          pphot->k1p[ip] = -std::cos(th);
          pphot->k2p[ip] = std::sin(th);
          pphot->k3p[ip] = 0.0;

        } else if (flag_incident_from_r) {

          // uniform sample in cos(th) = mu
          Real mumin = std::cos(thmin);
          Real mumax = std::cos(thmax);
          Real mu = pran->uniform() * (mumax-mumin) + mumin;
          Real th = std::acos(mu);

          // uniform sample in ph
          Real ph = pran->uniform() * (phmax-phmin) + phmin;

          pphot->x0p[ip] = 0.0;
          pphot->x1p[ip] = pmy_block->pmy_mesh->mesh_size.x1max;
          pphot->x2p[ip] = th;
          pphot->x3p[ip] = ph;
          pphot->k0p[ip] = 1.;
          pphot->k1p[ip] = -1.;
          pphot->k2p[ip] = 0.0;
          pphot->k3p[ip] = 0.0;

        } else if (flag_finite_star) {
          // sampling from a finite star and planet
          Real rplanet = pmy_block->pmy_mesh->mesh_size.x1max;
          Real lsp_min2 = SQR(sep - rplanet - rstar);
          //printf("sep=%g, rplanet=%g, rstar=%g, lspmin=%g\n", sep, rplanet, rstar, std::sqrt(lsp_min2));

          bool reject = true;
          int niters = 0;
          Real th, ph, kr, kth, kph;
          while (reject) {
            niters++;
            if (niters % 100000 == 0) printf("[WARNING] niters=%d in InitializePhoton\n", niters);

            // uniform sampling in cos(theta) = mu within meshblock
            Real mumin = std::cos(thmin);
            Real mumax = std::cos(thmax);
            Real mup = pran->uniform() * (mumax-mumin) + mumin;
            Real thp = std::acos(mup);

            // uniform sampling in phi within meshblock
            Real php = pran->uniform() * (phmax-phmin) + phmin;

            // uniform sampling in cos(theta) = mu on the full star
            Real ths = std::acos(2.*pran->uniform() - 1.);

            // uniform sampling in phi on the full star
            Real phs = pran->uniform() * 2*PI;

            //printf("sampled thp=%g, php=%g, ths=%g, phs=%g\n", thp, php, ths, phs);

            // compute trig functions
            Real sinthp = std::sin(thp);
            Real costhp = std::cos(thp);
            Real sinphp = std::sin(php);
            Real cosphp = std::cos(php);
            Real sinths = std::sin(ths);
            Real cosths = std::cos(ths);
            Real sinphs = std::sin(phs);
            Real cosphs = std::cos(phs);

            // compute star-planet vector and length
            Real xsp = sep + rplanet*sinthp*cosphp - rstar*sinths*cosphs;
            Real ysp = rplanet*sinthp*sinphp - rstar*sinths*sinphs;
            Real zsp = rplanet*costhp - rstar*cosths;
            Real lsp2 = SQR(xsp) + SQR(ysp) + SQR(zsp);
            //printf("found xsp=%g, ysp=%g, zsp=%g, lsp=%g\n", xsp, ysp, zsp, std::sqrt(lsp2));

            // compute n dot e_r for planet and star
            // for speed we don't normalize until the end
            Real ndoterp = xsp*sinthp*cosphp + ysp*sinthp*sinphp + zsp*costhp;
            Real ndoters = xsp*sinths*cosphs + ysp*sinths*sinphs + zsp*cosths;
            //printf("found ndoterp=%g, ndoters=%g\n", ndoterp/std::sqrt(lsp2), ndoters/std::sqrt(lsp2));

            // ensure that we are entering the planet
            if (ndoterp > 0.0) continue;

            // ensure that we are leaving the star
            if (ndoters < 0.0) continue;

            // compute the comparison function
            // we normalize here with another factor of lsp2
            Real fcomp = lsp_min2 / SQR(lsp2) * ndoters * (-ndoterp);
            Real rcomp = pran->uniform();
            //printf("found fcomp=%g, rcomp=%g\n", fcomp, rcomp);
            if (rcomp > fcomp) continue;
            reject = false;

            // sampling finished; cleanup
            th = thp;
            ph = php;
            kr = ndoterp;
            kth = xsp*costhp*cosphp + ysp*costhp*sinphp - zsp*sinthp;
            kph = -xsp*sinphp + ysp*cosphp;

            // normalize to unit vectors
            Real lsp = std::sqrt(lsp2);
            kr /= lsp;
            kth /= lsp;
            kph /= lsp;

           // printf("niters=%d, theta=%g, phi=%g\n", niters, th, ph);

          } // end while

          // initialize position
          pphot->x0p[ip] = 0.0;
          pphot->x1p[ip] = rplanet;
          pphot->x2p[ip] = th;
          pphot->x3p[ip] = ph;

          // initialize direction
          pphot->k0p[ip] = 1.;
          pphot->k1p[ip] = kr;
          pphot->k2p[ip] = kth;
          pphot->k3p[ip] = kph;

        } else { // NOT INCIDENT_FROM_Z or R

         // Sampling functions for phi and theta within meshblock boundaries
         // Assumes star is in the -x direction (psi = 0)

          Real ph = PI - std::asin(pran->uniform() * (sinphmax-sinphmin) + sinphmin);
          Real prob0 = 0.25 * (2*(thmax-thmin) + std::sin(2*thmin) - std::sin(2*thmax));
          Real probmax;
          if (thmax <= PI/2) {
            probmax = SQR(sinthmax) / prob0;
          } else if (thmin >= PI/2) {
            probmax = SQR(sinthmin) / prob0;
          } else {
            probmax = 1/prob0;
          }

          bool reject = true;
          int step = 0;
          Real th;
          while (reject) {
            Real th_samp = pran->uniform() * (thmax-thmin) + thmin;
            Real sth_samp = std::sin(th_samp);
            Real th_prob = SQR(sth_samp)/prob0;
            if (pran->uniform() <= th_prob) {
              reject = false;
              th = th_samp;
            }
            step++;
          }

          // initialize position
          pphot->x0p[ip] = 0.0;
          pphot->x1p[ip] = pmy_block->pmy_mesh->mesh_size.x1max;
          pphot->x2p[ip] = th;
          pphot->x3p[ip] = ph;

          // check for nightside emission
          if ((ph < 0.5*PI) || (ph > 1.5*PI)) {
            pphot->PrintPhoton("Warning: Photon on nightside",ip);
            pphot->statp[ip] = DESTROYED;
          }

          Real sth = std::sin(th);
          Real cth = std::cos(th);
          Real sph = std::sin(ph);
          Real cph = std::cos(ph);

          // Set direction vector - parallel to star-planet separation vector.
          // NB: k0p aliases ep on this branch, so it is NOT set to 1 here; the
          // photon energy is assigned from ep further below.
          pphot->k1p[ip] = sth*cph;
          pphot->k2p[ip] = cth*cph;
          pphot->k3p[ip] = -sph;

         // Real cph = std::cos(ph);
				 // Real sph = std::sin(ph);
         // Real xhat = sth*cph*pphot->k1p[ip] + cth*cph*pphot->k2p[ip] - sph*pphot->k3p[ip];
         // Real yhat = sth*sph*pphot->k1p[ip] + cth*sph*pphot->k2p[ip] + cph*pphot->k3p[ip];
         // Real zhat = cth*pphot->k1p[ip] - sth*pphot->k2p[ip];
         // printf("photon x, y, z directions: %g, %g, %g\n", xhat, yhat, zhat);

        } // END IF INCIDENT_FROM_Z

        break;
      } // END CASE SURFACE

    } // END SWITCH EMIS_GEOMETRY

    // Set the photon's energy and absorption / scattering opacities
    switch (phot_type) {

      case LYA: {
        if (emis_geometry == SURFACE) {
          if (flag_sample_lya) { // sample from input spectrum
            Real rsample = pran->uniform();
            Real wl = SampleSpectrumWavelength(spectrum_lya_wl, spectrum_lya_cdf, rsample);

            pphot->ep[ip] = h_cgs * c_cgs / wl;
            //printf("rsample=%g, wl=%g, ep=%g\n", rsample, wl, pphot->ep[ip]/energy_lya-1.);

          } else { // gaussian distribution
            Real energy;
            gasdev(pmy_block, energy_lya,linewidth,energy);
            while (std::fabs(energy - energy_lya) > linewidth_cutoff_energy) {
              gasdev(pmy_block, energy_lya,linewidth,energy);
            }
            pphot->ep[ip] = energy;
          }
        } else {
          // Photon is emitted via recombination - should have line center energy
          pphot->ep[ip] = energy_lya;
        }
        break;
      }

      case IONIZING: {
        Real nu_phot;
        if (!flag_pow_law) {
          nu_phot = numin + kb_cgs * chromo_temp / h_cgs * std::log(1./(1-pran->uniform()));
        } else {
          nu_phot = pow(pran->uniform() * (numaxpow - numinpow) + numinpow, nuexp);
        }
        pphot->ep[ip] = h_cgs * nu_phot;

        break;
      }
    } // end switch

    // Set status flag
    if (pphot->wp[ip] <= 0.0) {
      pphot->statp[ip] = DESTROYED;
    } else if (flag_init_escaped) {
      pphot->statp[ip] = ESCAPED;
    } else {
      pphot->statp[ip] = EVOLVING;
    }

    // Initialize the absorption and scattering extinction coefficients
    // to the values in the emitted zone
    pphot->acp[ip] = AbsorptionOpacity(this,pphot,ip);
    pphot->scp[ip] = ScatteringOpacity(this,pphot,ip);

    // initialize scattering number
    pphot->nscp[ip] = 0;

    // initialize start time in ms
    const auto now = std::chrono::system_clock::now() - global_start_time;
    Real diff = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    pphot->user[1][ip] = diff;

    //pphot->PrintPhoton(ip);

  } // end loop over ip
}


void MonteCarloBlock::UserWorkAfterTransfer(int etype) {
  if (!flag_update_nh)
    return;

  // only update ionization after transfer of ionizing photons
  if (etype != ion_str)
    return;

  // Uniform test
  //return;

  Real dt = pmy_block->pmy_mesh->dt;
  // for checking ionization
  //dt = 1.e10;

  Real tint = pmy_mc->tint; // in cgs
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {

        Real norm = 1./ (tint * pcoord->vol(k,j,i)); // tint, vol in cgs
        // Get temperatures to calculate recombination and impact excitation terms
        Real tempo1e4K = tgas(k,j,i) / 1.e4;
        Real invtemp = 1/tempo1e4K;

        // Do an implicit update of the neutral fraction to solve for the ionization state at the end of this step
        Real rho = pmy_block->phydro->u(IDN,k,j,i); // SWD: Why not use MCBlock rho?

        Real nh = pmy_block->pscalars->s(0,k,j,i);
        Real np = rho - nh;
        Real na = nh + np;

        // Calculate the photoionization rate, Gamma, from the number of photons absorbed
        // per cell per time
        Real absweight = sourceterms(MCNABS,k,j,i) * norm;
        Real Gamma;
        if (nh <= 0.0) {
          std::cout << "WARNING: neutral density is <= 0 in UpdateIonizationFraction, setting Gamma to 0.0" << std::endl;
          Gamma = 0.0;
        } else {
          Gamma = absweight / nh * (time_cgs/n_cgs);
        }

        // Calculate the recombination rate, alpha, from the temperature of this cell
        Real alpha = 2.54e-13 * std::pow(tempo1e4K, -0.8164-0.0208*std::log(tempo1e4K));
        alpha *= n_cgs * time_cgs; // conver to cgs
        Real nR = 1. / alpha / dt;
        Real nC = Gamma / alpha;

        // Calculate the update to the neutral H number density
        //CMF: see numerical recipes 5.6
        Real bb = 2*na + nC + nR;
        Real cc = nh*nR + SQR(na);
        Real dd = SQR(bb) - 4*cc;
        Real qq = 0.5*(bb + std::sqrt(dd));
        Real update = cc/qq;

        // Double-check that the implicit update gives a neutral fraction between 0 and 1 --- NOT guaranteed if dt is large
        Real neutral_frac = update/na;
        if (neutral_frac > 1.0000001) {
          printf("[WARNING] (Block %d) UpdateIonizationFraction: neutral fraction %g is greater than 1\n", pmy_block->lid, neutral_frac);
          neutral_frac = 1.0;
        } else if (neutral_frac < 0.0) {
          printf("[WARNING] (Block %d) UpdateIonizationFraction: neutral fraction %g is less than 0\n", pmy_block->lid, neutral_frac);
          neutral_frac = 0.0;
        }

        // check result
        //if (absweight > 0.0) {
        //  printf("nh'=%g, na=%g, nh0=%g, alpha=%g, Gamma=%g, dt=%g\n", update, na, nh, alpha, Gamma, dt);
        //}

        nh = neutral_frac * na;
        np = na - nh;

        pmy_block->pscalars->s(0,k,j,i) = nh;

        // calculate impact excitation cooling
        // see Christie, Arras, Li, 2013, eq.8 and Table 2
        Real c1s2s = 1.21e-8 * std::pow(invtemp, 0.455) * std::exp(-11.84/tempo1e4K);
        Real c1s2p = 1.71e-8 * std::pow(invtemp, 0.077) * std::exp(-11.84/tempo1e4K);
        Real ctot = c1s2s + c1s2p;

        Real eimp = 10.2 * MCConstants::ev_to_erg;
        Real cool = ctot * nh * np * eimp * SQR(n_cgs);
        // Result should be in cgs units multiply vol*tint to offset normalization
        Real vol = pcoord->vol(k,j,i);
        sourceterms(MCRS0,k,j,i) -= cool * vol * tint;
        pmy_block->user_out_var(5,k,j,i) += cool;

      }
    }
  }
}


void MonteCarloBlock::FinalizePhoton(Photon *pphot, int ip) {

  // store scatter number
  pphot->user[0][ip] = pphot->nscp[ip];

  // store wall time elapsed since initialization
  Real start = pphot->user[1][ip];
  const auto now = std::chrono::system_clock::now() - global_start_time;
  Real diff = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
  Real time_elapsed = diff - start;
  //printf("time_elapsed=%g\n", time_elapsed);
  pphot->user[1][ip] = time_elapsed;

}


// Definitions for namespace functions
namespace {

void GetTidalAcceleration(Real r, Real th, Real ph, int order, Real &a_r, Real &a_th, Real &a_ph) {

  // point source
  a_r  = 0;
  a_th = 0;
  a_ph = 0;
  if (order < 2) {
    return;
  }

  Real cosph = std::cos(ph);
  Real sinph = std::sin(ph);
  Real costh = std::cos(th);
  Real sinth = std::sin(th);

  Real x = r * cosph * sinth;
  Real y = r * sinph * sinth;
  Real z = r * costh;

  // unit vector conversions
  Real unit_xr = sinth*cosph;
  Real unit_xt = costh*cosph;
  Real unit_xp = -sinph;

  Real unit_yr = sinth*sinph;
  Real unit_yt = costh*sinph;
  Real unit_yp = cosph;

  Real unit_zr = costh;
  Real unit_zt = -sinth;

  // Quadratic (Hill)
  Real prefac2 = gm_star / (sep*sep*sep);
  a_r  += prefac2 * (3*x*unit_xr - z*unit_zr);
  a_th += prefac2 * (3*x*unit_xt - z*unit_zt);
  a_ph += prefac2 * (3*x*unit_xp);
  if (order < 3) {
    return;
  }

  // Cubic
  Real prefac3 = 1.5 * prefac2 / sep;
  a_r  += prefac3 * ((r*r - 3*x*x)*unit_xr + 2*x*y*unit_yr + 2*x*z*unit_zr);
  a_th += prefac3 * ((r*r - 3*x*x)*unit_xt + 2*x*y*unit_yt + 2*x*z*unit_zt);
  a_ph += prefac3 * ((r*r - 3*x*x)*unit_xp + 2*x*y*unit_yp);
}


void TwoPointMass(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar) {

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real r = pmb->pcoord->x1v(i);
        Real th = pmb->pcoord->x2v(j);
        Real ph = pmb->pcoord->x3v(k);
        Real rho = prim(IDN,k,j,i);

				// Tidal gravity + Centrifugal
        Real a_r, a_th, a_ph;
        GetTidalAcceleration(r, th, ph, rho, a_r, a_th, a_ph);

        // Coriolis
        // Omega = (GM/a^3)(costh e_r - sinth e_th), a_cor = -2Omega cross vel
        Real omega = std::sqrt((gm_star+gm_planet)/(sep*sep*sep));
        Real sinth = std::sin(th);
        Real costh = std::cos(th);
        Real vel_r = prim(IVX,k,j,i);
        Real vel_th = prim(IVY,k,j,i);
        Real vel_ph = prim(IVZ,k,j,i);
        a_r += 2*omega * vel_ph*sinth;
        a_th += 2*omega * vel_ph*costh;
        a_ph -= 2*omega * (vel_r*sinth + vel_th*costh);

        Real src1 = a_r*rho*dt;
        Real src2 = a_th*rho*dt;
        Real src3 = a_ph*rho*dt;
        cons(IM1,k,j,i) += src1;
        cons(IM2,k,j,i) += src2;
        cons(IM3,k,j,i) += src3;

        // Update the user meshblock quantities
        pmb->ruser_meshblock_data[0](0,k,j,i) = a_r;
        pmb->ruser_meshblock_data[0](1,k,j,i) = a_th;
        pmb->ruser_meshblock_data[0](2,k,j,i) = a_ph;

        // Update conserved gas energy
        cons(IEN,k,j,i) += src1*prim(IVX,k,j,i)
                         + src2*prim(IVY,k,j,i)
                         + src3*prim(IVZ,k,j,i);
      }
    }
  }
}


/*
 * Second order Hill potential
 * Assumes the star is in the -x direction, i.e. psi = 0
 * Inputs and outputs in code units
 */
void HillTidalGravity(MeshBlock *pmb, const Real time, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
    const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
    AthenaArray<Real> &cons_scalar) {
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real rho = prim(IDN,k,j,i);
        Real r = pmb->pcoord->x1v(i);
        Real th = pmb->pcoord->x2v(j);
        Real ph = pmb->pcoord->x3v(k);
        Real vel_r = prim(IVX,k,j,i);
        Real vel_th = prim(IVY,k,j,i);
        Real vel_ph = prim(IVZ,k,j,i);
        Real cosph = std::cos(ph);
        Real sinph = std::sin(ph);
        Real costh = std::cos(th);
        Real sinth = std::sin(th);
        Real x = r * cosph * sinth;
        //Real y = r * sinph * sinth;
        Real z = r * costh;

        // Tidal gravity, second order
        Real prefac = gm_star / (sep*sep*sep);
        Real a_r = prefac * (3*x*sinth*cosph - z*costh);
        Real a_th = prefac * (3*x*costh*cosph + z*sinth);
        Real a_ph = prefac * (-3*x*sinph);

        // Coriolis, second order
        Real omega = std::sqrt((gm_star+gm_planet)/(sep*sep*sep));
        a_r  += 2*omega * vel_ph*sinth;
        a_th += 2*omega * vel_ph*costh;
        a_ph -= 2*omega * (vel_r*sinth + vel_th*costh);

        Real src1 = a_r *rho*dt;
        Real src2 = a_th*rho*dt;
        Real src3 = a_ph*rho*dt;
        cons(IM1,k,j,i) += src1;
        cons(IM2,k,j,i) += src2;
        cons(IM3,k,j,i) += src3;

        // Update the user meshblock quantities
        pmb->ruser_meshblock_data[0](0,k,j,i) = a_r;
        pmb->ruser_meshblock_data[0](1,k,j,i) = a_th;
        pmb->ruser_meshblock_data[0](2,k,j,i) = a_ph;

        // Update conserved gas energy
        cons(IEN,k,j,i) += src1*prim(IVX,k,j,i)
                         + src2*prim(IVY,k,j,i)
                         + src3*prim(IVZ,k,j,i);
      }
    }
  }
}


/*
 * Third order Hill potential
 * Assumes the star is in the -x direction, i.e. psi = 0
 * Inputs and outputs in code units
 */
void ThirdOrderTidalGravity(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
  const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
  AthenaArray<Real> &cons_scalar) {
	Real prefac2 = gm_star / (sep*sep*sep);
	Real prefac3 = 1.5 * prefac2 / sep;
	Real omega = std::sqrt((gm_star + gm_planet) / (sep*sep*sep));

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
		Real ph = pmb->pcoord->x3v(k);
		Real cosph = std::cos(ph);
		Real sinph = std::sin(ph);

    for (int j=pmb->js; j<=pmb->je; ++j) {
			Real th = pmb->pcoord->x2v(j);
			Real costh = std::cos(th);
			Real sinth = std::sin(th);

			// unit vector conversions
			Real unit_xr = sinth*cosph;
			Real unit_xt = costh*cosph;
			Real unit_xp = -sinph;

			Real unit_yr = sinth*sinph;
			Real unit_yt = costh*sinph;
			Real unit_yp = cosph;

			Real unit_zr = costh;
			Real unit_zt = -sinth;

      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real r = pmb->pcoord->x1v(i);
				Real rho    = prim(IDN,k,j,i);
				Real vel_r  = prim(IVX,k,j,i);
				Real vel_th = prim(IVY,k,j,i);
				Real vel_ph = prim(IVZ,k,j,i);
				Real x = r * sinth * cosph;
				Real y = r * sinth * sinph;
				Real z = r * costh;

				// Tidal gravity, second order
        Real a_r  = prefac2 * (3*x*unit_xr - z*unit_zr);
				Real a_th = prefac2 * (3*x*unit_xt - z*unit_zt);
				Real a_ph = prefac2 * (3*x*unit_xp);

				// Tidal gravity, third order
				a_r  += prefac3 * ((r*r - 3*x*x)*unit_xr + 2*x*y*unit_yr + 2*x*z*unit_zr);
				a_th += prefac3 * ((r*r - 3*x*x)*unit_xt + 2*x*y*unit_yt + 2*x*z*unit_zt);
				a_ph += prefac3 * ((r*r - 3*x*x)*unit_xp + 2*x*y*unit_yp);

				// Coriolis
				a_r  += 2*omega * vel_ph*sinth;
				a_th += 2*omega * vel_ph*costh;
				a_ph -= 2*omega * (vel_r*sinth + vel_th*costh);

        Real src1 = a_r  * rho * dt;
        Real src2 = a_th * rho * dt;
        Real src3 = a_ph * rho * dt;
        cons(IM1,k,j,i) += src1;
        cons(IM2,k,j,i) += src2;
        cons(IM3,k,j,i) += src3;

        // Update the user meshblock quantities
        pmb->ruser_meshblock_data[0](0,k,j,i) = a_r;
        pmb->ruser_meshblock_data[0](1,k,j,i) = a_th;
        pmb->ruser_meshblock_data[0](2,k,j,i) = a_ph;

        // Update conserved gas energy
        cons(IEN,k,j,i) += src1*prim(IVX,k,j,i)
                         + src2*prim(IVY,k,j,i)
                         + src3*prim(IVZ,k,j,i);
      }
    }
  }
}


/*
 * Hydrostatic, isothermal density initial condition
 * Use tidal_order to set order of tidal source expansion
 * Inputs in code units
 * Outputs a dimensionless number to multiply by the substellar point base density
 */
Real HydrostaticIsothermalDensity(Real r, Real th, Real ph, Real r0, Real a2, int tidal_order) {
  Real roa  = r  / sep;
  Real r0oa = r0 / sep;
  Real xoa = roa * std::sin(th) * std::cos(ph);
  Real zoa = roa * std::cos(th);
  Real prefac = gm_star / sep / a2;
  Real mfrac = gm_planet / gm_star;

  // Point source
  Real U0 = -mfrac / r0oa;
  Real U  = -mfrac / roa;

  // Second order (Hill)
  if (tidal_order > 1) {
    U0 -= 1.5 * SQR(r0oa);
    U  -= 0.5 * (3*SQR(xoa) - SQR(zoa));
  }

  // Third order
  if (tidal_order > 2) {
    U0 -= r0oa*r0oa*r0oa;
    U  -= 0.5 * (3*SQR(roa)*xoa - 5*xoa*xoa*xoa);
  }

  return std::exp(prefac * (U0 - U));
}


/*
 * Hydrostatic, adiabatic density initial condition
 * Use tidal_order to set order of tidal source expansion
 * Inputs in code units
 * Outputs a dimensionless number to multiply by the substellar point base density
 */
Real HydrostaticAdiabaticDensity(Real r, Real th, Real ph, Real r0, Real a2, Real invgm1, int tidal_order) {
  Real roa  = r  / sep;
  Real r0oa = r0 / sep;
  Real xoa = roa * std::sin(th) * std::cos(ph);
  Real zoa = roa * std::cos(th);
  Real prefac = gm_star / sep / a2 / invgm1;
  Real mfrac = gm_planet / gm_star;

  // Point source
  Real U0 = -mfrac / r0oa;
  Real U  = -mfrac / roa;

  // Second order (Hill)
  if (tidal_order > 1) {
    U0 -= 1.5 * SQR(r0oa);
    U  -= 0.5 * (3*SQR(xoa) - SQR(zoa));
  }

  // Third order
  if (tidal_order > 2) {
    U0 -= r0oa*r0oa*r0oa;
    U  -= 0.5 * (3*SQR(roa)*xoa - 5*xoa*xoa*xoa);
  }

  // Correction for critical value
  Real critical = -prefac * (U0 - U);
  if (critical >= 1.) {
    return 0.0;
  }
  return std::pow(1. - critical, invgm1);
}


/*
 * Optically Thin, Hydrostatic Ionization Fraction
 * Assumes optically thin ionization rate and constant recombination
*/
Real HydrostaticIonization(Real n_a, Real n_c) {
  Real naonc = n_a / n_c;
  Real ion_factor = 0.5*(std::sqrt(1. + 4*naonc) - 1.);
  Real nhonc = naonc - ion_factor;
  return nhonc * n_c;
}


Real VolumeEmissivityLya(MonteCarloBlock *pmcb, int k, int j, int i, int etype) {
// Sets the value of the emission array which determines where Lya photons are
// likely to be emitted. The associated cooling of the gas is handled in
// UpdateSourceTerms, not here - this is just where the emissivity is calculated

  Real ntot = pmcb->rho(k,j,i) / MCConstants::mp_cgs;
  Real tempo1e4K = pmcb->tgas(k,j,i) / 1.0e4;
  Real invtemp = 1.0/tempo1e4K;

  // LYA COOLING TERMS
  // c.f. Christie Arras & Li (2013), Table 2
  // ----------------------------------------

  // Recombination rate as a function of temperature
  // Ionized hydrogen and free electrons combine to produce Lya
  // Energy per volume per time = 10.2 eV * alpha * np * ne
  Real alpha = 2.54e-13 * std::pow(tempo1e4K, -0.8164-0.0208*std::log(tempo1e4K));

  // Electron impact excitation rate as a function of temperature
  // We presume thermal electrons excite neutral H(1s) to the n=2 state (either 2s or 2p)
  // which then immediately de-excite, producing Lya
  // Energy per volume per time = 10.2 eV * ctot * nH * ne
  Real c1s2s = 1.21e-8 * std::pow(invtemp, 0.455) * std::exp(-11.84/tempo1e4K);
  Real c1s2p = 1.71e-8 * std::pow(invtemp, 0.077) * std::exp(-11.84/tempo1e4K);
  Real ctot = c1s2s + c1s2p;

  // Number density of neutral H, protons
	//Real nH = pmcb->pmy_block->pscalars->s(0,k,j,i);
  Real nH = pmcb->species(0,k,j,i);
	Real np = ntot - nH;

  Real recombination = alpha*SQR(np);
  Real impact = ctot*nH*np;
  Real emis = recombination + impact;
  //printf("rho=%g, ntot=%g, nH=%g, np=%g, recomb=%g, impact=%g, emis=%g\n",rho, ntot, nH, np, recombination, impact, emis);

  pmcb->pmy_block->user_out_var(0,k,j,i) = emis;

  return emis;
}


Real SurfaceEmissivityLya(MonteCarloBlock *pmcb, int k, int j, int i, int etype) {

  // number of photons intercepted per time through cell
  Real ndot; // number per time
  if (flag_finite_star) {
    Real nintens; // number / area / time / solid angle
    if (flag_sample_lya) {
      nintens = spectrum_lya_itot / spectrum_lya_energy; // itot from integral over spectrum
    } else {
      nintens = lya_flux / PI * SQR(sep / rstar) / energy_lya; // convert flux to intensity
    }
    Real area_omega = GetProjectedAreaOmegaFiniteStar(pmcb, k, j, i); // projected area on planet * solid angle on star
    ndot = nintens * area_omega;

  } else { // projections along coordinate directions
    Real nflux0; // number / area / time
    if (flag_sample_lya) {
      nflux0 = spectrum_lya_itot * PI * SQR(rstar / sep) / spectrum_lya_energy;
    } else {
      nflux0 = lya_flux / energy_lya;
    }
    Real projected_area = GetProjectedArea(pmcb, k, j, i); // projected area on planet
    ndot = nflux0 * projected_area;
  }

  // normalize to the area of the cell for a uniform flux
  Real nflux = ndot / pmcb->pmy_block->pcoord->GetFace1Area(k,j,i+1); // number / perp area / time
  pmcb->pmy_block->user_out_var(1,k,j,i) = ndot;
  return nflux;
}


Real SurfaceEmissivityIonizing(MonteCarloBlock *pmcb, int k, int j, int i, int etype) {
  Real energy_euv = MCConstants::h_cgs * mean_nu;

  // number of photons intercepted per time through cell
  Real ndot; // number per time
  if (flag_finite_star) {
    Real nintens = ion_flux / PI * SQR(sep / rstar) / energy_euv; // number / area / time / solid angle
    Real area_omega = GetProjectedAreaOmegaFiniteStar(pmcb, k, j, i); // projected area on planet * solid angle on star
    ndot = nintens * area_omega;

  } else { // projections along coordinate directions
    Real nflux0 = ion_flux / energy_euv; // number / area / time
    Real projected_area = GetProjectedArea(pmcb, k, j, i); // projected area on planet
    ndot = nflux0 * projected_area;
  }

  // normalize to the area of the cell for a uniform flux
  Real nflux = ndot / pmcb->pmy_block->pcoord->GetFace1Area(k,j,i+1); // number / perp area / time
  pmcb->pmy_block->user_out_var(1,k,j,i) = ndot;
  return nflux;
}


/*
 * Returns the projected area against an incident photon direction at a given cell face
 * returns zero if the cell is not on the surface
*/
Real GetProjectedArea(MonteCarloBlock *pmcb, int k, int j, int i) {
  Coordinates *pco = pmcb->pmy_block->pcoord;
  Real proj_area = 0.0;

  // ensure cell is not inside the volume
  if (pco->x1f(i+1) >= rout) {

    if (flag_incident_from_z) {
      // ensure cell is on the dayside
      if (pco->x2v(j) < PI/2.) {
        Real cthm = std::cos(pco->x2f(j));
        Real cthp = std::cos(pco->x2f(j+1));
        Real r2 = pco->x1f(i+1) * pco->x1f(i+1);
        Real domega = ( pco->x3f(k+1) - pco->x3f(k) ) * ( SQR(cthm) - SQR(cthp) );
        proj_area = 0.5*r2 * domega;
      }

    } else if (flag_incident_from_r) {
      Real cthm = std::cos(pco->x2f(j));
      Real cthp = std::cos(pco->x2f(j+1));
      Real r2 = pco->x1f(i+1) * pco->x1f(i+1);
      Real domega = ( pco->x3f(k+1) - pco->x3f(k) ) * ( cthm - cthp );
      proj_area = 0.5*r2 * domega;

    } else { // default incident from x
      Real phm = pco->x3f(k);
      Real php = pco->x3f(k+1);
      // ensure cell is on the dayside
      if ((phm > PI/2.) && (php < 3.*PI/2.)) {
        Real thm  = pco->x2f(j);
        Real thp  = pco->x2f(j+1);
        Real sthm = std::sin(thm);
        Real cthm = std::cos(thm);
        Real sthp = std::sin(thp);
        Real cthp = std::cos(thp);
        Real sphm = std::sin(phm);
        Real sphp = std::sin(php);
        Real r2   = pco->x1f(i+1) * pco->x1f(i+1);
        Real domega = -( thp - thm - (std::sin(thp - thm) * std::cos(thp + thm)) ) * ( sphp - sphm );
        proj_area = 0.5*r2 * domega;
      }
    }
  }

  return proj_area;
}


/*
 * Returns projected area*dOmega for finite star emissivity calculation
*/
Real GetProjectedAreaOmegaFiniteStar(MonteCarloBlock *pmcb, int k, int j, int i) {
  Coordinates *pco = pmcb->pmy_block->pcoord;

  // ensure cell is not inside the volume
  if (pco->x1f(i+1) < rout) return 0.;

  // integrate over stellar surface using same grid of theta, phi
  MeshBlock *pmb = pmcb->pmy_block;
  Real rplanet = pmb->pmy_mesh->mesh_size.x1max;
  Real php = pco->x3v(k);
  //Real thp = pco->x2v(j);
  Real mumin = std::cos(pco->x2f(j));
  Real mumax = std::cos(pco->x2f(j+1));
  Real mup = 0.5 * (mumin + mumax);
  Real dmup = mumin - mumax;
  Real thp = std::acos(mup);
  Real sinphp = std::sin(php);
  Real cosphp = std::cos(php);
  Real sinthp = std::sin(thp);
  Real costhp = std::cos(thp);
  Real dphp = pco->x3f(k+1) - pco->x3f(k);
  Real dthp = pco->x2f(j+1) - pco->x2f(j);
  //Real dAp = SQR(rplanet) * dphp * sinthp * dthp;
  Real dAp = SQR(rplanet) * dphp * dmup;

  int nk = pmb->pmy_mesh->mesh_size.nx3;
  int nj = pmb->pmy_mesh->mesh_size.nx2;
  Real dphs = 2.*PI / nk;
  Real dmus = 2. / nj;
  Real dAs = SQR(rstar) * dphs * dmus;

  Real area_omega = 0.0;
  for (int ks=0; ks<nk; ks++) {
    Real phs = dphs * (ks + 0.5);
    Real sinphs = std::sin(phs);
    Real cosphs = std::cos(phs);

    for (int js=0; js<nj; js++) {
      Real mus = 1. - dmus * (js + 0.5);
      Real ths = std::acos(mus);
      Real sinths = std::sin(ths);
      Real cosths = std::cos(ths);
      //printf("dmus=%g, mus=%g, ths=%g, dAs=%g\n", dmus, mus, ths, dAs);

      // compute star-planet vector and length
      Real xsp = sep + rplanet*sinthp*cosphp - rstar*sinths*cosphs;
      Real ysp = rplanet*sinthp*sinphp - rstar*sinths*sinphs;
      Real zsp = rplanet*costhp - rstar*cosths;
      Real lsp2 = SQR(xsp) + SQR(ysp) + SQR(zsp);
      //printf("found xsp=%g, ysp=%g, zsp=%g, lsp=%g\n", xsp, ysp, zsp, std::sqrt(lsp2));

      // compute n dot e_r for planet and star
      // for speed we don't normalize until the end
      Real ndoterp = xsp*sinthp*cosphp + ysp*sinthp*sinphp + zsp*costhp;
      Real ndoters = xsp*sinths*cosphs + ysp*sinths*sinphs + zsp*cosths;
      //printf("found ndoterp=%g, ndoters=%g\n", ndoterp/std::sqrt(lsp2), ndoters/std::sqrt(lsp2));

      // ensure that we are entering the planet
      if (ndoterp > 0.0) continue;

      // ensure that we are leaving the star
      if (ndoters < 0.0) continue;

      // add contribution from stellar surface
      area_omega += ndoters * (-ndoterp) / SQR(lsp2); // solid angle / area

    } // end for js
  } // end for ks

  area_omega *= dAs * dAp; // area * solid angle

  return area_omega;
}


// SWD: should be able to remove this
void ResonantScattering(MonteCarloBlock *pmcb, Photon *pphot, int ips, int ipe) {

  //for (int ip=ips; ip<=ipe; ip++) {
  //  pphot->statp[ip] = ESCAPED;
  //}
  //return;

  if (pphot->type[ips] != ion_str) {
    ScatterResonanceLine(pmcb, pphot, ips, ipe);
  } else {
    std::cout << "Warning: attempted to scatter ionizing photon\n";
  }
}


bool IsLyaPhot(Photon *pphot, int ip) {
  Real erg_away_from_lc = std::fabs(pphot->ep[ip] - energy_lya);
  return erg_away_from_lc <= linewidth_cutoff_energy;
}


bool IsEUVPhot(Photon *pphot, int ip) {
  Real emin = MCConstants::h_cgs * numin;
  return pphot->ep[ip] >= emin;
}


Real BoundFreeAbsorptionOpacity(MonteCarloBlock *pmcb, Photon *pphot, int ip) {

  Real h_cgs = MCConstants::h_cgs;
  Real emin = h_cgs * numin;
  Real opac = 0.;
  //if (pphot->ep[ip] >= emin) {
  if (IsEUVPhot(pphot, ip)) {
    int i1 = pphot->i1p[ip];
    int i2 = pphot->i2p[ip];
    int i3 = pphot->i3p[ip];
    Real erat = pphot->ep[ip] / emin;
    Real xsec = sigmamin * std::pow(erat, -3.0);
    Real epsilon = std::sqrt(erat - 1);
    Real correction = std::exp(4 - 4*std::atan(epsilon)/epsilon) / std::expm1(-2*PI / epsilon);
    xsec *= -correction / erat;
    //Real nH = pmcb->pmy_block->pscalars->s(0,i3,i2,i1) * n_cgs;
    Real nH = pmcb->species(0,i3,i2,i1);
    opac = xsec * nH; // opacities in cgs units
  }

  return opac;
}


Real ResonantScatteringOpacity(MonteCarloBlock *pmcb, Photon *pphot, int ip) {

  Real opac = 0.;
  //Real erg_away_from_lc = std::fabs(pphot->ep[ip] - energy_lya);
  //if (erg_away_from_lc <= linewidth_cutoff_energy) {
  if (IsLyaPhot(pphot, ip)) {
    opac = ResonanceLineOpacity(pmcb, pphot, ip);
  }
  return opac;
}


Real LyaPureAbsOpacity(MonteCarloBlock *pmcb, Photon *pphot, int ip) {
  Real opac = 0.;
  if (IsLyaPhot(pphot, ip)) {
    int i1 = pphot->i1p[ip];
    int i2 = pphot->i2p[ip];
    int i3 = pphot->i3p[ip];
    Real nH = pmcb->species(0,i3,i2,i1);
    Real nH_n2 = nH * 1.e-7; // simplification of Huang et al. 2017
    Real xsec = 1.e-20; // simplification of bound-free n=2
    opac = nH_n2 * xsec; 
  }
  return opac;
}

/*
 * Choose between bound-free or resonant absorption opacity functions
*/
Real ChooseAbsorptionOpacity(MonteCarloBlock *pmcb, Photon *pphot, int ip) {
  Real h_cgs = MCConstants::h_cgs;
  Real opac = 0;
  //if (pphot->ep[ip] < numin * h_cgs) {
  if (IsEUVPhot(pphot, ip)) {
    opac = BoundFreeAbsorptionOpacity(pmcb, pphot, ip);
  }
  if (IsLyaPhot(pphot, ip)) {
    opac = LyaPureAbsOpacity(pmcb, pphot, ip);
    //opac = ResonantScatteringOpacity(pmcb, pphot, ip);
  }
  return opac;
}


//! \fn void ReadSpectrumToCDF(...)
//! \brief Read two-column spectrum data from an ASCII file
//!   - Column 0: wavelength [cm]
//!   - Column 1: intensity [erg/cm^2/s/cm]
//! Modifies wavelength 'wl' and cumulative distribution function 'cdf' as vectors.
//! Also accumulates total intensity 'itot' and average energy 'emean'.

void ReadSpectrumToCDF(const std::string& filename, std::vector<Real>& wl,
                       std::vector<Real>& cdf, Real& itot, Real& emean) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("ReadSpectrumToCDF: cannot open file " + filename);
  }

  // clear any values that might already be in wl and cdf
  wl.clear();
  cdf.clear();

  std::vector<Real> ilam;
  std::string line;
  int line_number = 0;
  while (std::getline(file, line)) {
    ++line_number;

    // skip empty lines and comments
    if (line.empty()) continue;
    if (line[0] == '#') continue;

    std::istringstream iss(line);
    Real wavelength, intensity;

    // check for lines that don't read nicely into two doubles
    if (!(iss >> wavelength >> intensity)) {
      throw std::runtime_error("ReadSpectrumToCDF: malformed line " + std::to_string(line_number) + " in input file '" + filename + "'");
    }

    wl.push_back(wavelength);
    ilam.push_back(intensity);

  }
  file.close();

  const int nrows = wl.size();
  if (nrows == 0) {
    throw std::runtime_error("ReadSpectrumToCDF: no valid data in file " + filename);
  }
  if (nrows < 2) {
    throw std::runtime_error("ReadSpectrumToCDF: need at least 2 data points for interpolation, found " + std::to_string(nrows));
  }

  // PDF = ilam / (int dlambda*ilam) = ilam / itot
  // compute CDF[i] as int dlambda*ilam from wl[0] to wl[i]
  // compute mean energy as int dlambda * (ilam/itot) * (h*c/lambda)
  // assumes wavelengths are uniformly spaced and increasing
  Real dlambda = wl[1] - wl[0];
  cdf.push_back(0.);
  emean = 0.;

  // integrate using trapezoid rule
  for (int i = 1; i < nrows; ++i) {
    cdf.push_back(cdf[i-1] + 0.5*dlambda*(ilam[i-1] + ilam[i]));
    emean += 0.5*dlambda*(ilam[i-1]/wl[i-1] + ilam[i]/wl[i]);
  }
  emean *= MCConstants::h_cgs * MCConstants::c_cgs;

  // normalize: CDF runs from 0 to 1
  itot = cdf[nrows-1];
  if (itot == 0.) {
    throw std::runtime_error("ReadSpectrumToCDF: cdf norm is zero");
  }
  for (int i=0; i<nrows; ++i) {
    cdf[i] = cdf[i] / itot;
  }
  emean /= itot;

} // end ReadSpectrumToCDF


// \fn Real SampleSpectrumWavelength(...)
// \brief Inverse-transform sampling of the spectrum built by ReadSpectrumToCDF
//
// Returns wl(u) by piecewise-linear interpolation of the tabulated wl against cdf,
// for a uniform deviate u in [0,1). The cdf is non-decreasing by construction, so the
// containing bin is found by binary search. A bin of zero width (cdf flat across a
// stretch of zero intensity) contributes zero probability and is collapsed to its
// left edge rather than dividing by zero.

Real SampleSpectrumWavelength(const std::vector<Real>& wl, const std::vector<Real>& cdf,
                              Real u) {
  const int nrows = cdf.size();

  // first index with cdf > u; the containing bin is the one to its left
  int ihi = std::upper_bound(cdf.begin(), cdf.end(), u) - cdf.begin();

  // clamp so that [ilo, ilo+1] is always a valid bin, which also handles u outside
  // the tabulated range by extrapolating from the end bins
  int ilo = ihi - 1;
  if (ilo < 0) ilo = 0;
  if (ilo > nrows-2) ilo = nrows-2;

  Real dcdf = cdf[ilo+1] - cdf[ilo];
  Real frac = (dcdf > 0.0) ? (u - cdf[ilo]) / dcdf : 0.0;
  return wl[ilo] + frac * (wl[ilo+1] - wl[ilo]);

}


// SWD: redo this function?
void gasdev(MeshBlock *pmb, Real mean, Real sigma, Real &samp) {

  if (iset == 0) {
    Real v1, v2, rsq;
    do {
      v1 = 2. * pmb->pmy_mcb->pran->uniform() - 1.;
      v2 = 2. * pmb->pmy_mcb->pran->uniform() - 1.;
      rsq = v1 * v1 + v2 * v2;
    } while (rsq >= 1.0 || rsq == 0.0);

    Real fac = std::sqrt(-2. * std::log(rsq) / rsq);
    gset = v1 * fac * sigma + mean;
    samp = v2 * fac * sigma + mean;
    iset = 1;
  } else {
    samp = gset;
    iset = 0;
  }
  return;
}


void ForceEscape(MonteCarloBlock *pmcb, Photon *pphot, int ips, int ipe) {
  for (int ip=ips; ip<=ipe; ip++) {
    pphot->statp[ip] = ESCAPED;
  }
}


Real GetIsowindVelocity(Real x) {

  Real y = (x <= 1.0) ? std::exp(1.5-2.0/x) : 1.1;
  int l = 0;
  while (l<1000) {
    Real f = 0.5*y*y - std::log(y*x*x) - 2.0/x + 1.5;
    Real dfdy = y - 1./y;
    if (std::abs(f) < 1e-10)
      break;
    y -= f / dfdy;
    l++;
  }
  return y;
}

Real ConstantTimestep(MeshBlock *pmb) {
  if (user_dt > TINY_NUMBER) {
    return user_dt;
  } else {
    return std::numeric_limits<Real>().max();
  }
}

void GetIonizationTemperature(MonteCarloBlock *pmcb) {

  Hydro* phydro = pmcb->pmy_block->phydro;
  Real kb_cgs = MCConstants::kb_cgs;
  Real mp_cgs = MCConstants::mp_cgs;

  int il = pmcb->is; int iu = pmcb->ie;
  int jl = pmcb->js; int ju = pmcb->je;
  int kl = pmcb->ks; int ku = pmcb->ke;

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
				Real rho = phydro->u(IDN,k,j,i);
				Real mmw = 1./(2. - pmcb->pmy_block->pscalars->r(0,k,j,i));
        Real tgas = phydro->w(IPR,k,j,i) * mmw * mp_cgs / rho / kb_cgs * SQR(pmcb->vel_cgs);

        // apply temperature floor
        pmcb->tgas(k,j,i) = (tgas > pmcb->tfloor_cgs) ? tgas : pmcb->tfloor_cgs;
      }
    }
  }
}

void GetNH(MonteCarloBlock *pmcb) {

  for (int k=pmcb->ks; k<=pmcb->ke; ++k) {
    for (int j=pmcb->js; j<=pmcb->je; ++j) {
      for (int i=pmcb->is; i<=pmcb->ie; ++i) {
        pmcb->species(0,k,j,i) = pmcb->pmy_block->pscalars->s(0,k,j,i) * n_cgs; // nH
      }
    }
  }
}


void CoreSkipping(MonteCarloBlock *pmcb, Photon *pphot, PhotonPusher *ppusher, int ip) {
  MCCoord *pco = pmcb->pcoord;
  Real l_cgs = pmcb->l_cgs;
  //Real nu_lya = MCConstants::nu_lya;
  //Real h_cgs = MCConstants::h_cgs;
  Real kb_cgs = MCConstants::kb_cgs;
  Real mp_cgs = MCConstants::mp_cgs;
  Real c_cgs = MCConstants::c_cgs;
  //Real v_th = 1.e6; // need to calculate in cell
  int i1 = pphot->i1p[ip];
  int i2 = pphot->i2p[ip];
  int i3 = pphot->i3p[ip];
  Real v_th = std::sqrt(kb_cgs/mp_cgs * pmcb->tgas(i3,i2,i1));
  Real x = c_cgs/v_th * (pphot->ep[ip]/energy_lya - 1.0);

  Real x_crit = 3.3; // core-wing boundary, see McClellan et al. 2022
  Real e_crit = energy_lya * (1.0 + std::copysign(x_crit, x) * v_th / c_cgs);

  // core skip based on photon frequency
  //if ((x < x_crit) && (x > -x_crit)) {

  Real dist = pco->dmin(i3,i2,i1) * l_cgs;
  //Real dist = 0.0;
  Real taumin = 1000.;
  Real tauphot = dist * pphot->scp[ip];
  // core skip based on comoving optical depth
  if ((x < x_crit) && (x > -x_crit) && (tauphot > taumin)) {

    //printf("core skip: ip=%d, nscat=%d, temp=%g, v_th=%g, E=%g, E_crit=%g, x=%g\n", ip, pphot->nscp[ip], pmcb->tgas(i3,i2,i1), v_th, pphot->ep[ip], e_crit, x);
    pphot->ep[ip] = e_crit;
    pphot->acp[ip] = pmcb->AbsorptionOpacity(pmcb,pphot,ip);
    pphot->scp[ip] = pmcb->ScatteringOpacity(pmcb,pphot,ip);
  } else {
    //printf("no core skip: ip=%d, nscat=%d, temp=%g, v_th=%g, E=%g, E_crit=%g, x=%g\n", ip, pphot->nscp[ip], pmcb->tgas(i3,i2,i1), v_th, pphot->ep[ip], e_crit, x);
  }

}


void UpdateSourceTerms(MonteCarloBlock *pmcb, Photon *pphot, Real energy0, Real weight0,
                  Real k1p0, Real k2p0, Real k3p0, int ip) {

  // if continuous absorption, handle source terms in UpdateMoments()
  if (pmcb->pmy_mc->absorption_method[pphot->type[ip]] == ABSTAU)
    return;


  // Update sourceterms for ionizing radiation
  if (energy0 > MCConstants::energy_ly_edge) {
    Real heat = weight0 * (energy0 - MCConstants::energy_ly_edge);
    //printf("energy0, weight0: %g %g %g\n", energy0, weight0);
    if ((std::isinf(heat)) || (std::isnan(heat))) {
      std::cout << "Warning: UpdateSourceTerms heating is : " << heat << std::endl;
      pphot->PrintPhoton(ip);
      pphot->statp[ip] = DESTROYED;
    } else if (heat >= TINY_NUMBER) {
      int &i = pphot->i1p[ip];
      int &j = pphot->i2p[ip];
      int &k = pphot->i3p[ip];
      pmcb->sourceterms(MCRS0,k,j,i) += heat;
      pmcb->sourceterms(MCNABS,k,j,i) += weight0;
    }
  }

  // update momentum source terms as usual
  Real k1 = pphot->k1p[ip];
  Real k2 = pphot->k2p[ip];
  Real k3 = pphot->k3p[ip];

  // Normalize k vector if using general mover in spherical polar coords
  if ((COORDINATE_SYSTEM == "spherical_polar") && (pphot->general_pusher_flag)) {
    k2 *= pphot->x1p[ip];
    k3 *= pphot->x1p[ip] * sin(pphot->x2p[ip]);
    k2p0 *= pphot->x1p[ip];
    k3p0 *= pphot->x1p[ip] * sin(pphot->x2p[ip]);
  }
  Real norm0 = sqrt(SQR(k1p0) + SQR(k2p0) + SQR(k3p0));
  if ((fabs(norm0-1.) > 1.0e-8) && (norm0 > 1.0e-8)) {
    k1p0 /= norm0;
    k2p0 /= norm0;
    k3p0 /= norm0;
  }
  Real norm = sqrt(SQR(k1) + SQR(k2) + SQR(k3));
  if ((fabs(norm-1.) > 1.0e-8) && (norm > 1.0e-8)) {
    k1 /= norm;
    k2 /= norm;
    k3 /= norm;
  }

  Real c_cgs = MCConstants::c_cgs;
  // Components of momentum change --- assumes orthonormal basis
  Real dp1p = (pphot->wp[ip] * k1 * pphot->ep[ip]
               - weight0 * k1p0 * energy0) / c_cgs;
  Real dp2p = (pphot->wp[ip] * k2 * pphot->ep[ip]
               - weight0 * k2p0 * energy0) / c_cgs;
  Real dp3p = (pphot->wp[ip] * k3 * pphot->ep[ip]
               - weight0 * k3p0 * energy0) / c_cgs;

  Real cool = (pphot->wp[ip] * pphot->ep[ip]) - (weight0 * energy0);

  if ((std::isinf(cool)) || (std::isnan(cool))) {
    std::cout << "Warning: UpdateSourceTerms cooling is : " << cool << std::endl;
    pphot->PrintPhoton(ip);
    pphot->statp[ip] = DESTROYED;
  } else if ((std::isinf(dp1p)) || (std::isnan(dp1p))) {
    std::cout << "Warning: UpdateSourceTerms momentum change (k1p) is : "
              << dp1p << std::endl;
    pphot->PrintPhoton(ip);
  } else if ((std::isinf(dp2p)) || (std::isnan(dp2p))) {
    std::cout << "Warning: UpdateSourceTerms momentum change (k2p) is : "
              << dp2p << std::endl;
    pphot->PrintPhoton(ip);
  } else if ((std::isinf(dp3p)) || (std::isnan(dp3p))) {
    std::cout << "Warning: UpdateSourceTerms momentum change (k3p) is : "
              << dp3p << std::endl;
    pphot->PrintPhoton(ip);
    pphot->statp[ip] = DESTROYED;
    std::cout << "Warning: UpdateCooling cooling is : " << cool << std::endl;
  } else {
    int &i = pphot->i1p[ip];
    int &j = pphot->i2p[ip];
    int &k = pphot->i3p[ip];
    //sourceterms(MCRS0,k,j,i) -= cool; // BCM: We do not want the photons to cool here - handle cooling separately via explicit source terms
    pmcb->sourceterms(MCRS1,k,j,i) -= dp1p;
    pmcb->sourceterms(MCRS2,k,j,i) -= dp2p;
    pmcb->sourceterms(MCRS3,k,j,i) -= dp3p;
  }
}

// Boundary Conditions

/*
 * Bottom (r_inner) boundary condition: static inflow diode
 * If radial velocity is positive (outwards), copy it
 * If radial velocity is negative (inwards), set it to zero
 * Copy any theta and phi velocities
 * Density and pressure are fixed to initial values
*/
void StaticInflowInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
  FaceField &b, Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku,
  int ngh) {
  Real gamma = pmb->peos->GetGamma();
  Real invgm1 = 1/(gamma-1);

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {

        Real rad = pco->x1v(il-i);
        Real the = pco->x2v(j);
        Real phi = pco->x3v(k);
        //Real hydrostatic_profile = HydrostaticIsothermalDensity(r, th, ph, rin, a2, flag_tidal_gravity+1);
        Real hydrostatic_profile = HydrostaticAdiabaticDensity(rad, the, phi, rin, a2, invgm1, flag_tidal_gravity+1);
        Real rho = nbase * hydrostatic_profile;
        Real P = a2 * nbase / gamma * std::pow(hydrostatic_profile, gamma);

        // fix initial density and pressure
        prim(IDN,k,j,il-i) = rho;
        prim(IPR,k,j,il-i) = P;

        // outflow diode for velocity
        prim(IVY,k,j,il-i) = prim(IVY,k,j,il);
        prim(IVZ,k,j,il-i) = prim(IVZ,k,j,il);
        if (prim(IVX,k,j,il) >= 0.0) {
          prim(IVX,k,j,il-i) = prim(IVX,k,j,il);
        } else {
          prim(IVX,k,j,il-i) = 0.0;
        }

        // set neutral fraction to fully neutral
        pmb->pscalars->r(0,k,j,il-i) = 1.0;
        pmb->pscalars->s(0,k,j,il-i) = prim(IDN,k,j,il-i);
      }
    }
  }
  return;
}


/*
 * Bottom (r_inner) boundary condition: inflow diode
 * If radial velocity is positive (outwards), copy it
 * If radial velocity is negative (inwards), set it to zero
 * Density, ionization are fixed to initial values
 * Pressure is set to isothermal/adiabatic
 * Copy all other vars
*/
void InflowInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
  FaceField &b, Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku,
  int ngh) {

  Real gamma = pmb->peos->GetGamma();
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {

        // fix initial density and pressure
        prim(IDN,k,j,il-i) = rho_gz[il-i];
        prim(IPR,k,j,il-i) = prim(IPR,k,j,il) * std::pow(rho_gz[il-i]/prim(IDN,k,j,il), gamma);

        // outflow diode for velocity
        prim(IVY,k,j,il-i) = prim(IVY,k,j,il);
        prim(IVZ,k,j,il-i) = prim(IVZ,k,j,il);
        if (prim(IVX,k,j,il) >= 0.0) {
          prim(IVX,k,j,il-i) = prim(IVX,k,j,il);
        } else {
          prim(IVX,k,j,il-i) = 0.0;
        }

        // set neutral fraction to fully neutral
        pmb->pscalars->r(0,k,j,il-i) = 1.0;
        pmb->pscalars->s(0,k,j,il-i) = prim(IDN,k,j,il-i);
      }
    }
  }
  return;
}


/*
 * Top (r_outer) boundary condition: outflow diode
 * If velocity is positive (outwards), copy velocity, density and pressure
 * Otherwise, if velocity is negative (inwards), set all to zero
 * Copy all other vars
*/
void OutflowOuterX1 (MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
  FaceField &b, Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku,
  int ngh) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        Real dr = pco->x1v(iu+i) - pco->x1v(iu);
        if (prim(IVX,k,j,iu) >= 0.0) {
          prim(IVX,k,j,iu+i) = prim(IVX,k,j,iu);
          prim(IPR,k,j,iu+i) = prim(IPR,k,j,iu);
          prim(IDN,k,j,iu+i) = prim(IDN,k,j,iu);
        } else {
          prim(IVX,k,j,iu+i) = 0.0;
          prim(IPR,k,j,iu+i) = pfloor;
          prim(IDN,k,j,iu+i) = dfloor;
        }
        prim(IVY,k,j,iu+i) = prim(IVY,k,j,iu);
        prim(IVZ,k,j,iu+i) = prim(IVZ,k,j,iu);
      }
    }
  }
  return;
}


} // end namespace
