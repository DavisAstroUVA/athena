#ifndef MONTECARLO_HPP
#define MONTECARLO_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file montecarlo.hpp
//  \brief definitions for MonteCarlo class
//
// Current design focusses on implementing static post-processing so these class
// implementations will evolve.

#include <cmath>    // isnan, isinf
#include <sstream>
#include <complex>
#include <random>
#include <vector>
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif
// Athena++ classes headers
#include "../athena.hpp"
#include "../coordinates/coordinates.hpp"
#include "../outputs/outputs.hpp"
#include "../field/field.hpp"
#include "photon.hpp"
#include "mcbvals.hpp"
#include "mcoutput.hpp"
#include "mccoord.hpp"
#include "photon_frames.hpp"
#include "polarization.hpp"
#include "tetrad.hpp"

// GSL library
#if GSL
#include <gsl/gsl_randist.h>

#endif

class Mesh;
class MeshBlock;
class MonteCarloBlock;
class MCRankExchange;
class ParameterInput;
class Photon;
class PhotonPusher;
class MCRandom;
class MCBoundaryValues;
class MCOutoupt;
class MCCoord;

//! \brief Physical constants defined in c.g.s.
namespace MCConstants {

static const Real kb_cgs = 1.380649e-16; // Boltzmann constant
static const Real c_cgs = 2.99792458e+10; //  speed of light
static const Real h_cgs = 6.62607015e-27; // Planck's constant
static const Real ec = 4.80320425e-10; // elementary charge in esu
static const Real mp_cgs = 1.67262192595e-24; // proton mass
static const Real me_cgs = 9.1093837139e-28; // electron mass
static const Real amu_cgs = 1.660538782e-24; // atomic mass unit
static const Real mH_cgs = 1.007825 * amu_cgs; // mass of hydrogen
static const Real mec2 = me_cgs * c_cgs * c_cgs;
static const Real kmec2 = kb_cgs / (me_cgs * c_cgs * c_cgs); 
static const Real ev_to_erg = 1.602176634e-12; 
static const Real sigmat = 6.652487051e-25; // Thomson cross section
static const Real res_osc = PI*ec*ec / (me_cgs*c_cgs); // classic oscillator
static const Real lambda_lya = 1215.6701; // Lya wavelength in angstroms
static const Real nu_lya = 1.e8 * c_cgs / lambda_lya;  // frequency of Lya
static const Real lambda_ly_edge = 912.0; // ionization edge wavelength in angstroms
static const Real energy_ly_edge = h_cgs * 1.e8 * c_cgs / lambda_ly_edge; // energy of Lyman edge
static const Real lorwidth_lya = 6.265e8/(4.*PI); // natural line width of Lya
static const Real oscf_lya = 0.4164; // oscilator strength for Lya

} // namespace MCConstants

// Flags for controlling monte carlo emission, scattering, absorption, bcs
enum EmissionFlag {EMISUSER = 0, EMISNONE = 1, EMISFF = 2, EMISBB = 3, MULTI = 4};
enum EmissionGeometry {EMISVOL = 0, EMISAREA = 1, EMISGNONE = 2};
enum AbsorptionOpacityFlag {ABSUSER = 0, ABSNONE = 1, ABSFF = 2, ABSDUST =3};
enum AbsorptionMethodFlag {ABSWEIGHT = 0, ABSPROB = 1, ABSTAU = 2};
enum ScatteringFlag {SCATUSER = 0, SCATNONE =1, SCATISO = 2, SCATTHOM = 3, SCATCOMP =4,
                     SCATRES = 5, SCATDUST = 6};
enum MCBoundaryFlag {MC_PERIODIC_BNDRY = 0, MC_ESCAPE_BNDRY = 1, MC_ABSORB_BNDRY = 2,
                     MC_DESTROY_BNDRY = 3, MC_POLAR_BNDRY = 4, MC_REFLECT_BNDRY = 5,
                     MC_USER_BNDRY = 6, MC_BLOCK_BNDRY = 7};
// Array indices for monte carlo radiation moments
enum {MCIER=0, MCIFR1=1, MCIFR2=2, MCIFR3=3, MCIPR11=4, MCIPR22=5, MCIPR33=6,
      MCIPR12=7, MCIPR13=8, MCIPR23=9, MCIPR21=10, MCIPR31=11, MCIPR32=12};

//----------------------------------------------------------------------------------------
//! \brief storage slot for each component of the moment tensor.
//!
//! The moment arrays hold Er = T^00, Frmc_i = c T^0i and Prmc_ij = T^ij.  Keeping that
//! mapping in one table is what stops the accumulation and the frame transform, which are
//! inverses of each other, from drifting apart.  The 0i entries carry the extra factor of
//! c; MomentNeedsC() says which those are.
constexpr int MomentSlot[4][4] = {{MCIER, MCIFR1, MCIFR2, MCIFR3},
                              {MCIFR1, MCIPR11, MCIPR12, MCIPR13},
                              {MCIFR2, MCIPR12, MCIPR22, MCIPR23},
                              {MCIFR3, MCIPR13, MCIPR23, MCIPR33}};

constexpr bool MomentNeedsC(int a, int b) { return (a == 0) != (b == 0); }
enum SourceTermFlag {MCRS0 = 0, MCRS1=1, MCRS2=2, MCRS3=3, MCRF0=4, MCRF1=5,
                     MCRF2=6, MCRF3=7, MCNABS=8};
//----------------------------------------------------------------------------------------
// function pointer prototypes for user-defined modules set at runtime
typedef Real (*EmisFunc_t)(MonteCarloBlock *pmcb, int k, int j, int i, int etype);
typedef void (*DensFunc_t)(MonteCarloBlock *pmcb);
typedef void (*TempFunc_t)(MonteCarloBlock *pmcb);
typedef void (*NumbFunc_t)(MonteCarloBlock *pmcb);
typedef void (*MCBValFunc_t)(MonteCarloBlock *pmcb, MCCoord *pco, Photon *pphot, int ip);
typedef Real (*OpacFunc_t)(MonteCarloBlock *pmcb,  Photon *pphot, int ip);
typedef void (*ScatFunc_t)(MonteCarloBlock *pmcb, Photon *phot, int ips, int ipe);
typedef void (*UserMoveFunc_t)(MonteCarloBlock *pmcb, Photon *phot, PhotonPusher *ppusher,
                               int ip);
typedef void (*GetZonePos_t)(Photon *phot, MCRandom *pran, MCCoord *pco, int ip);
//! User moment functions receive the photon already projected into the frame they were
//! enrolled with, so they never have to reimplement a tetrad projection.  pphot is still
//! passed for weight, type and anything else photon-specific; anything frame-dependent
//! should come from the PhotonFrameState, including the path length.
typedef void (*UserMomentFunc_t)(MonteCarloBlock *pmcb, Photon *pphot, int ip, int imom,
                                 const PhotonFrameState &s);
typedef void (*UserSourcetermFunc_t)(MonteCarloBlock *pmcb, Photon *pphot,
                                    Real energy0, Real weight0, Real k1p0,
                                    Real k2p0, Real k3p0, int ip);
//--------------------- prototypes for opacity.cpp functions -----------------------------
Real NoOpacity(MonteCarloBlock *pmcb, Photon *pphot, int ip);
Real FreeFreeAbsorptionOpacity(MonteCarloBlock *pmcb, Photon *pphot, int ip);
Real DustAbsorptionOpacity(MonteCarloBlock *pmcb, Photon *pphot, int ip);
Real ThomsonOpacity(MonteCarloBlock *pmcb, Photon *pphot, int ip);
Real ComptonOpacity(MonteCarloBlock *pmcb,  Photon *pphot, int ip);
Real ResonanceLineOpacity(MonteCarloBlock *pmcb,  Photon *pphot, int ip);
Real DustScatteringOpacity(MonteCarloBlock *pmcb,  Photon *pphot, int ip);
void GenerateComptonTable(int io);
Real ComptonCrossSection(Real energy, Real theta);
Real Maxwell(Real theta, Real gamma);
Real KleinNishina(Real x);
Real ResLinePre();
Real XsecLorentzian(Real nu);
Real XsecDoppler(Real nu, Real tgas);
Real XsecVoigt(Real nu, Real tgas);
void InitializeAccelerationOpacity(MonteCarloBlock *pmcb);
//--------------------- prototypes for scatter.cpp functions -----------------------------
void NoScatter(MonteCarloBlock *pmcb, Photon *pphot, int ips, int ipe);
void ScatterIsotropic(MonteCarloBlock *pmcb, Photon *pphot, int ips, int ipe);
void ScatterThomsonPolarized(MonteCarloBlock *pmcb, Photon *pphot, int ips, int ipe);
void ScatterThomsonUnpolarized(MonteCarloBlock *pmcb, Photon *pphot, int ips, int ipe);
void ScatterComptonUnpolarized(MonteCarloBlock *pmcb, Photon *pphot, int ips, int ipe);
void ScatterComptonPolarized(MonteCarloBlock *pmcb, Photon *pphot, int ips, int ipe);
void ScatterResonanceLine(MonteCarloBlock *pmcb, Photon *pphot, int ips, int ipe);
void ScatterDust(MonteCarloBlock *pmcb, Photon *pphot, int ips, int ipe);
Real Bigy(Real x, Real xp);
Real SigmaHat(Real x);
Real ElectronDistPozdnyakov(Real tgas, MCRandom *pran);
Real ElectronDist(Real tgas, MCRandom *pran);
void SampleDipole(Real theta_in, Real phi_in, Real &theta_out, Real &phi_out,
                  MCRandom *pran);
Real SampleVelocityParallel(Real a, Real x_in, MCRandom *pran);
//--------------------- prototypes for emission.cpp functions ----------------------------
Real GetEmissionFreeFree(MonteCarloBlock *pmcb, int k, int j, int i, int etype);
void PhotonEmitFreeFree(MonteCarloBlock *pmcb, Photon *pphot, Real lemin, Real lemax,
                        int ip);
Real GetEmissionBlackbody(MonteCarloBlock *pmcb, int k, int j, int i, int etype);
void PhotonEmitBlackbody(MonteCarloBlock *pmcb, Photon *pphot, BoundaryFace face, int ip);
Real PlanckDist(Real temp, MCRandom *pran);
void GetZonePositionCartesian(Photon *pphot, MCRandom *pran, MCCoord *pco, int ip);
void GetZonePositionSphericalPolar(Photon *pphot, MCRandom *pran, MCCoord *pco, int ip);
void GetZonePositionCylindrical(Photon *pphot, MCRandom *pran, MCCoord *pco, int ip);
void GetZonePositionCartesianFace(Photon *pphot, MCRandom *pran, MCCoord *pcoord,
                                  BoundaryFace face, int ip);
//---------------------- prototypes for setting flags ------------------------------------
enum MCBoundaryFlag GetMCBoundaryFlag(std::string input_string);
enum EmissionFlag GetEmissionFlag(std::string input_string);
enum EmissionGeometry GetEmissionGeometry(std::string input_string);
enum BoundaryFace SetEmissionSurface(std::string input_face);
enum AbsorptionOpacityFlag GetAbsorptionOpacityFlag(std::string input_string);
enum AbsorptionMethodFlag GetAbsorptionMethodFlag(std::string input_string);
enum ScatteringFlag GetScatteringFlag(std::string input_string);
enum MCPolarization GetMCPolarizationFlag(std::string input_string);

//----------------------------------------------------------------------------------------
//! \struct MCBlockSize
//! \brief physical size of monte carlo block

typedef struct MCBlockSize {
  int nx1,nx2,nx3;
  int is,ie,js,je,ks,ke;

} MCBlockSize;

//----------------------------------------------------------------------------------------
//! \class MCRandom
//! \brief monte carlo random number generator

class MCRandom {
public:
  MCRandom(int iseed);
  ~MCRandom();

  Real uniform();
  Real chisquare(Real nu);
  int binomial(unsigned int n, Real p);
  void SampleMultinomial(int n, int m, Real *prob, int *counts);
  //! 64-bit version
  void SampleMultinomial(std::int64_t n, int m, const Real *prob, std::int64_t *counts);
  //! the generator's state as bytes, and back, so a block's stream survives a move
  std::string SaveState() const;
  void RestoreState(const std::string &state);

private:

#if GSL
  gsl_rng *dev;
#endif
  std::mt19937 gen;
  std::uniform_real_distribution<Real> uniform_dist;

};

//----------------------------------------------------------------------------------------
//! \class MonteCarlo
//! \brief monte carlo functions and data

class MonteCarlo {
  friend class MCBoundaryValues;

public:
  MonteCarlo(ParameterInput *pin, Mesh *pmesh);
  ~MonteCarlo();

  // data
  Mesh *pmy_mesh;
  MCOutput *pmcout;
  AthenaArray<MonteCarloBlock*> my_blocks;

  Real tint;   // Monte Carlo timestep
  Real tmax;   // Maximum evolution time
  Real weightratio; // used for setting minimum weight for absorption

  int ntype; // number of emission types
  int64_t nsamp;  // total number of photons to integrate per timestep/output
  int64_t *nsamptype; // number of sample per type
  //! Counts of actual photons run and scatterings performed, globally
  int64_t nphot_run, nscat_run;
  int nblocal; // number of montecarloblocks on this process
  int nbtotal; // total number of montecarloblocks
  int nout;  // number of outputs
  int64_t ncells; // total number of cells in mesh
  int iseed;  // seed to initialized random number generator(s)
 
  int list_size_init; // maximum number of photons run per output on any process
  int max_phots_init; // maximum number of photon elements
  int nuser_var, nuser_mom;
  int checkscat,capmove;
  int emission_method;
  int *emission_geometry;
  BoundaryFace *emission_face;

  enum EmissionFlag emission_flag;

  enum MCBoundaryFlag mc_bcs[6];
  enum ScatteringFlag scattering_meth;

  bool dynamic; // is monte carlo evolving with time
  bool coupled; // is monte carlo evolution coupled to hydro
  bool boosts;  // Compute lorentz transformations
  bool using_bfield; // set magnetic fields
  bool tetrads; // convert from coordinate frame
  bool emission_array;  // Compute and save cell emissivities
  bool *emission_eqwt; // Set initial weights equal
  bool *initialize_comoving; // Transform from comoving frame for emission
  enum AbsorptionMethodFlag *absorption_method; // absorption method for each emission type

  MCPolarization polarized;// how much of the polarization state is tracked
  bool acceleration;  // use MRW acceleration
  bool computedmin;
  //! <montecarlo>/compute_dmin: build MCCoord::dmin even without MRW acceleration, for a
  //! user hook that needs the smallest cell width
  bool compute_dmin;
  bool time_acc;  // use MRW acceleration with time limit
  bool raytrace_flag; // Will trace photons rather than scatter
  bool general_pusher_flag; // Use integration for photon movement
  bool verbose; // print out more information during run
  //! print the per-rank and per-block transport cost after each emission type
  bool lb_report;

  //! which metric the module integrates on; see SetCoordinateSystem
  MCCoordSystem coord_system;
  //! shape of (x1,x2,x3); derived from coord_system, cached because it is read per photon
  MCTopology topology;
  //! true when coord_system is a curved spacetime; derived from coord_system
  bool curved_metric;

  // canonical geometry/wavevector tag written to output headers; see SetGeometryTag
  std::string geometry_tag;
  // parameters of the metric, as "key=value,key=value"
  std::string metric_params;
  // frame tag for outputs
  std::string frame_tag;
  // true when geometry_tag denotes a curved (or at least GR-integrated) spacetime,
  // in which case list output reports the conserved energy -k_t rather than k^t
  bool relativistic_output;

  // function pointers
  UserMoveFunc_t UserWorkInMove;
  EmisFunc_t *GetEmission; // array of function pointers
  DensFunc_t UserGetDensity;
  TempFunc_t UserGetTemperature;
  NumbFunc_t UserGetNumberDensity;
  ScatFunc_t UserScattering;
  OpacFunc_t UserScatteringOpacity;
  OpacFunc_t UserAbsorptionOpacity;
  std::string *user_moment_names;
  UserMomentFunc_t *user_moment_func;
  MCFrame *user_moment_frame; // frame each user moment is accumulated in
  UserSourcetermFunc_t UserSourcetermFunc;

  // functions
  // SWD: some of these functions could/should be private
  void RunMonteCarlo(Outputs *pouts, Mesh *pmesh, ParameterInput *pinput);
  bool CheckAndBroadCastPhotonsRemaining();
  //! move photons between blocks on this rank; true when something landed here
  bool ExchangeLocal();
  //! flush receive buffers into their blocks; true when any block received something
  bool DrainArrivals();
  // send what was staged for other ranks, take delivery, and test for completion
  bool FinishRound();
  // transport every photon of this emission type to completion using photon counters
  void TransportAsync(int etype, ParameterInput *pin);
  //! one block's transport sweep, with timing for load balancing
  void TransportBlock(int nb, int etype);
  //! gather every block's window cost and counters and print the balance on rank 0
  void ReportLoadBalance(int etype);
  //! this rank's own sweep time prior to move
  double lb_rank_time;
  //! wall-clock seconds of the current emission type's transport
  double lb_transport_wall;
  //! this rank's wall time inside the asynchronous loop spent in passes with no
  //! work to sweep, and in the exchange
  double lb_idle_time, lb_exchange_time;
  //! exchange time split by call: completing the previous sends, taking delivery
  //! from other ranks, flushing receive buffers into blocks, the same-rank hand-off
  //! sweep, and posting the staged sends
  double lb_t_complete, lb_t_drain_in, lb_t_drain_arr, lb_t_local, lb_t_send;
  long lb_passes;

  //! the fluid-derived arrays of one block, from its MeshBlock's primitives: density,
  //! temperature, number density, free-free prefactor, frame, scalars, field.
  void SetupBlockFromFluid(MonteCarloBlock *pmcb);
  //! per-block cap on resident photons for the current nblocal (see Initialize)
  int ComputeLoopMax() const;

  // mesh's RedistributeAndRefineMeshBlocks calls the two hooks:
  // PackDeparting before it deletes the old MeshBlocks, with its new-rank and old/new gid
  // maps, and RebuildAfterRedistribution after Initialize(2) has refilled the new ones.
  void PackDeparting(const int *newrank, const int *newtoold, const int *oldtonew,
                     int ntot_new);
  void RebuildAfterRedistribution(ParameterInput *pin);
  //! problem-generator hook, run last: rebuild anything indexed by lid or sized to nblocal
  void UserWorkAfterRebalance(ParameterInput *pin);
  //! how many redistributions the module has followed
  int lb_epoch;

  //! One block's movable state. ib: photon integer properties, counters, the emission
  //! cursor and counts; rb: photon real properties, moments, source terms, emission,
  //! weights; sb: the random generator's state.
  struct BlockPayload {
    std::vector<int> ib;
    std::vector<Real> rb;
    std::vector<char> sb;
  };
  //! <montecarlo> lb_test_repack: none; local -- at the top of every RunMonteCarlo pack,
  //! destroy and rebuild every block in place, no mesh call, which must be bitwise
  //! neutral; mesh -- run the mesh's redistribution with unchanged costs and treat every
  //! block as departing and arriving, which exercises the hooks end to end.
  enum LbTestMode {LBTEST_NONE = 0, LBTEST_LOCAL = 1, LBTEST_MESH = 2};
  LbTestMode lb_test_repack;
  //! <montecarlo> lb_test_costs: measured (the default) or alternate -- synthetic block
  //! costs of 3 and 1 on the two halves of the gid range, swapped every balance, so that
  //! a known set of blocks changes rank each time.  Needs <loadbalancing> balancer =
  //! manual, whose cost path reads MeshBlock::cost_ as given.
  enum LbCostMode {LBCOST_MEASURED = 0, LBCOST_ALTERNATE = 1};
  LbCostMode lb_test_costs;
  //! the static run's balance point: between transports, on the previous transport's
  //! measured cost, through the mesh's own balancer and hooks
  void BalanceStatic(ParameterInput *pin);
  //! test costs if requested, the prediction guard, then the mesh's balancer with its
  //! cycle counter satisfied. Returnsrue if blocks were redistributed.
  bool BalanceNow(ParameterInput *pin);
  //! Mid-transport balancing in the synchronous round loop
  int lb_check_interval, lb_max_per_transport, lb_min_window;
  //! check for load balance fraction for asynchronous
  Real lb_check_fraction;
  void AssignTestCosts();
  //! <loadbalancing> cost_file: per-block transport costs written after every transport
  //! and read back at startup, so a run on the same mesh starts balanced
  std::string lb_cost_file;
  bool lb_costs_loaded;
  //! <montecarlo> lb_min_gain: a redistribution is taken only if the busiest rank of the
  //! partition the mesh would choose is at least this fraction below the current one.
  //! The mesh tests whether the current layout is imbalanced, not whether its greedy
  //! contiguous partition improves on it, and with few blocks per rank it can be worse.
  Real lb_min_gain;
  //! move every block's pending hand-off time into its cost, its window time and this
  //! rank's time.  Called before any of them is read.
  void FoldPendingCosts();
  //! this rank's blocks' costs as the balancer will see them (aged as it ages them),
  //! written into a gid-indexed list that a gather then completes
  void FillLocalBalancerCosts(std::vector<double> &cost);
  //! the same, gathered (blocking collective)
  void GatherBalancerCosts(std::vector<double> &cost);
  //! would the new partition improve on current layout
  bool WorthwhileFromCosts(const std::vector<double> &cost) const;
  //! gather and judge, in one blocking step
  bool RedistributionWorthwhile();
  //! the partition of a cost list the mesh will use when it redistributes: the
  //! module's optimal contiguous one under <montecarlo> lb_partition = optimal (the
  //! default), the mesh's greedy CalculateLoadBalance under greedy.  Called by the mesh
  //! from RedistributeAndRefineMeshBlocks and by the prediction guard, so the two agree.
  void Partition(double *cost, int nb, int *rlist, int *slist, int *nlist) const;
  bool lb_partition_optimal;
  //! <montecarlo> lb_cost_decay: after every balance check the accumulated block costs
  //! are scaled by this, so what the next check sees leans toward the recent window.
  //! 1 (the default) keeps everything since the last redistribution.
  Real lb_cost_decay;
  void DecayCosts();
  //! <montecarlo> lb_initial = none|photons: with photons and no cost file, the first
  //! transport is balanced on each block's share of the photons to emit before any is
  //! moved.  Informative for equal-weight emission, where that share follows the
  //! emissivity; uniform, and so useless, for the variable-weight scheme.
  bool lb_initial_photons;
  void WriteCostFile();
  bool ReadCostFile();

 private:
  MonteCarloBlock *RebuildArrival(MeshBlock *pmb, ParameterInput *pin,
                                  const BlockPayload &payload);
  void RepackAllLocal(ParameterInput *pin);
  void RelinkAll();
  std::vector<MonteCarloBlock*> lb_kept_;      //!> by new lid; null where a block arrives
  std::vector<int> lb_src_;                    //!> by new lid; source rank, -1 if kept
  std::vector<MonteCarloBlock*> lb_departed_;  //!> old blocks to delete after the rebuild
  std::vector<BlockPayload> lb_send_;          //!> payloads in flight to other ranks
  std::vector<BlockPayload> lb_local_;         //!> by new lid; test-mode payloads kept here
#ifdef MPI_PARALLEL
  std::vector<MPI_Request> lb_req_;
  MPI_Comm lb_comm_;                           //!> block transfers, apart from the mesh's
#endif
  int per_block_cap_, photon_budget_;          //!> inputs to ComputeLoopMax

 public:
  //! clock in the units MeshBlock::StartTimeMeasurement uses, so the costs add
  static double LoadBalanceClock();
  static double LoadBalanceSeconds(double clock_units);
  //! use the counter-based termination test instead of a collective every round
  bool async_term;
  // ceiling on consecutive same-rank transport sweeps before taking the global step,
  // so two blocks trading a photon cannot hold the other ranks at the barrier
  int local_max_sweeps;
  // Blocks taking part in the current transfer round. see CheckAndBroadCastPhotonsRemaining
  // for what puts a block in each.
  std::vector<int> send_list_, recv_list_;
  // moves photons between ranks a rank at a time; null when <montecarlo>/rank_exchange is off,
  // inactive on a single rank
  MCRankExchange *pexch;
  void InitUserMonteCarloData(ParameterInput *pin);
  // Enroll User functions
  void EnrollUserMCBoundaryFunction(enum BoundaryFace dir, MCBValFunc_t my_bc);
  void EnrollUserEmissionFunction(EmisFunc_t emissfunc);
  void EnrollUserEmissionFunction(EmisFunc_t emissfunc, int etype);
  void EnrollUserGetDensity(DensFunc_t densfunc);
  void EnrollUserGetTemperature(TempFunc_t tempfunc);
  void EnrollUserGetNumberDensity(NumbFunc_t numbunc);
  void EnrollUserWorkInMove(UserMoveFunc_t userfunc);
  void EnrollUserOpacityFunction(OpacFunc_t opacfunc, bool abs);
  void AllocateUserMoments(int n);
  void EnrollUserMoment(int i, UserMomentFunc_t my_func, const char *name,
                        MCFrame frame = MCFRAME_LAB);
  void EnrollUserSourcetermUpdate(UserSourcetermFunc_t my_func);
  void EnrollUserScatteringFunction(ScatFunc_t scatfunc);
  void Initialize(ParameterInput *pinput);
  void InitializeEmission(ParameterInput *pin);
  void SetCoordinateSystem(ParameterInput *pin);
  void SetGeometryTag(ParameterInput *pin);
  void SetMetricParams(ParameterInput *pin);
  void DistributeSamples(int etype);
  void NormalizeDomainOutputs(bool normalize);
private:

  // functions
  MCBValFunc_t BoundaryFunction_[6];

};

//----------------------------------------------------------------------------------------
//! \class MonteCarloBlock
//! \brief monte carlo functions and data contained on each mesh block

class MonteCarloBlock {
public:
  MonteCarloBlock(MeshBlock *pmb, MCBlockSize *pblsize, MonteCarlo *pmc,
                  ParameterInput *pin);
  ~MonteCarloBlock();

  // data
  MonteCarlo* pmy_mc; // MonteCarlo
  MeshBlock* pmy_block;    // MeshBlock corresponding to this MonteCarloBlock
  MonteCarloBlock *next;
  MCCoord *pcoord;

  Photon* pphot; // ptr to photon packet
  PhotonPusher* ppusher; // ptr to photon pusher
  MCRandom *pran; // ptr to random number generator
  MCBoundaryValues *pbval; // ptr to MC boundary values

  // ouput pointers
  Spectrum *pspec; // ptr to spectrum
  PhotonList *pphlist; // ptr to photon list
  PhotonTrajectoryList *ptraj; // ptr to traj list

  enum MCBoundaryFlag mcb_bcs[6];

  // function pointers
  GetZonePos_t GetZonePosition;
  OpacFunc_t AbsorptionOpacity;
  OpacFunc_t ScatteringOpacity;
  ScatFunc_t Scatter;

  int64_t nphrun; // Photons initialized thus far
  int64_t nphremain; // total number of photons to integrate
  double lb_time; // transport cost for this block
  int64_t lb_nstep;
  //! hand-off time charged by Photon's exchange functions since the last fold; see
  //! MonteCarlo::FoldPendingCosts
  double lb_pending;
  //! everything of this block that has to move with it and cannot be rebuilt from the
  //! fluid: resident photons, the accumulated moments and source terms, the emission
  //! array and cursor, counters, weights, the random generator
  void PackForTransfer(std::vector<int> &ib, std::vector<Real> &rb,
                       std::vector<char> &sb) const;
  void UnpackFromTransfer(const std::vector<int> &ib, const std::vector<Real> &rb,
                          const std::vector<char> &sb);
  int64_t nabs, nesc, ndes, nscat, nrem; // counters
  int loop_max_size;
  int nx1,nx2,nx3;
  int is,ie,js,je,ks,ke;
  int nsrc, nmom; // # of elements in sourcterm, moments arrays
  int nspec;
  int nf_scat;

  bool mom_flag_lab; // Compute/output moments
  bool mom_flag_com; // Compute moments in comoving frame
  bool mom_flag_coord; // Compute moments in the coordinate basis
  bool accumulate_com; // accumulate comoving moments directly rather than deriving them
  // The lab moments have to exist and be accumulated whenever the comoving ones are
  // derived from them, even if the lab moments are not themselves being output.
  bool need_lab_moments;
  bool mom_flag_src; // Compute source terms for output
  bool mom_flag_usr; // Compute user defined monte carlo moments
  bool mom_flag_scat; // Compute scattering source terms
  bool call_moments;
  bool call_srcterms;

  bool boosts;  // Compute lorentz transformations
  bool tetrads; // Compute tetrads
  //! are boost_cmv/boost_lab allocated and filled?  They serve the moment deposition
  //! and, outside GR, the legacy frame transforms; see the constructor for the condition.
  bool cache_tetrads;
  bool coupled; // Whether time dependent code is coupled to hydro
  bool coherent_scattering; // photon does notchange energy after scattering
  bool acceleration;  // use MRW acceleration
  bool computedmin;
  bool time_acc;  // use MRW acceleration with time limit

  // Set flags

  enum AbsorptionOpacityFlag absorption_opac;
  enum ScatteringFlag scattering_meth;

  //! mirrors of MonteCarlo::coord_system and friends, copied in so the hot paths do not
  //! chase a pointer per photon
  MCCoordSystem coord_system;
  MCTopology topology;
  bool curved_metric;

  //! true when the comoving frequency shift is identically one everywhere, so that an
  //! opacity depends on the cell alone and cannot change while a photon crosses it.
  //! Requires a flat metric (so the lapse is one) and a fluid at rest (so there is no
  //! Doppler term).  Set by ComputeTransformations, which is where the velocity is known.
  //! GeneralPusher uses it to skip the per-step opacity refresh, which is then provably a
  //! no-op; see the comment there.
  bool shift_unity;

  // Associated with general pusher
  // SWD some of these should be eliminated others moved to MonteCarlo?
  bool orthotet_flag; // use orthonormal tetrad for TransferPhotons()
  bool varystep_flag; // use variable (true) or constant (false) step

  Real rho_cgs, vel_cgs, tgas_cgs, tfloor_cgs, tceiling_cgs, l_cgs, time_cgs;
  // Helium abund and derived gas constant
  Real heabund, rgas;
  static Real GasConstant(Real heabund);
  Real betamax;
  Real stepsize;
  Real minweight;
  Real emiss_to_weight; // used relate weight to emission array
  Real emin_scat, emax_scat, dloge_scat; // min/max energy for scattering moments

  AthenaArray<Real> emission;
  AthenaArray<Real> moments;
  AthenaArray<Real> moments_com;
  AthenaArray<Real> moments_coord;
  AthenaArray<Real> moments_user;
  AthenaArray<Real> moments_scat;
  AthenaArray<Real> moments_scat_error;
  //! Contribution the photon currently being pushed has made to one (frequency bin, cell)
  //! pair so far, held back until the photon leaves that pair.
  Real scat_pend_sum_;
  int scat_pend_n_, scat_pend_i1_, scat_pend_i2_, scat_pend_i3_;
  AthenaArray<Real> energy_scat;
  AthenaArray<Real> freq_scat_mid;
  AthenaArray<Real> sourceterms;
  AthenaArray<Real> scalars;
  AthenaArray<Real> rho;
  AthenaArray<Real> species;
  AthenaArray<Real> tgas;
  // Free-free prefactor
  AthenaArray<Real> ff_cell;
  // Flat spacetime only: (gamma, gamma*beta^i) in the orthonormal frame, so consumers
  // divide by vel(...,0) to get beta^i. Unallocated in GR.
  AthenaArray<Real> vel;
  // General relativity only: the primitive (relative) three-velocity uu^i of the frame
  // the comoving tetrad is built on.
  AthenaArray<Real> uprim;
  AthenaArray<Real> bcc;
  AthenaArray<Real> boost_cmv;
  AthenaArray<Real> boost_lab;
  AthenaArray<Real> planck_opacity; // for acceleration
  AthenaArray<Real> planck_inv_opacity; // for acceleration

  // functions
  void InitUserMonteCarloBlockData(ParameterInput *pin);
  void MonteCarloProblemGenerator(ParameterInput *pin);
  void RayTracePhotonsOnBlock(int etype); // Ray trace photon on this block
  void TransferPhotonsOnBlock(int etype); // Transfer photons on this block
  void CoupleMonteCarloToFluid(Real dt);  // couple monte carlo to mesh
  void LorentzTransform(Photon *pphot, const Real sign, int ips, int ipe);
  Real LorentzTransformFrequencyShift(Photon *pphot, int ip);
  void InitializePhoton(Photon *pphot, int ips, int ipe, int etype);
  void FinalizePhoton(Photon *pphot, int ip);
  void UpdateMoments(Photon *pphot, Real dl, Real etau, int ip);
  void UpdateMoments(Photon *pphot, Real dl, int ip);
  void UpdateMomentsAcceleration(Photon *pphot, Real dl, Real pl, Real k1, Real k2,
                                 Real k3,Real etau, int ip);
  void NormalizeMoments(bool normalize);
  //! Fold the scattering-moment contribution into moments_scat_error
  void FlushScatError();
  void AccumulateMoments(AthenaArray<Real> &mom, int type, int i3, int i2, int i1,
                         const PhotonFrameState &s, Real wp);
  void ComovingFrameMatrix(int k, int j, int i, const AthenaArray<Real> &g,
                           const AthenaArray<Real> &gi, Real lam[4][4]);
  void DeriveComovingMoments();
  void ResetMoments();
  void UpdateSourceTerms(Photon *pphot, Real energy0, Real weight0,
                         Real k1p0, Real k2p0, Real k3p0, int ip);
  void NormalizeSourceTerms(bool normalize);
  void ResetSourceTerms();
  // Functions for handling distributed emission over cells
  void ComputeEmissionArray(int etype, Real &emm_min, Real &emm_max, Real &emm_tot);
  void ComputeEmissionSampleArray();
  //void ComputeEmissionSampleArray(BoundaryFace face);
  void SetEmissionCellWeight(Photon *pphot, int ips, int ipe);
  void SetEmissionCellWeightArea(Photon *pphot, BoundaryFace face, int ips, int ipe);
  // Index range the fluid-derived arrays are filled over: active cells plus ghosts.
  // Photons legitimately occupy a ghost cell while they wait to be handed to the
  // neighboring block, and the pusher reads rho, tgas, vel and the boost matrices at
  // whatever cell the photon is in, so filling active cells alone leaves those reads
  // returning zero.
  void FillBounds(int &il, int &iu, int &jl, int &ju, int &kl, int &ku) const;

  // Four-velocity of cell (i3,i2,i1)'s frame, rebuilt at the position x rather than read
  // from the cell center, so that u.u = -1 holds where the vector is actually used.
  // General relativity only.
  // x is not const because MCCoord::Metric and InverseMetric take a mutable Real[4].
  void FluidFourVelocity(Real x[4], int i3, int i2, int i1, Real ucon[4]) const;
  // The same, given the metric pair at x by a caller that already has it (the general
  // pusher carries the pair from step to step).  The first overload evaluates the pair
  // and calls this one.
  void FluidFourVelocity(const Real gcov[4][4], const Real gcon[4][4], int i3, int i2,
                         int i1, Real ucon[4]) const;

  void GetDensity();
  void GetNumberDensity();
  void ComputeFreeFreePrefactor();
  void GetScalars();
  void GetVelocity();
  void SetNormalObserver();
  void GetBField();
  void GetTemperature();
  void ComputeTransformations();
  void TransformToComoving(Photon *pphot, int ips, int ipe);
  void TransformToCoordinate(Photon *pphot, int ips, int ipe);
  Real FrequencyShiftComoving(Photon *pphot, int ips);
  // The same, given the metric pair at the photon; in general relativity this is the
  // body and the first overload evaluates the pair, outside it the pair is not needed
  // and the first overload is called.
  Real FrequencyShiftComoving(Photon *pphot, int ip, const Real gcov[4][4],
                              const Real gcon[4][4]);
  void UserWorkAfterTransfer(int etype);

private:
  int i1_, i2_, i3_; // used for emission
  AthenaArray<int> emit_count_; // used for emission
  void SetBoundaryValues(enum MCBoundaryFlag *input_bcs);
};

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::AccumulateMoments(...)
//! \brief add one photon's contribution to a moment array in a given frame
//!
//! The covariant estimator T^(a)(b) = sum w p^(a) p^(b) dlambda, written in terms of the
//! energy and unit direction PhotonFrames returns.  Every basis goes through here, so the
//! three cannot drift apart the way the lab and comoving paths once did, and it shares the
//! MomentSlot table with DeriveComovingMoments, which is its inverse.
//!
//! Defined here rather than in photon_frames.cpp because it runs once per cell crossing -- a few
//! hundred thousand times in even a small run -- and measurably loses about a percent when
//! it cannot inline into UpdateMoments across a translation unit boundary.

inline void MonteCarloBlock::AccumulateMoments(AthenaArray<Real> &mom, int type,
                                               int i3, int i2, int i1,
                                               const PhotonFrameState &s, Real wp) {
  const Real c_cgs = MCConstants::c_cgs;
  Real weight = wp * s.e * s.dl / c_cgs;
  const Real *n = s.n;
  // All ten components share (type,i3,i2,i1) and differ only in the moment slot, which is
  // the second index, so resolve the address once and step by the slot stride.  The
  // five-dimensional index arithmetic was 2.7% of total instructions when it was repeated
  // per component.  Layout is pdata_[i + nx1*(j + nx2*(k + nx3*(n + nx4*m)))], so
  // consecutive slots sit nx1*nx2*nx3 apart.
  Real *base = &mom(type,0,i3,i2,i1);
  const int stride = mom.GetDim1()*mom.GetDim2()*mom.GetDim3();
#ifdef DEBUG
  // This is the one place that depends on AthenaArray's internal layout.  If that ever
  // changes the arithmetic below goes silently wrong, so assert the stride rather than
  // trusting it.
  if (&mom(type,1,i3,i2,i1) - base != stride) {
    std::stringstream msg;
    msg << "### FATAL ERROR in AccumulateMoments" << std::endl
        << "moment slot stride is " << (&mom(type,1,i3,i2,i1) - base)
        << " but the layout assumption gives " << stride << std::endl;
    ATHENA_ERROR(msg);
  }
#endif
  // Split by whether the component carries the extra factor of c rather than testing it
  // per component; the branch inside the loop costs about a percent of total runtime.
  base[MomentSlot[0][0]*stride] += weight;
  Real wc = weight * c_cgs;
  for (int b=1; b<4; ++b)
    base[MomentSlot[0][b]*stride] += wc * n[b-1];
  for (int a=1; a<4; ++a)
    for (int b=a; b<4; ++b)
      base[MomentSlot[a][b]*stride] += weight * n[a-1] * n[b-1];
}


#endif // MONTECARLO_HPP
