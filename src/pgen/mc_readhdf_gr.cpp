//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mc_readhdf_gr.cpp
//! \brief Monte Carlo problem generator for an X-ray binary in Kerr-Schild coordinates,
//! initialized from an athdf snapshot.

// C headers

// C++ headers
#include <algorithm>  // max()
#include <string>     // c_str(), string
#include <vector>     // vector

// Athena++ headers
#include "../athena.hpp"              // Real
#include "../athena_arrays.hpp"       // AthenaArray
#include "../coordinates/coordinates.hpp"  // Coordinates
#include "../field/field.hpp"         // Field
#include "../globals.hpp"             // Globals
#include "../hydro/hydro.hpp"         // Hydro
#include "../eos/eos.hpp"                  // EquationOfState
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"     // ParameterInput
#include "../monte_carlo/mcgrid.hpp"
#include "../monte_carlo/mcsnapshot.hpp"
#include "../monte_carlo/montecarlo.hpp"
#include "../monte_carlo/photon.hpp"
#include "../monte_carlo/mcutils.hpp"
#include "../inputs/hdf5_reader.hpp"  // HDF5ReadRealArray()
#ifdef HDF5OUTPUT
#include <hdf5.h>
#endif

namespace {
  // Global variables
  bool tnorm;
  Real logemin, logemax;
  Real abh, mbh, r_hor;
  Real dcut = 1.e-20;   // code units
  Real tcut = 1.e20;    // Kelvin
  Real heabund = 0.09; // helium abundance by number
  bool biased = false;   // weights = biased
  std::string escape_file;  // <problem>/escape_file, from escape_table.py; empty = local
  // Floor on the sampled escape probability.  It bounds the largest weight at
  // ave/pesc_min
  Real pesc_min = 1.e-10;
  // emission = table: the module's tabulated opacities (<opacitytable>/opacity_file) supply
  // the emission and absorption; this generator masks the cells above tcut and builds
  // the escape tables on them
  bool table = false;

  //functions
  void InsideHorizon(MonteCarloBlock *pmcb, Photon *pphot, PhotonPusher *ppusher,int ip);
  Real KerrSchildRadius(Real x, Real y, Real z);
  bool InsideHorizonCell(MonteCarloBlock *pmcb, int k, int j, int i);
  Real FreeFreeOpacity(Real tgas, Real rho, Real energy);
  void ExcludeCells(MonteCarloBlock *pmcb);
  void BuildEscapeTables(MonteCarloBlock *pmcb);
  void FillEscapeLocal(MonteCarloBlock *pmcb);
  void ReadEscapeFile(MonteCarloBlock *pmcb);
  void ComputeImportance(MonteCarloBlock *pmcb);

  void CartesianKerrSchild(Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3);
}

void MonteCarlo::InitUserMonteCarloData(ParameterInput *pin) {

  nuser_var = 3;

  abh = pin->GetReal("coord","a");
  mbh = pin->GetOrAddReal("coord","m",1.0); // always 1?
  r_hor = mbh + sqrt(SQR(mbh) - SQR(abh));
  EnrollUserWorkInMove(InsideHorizon);
  dcut = pin->GetOrAddReal("problem", "dcut", dcut);
  tcut = pin->GetOrAddReal("problem", "tcut", tcut);
  heabund = pin->GetOrAddReal("problem", "heabund", heabund);
  biased = (pin->GetOrAddString("montecarlo","weights","emission") == "biased");
  if (biased) {
    pesc_min = pin->GetOrAddReal("problem", "pesc_min", pesc_min);
    escape_file = pin->GetOrAddString("problem", "escape_file", "");
  }

  {
    const std::string emis = pin->GetOrAddString("montecarlo","emission","none");
    const std::string absn = pin->GetOrAddString("montecarlo","absorption","none");
    if (emis == "user" || absn == "user") {
      std::stringstream msg;
      msg << "### FATAL ERROR in MonteCarlo::InitUserMonteCarloData" << std::endl
          << "mc_readhdf_gr has no user emission or absorption; use "
          << "emission = table and absorption = table with <opacitytable>/opacity_file"
          << std::endl;
      ATHENA_ERROR(msg);
    }
  }
  table = (emission_flag == EMISTABLE);
  EnrollUserGetNumberDensity(ExcludeCells);
  if (!table) {
    if (biased) {
      // Escape-table groups, log-spaced over every energy the free-free sampler can
      // draw: emin..emax in eV, or in units of kT over the temperature range with tnorm
      nescape = pin->GetOrAddInteger("problem", "nescape", 32);
      const Real kb = 1.380649e-16, everg = 1.6021772e-12;
      Real lo, hi;
      if (pin->GetOrAddBoolean("problem", "tnorm", false)) {
        Real tlo = pin->DoesParameterExist("problem", "tfloor_cgs")
                   ? pin->GetReal("problem", "tfloor_cgs") : 1.;
        Real thi = pin->DoesParameterExist("problem", "tceiling_cgs")
                   ? pin->GetReal("problem", "tceiling_cgs") : 1.e13;
        lo = std::log(kb * pin->GetReal("problem", "emin") * std::max(tlo, 1.));
        hi = std::log(kb * pin->GetReal("problem", "emax") * std::min(thi, 1.e13));
      } else {
        lo = std::log(everg * pin->GetReal("problem", "emin"));
        hi = std::log(everg * pin->GetReal("problem", "emax"));
      }
      escape_lne.NewAthenaArray(nescape+1);
      for (int l=0; l<=nescape; ++l)
        escape_lne(l) = lo + (hi-lo)*static_cast<Real>(l)/static_cast<Real>(nescape);
    }
    return;
  }

  if (biased) {
    // one escape group per group of the opacity table, which is what PhotonEmitTable's
    // energy-group draw assumes
    nescape = optab->ngroups;
    escape_lne.NewAthenaArray(nescape+1);
    for (int l=0; l<=nescape; ++l) escape_lne(l) = std::log(optab->gedge(l));
  }
}

void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin) {

  // dcut and tcut are read in InitUserMonteCarloData; see the note at their definition.
  if (!table) {
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
  // The opacity tables this builds on were filled by SetupBlockFromFluid, which runs
  // before this hook at startup and on a block's arrival after a redistribution.
  if (biased) BuildEscapeTables(this);
}

void Mesh::InitUserMeshData(ParameterInput *pin) {

  EnrollUserMetric(CartesianKerrSchild);
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief monte carlo test problem generator
//! Inputs:
//! - pin: parameters
//! Outputs: (none)
//! Notes:
//! - the primitives and the cell-centred field come from the athdf snapshot named by
//!   <problem>/input_filename, or by <montecarlo>/grid_from_file when the grid was built
//!   from that snapshot.  MCReadSnapshotBlock locates the variables by name, so the
//!   layout does not have to be described here.

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
    // Don't evelove cells emitted inside the horizon
    if (InsideHorizonCell(this,pphot->i3p[ip],pphot->i2p[ip],pphot->i1p[ip])) {
      pphot->wp[ip] = 0.;
      pphot->statp[ip] = REMOVED;
      continue;
    }
    // Obtain initial position within zone
    GetZonePosition(pphot,pran,pcoord,ip);

    // Set maximum integration time
    pphot->dtp[ip] = pphot->pmy_mcb->pmy_mc->tmax;

    //xs: store gas temperature
    pphot->user[0][ip] = tgas(pphot->i3p[ip],pphot->i2p[ip],pphot->i1p[ip]);

    if (!table) {
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


// Kerr-Schild radius of the point (x,y,z): the positive root of
// r^4 - (R^2 - a^2) r^2 - a^2 z^2 = 0 with R^2 = x^2 + y^2 + z^2.
Real KerrSchildRadius(Real x, Real y, Real z) {
  Real rr2 = SQR(x) + SQR(y) + SQR(z);
  Real a2 = SQR(abh);
  return std::sqrt(0.5*(rr2 - a2 + std::sqrt(SQR(rr2 - a2) + 4.0*a2*SQR(z))));
}

// Whether the center of cell (k,j,i) lies inside the horizon.  Used to zero the number
// densities there and to remove samples drawn there at emission.
bool InsideHorizonCell(MonteCarloBlock *pmcb, int k, int j, int i) {
  Coordinates *pco = pmcb->pmy_block->pcoord;
  return KerrSchildRadius(pco->x1v(i), pco->x2v(j), pco->x3v(k)) < r_hor;
}

void InsideHorizon(MonteCarloBlock *pmcb, Photon *pphot, PhotonPusher *ppusher, int ip) {

  Real r = KerrSchildRadius(pphot->x1p[ip], pphot->x2p[ip], pphot->x3p[ip]);
  if (r < r_hor) {
    pphot->statp[ip] = REMOVED;
    //printf("Photon absorbed inside horizon at r=%g\n",r);
  }
  return;
}

Real FreeFreeOpacity(Real tgas, Real rho, Real energy) {
  Real ffnrm = 3.692146e8;
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

// Escape probability per active cell and energy group from block-local columns, for
// weights = biased without an escape file.  The effective extinction
// sqrt(3 alpha_a (alpha_a + alpha_s)), the inverse of the thermalization length of a
// diffusing photon, is summed from the cell to the edge of its block along the six axis
// directions and the smallest column is used.  A cell in the face layer of a block sees
// no column past the face, so on a mesh where the disk spans many blocks this
// overestimates escape there; escape_table.py integrates through the whole mesh.
void FillEscapeLocal(MonteCarloBlock *pmcb) {
  MonteCarlo *pmc = pmcb->pmy_mc;
  const int ng = pmc->nescape;
  if (ng <= 0) return;
  const int is = pmcb->is, ie = pmcb->ie, js = pmcb->js, je = pmcb->je;
  const int ks = pmcb->ks, ke = pmcb->ke;
  const int nx1 = ie-is+1, nx2 = je-js+1, nx3 = ke-ks+1;
  const Real sigma_t = 6.6524587e-25, kb = 1.380649e-16, l_cgs = pmcb->l_cgs;
  MCCoord *pco = pmcb->pcoord;
  const AthenaArray<Real> &lne = pmc->escape_lne;

  // effective extinction per unit code length
  AthenaArray<Real> kap, tau;
  kap.NewAthenaArray(nx3,nx2,nx1,ng);
  tau.NewAthenaArray(nx3,nx2,nx1,ng);
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        const int kt = k-ks, jt = j-js, it = i-is;
        const Real ne = pmcb->species(0,k,j,i);
        const Real temp = pmcb->tgas(k,j,i), rhoc = pmcb->rho(k,j,i);
        for (int l=0; l<ng; ++l) {
          Real alpha_a;
          if (table && pmc->optab->groups == TABLE_GROUPS_EDGE)
            alpha_a = 0.5*(pmcb->TableExtinction(k,j,i,l) + pmcb->TableExtinction(k,j,i,l+1));
          else if (table)
            alpha_a = pmcb->TableExtinction(k,j,i,l);
          else
            alpha_a = FreeFreeOpacity(temp, rhoc, std::exp(0.5*(lne(l)+lne(l+1))));
          const Real alpha_s = sigma_t*ne;
          kap(kt,jt,it,l) = std::sqrt(3.*alpha_a*(alpha_a+alpha_s)) * l_cgs;
          tau(kt,jt,it,l) = HUGE_NUMBER;
        }
      }
    }
  }

  // Smallest effective depth to a block face.  Emission is uniform through the cell, so
  // the cell's own depth tau_c enters as the fraction (1 - e^{-tau_c})/tau_c that is
  // born within reach of the face, not as e^{-tau_c}: a thick surface cell keeps the
  // escape probability of its outer skin.
  std::vector<Real> cum(ng);
  auto sweep = [&](int kt, int jt, int it, Real dx) {
    for (int l=0; l<ng; ++l) {
      const Real own = kap(kt,jt,it,l)*dx;
      const Real skin = (own > 1.e-6) ? -std::log((1.-std::exp(-own))/own) : 0.;
      tau(kt,jt,it,l) = std::min(tau(kt,jt,it,l), cum[l] + skin);
      cum[l] += own;
    }
  };
  for (int kt=0; kt<nx3; ++kt) {
    for (int jt=0; jt<nx2; ++jt) {
      for (int dir=-1; dir<=1; dir+=2) {
        std::fill(cum.begin(), cum.end(), 0.);
        for (int n=0; n<nx1; ++n) {
          const int it = (dir > 0) ? nx1-1-n : n;
          sweep(kt, jt, it, pco->x1f(is+it+1) - pco->x1f(is+it));
        }
      }
    }
  }
  for (int kt=0; kt<nx3; ++kt) {
    for (int it=0; it<nx1; ++it) {
      for (int dir=-1; dir<=1; dir+=2) {
        std::fill(cum.begin(), cum.end(), 0.);
        for (int n=0; n<nx2; ++n) {
          const int jt = (dir > 0) ? nx2-1-n : n;
          sweep(kt, jt, it, pco->x2f(js+jt+1) - pco->x2f(js+jt));
        }
      }
    }
  }
  for (int jt=0; jt<nx2; ++jt) {
    for (int it=0; it<nx1; ++it) {
      for (int dir=-1; dir<=1; dir+=2) {
        std::fill(cum.begin(), cum.end(), 0.);
        for (int n=0; n<nx3; ++n) {
          const int kt = (dir > 0) ? nx3-1-n : n;
          sweep(kt, jt, it, pco->x3f(ks+kt+1) - pco->x3f(ks+kt));
        }
      }
    }
  }
  kap.DeleteAthenaArray();

  for (int kt=0; kt<nx3; ++kt)
    for (int jt=0; jt<nx2; ++jt)
      for (int it=0; it<nx1; ++it)
        for (int l=0; l<ng; ++l)
          pmcb->escape_prob(kt,jt,it,l) = std::max(std::exp(-tau(kt,jt,it,l)), pesc_min);
  tau.DeleteAthenaArray();
}

// Escape probabilities from the file escape_table.py wrote: one variable per energy
// band on the snapshot's block structure, plus the band edges.  Each escape-table group
// takes the band holding its centre, clamped at the ends.  The file's block order is the
// snapshot's, so the grid file's index map applies.
void ReadEscapeFile(MonteCarloBlock *pmcb) {
#ifdef HDF5OUTPUT
  MonteCarlo *pmc = pmcb->pmy_mc;
  const int ng = pmc->nescape;
  if (ng <= 0) return;
  MeshBlock *pmb = pmcb->pmy_block;
  const int nx1 = pmb->block_size.nx1, nx2 = pmb->block_size.nx2;
  const int nx3 = pmb->block_size.nx3;

  // band edges, read once
  static AthenaArray<Real> edges;
  static int nband = 0;
  if (nband == 0) {
    hid_t file = H5Fopen(escape_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in ReadEscapeFile" << std::endl
          << "cannot open <problem>/escape_file = " << escape_file << std::endl;
      ATHENA_ERROR(msg);
    }
    hid_t dset = H5Dopen(file, "pesc_edges", H5P_DEFAULT);
    hid_t space = H5Dget_space(dset);
    hsize_t dims[1];
    H5Sget_simple_extent_dims(space, dims, nullptr);
    nband = static_cast<int>(dims[0]) - 1;
    edges.NewAthenaArray(nband+1);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, edges.data());
    H5Sclose(space);
    H5Dclose(dset);
    H5Fclose(file);
    if (Globals::my_rank == 0)
      std::cout << "Escape probabilities from " << escape_file << ": " << nband
                << " bands, " << edges(0)/1.602176634e-12 << " to "
                << edges(nband)/1.602176634e-12 << " eV" << std::endl;
  }

  const int file_block = (MCGridFile::Loaded() != nullptr)
                         ? MCGridFile::Loaded()->FileIndex(pmb->gid) : pmb->gid;
  AthenaArray<Real> pesc;
  pesc.NewAthenaArray(nband,nx3,nx2,nx1);
  int start_file[5] = {0, file_block, 0, 0, 0};
  int count_file[5] = {nband, 1, nx3, nx2, nx1};
  int start_mem[4] = {0, 0, 0, 0};
  int count_mem[4] = {nband, nx3, nx2, nx1};
  HDF5ReadRealArray(escape_file.c_str(), "pesc", 5, start_file, count_file, 4,
                    start_mem, count_mem, pesc);

  // group -> band by the group's centre energy
  std::vector<int> band(ng);
  for (int l=0; l<ng; ++l) {
    const Real ec = std::exp(0.5*(pmc->escape_lne(l) + pmc->escape_lne(l+1)));
    int b = 0;
    while (b < nband-1 && ec >= edges(b+1)) ++b;
    band[l] = b;
  }
  for (int kt=0; kt<nx3; ++kt)
    for (int jt=0; jt<nx2; ++jt)
      for (int it=0; it<nx1; ++it)
        for (int l=0; l<ng; ++l)
          pmcb->escape_prob(kt,jt,it,l) = std::max(pesc(band[l],kt,jt,it), pesc_min);
#else
  (void)pmcb;
#endif
}

// The cell importance: its escape probability averaged over its emission spectrum, the
// free-free sampler's e^{-x} over the cell's energy range or the table's photon shares.
void ComputeImportance(MonteCarloBlock *pmcb) {
  MonteCarlo *pmc = pmcb->pmy_mc;
  const int ng = pmc->nescape;
  if (ng <= 0) return;
  const Real kb = 1.380649e-16;
  const AthenaArray<Real> &lne = pmc->escape_lne;
  for (int k=pmcb->ks; k<=pmcb->ke; ++k) {
    for (int j=pmcb->js; j<=pmcb->je; ++j) {
      for (int i=pmcb->is; i<=pmcb->ie; ++i) {
        const int kt = k-pmcb->ks, jt = j-pmcb->js, it = i-pmcb->is;
        Real wsum = 0., psum = 0.;
        const Real temp = pmcb->tgas(k,j,i);
        // the free-free sampler's range for this cell, in ln(energy/erg)
        const Real lo_c = logemin + (tnorm ? std::log(temp) : 0.);
        const Real hi_c = logemax + (tnorm ? std::log(temp) : 0.);
        const Real *cumv = nullptr;
        if (table) pmcb->TableEmissionCumulative(k,j,i,cumv);
        for (int l=0; l<ng; ++l) {
          const Real p = pmcb->escape_prob(kt,jt,it,l);
          Real w;
          if (table) {
            w = cumv[l+1] - cumv[l];
          } else {
            const Real d = std::min(lne(l+1),hi_c) - std::max(lne(l),lo_c);
            w = (d > 0.) ? d*std::exp(-std::exp(0.5*(lne(l)+lne(l+1)))/(kb*temp)) : 0.;
          }
          wsum += w;
          psum += w*p;
        }
        pmcb->importance(k,j,i) = (wsum > 0.) ? psum/wsum : 1.;
      }
    }
  }
}

void BuildEscapeTables(MonteCarloBlock *pmcb) {
  if (escape_file.empty())
    FillEscapeLocal(pmcb);
  else
    ReadEscapeFile(pmcb);
  ComputeImportance(pmcb);
}

void ExcludeCells(MonteCarloBlock *pmcb) {
  // The module's number densities, then no matter to speak of in the cells that take no
  // part in the radiation: below the density floor, above the temperature cut, or
  // inside the horizon.  The cells above tcut are also masked out of the table
  // opacities, which BuildOpacityTables reads after this.
  pmcb->DefaultNumberDensity();
  const Real dmin = dcut*pmcb->rho_cgs;
  const bool masked = (pmcb->tab_mask.GetSize() > 0);
  for (int k=pmcb->ks; k<=pmcb->ke; ++k) {
    for (int j=pmcb->js; j<=pmcb->je; ++j) {
      for (int i=pmcb->is; i<=pmcb->ie; ++i) {
        const bool hot = (pmcb->tgas(k,j,i) > tcut);
        if (pmcb->rho(k,j,i) < dmin || hot || InsideHorizonCell(pmcb,k,j,i)) {
          pmcb->species(0,k,j,i) = 0.;
          pmcb->species(1,k,j,i) = 0.;
        }
        if (masked) pmcb->tab_mask(k,j,i) = hot ? 0 : 1;
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
  Real m = pin->GetReal("coord", "m");

  // Calculate scalar quantities
  Real a2 = SQR(a);
  Real z2 = SQR(z);
  Real rr2 = SQR(x) + SQR(y) + z2;
  Real r2 = 0.5 * (rr2 - a2 + std::sqrt(SQR(rr2 - a2) + 4.0 * a2 * z2));
  Real r4 = SQR(r2);
  Real r = std::sqrt(r2);
  Real f = 2.0 * m * r * r2 / (r4 + a2 * z2);

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
  g(I23) = f * l_2 * l_3;
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
  g_inv(I23) = -f * l2 * l3;
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
