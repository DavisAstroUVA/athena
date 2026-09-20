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
#include <chrono>     // steady_clock
#include <string>     // c_str(), string

// Athena++ headers
#include "../athena.hpp"              // Real
#include "../athena_arrays.hpp"       // AthenaArray
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

namespace {
  // Global variables
  bool tnorm;
  Real logemin, logemax;
  Real abh, r_hor;
  Real dcut;
  Real tcut;
  constexpr Real CUT_VALUE = 1.e-20;
  std::string emission_type;
  // frequency table parameters
  int nfre, nrho, ntem;
  Real lmine, lmaxe, dle, lmint, lmaxt, dlt, lmind, lmaxd, dld;
  AthenaArray<Real> fre_grid;
  AthenaArray<Real> temp_grid;
  AthenaArray<Real> rho_grid;
  AthenaArray<Real> ross_gray_tab;
  AthenaArray<Real> plan_tab;
  AthenaArray<Real> emis_cum;
  AthenaArray<Real> emis_tot;
  AthenaArray<Real> opact;

  // Free-free fallback bookkeeping.  Two distinct things get counted: rows of the table
  // that are gray and are replaced wholesale at load, and cells whose (rho,T) falls off
  // the grid at setup.  Both are reported rather than left silent, because they decide
  // whether a run used the tabulated opacity or the analytic one.  Counting rather than
  // warning per cell is deliberate: on a mesh of this size a per-cell printf is a flood
  // that hides the very thing it is reporting.
  long long nff_cells = 0, ntab_cells = 0;
  long long noff_rho = 0, noff_temp = 0;
  long long ncut_cells = 0;  // above tcut, given CUT_VALUE instead of either
  double table_seconds = 0.;  // spent building the per-cell tables on this rank
  int ngray_rows = 0, ntable_rows = 0;

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

  //! \fn void CheckActiveCell(MonteCarloBlock *pmcb, int i3, int i2, int i1)
  //! \brief debug-only guard on the assumption opact/emis_tot/emis_cum are built around
  //
  // Those three are dimensioned on active cells, so a photon sitting in a ghost cell would
  // index past the end of its block's slice rather than into a harmless zero.  UpdateZone
  // changes a photon's status the moment it leaves the active range and these are only
  // reached while it is EVOLVING, so this cannot fire -- but the failure it guards against
  // is a silent heap overwrite, which is worth a check that costs nothing when off.
#ifdef DEBUG
  void CheckActiveCell(MonteCarloBlock *pmcb, int i3, int i2, int i1) {
    if (i1 < pmcb->is || i1 > pmcb->ie || i2 < pmcb->js || i2 > pmcb->je ||
        i3 < pmcb->ks || i3 > pmcb->ke) {
      std::stringstream msg;
      msg << "### FATAL ERROR in opacity/emission table lookup" << std::endl
          << "photon cell (" << i3 << "," << i2 << "," << i1 << ") is outside the active "
          << "range on gid " << pmcb->pmy_block->gid << "." << std::endl;
      ATHENA_ERROR(msg);
    }
  }
#else
  inline void CheckActiveCell(MonteCarloBlock *, int, int, int) {}
#endif

  void UserGetDensity(MonteCarloBlock *pmcb);
  void CartesianKerrSchild(Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3);
}

void MonteCarlo::InitUserMonteCarloData(ParameterInput *pin) {

  nuser_var = 3;

  abh = pin->GetReal("coord","a");
  // assumes mbh = 1 in code units
  r_hor = 1.0 + sqrt(1.0 - SQR(abh));
  EnrollUserWorkInMove(InsideHorizon);
  
  emission_type = pin->GetOrAddString("montecarlo","emission","none");
  // The opacity and emission tables of the table path are file-scope arrays indexed by
  // local block id and sized to the blocks this rank starts with, so they do not survive
  // a redistribution.  Refuse the combination until they are rebuilt in
  // UserWorkAfterRebalance, rather than read another block's table.
  if (emission_type != "freefree") {
    const bool balancing =
        pin->GetOrAddString("loadbalancing", "balancer", "default") != "default"
        || pin->GetOrAddString("montecarlo", "lb_test_repack", "none") != "none";
    if (balancing) {
      std::stringstream msg;
      msg << "### FATAL ERROR in MonteCarlo::InitUserMonteCarloData" << std::endl
          << "mc_readhdf_gr keeps per-block opacity tables that are not rebuilt after a"
          << " redistribution; load balancing is supported with emission = freefree only"
          << std::endl;
      ATHENA_ERROR(msg);
    }
  }
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
  // frequency integrated rosseland mean, read by GetNel to decide whether a cell is
  // cold enough to treat as neutral
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

  // rosseland mean for each frequency group
  // Read in but not used.  The values are still consumed rather than skipped, because
  // that is what leaves the file positioned at the Planck means below.
  for(int k=0; k<nfre; ++k) {
    for(int j=0; j<ntem; ++j) {
      for(int i=0; i<nrho; ++i) {
        fscanf(opac_file,"%lf",&buf);
      }
    }
  }

  // planck mean for each frequency group.  Kept as an opacity per gram (cm^2/g) rather
  // than pre-multiplied by the grid density: what gets interpolated is then kappa, which
  // varies weakly, instead of chi = kappa*rho, which carries an extra power of rho.  The
  // cell's own density is applied after the interpolation.
  for(int k=0; k<nfre; ++k) {
    for(int j=0; j<ntem; ++j) {
      for(int i=0; i<nrho; ++i) {
        fscanf(opac_file,"%lf",&(plan_tab(k,j,i)));
      }
    }
  }

  bool user_ff = pin->GetOrAddBoolean("problem", "userff", false);
  if (user_ff) {
    // Replaces plan_tab with free-free values (for testing purposes).  FreeFreeOpacity
    // returns chi in 1/cm, so divide out the grid density to store an opacity per gram.
    for(int k=0; k<nfre; ++k) {
      for(int j=0; j<ntem; ++j) {
        for(int i=0; i<nrho; ++i) {
          plan_tab(k,j,i) =
              FreeFreeOpacity(temp_grid(j),rho_grid(i),fre_grid(k)) / rho_grid(i);
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
      // Identify table values using gray opacity and replace with free-free.  A row
      // holding a zero cannot be tested by this ratio, so leave it as the file gives it
      // rather than relying on inf and NaN comparing false.
      ++ntable_rows;
      if ((min > 0.) && (max/min < 1.1)) {
        ++ngray_rows;
        for(int k=0; k<nfre; ++k) {
          plan_tab(k,j,i) =
              FreeFreeOpacity(temp_grid(j),rho_grid(i),fre_grid(k)) / rho_grid(i);
        }
      }
    }
  }
  fclose(opac_file);

  // Say so up front: a table that is gray over much of the (rho,T) plane means the run is
  // largely using the analytic free-free opacity, whatever the input file is called.
  if (Globals::my_rank == 0) {
    printf("Opacity table: %d of %d (T,rho) rows are gray and were replaced by "
           "free-free (%.1f%%)\n", ngray_rows, ntable_rows,
           100.*static_cast<Real>(ngray_rows)/static_cast<Real>(ntable_rows));
  }


  EnrollUserEmissionFunction(TableEmission);
  EnrollUserOpacityFunction(TableOpacity,true);

  EnrollUserGetNumberDensity(GetNel);

  int nx1 = pin->GetInteger("meshblock", "nx1");
  int nx2 = pin->GetInteger("meshblock", "nx2");
  int nx3 = pin->GetInteger("meshblock", "nx3");
  int nblocal =  pmy_mesh->nblocal;
  // Active cells only.  The fill loops below run ks..ke and every read comes from a
  // photon in an active cell, so a ghost layer here would be allocated, zero-filled by
  // NewAthenaArray -- which touches every page -- and then never used.  With 8x4x8 blocks
  // and NGHOST=2 that is 1152 cells carried for 256, and on a mesh of this size the
  // difference is tens of gigabytes of resident memory.
  opact.NewAthenaArray(nblocal,nx3,nx2,nx1,nfre);
  emis_tot.NewAthenaArray(nblocal,nx3,nx2,nx1);
  emis_cum.NewAthenaArray(nblocal,nx3,nx2,nx1,nfre);

}

void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin) {

  dcut = pin->GetOrAddReal("problem", "dcut",1.e-20);   // code units
  tcut = pin->GetOrAddReal("problem", "tcut",1.e20);    // Kelvin
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
    // Time the table build on this rank, summed over blocks and reported after the last
    const std::chrono::steady_clock::time_point table_start =
        std::chrono::steady_clock::now();

    int lid = pmy_block->lid;
    // Compute opacity table corresponding to each cell and frequency
    for(int k=ks; k<=ke; ++k) {
      for(int j=js; j<=je; ++j) {
        for(int i=is; i<=ie; ++i) {
          // Tables are indexed from the first active cell, not from the ghost zone.
          const int kt = k-ks, jt = j-js, it = i-is;
          Real temp = tgas(k,j,i);
          // A cell above tcut takes no part: a negligible extinction at every frequency,
          // which also makes its emissivity table below negligible.
          if (temp > tcut) {
            ++ncut_cells;
            for(int l=0; l<nfre; ++l) opact(lid,kt,jt,it,l) = CUT_VALUE;
            continue;
          }
          bool on_grid = true;
          Real ld = log10(rho(k,j,i));
          //ld = (ld < lmind) ? lmind : ld;
          //ld = (ld > lmaxd) ? lmaxd : ld;
          Real lt = log10(temp);
          //lt = (lt < lmint) ? lmint : lt;
          //lt = (lt > lmaxt) ? lmaxt : lt;
          Real xi = (ld - lmind) / dld;
          int ii = std::floor(xi);
          if (ii < 0) {
            ii = 0;
            ++noff_rho;
            on_grid = false;
          } else if (ii > nrho-2) {
            ii = nrho-2;
            ++noff_rho;
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
          // Test the interpolation weights, not the bracket index.  jj was clamped into
          // [0, ntem-2] above and neither while loop can take it back out, so a test on
          // jj alone can never fire; a cell off either end of the temperature grid shows
          // up here instead, as a weight outside [0,1].  Extrapolating on it can drive
          // plan_tab negative, which makes emis_cum non-monotonic and the bisection in
          // SampleEmissivity meaningless.  Same for the density weight.
          // Fractional position in log T, to match xi, which is already fractional in
          // log rho.  The grid is log-spaced in both (the temperature axis unevenly so,
          // which is why jj came from a search rather than a formula).
          xj = std::log(temp/temp_grid(jj))
               / std::log(temp_grid(jj+1)/temp_grid(jj));
          if ((xj < 0.) || (xj > 1.)) {
            ++noff_temp;
            on_grid = false;
          }
          // xi can only leave [0,1] when ii was clamped, which already counted it.
          if ((xi < 0.) || (xi > 1.)) on_grid = false;
          if (on_grid) {
            ++ntab_cells;
            const Real rhoc = rho(k,j,i);
            for(int l=0; l<nfre; ++l) {
              const Real k00 = plan_tab(l,jj  ,ii  ), k10 = plan_tab(l,jj+1,ii  );
              const Real k01 = plan_tab(l,jj  ,ii+1), k11 = plan_tab(l,jj+1,ii+1);
              Real kap;
              // Log-log in (rho,T).  These opacities are power laws over most of the
              // plane, so interpolating the logarithm is far closer to the truth than
              // interpolating the value: reconstructing dropped points from the 64-group
              // table gives a median error of 2.5% this way against 7.8% linearly in
              // temperature, and 0.0% against 3.3% in density.  Any non-positive corner
              // drops back to linear, where the logarithm is not defined.
              if ((k00 > 0.) && (k10 > 0.) && (k01 > 0.) && (k11 > 0.)) {
                kap = std::exp((1.-xi)*((1.-xj)*std::log(k00) + xj*std::log(k10))
                                 + xi *((1.-xj)*std::log(k01) + xj*std::log(k11)));
              } else {
                kap = (1.-xi)*((1.-xj)*k00 + xj*k10) + xi*((1.-xj)*k01 + xj*k11);
              }
              // plan_tab is per gram; the extinction coefficient carries the cell's own
              // density, so the strong rho dependence is exact rather than interpolated.
              opact(lid,kt,jt,it,l) = kap * rhoc;
            }
          } else {
              ++nff_cells;
              // FreeFreeOpacity already returns chi in 1/cm for this cell's density, so
              // this branch needs no further factor of rho.
              for(int l=0; l<nfre; ++l) {
                opact(lid,kt,jt,it,l) = FreeFreeOpacity(temp,rho(k,j,i),fre_grid(l));
              }
          }
        }
      }
    }

    // Compute emissivity table for each cell and frequncy
    AthenaArray<Real> eta_nu_tab;
    eta_nu_tab.NewAthenaArray(nx3,nx2,nx1,nfre);
    Real h_cgs = 6.62607015e-27;
    for(int l=0; l<nfre; ++l) {
      Real nu = fre_grid(l)/h_cgs;
      for(int k=ks; k<=ke; ++k) {
        for(int j=js; j<=je; ++j) {
          for(int i=is; i<=ie; ++i) {
            Real temp = tgas(k,j,i);
            eta_nu_tab(k-ks,j-js,i-is,l) = Planck(temp,nu)
                                          * opact(lid,k-ks,j-js,i-is,l);
          }
        }
      }
    }

    // Compute integratred emission tables (total and cumulative) for each cell
    for(int k=ks; k<=ke; ++k) {
      for(int j=js; j<=je; ++j) {
        for(int i=is; i<=ie; ++i) {
          const int kt = k-ks, jt = j-js, it = i-is;
          emis_cum(lid,kt,jt,it,0) = 0.;
          for(int l=1; l<nfre; ++l) {
            Real nup = fre_grid(l)/h_cgs;
            Real num = fre_grid(l-1)/h_cgs;
            Real dlnu = std::log(nup/num);
            Real eta_ave = 0.5*(eta_nu_tab(kt,jt,it,l)+eta_nu_tab(kt,jt,it,l-1));
            emis_cum(lid,kt,jt,it,l) = emis_cum(lid,kt,jt,it,l-1)
                                       + 4.*PI/h_cgs*eta_ave*dlnu;
          }
          emis_tot(lid,kt,jt,it) = emis_cum(lid,kt,jt,it,nfre-1);
          // A cell with no emission leaves the cumulative array at zero rather than
          // dividing by it.  Cells are drawn uniformly in SetEmissionCellWeight and only
          // then weighted by the emission array, so a non-emitting cell is still handed
          // to SampleEmissivity; normalizing here would give it a table of NaNs, which
          // the zero weight would not stop from reaching the opacities.
          if (emis_tot(lid,kt,jt,it) > 0.) {
            for(int l=1; l<nfre; ++l) {
              emis_cum(lid,kt,jt,it,l) /= emis_tot(lid,kt,jt,it);
            }
          }
          // A cell above tcut emits CUT_VALUE outright, whatever the Planck function at
          // its temperature made of the CUT_VALUE extinction.  Its cumulative array,
          // normalized above from that shape, still gives SampleEmissivity a valid
          // distribution for the negligible weight it will carry.
          if (tgas(k,j,i) > tcut) emis_tot(lid,kt,jt,it) = CUT_VALUE;
        }
      }
    }
    eta_nu_tab.DeleteAthenaArray();

    table_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - table_start).count();
    if (Globals::my_rank == 0 && lid == pmy_block->pmy_mesh->nblocal - 1) {
      std::cout << "  opacity and emission tables on rank 0: "
                << pmy_block->pmy_mesh->nblocal << " blocks, " << table_seconds << " s"
                << std::endl;
    }
  }

}


//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//! \brief report how many cells used the tabulated opacity and how many fell back to
//!        analytic free-free
//
// Accumulated per block during setup and summed here, which is a point every rank reaches
// exactly once (main calls it unconditionally), so the reduction cannot mismatch.  The
// gray-row fraction is reported separately at load; this is the complementary number --
// how much of the actual domain landed off the grid, and on which axis.
//========================================================================================

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {

  if (emission_type == "freefree") return;  // no table was ever read

  long long tot[5] = {ntab_cells, nff_cells, noff_rho, noff_temp, ncut_cells};
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, tot, 5, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (Globals::my_rank == 0) {
    const long long ncell = tot[0] + tot[1];
    printf("Opacity source: %lld cells from the table, %lld off-grid using free-free "
           "(%.2f%%)\n", tot[0], tot[1],
           (ncell > 0) ? 100.*static_cast<Real>(tot[1])/static_cast<Real>(ncell) : 0.);
    if (tot[1] > 0)
      printf("                off-grid in density: %lld, in temperature: %lld\n",
             tot[2], tot[3]);
    if (tot[4] > 0)
      printf("                %lld cells above tcut = %g K given opacity and emission "
             "%g\n", tot[4], tcut, CUT_VALUE);
  }
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
    pphot->statp[ip] = REMOVED;
    //printf("Photon absorbed inside horizon at r=%g\n",r);
  }
  //Real keverg = 1.602176634e-9;
  //if (pphot->ep[ip] > 2.e3*keverg)
  //  pphot->statp[ip] = DESTROYED;
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
  int lid = pmcb->pmy_block->lid;
  CheckActiveCell(pmcb,i3,i2,i1);
  const int t3 = i3-pmcb->ks, t2 = i2-pmcb->js, t1 = i1-pmcb->is;
  return (1.-xk) * opact(lid,t3,t2,t1,k) + xk * opact(lid,t3,t2,t1,k+1);

}

Real IntegrateEmission(Real temp, Real num, Real nup, Real am, Real ap) {

  int n = 20;
  Real h_cgs = 6.62607015e-27;
  Real dlnu = std::log(nup/num)/static_cast<Real>(n);
  Real dadnu = (ap-am)/(nup-num);
  Real lnu = std::log(num);
  Real sum = Planck(temp,num)*am*dlnu/h_cgs/2.;
  // Interior nodes run to n-1: the composite trapezoid rule over n intervals weights
  // nodes 1..n-1 fully and the two endpoints by a half.
  for(int i=1; i<n; ++i) {
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
  CheckActiveCell(pmcb,i3,i2,i1);
  return emis_tot(lid,i3-pmcb->ks,i2-pmcb->js,i1-pmcb->is);
}


Real SampleEmissivity(MonteCarloBlock *pmcb, Photon *pphot, int ip) {

  Real dev = pmcb->pran->uniform();
  int i1 = pphot->i1p[ip];
  int i2 = pphot->i2p[ip];
  int i3 = pphot->i3p[ip];
  int lid = pmcb->pmy_block->lid;

  CheckActiveCell(pmcb,i3,i2,i1);
  Real *prob = &(emis_cum(lid,i3-pmcb->ks,i2-pmcb->js,i1-pmcb->is,0));
  // A non-emitting cell has an unnormalized (all-zero) cumulative array; it can still be
  // drawn, since cells are picked uniformly and only then weighted.  There is no spectrum
  // to sample, and the photon carries zero weight, so return the lowest tabulated energy
  // rather than dividing by a zero bin width.
  if (prob[nfre-1] <= 0.) return fre_grid(0);
  int i = mcbisect(dev,prob,nfre);
  Real a = (dev-prob[i])/(prob[i+1]-prob[i]);
  Real a1 = 1.-a;
  //printf("%d %g %g\n",i,a,a1);
  if ((a < 0.) || (a > 1.)) {
    printf("%d %d %d\n",i3,i2,i1);
    for (int j=0; j< nfre; ++j)
      printf("%d %e\n",j,1-prob[j]);
    printf("%d %g %g %g %g\n",i,dev,fre_grid(i),a,a1);
  }
  // Log interpolation within the bin.  The frequency grid is log-spaced -- 0.0625 dex
  // for the 64-group table -- so drawing linearly in energy inside a bin puts the sample
  // systematically high; interpolating the logarithm places it where the grid says.
  Real nu = std::exp(a*std::log(fre_grid(i+1)) + a1*std::log(fre_grid(i)));
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
        // below the density floor or above the temperature cut: no matter to speak of
        Real rho = pmcb->rho(k,j,i);
        if (rho < dmin || pmcb->tgas(k,j,i) > tcut) rho = 1.e-30;
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
        // above the temperature cut: no matter to speak of, as in GetNelFloor
        if (pmcb->tgas(k,j,i) > tcut) rho = 1.e-30;
        Real nh = rho / (mp*(1.+4.*heabund));
        Real nhe = nh*heabund;
        // species(1) is the ion density read by the free-free opacity and emission in
        // opacity.cpp and emission.cpp.  Set it here as GetNelFloor and the default in
        // MonteCarloBlock do, so this hook does not depend on the table functions fully
        // displacing those paths.
        pmcb->species(1,k,j,i) = nh + 4. * nhe;
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

void UserGetDensity(MonteCarloBlock *pmcb) {

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
	Real dx2 = pmcb->pmy_block->pcoord->dx2f(j);
	Real dx3 = pmcb->pmy_block->pcoord->dx3f(k);
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
