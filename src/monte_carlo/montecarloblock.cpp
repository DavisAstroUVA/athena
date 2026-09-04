//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file montecarloblock.cpp
//! \brief implementation of functions in class MonteCarloBlock

// C++ headers
#include <cstring>   // strcmp
#include <iostream>
#include <stdexcept>  // runtime_error

// Athena++ headers
#include "montecarlo.hpp"
#include "polarization.hpp"
#include "tetrad.hpp"
#include "photon.hpp"
#include "photonpusher.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../mesh/mesh.hpp"
#include "../hydro/hydro.hpp"
#include "../globals.hpp"
#include "../scalars/scalars.hpp"


//----------------------------------------------------------------------------------------
//! MonteCarloBlock constructor, builds MonteCarloBlock from parameter input

MonteCarloBlock::MonteCarloBlock(MeshBlock *pmb,  MCBlockSize *pblsize, MonteCarlo *pmc,
                                 ParameterInput *pin) {

  pmy_mc = pmc;

  // Set related meshblock, coordinate
  pmy_block = pmb;

  // Set pointer to this monte carlo block in pmb if not nullptr
  if (pmb != nullptr) {
    pmb->pmy_mcb = this;
  }

  // Construct pointer to photon
  pphot  = new Photon(this,pin);

  // Initialize to nullptr and set below
  ppusher = nullptr;
  pcoord = nullptr;

  // get seed and intitialize randon number generator
  int rank = Globals::my_rank;
  int iseed = pmy_mc->iseed+pmy_block->gid*10;  // temporary solution

  pran = new MCRandom(iseed);

  next=nullptr;

  // SWD: eliminate some or all of these?
  // set local flags based on monte_carlo
  coord_system = pmy_mc->coord_system;
  topology = pmy_mc->topology;
  curved_metric = pmy_mc->curved_metric;
  boosts = pmy_mc->boosts;
  coupled = pmy_mc->coupled;
  acceleration = pmy_mc->acceleration;
  time_acc = pmy_mc->time_acc;
  // set in mcoutput if output requested
  mom_flag_lab = pmy_mc->pmcout->mom_flag_lab;
  mom_flag_com = pmy_mc->pmcout->mom_flag_com;
  mom_flag_coord = pmy_mc->pmcout->mom_flag_coord;
  // Comoving moments currently computed from lab moments. This flag accumulates them directly
  accumulate_com = pin->GetOrAddBoolean("montecarlo","accumulate_comoving",false);
  if (mom_flag_com && !boosts) {
    std::stringstream msg;
    msg << "FATAL ERROR: comoving frame moments requested but booosts set to false."
        << std::endl;
    ATHENA_ERROR(msg);
  }
  mom_flag_src = pmy_mc->pmcout->mom_flag_src;
  mom_flag_usr = pmy_mc->pmcout->mom_flag_usr || (pmy_mc->nuser_mom > 0);
  mom_flag_scat = pmy_mc->pmcout->mom_flag_scat;

  call_srcterms = coupled || mom_flag_src;
  // Call UpdateMoments if any of these moments flags are set or source terms requested
  call_moments = mom_flag_lab || mom_flag_com || mom_flag_coord || call_srcterms
                 || mom_flag_usr || mom_flag_scat;
  // Set boundary values for this block
  SetBoundaryValues(pmy_mc->mc_bcs);

  // Initialize pbval after mcb_bcs is set
  pbval = new MCBoundaryValues(this,pin);

  // Setup outputs
  pspec = pmy_mc->pmcout->pspec;
  pphlist = pmy_mc->pmcout->pphlist;
  ptraj = pmy_mc->pmcout->ptraj;

  // set local mesh parameters to correspond to mesh block
  if (pmb != nullptr) {
    is = pmb->is; ie = pmb->ie;
    js = pmb->js; je = pmb->je;
    ks = pmb->ks; ke = pmb->ke;
    nx1 = pmb->block_size.nx1;
    nx2 = pmb->block_size.nx2;
    nx3 = pmb->block_size.nx3;
  } else {
    if (pblsize == nullptr) {
      std::stringstream msg;
      msg << "### FATAL ERROR Monte Carlo Block Constructor" << std::endl
          << "Both input Mesh Block and Block Size are nullptr." << std::endl;
      ATHENA_ERROR(msg);
    } else {
      is = pblsize->is; ie = pblsize->ie;
      js = pblsize->js; je = pblsize->je;
      ks = pblsize->ks; ke = pblsize->ke;
      nx1 = pblsize->nx1;
      nx2 = pblsize->nx2;
      nx3 = pblsize->nx3;
      //pcoord = new MCCoord(nx1+2*(NGHOST),nx2+2*(NGHOST),nx3+2*(NGHOST),acceleration);
    }
  }
  // default cgs conversion from code units
  rho_cgs = pin->GetOrAddReal("problem","rho_cgs",1.);
  vel_cgs = pin->GetOrAddReal("problem","vel_cgs",1.);
  tgas_cgs = pin->GetOrAddReal("problem","tgas_cgs",-1.);
  tfloor_cgs = pin->GetOrAddReal("problem","tfloor_cgs",0.);
  tceiling_cgs = pin->GetOrAddReal("problem","tceiling_cgs",HUGE_NUMBER);
  l_cgs = pin->GetOrAddReal("problem","l_cgs",1.);
  time_cgs = l_cgs/vel_cgs;
  betamax = pin->GetOrAddReal("problem","betamax",0.999);

  // SWD:  stepsize control needs to be modified
  stepsize = pin->GetOrAddReal("montecarlo","stepsize",1.0e-3);

  // Flags for handling photon steps.  boyerlindquist is not read here any more: it is one
  // of the inputs SetCoordinateSystem folds into coord_system.
  orthotet_flag = pin->GetOrAddBoolean("montecarlo", "orthotet", false);
  varystep_flag = pin->GetOrAddBoolean("montecarlo", "varystep", false);

  // Scattering
  scattering_meth = GetScatteringFlag(pin->GetOrAddString("montecarlo","scattering",
                                                          "none"));
  if (scattering_meth == SCATUSER) {
    ScatteringOpacity = pmy_mc->UserScatteringOpacity;
    Scatter = pmy_mc->UserScattering;
  } else if (scattering_meth == SCATNONE) {
    ScatteringOpacity = NoOpacity;
    Scatter = NoScatter;  // should not be called
    coherent_scattering = true;
  } else if (scattering_meth == SCATISO) {
    if (IsPolarized(pmy_mc->polarized)) {
      std::stringstream msg;
      msg << "### ERROR in MonteCarloBlock constructor" << std::endl
          << "Istropic scattering not suppored for polarized = "
          << GetMCPolarizationName(pmy_mc->polarized) << std::endl;
      ATHENA_ERROR(msg);
    } else {
      ScatteringOpacity = ThomsonOpacity;
      Scatter = ScatterIsotropic;
      coherent_scattering = true;
    }
  } else if (scattering_meth == SCATTHOM) {
    ScatteringOpacity = ThomsonOpacity;
    if (IsPolarized(pmy_mc->polarized)) {
      Scatter = ScatterThomsonPolarized;
    } else
      Scatter = ScatterThomsonUnpolarized;
    coherent_scattering = true;
  } else if (scattering_meth == SCATCOMP) {
    int comptonio = pin->GetOrAddInteger("montecarlo","comptonio",1);
    if (pmy_block->gid == 0) {
      if (comptonio > 0)
        std::cout << "Creating table for Compton cross section." << std::endl;
      else
        std::cout << "Reading in table for Compton cross section." << std::endl;
    }
    GenerateComptonTable(comptonio);
    ScatteringOpacity = ComptonOpacity;
    if (IsPolarized(pmy_mc->polarized)) {
      Scatter = ScatterComptonPolarized;
    } else {
      Scatter = ScatterComptonUnpolarized;
    }
    coherent_scattering = false;
  } else if (scattering_meth == SCATRES) {
    ScatteringOpacity = ResonanceLineOpacity;
    if (IsPolarized(pmy_mc->polarized)) {
      std::stringstream msg;
      msg << "### ERROR in MonteCarloBlock constructor" << std::endl
          << "Lyman alpha scattering not suppored for polarized = "
          << GetMCPolarizationName(pmy_mc->polarized) << std::endl;
      ATHENA_ERROR(msg);
    } else {
      Scatter = ScatterResonanceLine;
      coherent_scattering = false;
    }
  } else if (scattering_meth == SCATDUST) {
    ScatteringOpacity = DustScatteringOpacity;
    if (IsPolarized(pmy_mc->polarized)) {
      Scatter = ScatterDust;
      coherent_scattering = true;
    } else {
      std::stringstream msg;
      msg << "### ERROR in MonteCarloBlock constructor" << std::endl
          << "Dust scattering not suppored for polarized = "
          << GetMCPolarizationName(pmy_mc->polarized) << std::endl;
      ATHENA_ERROR(msg);
    }
  }

  // Set up photon movement and initialization methods
  computedmin = false;
  if (acceleration)
    computedmin = true;
  pmy_mc->computedmin = computedmin;
  tetrads = true;
  // Number of cells including ghosts, for the standalone (pmb == nullptr) constructors.
  const int mc1 = nx1+2*(NGHOST), mc2 = nx2+2*(NGHOST), mc3 = nx3+2*(NGHOST);
  // The MCCoord(Coordinates*, ...) constructors copy mass and spin off the Athena
  // coordinate object, so they only have to be read from the input when there is no
  // MeshBlock to copy from.
  bool set_bh_params = false;

  switch (coord_system) {
    case MCCOORD_CARTESIAN:
      GetZonePosition = GetZonePositionCartesian;
      if (pmy_mc->general_pusher_flag) {
        ppusher = new GeneralPusher(this);
        pcoord = (pmb != nullptr) ? new MCCartesian(pmb->pcoord,this)
                                  : new MCCartesian(mc1,mc2,mc3,computedmin);
      } else {
        tetrads = false;
        ppusher = new CartesianPusher(this);
        pcoord = (pmb != nullptr) ? new MCCoord(pmb->pcoord,this)
                                  : new MCCoord(mc1,mc2,mc3,computedmin);
      }
      break;

    case MCCOORD_SPHERICAL_POLAR:
      GetZonePosition = GetZonePositionSphericalPolar;
      if (pmy_mc->general_pusher_flag) {
        ppusher = new GeneralPusher(this);
        pcoord = (pmb != nullptr) ? new MCSphericalPolar(pmb->pcoord,this)
                                  : new MCSphericalPolar(mc1,mc2,mc3,computedmin);
      } else {
        tetrads = false;
        ppusher = new SphericalPolarPusher(this);
        pcoord = (pmb != nullptr) ? new MCCoord(pmb->pcoord,this)
                                  : new MCCoord(mc1,mc2,mc3,computedmin);
      }
      break;

    case MCCOORD_CYLINDRICAL:
      GetZonePosition = GetZonePositionCylindrical;
      ppusher = new GeneralPusher(this);
      pcoord = (pmb != nullptr) ? new MCCylindrical(pmb->pcoord,this)
                                : new MCCylindrical(mc1,mc2,mc3,computedmin);
      break;

    case MCCOORD_MINKOWSKI:
      GetZonePosition = GetZonePositionCartesian;
      ppusher = new GeneralPusher(this);
      pcoord = (pmb != nullptr) ? new MCMinkowski(pmb->pcoord,this)
                                : new MCMinkowski(mc1,mc2,mc3,computedmin);
      break;

    case MCCOORD_KERR_SCHILD:
      GetZonePosition = GetZonePositionSphericalPolar;//approximate
      ppusher = new GeneralPusher(this);
      pcoord = (pmb != nullptr) ? new MCKerrSchild(pmb->pcoord,this)
                                : new MCKerrSchild(mc1,mc2,mc3,computedmin);
      set_bh_params = (pmb == nullptr);
      break;

    case MCCOORD_BOYER_LINDQUIST:
      GetZonePosition = GetZonePositionSphericalPolar;//approximate
      ppusher = new GeneralPusher(this);
      pcoord = (pmb != nullptr) ? new MCBoyerLindquist(pmb->pcoord,this)
                                : new MCBoyerLindquist(mc1,mc2,mc3,computedmin);
      set_bh_params = (pmb == nullptr);
      break;

    case MCCOORD_KERR_SCHILD_CARTESIAN:
      GetZonePosition = GetZonePositionCartesian;
      ppusher = new GeneralPusher(this);
      // Previously this branch built no coordinate object at all when pmb was nullptr and
      // then dereferenced it for SetSpin, and read coord/a and coord/m unconditionally
      // even though the MeshBlock path already supplies them.
      pcoord = (pmb != nullptr) ? new MCKerrSchildCartesian(pmb->pcoord,this)
                                : new MCKerrSchildCartesian(mc1,mc2,mc3,computedmin);
      set_bh_params = (pmb == nullptr);
      break;

    case MCCOORD_SNAKE:
      GetZonePosition = GetZonePositionCartesian;
      ppusher = new GeneralPusher(this);
      pcoord = (pmb != nullptr) ? new MCSnake(pmb->pcoord,this)
                                : new MCSnake(mc1,mc2,mc3,computedmin);
      // Not <coord>/a: gr_user requires that name for the black hole spin and GRUser
      // reads it unconditionally, so the shear amplitude needs its own key.  Set on both
      // paths, since the MCCoord(Coordinates*) constructor has no snake parameters to
      // copy the way it copies mass and spin.
      static_cast<MCSnake*>(pcoord)->SetSnakeParams(
          pin->GetOrAddReal("coord", "snake_a", 0.0),
          pin->GetOrAddReal("coord", "snake_k", 0.0));
      break;
  }

  if (set_bh_params) {
    pcoord->SetSpin(pin->GetReal("coord", "a"));
    pcoord->SetMass(pin->GetReal("coord", "m"));
  }
  pmy_mc->tetrads = tetrads;

  // Set pcoord in ppusher
  ppusher->pcoord = pcoord;

  // Set absorption opacity type
  absorption_opac = GetAbsorptionOpacityFlag(pin->GetOrAddString("montecarlo",
                                                                 "absorption","none"));
  if (absorption_opac == ABSUSER) {
    AbsorptionOpacity = pmy_mc->UserAbsorptionOpacity;
  } else if (absorption_opac == ABSNONE) {
    AbsorptionOpacity = NoOpacity;
  } else if (absorption_opac == ABSFF) {
    AbsorptionOpacity = FreeFreeAbsorptionOpacity;
  } else if (absorption_opac == ABSDUST) {
    AbsorptionOpacity = DustAbsorptionOpacity;
  }
  // Get number of species
  nspec = pin->GetOrAddInteger("problem","nspec",2);

  // Allocate (/initialize) variable arrays needed for evolution/output
  int ncells1 = nx1 + 2*(NGHOST);
  int ncells2 = 1, ncells3 = 1;
  if (nx2 > 1) ncells2 = nx2 + 2*(NGHOST);
  if (nx3 > 1) ncells3 = nx3 + 2*(NGHOST);
  rho.NewAthenaArray(ncells3,ncells2,ncells1);
  species.NewAthenaArray(nspec,ncells3,ncells2,ncells1);
  tgas.NewAthenaArray(ncells3,ncells2,ncells1);
  if (boosts || tetrads) {
    boost_cmv.NewAthenaArray(ncells3,ncells2,ncells1,4,4);
    boost_lab.NewAthenaArray(ncells3,ncells2,ncells1,4,4);
  }
  // The two storage conventions, which are deliberately not the same thing and used to
  // share one array name:
  //
  //   flat spacetime -- vel holds (gamma, gamma*beta^i) in the orthonormal frame.  The
  //     metric is constant across a cell, so a vector normalized at the cell center is
  //     still normalized anywhere in it and there is nothing to reconstruct.  Consumers
  //     divide by vel(...,0) to recover beta^i.
  //
  //   general relativity -- uprim holds the primitive uu^i and there is no stored
  //     four-velocity at all.  FluidFourVelocity assembles one on demand at whatever
  //     point it is asked about, which is the only way to get u.u = -1 where the vector
  //     is used rather than only where it was built.
  if (GENERAL_RELATIVITY) {
    uprim.NewAthenaArray(ncells3,ncells2,ncells1,3);
  } else if (boosts || IsPolarized(pmy_mc->polarized)) {
    vel.NewAthenaArray(ncells3,ncells2,ncells1,4);
    if (!boosts) {
      // Value is constant in time, unlike the fluid velocity, so it is set once rather than
      // refreshed each cycle. g_tt = -1 in every flat metric the module supports, so
      // u = (1,0,0,0) is already normalized; ConstructTetrad renormalizes regardless.
      for (int k=0; k<ncells3; ++k) {
        for (int j=0; j<ncells2; ++j) {
          for (int i=0; i<ncells1; ++i) {
            vel(k,j,i,0) = 1.0;
            vel(k,j,i,1) = 0.0;
            vel(k,j,i,2) = 0.0;
            vel(k,j,i,3) = 0.0;
          }
        }
      }
    }
  }
  if (NSCALARS > 0) scalars.NewAthenaArray(ncells3,ncells2,ncells1);
  // moments is 1 (Er) + 3 (Fr) + 9 (Pr) + 1 (Eave) + 1 (net cool)
  nmom = 13;
  int ntype = pmy_mc->ntype;
  if (mom_flag_lab) moments.NewAthenaArray(ntype,nmom,ncells3,ncells2,ncells1);
  if (mom_flag_com) moments_com.NewAthenaArray(ntype,nmom,ncells3,ncells2,ncells1);
  if (mom_flag_coord)
    moments_coord.NewAthenaArray(ntype,nmom,ncells3,ncells2,ncells1);
  if (pmy_mc->nuser_mom > 0)
    moments_user.NewAthenaArray(pmy_mc->nuser_mom,ncells3,ncells2,ncells1);
  nsrc = 10;
  if (call_srcterms) sourceterms.NewAthenaArray(nsrc,ncells3,ncells2,ncells1);
  if (mom_flag_scat) {
    nf_scat = pin->GetInteger("montecarlo","nf_scat");
    Real everg = 1.602176634e-12;
    emin_scat = pin->GetReal("montecarlo","emin_scat") * everg;
    emax_scat = pin->GetReal("montecarlo","emax_scat") * everg;
    moments_scat.NewAthenaArray(nf_scat,ncells3,ncells2,ncells1);
    moments_scat_error.NewAthenaArray(nf_scat,ncells3,ncells2,ncells1);
    dloge_scat = (std::log10(emax_scat/emin_scat))/static_cast<Real>(nf_scat);
    energy_scat.NewAthenaArray(nf_scat+1);
    freq_scat_mid.NewAthenaArray(nf_scat);
    Real h_cgs = 6.62607015e-27;
    for (int i=0; i<=nf_scat; i++) {
      energy_scat(i) = std::log10(emin_scat) + static_cast<Real>(i)*dloge_scat; // keep log
      if (i > 0) {
        //        freq_scat_mid(i-1) = 0.5*( pow(10.,energy_scat(i-1)) + pow(10.,energy_scat(i)) )/h_cgs;
        freq_scat_mid(i-1) = pow(10.,0.5*(energy_scat(i-1) + energy_scat(i)))/h_cgs;
      }
    }
    if ((Globals::my_rank ==0) && (pmy_block->lid == 0)) {
      FILE *pfile;
      std::stringstream msg;
      if ((pfile = fopen("sourceterm_frequencies.txt","w")) == NULL) {
        msg << "### FATAL ERROR in MonteCarloBlock Constructor" << std::endl
            << "Output file sourceterm_frequencies.txt could not be opened";
        throw std::runtime_error(msg.str().c_str());
      }
      fprintf(pfile,"%d\n", nf_scat);
      for (int i=0; i< nf_scat; i++) {
        fprintf(pfile,"%d %e %e %e\n",i,freq_scat_mid(i),pow(10.,energy_scat(i))/h_cgs,
                pow(10.,energy_scat(i+1))/h_cgs);
      }
      fclose(pfile);
    }
  }
  if (pmy_mc->emission_array) emission.NewAthenaArray(ncells3,ncells2,ncells1);
  if (pmy_mc->emission_eqwt[0]) emit_count_.NewAthenaArray(ncells3,ncells2,ncells1);
  if (acceleration && !(coherent_scattering) && !(scattering_meth == SCATRES)) {
    planck_opacity.NewAthenaArray(ncells3,ncells2,ncells1);
    planck_inv_opacity.NewAthenaArray(ncells3,ncells2,ncells1);
  }

 // Create user monte carlo block data
  InitUserMonteCarloBlockData(pin);

}

//----------------------------------------------------------------------------------------
//! destructor

MonteCarloBlock::~MonteCarloBlock() {

  delete pphot;
  delete ppusher;
  delete pcoord;
  delete pbval;
  delete pran;
  //delete pspec;
  //delete pphlist;
  //delete ptraj;

  rho.DeleteAthenaArray();
  species.DeleteAthenaArray();
  tgas.DeleteAthenaArray();
  if (boosts || tetrads) {
    boost_cmv.DeleteAthenaArray();
    boost_lab.DeleteAthenaArray();
  }
  // Unconditional: DeleteAthenaArray handles the never-allocated case, and matching the
  // allocation conditions by hand is how vel came to be leaked on flat polarized runs,
  // where it was allocated but not freed.
  vel.DeleteAthenaArray();
  uprim.DeleteAthenaArray();
  if (NSCALARS > 0) scalars.DeleteAthenaArray();
  if (mom_flag_lab) moments.DeleteAthenaArray();
  if (mom_flag_com) moments_com.DeleteAthenaArray();
  if (mom_flag_coord) moments_coord.DeleteAthenaArray();
  if (pmy_mc->nuser_mom > 0) moments_user.DeleteAthenaArray();
  if (call_srcterms) sourceterms.DeleteAthenaArray();
  if (pmy_mc->emission_array) emission.DeleteAthenaArray();
  if (pmy_mc->emission_eqwt[0]) emit_count_.DeleteAthenaArray();
  if (acceleration && !(coherent_scattering) && !(scattering_meth == SCATRES)) {
    planck_opacity.DeleteAthenaArray();
    planck_inv_opacity.DeleteAthenaArray();
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::RayTracePhotonsOnBlock(int etype)
//! \brief Integrate photons to termination condtion without scattering

void MonteCarloBlock::RayTracePhotonsOnBlock(int etype) {

  Real const to_comv = 1.0;
  Real const to_eulr = -1.0;
  int nbuf = 0;

  printf("remain: %d \n",nphremain);
  // Emit photons to replace those that left meshblock or were terminated
  // Limit ntodo to number of remaining photons on block
  int ntodo = (loop_max_size > nphremain) ? nphremain : loop_max_size;

  // if photons remain to transfer, make space for new photons
  if (ntodo > 0) {
    int nold = pphot->nphot;
    pphot->AllocatePhotons(nold+ntodo);
    nphremain -= ntodo;
    nphrun += ntodo;

    // user definied photon initialization
    InitializePhoton(pphot,nold,pphot->nphot-1,etype);
    //if (ptraj != nullptr) {
    //  for (int ip=nold; ip < pphot->nphot; ip++)
    //    ptraj->InitializeTrajectory(pphot->trp[ip]);
    //}
  }
  int ntot = pphot->nphot;

  // Photon initialized in coordinate frame
  // move photon until  stopping condition
  ppusher->Move(pphot,0,pphot->nphot-1);

  for (int ip=pphot->nphot-1; ip >= 0; ip--) {
    if (pphot->statp[ip] != EVOLVING) {

      if (pphot->statp[ip] != BUFFERED) {
        // Bring the Stokes parameters up to date with the transported coherency tensor
        // before finalizing the photon, writing outputs
        if (IsPolarized(pmy_mc->polarized)) CoherencyToObserverStokes(this, pphot, ip);
        // User defined completion work
        FinalizePhoton(pphot,ip);

        //if (ptraj != nullptr) {
        //  ptraj->CompleteTrajectory(pphot->trp[ip]);
        //}
      }
      if (pphot->statp[ip] == ESCAPED) {
        // loop over outputs for escaping photons and update
        Spectrum *pspect = pspec;
        while (pspect != nullptr) {
          pspect->UpdateSpectrum(pphot,ip);
          pspect = pspect->next;
        }
        if (pphlist != nullptr) {
          pphlist->AddPhoton(pphot,ip);
        }
        nesc++;
        pphot->RemoveOneParticle(ip);
      } else if (pphot->statp[ip] == ABSORBED) {
        nabs++;
        pphot->RemoveOneParticle(ip);
      } else if (pphot->statp[ip] == DESTROYED) {
        ndes++;
        pphot->PrintPhoton("destroy in ray tracing",ip);
        pphot->RemoveOneParticle(ip);
      } else if (pphot->statp[ip] == BUFFERED) {
        nbuf++;
      } else if (pphot->statp[ip] == REMOVED) {
        pphot->RemoveOneParticle(ip);
        nrem++;
      }
    }
  } // end loop over ip
  //if ((pphot->nphot > 0) && (Globals::my_rank == 0))
  //  pphot->PrintPhoton(pphot->nphot-1);
  /*std::cout  << "rank, ntot, nnew, nesc, nabs, ndes, nbuf, nscat: " << Globals::my_rank
             << ' ' << ntot << ' ' << ntodo << ' ' << nesc
             << ' ' << nabs << ' ' << ndes << ' ' << nbuf
             << ' ' << nscat << std::endl;*/
}


//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::TransferPhotonsOnBlock(int etype)
//! \brief perform radiation transfer

void MonteCarloBlock::TransferPhotonsOnBlock(int etype) {

  // Set absorption method for this photon type
  enum AbsorptionMethodFlag absorption_meth = pmy_mc->absorption_method[etype];

  //int nbuf = 0;
  int nold = pphot->nphot;
  int ntot = nold + nphremain;

  // Emit photons to replace those that left meshblock or were terminated
  // limit ntot < loop_max_size unless nold is larger than loop_max_size
  ntot = (loop_max_size > ntot) ? ntot : loop_max_size;
  ntot = (nold > ntot) ? nold : ntot;
  int nnew = ntot - nold;

  if (ntot == 0) // nothing to do
    return;

  // if photons remain to transfer, make space for new photons
  if (nnew > 0) {
    pphot->AllocatePhotons(ntot);
    nphremain -= nnew;
    nphrun += nnew;

    // user definied photon initialization
    InitializePhoton(pphot,nold,pphot->nphot-1,etype);

    // Convert the emitted state from the comoving frame to the coordinate frame, and
    // update opacities.  Only for newly emitted samples.
    if ((boosts || tetrads) && pmy_mc->initialize_comoving[etype]) {
      if (IsPolarized(pmy_mc->polarized)) {
        for (int ip = nold; ip < pphot->nphot; ip++)
          ScatteringStokesToCoherency(this, pphot, ip);
      }
      TransformToCoordinate(pphot,nold,pphot->nphot-1);
    }

    // Update the absorption and scattering extinction coefficients
    if (call_srcterms) {
      // Update source terms to reflect newly emitted samples
      for (int ip=nold; ip<pphot->nphot; ip++) {
        UpdateSourceTerms(pphot,0.,0.,0.,0.,0.,ip);
      }
    }
  }
  //if (pphot->nphot == 1) {
  //  printf("gid: %d\n",pmy_block->gid);
  //  printf("%g %g",pcoord->x1f(is),pcoord->x1f(ie+1));
  //  pphot->PrintPhoton("before",0);
  //}
  // move all samples to next interaction or boundary
  //printf("%d %d\n",pmy_block->gid, pphot->nphot);
  ppusher->Move(pphot,0,pphot->nphot-1);
  //if (pphot->nphot == 1) {
  //  printf("gid: %d\n",pmy_block->gid);
  //  pphot->PrintPhoton("after",0);
  //}
  //printf("%d done\n",pmy_block->gid);
  // perform all absorption and scattering related tasks for all samples
  for (int ip=0; ip<pphot->nphot; ip++) {
    // record initial weight and direction
    Real weight0 = pphot->wp[ip];
    Real e_pre_scat = pphot->ep[ip];
    Real k1p0 = pphot->k1p[ip];
    Real k2p0 = pphot->k2p[ip];
    Real k3p0 = pphot->k3p[ip];
    // account for absorption
    if (pphot->statp[ip] == EVOLVING) {
      if (absorption_meth == ABSWEIGHT) {
        pphot->wp[ip] *= (pphot->scp[ip]/(pphot->scp[ip]+pphot->acp[ip]));
        if(pphot->wp[ip] <= minweight) {
          pphot->statp[ip] = ABSORBED;
        }
      } else if (absorption_meth == ABSPROB) {
        if (pran->uniform() > (pphot->scp[ip]/(pphot->scp[ip]+pphot->acp[ip])) )
          pphot->wp[ip] = 0.;
        pphot->statp[ip] = ABSORBED;
      } else if (absorption_meth == ABSTAU) {
        if(pphot->wp[ip] <= minweight) {
          pphot->statp[ip] = ABSORBED;
        }
      }
    } // status == evolving

    // account for scattering if not absorbed
    if (pphot->statp[ip] == EVOLVING) {
      // Lorentz transform to comoving frame for scattering
      if (boosts || tetrads) {
        TransformToComoving(pphot,ip,ip);
      }
      // Convert coherency tensor to Stokes parameters for scattering (if needed)
      if (IsPolarized(pmy_mc->polarized)) CoherencyToScatteringStokes(this, pphot, ip);
      // call scattering function and update counters
      Scatter(this,pphot,ip,ip);
      if (IsPolarized(pmy_mc->polarized)) ScatteringStokesToCoherency(this, pphot, ip);
      nscat++;
      pphot->nscp[ip]++;
      if (pphot->nscp[ip] % pmy_mc->checkscat == 0) {
        //pphot->PrintPhoton("check scat",ip);
        // Check for possible infinite loop due to NaN in photon
        if (pphot->IsNanPhoton(ip)) {
          pphot->statp[ip] = DESTROYED;
          if (pmy_mc->verbose) {
            pphot->PrintPhoton("Warning: Nan encounterd in TransferPhotons(),"
                               " photon destroyed",ip);
          }
        }
      }

      // Update the absorption and scattering extinction coefficients
      // if scattering can change sample energy.
      if (!coherent_scattering) {
        pphot->acp[ip] = AbsorptionOpacity(this,pphot,ip);
        pphot->scp[ip] = ScatteringOpacity(this,pphot,ip);
      }
      // Lorentz transform to Eulerian frame and shift opacities
      if (boosts || tetrads) {
        TransformToCoordinate(pphot,ip,ip);
      }
    } // status == evolving
      // Update moments that compute radiation force and net heating/cooling
    if (call_srcterms && ((pphot->statp[ip] == EVOLVING) || (pphot->statp[ip] == ABSORBED))) {
      UpdateSourceTerms(pphot,e_pre_scat,weight0,k1p0,k2p0,k3p0,ip);
    }

  } // End loop over ip

  // Perform tasks for samples that have left meshblock or otherwise terminated
  // their evolution.  Loop is reversed because of way particles (photon samples)
  // are popped
  for (int ip=pphot->nphot-1; ip >= 0; ip--) {
    if (pphot->statp[ip] != EVOLVING) {
      if (pphot->statp[ip] != BUFFERED) {
        // Bring the Stokes parameters up to date with the transported coherency tensor
        // before finalizing the photon, writing outputs
        FinalizePhoton(pphot,ip);
      }

      if (pphot->statp[ip] == ESCAPED) {
       // loop over outputs for escaping photons and update
        Spectrum *pspect = pspec;
        while (pspect != nullptr) {
          pspect->UpdateSpectrum(pphot,ip);
          pspect = pspect->next;
        }
        if (pphlist != nullptr) {
          pphlist->AddPhoton(pphot,ip);
        }
        nesc++;
        pphot->RemoveOneParticle(ip);
      } else if (pphot->statp[ip] == ABSORBED) {
        nabs++;
        pphot->RemoveOneParticle(ip);
      } else if (pphot->statp[ip] == DESTROYED) {
        pphot->RemoveOneParticle(ip);
        ndes++;
      } else if (pphot->statp[ip] == REMOVED) {
        pphot->RemoveOneParticle(ip);
        nrem++;
      }
    }
  } // End loop over ip

}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::CoupleMonteCarloToFluid(Real dt)
//! \brief update hydro momentum and energy based on radiative cooling and forces from MC

void MonteCarloBlock::CoupleMonteCarloToFluid(Real dt) {

  if (!coupled) return;

  Real edot_cgs_inv = time_cgs / (rho_cgs * SQR(vel_cgs));
  Real pdot_cgs_inv = time_cgs / (rho_cgs * vel_cgs);

  NormalizeSourceTerms(true);

  MeshBlock *pmb = pmy_block;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          pmb->phydro->u(IEN,k,j,i) += dt * edot_cgs_inv * sourceterms(MCRS0,k,j,i);
          pmb->phydro->u(IM1,k,j,i) += dt * pdot_cgs_inv * sourceterms(MCRF1,k,j,i)
                                          * pmb->phydro->u(IDN,k,j,i);
          pmb->phydro->u(IM2,k,j,i) += dt * pdot_cgs_inv * sourceterms(MCRF2,k,j,i)
                                          * pmb->phydro->u(IDN,k,j,i);
          pmb->phydro->u(IM3,k,j,i) += dt * pdot_cgs_inv * sourceterms(MCRF3,k,j,i)
                                          * pmb->phydro->u(IDN,k,j,i);
        }
      }
  }
  NormalizeSourceTerms(false);
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::LorentzTransform(Photon *pphot, const Real sign, int ips,
//!                                            int ipe)
//! \brief Lorentz transform photon sample
//
// Does not transform stokes vectors but this seems
// to be correct -- the plane of polarization is invariant under lorentz
// transformation as discussed in Cocke & Holm (1972) Nature letter.
// weight is not transformed either as weight represents number of photons
// in the packet which is invariant.
// to_comv: sign = 1.0;
// to_eulr: sign = -1.0;

void MonteCarloBlock::LorentzTransform(Photon *pphot, const Real sign, int ips,
                                       int ipe) {

  for (int ip=ips; ip<=ipe; ip++) {

    Real &k1 = pphot->k1p[ip];
    Real &k2 = pphot->k2p[ip];
    Real &k3 = pphot->k3p[ip];

    int i1 = pphot->i1p[ip];
    int i2 = pphot->i2p[ip];
    int i3 = pphot->i3p[ip];

    Real beta[3];
    for (int i=0; i<3; ++i) {
      beta[i] = sign * vel(i3,i2,i1,i+1)/vel(i3,i2,i1,0);
    }
    Real beta2= SQR(beta[0]) + SQR(beta[1]) + SQR(beta[2]);

    if(beta2 > 0.) {
      // SWD: pretabulate gamma for each cell?
      Real gamma = 1. / sqrt(1. - beta2); // assumes v^2 < c^2 checked elsewhere
      Real bdk = k1 * beta[0] + k2 * beta[1] + k3 * beta[2];
      Real gonembdk = gamma * (1. - bdk);
      Real aber = gamma*(1.-gamma*bdk/(gamma+1.));

      pphot->ep[ip] *= gonembdk;
      k1 = (k1 - aber * beta[0]) / gonembdk;
      k2 = (k2 - aber * beta[1]) / gonembdk;
      k3 = (k3 - aber * beta[2]) / gonembdk;

      // Transform opacities
      // Must be performed even when transforming to comoving frame because inverse
      // process is performed to go back to Eulerian frame in cases where scattering
      // is computed

      pphot->acp[ip] /= gonembdk;
      pphot->scp[ip] /= gonembdk;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real MonteCarloBlock::LorentzTransformFrequencyShift(Photon *pphot, int ip)
//!  \brief Returns frequency shift from Lorentz Trransformation

Real MonteCarloBlock::LorentzTransformFrequencyShift(Photon *pphot, int ip) {

  Real k1 = pphot->k1p[ip];
  Real k2 = pphot->k2p[ip];
  Real k3 = pphot->k3p[ip];
  int i1 = pphot->i1p[ip], i2 = pphot->i2p[ip], i3 = pphot->i3p[ip];

  Real beta[3];
  for (int i=0; i<3; ++i) {
    beta[i] = vel(i3,i2,i1,i+1)/vel(i3,i2,i1,0);
  }
  Real beta2= SQR(beta[0]) + SQR(beta[1]) + SQR(beta[2]);
  Real gonembdk;
  if(beta2 > 0.) {
    Real gamma = 1. / sqrt(1. - beta2); // assumes v^2 < c^2 checked elsewhere
    Real bdk = k1 * beta[0] + k2 * beta[1] + k3 * beta[2];

    gonembdk = gamma * (1. - bdk);
  } else {
    gonembdk = 1.;
  }
  // Always called from lab frame so returna nu'/nu
  return gonembdk;
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::UpdateMoments(Photon *pphot, Real dl, Real etau, int ip)
//! \brief Overload for UpdateMoments with additional extinction argument

void MonteCarloBlock::UpdateMoments(Photon *pphot, Real dl, Real etau, int ip) {

  // Account for attenuation along ray
  Real leff;
  if (fabs(1.-etau) < TINY_NUMBER) {
    leff = dl;
  } else {
    leff = (1.-etau)/pphot->acp[ip];;
  }
  UpdateMoments(pphot, leff, ip);
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::UpdateMoments(Photon *pphot, Real dl, int ip)
//! \brief add contribution to radiation moments in current cell
//
// In general relativity the frame transformations -- FrequencyShiftComoving,
// TransformToComoving, TransformToCoordinate -- rebuild the fluid four-velocity at the
// photon's own position, so that u.u = -1 exactly where it is contracted with k.  The
// moments do not: boost_lab and boost_cmv are built once per cell at the cell center by
// ComputeTransformations, and PhotonFrames applies those same matrices to every photon
// crossing the cell.
//
// Two related approximations: the cell-center tetrad is applied to a wavevector carried
// at the photon without parallel transport, and the opacity is refreshed only at cell
// face crossings.

void MonteCarloBlock::UpdateMoments(Photon *pphot, Real dl, int ip) {

  int type = pphot->type[ip];
  int i1 = pphot->i1p[ip];
  int i2 = pphot->i2p[ip];
  int i3 = pphot->i3p[ip];

  const Real c_cgs = MCConstants::c_cgs;
  const Real wp = pphot->wp[ip];

  // Projects this photon into whichever frames are asked for below, at most once each
  // to avoid repated transformations to same basis.
  PhotonFrames frames(this, pphot, ip, dl);

  if (mom_flag_lab) {
    const PhotonFrameState &s = frames.Get(MCFRAME_LAB);
    if (!s.Finite(wp)) {
      pphot->statp[ip] = DESTROYED;
      if (pmy_mc->verbose) {
        pphot->PrintPhoton("Warning: Nan/Inf encountered in UpdateMoments(),"
                           " photon destroyed",ip);
      }
      return;
    }
    AccumulateMoments(moments, type, i3, i2, i1, s, wp);
  }

  // Coordinate basis moments. Note that only the general pusher stores the coordinate four-vector
  // it is built from.
  if (mom_flag_coord && frames.Available(MCFRAME_COORD)) {
    const PhotonFrameState &s = frames.Get(MCFRAME_COORD);
    if (s.Finite(wp))
      AccumulateMoments(moments_coord, type, i3, i2, i1, s, wp);
  }

  // Mean intensities for evaluating scattering source terms
  if (mom_flag_scat) {
    Real weight_scat, e_scat;
    if (frames.GRTetrad() && boosts) {
      // Only the comoving energy is needed so FrequencyShiftComoving
      // rebuilds the fluid four-velocity at the photon and contracts it with k locally,
      // in contrast to the cell-center projection used for other moments.
      Real shift = FrequencyShiftComoving(pphot, ip);
      e_scat = pphot->ep[ip] * shift;
      weight_scat = wp * e_scat * dl * shift / c_cgs;
    } else {
      const PhotonFrameState &sl = frames.Get(MCFRAME_LAB);
      e_scat = pphot->ep[ip];
      weight_scat = wp * sl.e * sl.dl / c_cgs;
    }

    Real loge = std::log10(e_scat);
    Real log10 = 2.302585092994046;
    int n = std::floor((loge-energy_scat(0))/dloge_scat);

    if (n >= 0 && n < nf_scat) {
      Real norm = c_cgs/(4.*PI*freq_scat_mid(n)*dloge_scat*log10);
      moments_scat(n,i3,i2,i1) += norm * weight_scat;
      moments_scat_error(n,i3,i2,i1) += SQR(norm) * SQR(weight_scat);
      //moments_scat(n,i3,i2,i1) += norm * pphot->scp[ip] * weight * k0 * k0;
    }
  }

  if (mom_flag_com && accumulate_com) {
    const PhotonFrameState &s = frames.Get(MCFRAME_COMOVING);
    if (!s.Finite(wp)) {
      pphot->statp[ip] = DESTROYED;
      if (pmy_mc->verbose) {
        pphot->PrintPhoton("Warning: Nan/Inf encountered in UpdateMoments(),"
                           " comoving frame",ip);
      }
      return;
    }
    AccumulateMoments(moments_com, type, i3, i2, i1, s, wp);
  }

  if (mom_flag_usr) {
    // PhotonFrames caches, so several user moments sharing a frame cost one projection.
    for (int i=0; i<pmy_mc->nuser_mom; i++) {
      MCFrame f = pmy_mc->user_moment_frame[i];
      if (!frames.Available(f)) continue;
      pmy_mc->user_moment_func[i](this,pphot,ip,i,frames.Get(f));
    }
  }

  if (call_srcterms) {
    // Radiative Acceleration from flux (always lab frame)
    Real weight = pphot->wp[ip] * pphot->ep[ip] * dl / c_cgs;
    Real abs_coef = pphot->acp[ip];
    Real sct_coef = pphot->scp[ip];
    // radiative force follows the lab-frame propagation direction
    const Real *nl = frames.Get(MCFRAME_LAB).n;
    sourceterms(MCRF1,i3,i2,i1) += (sct_coef+abs_coef) * weight * nl[0];
    sourceterms(MCRF2,i3,i2,i1) += (sct_coef+abs_coef) * weight * nl[1];
    sourceterms(MCRF3,i3,i2,i1) += (sct_coef+abs_coef) * weight * nl[2];

    if (pmy_mc->absorption_method[pphot->type[ip]] == ABSTAU) {
        Real threshold = 3.28808816e+15 * MCConstants::h_cgs;
        // Update soucterms for ionizing radiation
        if (pphot->ep[ip] > threshold) {
          Real weight = pphot->wp[ip] * dl * abs_coef;
          Real heat = weight * (pphot->ep[ip] - threshold);
          sourceterms(MCRS0,i3,i2,i1) += heat;
          sourceterms(MCNABS,i3,i2,i1) += weight;
        }
    }
  }


}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::UpdateMomentsAcceleration(Photon *pphot, Real dl, Real pl,
//        Real k1, Real k2, Real k3,Real etau, int ip)
//! \brief add contribution to radiation moments in current cell for acceleration

void MonteCarloBlock::UpdateMomentsAcceleration(Photon *pphot, Real dl, Real pl, Real k1,
                                                Real k2, Real k3, Real etau, int ip) {

  int type = pphot->type[ip];
  const Real c_cgs = 2.99792458e10;;
  Real k1p = pphot->k1p[ip];
  Real k2p = pphot->k2p[ip];
  Real k3p = pphot->k3p[ip];

  // Normalize k vector if using general pusher in spherical polar coords.
  // See UpdateSourceTerms: keyed on the metric, not the topology, because r and
  // r sin(theta) are the flat orthonormalization factors.

  if ((coord_system == MCCOORD_SPHERICAL_POLAR) && (pphot->general_pusher_flag)) {
    k2p *= pphot->x1p[ip];
    k3p *= pphot->x1p[ip] * sin(pphot->x2p[ip]);
  }

  // Start from the Eulerian values so that every path below leaves all four defined.
  // The comoving branch used to assign abs_coef, sct_coef and step only inside
  // if (beta2 > 0.), with no else, so a cell with a fluid exactly at rest fell through
  // and leff = (1.-etau)/abs_coef read uninitialized stack.
  Real energy = pphot->ep[ip];
  Real abs_coef = pphot->acp[ip];
  Real sct_coef = pphot->scp[ip];
  Real step = dl;
  // BCM: Comoving moments currently do not work with code acceleration
  if (mom_flag_com) {
    // boost relevant quanitities to comoving frame
    int i1 = pphot->i1p[ip], i2 = pphot->i2p[ip], i3 = pphot->i3p[ip];
    // vel holds a four-velocity, so the three-velocity needs the u^0 division -- the
    // same convention GetDopplerFactor() uses.
    Real beta[3];
    for (int i=0; i<3; ++i) {
      beta[i] = vel(i3,i2,i1,i+1)/vel(i3,i2,i1,0);
    }
    Real beta2= SQR(beta[0]) + SQR(beta[1]) + SQR(beta[2]);

    if(beta2 > 0.) {
      Real gamma = 1. / sqrt(1. - beta2); // assumes v^2 < c^2 checked elsewhere
      Real bdk = k1p * beta[0] + k2p * beta[1] + k3p * beta[2];
      Real gonembdk = gamma * (1. - bdk);
      Real aber = gamma*(1.-gamma*bdk/(gamma+1.));

      energy *= gonembdk;
      k1p = (k1p - aber * beta[0]) / gonembdk;
      k2p = (k2p - aber * beta[1]) / gonembdk;
      k3p = (k3p - aber * beta[2]) / gonembdk;
      abs_coef = pphot->acp[ip] / gonembdk;
      sct_coef = pphot->scp[ip] / gonembdk;
      step = dl * gonembdk;
    }
  }
  // Account for attenuation along ray
  Real leff;
  if (fabs(1.-etau) < TINY_NUMBER) {
    leff = step;
  } else {
    leff = (1.-etau)/abs_coef;
  }

  // Weight moments by time spent in domain

  Real weight = pphot->wp[ip] * energy * leff / c_cgs;
  Real path_weight = weight * (pl / dl);

  if ((std::isinf(weight)) || (std::isnan(weight))) {
    pphot->statp[ip] = DESTROYED;
    if (pmy_mc->verbose) {
      pphot->PrintPhoton("Warning: Nan/Inf encountered in UpdateMoments(),"
                         " photon destroyed",ip);
    }
  } else {
    // Higher order moments are weighted by displacement direction vector k
    Real weight1 = weight * k1;
    Real weight2 = weight * k2;
    Real weight3 = weight * k3;

    int i = pphot->i1p[ip];
    int j = pphot->i2p[ip];
    int k = pphot->i3p[ip];

    if (mom_flag_lab) {
      // Add contribution to corresponding moments
      // Energy density
      moments(type,MCIER,k,j,i) += path_weight;
      // Flux
      moments(type,MCIFR1,k,j,i) += weight1 * c_cgs;
      moments(type,MCIFR2,k,j,i) += weight2 * c_cgs;
      moments(type,MCIFR3,k,j,i) += weight3 * c_cgs;
      // Radiation Pressure
      Real weightp = weight1 * k1p;
      moments(type,MCIPR11,k,j,i) += weightp;
      weightp = weight2 * k2p;
      moments(type,MCIPR22,k,j,i) += weightp;
      weightp = weight3 * k3p;
      moments(type,MCIPR33,k,j,i) += weightp;
      weightp = weight1 * k2p;
      moments(type,MCIPR12,k,j,i) += weightp;
      weightp = weight1 * k3p;
      moments(type,MCIPR13,k,j,i)  += weightp;
      weightp = weight2 * k3p;
      moments(type,MCIPR23,k,j,i) += weightp;
    }

    if (call_srcterms) {
      // Radiative Acceleration from flux
      sourceterms(MCRF1,k,j,i) += (sct_coef+abs_coef) * weight1;
      sourceterms(MCRF2,k,j,i) += (sct_coef+abs_coef) * weight2;
      sourceterms(MCRF3,k,j,i) += (sct_coef+abs_coef) * weight3;
    }
  }

}

//------------------------------------------------- --------------------------------------
//! \fn void MonteCarloBlock::NormalizeMoments(bool normalize)
//! \brief (un)normalized moments for output and copy symmetric elements


void MonteCarloBlock::NormalizeMoments(bool normalize) {

  // Derive the comoving moments from the lab accumulation unless they were accumulated
  // directly.  Done before the normalization factor is applied; it is a scalar so the two
  // commute, but doing it here keeps the derived array in step with what is written out.
  if (mom_flag_com && !accumulate_com && normalize) DeriveComovingMoments();

  // Get integration time
  // Fix for dynamic MC
  Real tint = pmy_mc->tint;
  Real norm;

  if (mom_flag_lab) {
    for (int m=0; m<pmy_mc->ntype; ++m) {
      for (int n=0; n<nmom-3; ++n) {
        for (int k=ks; k<=ke; ++k) {
          for (int j=js; j<=je; ++j) {
            for (int i=is; i<=ie; ++i) {
              if (normalize) {
                norm = 1./ (tint * pcoord->vol(k,j,i));
              } else {
                norm = tint * pcoord->vol(k,j,i);
              }
              moments(m,n,k,j,i) *= norm;
            }
          }
        }
      }
    }
    // Copy normalized moments to symmetric elements
    for (int m=0; m<pmy_mc->ntype; ++m) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            moments(m,MCIPR21,k,j,i) = moments(m,MCIPR12,k,j,i);
            moments(m,MCIPR31,k,j,i) = moments(m,MCIPR13,k,j,i);
            moments(m,MCIPR32,k,j,i) = moments(m,MCIPR23,k,j,i);
          }
        }
      }
    }
  } // end if (mom_flag_lab)

  if (mom_flag_com) {
    for (int m=0; m<pmy_mc->ntype; ++m) {
      for (int n=0; n<nmom-3; ++n) {
        for (int k=ks; k<=ke; ++k) {
          for (int j=js; j<=je; ++j) {
            for (int i=is; i<=ie; ++i) {
              if (normalize)
                norm = 1./ (tint * pcoord->vol(k,j,i));
              else
                norm = tint * pcoord->vol(k,j,i);
              moments_com(m,n,k,j,i) *= norm;
            }
          }
        }
      }
    }
    // Copy normalized moments to symmetric elements
    for (int m=0; m<pmy_mc->ntype; ++m) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            moments_com(m,MCIPR21,k,j,i) = moments_com(m,MCIPR12,k,j,i);
            moments_com(m,MCIPR31,k,j,i) = moments_com(m,MCIPR13,k,j,i);
            moments_com(m,MCIPR32,k,j,i) = moments_com(m,MCIPR23,k,j,i);
          }
        }
      }
    }
  }

  if (mom_flag_coord) {
    for (int m=0; m<pmy_mc->ntype; ++m) {
      for (int n=0; n<nmom-3; ++n) {
        for (int k=ks; k<=ke; ++k) {
          for (int j=js; j<=je; ++j) {
            for (int i=is; i<=ie; ++i) {
              if (normalize)
                norm = 1./ (tint * pcoord->vol(k,j,i));
              else
                norm = tint * pcoord->vol(k,j,i);
              moments_coord(m,n,k,j,i) *= norm;
            }
          }
        }
      }
    }
    // Copy normalized moments to symmetric elements
    for (int m=0; m<pmy_mc->ntype; ++m) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            moments_coord(m,MCIPR21,k,j,i) = moments_coord(m,MCIPR12,k,j,i);
            moments_coord(m,MCIPR31,k,j,i) = moments_coord(m,MCIPR13,k,j,i);
            moments_coord(m,MCIPR32,k,j,i) = moments_coord(m,MCIPR23,k,j,i);
          }
        }
      }
    }
  } // end if (mom_flag_com)

  if (mom_flag_scat) {
    for (int n=0; n<nf_scat; ++n) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            if (normalize)
              norm = 1./ (tint * pcoord->vol(k,j,i));
            else
              norm = tint * pcoord->vol(k,j,i);
            moments_scat(n,k,j,i) *= norm;
            if (normalize) {
              moments_scat_error(n,k,j,i) = std::sqrt(moments_scat_error(n,k,j,i))*norm;
            } else {
              Real mom2 = moments_scat_error(n,k,j,i)*norm;
              moments_scat_error(n,k,j,i) = SQR(mom2);
            }
          }
        }
      }
    }
  }

  if (mom_flag_usr) {
    for (int n=0; n<pmy_mc->nuser_mom; ++n) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            if (normalize)
              norm = 1./ (tint * pcoord->vol(k,j,i));
            else
              norm = tint * pcoord->vol(k,j,i);
            moments_user(n,k,j,i) *= norm;
          }
        }
      }
    }
  }

}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::ResetMoments()
//! \brief set moments to zero on block

void MonteCarloBlock::ResetMoments() {

  // set moments to zero
  for (int m=0; m<pmy_mc->ntype; ++m) {
    for (int n=0; n<nmom-3; ++n) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            moments(m,n,k,j,i) = 0.;
          }
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::UpdateSourceTerms(Photon *pphot, Real energy0,
//                                              Real weight0, int ip, Real k1p0,
//                                              Real k2p0, Real k3p0)
//! \brief compute net photon cooling rate and momentum change

void MonteCarloBlock::UpdateSourceTerms(Photon *pphot, Real energy0,
                                        Real weight0, Real k1p0,
                                        Real k2p0, Real k3p0, int ip) {

  if (pmy_mc->UserSourcetermFunc != nullptr) {
    pmy_mc->UserSourcetermFunc(this,pphot,energy0,weight0,
                              k1p0,k2p0,k3p0,ip);
    return; // SWD: should give option to still do standard updates?
  }

  // Updates
  Real c_cgs = 2.99792458e10;
  Real k1 = pphot->k1p[ip];
  Real k2 = pphot->k2p[ip];
  Real k3 = pphot->k3p[ip];

  // Normalize k vector if using general pusher in spherical polar coords.
  // Keyed on the metric rather than on the topology: r and r sin(theta) are the flat
  // orthonormalization factors, exact only for the spherical_polar metric.  Kerr-Schild
  // shares the topology but not the scale factors, so it is deliberately excluded here;
  // doing it properly means going through pcoord->InverseTetrad as UpdateMoments does.
  if ((coord_system == MCCOORD_SPHERICAL_POLAR) && (pphot->general_pusher_flag)) {
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

  // Components of momentum change --- assumes orthonormal basis
  Real dp1p = pphot->wp[ip] * k1 * pphot->ep[ip] / c_cgs
              - weight0 * k1p0 * energy0 / c_cgs;
  Real dp2p = pphot->wp[ip] * k2 * pphot->ep[ip] / c_cgs
              - weight0 * k2p0 * energy0 / c_cgs;
  Real dp3p = pphot->wp[ip] * k3 * pphot->ep[ip] / c_cgs
              - weight0 * k3p0 * energy0 / c_cgs;

  Real cool = (pphot->wp[ip] * pphot->ep[ip]) - (weight0 * energy0);

  if ((std::isinf(cool)) || (std::isnan(cool))) {
    std::cout << "Warning: UpdateSourceTerms cooling is : " << cool << std::endl;
    pphot->PrintPhoton(ip);
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
    sourceterms(MCRS0,k,j,i) -= cool;
    sourceterms(MCRS1,k,j,i) -= dp1p;
    sourceterms(MCRS2,k,j,i) -= dp2p;
    sourceterms(MCRS3,k,j,i) -= dp3p;
  }

}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::NormalizeSourceTerms(bool normalize)
//! \brief (un)normalized source terms for output

void MonteCarloBlock::NormalizeSourceTerms(bool normalize) {

  // Get integration time
  Real tint = pmy_mc->tint;

  // Normalize sourcterms
  for (int n=0; n<nsrc; ++n) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          if (normalize)
            sourceterms(n,k,j,i) /= (tint * pcoord->vol(k,j,i));
          else
            sourceterms(n,k,j,i) *= (tint * pcoord->vol(k,j,i));
        }
      }
    }
  }

}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::ResetSourceTerms()
//! \brief set sourceterms to zero on block

void MonteCarloBlock::ResetSourceTerms() {

  // set sourceterms to zero
  for (int n=0; n<nsrc; ++n) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          sourceterms(n,k,j,i) = 0.;
        }
      }
    }
  }

}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::SetBoundaryValues(enum MCBoundaryFlag *input_bcs)
//! \brief set boundary values on monte carlo block

void MonteCarloBlock::SetBoundaryValues(enum MCBoundaryFlag *input_bcs) {

  // set x1 boundaries
  if(pmy_block->pbval->block_bcs[BoundaryFace::inner_x1] == BoundaryFlag::block)
    mcb_bcs[BoundaryFace::inner_x1] = MC_BLOCK_BNDRY;
  else
    mcb_bcs[BoundaryFace::inner_x1] = input_bcs[BoundaryFace::inner_x1];

  if(pmy_block->pbval->block_bcs[BoundaryFace::outer_x1] == BoundaryFlag::block)
    mcb_bcs[BoundaryFace::outer_x1] = MC_BLOCK_BNDRY;
  else
    mcb_bcs[BoundaryFace::outer_x1] = input_bcs[BoundaryFace::outer_x1];

  // set x2 boundaries
  if(pmy_block->pbval->block_bcs[BoundaryFace::inner_x2] == BoundaryFlag::block)
    mcb_bcs[BoundaryFace::inner_x2] = MC_BLOCK_BNDRY;
  else
    mcb_bcs[BoundaryFace::inner_x2] = input_bcs[BoundaryFace::inner_x2];

  if(pmy_block->pbval->block_bcs[BoundaryFace::outer_x2] == BoundaryFlag::block)
    mcb_bcs[BoundaryFace::outer_x2] = MC_BLOCK_BNDRY;
  else
    mcb_bcs[BoundaryFace::outer_x2] = input_bcs[BoundaryFace::outer_x2];

  // set x3 boundaries
  if(pmy_block->pbval->block_bcs[BoundaryFace::inner_x3] == BoundaryFlag::block)
    mcb_bcs[BoundaryFace::inner_x3] = MC_BLOCK_BNDRY;
  else
    mcb_bcs[BoundaryFace::inner_x3] = input_bcs[BoundaryFace::inner_x3];

  if(pmy_block->pbval->block_bcs[BoundaryFace::outer_x3] == BoundaryFlag::block)
    mcb_bcs[BoundaryFace::outer_x3] = MC_BLOCK_BNDRY;
  else
    mcb_bcs[BoundaryFace::outer_x3] = input_bcs[BoundaryFace::outer_x3];

}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::ComputeEmissionArray(int etype, Real &em_min, Real &em_max, Real
//!                                                &em_tot)
//! \brief compute emission array

void MonteCarloBlock::ComputeEmissionArray(int etype, Real &em_min, Real &em_max, Real &em_tot) {


  Real tint = pmy_mc->tint;
  em_min = SQR(HUGE_NUMBER);
  em_max = -HUGE_NUMBER;
  em_tot = 0.;

  EmisFunc_t GetEmission = pmy_mc->GetEmission[etype];
  if (pmy_mc->emission_geometry[etype] == EMISVOL) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          Real vol = pcoord->vol(k,j,i);
          emission(k,j,i) = GetEmission(this,k,j,i,etype) * tint * vol;
          //printf("emission[%d %d %d]: %g %g %g %g\n",i,j,k,tint,vol,emission(k,j,i),pmy_mc->GetEmission[etype](this,k,j,i,etype));
          em_tot += emission(k,j,i);
          if (emission(k,j,i) > em_max) em_max = emission(k,j,i);
          if (emission(k,j,i) < em_min) em_min = emission(k,j,i);
          if (std::isnan(emission(k,j,i)))
            printf("emission[%d %d %d]: %g %g %g\n",i,j,k,vol,emission(k,j,i),pmy_mc->GetEmission[etype](this,k,j,i,etype));
        }
      }
    }
  } else if (pmy_mc->emission_geometry[etype] == EMISAREA) {
    // emmision array is reused so loop over block and reset to zero
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          emission(k,j,i) = 0.;
        }
      }
    }
    int il=is, iu=ie, jl=js, ju=je, kl=ks, ku=ke;
    int ip=0, jp=0, kp=0;
    int iface;
    BoundaryFace face = pmy_mc->emission_face[etype];
    // Check if we are on a physical boundary corresponding to emission face
    // Could use the mesh information, but this will generalize to arbitrary
    // surface in future
    Mesh *pm = pmy_block->pmy_mesh;
    Real tol = 0.001;
    bool physical_boundary = false;
    switch(face) {
      case BoundaryFace::inner_x1: {
        Real diff = std::fabs(pm->mesh_size.x1min-pcoord->x1f(is));
        Real dx = std::fabs(pcoord->x1f(is+1)-pcoord->x1f(is));
        if (diff < tol*dx) {
          physical_boundary = true;
          il = iu = is;
          iface = 0;
        }
        break;
      }
      case BoundaryFace::outer_x1: {
        Real diff = std::fabs(pm->mesh_size.x1max-pcoord->x1f(ie+1));
        Real dx = std::fabs(pcoord->x1f(ie+1)-pcoord->x1f(ie));
        if (diff < tol*dx) {
          physical_boundary = true;
          il = iu = ie;
          iface = 3;
        }
        break;
      }
      case BoundaryFace::inner_x2: {
        Real diff = std::fabs(pm->mesh_size.x2min-pcoord->x2f(js));
        Real dx = std::fabs(pcoord->x2f(js+1)-pcoord->x2f(js));
        if (diff < tol*dx) {
          physical_boundary = true;
          jl = ju = js;
          iface = 1;
        }
        break;
      }
      case BoundaryFace::outer_x2: {
        Real diff = std::fabs(pm->mesh_size.x2max-pcoord->x2f(je+1));
        Real dx = std::fabs(pcoord->x2f(je+1)-pcoord->x2f(je));
        if (diff < tol*dx) {
          physical_boundary = true;
          jl = ju = je;
          iface = 4;
        }
        break;
      }
      case BoundaryFace::inner_x3: {
        Real diff = std::fabs(pm->mesh_size.x3min-pcoord->x3f(ks));
        Real dx = std::fabs(pcoord->x3f(ks+1)-pcoord->x3f(ks));
        if (diff < tol*dx) {
          physical_boundary = true;
          kl = ku = ks;
          iface = 2;
        }
        break;
      }
      case BoundaryFace::outer_x3: {
        Real diff = std::fabs(pm->mesh_size.x3max-pcoord->x3f(ke+1));
        Real dx = std::fabs(pcoord->x3f(ke+1)-pcoord->x3f(ke));
        if (diff < tol*dx) {
          physical_boundary = true;
          kl = iu = ke;
          iface = 5;
        }
        break;
      }
      default:
        std::stringstream msg;
        msg << "### FATAL ERROR in function [MonteCarloBlock::SetEmissionCellWeightArea]"
            << std::endl << "Face not valid" << std::endl;
        throw std::runtime_error(msg.str().c_str());
        break;
    }
    if (physical_boundary) { // valid boundary on meshblock
      Coordinates *pbcoord = pmy_block->pcoord; // SWD: Maybe should improve this
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
          for (int i=il; i<=iu; ++i) {
            Real area;
            if (iface == 0)
              area = pbcoord->GetFace1Area(k,j,i);
            else if (iface == 1)
              area = pbcoord->GetFace2Area(k,j,i);
            else if (iface == 2)
              area = pbcoord->GetFace3Area(k,j,i);
            else if (iface == 3)
              area = pbcoord->GetFace1Area(k,j,i+1);
            else if (iface == 4)
              area = pbcoord->GetFace2Area(k,j+1,i);
            else if (iface == 5)
              area = pbcoord->GetFace3Area(k+1,j,i);
            area *= l_cgs*l_cgs; // convert area to cgs
            emission(k,j,i) = GetEmission(this,k,j,i,etype) * tint * area;

            //printf("emission[%d %d %d %d]: %g %g %g %g\n",pmy_block->gid,i,j,k,tint,area,emission(k,j,i),pmy_mc->GetEmission[etype](this,k,j,i,etype));
            em_tot += emission(k,j,i);
            if (emission(k,j,i) > em_max) em_max = emission(k,j,i);
            if (emission(k,j,i) < em_min) em_min = emission(k,j,i);
            if (std::isnan(emission(k,j,i)))
              printf("emission[%d %d %d]: %g %g %g\n",i,j,k,area,emission(k,j,i),
                     GetEmission(this,k,j,i,etype));
          }
        }
      }
    } else {
      // not a physical boundary so there is emission
      em_tot = em_max = em_min = 0.;
    }
  } // end if EMISAREA
  // if using equal weight scheme, intialize variables for SetEmissionCellWeight
  i1_ = -1; i2_= -1; i3_ = -1;

}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::ComputeEmissionSampleArray()
//! \brief compute emission array for equal weight scheme

void MonteCarloBlock::ComputeEmissionSampleArray() {

  int ncells = nx1 * nx2 * nx3;
  Real prob[ncells];
  int count[ncells];

  // contruct probability array
  Real total_emission = 0.;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        int n = (k-ks)*nx2*nx1 + (j-js)*nx1 + i-is;
        prob[n] = emission(k,j,i);
        total_emission += emission(k,j,i);
      }
    }
  }

  for (int i=0; i<ncells; ++i) {
    prob[i] /= total_emission;
  }
  // sample multinomial distribution
  pran->SampleMultinomial(nphremain,ncells,prob,count);
  // set counts in emit_count_ array
  int sum = 0;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        int n = (k-ks)*nx2*nx1 + (j-js)*nx1 + i-is;
        emit_count_(k,j,i) = count[n];
        sum += count[n];
      }
    }
  }
  if (sum != nphremain) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [MonteCarloBlock::ComputeEmissionSampleArray]"
        << std::endl << "Sum of counts does not equal number of photons to emit" << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

}


//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::SetEmissionCellWeight(Photon *pphot, int ips, int ipe)
//! \brief set emission cell and weight for photon via emission array

void MonteCarloBlock::SetEmissionCellWeight(Photon *pphot, int ips, int ipe) {

  if (pmy_mc->emission_eqwt[0]) {
    // Set intial cell based on probability within cell

    for (int ip=ips; ip<=ipe; ip++) {
      bool this_zone = false;
      while (!this_zone) {
        int i = i1_ + is;
        int j = i2_ + js;
        int k = i3_ + ks;
        if (emit_count_(k,j,i) > 0) {
          pphot->i1p[ip] = i;
          pphot->i2p[ip] = j;
          pphot->i3p[ip] = k;
          this_zone = true;
          emit_count_(k,j,i) -= 1;
        } else {
          // Update cell
          this_zone = false;
          i3_++;
          if (i3_ >= nx3) {
            i3_ = 0;
            i2_++;
            if (i2_ >= nx2) {
              i2_ = 0;
              i1_++;
              if (i1_ >= nx1)
                i1_ = 0;
            }
          }
        }
      } // end while (!this_zone)
      // Set weight to constant value for all photons
      pphot->wp[ip] = emiss_to_weight;

    } // end loop over ip
  } else {
    for (int ip=ips; ip<=ipe; ip++) {
      // Randomly assign emission cell
      pphot->i1p[ip] = static_cast<int>(pran->uniform()*nx1)+is;
      pphot->i2p[ip] = static_cast<int>(pran->uniform()*nx2)+js;
      pphot->i3p[ip] = static_cast<int>(pran->uniform()*nx3)+ks;

      // Set weight according to the emission array, which is the relative number of
      // photons emitted in each cell
      pphot->wp[ip] = emission(pphot->i3p[ip],pphot->i2p[ip],pphot->i1p[ip])
        * emiss_to_weight;
    } // end loop over ip
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::SetEmissionCellWeightArea(Photon *pphot, BoundaryFace face,
//        int ips, int ipe)
//! \brief set emission cell and weight for photon via emission array

void MonteCarloBlock::SetEmissionCellWeightArea(Photon *pphot, BoundaryFace face, int ips,
                                                int ipe) {

  if (pmy_mc->emission_eqwt[0]) {
    // Set intial cell based on probability within cell
    for (int ip=ips; ip<=ipe; ip++) {
      bool i1flag = true;
      bool i2flag = true;
      bool i3flag = true;
      switch(face) {
        case BoundaryFace::inner_x1:
          i1_ = 0;
          i1flag = false;
          break;
        case BoundaryFace::outer_x1:
          i1_ = nx1-1;
          i1flag = false;
          break;
        case BoundaryFace::inner_x2:
          i2_ = 0;
          i2flag = false;
          break;
        case BoundaryFace::outer_x2:
          i2_ = nx2-1;
          i2flag = false;
          break;
        case BoundaryFace::inner_x3:
          i3_ = 0;
          i3flag = false;
          break;
        case BoundaryFace::outer_x3:
          i3_ = nx3-1;
          i3flag = false;
          break;
        default:
          std::stringstream msg;
          msg << "### FATAL ERROR in function [MonteCarloBlock::SetEmissionCellWeightArea]"
              << std::endl << "Face not valid" << std::endl;
          throw std::runtime_error(msg.str().c_str());
          break;
      }
      bool this_zone = false;
      while (!this_zone) {
        int i = i1_ + is;
        int j = i2_ + js;
        int k = i3_ + ks;

        if (emit_count_(k,j,i) > 0) {
          pphot->i1p[ip] = i;
          pphot->i2p[ip] = j;
          pphot->i3p[ip] = k;
          this_zone = true;
          emit_count_(k,j,i) -= 1;
        } else {
          // Update cell
          this_zone = false;
          if (!i3flag) {
            i2_++;
            if (i2_ >= nx2) {
              i2_ = 0;
              i1_++;
              if (i1_ >= nx1)
                i1_ = 0;
            }
          } else if (!i2flag) {
            i3_++;
            if (i3_ >= nx3) {
              i3_ = 0;
              i1_++;
              if (i1_ >= nx1)
                i1_ = 0;
            }
          } else if (!i1flag) {
            i3_++;
            if (i3_ >= nx3) {
              i3_ = 0;
              i2_++;
              if (i2_ >= nx2) {
                i2_ = 0;
              }
            }
          }
        }
      } // end while (!this_zone)
      // Set weight to constant value for all photons
      pphot->wp[ip] = emiss_to_weight;

    } // end loop over ip
  } else {

    for (int ip=ips; ip<=ipe; ip++) {
      // Randomly assign emission cell
      Real weight_reduce;
      switch(face) {
        case BoundaryFace::inner_x1:
          pphot->i1p[ip] = is;
          pphot->i2p[ip] = static_cast<int>(pran->uniform()*nx2)+js;
          pphot->i3p[ip] = static_cast<int>(pran->uniform()*nx3)+ks;
          weight_reduce = 1./static_cast<Real>(nx1);
          break;
        case BoundaryFace::outer_x1:
          pphot->i1p[ip] = ie;
          pphot->i2p[ip] = static_cast<int>(pran->uniform()*nx2)+js;
          pphot->i3p[ip] = static_cast<int>(pran->uniform()*nx3)+ks;
          weight_reduce = 1./static_cast<Real>(nx1);
          break;
        case BoundaryFace::inner_x2:
          pphot->i1p[ip] = static_cast<int>(pran->uniform()*nx1)+is;
          pphot->i2p[ip] = js;
          pphot->i3p[ip] = static_cast<int>(pran->uniform()*nx3)+ks;
          weight_reduce = 1./static_cast<Real>(nx2);
          break;
        case BoundaryFace::outer_x2:
          pphot->i1p[ip] = static_cast<int>(pran->uniform()*nx1)+is;
          pphot->i2p[ip] = je;
          pphot->i3p[ip] = static_cast<int>(pran->uniform()*nx3)+ks;
          weight_reduce = 1./static_cast<Real>(nx2);
          break;
        case BoundaryFace::inner_x3:
          pphot->i1p[ip] = static_cast<int>(pran->uniform()*nx1)+is;
          pphot->i2p[ip] = static_cast<int>(pran->uniform()*nx2)+js;
          pphot->i3p[ip] = ks;
          weight_reduce = 1./static_cast<Real>(nx3);
          break;
        case BoundaryFace::outer_x3:
          pphot->i1p[ip] = static_cast<int>(pran->uniform()*nx1)+is;
          pphot->i2p[ip] = static_cast<int>(pran->uniform()*nx2)+js;
          pphot->i3p[ip] = ke;
          weight_reduce = 1./static_cast<Real>(nx3);
          break;
        default:
          std::stringstream msg;
          msg << "### FATAL ERROR in function [MonteCarloBlock::SetEmissionCellWeightArea]"
              << std::endl << "Face not valid" << std::endl;
          throw std::runtime_error(msg.str().c_str());
          break;
      }

      // Set weight according to the emission array, which is the relative number of
      // photons emitted in each cell. Note that emiss_to_weight assumes all cells
      // in block contribute so needs to be receduced by number of cell in direction
      // normal to the face
      pphot->wp[ip] = emission(pphot->i3p[ip],pphot->i2p[ip],pphot->i1p[ip])
        * emiss_to_weight * weight_reduce;
    } // end loop over ip
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::GetDensity()
//! \brief Make hard copy of density from MeshBlock to MonteCarloBlock.
//  Uses hard copy so that rho is always in cgs units

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::FillBounds(int &il, int &iu, int &jl, int &ju,
//!                                      int &kl, int &ku) const
//! \brief index range the fluid-derived arrays are filled over
//
// Active cells plus ghosts, widened only in the dimensions the block actually has, which
// is the same rule MeshBlock::ProblemGenerator and Mesh::Initialize use.
//
// The ghosts matter because a photon can sit in one: it keeps its old cell indices while
// it waits to be handed to the neighboring block, and the pusher goes on reading rho,
// tgas, vel and the boost matrices at those indices.  Filling active cells only left each
// of those reads returning zero, which for vel meant a null four-velocity reaching
// FrequencyShiftComoving and the two transform routines.
//
// This assumes the source primitives are themselves valid in the ghosts.  They are for a
// problem generator that fills its full range, which is the Athena++ convention, and for
// non-GR runs Mesh::Initialize refreshes them via ConservedToPrimitive.  Note that
// ConservedToPrimitive is deliberately skipped when MONTE_CARLO_ENABLED and
// GENERAL_RELATIVITY are both on, so in that case the ghosts are exactly what the problem
// generator wrote and nothing else.

void MonteCarloBlock::FillBounds(int &il, int &iu, int &jl, int &ju,
                                 int &kl, int &ku) const {
  il = is - NGHOST;
  iu = ie + NGHOST;
  jl = js;
  ju = je;
  if (nx2 > 1) {
    jl -= NGHOST;
    ju += NGHOST;
  }
  kl = ks;
  ku = ke;
  if (nx3 > 1) {
    kl -= NGHOST;
    ku += NGHOST;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::FluidFourVelocity(Real x[4], int i3, int i2, int i1,
//!                                             Real ucon[4]) const
//! \brief four-velocity of cell (i3,i2,i1)'s frame, evaluated at the position x
//
// The normal uu^i carries no normalization constraint, so covariant velocities can
// be obtained via the metric at specific x. The fluid state is assumed to be
// piecewise constant.
//
// With boosts off uprim is zero and this returns the normal observer at x, which is both
// the right answer and an exact one, since no cell-center quantity enters at all.

void MonteCarloBlock::FluidFourVelocity(Real x[4], int i3, int i2, int i1,
                                        Real ucon[4]) const {

  Real gcov[4][4], gcon[4][4];
  pcoord->Metric(x, gcov);
  pcoord->InverseMetric(x, gcon);

  const Real uu1 = uprim(i3,i2,i1,0);
  const Real uu2 = uprim(i3,i2,i1,1);
  const Real uu3 = uprim(i3,i2,i1,2);

  const Real gamma2 = 1. + gcov[IMC1][IMC1]*uu1*uu1 + gcov[IMC2][IMC2]*uu2*uu2
                    + gcov[IMC3][IMC3]*uu3*uu3
                    + 2.*gcov[IMC1][IMC2]*uu1*uu2 + 2.*gcov[IMC1][IMC3]*uu1*uu3
                    + 2.*gcov[IMC2][IMC3]*uu2*uu3;
  const Real gamma = std::sqrt(gamma2);
  const Real alpha = 1.0/std::sqrt(-gcon[IMC0][IMC0]);

  ucon[IMC0] = -gamma*alpha*gcon[IMC0][IMC0];
  ucon[IMC1] = uu1 - gamma*alpha*gcon[IMC0][IMC1];
  ucon[IMC2] = uu2 - gamma*alpha*gcon[IMC0][IMC2];
  ucon[IMC3] = uu3 - gamma*alpha*gcon[IMC0][IMC3];
}

//----------------------------------------------------------------------------------------

void MonteCarloBlock::GetDensity() {

  if (pmy_mc->UserGetDensity != nullptr) {
    pmy_mc->UserGetDensity(this);
    return;
  }
  int il, iu, jl, ju, kl, ku;
  FillBounds(il, iu, jl, ju, kl, ku);
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        rho(k,j,i) = rho_cgs * pmy_block->phydro->w(IDN,k,j,i);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::GetNumberDensity()
//! \brief default function for computing number densities if no user function provided.

void MonteCarloBlock::GetNumberDensity() {

  if (pmy_mc->UserGetNumberDensity != nullptr) {
    pmy_mc->UserGetNumberDensity(this);
    return;
  }

  if (pmy_mc->scattering_meth == SCATRES) {
    // Default for resonant scattering assumes pure hydrogen
    // with 100% neutral fraction.
    Real mp = 1.67262192369e-24;
    int il, iu, jl, ju, kl, ku;
    FillBounds(il, iu, jl, ju, kl, ku);
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          species(0,k,j,i) = rho(k,j,i) / mp;
        }
      }
    }
    return;
  }

  // For all other cases, the default assumes a compostion of hydrogen
  // and helium with a given abundance and fully ionized. We
  // compute the electron and ion number desnities. This is used
  // for electron scattering and free-free emission/absorption.

  Real heabund = 0.09; //hardcode for now (should be parameter)
  Real mp = 1.67262192369e-24;

  int il, iu, jl, ju, kl, ku;
  FillBounds(il, iu, jl, ju, kl, ku);
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        Real nh = rho(k,j,i) / (mp*(1.+4.*heabund));
        Real nhe = nh*heabund;
        species(1,k,j,i) = nh + 4. * nhe; // nion
        species(0,k,j,i) = nh + 2. * nhe; // nel
      }
    }
  }
}


//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::GetScalars(MonteCarloBlock *pmcb)
//! \brief Make a hard copy of scalars from MeshBlock to MonteCarloBlock.

void MonteCarloBlock::GetScalars() {

  int il, iu, jl, ju, kl, ku;
  FillBounds(il, iu, jl, ju, kl, ku);
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        scalars(k,j,i) = pmy_block->pscalars->s(0,k,j,i);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::GetVelocities()
//! \brief Make hard copy of velocites from MeshBlock to MonteCarloBlock.
//  Uses hard copy so that velocities is always fraction of speed of light

void MonteCarloBlock::GetVelocity() {

  int il, iu, jl, ju, kl, ku;
  FillBounds(il, iu, jl, ju, kl, ku);

  if (GENERAL_RELATIVITY) {
    // Only the primitives are stored. They carry no normalization constraint, so
    // FluidFourVelocity can assemble the four-velocity at any x.
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          uprim(k,j,i,0) = pmy_block->phydro->w(IVX,k,j,i);
          uprim(k,j,i,1) = pmy_block->phydro->w(IVY,k,j,i);
          uprim(k,j,i,2) = pmy_block->phydro->w(IVZ,k,j,i);
        }
      }
    }
  } else {
    Real c_cgs = 2.99792458e10;
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          Real rho = pmy_block->phydro->u(IDN,k,j,i);
          if (!(rho > 0.)) {
            // Empty ghost cell: leave the fluid at rest rather than dividing by zero.
            // vel must still be a valid four-velocity, since consumers contract it.
            vel(k,j,i,0) = 1.0;
            vel(k,j,i,1) = 0.0;
            vel(k,j,i,2) = 0.0;
            vel(k,j,i,3) = 0.0;
            continue;
          }
          vel(k,j,i,1) = vel_cgs * pmy_block->phydro->u(IM1,k,j,i) / (rho * c_cgs);
          vel(k,j,i,2) = vel_cgs * pmy_block->phydro->u(IM2,k,j,i) / (rho * c_cgs);
          vel(k,j,i,3) = vel_cgs * pmy_block->phydro->u(IM3,k,j,i) / (rho * c_cgs);
          Real beta0 = std::sqrt(SQR(vel(k,j,i,1))+SQR(vel(k,j,i,2))+SQR(vel(k,j,i,3)));
          Real beta = (beta0 > betamax) ? betamax : beta0;
          Real gamma = 1. / std::sqrt(1. - beta*beta);
          vel(k,j,i,0) = gamma;
          //gamma = 1;
          if (beta0 > 0.) {
            for (int l=1; l<4; ++l) {
              vel(k,j,i,l) *= gamma * beta / beta0 ;
            }
          }
        }
      }
    }
  } // end if (GENERAL_RELATIVITY) else
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::SetNormalObserver()
//! \brief declare the frame to be that of the normal (Eulerian) observer.
//
// Used for general relativistic problems run without boosts.  There is no fluid
// velocity to define a comoving frame, but the GR frame transformations still need a
// four-velocity to build the tetrad on, and the natural choice is the observer normal
// to the spatial slices: n^mu = -alpha g^{mu t} with lapse alpha = 1/sqrt(-g^{tt}).
// This is the zero-three-velocity limit of GetVelocity(), and unlike the static
// observer it stays well defined inside the ergosphere.

void MonteCarloBlock::SetNormalObserver() {

  // Zero primitives are the normal observer: with uu^i = 0 the Lorentz factor in
  // FluidFourVelocity is 1 and it returns u^mu = -alpha g^{mu t}, evaluated wherever it
  // is asked for.
  int il, iu, jl, ju, kl, ku;
  FillBounds(il, iu, jl, ju, kl, ku);
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        uprim(k,j,i,0) = 0.0;
        uprim(k,j,i,1) = 0.0;
        uprim(k,j,i,2) = 0.0;
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::GetBField()
//! \brief Make copy of magnetic fields from MeshBlock to MonteCarloBlock.

void MonteCarloBlock::GetBField() {

  Field *pfld = pmy_block->pfield;
  bcc.InitWithShallowSlice(pfld->bcc, 4, IB1, 3);
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::GetTemperature()
//! \brief default function for computing temperature if no user function provided.
//  Assumes EOS of the form P=RTd.

void MonteCarloBlock::GetTemperature() {

  if (pmy_mc->UserGetTemperature != nullptr) {
    pmy_mc->UserGetTemperature(this);
    return;
  }

  Real rideal = 8.314e7;
  Hydro* phydro = pmy_block->phydro;

  Real tconv;
  if (tgas_cgs <= 0.)
    tconv = 1. / rideal;
  else
    tconv = tgas_cgs;

  // compute temperature from pressure and density
  int il, iu, jl, ju, kl, ku;
  FillBounds(il, iu, jl, ju, kl, ku);
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {

        // A ghost cell a problem generator never wrote leaves both of these at zero, and
        // 0/0 is a NaN that the floor and ceiling below cannot clamp -- every comparison
        // against a NaN is false, so it would propagate into the opacities.  Fall back to
        // the floor instead, which is what an empty cell should read as anyway.
        Real dens = phydro->w(IDN,k,j,i);
        Real temp = (dens > 0.) ? tconv * phydro->w(IEN,k,j,i) / dens : tfloor_cgs;
        // apply temperature floor
        temp = (temp > tfloor_cgs) ? temp : tfloor_cgs;
        temp = (temp < tceiling_cgs) ? temp : tceiling_cgs;
        tgas(k,j,i) = temp;
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::ComputeTransformations()
//! \brief compute transformation matrices between comoving frame and lab frame

void MonteCarloBlock::ComputeTransformations() {

  int il, iu, jl, ju, kl, ku;
  FillBounds(il, iu, jl, ju, kl, ku);

  // Decide whether the comoving frequency shift is identically one, which lets the
  // general pusher skip its per-step opacity refresh.  Two conditions, both needed:
  //
  //   flat metric   -- the shift for an observer at rest is the lapse, and alpha = 1 for
  //                    every flat metric the module supports (g_tt = -1, no time cross
  //                    terms).  In a curved metric alpha varies from point to point.
  //   fluid at rest -- otherwise there is a Doppler term, and it varies within a cell
  //                    whenever the flow has a component along a direction the metric
  //                    depends on.  Snake is a poor test of that because its metric
  //                    depends on x1 alone while mc_snake only drives flow along x3, so
  //                    k_3 is conserved and the shift comes out constant; spherical polar
  //                    with any radial flow is the case that does vary.
  //
  // The test is for exact zeros.  A pgen that sets a tiny but nonzero velocity gets the
  // refresh, which costs time but cannot be wrong.
  bool fluid_at_rest = true;
  for (int k=kl; k<=ku && fluid_at_rest; ++k) {
    for (int j=jl; j<=ju && fluid_at_rest; ++j) {
      for (int i=il; i<=iu && fluid_at_rest; ++i) {
        if (GENERAL_RELATIVITY) {
          if (uprim(k,j,i,0) != 0. || uprim(k,j,i,1) != 0. || uprim(k,j,i,2) != 0.)
            fluid_at_rest = false;
        } else {
          if (vel(k,j,i,1) != 0. || vel(k,j,i,2) != 0. || vel(k,j,i,3) != 0.)
            fluid_at_rest = false;
        }
      }
    }
  }
  shift_unity = fluid_at_rest && !curved_metric;

  if (GENERAL_RELATIVITY) {
    // In GR the map to an orthonormal frame is not a flat Lorentz boost.  vel holds a
    // coordinate-frame four-velocity, so treating its components as gamma and
    // gamma*beta -- which is what the flat branch below does -- is not a Lorentz
    // transformation at all: for a static observer in Kerr-Schild the spatial part picks
    // up the shift vector and the components fail the Minkowski norm.  Instead store the
    // covariant legs of two tetrads, so that one matrix multiply carries a coordinate
    // four-vector straight to orthonormal components:
    //
    //   boost_lab -> normal (Eulerian) observer; the basis lab moments are reported in.
    //   boost_cmv -> the frame vel is built on: the fluid with boosts enabled, and the
    //                normal observer without, in which case the two agree by construction.
    //
    // Both are evaluated at the cell center, which is where the fluid velocity already
    // lives, so the moments are built in a single well defined per-cellframe.
    AthenaArray<Real> g, gi;
    g.NewAthenaArray(NMETRIC,iu+1);
    gi.NewAthenaArray(NMETRIC,iu+1);
    for (int k=kl; k<=ku; k++) {
      for (int j=jl; j<=ju; j++) {
        pmy_block->pcoord->CellMetric(k,j,il,iu,g,gi);
        for (int i=il; i<=iu; i++) {
          Real x[4];
          x[IMC0] = 0.;
          x[IMC1] = pmy_block->pcoord->x1v(i);
          x[IMC2] = pmy_block->pcoord->x2v(j);
          x[IMC3] = pmy_block->pcoord->x3v(k);
          Real gcov[4][4];
          pcoord->Metric(x, gcov);
          Real econ[4][4], ecov[4][4];
          // gcov defined at x, g, gi defined at cell center
          // normal observer, n^mu = -alpha g^{mu t}
          Real alpha = 1.0/std::sqrt(-gi(I00,i));
          Real ncon[4];
          ncon[IMC0] = -alpha*gi(I00,i);
          ncon[IMC1] = -alpha*gi(I01,i);
          ncon[IMC2] = -alpha*gi(I02,i);
          ncon[IMC3] = -alpha*gi(I03,i);
          ConstructTetrad(ncon, gcov, econ, ecov);
          for (int a=0; a<4; a++)
            for (int m=0; m<4; m++)
              boost_lab(k,j,i,a,m) = ecov[a][m];

          // The frame the fluid is at rest in, rebuilt at the cell center.  Deliberately
          // the cell center and not the photon: these matrices are per cell and back the
          // moments, which are cell averages.  See DeriveComovingMoments.
          Real ucon[4];
          FluidFourVelocity(x, k, j, i, ucon);
          ConstructTetrad(ucon, gcov, econ, ecov);
          for (int a=0; a<4; a++)
            for (int m=0; m<4; m++)
              boost_cmv(k,j,i,a,m) = ecov[a][m];
        }
      }
    }
    g.DeleteAthenaArray();
    gi.DeleteAthenaArray();
    return;
  }

  // loop over all cells on block
  for (int k=kl; k<=ku; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {
        boost_cmv(k,j,i,0,0) = vel(k,j,i,0);
        boost_lab(k,j,i,0,0) = vel(k,j,i,0);
        for (int m=1; m<4; m++) {
          boost_cmv(k,j,i,0,m) = -vel(k,j,i,m);
          boost_lab(k,j,i,0,m) = vel(k,j,i,m);
        }
        for (int l=1; l<4; l++) {
          boost_cmv(k,j,i,l,0) = -vel(k,j,i,l);
          boost_lab(k,j,i,l,0) = vel(k,j,i,l);
          for (int m=1; m<4; m++) {
            boost_cmv(k,j,i,l,m) = vel(k,j,i,l)*vel(k,j,i,m)/(1.+vel(k,j,i,0));
            boost_lab(k,j,i,l,m) = boost_cmv(k,j,i,l,m);
          }
          boost_cmv(k,j,i,l,l) += 1.;
          boost_lab(k,j,i,l,l) += 1.;
        }
      } // loop over i
    } // loop over j
  } // loop over k
}

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::TransformToComoving(Photon *pphot, int ips, int ipe)
//! \brief convert momentum vector to comoving (or tetrad) frame

void MonteCarloBlock::TransformToComoving(Photon *pphot, int ips, int ipe) {

  if (GENERAL_RELATIVITY) {
    for(int ip=ips; ip<=ipe; ip++) {
      // Construct the tetrad
      Real gcov[4][4];
      Real x[4];
      x[IMC0] = pphot->x0p[ip];
      x[IMC1] = pphot->x1p[ip];
      x[IMC2] = pphot->x2p[ip];
      x[IMC3] = pphot->x3p[ip];
      pcoord->Metric(x, gcov);

      // Create tetrad basis on a four-velocity rebuilt here, so it is a unit timelike
      // vector at the photon rather than at the cell center.  This is the inverse of
      // TransformToCoordinate and the two are called around Scatter, which does not move
      // the photon, so both see the same x and the round trip stays exact.
      Real ucon[4];
      FluidFourVelocity(x, pphot->i3p[ip], pphot->i2p[ip], pphot->i1p[ip], ucon);
      Real econ[4][4], ecov[4][4];
      ConstructTetrad(ucon, gcov, econ, ecov);

      Real k0init = pphot->k0p[ip];
      // Transform to comoving tetrad.  In GR the coordinate-frame spatial components are
      // dimensional, the comoving ones are a unit direction.
      Real kcopy[4], k[4];
      pphot->GetFourVector(ip, false, kcopy);
      CoordinateToTetrad(kcopy, k, ecov);

      // nufact must be taken from the local array: k0p is the energy, so writing it
      // below also writes ep.
      Real nufact = k[IMC0]/k0init;
      pphot->SetFourVector(ip, true, k);

      pphot->acp[ip] /= nufact;
      pphot->scp[ip] /= nufact;
    }
  } else {
    for(int ip=ips; ip<=ipe; ip++) {

      int i1 = pphot->i1p[ip];
      int i2 = pphot->i2p[ip];
      int i3 = pphot->i3p[ip];

      Real k0init = pphot->k0p[ip];
      // The general pusher stores dimensional spatial components in the coordinate
      // frame; the legacy pushers store a unit direction.
      Real ki[4], kf[4];
      pphot->GetFourVector(ip, !pmy_mc->general_pusher_flag, kf);

      if (pmy_mc->general_pusher_flag) {
        for (int i=0; i<4; i++) ki[i] = kf[i];
        Real x[4], invtet[4][4];
        x[0] = pphot->x0p[ip];
        x[1] = pphot->x1p[ip];
        x[2] = pphot->x2p[ip];
        x[3] = pphot->x3p[ip];
        pcoord->InverseTetrad(x,invtet);
        for (int j=0; j<4; j++) {
          kf[j] = 0.;
          for (int i=0; i<4; i++) {
            kf[j] += invtet[j][i] * ki[i];
          }
        }
      }
      if (boosts) {
        for (int i=0; i<4; i++) ki[i] = kf[i];
        for (int j=0; j<4; j++) {
          kf[j] = 0.;
          for (int i=0; i<4; i++) {
            kf[j] += boost_cmv(i3,i2,i1,j,i) * ki[i];
          }
        }
      }

      // Take nufact before storing: k0p is the energy, so the store also writes ep.
      Real nufact = kf[IMC0]/k0init;
      // comoving frame always keeps a unit propagation direction
      pphot->SetFourVector(ip, true, kf);

      // Into the basis the polarized scattering routines assume; see polarization.hpp
      if (IsPolarized(pmy_mc->polarized)) ToScatteringBasis(this, pphot, ip);
      pphot->acp[ip] /= nufact;
      pphot->scp[ip] /= nufact;

    } //end loop overip
  } // end if GENERAL_RELATIVITY else

}


//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::TransformToCoordinate(Photon *pphot, int ips, int ipe)
//! \brief convert momentum vector to coordinate frame

void MonteCarloBlock::TransformToCoordinate(Photon *pphot, int ips, int ipe) {

  if (GENERAL_RELATIVITY) {
    for(int ip=ips; ip<=ipe; ip++) {
      Real gcov[4][4];
      Real x[4];
      x[IMC0] = pphot->x0p[ip];
      x[IMC1] = pphot->x1p[ip];
      x[IMC2] = pphot->x2p[ip];
      x[IMC3] = pphot->x3p[ip];
      pcoord->Metric(x, gcov);

      // Create tetrad basis on a four-velocity rebuilt here, so it is a unit timelike
      // vector at the photon rather than at the cell center.
      Real ucon[4];
      FluidFourVelocity(x, pphot->i3p[ip], pphot->i2p[ip], pphot->i1p[ip], ucon);

      Real econ[4][4], ecov[4][4];
      ConstructTetrad(ucon, gcov, econ, ecov);

      // Transform out of the comoving tetrad.  Comoving stores a unit direction, the GR
      // coordinate frame stores dimensional components.
      Real kcopy[4], k[4];
      pphot->GetFourVector(ip, true, kcopy);
      Real k0init = kcopy[IMC0];

      TetradToCoordinate(kcopy, k, econ);

      // nufact must be taken before storing: writing k0p also writes ep.
      Real nufact = k[IMC0]/k0init;
      pphot->SetFourVector(ip, false, k);

      pphot->acp[ip] /= nufact;
      pphot->scp[ip] /= nufact;
    }
  } else {
    for(int ip=ips; ip<=ipe; ip++) {
      int i1 = pphot->i1p[ip];
      int i2 = pphot->i2p[ip];
      int i3 = pphot->i3p[ip];

      // Back out of the scattering basis.  Guarded on the legacy pusher exactly as
      // before: the general pusher reaches this function through a different branch.
      if (!pmy_mc->general_pusher_flag && IsPolarized(pmy_mc->polarized))
        FromScatteringBasis(this, pphot, ip);

      Real k0init = pphot->k0p[ip];
      // comoving frame stores a unit direction
      Real ki[4], kf[4];
      pphot->GetFourVector(ip, true, kf);

      if (boosts) {
        for (int i=0; i<4; i++) ki[i] = kf[i];
        for (int j=0; j<4; j++) {
          kf[j] = 0.;
          for (int i=0; i<4; i++) {
            kf[j] += boost_lab(i3,i2,i1,j,i) * ki[i];
          }
        }
      }
      if (pmy_mc->general_pusher_flag) {
        for (int i=0; i<4; i++) ki[i] = kf[i];
        Real x[4], tetrad[4][4];
        x[0] = pphot->x0p[ip];
        x[1] = pphot->x1p[ip];
        x[2] = pphot->x2p[ip];
        x[3] = pphot->x3p[ip];
        pcoord->Tetrad(x,tetrad);
        for (int j=0; j<4; j++) {
          kf[j] = 0.;
          for (int i=0; i<4; i++) {
            kf[j] += tetrad[j][i] * ki[i];
          }
        }
      }

      // Take nufact before storing: writing k0p also writes ep.  The general pusher
      // keeps dimensional spatial components in the coordinate frame; the legacy
      // pushers keep a unit direction.
      Real nufact = kf[IMC0]/k0init;
      pphot->SetFourVector(ip, !pmy_mc->general_pusher_flag, kf);
      // update opacities
      pphot->acp[ip] /= nufact;
      pphot->scp[ip] /= nufact;

    } // for ip loop
  } // end if GENERAL_RELATIVITY else
}

//----------------------------------------------------------------------------------------
//! \fn Real MonteCarloBlock::FrequencyShiftComoving(Photon *pphot, int ip)
//! \brief frequency shift factor from coordinate to comoving

Real  MonteCarloBlock::FrequencyShiftComoving(Photon *pphot, int ip) {

  if (GENERAL_RELATIVITY) {
    // vel holds the fluid four-velocity when boosts are on and the normal observer
    // otherwise, so the tetrad is well defined either way.  With boosts off the shift
    // returned here is the purely gravitational one.
    Real gcov[4][4];
    Real x[4];
    x[IMC0] = pphot->x0p[ip];
    x[IMC1] = pphot->x1p[ip];
    x[IMC2] = pphot->x2p[ip];
    x[IMC3] = pphot->x3p[ip];
    pcoord->Metric(x, gcov);

    // Rebuilt at the photon rather than read from the cell center, so u.u = -1 holds
    // here, where it is about to be contracted with k.
    Real ucon[4];
    FluidFourVelocity(x, pphot->i3p[ip], pphot->i2p[ip], pphot->i1p[ip], ucon);

    Real k0init = pphot->k0p[ip];
    // Called from the coordinate frame, where GR keeps dimensional components.
    Real kcopy[4];
    pphot->GetFourVector(ip, false, kcopy);

    // Only the observer's energy is wanted, so build only the tetrad leg that carries it.
    // ObserverEnergy reproduces ConstructTetrad + CoordinateToTetrad on this component
    // bit for bit while skipping the Gram-Schmidt for the three spatial legs, which is
    // roughly ten metric contractions and four square roots that never reach the answer.
    // This sits in the general pusher's inner loop through UpdateOpacities, so the saving
    // is what makes refreshing opacities more often than once per cell affordable.
    return ObserverEnergy(ucon, kcopy, gcov)/k0init;
  } else {
    int i1 = pphot->i1p[ip];
    int i2 = pphot->i2p[ip];
    int i3 = pphot->i3p[ip];

    Real k0init = pphot->k0p[ip];
    // Called from the coordinate frame: the general pusher keeps dimensional spatial
    // components there, the legacy pushers keep a unit direction.
    Real ki[4], kf[4];
    pphot->GetFourVector(ip, !pmy_mc->general_pusher_flag, kf);

    if (tetrads) {
      for (int i=0; i<4; i++) ki[i] = kf[i];
      Real x[4], invtet[4][4];
      x[0] = pphot->x0p[ip];
      x[1] = pphot->x1p[ip];
      x[2] = pphot->x2p[ip];
      x[3] = pphot->x3p[ip];
      pcoord->InverseTetrad(x,invtet);
      for (int j=0; j<4; j++) {
        kf[j] = 0.;
        for (int i=0; i<4; i++) {
          kf[j] += invtet[j][i] * ki[i];
        }
      }
    }

    // Boost from the tetrad frame into the fluid frame.  With boosts disabled the tetrad
    // frame is already the comoving frame, and boost_cmv has never been filled by
    // ComputeTransformations(), so it must not be applied here.
    Real k0f;
    if (boosts) {
      k0f = 0.;
      for (int i=0; i<4; i++) {
        k0f += boost_cmv(i3,i2,i1,0,i) * kf[i];
      }
    } else {
      k0f = kf[0];
    }
    /* Real nufact =k0f/k0init;
     (std::isinf(nufact) || std::isnan(nufact)) {
      printf("%g %g %g %g\n",boost_cmv(i3,i2,i1,0,0),boost_cmv(i3,i2,i1,0,1),
            boost_cmv(i3,i2,i1,0,2),boost_cmv(i3,i2,i1,0,3));
      printf("%g %g %g %g %g %g\n",k0init,k0f,kf[0],kf[1],kf[2],kf[3]);
      pphot->PrintPhoton("shift: nan in boost",ip);
      }*/
    return k0f/k0init;
  }

 }

//----------------------------------------------------------------------------------------
//! \fn void MonteCarloBlock::SetBoundaryValues(enum MCBoundaryFlag *input_bcs)
//! \brief set boundary values on monte carlo block

/*void MonteCarloBlock::SetBoundaryValues(enum MCBoundaryFlag *input_bcs) {

  // set x1 boundaries
  if (pmy_block->pbval->block_bcs[BoundaryFace::inner_x1] == BLOCK_BNDRY) {
    mcb_bcs[BoundaryFace::inner_x1] = MC_BLOCK_BNDRY;
  } else {
    mcb_bcs[BoundaryFace::inner_x1] = input_bcs[BoundaryFace::inner_x1];
  }
  if (pmy_block->pbval->block_bcs[BoundaryFace::outer_x1] == BLOCK_BNDRY) {
    mcb_bcs[BoundaryFace::outer_x1] = MC_BLOCK_BNDRY;
  } else {
    mcb_bcs[BoundaryFace::outer_x1] = input_bcs[BoundaryFace::outer_x1];
  }

  // set x2 boundaries
  if (pmy_block->pbval->block_bcs[BoundaryFace::inner_x2] == BLOCK_BNDRY) {
    mcb_bcs[BoundaryFace::inner_x2] = MC_BLOCK_BNDRY;
  } else {
    mcb_bcs[BoundaryFace::inner_x2] = input_bcs[BoundaryFace::inner_x2];
  }
  if (pmy_block->pbval->block_bcs[BoundaryFace::outer_x2] == BLOCK_BNDRY) {
    mcb_bcs[BoundaryFace::outer_x2] = MC_BLOCK_BNDRY;
  } else {
    mcb_bcs[BoundaryFace::outer_x2] = input_bcs[BoundaryFace::outer_x2];
  }

  // set x3 boundaries
  if (pmy_block->pbval->block_bcs[BoundaryFace::inner_x3] == BLOCK_BNDRY) {
    mcb_bcs[BoundaryFace::inner_x3] = MC_BLOCK_BNDRY;
  } else {
    mcb_bcs[BoundaryFace::inner_x3] = input_bcs[BoundaryFace::inner_x3];
  }
  if (pmy_block->pbval->block_bcs[BoundaryFace::outer_x3] == BLOCK_BNDRY) {
    mcb_bcs[BoundaryFace::outer_x3] = MC_BLOCK_BNDRY;
  } else {
    mcb_bcs[BoundaryFace::outer_x3] = input_bcs[BoundaryFace::outer_x3];
  }
  }*/
