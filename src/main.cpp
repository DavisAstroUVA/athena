//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//================================= Athena++ Main Program ================================
//! \file main.cpp
//! \brief Athena++ main program
//!
//! Based on the Athena MHD code (Cambridge version), originally written in 2002-2005 by
//! Jim Stone, Tom Gardiner, and Peter Teuben, with many important contributions by many
//! other developers after that, i.e. 2005-2014.
//!
//! Athena++ was started in Jan 2014.  The core design was finished during 4-7/2014 at the
//! KITP by Jim Stone.  GR was implemented by Chris White and AMR by Kengo Tomida during
//! 2014-2016.  Contributions from many others have continued to the present.
//========================================================================================

// C headers

// C++ headers
#include <cmath>      // sqrt()
#include <csignal>    // ISO C/C++ signal() and sigset_t, sigemptyset() POSIX C extensions
#include <cstdint>    // int64_t
#include <cstdio>     // sscanf()
#include <chrono>     // steady_clock
#include <cstdlib>    // strtol
#include <ctime>      // clock(), CLOCKS_PER_SEC, clock_t
#include <exception>  // exception
#include <iomanip>    // setprecision()
#include <iostream>   // cout, endl
#include <limits>     // max_digits10
#include <new>        // bad_alloc
#include <string>     // string

// Athena++ headers
#include "athena.hpp"
#include "fft/turbulence.hpp"
#include "globals.hpp"
#include "gravity/fft_gravity.hpp"
#include "gravity/mg_gravity.hpp"
#include "mesh/mesh.hpp"
#include "outputs/io_wrapper.hpp"
#include "outputs/outputs.hpp"
#include "parameter_input.hpp"
#include "utils/utils.hpp"
#include "monte_carlo/mcgrid.hpp"
#include "monte_carlo/montecarlo.hpp"

//! \fn static long PeakResidentKB()
//! \brief peak resident set size of this process in kB, from /proc/self/status
//
// Linux only; returns 0 where the file is missing, and the caller then prints nothing.
// Read at the end of the run rather than sampled during it because VmHWM is already the
// high-water mark the kernel keeps.
static long PeakResidentKB() {
  std::FILE *f = std::fopen("/proc/self/status", "r");
  if (f == nullptr) return 0;
  char line[256];
  long kb = 0;
  while (std::fgets(line, sizeof(line), f) != nullptr) {
    if (std::sscanf(line, "VmHWM: %ld", &kb) == 1) break;
  }
  std::fclose(f);
  return kb;
}

// MPI/OpenMP headers
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

//----------------------------------------------------------------------------------------
//! \fn int main(int argc, char *argv[])
//! \brief Athena++ main program

int main(int argc, char *argv[]) {
  std::string athena_version = "version 21.0 - January 2021";
  char *input_filename = nullptr, *restart_filename = nullptr;
  char *prundir = nullptr;
  int res_flag = 0;   // set to 1 if -r        argument is on cmdline
  int narg_flag = 0;  // set to 1 if -n        argument is on cmdline
  int iarg_flag = 0;  // set to 1 if -i <file> argument is on cmdline
  int mesh_flag = 0;  // set to <nproc> if -m <nproc> argument is on cmdline
  int wtlim = 0;
  std::uint64_t mbcnt = 0;

  //--- Step 1. --------------------------------------------------------------------------
  // Initialize MPI environment, if necessary

#ifdef MPI_PARALLEL
#ifdef OPENMP_PARALLEL
  int mpiprv;
  if (MPI_SUCCESS != MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpiprv)) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "MPI Initialization failed." << std::endl;
    return(0);
  }
  if (mpiprv != MPI_THREAD_MULTIPLE) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "MPI_THREAD_MULTIPLE must be supported for the hybrid parallelzation. "
              << MPI_THREAD_MULTIPLE << " : " << mpiprv
              << std::endl;
    MPI_Finalize();
    return(0);
  }
#else  // no OpenMP
  if (MPI_SUCCESS != MPI_Init(&argc, &argv)) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "MPI Initialization failed." << std::endl;
    return(0);
  }
#endif  // OPENMP_PARALLEL
  // Get process id (rank) in MPI_COMM_WORLD
  if (MPI_SUCCESS != MPI_Comm_rank(MPI_COMM_WORLD, &(Globals::my_rank))) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "MPI_Comm_rank failed." << std::endl;
    MPI_Finalize();
    return(0);
  }

  // Get total number of MPI processes (ranks)
  if (MPI_SUCCESS != MPI_Comm_size(MPI_COMM_WORLD, &Globals::nranks)) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "MPI_Comm_size failed." << std::endl;
    MPI_Finalize();
    return(0);
  }
#else  // no MPI
  Globals::my_rank = 0;
  Globals::nranks  = 1;
#endif  // MPI_PARALLEL

  // Monte Carlo builds only: wall clock from here, so the phases of setup can be stamped
  // as they finish and their total reported at the end.
  const std::chrono::steady_clock::time_point program_start =
      std::chrono::steady_clock::now();
  auto setup_seconds = [&program_start]() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - program_start).count();
  };
  auto setup_stamp = [&setup_seconds](const char *what) {
    if (MONTE_CARLO_ENABLED && Globals::my_rank == 0) {
      char stamp[32];
      std::snprintf(stamp, sizeof(stamp), "[setup %.1f s] ", setup_seconds());
      std::cout << stamp << what << std::endl;
    }
  };

  //--- Step 2. --------------------------------------------------------------------------
  // Check for command line options and respond.

  for (int i=1; i<argc; i++) {
    // If argv[i] is a 2 character string of the form "-?" then:
    if (*argv[i] == '-'  && *(argv[i]+1) != '\0' && *(argv[i]+2) == '\0') {
      // check validity of command line options + arguments:
      char opt_letter = *(argv[i]+1);
      switch(opt_letter) {
        // options that do not take arguments:
        case 'n':
        case 'c':
        case 'h':
          break;
          // options that require arguments:
        default:
          if ((i+1 >= argc) // flag is at the end of the command line options
              || (*argv[i+1] == '-') ) { // flag is followed by another flag
            if (Globals::my_rank == 0) {
              std::cout << "### FATAL ERROR in main" << std::endl
                        << "-" << opt_letter << " must be followed by a valid argument\n";
#ifdef MPI_PARALLEL
              MPI_Finalize();
#endif
              return(0);
            }
          }
      }
      switch(*(argv[i]+1)) {
        case 'i':                      // -i <input_filename>
          input_filename = argv[++i];
          iarg_flag = 1;
          break;
        case 'r':                      // -r <restart_file>
          res_flag = 1;
          restart_filename = argv[++i];
          break;
        case 'd':                      // -d <run_directory>
          prundir = argv[++i];
          break;
        case 'n':
          narg_flag = 1;
          break;
        case 'm':                      // -m <nproc>
          mesh_flag = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
          break;
        case 't':                      // -t <hh:mm:ss>
          int wth, wtm, wts;
          std::sscanf(argv[++i], "%d:%d:%d", &wth, &wtm, &wts);
          wtlim = wth*3600 + wtm*60 + wts;
          break;
        case 'c':
          if (Globals::my_rank == 0) ShowConfig();
#ifdef MPI_PARALLEL
          MPI_Finalize();
#endif
          return(0);
          break;
        case 'h':
        default:
          if (Globals::my_rank == 0) {
            std::cout << "Athena++ " << athena_version << std::endl;
            std::cout << "Usage: " << argv[0] << " [options] [block/par=value ...]\n";
            std::cout << "Options:" << std::endl;
            std::cout << "  -i <file>       specify input file [athinput]\n";
            std::cout << "  -r <file>       restart with this file\n";
            std::cout << "  -d <directory>  specify run dir [current dir]\n";
            std::cout << "  -n              parse input file and quit\n";
            std::cout << "  -c              show configuration and quit\n";
            std::cout << "  -m <nproc>      output mesh structure and quit\n";
            std::cout << "  -t hh:mm:ss     wall time limit for final output\n";
            std::cout << "  -h              this help\n";
            ShowConfig();
          }
#ifdef MPI_PARALLEL
          MPI_Finalize();
#endif
          return(0);
          break;
      }
    } // else if argv[i] not of form "-?" ignore it here (tested in ModifyFromCmdline)
  }

  if (restart_filename == nullptr && input_filename == nullptr) {
    // no input file is given
    std::cout << "### FATAL ERROR in main" << std::endl
              << "No input file or restart file is specified." << std::endl;
#ifdef MPI_PARALLEL
    MPI_Finalize();
#endif
    return(0);
  }

  // Set up the signal handler
  SignalHandler::SignalHandlerInit();
  if (Globals::my_rank == 0 && wtlim > 0)
    SignalHandler::SetWallTimeAlarm(wtlim);

  // Note steps 3-6 are protected by a simple error handler
  //--- Step 3. --------------------------------------------------------------------------
  // Construct object to store input parameters, then parse input file and command line.
  // With MPI, the input is read by every process in parallel using MPI-IO.

  ParameterInput *pinput;
  IOWrapper infile, restartfile;
#ifdef ENABLE_EXCEPTIONS
  try {
#endif
    pinput = new ParameterInput;
    if (res_flag == 1) {
      restartfile.Open(restart_filename, IOWrapper::FileMode::read);
      pinput->LoadFromFile(restartfile);
      // If both -r and -i are specified, make sure next_time gets corrected.
      // This needs to be corrected on the restart file because we need the old dt.
      if (iarg_flag == 1) pinput->RollbackNextTime();
      // leave the restart file open for later use
    }
    if (iarg_flag == 1) {
      // if both -r and -i are specified, override the parameters using the input file
      infile.Open(input_filename, IOWrapper::FileMode::read);
      pinput->LoadFromFile(infile);
      infile.Close();
    }
    pinput->ModifyFromCmdline(argc ,argv);
#ifdef ENABLE_EXCEPTIONS
  }
  catch(std::bad_alloc& ba) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "memory allocation failed initializing class ParameterInput: "
              << ba.what() << std::endl;
    if (res_flag == 1) restartfile.Close();
#ifdef MPI_PARALLEL
    MPI_Finalize();
#endif
    return(0);
  }
  catch(std::exception const& ex) {
    std::cout << ex.what() << std::endl;  // prints diagnostic message
    if (res_flag == 1) restartfile.Close();
#ifdef MPI_PARALLEL
    MPI_Finalize();
#endif
    return(0);
  }
#endif // ENABLE_EXCEPTIONS

  //--- Step 4. --------------------------------------------------------------------------
  // Construct and initialize Mesh

  // Take the grid from an athdf snapshot if one was named.  This rewrites the <mesh> and
  // <meshblock> parameters before Mesh reads them in its member initializer list, so the
  // Monte Carlo run is built on exactly the grid the snapshot was written from.  A
  // restart carries its own mesh, so this applies only to a fresh start.
  if (MONTE_CARLO_ENABLED) {
    if (res_flag == 0 && MCGridFile::Requested(pinput))
      MCGridFile::Load(pinput)->InjectMeshParameters(pinput);
  }

  Mesh *pmesh;
#ifdef ENABLE_EXCEPTIONS
  try {
#endif
    if (res_flag == 0) {
      pmesh = new Mesh(pinput, mesh_flag);
    } else {
      pmesh = new Mesh(pinput, restartfile, mesh_flag);
    }
#ifdef ENABLE_EXCEPTIONS
  }
  catch(std::bad_alloc& ba) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "memory allocation failed initializing class Mesh: "
              << ba.what() << std::endl;
    if (res_flag == 1) restartfile.Close();
#ifdef MPI_PARALLEL
    MPI_Finalize();
#endif
    return(0);
  }
  catch(std::exception const& ex) {
    std::cout << ex.what() << std::endl;  // prints diagnostic message
    if (res_flag == 1) restartfile.Close();
#ifdef MPI_PARALLEL
    MPI_Finalize();
#endif
    return(0);
  }
#endif // ENABLE_EXCEPTIONS

  // With current mesh time possibly read from restart file, correct next_time for outputs
  if (iarg_flag == 1 && res_flag == 1) {
    // if both -r and -i are specified, ensure that next_time  >= mesh_time - dt
    pinput->ForwardNextTime(pmesh->time);
  }

  // Dump input parameters and quit if code was run with -n option.
  if (narg_flag) {
    if (Globals::my_rank == 0) pinput->ParameterDump(std::cout);
    if (res_flag == 1) restartfile.Close();
#ifdef MPI_PARALLEL
    MPI_Finalize();
#endif
    return(0);
  }

  if (res_flag == 1) restartfile.Close(); // close the restart file here

  // Quit if -m was on cmdline.  This option builds and outputs mesh structure.
  if (mesh_flag > 0) {
#ifdef MPI_PARALLEL
    MPI_Finalize();
#endif
    return(0);
  }

  MonteCarlo *pmc;
  if (MONTE_CARLO_ENABLED) {
    try {
      pmc = new MonteCarlo(pinput,pmesh);
    }
    catch(std::bad_alloc& ba) {
      std::cout << "### FATAL ERROR in main" << std::endl << "memory allocation failed "
                << "in creating monte carlo " << ba.what() << std::endl;
#ifdef MPI_PARALLEL
      MPI_Finalize();
#endif
      return(0);
    }
  }

  setup_stamp("mesh and Monte Carlo objects constructed");

  //--- Step 5. --------------------------------------------------------------------------
  // Construct and initialize TaskList

  TimeIntegratorTaskList *ptlist;
#ifdef ENABLE_EXCEPTIONS
  try {
#endif
    ptlist = new TimeIntegratorTaskList(pinput, pmesh);
#ifdef ENABLE_EXCEPTIONS
  }
  catch(std::bad_alloc& ba) {
    std::cout << "### FATAL ERROR in main" << std::endl << "memory allocation failed "
              << "in creating task list " << ba.what() << std::endl;
#ifdef MPI_PARALLEL
    MPI_Finalize();
#endif
    return(0);
  }
#endif // ENABLE_EXCEPTIONS

  SuperTimeStepTaskList *pststlist = nullptr;
  if (STS_ENABLED) {
#ifdef ENABLE_EXCEPTIONS
    try {
#endif
      pststlist = new SuperTimeStepTaskList(pinput, pmesh, ptlist);
#ifdef ENABLE_EXCEPTIONS
    }
    catch(std::bad_alloc& ba) {
      std::cout << "### FATAL ERROR in main" << std::endl << "memory allocation failed "
                << "in creating task list " << ba.what() << std::endl;
#ifdef MPI_PARALLEL
      MPI_Finalize();
#endif
      return(0);
    }
#endif // ENABLE_EXCEPTIONS
  }

  //--- Step 6. --------------------------------------------------------------------------
  // Set initial conditions by calling problem generator, or reading restart file

#ifdef ENABLE_EXCEPTIONS
  try {
#endif
    pmesh->Initialize(res_flag, pinput);
#ifdef ENABLE_EXCEPTIONS
  }
  catch(std::bad_alloc& ba) {
    std::cout << "### FATAL ERROR in main" << std::endl << "memory allocation failed "
              << "in problem generator " << ba.what() << std::endl;
#ifdef MPI_PARALLEL
    MPI_Finalize();
#endif
    return(0);
  }
  catch(std::exception const& ex) {
    std::cout << ex.what() << std::endl;  // prints diagnostic message
#ifdef MPI_PARALLEL
    MPI_Finalize();
#endif
    return(0);
  }
#endif // ENABLE_EXCEPTIONS
  setup_stamp("mesh initialized: problem generator (snapshot read), boundaries, "
              "primitives");

#ifdef ENABLE_EXCEPTIONS
  try {
#endif
    if (MONTE_CARLO_ENABLED)
      pmc->Initialize(pinput);
#ifdef ENABLE_EXCEPTIONS
  }
  catch(std::bad_alloc& ba) {
    std::cout << "### FATAL ERROR in main" << std::endl << "memory allocation failed "
              << "in MonteCarlo problem generator " << ba.what() << std::endl;
#ifdef MPI_PARALLEL
    MPI_Finalize();
#endif
    return(0);
  }
  catch(std::exception const& ex) {
    std::cout << ex.what() << std::endl;  // prints diagnostic message
#ifdef MPI_PARALLEL
    MPI_Finalize();
#endif
    return(0);
  }
#endif // ENABLE_EXCEPTIONS


  //--- Step 7. --------------------------------------------------------------------------
  // Change to run directory, initialize outputs object, and make output of ICs

  Outputs *pouts;
#ifdef ENABLE_EXCEPTIONS
  try {
#endif
    ChangeRunDir(prundir);
    pouts = new Outputs(pmesh, pinput);
    if (res_flag==0) {
      if (MONTE_CARLO_ENABLED) {
        if (pmc->dynamic)
          pouts->MakeOutputs(pmesh,pmc,pinput);
      } else {
        pouts->MakeOutputs(pmesh,pinput);
      }
    }
#ifdef ENABLE_EXCEPTIONS
  }
  catch(std::bad_alloc& ba) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "memory allocation failed setting initial conditions: "
              << ba.what() << std::endl;
#ifdef MPI_PARALLEL
    MPI_Finalize();
#endif
    return(0);
  }
  catch(std::exception const& ex) {
    std::cout << ex.what() << std::endl;  // prints diagnostic message
#ifdef MPI_PARALLEL
    MPI_Finalize();
#endif
    return(0);
  }
#endif // ENABLE_EXCEPTIONS

  //=== Step 8. === START OF MAIN INTEGRATION LOOP =======================================
  // For performance, there is no error handler protecting this step (except outputs)

  setup_stamp("Monte Carlo blocks set up (opacity tables)");
  const double setup_time = setup_seconds();
  if (Globals::my_rank == 0) {
    if (MONTE_CARLO_ENABLED) {
      std::cout << "\nSetup complete after " << setup_time
                << " s, entering main loop...\n" << std::endl;
    } else {
      std::cout << "\nSetup complete, entering main loop...\n" << std::endl;
    }
  }

  clock_t tstart = clock();
  // Wall time as well as CPU time.  A Monte Carlo run spends part of its CPU time spinning
  // in the termination reductions, so CPU seconds alone neither measure how long the run
  // took nor how much work it did; both numbers are reported at the end.
  std::chrono::steady_clock::time_point wall_start = std::chrono::steady_clock::now();
#ifdef OPENMP_PARALLEL
  double omp_start_time = omp_get_wtime();
#endif
  int mc_ncycle0 = 0;
  if (MONTE_CARLO_ENABLED) {
    // Simple method for modifying main loop
    if (!pmc->dynamic) {
      pmesh->tlim = pmesh->start_time + static_cast<Real>(pmc->nout)*pmc->tint;
      pmesh->dt = pmc->tint;
      mc_ncycle0 = pmesh->ncycle;
    }
  }
  while ((pmesh->time < pmesh->tlim) &&
         (pmesh->nlim < 0 || pmesh->ncycle < pmesh->nlim)) {

    if (Globals::my_rank == 0)
      pmesh->OutputCycleDiagnostics();

    if (STS_ENABLED) {
      pmesh->sts_loc = TaskType::op_split_before;
      // compute nstages for this STS
      if (pmesh->sts_integrator == "rkl2") { // default
        pststlist->nstages =
            static_cast<int>
              (0.5*(-1. + std::sqrt(9. + 16.*(0.5*pmesh->dt)/pmesh->dt_parabolic))) + 1;
      } else { // rkl1
        pststlist->nstages =
            static_cast<int>
              (0.5*(-1. + std::sqrt(1. + 8.*pmesh->dt/pmesh->dt_parabolic))) + 1;
      }
      if (pststlist->nstages % 2 == 0) { // guarantee odd nstages for STS
        pststlist->nstages += 1;
      }
      // take super-timestep
      for (int stage=1; stage<=pststlist->nstages; ++stage)
        pststlist->DoTaskListOneStage(pmesh, stage);

      pmesh->sts_loc = TaskType::main_int;
    }

    if (pmesh->turb_flag > 1) pmesh->ptrbd->Driving(); // driven turbulence

    if ( !(MONTE_CARLO_ENABLED) || pmc->dynamic) {
      pmc->RunMonteCarlo(pouts,pmesh,pinput);

      for (int stage=1; stage<=ptlist->nstages; ++stage) {
        ptlist->DoTaskListOneStage(pmesh, stage);
        if (ptlist->CheckNextMainStage(stage)) {
          if (SELF_GRAVITY_ENABLED == 1) // fft (0: discrete kernel, 1: continuous kernel)
            pmesh->pfgrd->Solve(stage, 0);
          else if (SELF_GRAVITY_ENABLED == 2) // multigrid
            pmesh->pmgrd->Solve(stage);
        }
      }
    } else {
      pmc->RunMonteCarlo(pouts,pmesh,pinput);
      pmesh->dt = pmc->tint;
    } // end not MONTE_CARLO_ENABLED or dynamic MC

    if (STS_ENABLED && pmesh->sts_integrator == "rkl2") {
      pmesh->sts_loc = TaskType::op_split_after;
      // take super-timestep
      for (int stage=1; stage<=pststlist->nstages; ++stage)
        pststlist->DoTaskListOneStage(pmesh, stage);
    }

    pmesh->UserWorkInLoop();

    pmesh->ncycle++;
    pmesh->time += pmesh->dt;
    // Forming the time the same way tlim was formed makes the two agree to the bit
    // on the last cycle, so the run does exactly nout outputs.
    if (MONTE_CARLO_ENABLED && !pmc->dynamic)
      pmesh->time = pmesh->start_time
                    + static_cast<Real>(pmesh->ncycle - mc_ncycle0)*pmc->tint;
    mbcnt += pmesh->nbtotal;
    pmesh->step_since_lb++;

    // A static Monte Carlo run balances from inside RunMonteCarlo, on the transport
    // cost, at points where no photon is in flight; the per-cycle hydro balancer would
    // redistribute on hydro cost the Monte Carlo blocks never see.
    if (!(MONTE_CARLO_ENABLED && pmesh->mc_static))
      pmesh->LoadBalancingAndAdaptiveMeshRefinement(pinput);

    pmesh->NewTimeStep();

#ifdef ENABLE_EXCEPTIONS
    try {
#endif
      if (pmesh->time < pmesh->tlim) // skip the final output as it happens later
        if (MONTE_CARLO_ENABLED) {
          //if (pmc->dynamic) {
            pouts->MakeOutputs(pmesh,pmc,pinput);
          //}
        } else {
          pouts->MakeOutputs(pmesh,pinput);
        }
#ifdef ENABLE_EXCEPTIONS
    }
    catch(std::bad_alloc& ba) {
      std::cout << "### FATAL ERROR in main" << std::endl
                << "memory allocation failed during output: " << ba.what() <<std::endl;
#ifdef MPI_PARALLEL
      MPI_Finalize();
#endif
      return(0);
    }
    catch(std::exception const& ex) {
      std::cout << ex.what() << std::endl;  // prints diagnostic message
#ifdef MPI_PARALLEL
      MPI_Finalize();
#endif
      return(0);
    }
#endif // ENABLE_EXCEPTIONS

    // check for signals
    if (SignalHandler::CheckSignalFlags() != 0) {
      break;
    }
  } // END OF MAIN INTEGRATION LOOP ======================================================
  // Make final outputs, print diagnostics, clean up and terminate

  if (Globals::my_rank == 0 && wtlim > 0)
    SignalHandler::CancelWallTimeAlarm();


  //--- Step 9. --------------------------------------------------------------------------
  // Output the final cycle diagnostics and make the final outputs

  if (Globals::my_rank == 0)
    pmesh->OutputCycleDiagnostics();

  pmesh->UserWorkAfterLoop(pinput);

#ifdef ENABLE_EXCEPTIONS
  try {
#endif
    if (MONTE_CARLO_ENABLED) {
      //if (pmc->dynamic)
        pouts->MakeOutputs(pmesh,pmc,pinput,true);
    } else {
      pouts->MakeOutputs(pmesh,pinput,true);
    }
#ifdef ENABLE_EXCEPTIONS
  }
  catch(std::bad_alloc& ba) {
    std::cout << "### FATAL ERROR in main" << std::endl
            << "memory allocation failed during output: " << ba.what() <<std::endl;
#ifdef MPI_PARALLEL
    MPI_Finalize();
#endif
    return(0);
  }
  catch(std::exception const& ex) {
    std::cout << ex.what() << std::endl;  // prints diagnostic message
#ifdef MPI_PARALLEL
    MPI_Finalize();
#endif
    return(0);
  }
#endif // ENABLE_EXCEPTIONS

  //--- Step 10. -------------------------------------------------------------------------
  // Print diagnostic messages related to the end of the simulation

  bool write_fluid_diagnostics = true;
  if ((MONTE_CARLO_ENABLED) && !(pmc->dynamic))
    write_fluid_diagnostics = false;

  // Peak resident memory, from /proc/self/status (zero where that file does not exist).
  // Collective: every rank contributes before rank 0 prints.  The per-rank maximum is what
  // has to fit on a node; the total is what the job charged.
  long mem_max_kb = PeakResidentKB();
  long mem_sum_kb = mem_max_kb;
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &mem_max_kb, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &mem_sum_kb, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif

  // Wall and CPU time.  clock() measures this rank's process only, so the rate a rank-0
  // CPU time implies is not the rate the job achieved; the sum over ranks is what the job
  // charged and the wall time is how long it took.  Both are collected here, before the
  // rank-0 print, for the same reason the memory figures are.
  const double wall_time = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - wall_start).count();
  clock_t tstop = clock();
  double cpu_time = (tstop>tstart ? static_cast<double> (tstop-tstart) :
                     1.0)/static_cast<double> (CLOCKS_PER_SEC);
  double cpu_time_rank0 = cpu_time;
  double cpu_time_sum = cpu_time;
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &cpu_time_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  if (Globals::my_rank == 0) {
    if (write_fluid_diagnostics) {
      if (SignalHandler::GetSignalFlag(SIGTERM) != 0) {
        std::cout << std::endl << "Terminating on Terminate signal" << std::endl;
      } else if (SignalHandler::GetSignalFlag(SIGINT) != 0) {
        std::cout << std::endl << "Terminating on Interrupt signal" << std::endl;
      } else if (SignalHandler::GetSignalFlag(SIGALRM) != 0) {
        std::cout << std::endl << "Terminating on wall-time limit" << std::endl;
      } else if (pmesh->ncycle == pmesh->nlim) {
        std::cout << std::endl << "Terminating on cycle limit" << std::endl;
      } else {
        std::cout << std::endl << "Terminating on time limit" << std::endl;
      }

      std::cout << "time=" << pmesh->time << " cycle=" << pmesh->ncycle << std::endl;
      std::cout << "tlim=" << pmesh->tlim << " nlim=" << pmesh->nlim << std::endl;

      if (pmesh->adaptive) {
        std::cout << std::endl << "Number of MeshBlocks = " << pmesh->nbtotal
                  << "; " << pmesh->nbnew << "  created, " << pmesh->nbdel
                  << " destroyed during this simulation." << std::endl;
      }
    }

    // Calculate and print the zone-cycles/cpu-second and wall-second
#ifdef OPENMP_PARALLEL
    double omp_time = omp_get_wtime() - omp_start_time;
#endif
    std::uint64_t zonecycles = mbcnt
      *static_cast<std::uint64_t> (pmesh->my_blocks(0)->GetNumberOfMeshBlockCells());
    double zc_cpus = static_cast<double> (zonecycles) / cpu_time;
    if (write_fluid_diagnostics) {
      std::cout << std::endl << "zone-cycles = " << zonecycles << std::endl;
      std::cout << "cpu time used  = " << cpu_time << std::endl;
      std::cout << "zone-cycles/cpu_second = " << zc_cpus << std::endl;
    } else { // Static Monte Carlo output
      // What the run actually transported, counted as photons were retired, rather than
      // the number of samples that were asked for: the two differ whenever a photon is
      // re-emitted, a type emits fewer samples than requested, or a run stops early.
      // Scatterings are the unit of work in a resonant-line problem, where a single
      // photon can carry 1e5 of them, so the per-scattering cost is the stable number and
      // the per-photon cost is only meaningful beside it.
      const double phot = static_cast<double>(pmc->nphot_run);
      const double scat = static_cast<double>(pmc->nscat_run);
      std::cout << std::endl
                << "wall time used = " << wall_time << std::endl
                << "setup wall time = " << setup_time << " (before the main loop; not "
                << "in the figures below)" << std::endl
                << "cpu time used  = " << cpu_time_sum << " summed over "
                << Globals::nranks << " rank(s) (" << cpu_time_rank0 << " on rank 0)"
                << std::endl
                << "photons transported = " << pmc->nphot_run
                << ", scatterings = " << pmc->nscat_run << std::endl;
      if (phot > 0.0) {
        std::cout << "photons/wall_second = " << phot/wall_time << std::endl
                  << "cpu_seconds/photon  = " << cpu_time_sum/phot << std::endl;
      }
      if (scat > 0.0) {
        std::cout << "cpu_microseconds/scattering = " << 1.0e6*cpu_time_sum/scat
                  << std::endl;
      }
      const double busy = (wall_time > 0.0 && Globals::nranks > 0)
                          ? cpu_time_sum/(wall_time*Globals::nranks) : 0.0;
      std::cout << "cpu/wall per rank   = " << busy
                << " (1 = every rank busy the whole run)" << std::endl;
    }
    if (mem_max_kb > 0) {
      std::cout << "peak resident memory = " << mem_max_kb/1024.0 << " MB per rank (max), "
                << mem_sum_kb/1024.0 << " MB total" << std::endl;
    }
#ifdef OPENMP_PARALLEL
    double zc_omps = static_cast<double> (zonecycles) / omp_time;
    std::cout << std::endl << "omp wtime used = " << omp_time << std::endl;
    std::cout << "zone-cycles/omp_wsecond = " << zc_omps << std::endl;
#endif
  }

  delete pinput;
  // Before the Mesh: MonteCarloBlock holds a MeshBlock pointer, so tearing the Mesh down
  // first leaves ~MonteCarlo walking blocks whose MeshBlocks are already gone.
  if (MONTE_CARLO_ENABLED) {
    delete pmc;
    MCGridFile::Free();
  }
  delete pmesh;
  delete ptlist;
  delete pouts;

#ifdef MPI_PARALLEL
  MPI_Finalize();
#endif

  return(0);
}
