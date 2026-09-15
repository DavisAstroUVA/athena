//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file default_pgen.cpp
//! \brief Provides default (empty) versions of all functions in problem generator files
//! This means user does not have to implement these functions if they are not needed.
//!
//! The attribute "weak" is used to ensure the loader selects the user-defined version of
//! functions rather than the default version given here.
//!
//! The attribute "alias" may be used with the "weak" functions (in non-defining
//! declarations) in order to have them refer to common no-operation function definition
//! in the same translation unit. Target function must be specified by mangled name
//! unless C linkage is specified.
//!
//! This functionality is not in either the C nor the C++ standard. These GNU extensions
//! are largely supported by LLVM, Intel, IBM, but may affect portability for some
//! architecutres and compilers. In such cases, simply define all 6 of the below class
//! functions in every pgen/*.cpp file (without any function attributes).

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../monte_carlo/montecarlo.hpp"
#include "../parameter_input.hpp"

// 3x members of Mesh class:

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in Mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================

void __attribute__((weak)) Mesh::InitUserMeshData(ParameterInput *pin) {
  // do nothing
  return;
}

//========================================================================================
//! \fn void Mesh::UserWorkInLoop()
//! \brief Function called once every time step for user-defined work.
//========================================================================================

void __attribute__((weak)) Mesh::UserWorkInLoop() {
  // do nothing
  return;
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//! \brief Function called after main loop is finished for user-defined work.
//========================================================================================

void __attribute__((weak)) Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  // do nothing
  return;
}

// 4x members of MeshBlock class:

//========================================================================================
//! \fn void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in MeshBlock class.  Can also be
//! used to initialize variables which are global to other functions in this file.
//! Called in MeshBlock constructor before ProblemGenerator.
//========================================================================================

void __attribute__((weak)) MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  // do nothing
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Should be used to set initial conditions.
//========================================================================================

void __attribute__((weak)) MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // In practice, this function should *always* be replaced by a version
  // that sets the initial conditions for the problem of interest.
  return;
}

//========================================================================================
//! \fn void MeshBlock::UserWorkInLoop()
//! \brief Function called once every time step for user-defined work.
//========================================================================================

void __attribute__((weak)) MeshBlock::UserWorkInLoop() {
  // do nothing
  return;
}

//========================================================================================
//! \fn void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin)
//! \brief Function called before generating output files
//========================================================================================

void __attribute__((weak)) MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  // do nothing
  return;
}

//========================================================================================
//! \fn void MonteCarlo::InitUserMonteCarloData(ParameterInput *pin)
//! \brief Initializes user data specific to MonteCarlo class
//========================================================================================

void __attribute__((weak)) MonteCarlo::InitUserMonteCarloData(ParameterInput *pin){

}

//========================================================================================
//! \fn void MonteCarlo::InitUserMonteCarloBlockData(ParameterInput *pin)
//! \brief Initializes user data specific to MonteCarloBlock class
//========================================================================================

void __attribute__((weak)) MonteCarloBlock::InitUserMonteCarloBlockData(ParameterInput *pin){

}

//========================================================================================
//! \fn void MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin)
//! \brief Analogous to problem generator but used in support of InitializePhoton
//========================================================================================

void __attribute__((weak)) MonteCarloBlock::MonteCarloProblemGenerator(ParameterInput *pin){

}

//========================================================================================
//! \fn void MonteCarloBlock::InitializePhoton(Photon *pphot, int ips, int ipe)
//! \brief Initializes Photon packets before integration
//========================================================================================

void __attribute__((weak)) MonteCarloBlock::InitializePhoton(Photon *pphot, int ips,
                                                             int ipe, int etype) {

}

//========================================================================================
//! \fn void MonteCarloBlock::FinalizePhoton(Photon *pphot, int ip)
//! \brief Complete work at end of photon packets before integration
//========================================================================================

void __attribute__((weak)) MonteCarloBlock::FinalizePhoton(Photon *pphot, int ip) {

}

//========================================================================================
//! \fn void MonteCarlo::UserWorkAfterRebalance(ParameterInput *pin)
//! \brief Called after the Monte Carlo blocks have followed a mesh redistribution, so a
//! problem generator can rebuild anything it keeps per local block or sized to nblocal
//========================================================================================

void __attribute__((weak)) MonteCarlo::UserWorkAfterRebalance(ParameterInput *pin) {
  return;
}

//========================================================================================
//! \fn void MonteCarloBlock::UserWorkAfterTransfer(int etype)
//! \brief Do work after each stage of photon transfer (multiple emission types)
//========================================================================================

void __attribute__((weak)) MonteCarloBlock::UserWorkAfterTransfer(int etype) {

}