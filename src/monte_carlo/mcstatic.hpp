#ifndef MCSTATIC_HPP
#define MCSTATIC_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mcstatic.hpp
//! \brief checks whether fluid is static (e.g. Monte Carlo post-processing)
//
// A Monte Carlo post-processing run (MONTE_CARLO_ENABLED with <montecarlo>/dynamic false)
// never enters the time integrator: main.cpp hands every cycle to RunMonteCarlo instead.
// The integrator's registers and fluxes, the constrained-transport EMFs, and the
// face-centered metric and frame-transformation arrays are uneeded. They are left empty
// in Monte Carlo post-processingto reduce the memory footprint.
//
// Mesh is constructed before MonteCarlo (see main.cpp), so this reads the input directly
// rather than asking the MonteCarlo object; the key and the default are the ones
// MonteCarlo::MonteCarlo uses.  A build without the Monte Carlo module always answers
// false, so hydro builds are untouched.

#include "../athena.hpp"
#include "../parameter_input.hpp"

inline bool HydroIsStatic(ParameterInput *pin) {
  if (!MONTE_CARLO_ENABLED) return false;
  return !pin->GetOrAddBoolean("montecarlo", "dynamic", false);
}

#endif // MCSTATIC_HPP
