#ifndef POLARIZATION_HPP
#define POLARIZATION_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file polarization.hpp
//! \brief the polarization interface shared by the legacy and general pushers
//
// The polarized scattering routines in scattering.cpp do not work in an arbitrary basis.
// They require:
//
//   - the wavevector in cartesian components, and
//   - Q and U referenced to the meridian plane containing the global cartesian z axis
//     and k-hat.  ScatterThomsonPolarized reads that reference straight off the stored
//     components (mu = kz, phi = acos(kx/sin(theta)))
//
// Both legacy pushers run only in flat spacetime (cartesian, spherical-polar).  A flat
// spacetime has a globally covariantly-constant cartesian frame and the
// convention above references Q and U to exactly that frame.  A photon between
// scatterings travels a straight line, so k-hat is constant in cartesian components, the
// meridian plane is constant, and Q and U do not change.  The stored components of k in
// spherical does change along the ray, which is why the wavevector has to be
// rotated at a scattering even though the Stokes parameters do not.
//
// In a curved spacetime no global frame exists, so the general pusher must parallel
// transport the coherency tensor step by step.
//
// Faraday rotation rotates Q into U continuously along the ray, so the Stokes parameters
// stop being constant between scatterings even in flat spacetime.

// Real, and nothing else.  This header is included from both montecarlo.hpp and
// photon.hpp, which include each other, so it must not pull in anything from the
// module itself; athena.hpp is a leaf and safe.
#include "../athena.hpp"

// enum for checking polarization treatment
// Byte-backed so the member occupies the same storage a bool did and the hot-path test
// compiles to the same single compare-against-memory.
enum MCPolarization : unsigned char {MCPOL_NONE = 0, MCPOL_LINEAR = 1,
                                     MCPOL_CIRCULAR = 2};

//! \brief true when any polarization is tracked
inline bool IsPolarized(MCPolarization p) {return p != MCPOL_NONE;}
//! \brief true when Stokes V is carried and stored
inline bool TracksCircular(MCPolarization p) {return p == MCPOL_CIRCULAR;}

//! \brief number of Stokes planes stored alongside I: none, then Q and U, then V
inline int NumStokesStored(MCPolarization p) {
  if (p == MCPOL_CIRCULAR) return 3;
  if (p == MCPOL_LINEAR) return 2;
  return 0;
}

// enum slots of the stored Stokes planes, in file order
enum {MCISQ = 0, MCISU = 1, MCISV = 2};

//! \brief the input-file spelling of a mode, for error messages and file headers
inline const char *GetMCPolarizationName(MCPolarization p) {
  if (p == MCPOL_LINEAR) return "linear";
  if (p == MCPOL_CIRCULAR) return "circular";
  return "none";
}

class MonteCarloBlock;
class Photon;

// rotate the photon into the basis the polarized scattering routines assume
void ToScatteringBasis(MonteCarloBlock *pmcb, Photon *pphot, int ip);

// coherency tensor -> Stokes, in the scattering basis (general pusher)
void CoherencyToScatteringStokes(MonteCarloBlock *pmcb, Photon *pphot, int ip);

// Stokes -> coherency tensor, undoing CoherencyToScatteringStokes
void ScatteringStokesToCoherency(MonteCarloBlock *pmcb, Photon *pphot, int ip);

// the normal observer's tetrad at the photon and the wavevector in it; the one place
// the output frame is built, so every output measures against the same axes
bool NormalFrameWavevector(MonteCarloBlock *pmcb, Photon *pphot, int ip,
                           Real econ[4][4], Real ecov[4][4], Real ktet[4]);

// coherency tensor -> Stokes in the normal observer's frame, for the outputs
void CoherencyToObserverStokes(MonteCarloBlock *pmcb, Photon *pphot, int ip);

// rotate the photon back out of the scattering basis
void FromScatteringBasis(MonteCarloBlock *pmcb, Photon *pphot, int ip);

// The meridian pair for a unit direction n in some orthonormal frame:
//   l = normalize(zref - (zref.n) n),   r = n x l,
// with zref a unit vector in the same components.  Public so that a problem generator
// that hands over Stokes parameters can reference them to exactly the pair the framework
// will read them against.  When n is parallel to zref the meridian is undefined and a
// deterministic pair is returned instead, the limit approached along the phi = 0
// meridian; see the definition for the convention.  Returns false only when n has no
// magnitude to normalize.
bool MeridianPair(const Real n[3], const Real zref[3], Real lhat[3], Real rhat[3]);

#endif // POLARIZATION_HPP
