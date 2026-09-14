#ifndef MONTE_CARLO_TETRAD_HPP_
#define MONTE_CARLO_TETRAD_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file tetrad.hpp
//! \brief orthonormal frames and the vector algebra that builds them.
//
// These functions act on four-vectors.  Frame transformatinos for Monte Carlo quantities
// like photons are found in photon_frames.cpp.
//
// Implemented in tetrad.cpp.

// C++ headers
#include <complex>

// Athena++ headers
#include "../athena.hpp"

// build an orthonormal tetrad by Gram-Schmidt against one, two or three given
// directions; the timelike leg is taken parallel to ucon and the remaining legs are
// filled out from coordinate directions.
void ConstructTetrad(Real ucon[4], Real gcov[4][4],
                     Real econ[4][4], Real ecov[4][4]);
void ConstructTetrad(Real ucon[4], Real vcon[4], Real gcov[4][4],
                     Real econ[4][4], Real ecov[4][4]);
void ConstructTetrad(Real ucon[4], Real vcon[4], Real wcon[4],
                     Real gcov[4][4], Real econ[4][4],
                     Real ecov[4][4]);
void InitializeLeviCivita(Real levi[4][4][4][4]);
void ImposeRightHanded(Real econ[4][4], Real gcov[4][4]);
Real KroneckerDelta(int i, int j);

// Energy of kcon measured by an observer with four-velocity ucon.  Equal to the time
// component ConstructTetrad + CoordinateToTetrad would give, but skips the Gram-Schmidt
// for legs 1-3, which do not enter it.  For callers in the pusher's inner loop that want
// a frequency shift and nothing else.
Real ObserverEnergy(Real ucon[4], Real kcon[4], Real gcov[4][4]);

// vector algebra in a metric
void ProjectVecSub(Real ucon[4], Real vcon[4], Real gcov[4][4]);
Real DotVec(Real ucon[4], Real vcon[4], Real gcov[4][4]);
void NormalizeVec(Real ucon[4], Real gcov[4][4]);
void ConToCov(Real ucon[4], Real ucov[4], Real gcov[4][4]);
void CovToCon(Real ucov[4], Real ucon[4], Real gcon[4][4]);

// four-velocity of the normal (Eulerian) observer, n^mu = -alpha g^{mu t}.  Shared so
// that everything referencing an output to "the normal frame" means the same frame.
bool NormalObserver(Real gcon[4][4], Real ncon[4]);

// move a four-vector between the coordinate basis and a tetrad
void CoordinateToTetrad(Real ucoord[4], Real utet[4], Real ecov[4][4]);
void TetradToCoordinate(Real utet[4], Real ucoord[4], Real econ[4][4]);

// polarization, carried as an invariant tensor between scatterings
// The canonical statement of the Stokes <-> coherency-tensor convention
// (Moscibrodzka & Gammie 2018, equations 13 and 14):
//
//   N11 = I + Q,  N12 = U - iV,  N21 = U + iV,  N22 = I - Q.
//
// Both assume the tensor occupies the (1,2) block, which holds when the tetrad's third
// spatial leg is along k so that the polarization is transverse.  Callers working in some
// other pair of transverse directions -- polarization.cpp uses the meridian pair (l,r) --
// can pack that pair into the same block and use these unchanged
//
// Note that the two are not exact inverses: TensorToStokes divides through by I, returning
// stokes[0] = 1, while StokesToTensor does not.
void StokesToTensor(Real stokes[4], std::complex<Real> tensor[4][4]);
void TensorToStokes(std::complex<Real> tensor[4][4], Real stokes[4]);

void LorentzBoostVector(Real vel[4], Real kold[4]);

#endif // MONTE_CARLO_TETRAD_HPP_
