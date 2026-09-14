//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file tetrad.cpp
//! \brief implementation of functions for constructing and transforming tetrad frames

// C++ libraries
#include <cmath>
#include <complex>

// Athena++ classes headers
#include "tetrad.hpp"
#include "montecarlo.hpp"

#define SMALL_NUMBER 1.e-30
//#define DEBUG_FT

//----------------------------------------------------------------------------------------
//! \fn void ConstructTetrad(Real ucon[4], Real gcov[4][4],
//!                          Real ecov[4][4], Real econ[4][4]
//! \brief construct an orthonormal tetrad from one trial vector
// uses the Gram-Schmidt algorithm

void ConstructTetrad(Real ucon[4], Real gcov[4][4],
                     Real econ[4][4], Real ecov[4][4]) {

  // make a trial tetrad where time component is parallel to ucon, usually chosen
  // to be fluid velocity
  Real mag = sqrt(fabs(DotVec(ucon, ucon, gcov)));
  if (mag > SMALL_NUMBER) {
    //set 0 vector to normalized ucon
    for (int j = 0; j < 4; j++)
      econ[IMC0][j] = ucon[j]/mag;
  } else {
    // set 0 vector to time direction
    econ[IMC0][IMC0] = 1.; econ[IMC0][IMC1] = 0.;
    econ[IMC0][IMC2] = 0.; econ[IMC0][IMC3] = 0.;
    NormalizeVec(econ[IMC0], gcov);
  }

  // Construct rest of contravariant tetrad using coordinate directions as defaults
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      if (i != IMC0)
        econ[i][j] = KroneckerDelta(i,j); // diagonal trial
    }
  }

  ProjectVecSub(econ[IMC1], econ[IMC0], gcov);
  NormalizeVec(econ[IMC1], gcov);

  ProjectVecSub(econ[IMC2], econ[IMC0], gcov);
  ProjectVecSub(econ[IMC2], econ[IMC1], gcov);
  NormalizeVec(econ[IMC2], gcov);

  ProjectVecSub(econ[IMC3], econ[IMC0], gcov);
  ProjectVecSub(econ[IMC3], econ[IMC1], gcov);
  ProjectVecSub(econ[IMC3], econ[IMC2], gcov);
  NormalizeVec(econ[IMC3], gcov);

  // contravariant tetrad construction complete
  // begin construction covariant tetrad
  for (int i = 0; i < 4; i++) {
    ConToCov(econ[i], ecov[i], gcov);
    if (i == IMC0) {
      for (int j = 0; j < 4; j++) ecov[IMC0][j] *= -1;
    }
  }
  // covariant tetrad construction complete

  // check for orthonormality
#ifdef DEBUG_FT
  printf("--------------------------------------------------\n");
  printf("Othonormality check:\n");
  printf("econ[IMC0]: %g %g %g %g\n", econ[IMC0][IMC0], econ[IMC0][IMC1], econ[IMC0][IMC2], econ[IMC0][IMC3]);
  printf("econ[IMC1]: %g %g %g %g\n", econ[IMC1][IMC0], econ[IMC1][IMC1], econ[IMC1][IMC2], econ[IMC1][IMC3]);
  printf("econ[IMC2]: %g %g %g %g\n", econ[IMC2][IMC0], econ[IMC2][IMC1], econ[IMC2][IMC2], econ[IMC2][IMC3]);
  printf("econ[IMC3]: %g %g %g %g\n", econ[IMC3][IMC0], econ[IMC3][IMC1], econ[IMC3][IMC2], econ[IMC3][IMC3]);
  printf("ecov[IMC0]: %g %g %g %g\n", ecov[IMC0][IMC0], ecov[IMC0][IMC1], ecov[IMC0][IMC2], ecov[IMC0][IMC3]);
  printf("ecov[IMC1]: %g %g %g %g\n", ecov[IMC1][IMC0], ecov[IMC1][IMC1], ecov[IMC1][IMC2], ecov[IMC1][IMC3]);
  printf("ecov[IMC2]: %g %g %g %g\n", ecov[IMC2][IMC0], ecov[IMC2][IMC1], ecov[IMC2][IMC2], ecov[IMC2][IMC3]);
  printf("ecov[IMC3]: %g %g %g %g\n", ecov[IMC3][IMC0], ecov[IMC3][IMC1], ecov[IMC3][IMC2], ecov[IMC3][IMC3]);
  printf("gcov[IMC0]: %g %g %g %g\n", gcov[IMC0][IMC0], gcov[IMC0][IMC1], gcov[IMC0][IMC2], gcov[IMC0][IMC3]);
  printf("gcov[IMC1]: %g %g %g %g\n", gcov[IMC1][IMC0], gcov[IMC1][IMC1], gcov[IMC1][IMC2], gcov[IMC1][IMC3]);
  printf("gcov[IMC2]: %g %g %g %g\n", gcov[IMC2][IMC0], gcov[IMC2][IMC1], gcov[IMC2][IMC2], gcov[IMC2][IMC3]);
  printf("gcov[IMC3]: %g %g %g %g\n", gcov[IMC3][IMC0], gcov[IMC3][IMC1], gcov[IMC3][IMC2], gcov[IMC3][IMC3]);

  Real sum;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      sum = 0.;
      for (int k = 0; k < 4; k++) {
        sum += econ[i][k] * ecov[j][k];
      }
    }
    printf("sum: %g  index: %d\n",sum, i);


  }
  printf("--------------------------------------------------\n");
#endif

}



//----------------------------------------------------------------------------------------
//! \fn Real ObserverEnergy(Real ucon[4], Real kcon[4], Real gcov[4][4])
//! \brief the energy of kcon as measured by an observer with four-velocity ucon
//
// This is the time component that ConstructTetrad(ucon,...) followed by
// CoordinateToTetrad would produce, obtained without building the other three legs.
// Row zero of the covariant tetrad is fixed entirely by ucon: ConstructTetrad sets
// econ[0] = ucon/sqrt(|ucon.ucon|), lowers it, and flips the sign, so
//
//     ecov[0][mu] = -u_mu / sqrt(|u.u|)   and   k^(0) = -u_mu k^mu / sqrt(|u.u|).
//
// The Gram-Schmidt orthogonalization of legs 1-3 costs roughly ten metric contractions
// and four square roots and does not enter the answer, which matters because the callers
// that only want an energy -- the opacity frequency shift above all -- sit in the inner
// loop of the general pusher.
//
// The normalization is now a no-op for the Monte Carlo's own callers and is kept for
// generality only.  It used to do real work: a four-velocity was assembled and normalized
// at the cell center and then contracted with the metric at a photon, where u.u was -1
// only to first order in the offset -- 3e-3 at r = 2M on a Kerr-Schild grid.  Callers now
// obtain ucon from MonteCarloBlock::FluidFourVelocity, which rebuilds it from the stored
// primitives against the metric at the very point gcov is evaluated, so u.u = -1 there to
// roundoff.  Anything passing a vector normalized somewhere else still gets the division,
// which is why it stays; note it only ever corrected the magnitude, never the direction.
//
// The arithmetic below deliberately mirrors ConstructTetrad step for step, dividing ucon
// by the norm before lowering rather than after, and keeps the same degenerate branch.
// Lowering first would be algebraically identical but would round differently, and
// reproducing the operation order exactly means this can be swapped in for the tetrad
// without perturbing a single output bit.

Real ObserverEnergy(Real ucon[4], Real kcon[4], Real gcov[4][4]) {

  Real e0[4];
  Real mag = sqrt(fabs(DotVec(ucon, ucon, gcov)));
  if (mag > SMALL_NUMBER) {
    Real unorm[4];
    for (int j = 0; j < 4; j++) unorm[j] = ucon[j]/mag;
    ConToCov(unorm, e0, gcov);
  } else {
    // ucon is degenerate; fall back to the coordinate time direction, as ConstructTetrad
    // does in the same situation.
    Real unorm[4] = {1., 0., 0., 0.};
    NormalizeVec(unorm, gcov);
    ConToCov(unorm, e0, gcov);
  }

  Real energy = 0.;
  for (int j = 0; j < 4; j++) energy += -e0[j] * kcon[j];
  return energy;
}
//----------------------------------------------------------------------------------------
//! \fn void ConstructTetrad(Real ucon[4], Real vcon[4],
//!                          Real gcov[4][4], Real ecov[4][4],
//!                          Real econ[4][4]
//! \brief construct an orthonormal tetrad from two trial vectors
// uses the Gram-Schmidt algorithm

//SWD: Make calculation of ecov optional, could overload?
void ConstructTetrad(Real ucon[4], Real vcon[4], Real gcov[4][4],
                     Real econ[4][4], Real ecov[4][4]) {


  // make a trial tetrad where time component is parallel to ucon, usually chosen
  // to be fluid velocity
  Real mag = sqrt(fabs(DotVec(ucon, ucon, gcov)));
  if (mag > SMALL_NUMBER) {
    //set 0 vector to normalized ucon
    for (int j = 0; j < 4; j++)
      econ[IMC0][j] = ucon[j]/mag;
  } else {
    // set 0 vector to time direction
    econ[IMC0][IMC0] = 1.; econ[IMC0][IMC1] = 0.;
    econ[IMC0][IMC2] = 0.; econ[IMC0][IMC3] = 0.;
    NormalizeVec(econ[IMC0], gcov);
  }
  // make a trial tetrad where 3rd coordinate is pararallel to vcon
  for (int j = 0; j < 4; j++)
    econ[IMC3][j] = vcon[j];
  ProjectVecSub(econ[IMC3], econ[IMC0], gcov);
  mag = sqrt(fabs(DotVec(econ[IMC3], econ[IMC3], gcov)));
  if (mag > SMALL_NUMBER) {
    //set 3 vector to normalized vcon
    for (int j = 0; j < 4; j++)
      econ[IMC3][j] /= mag;
  } else {
    // set 3 vector to coordinate direction
    econ[IMC3][IMC0] = 0.; econ[IMC3][IMC1] = 0.;
    econ[IMC3][IMC2] = 0.; econ[IMC3][IMC3] = 1.;
    ProjectVecSub(econ[IMC3], econ[IMC0], gcov);
    mag = sqrt(fabs(DotVec(econ[IMC3], econ[IMC3], gcov)));
    for (int j = 0; j < 4; j++)
      econ[IMC3][j] /= mag;
  }
  // Construct rest of contravariant tetrad using coordinate directions as defaults
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      if ((i == IMC1) || (i == IMC2))
        econ[i][j] = KroneckerDelta(i,j); // diagonal trial
    }
  }
  ProjectVecSub(econ[IMC1], econ[IMC0], gcov);
  ProjectVecSub(econ[IMC1], econ[IMC3], gcov);
  NormalizeVec(econ[IMC1], gcov);

  ProjectVecSub(econ[IMC2], econ[IMC0], gcov);
  ProjectVecSub(econ[IMC2], econ[IMC3], gcov);
  ProjectVecSub(econ[IMC2], econ[IMC1], gcov);
  NormalizeVec(econ[IMC2], gcov);

  // contravariant tetrad construction complete
  // begin construction covariant tetrad
  for (int i = 0; i < 4; i++) {
    ConToCov(econ[i], ecov[i], gcov);
    if (i == IMC0) {
      for (int j = 0; j < 4; j++) ecov[IMC0][j] *= -1;
    }
  }


}

//----------------------------------------------------------------------------------------
//! \fn void ConstructTetrad(Real ucon[4], Real vcon[4], Real wcon[4],
//                           Real gcov[4][4], Real ecov[4][4],
//                           Real econ[4][4])
//  \brief construct an orthonormal tetrad using the fluid frame vector ucon and defining
//         e^mu_(3) = k^\mu

void ConstructTetrad(Real ucon[4], Real vcon[4], Real wcon[4],
                     Real gcov[4][4], Real econ[4][4],
                     Real ecov[4][4]) {

  // make a trial vector where time component is parallel to ucon, usually chosen
  // to be fluid velocity

  Real mag = sqrt(fabs(DotVec(ucon, ucon, gcov)));
  if (mag > SMALL_NUMBER) {
    //set 0 vector to normalized ucon
    for (int j = 0; j < 4; j++)
      econ[IMC0][j] = ucon[j]/mag;
  } else {
    // set 0 vector to time direction
    econ[IMC0][IMC0] = 1.; econ[IMC0][IMC1] = 0.;
    econ[IMC0][IMC2] = 0.; econ[IMC0][IMC3] = 0.;
    NormalizeVec(econ[IMC0], gcov);
  }

  // make a trial vector where 3 coordinate is pararallel to vcon
  for (int j = 0; j < 4; j++)
    econ[IMC3][j] = vcon[j];
  ProjectVecSub(econ[IMC3], econ[IMC0], gcov);
  mag = sqrt(fabs(DotVec(econ[IMC3], econ[IMC3], gcov)));
  if (mag > SMALL_NUMBER) {
    //set 3 vector to normalized vcon
    for (int j = 0; j < 4; j++)
      econ[IMC3][j] /= mag;
  } else {
    // set 3 vector to coordinate direction
    econ[IMC3][IMC0] = 0.; econ[IMC3][IMC1] = 0.;
    econ[IMC3][IMC2] = 0.; econ[IMC3][IMC3] = 1.;
    ProjectVecSub(econ[IMC3], econ[IMC0], gcov);
    mag = sqrt(fabs(DotVec(econ[IMC3], econ[IMC3], gcov)));
    for (int j = 0; j < 4; j++)
      econ[IMC3][j] /= mag;
  }

  // make a trial vector where 2 coordinate is pararallel to wcon
  for (int j = 0; j < 4; j++)
    econ[IMC2][j] = wcon[j];
  ProjectVecSub(econ[IMC2], econ[IMC0], gcov);
  ProjectVecSub(econ[IMC2], econ[IMC3], gcov);
  mag = sqrt(fabs(DotVec(econ[IMC2], econ[IMC2], gcov)));
  if (mag > SMALL_NUMBER) {
    //set 2 vector to normalized wcon
    for (int j = 0; j < 4; j++)
      econ[IMC2][j] /=mag;
  } else {
    // set 2 vector to coordinate direction
    econ[IMC2][0] = 0.; econ[IMC2][1] = 0.; econ[IMC2][2] = 1.; econ[IMC2][3] = 0.;
    ProjectVecSub(econ[IMC2], econ[IMC0], gcov);
    ProjectVecSub(econ[IMC2], econ[IMC3], gcov);
    mag = sqrt(fabs(DotVec(econ[IMC2], econ[IMC2], gcov)));
    for (int j = 0; j < 4; j++)
      econ[IMC2][j] /= mag;
  }

  // make a trial vector
  econ[IMC1][IMC0] = 1.; econ[IMC1][IMC1] = 1.;
  econ[IMC1][IMC2] = 1.; econ[IMC1][IMC3] = 1.;
  ProjectVecSub(econ[IMC1], econ[IMC0], gcov);
  ProjectVecSub(econ[IMC1], econ[IMC3], gcov);
  ProjectVecSub(econ[IMC1], econ[IMC2], gcov);
  NormalizeVec(econ[IMC1], gcov);

  // SWD: Might rethink if this is needed here
  ImposeRightHanded(econ,gcov);
  //printf("econ0: %e %e %e %e\n",econ[IMC0][IMC0],econ[IMC0][IMC1],econ[IMC0][IMC2],econ[IMC0][IMC3]);
  //printf("econ3: %e %e %e %e\n",econ[IMC3][IMC0],econ[IMC3][IMC1],econ[IMC3][IMC2],econ[IMC3][IMC3]);
  //printf("econ2: %e %e %e %e\n",econ[IMC2][IMC0],econ[IMC2][IMC1],econ[IMC2][IMC2],econ[IMC2][IMC3]);
  //printf("econ1: %e %e %e %e\n",econ[IMC1][IMC0],econ[IMC1][IMC1],econ[IMC1][IMC2],econ[IMC1][IMC3]);
  // contravariant tetrad construction complete
  // begin construction covariant tetrad
  for (int i = 0; i < 4; i++) {
    ConToCov(econ[i], ecov[i], gcov);
  }
  for (int j = 0; j < 4; j++) ecov[IMC0][j] *= -1;

  /*Real sum1,sum2;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      sum1 = 0.;
      sum2 = 0.;
      for (int k = 0; k < 4; k++) {
        sum1 += econ[i][k] * ecov[j][k];
        for (int l = 0; l < 4; l++) {
          sum2 += econ[i][k]*gcov[k][l]*econ[j][l];
        }
      }
      printf("orth: %d %d %g %g\n",i,j,sum1,sum2);
      }}*/

}


//SWD: Modify so this function is called only once
//----------------------------------------------------------------------------------------
//! \fn void ImposeRightHanded(Real econ[4][4], Real gcov[4][4])
//! \brief Check if tetrad is right-handed and reverse if not

void InitializeLeviCivita(Real levi[4][4][4][4]) {

  // Ensure that levi uses correct IMC values
  int conv[4];
  conv[0] = IMC0;
  conv[1] = IMC1;
  conv[2] = IMC2;
  conv[3] = IMC3;

  for (int i = 0; i < 4; i++) {
    int ic = conv[i];
    for (int j = 0; j < 4; j++) {
      int jc = conv[j];
      for (int k = 0; k < 4; k++) {
        int kc = conv[k];
        for (int l = 0; l < 4; l++) {
          int lc = conv[l];
          if (ic == jc || ic == kc || ic == lc || jc == kc || jc == lc || kc == lc)
            levi[ic][jc][kc][lc] = 0.;
          else {
            // Uses the simple procedure for ndim=3
            int diffprod = (jc-kc)*(kc-lc)*(lc-jc);
            diffprod /= abs(diffprod);
            if (ic == 0 || ic == 2)
              levi[ic][jc][kc][lc] = static_cast<int>(diffprod);
            else
              levi[ic][jc][kc][lc] = static_cast<int>(-diffprod);
            //printf("%d %d %d %d %g\n",ic,jc,kc,lc,levi[ic][jc][kc][lc]);
          }
        }}}}
}

//----------------------------------------------------------------------------------------
//! \fn void ImposeRightHanded(Real econ[4][4], Real gcov[4][4])
//! \brief Check if tetrad is right-handed and reverse if not

void ImposeRightHanded(Real econ[4][4], Real gcov[4][4]) {

  static Real levi[4][4][4][4];
  static bool init = false;
  if (!init) {
    InitializeLeviCivita(levi);
    init = true;
  }

  Real sum = 0.;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        for (int l = 0; l < 4; l++) {
          sum += levi[i][j][k][l]*econ[0][i]*econ[1][j]*econ[2][k]*econ[3][l];
        }}}}

  if (sum < 0.) {
    for (int i = 0; i < 4; i++)
      econ[IMC1][i] *= -1.;
  }
}


//----------------------------------------------------------------------------------------
//! \fn Real KroneckerDelta(int i, int j)
//! \brief Kronecker Delta function

Real KroneckerDelta(int i, int j) {

  if (i == j)
    return 1.;
  else
    return 0.0;

}

//----------------------------------------------------------------------------------------
//! \fn void ProjectVecSub(Real ucon[4], Real vcon[4],
//                          Real gcov[4][4])
//  \brief subtract four-vector projection of ucon onto vcon from ucon in metric gcov

void ProjectVecSub(Real ucon[4], Real vcon[4], Real gcov[4][4]) {

  Real vcon_dot_vcon = DotVec(vcon, vcon, gcov);
  Real ucon_dot_vcon = DotVec(ucon, vcon, gcov);

  if (fabs(vcon_dot_vcon) < SMALL_NUMBER) {
    printf("Warning: attempted to project out using a zero vector. Vector left as is.\n");
#ifdef DEBUG_FT
    printf("vdotv: %g\n", vcon_dot_vcon);
    printf("vcon: %g %g %g %g\n", vcon[IMC0], vcon[IMC1], vcon[IMC2], vcon[IMC3]);
    printf("gcov[IMC0]: %g %g %g %g\n", gcov[IMC0][IMC0], gcov[IMC0][IMC1], gcov[IMC0][IMC2], gcov[IMC0][IMC3]);
    printf("gcov[IMC1]: %g %g %g %g\n", gcov[IMC1][IMC0], gcov[IMC1][IMC1], gcov[IMC1][IMC2], gcov[IMC1][IMC3]);
    printf("gcov[IMC2]: %g %g %g %g\n", gcov[IMC2][IMC0], gcov[IMC2][IMC1], gcov[IMC2][IMC2], gcov[IMC2][IMC3]);
    printf("gcov[IMC3]: %g %g %g %g\n", gcov[IMC3][IMC0], gcov[IMC3][IMC1], gcov[IMC3][IMC2], gcov[IMC3][IMC3]);
#endif
    return;
  }

  for (int i = 0; i < 4; i++)
    ucon[i] -= vcon[i] * ucon_dot_vcon / vcon_dot_vcon;

}


//----------------------------------------------------------------------------------------
//! \fn Real DotVec(Real ucon[4], Real vcon[4], Real gcov[4][4])
//! \brief return dot (scalar) product of <ucon, vcon> in metric gcov

Real DotVec(Real ucon[4], Real vcon[4], Real gcov[4][4]) {

  Real dot = 0;

  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      dot += ucon[i] * gcov[i][j] * vcon[j];

  return dot;

}

//----------------------------------------------------------------------------------------
//! \fn void NormalizeVec(Real ucon[4], Real gcov[4][4])
//! \brief normalize vector ucon in metric gcov

void NormalizeVec(Real ucon[4], Real gcov[4][4]) {

  Real mag = sqrt(fabs(DotVec(ucon, ucon, gcov)));

  if (mag < SMALL_NUMBER) {
    printf("Warning: attempted to normalize a zero vector. Vector left as is.\n");
    printf("ucon: %g %g %g %g\n",ucon[IMC0],ucon[IMC1],ucon[IMC2],ucon[IMC3]);
    return;
  }

  for (int i = 0; i < 4; i ++)
    ucon[i] /= mag;

}

//----------------------------------------------------------------------------------------
//! \fn void ConToCov(Real ucon[4], Real ucov[4], Real gcov[4][4])
//! \brief converts contravariant vector ucon to covariant vector ucov in metric gcov

void ConToCov(Real ucon[4], Real ucov[4], Real gcov[4][4]) {

  for (int i = 0; i < 4; i++) {
    ucov[i] = 0.;
    for (int j = 0; j < 4; j++)
      ucov[i] += gcov[i][j] * ucon[j];
  }

}

//----------------------------------------------------------------------------------------
//! \fn void CovToCon(Real ucov[4], Real ucon[4], Real gcon[4][4])
//! \brief converts covariant vector ucov to contravariant vector ucon in metric gcon

void CovToCon(Real ucov[4], Real ucon[4], Real gcon[4][4]) {

  for (int i = 0; i < 4; i++) {
    ucon[i] = 0.;
    for (int j = 0; j < 4; j++)
      ucon[i] += gcon[i][j] * ucov[j];
  }

}

//----------------------------------------------------------------------------------------
//! \fn bool NormalObserver(Real gcon[4][4], Real ncon[4])
//! \brief four-velocity of the normal (Eulerian) observer
//
// n^mu = -alpha g^{mu t} with lapse alpha = 1/sqrt(-g^{tt}).  This is the frame the
// outputs reference directions and polarization to, so it lives here rather than being
// rebuilt at each call site: the wavevector and the Stokes parameters must be projected
// onto the same frame
//
// Returns false when g^{tt} is not negative, meaning the slice is not spacelike and there
// is no such observer.

bool NormalObserver(Real gcon[4][4], Real ncon[4]) {

  if (gcon[IMC0][IMC0] >= 0.) return false;
  Real alpha = 1.0/std::sqrt(-gcon[IMC0][IMC0]);
  for (int m = 0; m < 4; ++m) ncon[m] = -alpha*gcon[m][IMC0];
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn void CoordinateToTetrad(Real ucoord[4], Real utet[4],
//!                             Real ecov[4][4])
//! \brief transform vector ucoord from coordinate frame to utet in the tetrad frame

void CoordinateToTetrad(Real ucoord[4], Real utet[4],
                        Real ecov[4][4]) {

  for (int i = 0; i < 4; i++) {
    utet[i] = 0.;
    for (int j = 0; j< 4; j++)
      utet[i] += ecov[i][j] * ucoord[j];
  }
}


//----------------------------------------------------------------------------------------
//! \fn void TetradToCoordinate(Real utet[4], Real ucoord[4],
//!                             Real econ[4][4])
//! \brief transform vector utet from tetrad frame to ucoord in the coordinate frame

void TetradToCoordinate(Real utet[4], Real ucoord[4],
                        Real econ[4][4]) {

  for (int i = 0; i < 4; i++) {
    ucoord[i] = 0.;
    for (int j = 0; j < 4; j++)
      ucoord[i] += econ[j][i] * utet[j];
  }

}

//----------------------------------------------------------------------------------------
//! \fn void StokesToTensor(Real stokes[4], std::complex<Real> tensor[4][4])
//! \brief transform stokes vector invariant polarization tensor

void StokesToTensor(Real stokes[4], std::complex<Real> tensor[4][4]) {

  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) {
      tensor[i][j] = std::complex<Real>(0.,0.);
    }
  // Follows the conventions defined in Moscibrodzka&Gammie 2018
  tensor[IMC1][IMC1] = std::complex<Real>((stokes[0] + stokes[1]),0.);
  tensor[IMC1][IMC2] = std::complex<Real>(stokes[2], -stokes[3]);
  tensor[IMC2][IMC1] = std::complex<Real>(stokes[2], stokes[3]);
  tensor[IMC2][IMC2] = std::complex<Real>((stokes[0] - stokes[1]),0.);

}
//----------------------------------------------------------------------------------------
//! \fn void TensorToStokes(std::complex<Real> tensor[4][4], Real stokes[4])
//! \brief transform invariant polarization tensor to stokes vector

void TensorToStokes(std::complex<Real> tensor[4][4], Real stokes[4]) {

  // Follows the conventions defined in Moscibrodzka&Gammie 2018
  stokes[0] = 0.5 * (tensor[IMC1][IMC1] + tensor[IMC2][IMC2]).real();
  stokes[1] = 0.5 * (tensor[IMC1][IMC1] - tensor[IMC2][IMC2]).real();
  stokes[2] = 0.5 * (tensor[IMC1][IMC2] + tensor[IMC2][IMC1]).real();
  stokes[3] = 0.5 * (tensor[IMC2][IMC1] - tensor[IMC1][IMC2]).imag();

  //printf("stokes: %e %e %e %e\n",stokes[0],stokes[1],stokes[2],stokes[3]);
  //printf("stokes: %e %e\n",SQR(stokes[1])+SQR(stokes[2]),stokes[0]*stokes[0]);
  Real norm = stokes[0];
  for (int i = 0; i < 4; i++)
    stokes[i] /= norm;

}

//----------------------------------------------------------------------------------------
//! \fn void LorentzBoostVector(Real vel[4], Real kold[4])
//! \brief boost photon momentum vector

void LorentzBoostVector(Real vel[4], Real kold[4]) {

  Real knew[4];

  knew[0] = vel[0] * kold[0];
  for (int l=1; l<4; l++)
    knew[0] -= vel[l] * kold[l];
  for (int m=1; m<4; m++) {
    knew[m] = kold[m] - vel[m] * kold[0];
    for (int l=0; l<4; l++) {
      knew[m] += vel[l]*vel[m]/(1.+vel[0]);
    }
  }
  for (int l=0; l<4; l++)
    kold[l] = knew[l];

}
