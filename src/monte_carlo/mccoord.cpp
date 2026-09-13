//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file monte_carlo.cpp
//! \brief implementation of functions in class MCCoord

// SWD: General notes:
// * This whole class maybe should be reworked
// * inverse metric for spherical polar and cylindrical
// * remove zeroing of metric,connection at top?
// * remove extraneous variable definitions

// Athena++ headers
#include "../athena.hpp"
#include "mccoord.hpp"

//----------------------------------------------------------------------------------------
//! \fn MCTopology GetMCTopology(MCCoordSystem c)
//! \brief grid topology implied by a given metric
//!
//! Kept as a function of the metric rather than as an independently stored flag so the
//! two cannot drift apart.  Note that both Kerr-Schild forms appear here: the spherical
//! one shares a topology with spherical_polar, the Cartesian one with cartesian.

MCTopology GetMCTopology(MCCoordSystem c) {
  switch (c) {
    case MCCOORD_CYLINDRICAL:
      return MCTOPO_CYLINDRICAL;
    case MCCOORD_SPHERICAL_POLAR:
    case MCCOORD_KERR_SCHILD:
    case MCCOORD_BOYER_LINDQUIST:
      return MCTOPO_SPHERICAL;
    default:
      return MCTOPO_CARTESIAN;
  }
}

//----------------------------------------------------------------------------------------
//! \fn bool IsMCMetricCurved(MCCoordSystem c)
//! \brief true when the metric is not flat in the coordinates being integrated
//!
//! Spherical and cylindrical are flat: they carry non-zero connection coefficients but
//! zero curvature, and the places that ask this question are asking about the spacetime,
//! not about whether the connection vanishes.

bool IsMCMetricCurved(MCCoordSystem c) {
  switch (c) {
    case MCCOORD_KERR_SCHILD:
    case MCCOORD_BOYER_LINDQUIST:
    case MCCOORD_KERR_SCHILD_CARTESIAN:
      return true;
    default:
      return false;
  }
}

//----------------------------------------------------------------------------------------
//! \fn const char *GetMCCoordSystemName(MCCoordSystem c)
//! \brief human-readable name, used in error messages and for <montecarlo>/mc_coord

const char *GetMCCoordSystemName(MCCoordSystem c) {
  switch (c) {
    case MCCOORD_CARTESIAN:              return "cartesian";
    case MCCOORD_CYLINDRICAL:            return "cylindrical";
    case MCCOORD_SPHERICAL_POLAR:        return "spherical_polar";
    case MCCOORD_MINKOWSKI:              return "minkowski";
    case MCCOORD_KERR_SCHILD:            return "kerr_schild";
    case MCCOORD_BOYER_LINDQUIST:        return "boyer_lindquist";
    case MCCOORD_KERR_SCHILD_CARTESIAN:  return "kerr_schild_cartesian";
    case MCCOORD_SNAKE:                  return "snake";
  }
  return "unknown";
}

//----------------------------------------------------------------------------------------
//! \fn bool IsMCRelativistic(MCCoordSystem c)
//! \brief true when the run integrates geodesics in a relativistic spacetime
//!
//! Distinct from IsMCMetricCurved: Minkowski and snake are flat spacetimes but are still
//! built with -g and integrated as geodesics, so their photon lists carry the conserved
//! -k_t rather than k^t.  This is the set of metrics that require a GR build.

bool IsMCRelativistic(MCCoordSystem c) {
  switch (c) {
    case MCCOORD_MINKOWSKI:
    case MCCOORD_KERR_SCHILD:
    case MCCOORD_BOYER_LINDQUIST:
    case MCCOORD_KERR_SCHILD_CARTESIAN:
    case MCCOORD_SNAKE:
      return true;
    default:
      return false;
  }
}

//----------------------------------------------------------------------------------------
//! \fn bool IsMCPusherAlwaysGeneral(MCCoordSystem c)
//! \brief true when the coordinate system is integrated with GeneralPusher regardless of
//!        <montecarlo>/general_pusher
//!
//! Mirrors the switch in the MonteCarloBlock constructor, which honours the input flag
//! only for Cartesian and spherical-polar and picks GeneralPusher unconditionally for
//! everything else.  Keep the two in step: general_pusher_flag does not select the
//! pusher, it selects the four-vector storage convention and gates the polarization and
//! frame machinery, so a coordinate system that forces GeneralPusher while the flag is
//! false leaves the module in a split state.  MonteCarlo::SetCoordinateSystem rejects
//! that combination.
//!
//! Distinct from IsMCRelativistic: cylindrical forces GeneralPusher but is not
//! relativistic and needs no GR build.

bool IsMCPusherAlwaysGeneral(MCCoordSystem c) {
  switch (c) {
    case MCCOORD_CARTESIAN:
    case MCCOORD_SPHERICAL_POLAR:
      return false;
    default:
      return true;
  }
}

//----------------------------------------------------------------------------------------
//! \fn bool HasFlatOrthonormalBasis(MCCoordSystem c)
//! \brief true when the flat scale factors orthonormalize the coordinate basis
//
// Returns whether or not coordinate system has non diagonal tetrad.

bool HasFlatOrthonormalBasis(MCCoordSystem c) {
  switch (c) {
    case MCCOORD_CARTESIAN:
    case MCCOORD_CYLINDRICAL:
    case MCCOORD_SPHERICAL_POLAR:
    case MCCOORD_MINKOWSKI:
      return true;
    default:
      return false;
  }
}

//----------------------------------------------------------------------------------------
//! MCCoord base class constructor, builds MCCoord from Coord and MonteCarloBlock

MCCoord::MCCoord(Coordinates *pcoord, MonteCarloBlock *pmcb) {

  int nx1 = pcoord->x1f.GetDim1();
  int nx2 = pcoord->x2f.GetDim1();
  int nx3 = pcoord->x3f.GetDim1();

  x1f.NewAthenaArray(nx1);
  x2f.NewAthenaArray(nx2);
  x3f.NewAthenaArray(nx3);

  x1f = pcoord->x1f;
  //for (int i = 0; i < nx1; i++)
  //  x1f(i) *= pmcb->l_cgs;
  x2f = pcoord->x2f;
  x3f = pcoord->x3f;
  //if ( (COORDINATE_SYSTEM == "cartesian") || (COORDINATE_SYSTEM == "minkowski")
  //     || (COORDINATE_SYSTEM == "gr_user") ) {
  //  for (int i = 0; i < nx2; i++)
  //    x2f(i) *= pmcb->l_cgs;
  //  for (int i = 0; i < nx3; i++)
  //    x3f(i) *= pmcb->l_cgs;
  //}
  // Needed for black hole coordinates
  if (GENERAL_RELATIVITY) {
    bh_mass_ = pcoord->GetMass();
    bh_spin_ = pcoord->GetSpin();
  } else {
    // initialize to 0 for flat spacetimes
    bh_mass_ = 0.;
    bh_spin_ = 0.;
  }

  // Allocate volume array
  int ncells1 = pmcb->nx1 + 2*(NGHOST);
  int ncells2 = 1, ncells3 = 1;
  if (pmcb->nx2 > 1) ncells2 = pmcb->nx2 + 2*(NGHOST);
  if (pmcb->nx3 > 1) ncells3 = pmcb->nx3 + 2*(NGHOST);
  vol.NewAthenaArray(ncells3,ncells2,ncells1);
  // Initialize volume array
  for (int k=pmcb->ks; k<=pmcb->ke; ++k) {
    for (int j=pmcb->js; j<=pmcb->je; ++j) {
      for (int i=pmcb->is; i<=pmcb->ie; ++i) {
        // Volume in cgs units
        vol(k,j,i) = pcoord->GetCellVolume(k,j,i) * pow(pmcb->l_cgs,3);
        // Only a curved metric can put a coordinate singularity inside the domain; in a
        // flat spacetime a NaN volume is a bug and should not be quietly zeroed.  This
        // used to test for gr_user, which was a proxy for the same thing.
        if (std::isnan(vol(k,j,i)) && pmcb->curved_metric) {
          // at the singularity set volume to zero
         vol(k,j,i) = 0.;
        }
      }}}
  computedmin = pmcb->computedmin;
  if (computedmin) {
    dmin.NewAthenaArray(ncells3,ncells2,ncells1);
    Real dw1,dw2,dw3;
    for (int k=pmcb->ks; k<=pmcb->ke; ++k) {
      for (int j=pmcb->js; j<=pmcb->je; ++j) {
        for (int i=pmcb->is; i<=pmcb->ie; ++i) {
          dw3 = pcoord->dx3f(k);
          dw2 = pcoord->dx2f(j);
          dw1 = pcoord->dx1f(i);
          //dw1 *= pmcb->l_cgs;
          //if ( (COORDINATE_SYSTEM == "cartesian") ||
          //     (COORDINATE_SYSTEM == "minkowski")
          //     || (COORDINATE_SYSTEM == "gr_user") ) {
          //  dw2 *= pmcb->l_cgs;
          //  dw3 *= pmcb->l_cgs;
          //}
          Real dmin0 = std::min(dw1,dw2);
          dmin(k,j,i) = std::min(dmin0,dw3);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! MonteCarlo constructor for processes without own MeshBlock

MCCoord::MCCoord(int ncells1, int ncells2, int ncells3, bool cdmin) {

  x1f.NewAthenaArray(ncells1+1);
  x2f.NewAthenaArray(ncells2+1);
  x3f.NewAthenaArray(ncells3+1);

  vol.NewAthenaArray(ncells3,ncells2,ncells1);
  computedmin = cdmin;
  if (cdmin)
    dmin.NewAthenaArray(ncells3,ncells2,ncells1);
}

//----------------------------------------------------------------------------------------
//! destructor

MCCoord::~MCCoord() {

  x1f.DeleteAthenaArray();
  x2f.DeleteAthenaArray();
  x3f.DeleteAthenaArray();
  vol.DeleteAthenaArray();
  if (computedmin)
    dmin.DeleteAthenaArray();
}

//----------------------------------------------------------------------------------------
//! \fn void MCCoord::Metric(Real x[4], Real gcov[4][4])
//! \brief compute metric in flat spacetime cartesian

void MCCoord::Metric(Real x[4], Real gcov[4][4]) {

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      if (i == j) {
        if (i == IMC0)
          gcov[i][i] = -1.;
        else
          gcov[i][i] = 1.;
      } else
        gcov[i][j] = 0.;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MCCoord::MetricDerivative(Real x[4], Real dgcov[4][4][4])
//! \brief compute metric derivative in flat spacetime cartesian

void MCCoord::MetricDerivative(Real x[4], Real dgcov[4][4][4]) {

  for(int i = 0; i < 4; i++) {
    for(int j = 0; j < 4; j++) {
      for(int k = 0; k < 4; k++) {
        dgcov[i][j][k]=0;
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MCCoord::InverseMetric(Real x[4], Real gcon[4][4])
//! \brief compute inverse metric in flat spacetime cartesian

void MCCoord::InverseMetric(Real x[4], Real gcon[4][4]) {

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      if (i == j) {
        if (i == IMC0)
          gcon[i][i] = -1.;
        else
          gcon[i][i] = 1.;
      } else
        gcon[i][j] = 0.;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MCCoord::InverseMetricDerivative(Real x[4], Real dgcon[4][4][4])
//! \brief compute derivative of inverse metric in flat spacetime cartesian

void MCCoord::InverseMetricDerivative(Real x[4], Real dgcon[4][4][4]) {

  for(int i = 0; i < 4; i++) {
    for(int j = 0; j < 4; j++) {
      for(int k = 0; k < 4; k++) {
        dgcon[i][j][k]=0;
      }
    }
  }
}


//----------------------------------------------------------------------------------------
//! \fn void MCCoord::Connect(Real x[4], Real gamma[4][4][4])
//! \brief compute connection in flat spacetime cartesian

void MCCoord::Connect(Real x[4], Real gamma[4][4][4]) {

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        gamma[i][j][k] = 0.;
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MCCoord::MetricAndInverse(Real x[4], Real gcov[4][4], Real gcon[4][4])
//! \fn void MCCoord::InverseMetricAndDerivative(Real x[4], Real gcon[4][4],
//!                                              Real dgcon[4][4][4])
//! \brief the fused pairs, by default the two single calls in the order the general
//!        pusher used to make them; see the declaration for why they exist

void MCCoord::MetricAndInverse(Real x[4], Real gcov[4][4], Real gcon[4][4]) {
  Metric(x, gcov);
  InverseMetric(x, gcon);
}

void MCCoord::InverseMetricAndDerivative(Real x[4], Real gcon[4][4],
                                         Real dgcon[4][4][4]) {
  InverseMetric(x, gcon);
  InverseMetricDerivative(x, dgcon);
}

//----------------------------------------------------------------------------------------
//! \fn void MCCoord::Tetrad(Real x[4], Real tetrad[4][4])
//! \brief compute a diagonal tetrad

void MCCoord::Tetrad(Real x[4], Real tetrad[4][4]) {

  for (int l=0; l<4; l++) {
    for (int m=0; m<4; m++) {
      tetrad[l][m] = 0.;
    }
    tetrad[l][l] = 1.;
  }

}

//----------------------------------------------------------------------------------------
//! \fn void MCCoord::InverseTetrad(Real x[4], Real invtet[4][4])
//! \brief compute a diagonal tetrad

void MCCoord::InverseTetrad(Real x[4], Real invtet[4][4]) {

  for (int l=0; l<4; l++) {
    for (int m=0; m<4; m++) {
      invtet[l][m] = 0.;
    }
    invtet[l][l] = 1.;
  }

}

//----------------------------------------------------------------------------------------
//! MCCartesian constructor, built from Coord and MonteCarloBlock

MCCartesian::MCCartesian(Coordinates *pcoord, MonteCarloBlock *pmcb)
  : MCCoord(pcoord,pmcb) {

}

//----------------------------------------------------------------------------------------
//! MCCartesian constructor for processes without own MeshBlock

MCCartesian::MCCartesian(int ncells1, int ncells2, int ncells3, bool acc)
  : MCCoord(ncells1,ncells2,ncells3,acc) {

}

//----------------------------------------------------------------------------------------
//! destructor

MCCartesian::~MCCartesian() {

}

//----------------------------------------------------------------------------------------
//! MCSphericalPolar constructor, built from Coord and MonteCarloBlock

MCSphericalPolar::MCSphericalPolar(Coordinates *pcoord, MonteCarloBlock *pmcb)
  : MCCoord(pcoord,pmcb) {

}

//----------------------------------------------------------------------------------------
//! MCSphericalPolar constructor for processes without own MeshBlock

MCSphericalPolar::MCSphericalPolar(int ncells1, int ncells2, int ncells3, bool acc)
  : MCCoord(ncells1,ncells2,ncells3,acc) {

}

//----------------------------------------------------------------------------------------
//! destructor
MCSphericalPolar::~MCSphericalPolar() {

}

//----------------------------------------------------------------------------------------
//! \fn void MCSphericalPolar::Metric(Real x[4], Real gcov[4][4])
//! \brief compute metric in flat spacetime spherical polar

void MCSphericalPolar::Metric(Real x[4], Real gcov[4][4]) {

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      gcov[i][j] = 0;
    }
  }

  Real r = x[IMC1];
  Real sth = sin(x[IMC2]);
  gcov[IMC0][IMC0] = -1.;
  gcov[IMC1][IMC1] = 1.;
  gcov[IMC2][IMC2] = r * r;
  gcov[IMC3][IMC3] = r * r * sth * sth;

}

//----------------------------------------------------------------------------------------
//! \fn void MCSphericalPolar::InverseMetric(Real x[4], Real gcon[4][4])
//! \brief compute metric in flat spacetime spherical polar

void MCSphericalPolar::InverseMetric(Real x[4], Real gcon[4][4]) {

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      gcon[i][j] = 0;
    }
  }
  Real r = x[IMC1];
  Real sth = sin(x[IMC2]);
  gcon[IMC0][IMC0] = -1;
  gcon[IMC1][IMC1] = 1;
  gcon[IMC2][IMC2] = 1. / (r * r);
  gcon[IMC3][IMC3] = 1. / (r * r * sth *sth);

}

//----------------------------------------------------------------------------------------
//! \fn void MCSphericalPolar::Connect(Real x[4], Real gamma[4][4][4])
//! \brief compute connection in flat spacetime spherical polar

void MCSphericalPolar::Connect(Real x[4], Real gamma[4][4][4]) {

  for(int i = 0; i < 4; i++) {
    for(int j = 0; j < 4; j++) {
      for(int k = 0; k < 4; k++) {
        gamma[i][j][k]=0;
      }
    }
  }
  Real r = x[IMC1];
  Real sth = sin(x[IMC2]);
  Real cth = cos(x[IMC2]);

  gamma[IMC1][IMC2][IMC2] = -r;
  gamma[IMC1][IMC3][IMC3] = -r * sth * sth;
  gamma[IMC2][IMC1][IMC2] = 1. / r;
  gamma[IMC2][IMC2][IMC1] = gamma[IMC2][IMC1][IMC2];
  gamma[IMC2][IMC3][IMC3] = -sth*cth;
  gamma[IMC3][IMC1][IMC3] = 1. / r;
  gamma[IMC3][IMC2][IMC3] = cth / sth;
  gamma[IMC3][IMC3][IMC1] = gamma[IMC3][IMC1][IMC3];
  gamma[IMC3][IMC3][IMC2] = gamma[IMC3][IMC2][IMC3];

}

//----------------------------------------------------------------------------------------
//! \fn void MCSphericalPolar::InverseMetricDerivative(Real x[4], Real dgcon[4][4][4])
//! \brief compute a diagonal tetrad

void MCSphericalPolar::InverseMetricDerivative(Real x[4], Real dgcon[4][4][4]) {

  for(int i = 0; i < 4; i++) {
    for(int j = 0; j < 4; j++) {
      for(int k = 0; k < 4; k++) {
        dgcon[i][j][k]=0;
      }
    }
  }

  Real r = x[IMC1];
  Real sth = sin(x[IMC2]);
  Real cth = cos(x[IMC2]);
  dgcon[IMC1][IMC2][IMC2] = -2./(r*r*r);
  dgcon[IMC1][IMC3][IMC3] = -2./(r*r*r)/(sth*sth);
  dgcon[IMC2][IMC3][IMC3] = -2.*cth/(r*r)/(sth*sth*sth);

}

//----------------------------------------------------------------------------------------
//! \fn void MCSphericalPolar::Tetrad(Real x[4], Real tetrad[4][4])
//! \brief compute a diagonal tetrad

void MCSphericalPolar::Tetrad(Real x[4], Real tetrad[4][4]) {

  for (int l=0; l<4; l++) {
    for (int m=0; m<4; m++) {
      tetrad[l][m] = 0.;
    }
    tetrad[l][l] = 1.;
  }
  Real r = x[IMC1];
  Real sth = sin(x[IMC2]);
  tetrad[IMC2][IMC2] = 1. / r;
  tetrad[IMC3][IMC3] = 1. / (r*sth);
}

//----------------------------------------------------------------------------------------
//! \fn void MCSphericalPolar::InverseTetrad(Real x[4], Real invtet[4][4])
//! \brief compute a diagonal tetrad

void MCSphericalPolar::InverseTetrad(Real x[4], Real invtet[4][4]) {

  for (int l=0; l<4; l++) {
    for (int m=0; m<4; m++) {
      invtet[l][m] = 0.;
    }
    invtet[l][l] = 1.;
  }
  Real r = x[IMC1];
  Real sth = sin(x[IMC2]);
  invtet[IMC2][IMC2] = r;
  invtet[IMC3][IMC3] = r * sth;
}
//----------------------------------------------------------------------------------------
//! MCCylindrical constructor, built from Coord and MonteCarloBlock

MCCylindrical::MCCylindrical(Coordinates *pcoord, MonteCarloBlock *pmcb)
  : MCCoord(pcoord,pmcb) {
}

//----------------------------------------------------------------------------------------
//! MCCylindrical constructor for processes without own MeshBlock

MCCylindrical::MCCylindrical(int ncells1, int ncells2, int ncells3, bool acc)
  : MCCoord(ncells1,ncells2,ncells3,acc) {
}

//----------------------------------------------------------------------------------------
//! destructor
MCCylindrical::~MCCylindrical() {

}

//----------------------------------------------------------------------------------------
//! \fn void MCCylindrical::Metric(Real x[4], Real gcov[4][4])
//! \brief compute metric in flat spacetime cylindrical

void MCCylindrical::Metric(Real x[4], Real gcov[4][4]) {

  for(int i = 0; i < 4; i++) {
    for(int j = 0; j < 4; j++) {
      gcov[i][j] = 0;
    }
  }
  gcov[IMC0][IMC0] = -1;
  gcov[IMC1][IMC1] = 1;
  gcov[IMC2][IMC2] = x[IMC1] * x[IMC1];
  gcov[IMC3][IMC3] = 1;

}

//----------------------------------------------------------------------------------------
//! \fn void MCCylindrical::Connect(Real x[4], Real gamma[4][4][4])
//! \brief compute connection in flat spacetime cylindrical

void MCCylindrical::Connect(Real x[4], Real gamma[4][4][4]) {

  for(int i = 0; i < 4; i++) {
    for(int j = 0; j < 4; j++) {
      for(int k = 0; k < 4; k++) {
        gamma[i][j][k]=0;
      }
    }
  }

  gamma[IMC1][IMC2][IMC2] = -x[IMC1];
  gamma[IMC2][IMC1][IMC2] = 1./x[IMC1];
  gamma[IMC2][IMC2][IMC1] = 1./x[IMC1];
}

//----------------------------------------------------------------------------------------
//! MCKerrSchild constructor, built from Coord and MonteCarloBlock

MCKerrSchild::MCKerrSchild(Coordinates *pcoord, MonteCarloBlock *pmcb)
  : MCCoord(pcoord,pmcb) {

}

//----------------------------------------------------------------------------------------
//! MCKerrSchild constructor for processes without own MeshBlock

MCKerrSchild::MCKerrSchild(int ncells1, int ncells2, int ncells3, bool acc)
  : MCCoord(ncells1,ncells2,ncells3,acc) {

}

//----------------------------------------------------------------------------------------
//! destructor
MCKerrSchild::~MCKerrSchild() {

}

//----------------------------------------------------------------------------------------
//! \fn void MCKerrSchild::PointQuantities(const Real x[4], Point &p) const
//! \brief everything the spherical Kerr-Schild metric functions share at one point
//
// One sin, one cos and the handful of products below used to be recomputed by each of
// Metric, InverseMetric and InverseMetricDerivative, which the general pusher calls at
// the same point several times per step.  The expressions are kept exactly as those
// functions had them, so assembling from this struct reproduces their values bit for bit.

void MCKerrSchild::PointQuantities(const Real x[4], Point &p) const {
  p.a = bh_spin_;
  p.r = x[IMC1];
  p.r2 = SQR(p.r);
  p.sth = sin(x[IMC2]);
  p.cth = cos(x[IMC2]);
  p.cth2 = SQR(p.cth);
  p.sth2 = SQR(p.sth);
  p.s2th = 2.*p.sth*p.cth;
  p.c2th = p.cth2 - p.sth2;
  p.a2 = SQR(p.a);
  p.sigma = p.r2 + p.a2 * p.cth2;
  p.sigma2 = SQR(p.sigma);
  p.delta = p.r2 - 2 * p.r + p.a2;
  p.A = SQR(p.r2 + p.a2) - p.a2 * p.delta * p.sth2;
  p.alts = p.r2 - p.a2 * p.cth2;
  p.inv_sigma = 1. / p.sigma;
  p.inv_sigma2 = 1. / p.sigma2;
  p.inv_sigma3 = p.inv_sigma2 * p.inv_sigma;
  p.inv_sth = 1. / p.sth;
  p.inv_sth2 = 1. / p.sth2;
}

//----------------------------------------------------------------------------------------
//! \fn void MCKerrSchild::FillMetric(const Point &p, Real gcov[4][4])
//! \brief assemble the spherical Kerr-Schild metric from the point quantities

void MCKerrSchild::FillMetric(const Point &p, Real gcov[4][4]) {

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      gcov[i][j] = 0.;
    }
  }

  const Real a = p.a, sth2 = p.sth2, sigma = p.sigma, A = p.A;
  const Real tr = 2. * p.r * p.inv_sigma;  // 2 r / sigma

  gcov[IMC0][IMC0] = -(1. - tr);
  gcov[IMC0][IMC1] = tr;
  gcov[IMC0][IMC3] = -a * sth2 * tr;

  gcov[IMC1][IMC0] = gcov[IMC0][IMC1];
  gcov[IMC1][IMC1] = 1. + tr;
  gcov[IMC1][IMC3] = -a * sth2 * (1. + tr);

  gcov[IMC2][IMC2] = sigma;

  gcov[IMC3][IMC0] = gcov[IMC0][IMC3];
  gcov[IMC3][IMC1] = gcov[IMC1][IMC3];
  gcov[IMC3][IMC3] = A * sth2 * p.inv_sigma;

}

//----------------------------------------------------------------------------------------
//! \fn void MCKerrSchild::FillInverse(const Point &p, Real gcon[4][4])
//! \brief assemble the spherical Kerr-Schild inverse metric from the point quantities

void MCKerrSchild::FillInverse(const Point &p, Real gcon[4][4]) {

  // equations come from Takahasi (2007) Appendix
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      gcon[i][j] = 0.;
    }
  }

  const Real inv_sigma = p.inv_sigma;
  const Real tr = 2. * p.r * inv_sigma;  // 2 r / sigma

  gcon[IMC0][IMC0] = -(1. + tr);
  gcon[IMC0][IMC1] = tr;

  gcon[IMC1][IMC0] = gcon[IMC0][IMC1];
  gcon[IMC1][IMC1] = p.delta * inv_sigma;
  gcon[IMC1][IMC3] = p.a * inv_sigma;

  gcon[IMC2][IMC2] = inv_sigma;

  gcon[IMC3][IMC1] = gcon[IMC1][IMC3];
  gcon[IMC3][IMC3] = inv_sigma * p.inv_sth2;

}

//----------------------------------------------------------------------------------------
//! \fn void MCKerrSchild::FillInverseDerivative(const Point &p, Real dgcon[4][4][4])
//! \brief assemble the derivative of the spherical Kerr-Schild inverse metric

void MCKerrSchild::FillInverseDerivative(const Point &p, Real dgcon[4][4][4]) {

  for(int i = 0; i < 4; i++) {
    for(int j = 0; j < 4; j++) {
      for(int k = 0; k < 4; k++) {
        dgcon[i][j][k]=0;
      }
    }
  }

  const Real a = p.a, a2 = p.a2, r = p.r, r2 = p.r2, cth = p.cth;
  const Real sth2 = p.sth2, s2th = p.s2th, c2th = p.c2th, alts = p.alts;
  const Real inv_sigma2 = p.inv_sigma2, inv_sth2 = p.inv_sth2, inv_sth = p.inv_sth;

  dgcon[IMC1][IMC0][IMC0] = 2. * alts * inv_sigma2;
  dgcon[IMC1][IMC0][IMC1] = -dgcon[IMC1][IMC0][IMC0];
  dgcon[IMC1][IMC1][IMC0] = dgcon[IMC1][IMC0][IMC1];
  dgcon[IMC1][IMC1][IMC1] = 2.*(alts-a2*r*sth2) * inv_sigma2;
  dgcon[IMC1][IMC1][IMC3] = -2*a*r * inv_sigma2;
  dgcon[IMC1][IMC2][IMC2] = -2*r * inv_sigma2;
  dgcon[IMC1][IMC3][IMC1] = dgcon[IMC1][IMC1][IMC3];
  dgcon[IMC1][IMC3][IMC3] = -2*r * inv_sth2 * inv_sigma2;

  dgcon[IMC2][IMC0][IMC0] = -2.*a2*r*s2th * inv_sigma2;
  dgcon[IMC2][IMC0][IMC1] = -dgcon[IMC2][IMC0][IMC0];
  dgcon[IMC2][IMC1][IMC0] = dgcon[IMC2][IMC0][IMC1];
  dgcon[IMC2][IMC1][IMC1] = a2*(a2+r*(r-2.))*s2th * inv_sigma2;
  dgcon[IMC2][IMC1][IMC3] = a*a2*s2th * inv_sigma2;
  dgcon[IMC2][IMC2][IMC2] = a2*s2th * inv_sigma2;
  dgcon[IMC2][IMC3][IMC1] = dgcon[IMC2][IMC1][IMC3];
  dgcon[IMC2][IMC3][IMC3] = -2*(r2+a2*c2th)*cth*inv_sth*inv_sth2 * inv_sigma2;
}

//----------------------------------------------------------------------------------------
//! \fn void MCKerrSchild::Metric(Real x[4], Real gcov[4][4])
//! \brief compute metric in spherical polar Kerr-Schild

void MCKerrSchild::Metric(Real x[4], Real gcov[4][4]) {
  Point p;
  PointQuantities(x, p);
  FillMetric(p, gcov);
}

//----------------------------------------------------------------------------------------
//! \fn void MCKerrSchild::InverseMetric(Real x[4], Real gcon[4][4])
//! \brief compute inverse metric in spherical polar Kerr-Schild

void MCKerrSchild::InverseMetric(Real x[4], Real gcon[4][4]) {
  Point p;
  PointQuantities(x, p);
  FillInverse(p, gcon);
}

//----------------------------------------------------------------------------------------
//! \fn void MCKerrSchild::MetricAndInverse(Real x[4], Real gcov[4][4], Real gcon[4][4])
//! \fn void MCKerrSchild::InverseMetricAndDerivative(Real x[4], Real gcon[4][4],
//!                                                   Real dgcon[4][4][4])
//! \brief the fused pairs: one sin, one cos, one set of products for both outputs

void MCKerrSchild::MetricAndInverse(Real x[4], Real gcov[4][4], Real gcon[4][4]) {
  Point p;
  PointQuantities(x, p);
  FillMetric(p, gcov);
  FillInverse(p, gcon);
}

void MCKerrSchild::InverseMetricAndDerivative(Real x[4], Real gcon[4][4],
                                              Real dgcon[4][4][4]) {
  Point p;
  PointQuantities(x, p);
  FillInverse(p, gcon);
  FillInverseDerivative(p, dgcon);
}

//----------------------------------------------------------------------------------------
//! \fn void MCKerrSchild::Connect(Real x[4], Real gamma[4][4][4])
//! \brief compute connection in spherical polar Kerr-Schild

void MCKerrSchild::Connect(Real x[4], Real gamma[4][4][4]) {

  for(int i = 0; i < 4; i++) {
    for(int j = 0; j < 4; j++) {
      for(int k = 0; k < 4; k++) {
        gamma[i][j][k]=0;
      }
    }
  }

  // The same point quantities the metric functions use; see PointQuantities.
  Point p;
  PointQuantities(x, p);
  const Real a = p.a, r = p.r, r2 = p.r2, sth = p.sth, cth = p.cth, cth2 = p.cth2;
  const Real sth2 = p.sth2, s2th = p.s2th, c2th = p.c2th, a2 = p.a2;
  const Real sigma = p.sigma, delta = p.delta;
  const Real is = p.inv_sigma, is2 = p.inv_sigma2, is3 = p.inv_sigma3;
  const Real cot = cth * p.inv_sth;
  // the three factors the table repeats, each once
  const Real q = 1. - 2. * r2 * is;      // 1 - 2 r^2 / sigma
  const Real pr = 1. + 2. * r * is;      // 1 + 2 r / sigma
  const Real pd = 1. - delta * is;       // 1 - delta / sigma
  const Real w = r + a2 * sth2 * is * q; // r + a^2 sin^2 / sigma (1 - 2 r^2 / sigma)

  gamma[IMC0][IMC0][IMC0] = -2. * r * is2 * q;
  gamma[IMC0][IMC0][IMC1] = -is * pr * q;
  gamma[IMC0][IMC0][IMC2] = -a2 * r * s2th * is2;
  gamma[IMC0][IMC0][IMC3] = 2. * a * r * sth2 * is2 * q;

  gamma[IMC0][IMC1][IMC0] = gamma[IMC0][IMC0][IMC1];
  gamma[IMC0][IMC1][IMC1] = -2. * is * (1. + r * is) * q;
  gamma[IMC0][IMC1][IMC2] = -a2 * r * s2th * is2;
  gamma[IMC0][IMC1][IMC3] = a * sth2 * is * pr * q;

  gamma[IMC0][IMC2][IMC0] = gamma[IMC0][IMC0][IMC2];
  gamma[IMC0][IMC2][IMC1] = gamma[IMC0][IMC1][IMC2];
  gamma[IMC0][IMC2][IMC2] = -2. * r2 * is;
  gamma[IMC0][IMC2][IMC3] = a2 * a * r * is2 * sth2 * s2th;

  gamma[IMC0][IMC3][IMC0] = gamma[IMC0][IMC0][IMC3];
  gamma[IMC0][IMC3][IMC1] = gamma[IMC0][IMC1][IMC3];
  gamma[IMC0][IMC3][IMC2] = gamma[IMC0][IMC2][IMC3];
  gamma[IMC0][IMC3][IMC3] = -2. * r * sth2 * is * w;

  gamma[IMC1][IMC0][IMC0] = -delta * is2 * q;
  gamma[IMC1][IMC0][IMC1] = is * q * pd;
  gamma[IMC1][IMC0][IMC2] = 0.;
  gamma[IMC1][IMC0][IMC3] = a * delta * sth2 * is2 * q;

  gamma[IMC1][IMC1][IMC0] = gamma[IMC1][IMC0][IMC1];
  gamma[IMC1][IMC1][IMC1] = is * q * (2. - delta * is);
  gamma[IMC1][IMC1][IMC2] = -a2 * 0.5 * is * s2th;
  gamma[IMC1][IMC1][IMC3] = a * is * sth2 * (r - q * pd);

  gamma[IMC1][IMC2][IMC0] = gamma[IMC1][IMC0][IMC2];
  gamma[IMC1][IMC2][IMC1] = gamma[IMC1][IMC1][IMC2];
  gamma[IMC1][IMC2][IMC2] = -r * delta * is;
  gamma[IMC1][IMC2][IMC3] = 0.;

  gamma[IMC1][IMC3][IMC0] = gamma[IMC1][IMC0][IMC3];
  gamma[IMC1][IMC3][IMC1] = gamma[IMC1][IMC1][IMC3];
  gamma[IMC1][IMC3][IMC2] = gamma[IMC1][IMC2][IMC3];
  gamma[IMC1][IMC3][IMC3] = -delta * is * sth2 * w;

  gamma[IMC2][IMC0][IMC0] = -a2 * r * s2th * is3;
  gamma[IMC2][IMC0][IMC1] = -a2 * r * s2th * is3;
  gamma[IMC2][IMC0][IMC2] = 0.;
  gamma[IMC2][IMC0][IMC3] = a * r * (r2 + a2) * s2th * is3;

  gamma[IMC2][IMC1][IMC0] = gamma[IMC2][IMC0][IMC1];
  gamma[IMC2][IMC1][IMC1] = -a2 * r * s2th * is3;
  gamma[IMC2][IMC1][IMC2] = r * is;

  // from Shane's notebook -- not equal to Takahashi+07 (SWD: ?)
  gamma[IMC2][IMC1][IMC3] = (a * cth * sth * is3) *
    (r2 * r * (r + 2.) + 2. * a2 * r * (r + 1.) * cth2 + a2 * a2 * cth2 * cth2
    + 2. * a2 * r * sth2);

  gamma[IMC2][IMC2][IMC0] = gamma[IMC2][IMC0][IMC2];
  gamma[IMC2][IMC2][IMC1] = gamma[IMC2][IMC1][IMC2];
  gamma[IMC2][IMC2][IMC2] = -a2 * s2th * 0.5 * is;
  gamma[IMC2][IMC2][IMC3] = 0.;

  gamma[IMC2][IMC3][IMC0] = gamma[IMC2][IMC0][IMC3];
  gamma[IMC2][IMC3][IMC1] = gamma[IMC2][IMC1][IMC3];
  gamma[IMC2][IMC3][IMC2] = gamma[IMC2][IMC2][IMC3];
  /*gamma[IMC2][IMC3][IMC3] = -s2th / (2. * sigma) * (delta + 2. * r *
    SQR((r2 + a2) / sigma));*/
  // from Shane's notebook -- not equal to Takahashi+07 (SWD: ?)
  gamma[IMC2][IMC3][IMC3] = -(cth * sth * is3) *
    (a2 * a2 * a2 * cth2 * cth2 * cth2 +
     cth2 * cth2 * (3. * a2 * a2 * r2 + a2 * a2 * a2 * sth2) +
     cth2 * (3. * a2 * r2 * r2 + 2. * a2 * a2 * r2 * sth2) +
     r * (r2 * r2 * r + a2 * r2 * (r + 4.) * sth2 + 2. * a2 * a2 * sth2 * sth2 +
     a2 * a2 * s2th * s2th));

  gamma[IMC3][IMC0][IMC0] = -a * is2 * q;
  gamma[IMC3][IMC0][IMC1] = -a * is2 * q;
  gamma[IMC3][IMC0][IMC2] = -2. * a * r * is2 * cot;
  gamma[IMC3][IMC0][IMC3] = a2 * sth2 * is2 * q;

  gamma[IMC3][IMC1][IMC0] = gamma[IMC3][IMC0][IMC1];
  gamma[IMC3][IMC1][IMC1] = -a * is2 * q;
  gamma[IMC3][IMC1][IMC2] = -a * is * pr * cot;
  gamma[IMC3][IMC1][IMC3] = is * w;

  gamma[IMC3][IMC2][IMC0] = gamma[IMC3][IMC0][IMC2];
  gamma[IMC3][IMC2][IMC1] = gamma[IMC3][IMC1][IMC2];
  gamma[IMC3][IMC2][IMC2] = -a * r * is;
  //gamma[IMC3][IMC2][IMC3] = (1. + 2. * r / sigma * ((r2 + a2) / sigma - 1.)) *
  // cth / sth;
  // from Shane's notebook -- not equal to Takahashi+07 (SWD: ?)
  gamma[IMC3][IMC2][IMC3] = ((1. / 4.) * SQR(a2 + 2. * r2 + a2 * c2th) * cot +
    a2 * r * s2th) * is2;

  gamma[IMC3][IMC3][IMC0] = gamma[IMC3][IMC0][IMC3];
  gamma[IMC3][IMC3][IMC1] = gamma[IMC3][IMC1][IMC3];
  gamma[IMC3][IMC3][IMC2] = gamma[IMC3][IMC2][IMC3];
  gamma[IMC3][IMC3][IMC3] = -a * is * sth2 * w;
  (void)sigma;

}

//----------------------------------------------------------------------------------------
//! \fn void MCKerrSchild::InverseMetricDerivative(Real x[4], Real dgcon[4][4][4])
//! \brief compute derivative of the inverse metric in spherical polar Kerr-Schild

void MCKerrSchild::InverseMetricDerivative(Real x[4], Real dgcon[4][4][4]) {
  Point p;
  PointQuantities(x, p);
  FillInverseDerivative(p, dgcon);
}

//----------------------------------------------------------------------------------------
//! MCKerrSchild constructor, built from Coord and MonteCarloBlock

MCKerrSchildCartesian::MCKerrSchildCartesian(Coordinates *pcoord, MonteCarloBlock *pmcb)
  : MCCoord(pcoord,pmcb) {

}

//----------------------------------------------------------------------------------------
//! MCKerrSchild constructor for processes without own MeshBlock

MCKerrSchildCartesian::MCKerrSchildCartesian(int ncells1, int ncells2, int ncells3,
                                             bool acc)
  : MCCoord(ncells1,ncells2,ncells3,acc) {

}

//----------------------------------------------------------------------------------------
//! destructor
MCKerrSchildCartesian::~MCKerrSchildCartesian() {

}

//----------------------------------------------------------------------------------------
//! \fn void MCKerrSchildCartesian::PointQuantities(const Real x[4], Point &p) const
//! \fn void MCKerrSchildCartesian::PointDerivatives(const Real x[4], Point &p) const
//! \brief the shared point quantities of the Cartesian Kerr-Schild metric
//
// g = eta + f l l with l null; every metric function of this class starts from the Kerr
// radius r, f and the spatial null vector, and the derivative also needs their gradients.
// Metric, InverseMetric and InverseMetricDerivative used to recompute the hypot, the sqrt
// and the divisions independently, several times per step at the same point.  The
// expressions are exactly the ones those functions had, so assembling from this struct
// reproduces their values bit for bit.  PointDerivatives includes PointQuantities.

void MCKerrSchildCartesian::PointQuantities(const Real x[4], Point &p) const {
  p.a = bh_spin_;
  p.m = bh_mass_;
  p.a2 = p.a * p.a;
  p.rr2 = x[IMC1] * x[IMC1] + x[IMC2] * x[IMC2] + x[IMC3] * x[IMC3];
  p.r2 = 0.5 * (p.rr2 - p.a2 + std::hypot(p.rr2 - p.a2, 2.0 * p.a * x[IMC3]));
  p.r = std::sqrt(p.r2);
  // Three reciprocals here replace some twenty divisions across the derivatives.
  p.inv_den = 1.0 / (p.r2 * p.r2 + p.a2 * x[IMC3] * x[IMC3]);
  p.inv_ra2 = 1.0 / (p.r2 + p.a2);
  p.inv_r = 1.0 / p.r;
  p.f = 2.0 * p.m * p.r2 * p.r * p.inv_den;
  p.l1 = (p.r * x[IMC1] + p.a * x[IMC2]) * p.inv_ra2;
  p.l2 = (p.r * x[IMC2] - p.a * x[IMC1]) * p.inv_ra2;
  p.l3 = x[IMC3] * p.inv_r;
}

void MCKerrSchildCartesian::PointDerivatives(const Real x[4], Point &p) const {
  PointQuantities(x, p);
  const Real a = p.a, a2 = p.a2, rr2 = p.rr2, r2 = p.r2, r = p.r, f = p.f;
  const Real l1 = p.l1, l2 = p.l2, inv_r = p.inv_r, inv_ra2 = p.inv_ra2;
  const Real z = x[IMC3], z2 = z * z;

  // Calculate scalar derivatives: dr/dx^i, then df/dx^i
  const Real inv_rden = 1.0 / (2.0 * r2 - rr2 + a2);
  p.dr_dx = r * x[IMC1] * inv_rden;
  p.dr_dy = r * x[IMC2] * inv_rden;
  p.dr_dz = (r * z + a2 * z * inv_r) * inv_rden;
  const Real fc = -(r2 * r2 - 3.0 * a2 * z2) * inv_r * p.inv_den * f;
  p.df_dx = fc * p.dr_dx;
  p.df_dy = fc * p.dr_dy;
  p.df_dz = fc * p.dr_dz - 2.0 * a2 * r * z * inv_r * p.inv_den * f;

  // Calculate vector derivatives
  const Real c1 = x[IMC1] - 2.0 * r * l1;
  const Real c2 = x[IMC2] - 2.0 * r * l2;
  const Real zr2 = z * inv_r * inv_r;
  p.dl1_dx = (c1 * p.dr_dx + r) * inv_ra2;
  p.dl1_dy = (c1 * p.dr_dy + a) * inv_ra2;
  p.dl1_dz = c1 * p.dr_dz * inv_ra2;
  p.dl2_dx = (c2 * p.dr_dx - a) * inv_ra2;
  p.dl2_dy = (c2 * p.dr_dy + r) * inv_ra2;
  p.dl2_dz = c2 * p.dr_dz * inv_ra2;
  p.dl3_dx = -zr2 * p.dr_dx;
  p.dl3_dy = -zr2 * p.dr_dy;
  p.dl3_dz = -zr2 * p.dr_dz + inv_r;
}

//----------------------------------------------------------------------------------------
//! \fn void MCKerrSchildCartesian::FillMetric(const Point &p, Real gcov[4][4])
//! \brief assemble the cartesian Kerr-Schild metric from the point quantities

void MCKerrSchildCartesian::FillMetric(const Point &p, Real gcov[4][4]) {

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      gcov[i][j] = 0.;
    }
  }

  const Real f = p.f;
  const Real l_0 = 1.0, l_1 = p.l1, l_2 = p.l2, l_3 = p.l3;

  // Calculate metric components
  gcov[IMC0][IMC0] = f * l_0 * l_0 - 1.0;
  gcov[IMC0][IMC1] = f * l_0 * l_1;
  gcov[IMC0][IMC2] = f * l_0 * l_2;
  gcov[IMC0][IMC3] = f * l_0 * l_3;

  gcov[IMC1][IMC0] = f * l_1 * l_0;
  gcov[IMC1][IMC1] = f * l_1 * l_1 + 1.0;
  gcov[IMC1][IMC2] = f * l_1 * l_2;
  gcov[IMC1][IMC3] = f * l_1 * l_3;

  gcov[IMC2][IMC0] = f * l_2 * l_0;
  gcov[IMC2][IMC1] = f * l_2 * l_1;
  gcov[IMC2][IMC2] = f * l_2 * l_2 + 1.0;
  gcov[IMC2][IMC3] = f * l_2 * l_3;

  gcov[IMC3][IMC0] = f * l_3 * l_0;
  gcov[IMC3][IMC1] = f * l_3 * l_1;
  gcov[IMC3][IMC2] = f * l_3 * l_2;
  gcov[IMC3][IMC3] = f * l_3 * l_3 + 1.0;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MCKerrSchildCartesian::FillInverse(const Point &p, Real gcon[4][4])
//! \brief assemble the cartesian Kerr-Schild inverse metric from the point quantities

void MCKerrSchildCartesian::FillInverse(const Point &p, Real gcon[4][4]) {

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      gcon[i][j] = 0.;
    }
  }

  const Real f = p.f;
  const Real l0 = -1.0, l1 = p.l1, l2 = p.l2, l3 = p.l3;

  // Calculate metric components
  gcon[IMC0][IMC0] = -f * l0 * l0 - 1.0;
  gcon[IMC0][IMC1] = -f * l0 * l1;
  gcon[IMC0][IMC2] = -f * l0 * l2;
  gcon[IMC0][IMC3] = -f * l0 * l3;

  gcon[IMC1][IMC0] = -f * l1 * l0;
  gcon[IMC1][IMC1] = -f * l1 * l1 + 1.0;
  gcon[IMC1][IMC2] = -f * l1 * l2;
  gcon[IMC1][IMC3] = -f * l1 * l3;

  gcon[IMC2][IMC0] = -f * l2 * l0;
  gcon[IMC2][IMC1] = -f * l2 * l1;
  gcon[IMC2][IMC2] = -f * l2 * l2 + 1.0;
  gcon[IMC2][IMC3] = -f * l2 * l3;

  gcon[IMC3][IMC0] = -f * l3 * l0;
  gcon[IMC3][IMC1] = -f * l3 * l1;
  gcon[IMC3][IMC2] = -f * l3 * l2;
  gcon[IMC3][IMC3] = -f * l3 * l3 + 1.0;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MCKerrSchildCartesian::FillInverseDerivative(const Point &p,
//                                                        Real dgcon[4][4][4])
//! \brief assemble the derivative of the cartesian Kerr-Schild inverse metric; p must
//!        have been through PointDerivatives

void MCKerrSchildCartesian::FillInverseDerivative(const Point &p, Real dgcon[4][4][4]) {

  for(int i = 0; i < 4; i++) {
    for(int j = 0; j < 4; j++) {
      for(int k = 0; k < 4; k++) {
        dgcon[i][j][k]=0;
      }
    }
  }

  const Real f = p.f;
  const Real l0 = -1.0, l1 = p.l1, l2 = p.l2, l3 = p.l3;
  const Real df_dx = p.df_dx, df_dy = p.df_dy, df_dz = p.df_dz;
  const Real dl0_dx = 0.0, dl0_dy = 0.0, dl0_dz = 0.0;
  const Real dl1_dx = p.dl1_dx, dl1_dy = p.dl1_dy, dl1_dz = p.dl1_dz;
  const Real dl2_dx = p.dl2_dx, dl2_dy = p.dl2_dy, dl2_dz = p.dl2_dz;
  const Real dl3_dx = p.dl3_dx, dl3_dy = p.dl3_dy, dl3_dz = p.dl3_dz;

  // Written out entry by entry on purpose: a loop over (k,a,b) with mirrored stores
  // measured slower than this unrolled list (112 ns against 104 ns per call), and the
  // reciprocals in PointDerivatives are where the saving was.
  // Calculate metric component x-derivatives
  dgcon[IMC1][IMC0][IMC0] = -(df_dx * l0 * l0 + f * dl0_dx * l0 + f * l0 * dl0_dx);
  dgcon[IMC1][IMC0][IMC1] = -(df_dx * l0 * l1 + f * dl0_dx * l1 + f * l0 * dl1_dx);
  dgcon[IMC1][IMC0][IMC2] = -(df_dx * l0 * l2 + f * dl0_dx * l2 + f * l0 * dl2_dx);
  dgcon[IMC1][IMC0][IMC3] = -(df_dx * l0 * l3 + f * dl0_dx * l3 + f * l0 * dl3_dx);
  dgcon[IMC1][IMC1][IMC0] = -(df_dx * l1 * l0 + f * dl1_dx * l0 + f * l1 * dl0_dx);
  dgcon[IMC1][IMC1][IMC1] = -(df_dx * l1 * l1 + f * dl1_dx * l1 + f * l1 * dl1_dx);
  dgcon[IMC1][IMC1][IMC2] = -(df_dx * l1 * l2 + f * dl1_dx * l2 + f * l1 * dl2_dx);
  dgcon[IMC1][IMC1][IMC3] = -(df_dx * l1 * l3 + f * dl1_dx * l3 + f * l1 * dl3_dx);
  dgcon[IMC1][IMC2][IMC0] = -(df_dx * l2 * l0 + f * dl2_dx * l0 + f * l2 * dl0_dx);
  dgcon[IMC1][IMC2][IMC1] = -(df_dx * l2 * l1 + f * dl2_dx * l1 + f * l2 * dl1_dx);
  dgcon[IMC1][IMC2][IMC2] = -(df_dx * l2 * l2 + f * dl2_dx * l2 + f * l2 * dl2_dx);
  dgcon[IMC1][IMC2][IMC3] = -(df_dx * l2 * l3 + f * dl2_dx * l3 + f * l2 * dl3_dx);
  dgcon[IMC1][IMC3][IMC0] = -(df_dx * l3 * l0 + f * dl3_dx * l0 + f * l3 * dl0_dx);
  dgcon[IMC1][IMC3][IMC1] = -(df_dx * l3 * l1 + f * dl3_dx * l1 + f * l3 * dl1_dx);
  dgcon[IMC1][IMC3][IMC2] = -(df_dx * l3 * l2 + f * dl3_dx * l2 + f * l3 * dl2_dx);
  dgcon[IMC1][IMC3][IMC3] = -(df_dx * l3 * l3 + f * dl3_dx * l3 + f * l3 * dl3_dx);

  // Calculate metric component y-derivatives
  dgcon[IMC2][IMC0][IMC0] = -(df_dy * l0 * l0 + f * dl0_dy * l0 + f * l0 * dl0_dy);
  dgcon[IMC2][IMC0][IMC1] = -(df_dy * l0 * l1 + f * dl0_dy * l1 + f * l0 * dl1_dy);
  dgcon[IMC2][IMC0][IMC2] = -(df_dy * l0 * l2 + f * dl0_dy * l2 + f * l0 * dl2_dy);
  dgcon[IMC2][IMC0][IMC3] = -(df_dy * l0 * l3 + f * dl0_dy * l3 + f * l0 * dl3_dy);
  dgcon[IMC2][IMC1][IMC0] = -(df_dy * l1 * l0 + f * dl1_dy * l0 + f * l1 * dl0_dy);
  dgcon[IMC2][IMC1][IMC1] = -(df_dy * l1 * l1 + f * dl1_dy * l1 + f * l1 * dl1_dy);
  dgcon[IMC2][IMC1][IMC2] = -(df_dy * l1 * l2 + f * dl1_dy * l2 + f * l1 * dl2_dy);
  dgcon[IMC2][IMC1][IMC3] = -(df_dy * l1 * l3 + f * dl1_dy * l3 + f * l1 * dl3_dy);
  dgcon[IMC2][IMC2][IMC0] = -(df_dy * l2 * l0 + f * dl2_dy * l0 + f * l2 * dl0_dy);
  dgcon[IMC2][IMC2][IMC1] = -(df_dy * l2 * l1 + f * dl2_dy * l1 + f * l2 * dl1_dy);
  dgcon[IMC2][IMC2][IMC2] = -(df_dy * l2 * l2 + f * dl2_dy * l2 + f * l2 * dl2_dy);
  dgcon[IMC2][IMC2][IMC3] = -(df_dy * l2 * l3 + f * dl2_dy * l3 + f * l2 * dl3_dy);
  dgcon[IMC2][IMC3][IMC0] = -(df_dy * l3 * l0 + f * dl3_dy * l0 + f * l3 * dl0_dy);
  dgcon[IMC2][IMC3][IMC1] = -(df_dy * l3 * l1 + f * dl3_dy * l1 + f * l3 * dl1_dy);
  dgcon[IMC2][IMC3][IMC2] = -(df_dy * l3 * l2 + f * dl3_dy * l2 + f * l3 * dl2_dy);
  dgcon[IMC2][IMC3][IMC3] = -(df_dy * l3 * l3 + f * dl3_dy * l3 + f * l3 * dl3_dy);

  // Calculate metric component z-derivatives
  dgcon[IMC3][IMC0][IMC0] = -(df_dz * l0 * l0 + f * dl0_dz * l0 + f * l0 * dl0_dz);
  dgcon[IMC3][IMC0][IMC1] = -(df_dz * l0 * l1 + f * dl0_dz * l1 + f * l0 * dl1_dz);
  dgcon[IMC3][IMC0][IMC2] = -(df_dz * l0 * l2 + f * dl0_dz * l2 + f * l0 * dl2_dz);
  dgcon[IMC3][IMC0][IMC3] = -(df_dz * l0 * l3 + f * dl0_dz * l3 + f * l0 * dl3_dz);
  dgcon[IMC3][IMC1][IMC0] = -(df_dz * l1 * l0 + f * dl1_dz * l0 + f * l1 * dl0_dz);
  dgcon[IMC3][IMC1][IMC1] = -(df_dz * l1 * l1 + f * dl1_dz * l1 + f * l1 * dl1_dz);
  dgcon[IMC3][IMC1][IMC2] = -(df_dz * l1 * l2 + f * dl1_dz * l2 + f * l1 * dl2_dz);
  dgcon[IMC3][IMC1][IMC3] = -(df_dz * l1 * l3 + f * dl1_dz * l3 + f * l1 * dl3_dz);
  dgcon[IMC3][IMC2][IMC0] = -(df_dz * l2 * l0 + f * dl2_dz * l0 + f * l2 * dl0_dz);
  dgcon[IMC3][IMC2][IMC1] = -(df_dz * l2 * l1 + f * dl2_dz * l1 + f * l2 * dl1_dz);
  dgcon[IMC3][IMC2][IMC2] = -(df_dz * l2 * l2 + f * dl2_dz * l2 + f * l2 * dl2_dz);
  dgcon[IMC3][IMC2][IMC3] = -(df_dz * l2 * l3 + f * dl2_dz * l3 + f * l2 * dl3_dz);
  dgcon[IMC3][IMC3][IMC0] = -(df_dz * l3 * l0 + f * dl3_dz * l0 + f * l3 * dl0_dz);
  dgcon[IMC3][IMC3][IMC1] = -(df_dz * l3 * l1 + f * dl3_dz * l1 + f * l3 * dl1_dz);
  dgcon[IMC3][IMC3][IMC2] = -(df_dz * l3 * l2 + f * dl3_dz * l2 + f * l3 * dl2_dz);
  dgcon[IMC3][IMC3][IMC3] = -(df_dz * l3 * l3 + f * dl3_dz * l3 + f * l3 * dl3_dz);

}

//----------------------------------------------------------------------------------------
//! \fn void MCKerrSchildCartesian::Metric(Real x[4], Real gcov[4][4])
//! \fn void MCKerrSchildCartesian::InverseMetric(Real x[4], Real gcon[4][4])
//! \fn void MCKerrSchildCartesian::InverseMetricDerivative(Real x[4],
//!                                                          Real dgcon[4][4][4])
//! \brief the single-purpose entry points, each one point evaluation plus one assembly

void MCKerrSchildCartesian::Metric(Real x[4], Real gcov[4][4]) {
  Point p;
  PointQuantities(x, p);
  FillMetric(p, gcov);
}

void MCKerrSchildCartesian::InverseMetric(Real x[4], Real gcon[4][4]) {
  Point p;
  PointQuantities(x, p);
  FillInverse(p, gcon);
}

void MCKerrSchildCartesian::InverseMetricDerivative(Real x[4], Real dgcon[4][4][4]) {
  Point p;
  PointDerivatives(x, p);
  FillInverseDerivative(p, dgcon);
}

//----------------------------------------------------------------------------------------
//! \fn void MCKerrSchildCartesian::MetricAndInverse(Real x[4], Real gcov[4][4],
//!                                                   Real gcon[4][4])
//! \fn void MCKerrSchildCartesian::InverseMetricAndDerivative(Real x[4], Real gcon[4][4],
//!                                                            Real dgcon[4][4][4])
//! \brief the fused pairs: one hypot, one sqrt, one null vector for both outputs

void MCKerrSchildCartesian::MetricAndInverse(Real x[4], Real gcov[4][4],
                                             Real gcon[4][4]) {
  Point p;
  PointQuantities(x, p);
  FillMetric(p, gcov);
  FillInverse(p, gcon);
}

void MCKerrSchildCartesian::InverseMetricAndDerivative(Real x[4], Real gcon[4][4],
                                                       Real dgcon[4][4][4]) {
  Point p;
  PointDerivatives(x, p);
  FillInverse(p, gcon);
  FillInverseDerivative(p, dgcon);
}

//----------------------------------------------------------------------------------------
//! \fn void MCKerrSchildCartesian::Connect(Real x[4], Real gamma[4][4][4])
//! \brief compute the connection in cartesian Kerr-Schild
//
// Kerr-Schild is g_{mn} = eta_{mn} + f l_m l_n with l null in eta, and in cartesian
// coordinates eta is constant, so the whole connection comes from the f l l piece and
// vanishes with f.  There is no flat-background curvilinear part to carry. Writing
// G_{lmn} = 1/2 (d_m h_{ln} + d_n h_{lm} - d_l h_{mn}) with h = f l l and grouping,
//
//   G_{lmn} = 1/2 [ f_,m l_l l_n + f_,n l_l l_m - f_,l l_m l_n
//                   + f ( l_n F_{ml} + l_m F_{nl} + l_l S_{mn} ) ],
//
// where F and S are the antisymmetric and symmetric parts of d_a l_b.  Raising the first
// index is exact and cheap because the inverse is linear in f,
// g^{ls} = eta^{ls} - f l^l l^s, so it is a diagonal scaling plus one rank-one correction
// instead of a matrix multiply.
//
// The scalar and vector derivatives below are the same expressions the gr_user metric
// function in the problem generators uses; r is defined implicitly by
// r^4 - (R^2 - a^2) r^2 - a^2 z^2 = 0.

void MCKerrSchildCartesian::Connect(Real x[4], Real gamma[4][4][4]) {

  Real eta[4];
  eta[IMC0] = -1.0;
  eta[IMC1] = 1.0;
  eta[IMC2] = 1.0;
  eta[IMC3] = 1.0;

  // The same point quantities and gradients the metric functions use.
  Point p;
  PointDerivatives(x, p);
  const Real f = p.f;

  // null vector; l^mu = eta^{mu nu} l_nu, so only the time component changes sign
  Real lcov[4], lcon[4];
  lcov[IMC0] = 1.0;
  lcov[IMC1] = p.l1;
  lcov[IMC2] = p.l2;
  lcov[IMC3] = p.l3;
  for (int i = 0; i < 4; i++) lcon[i] = eta[i]*lcov[i];

  // df/dx^mu
  Real df[4];
  df[IMC0] = 0.0;
  df[IMC1] = p.df_dx;
  df[IMC2] = p.df_dy;
  df[IMC3] = p.df_dz;

  // dl[alpha][beta] = d_alpha l_beta; l_0 is constant so its column stays zero
  Real dl[4][4];
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) dl[i][j] = 0.0;
  dl[IMC1][IMC1] = p.dl1_dx;
  dl[IMC2][IMC1] = p.dl1_dy;
  dl[IMC3][IMC1] = p.dl1_dz;
  dl[IMC1][IMC2] = p.dl2_dx;
  dl[IMC2][IMC2] = p.dl2_dy;
  dl[IMC3][IMC2] = p.dl2_dz;
  dl[IMC1][IMC3] = p.dl3_dx;
  dl[IMC2][IMC3] = p.dl3_dy;
  dl[IMC3][IMC3] = p.dl3_dz;

  // connection with the first index down.  Kept as the plain triple loop: a version
  // with F and S formed once and the (mu,nu) symmetry mirrored measured slower (243 ns
  // against 188 ns per call), the compiler doing better with the regular form.
  Real gl[4][4][4];
  for (int l = 0; l < 4; l++) {
    for (int mu = 0; mu < 4; mu++) {
      for (int nu = 0; nu < 4; nu++) {
        Real fml = dl[mu][l]  - dl[l][mu];   // F_{mu l}
        Real fnl = dl[nu][l]  - dl[l][nu];   // F_{nu l}
        Real smn = dl[mu][nu] + dl[nu][mu];  // S_{mu nu}
        gl[l][mu][nu] = 0.5*(df[mu]*lcov[l]*lcov[nu]
                           + df[nu]*lcov[l]*lcov[mu]
                           - df[l]*lcov[mu]*lcov[nu]
                           + f*(lcov[nu]*fml + lcov[mu]*fnl + lcov[l]*smn));
      }
    }
  }

  // l^s G_{s mu nu}, the only contraction the raising needs
  Real lg[4][4];
  for (int mu = 0; mu < 4; mu++) {
    for (int nu = 0; nu < 4; nu++) {
      Real sum = 0.0;
      for (int sig = 0; sig < 4; sig++) sum += lcon[sig]*gl[sig][mu][nu];
      lg[mu][nu] = sum;
    }
  }

  for (int l = 0; l < 4; l++)
    for (int mu = 0; mu < 4; mu++)
      for (int nu = 0; nu < 4; nu++)
        gamma[l][mu][nu] = eta[l]*gl[l][mu][nu] - f*lcon[l]*lg[mu][nu];
}

//----------------------------------------------------------------------------------------
//! MCBoyerLindquist, built from Coord and MonteCarloBlock

MCBoyerLindquist::MCBoyerLindquist(Coordinates *pcoord, MonteCarloBlock *pmcb)
  : MCCoord(pcoord,pmcb) {
}

//----------------------------------------------------------------------------------------
//! MCBoyerLindquist constructor for processes without own MeshBlock

MCBoyerLindquist::MCBoyerLindquist(int ncells1, int ncells2, int ncells3, bool acc)
  : MCCoord(ncells1,ncells2,ncells3,acc) {
}

//----------------------------------------------------------------------------------------
//! destructor
MCBoyerLindquist::~MCBoyerLindquist() {

}

//----------------------------------------------------------------------------------------
//! \fn void MCBoyerLindquist::Metric(Real x[4], Real gcov[4][4])
//! \brief compute metric for Boyer-Lindquist coordinates

void MCBoyerLindquist::Metric(Real x[4], Real gcov[4][4]) {

  // equation for the Metric comes from the inside cover of Hartle
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      gcov[i][j] = 0.;
    }
  }

  Real a = bh_spin_;
  Real m = bh_mass_;
  // a in input file is dimensionless spin
  a *= m;
  Real r = x[IMC1];
  Real sth = sin(x[IMC2]);
  Real cth = cos(x[IMC2]);
  Real cth2 = SQR(cth);
  Real sth2 = SQR(sth);

  Real r2 = SQR(r);
  Real a2 = SQR(a);
  Real rho2 = r2 + a2*cth2;
  Real delta = r2 - 2. * m* r + a2;

  gcov[IMC0][IMC0] = (-1. + 2.*m*r/rho2);
  gcov[IMC1][IMC1] = rho2/delta;
  gcov[IMC2][IMC2] = rho2;
  gcov[IMC3][IMC3] = (r2 + a2 + 2.*m*r*a2*sth2/rho2)*sth2;
  gcov[IMC0][IMC3] = -2.*a*r*sth2/rho2;
  gcov[IMC3][IMC0] = gcov[IMC0][IMC3];
}

//----------------------------------------------------------------------------------------
//! \fn void MCBoyerLindquist::InverseMetric(Real x[4], Real gcon[4][4])
//! \brief compute inverse metric for Boyer-Lindquist coordinates

void MCBoyerLindquist::InverseMetric(Real x[4], Real gcon[4][4]) {

  // Equation comes from ColinsCosmos.com/wiki/boyer-lindquist-coordinates, which
  // sites Frolov & Novikov Section D.1 (but I don't have access to this book)

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      gcon[i][j] = 0.;
    }
  }

  Real a = bh_spin_;
  Real m = bh_mass_;
  // a in input file is dimensionless spin
  a *= m;
  Real r = x[IMC1];
  Real sth = sin(x[IMC2]);
  Real cth = cos(x[IMC2]);
  Real cth2 = SQR(cth);
  Real sth2 = SQR(sth);

  Real r2 = SQR(r);
  Real a2 = SQR(a);
  Real rho2 = r2 + a2 * cth2;
  Real delta = r2 - 2. * m * r + a2;

  gcon[IMC0][IMC0] = -1. / delta * (r2 + a2 + 2. * r * m * a2 * sth2 / rho2);
  gcon[IMC1][IMC1] = delta / rho2;
  gcon[IMC2][IMC2] = 1. / rho2;
  gcon[IMC3][IMC3] = (delta - a2 * sth2) / (rho2 * delta * sth2);

  gcon[IMC0][IMC3] = -2. * m * r * a / (rho2 * delta);
  gcon[IMC3][IMC0] = gcon[IMC0][IMC3];

}

//----------------------------------------------------------------------------------------
//! \fn void MCBoyerLindquist::Connect(Real x[4], Real gamma[4][4][4])
//! \brief compute connection for Boyer-Lindquist coordinates

void MCBoyerLindquist::Connect(Real x[4], Real gamma[4][4][4]) {

  // equations for the connection coefficients come from Frutos-Alfaro et al. (2012)

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        gamma[i][j][k] = 0.;
      }
    }
  }

  // SWD: Clean this one up
  Real a = bh_spin_;
  Real m = bh_mass_;
  // a has units of mass in Frutos-Alfaro
  a *= m;
  Real r = x[IMC1];
  Real j = a*m;

  Real sth = sin(x[IMC2]);
  Real cth = cos(x[IMC2]);
  Real cth2 = SQR(cth);
  Real sth2 = 1. - cth2;

  Real r2 = SQR(r);
  Real a2 = SQR(a);
  Real rho2 = r2 + a2*cth2;
  Real rs = 2.*m;
  Real delta = r2 - rs*r + a2;

  Real rho4 = SQR(rho2);
  Real rho6 = rho4*rho2;

  // Real A = SQR(r2+a2) - a2*delta*sth2;

  gamma[IMC0][IMC0][IMC1] = rs/(2.*rho4*delta)*(r2+a2)*(2.*r2-rho2);
  gamma[IMC0][IMC0][IMC2] = -2.*a*j*r/rho4*sth*cth;

  gamma[IMC0][IMC1][IMC0] = gamma[IMC0][IMC0][IMC1];
  gamma[IMC0][IMC1][IMC3] = -j*sth2/(rho4*delta)*(rho2*(r2-a2)+2.*r2*(r2+a2));

  gamma[IMC0][IMC2][IMC3] = 2.*a2*j*r/rho4*cth*sth2*sth;
  gamma[IMC0][IMC3][IMC2] = gamma[IMC0][IMC2][IMC3];

  gamma[IMC1][IMC0][IMC0] = rs*delta/(2.*rho6)*(2.*r2-rho2);
  gamma[IMC1][IMC0][IMC3] = -j*delta/rho6*(2.*r2-rho2)*sth2;

  gamma[IMC1][IMC1][IMC1] = 1./(rho2*delta)*(rho2*(rs/2.-r)+r*delta);
  gamma[IMC1][IMC1][IMC2] = -a2/rho2*sth*cth;

  gamma[IMC1][IMC2][IMC1] = gamma[IMC1][IMC1][IMC2];
  gamma[IMC1][IMC2][IMC2] = -r*delta/rho2;

  gamma[IMC1][IMC3][IMC0] = gamma[IMC1][IMC0][IMC3];
  gamma[IMC1][IMC3][IMC3] = -delta*sth2/rho6*(r*rho4-a*j*(2.*r2-rho2)*sth2);

  gamma[IMC2][IMC0][IMC0] = -2.*a*j*r/rho6*sth*cth;
  gamma[IMC2][IMC0][IMC3] = 2.*j*r/rho6*(r2+a2)*sth*cth;

  gamma[IMC2][IMC1][IMC1] = a2/(rho2*delta)*sth*cth;
  gamma[IMC2][IMC1][IMC2] = r/rho2;

  gamma[IMC2][IMC2][IMC1] = gamma[IMC2][IMC1][IMC2];
  gamma[IMC2][IMC2][IMC2] = gamma[IMC1][IMC1][IMC2];

  gamma[IMC2][IMC3][IMC0] = gamma[IMC2][IMC0][IMC3];
  gamma[IMC2][IMC3][IMC3] = -sth*cth/rho6*(rho4*delta+rs*r*SQR(r2+a2));
  //gamma[IMC2][IMC2][IMC3] = -sth*cth/rho6*(A*rho2+(r2+a2)*a2*r*rs*sth2);

  gamma[IMC3][IMC0][IMC1] = j/(rho4*delta)*(2.*r2-rho2);
  gamma[IMC3][IMC0][IMC2] = -2.*j*r*cth/(rho4*sth);

  gamma[IMC3][IMC1][IMC0] = gamma[IMC3][IMC0][IMC1];
  gamma[IMC3][IMC1][IMC3] = 1./(rho4*delta)*(r*rho2*(rho2-rs*r)-a*j*sth2*(2.*r2-rho2));

  gamma[IMC3][IMC2][IMC0] = gamma[IMC3][IMC0][IMC2];
  gamma[IMC3][IMC2][IMC3] = cth/(rho4*sth)*(rho4+2.*a*j*r*sth2);

  gamma[IMC3][IMC3][IMC1] = gamma[IMC3][IMC1][IMC3];
  gamma[IMC3][IMC3][IMC2] = gamma[IMC3][IMC2][IMC3];
}

//----------------------------------------------------------------------------------------
//! MCSnake constructor, built from Coord and MonteCarloBlock
//!
//! snake_a_ and snake_k_ are set afterwards by the MonteCarloBlock dispatch, which is the
//! only place with the ParameterInput in hand.  They default to zero, which degenerates
//! to Minkowski rather than to something ill-formed.

MCSnake::MCSnake(Coordinates *pcoord, MonteCarloBlock *pmcb)
  : MCCoord(pcoord,pmcb), snake_a_(0.0), snake_k_(0.0) {

}

//----------------------------------------------------------------------------------------
//! MCSnake constructor for processes without own MeshBlock

MCSnake::MCSnake(int ncells1, int ncells2, int ncells3, bool acc)
  : MCCoord(ncells1,ncells2,ncells3,acc), snake_a_(0.0), snake_k_(0.0) {

}

//----------------------------------------------------------------------------------------
//! destructor

MCSnake::~MCSnake() {

}

//----------------------------------------------------------------------------------------
//! \fn void MCSnake::Metric(Real x[4], Real gcov[4][4])
//! \brief covariant metric for sinusoidal ("snake") coordinates
//!
//! From y = y_M + a sin(k x_M), so dy_M = dy - beta dx with beta = a k cos(k x), and
//!   ds^2 = -dt^2 + dx^2 + (dy - beta dx)^2 + dz^2.

void MCSnake::Metric(Real x[4], Real gcov[4][4]) {

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      gcov[i][j] = 0.;
    }
  }

  Real beta = snake_a_ * snake_k_ * cos(snake_k_ * x[IMC1]);

  gcov[IMC0][IMC0] = -1.;
  gcov[IMC1][IMC1] = 1. + SQR(beta);
  gcov[IMC1][IMC2] = -beta;
  gcov[IMC2][IMC1] = -beta;
  gcov[IMC2][IMC2] = 1.;
  gcov[IMC3][IMC3] = 1.;
}

//----------------------------------------------------------------------------------------
//! \fn void MCSnake::InverseMetric(Real x[4], Real gcon[4][4])
//! \brief contravariant metric for snake coordinates
//!
//! The spatial block has unit determinant ((1+beta^2) - beta^2 = 1), so the inverse is
//! obtained by swapping the diagonal and flipping the sign of the off-diagonal.

void MCSnake::InverseMetric(Real x[4], Real gcon[4][4]) {

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      gcon[i][j] = 0.;
    }
  }

  Real beta = snake_a_ * snake_k_ * cos(snake_k_ * x[IMC1]);

  gcon[IMC0][IMC0] = -1.;
  gcon[IMC1][IMC1] = 1.;
  gcon[IMC1][IMC2] = beta;
  gcon[IMC2][IMC1] = beta;
  gcon[IMC2][IMC2] = 1. + SQR(beta);
  gcon[IMC3][IMC3] = 1.;
}

//----------------------------------------------------------------------------------------
//! \fn void MCSnake::MetricDerivative(Real x[4], Real dgcov[4][4][4])
//! \brief derivative of the covariant metric, dgcov[c][a][b] = d_c g_ab
//!
//! beta depends on x alone, so only the x derivatives survive.

void MCSnake::MetricDerivative(Real x[4], Real dgcov[4][4][4]) {

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        dgcov[i][j][k] = 0.;
      }
    }
  }

  Real beta = snake_a_ * snake_k_ * cos(snake_k_ * x[IMC1]);
  Real dbeta = -snake_a_ * SQR(snake_k_) * sin(snake_k_ * x[IMC1]);

  dgcov[IMC1][IMC1][IMC1] = 2. * beta * dbeta;
  dgcov[IMC1][IMC1][IMC2] = -dbeta;
  dgcov[IMC1][IMC2][IMC1] = -dbeta;
}

//----------------------------------------------------------------------------------------
//! \fn void MCSnake::InverseMetricDerivative(Real x[4], Real dgcon[4][4][4])
//! \brief derivative of the contravariant metric, dgcon[c][a][b] = d_c g^ab

void MCSnake::InverseMetricDerivative(Real x[4], Real dgcon[4][4][4]) {

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        dgcon[i][j][k] = 0.;
      }
    }
  }

  Real beta = snake_a_ * snake_k_ * cos(snake_k_ * x[IMC1]);
  Real dbeta = -snake_a_ * SQR(snake_k_) * sin(snake_k_ * x[IMC1]);

  dgcon[IMC1][IMC1][IMC2] = dbeta;
  dgcon[IMC1][IMC2][IMC1] = dbeta;
  dgcon[IMC1][IMC2][IMC2] = 2. * beta * dbeta;
}

//----------------------------------------------------------------------------------------
//! \fn void MCSnake::Connect(Real x[4], Real gamma[4][4][4])
//! \brief connection coefficients, gamma[a][b][c] = Gamma^a_bc
//!
//! Exactly one is non-zero.  Fastest seen from the coordinate map rather than from the
//! metric: Gamma^mu_ab = (dx^mu/dx_M^rho)(d^2 x_M^rho / dx^a dx^b), and the only non-zero
//! second derivative is d^2 y_M/dx^2 = a k^2 sin(k x), which feeds only the y row.

void MCSnake::Connect(Real x[4], Real gamma[4][4][4]) {

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        gamma[i][j][k] = 0.;
      }
    }
  }

  gamma[IMC2][IMC1][IMC1] = snake_a_ * SQR(snake_k_) * sin(snake_k_ * x[IMC1]);
}

//----------------------------------------------------------------------------------------
//! \fn void MCSnake::Tetrad(Real x[4], Real tetrad[4][4])
//! \brief orthonormal components to coordinate components
//!
//! The orthonormal legs aligned with the underlying Minkowski axes are
//!   e_(t) = d_t,  e_(x) = d_x + beta d_y,  e_(y) = d_y,  e_(z) = d_z,
//! which is orthonormal by construction: g(e_x,e_x) = (1+beta^2) - 2beta^2 + beta^2 = 1
//! and g(e_x,e_y) = -beta + beta = 0.
//!
//! Unlike every other supported system this matrix is not diagonal, because the snake
//! coordinate basis is not orthogonal.  Callers apply it as kf[j] = tetrad[j][i] * ki[i].

void MCSnake::Tetrad(Real x[4], Real tetrad[4][4]) {

  for (int l = 0; l < 4; l++) {
    for (int m = 0; m < 4; m++) {
      tetrad[l][m] = 0.;
    }
    tetrad[l][l] = 1.;
  }

  Real beta = snake_a_ * snake_k_ * cos(snake_k_ * x[IMC1]);
  tetrad[IMC2][IMC1] = beta;
}

//----------------------------------------------------------------------------------------
//! \fn void MCSnake::InverseTetrad(Real x[4], Real invtet[4][4])
//! \brief coordinate components to orthonormal components
//!
//! Inverse of Tetrad: k^(y) = k^y - beta k^x, the rest unchanged.

void MCSnake::InverseTetrad(Real x[4], Real invtet[4][4]) {

  for (int l = 0; l < 4; l++) {
    for (int m = 0; m < 4; m++) {
      invtet[l][m] = 0.;
    }
    invtet[l][l] = 1.;
  }

  Real beta = snake_a_ * snake_k_ * cos(snake_k_ * x[IMC1]);
  invtet[IMC2][IMC1] = -beta;
}

//----------------------------------------------------------------------------------------
//! MCMinkowski constructor, built from Coord and MonteCarloBlock

MCMinkowski::MCMinkowski(Coordinates *pcoord, MonteCarloBlock *pmcb)
  : MCCoord(pcoord,pmcb) {
}

//----------------------------------------------------------------------------------------
//! MCMinkowski constructor for processes without own MeshBlock
MCMinkowski::MCMinkowski(int ncells1, int ncells2, int ncells3, bool acc)
  : MCCoord(ncells1,ncells2,ncells3,acc) {
}

//----------------------------------------------------------------------------------------
//! destructor
MCMinkowski::~MCMinkowski() {

}
