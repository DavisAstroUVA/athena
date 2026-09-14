#ifndef PHOTON_HPP
#define PHOTON_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file photon.hpp
//! \brief definitions for Photon class

// C++ libraries
#include <complex>
#include <vector>

// Athena++ classes headers
#include "../athena.hpp"
#include "montecarlo.hpp"
#include "polarization.hpp"  // MCPolarization; no includes of its own, so no cycle

class MonteCarloBlock;

//! Whether the Verlet is used and relevant quantities stored
#define MC_VERLET_DK 0

// photon status identifiers
enum PhotonStatus {EVOLVING = 0, ESCAPED = 1, ABSORBED = 2, DESTROYED = 3, BUFFERED = 4, REMOVED = 5};
enum {IMC0 = 0, IMC1 = 1, IMC2 = 2, IMC3 = 3};
//enum {IMC1 = 0, IMC2 = 1, IMC3 = 2, IMC0 = 3};

//---------------------------------------------------------------------------------------
//! \class Photon
//! \brief photon data and functions

class Photon : public Particles {
public:
  Photon(MonteCarloBlock *pmcb, ParameterInput *pin);
  ~Photon();

  // public functions
  //void RemoveOneParticle(int k);
  void PrintPhoton(const std::string &msg, int ip) const;

  //! pack/unpack the stored wavevector as a genuine contravariant four-vector.  The
  //! spatial components are stored either as a dimensional k^i or as a unit direction,
  //! and these hide that difference from everything that just wants the four-vector.
  void GetFourVector(int ip, bool unit_spatial, Real k[4]) const;
  void SetFourVector(int ip, bool unit_spatial, const Real k[4]);
  void PrintPhoton(int ip) const;
  bool IsNanPhoton(int ip);
  //! the cheap subset of IsNanPhoton for a per-step check: weight, position, wavevector
  bool IsNanTransport(int ip) const;
  void PolarizationToTetrad(std::complex<Real> ttet[4][4], Real ecov[4][4], const int ip);
  void PolarizationToCoord(std::complex<Real> ttet[4][4], Real econ[4][4], const int ip);

  //! The coherency tensor is Hermitian, so it is stored as sixteen reals rather than
  //! sixteen complex numbers: the four real diagonal entries in slots 0..3, then the real
  //! and imaginary parts of the six entries above the diagonal, in the order (0,1) (0,2)
  //! (0,3) (1,2) (1,3) (2,3).  Nothing outside these four functions knows the layout.
  //! LoadTensor expands to the full complex 4x4, mirroring the lower triangle as the
  //! conjugate; StoreTensor keeps the diagonal's real part and the upper triangle.
  static int TensorSlot(int i, int j, bool imag);
  void LoadTensor(int ip, std::complex<Real> n[4][4]) const;
  void StoreTensor(int ip, const std::complex<Real> n[4][4]);
  std::complex<Real> Tensor(int ip, int i, int j) const;
  void AllocatePhotons(int nphot);
  void SendToNeighbors();
  void ApplyPeriodicBoundary(Real &x1, Real &x2, Real &x3, int k);
  bool ReceiveFromNeighbors();
  void GetPositionIndices(int ibegin, int iend);
  static void Initialize(MonteCarlo *pmc, ParameterInput *pin);

  //! reset the boundary state between transfer rounds
  //!
  //! Overrides Particles::ClearBoundary, which leaves every neighbor "waiting" so that
  //! each one has to report in even when it has nothing to send.  A photon handed to a
  //! same-rank neighbor is written straight into that block's receive buffer during the
  //! send sweep, so for those neighbors there is nothing to wait for: they start
  //! "completed" here and a sender marks one "arrived" only when it actually delivers.
  //! Off-rank neighbors keep the waiting/poll protocol, which MPI still needs.
  void ClearBoundary();

  //! record, once, whether any neighbor of this block lives on another rank.  Such a
  //! block has to take part in every round to keep the MPI protocol matched, whether or
  //! not it holds photons.
  void SetOffRankNeighborFlag();

  //! does this block need the boundary sweeps this round?  A block with no photons, no
  //! delivery from a same-rank neighbor and no off-rank neighbor cannot send, receive or
  //! dirty any boundary state, so every sweep can skip it.
  bool NeedsBoundaryWork() const {
    return nphot > 0 || has_incoming_ || has_offrank_neighbor_;
  }

  //! add every rank this block has a neighbor on to `seen`, for the peer list
  void CollectPeerRanks(std::vector<bool> &seen) const;

  //! copy photons that arrived through the rank exchange into one receive slot, exactly
  //! as an off-rank MPI receive used to fill it
  void AcceptPhotons(int bufid, const int *ib, const Real *rb,
                     const std::complex<Real> *cb, int npar);

  //! per-photon property counts, so the exchange can size its buffers without reaching
  //! into ParticleBuffer, which it is not a friend of
  static int PropertyCountInt();
  static int PropertyCountReal();
  static int PropertyCountCplx();

  //! set by a sender that has just deposited photons into this block's receive buffer
  bool has_incoming_;
  bool has_offrank_neighbor_;

  // public data
  // SWD: should be reorganized with tighter access control for some variables
  MonteCarloBlock* pmy_mcb; // ptr to MonteCarlo currently containing this Photon

  int nuser_var;
  int nphot_limit;
  int &nphot;

  static int istatp, inscp, ityp, inmvp;
  static int ii1p, ii2p, ii3p;
  static int ix0p, ix1p, ix2p, ix3p;
  static int ik0p, ik1p, ik2p, ik3p;
  static int idk0p, idk1p, idk2p, idk3p;
  static int iep, iwp, iscp, iacp;
  static int isip, isqp, isup, isvp;
  static int iuserp;
  static int ipolp;
  static int idtp;

  std::vector<int> &statp, &nscp, &type;
  //! count of pusher steps taken since this photon last scattered, accumulated across
  //! every Move call and every block it has crossed in that free flight.  Registered as a
  //! photon property so that it rides along in the MPI buffers and survives a transfer to
  //! another block; `capmove` retires a photon whose free flight runs past the cap.
  std::vector<int> &nmvp;
  std::vector<int> &i1p, &i2p, &i3p;
  std::vector<Real> &x0p, &x1p, &x2p, &x3p;
  std::vector<Real> &k0p, &k1p, &k2p, &k3p;
  //! Verlet's dk/dlambda; see MC_VERLET_DK.  With it off these are four references to
  //! dk_scratch_, kept at least as long as the photon arrays so writes stay in bounds.
  std::vector<Real> &dk0p, &dk1p, &dk2p, &dk3p;
  static std::vector<Real> dk_scratch_;
  void EnsureScratch();
  std::vector<Real> &ep, &wp, &scp, &acp;
  std::vector<Real> &sip, &sqp, &sup, &svp;
  std::vector<Real> &dtp;
  std::vector<Real> *user;     //!>   user variable arrays
  //! the sixteen real columns of the coherency tensor; see TensorSlot for the layout, and
  //! go through LoadTensor/StoreTensor/Tensor rather than indexing this directly
  std::vector<Real> *polten;

  static bool initialized;
  static MCPolarization polarized;
  static bool general_pusher_flag;

  //#ifdef MPI_PARALLEL
  //static MPI_Comm my_comm;   //!> my MPI communicator
  //ParticleBuffer send_[56];  //!> particle send buffers
  //#endif


};
#endif // PHOTON_HPP
