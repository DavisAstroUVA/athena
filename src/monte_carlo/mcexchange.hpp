#ifndef MCEXCHANGE_HPP
#define MCEXCHANGE_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mcexchange.hpp
//! \brief rank-aggregated photon exchange between MeshBlocks on different processes
//
// The per-block protocol sends one count message down every neighbor link that crosses a
// rank boundary, every transfer round, whether or not there is a photon to go with it,
// and then polls each of them to completion.  That is fine when blocks are large and few.
// It is not fine on a mesh read from an AMR snapshot: with 83231 blocks of 8x4x8 cells on
// 32 ranks, roughly a third of the ~56 neighbor links per block cross a rank boundary, so
// each rank issues on the order of 50000 blocking sends per round and every rank waits on
// the slowest of them.  Measured on a 4096-block test problem, going from 1 rank to 8
// made the run more than ten times *slower*.
//
// Each rank has only 11 to 17 distinct peers in that same measurement, though, so this
// collects everything leaving for a given rank into one buffer and sends it once.  The
// photons themselves are untouched: the per-block packing, and the direct hand-off
// between blocks that share a rank, both work exactly as before.  What changes is that a
// block no longer talks to a neighbor block across a rank boundary; a rank talks to a
// rank.

// C++ headers
#include <complex>
#include <vector>

// Athena++ headers
#include "../athena.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

// Forward declarations
class MonteCarlo;

//----------------------------------------------------------------------------------------
//! \class MCRankExchange
//! \brief collects photons bound for other ranks and moves them in one message per peer

class MCRankExchange {
 public:
  explicit MCRankExchange(MonteCarlo *pmc);

  //! Work out which ranks this one shares a boundary with.  Fixed for the life of the
  //! mesh, so it is done once after the neighbor lists are linked.
  void BuildPeerList();

  //! Empty the staging buffers at the start of a transfer round.
  void Reset();

  //! Add one block's outgoing photons for a neighbor on another rank.  lid and bufid say
  //! where they belong on the receiving side: the neighbor's local block index and the
  //! buffer slot it expects them in.
  void Stage(int dest_rank, int lid, int bufid, const int *ib, const Real *rb,
             const std::complex<Real> *cb, int npar);

  //! Move everything staged, one exchange per peer, and hand the arrivals to the blocks
  //! they belong to.  Collective over the peer set, so every rank calls it every round.
  void ExchangeAndDeliver();

  //! true when the mesh actually spans more than one rank and there is anything to do
  bool Active() const { return active_; }

  //! number of distinct peer ranks, for diagnostics
  int NumPeers() const { return static_cast<int>(peers_.size()); }

 private:
  MonteCarlo *pmy_mc_;
  bool active_;
  std::vector<int> peers_;        //!> distinct ranks sharing a boundary with this one
  std::vector<int> peer_of_rank_; //!> rank -> index into peers_, -1 if not a peer

  // Staging, one entry per peer.  The header holds (lid, bufid, npar) per contributing
  // block; the three streams hold the photon properties back to back in that order.
  std::vector<std::vector<int> > shdr_, sint_;
  std::vector<std::vector<Real> > sreal_;
  std::vector<std::vector<std::complex<Real> > > scplx_;
  std::vector<std::vector<int> > rhdr_, rint_;
  std::vector<std::vector<Real> > rreal_;
  std::vector<std::vector<std::complex<Real> > > rcplx_;
};

#endif  // MCEXCHANGE_HPP
