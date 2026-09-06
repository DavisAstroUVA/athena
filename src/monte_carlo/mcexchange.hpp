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

  //! Post what is staged, without waiting for anyone.  A peer with nothing to receive is
  //! sent nothing at all, so a rank that is idle this round puts no message on the wire.
  //! That is what the size handshake in ExchangeAndDeliver cannot do: it has to tell
  //! every peer "nothing for you", which makes an idle round cost 2*npeers messages on
  //! every rank and couples the whole domain to the slowest one.
  void SendStaged();

  //! Take delivery of whatever has arrived, from anyone, and hand it to the blocks that
  //! own it.  Returns the number of photons delivered.  Never blocks: it drains what is
  //! there and returns, so a rank with no work spins cheaply instead of waiting on peers
  //! that may have nothing to say this round.
  int DrainIncoming();

  //! Wait for the posted sends to release their buffers, so Reset can reuse them, taking
  //! delivery of anything that arrives while waiting.  A large send completes only once
  //! the peer has matched it, so this has to keep receiving or two ranks each waiting on
  //! their own sends would wait on each other forever.  It is still not a synchronization
  //! point: termination is decided from the counters below, never from send completion.
  void CompleteSends();

  //! Cumulative photons handed to other ranks and taken from them.  Termination needs a
  //! quantity that accounts for what is in flight, and these are it: a photon staged here
  //! is counted out at once and counted in only when it lands in a block, so the global
  //! sums differ by exactly the number of photons on the wire.
  int64_t NumSent() const { return nsent_; }
  int64_t NumRecv() const { return nrecv_; }

  //! true when the mesh actually spans more than one rank and there is anything to do
  bool Active() const { return active_; }

  //! number of distinct peer ranks, for diagnostics
  int NumPeers() const { return static_cast<int>(peers_.size()); }

 private:
  MonteCarlo *pmy_mc_;
  bool active_;
  std::vector<int> peers_;        //!> distinct ranks sharing a boundary with this one
  std::vector<int> peer_of_rank_; //!> rank -> index into peers_, -1 if not a peer

  int64_t nsent_, nrecv_;         //!> cumulative photons out of and into this rank

  // Buffers for the probe-driven path.  The int stream is sent as one message that leads
  // with the header, so a receiver that has probed it can work out the length of every
  // other stream from its contents and post exact receives for them.
  std::vector<std::vector<int> > smsg_;
#ifdef MPI_PARALLEL
  std::vector<MPI_Request> sreq_;
#endif

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
