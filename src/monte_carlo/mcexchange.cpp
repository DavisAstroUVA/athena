//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mcexchange.cpp
//! \brief rank-aggregated photon exchange between MeshBlocks on different processes

// C++ headers
#include <complex>
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../globals.hpp"
#include "../mesh/mesh.hpp"
#include "mcexchange.hpp"
#include "montecarlo.hpp"
#include "photon.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

// Tags for the four streams.  The peer set is fixed and each pair of ranks exchanges at
// most one message of each kind per round, so a constant tag per stream is enough to keep
// them apart.
namespace {
const int kTagSizes = 900;
const int kTagHdr   = 901;
const int kTagInt   = 902;
const int kTagReal  = 903;
const int kTagCplx  = 904;
const int kHdrWords = 3;   // (lid, bufid, npar) per contributing block
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn MCRankExchange::MCRankExchange(MonteCarlo *pmc)

MCRankExchange::MCRankExchange(MonteCarlo *pmc)
    : pmy_mc_(pmc), active_(false), nsent_(0), nrecv_(0) {}

//----------------------------------------------------------------------------------------
//! \fn void MCRankExchange::BuildPeerList()
//! \brief collect the distinct ranks this one shares a block boundary with
//!
//! The relation is symmetric -- if a block here neighbors a block there, that block
//! neighbors this one -- so the peer sets match up on both sides without any negotiation,
//! which is what lets the exchange below be a simple paired Isend/Irecv per peer.

void MCRankExchange::BuildPeerList() {
  peers_.clear();
  peer_of_rank_.assign(Globals::nranks, -1);
  active_ = false;
#ifdef MPI_PARALLEL
  if (Globals::nranks < 2) return;
  std::vector<bool> seen(Globals::nranks, false);
  for (int nb = 0; nb < pmy_mc_->nblocal; ++nb)
    pmy_mc_->my_blocks(nb)->pphot->CollectPeerRanks(seen);
  for (int r = 0; r < Globals::nranks; ++r) {
    if (seen[r] && r != Globals::my_rank) {
      peer_of_rank_[r] = static_cast<int>(peers_.size());
      peers_.push_back(r);
    }
  }
  const std::size_t np = peers_.size();
  shdr_.resize(np);  sint_.resize(np);  sreal_.resize(np);  scplx_.resize(np);
  rhdr_.resize(np);  rint_.resize(np);  rreal_.resize(np);  rcplx_.resize(np);
  smsg_.resize(np);
  active_ = !peers_.empty();
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void MCRankExchange::Reset()

void MCRankExchange::Reset() {
  for (std::size_t p = 0; p < peers_.size(); ++p) {
    shdr_[p].clear();  sint_[p].clear();  sreal_[p].clear();  scplx_[p].clear();
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MCRankExchange::Stage(...)
//! \brief add one block's outgoing photons to the buffer for their destination rank

void MCRankExchange::Stage(int dest_rank, int lid, int bufid, const int *ib,
                           const Real *rb, const std::complex<Real> *cb, int npar) {
  if (npar <= 0) return;
  const int p = peer_of_rank_[dest_rank];
  if (p < 0) return;   // not a peer: cannot happen, but do not corrupt memory if it does

  shdr_[p].push_back(lid);
  shdr_[p].push_back(bufid);
  shdr_[p].push_back(npar);
  nsent_ += npar;

  const int ni = Photon::PropertyCountInt();
  const int nr = Photon::PropertyCountReal();
  const int nc = Photon::PropertyCountCplx();
  sint_[p].insert(sint_[p].end(), ib, ib + npar*ni);
  sreal_[p].insert(sreal_[p].end(), rb, rb + npar*nr);
  if (nc > 0 && cb != nullptr)
    scplx_[p].insert(scplx_[p].end(), cb, cb + npar*nc);
}

//----------------------------------------------------------------------------------------
//! \fn void MCRankExchange::ExchangeAndDeliver()
//! \brief one exchange per peer, then hand the arrivals to the blocks they belong to
//!
//! Two phases.  The first tells each peer how much is coming, which every peer needs
//! before it can post a receive of the right size; it is four ints, and it goes out even
//! when nothing follows.  The second moves the payload only between peers that have one.
//! That is at most five small messages per peer per round, against one per off-rank
//! neighbor link in the per-block scheme.

void MCRankExchange::ExchangeAndDeliver() {
#ifdef MPI_PARALLEL
  if (!active_) return;
  const std::size_t np = peers_.size();
  const int nc = Photon::PropertyCountCplx();

  // ---- phase one: sizes ----
  std::vector<int> ssz(4*np), rsz(4*np);
  std::vector<MPI_Request> req;
  req.reserve(4*np);
  for (std::size_t p = 0; p < np; ++p) {
    ssz[4*p+0] = static_cast<int>(shdr_[p].size());
    ssz[4*p+1] = static_cast<int>(sint_[p].size());
    ssz[4*p+2] = static_cast<int>(sreal_[p].size());
    ssz[4*p+3] = static_cast<int>(scplx_[p].size());
  }
  for (std::size_t p = 0; p < np; ++p) {
    MPI_Request r;
    MPI_Irecv(&rsz[4*p], 4, MPI_INT, peers_[p], kTagSizes, MPI_COMM_WORLD, &r);
    req.push_back(r);
    MPI_Isend(&ssz[4*p], 4, MPI_INT, peers_[p], kTagSizes, MPI_COMM_WORLD, &r);
    req.push_back(r);
  }
  MPI_Waitall(static_cast<int>(req.size()), req.data(), MPI_STATUSES_IGNORE);
  req.clear();

  // ---- phase two: payload, only where there is one ----
  for (std::size_t p = 0; p < np; ++p) {
    rhdr_[p].resize(rsz[4*p+0]);
    rint_[p].resize(rsz[4*p+1]);
    rreal_[p].resize(rsz[4*p+2]);
    rcplx_[p].resize(rsz[4*p+3]);
    MPI_Request r;
    if (rsz[4*p+0] > 0) {
      MPI_Irecv(rhdr_[p].data(), rsz[4*p+0], MPI_INT, peers_[p], kTagHdr,
                MPI_COMM_WORLD, &r);
      req.push_back(r);
      MPI_Irecv(rint_[p].data(), rsz[4*p+1], MPI_INT, peers_[p], kTagInt,
                MPI_COMM_WORLD, &r);
      req.push_back(r);
      MPI_Irecv(rreal_[p].data(), rsz[4*p+2], MPI_ATHENA_REAL, peers_[p], kTagReal,
                MPI_COMM_WORLD, &r);
      req.push_back(r);
      if (rsz[4*p+3] > 0) {
        MPI_Irecv(rcplx_[p].data(), rsz[4*p+3], MPI_ATHENA_COMPLEX, peers_[p], kTagCplx,
                  MPI_COMM_WORLD, &r);
        req.push_back(r);
      }
    }
    if (!shdr_[p].empty()) {
      MPI_Isend(shdr_[p].data(), static_cast<int>(shdr_[p].size()), MPI_INT,
                peers_[p], kTagHdr, MPI_COMM_WORLD, &r);
      req.push_back(r);
      MPI_Isend(sint_[p].data(), static_cast<int>(sint_[p].size()), MPI_INT,
                peers_[p], kTagInt, MPI_COMM_WORLD, &r);
      req.push_back(r);
      MPI_Isend(sreal_[p].data(), static_cast<int>(sreal_[p].size()), MPI_ATHENA_REAL,
                peers_[p], kTagReal, MPI_COMM_WORLD, &r);
      req.push_back(r);
      if (!scplx_[p].empty()) {
        MPI_Isend(scplx_[p].data(), static_cast<int>(scplx_[p].size()),
                  MPI_ATHENA_COMPLEX, peers_[p], kTagCplx, MPI_COMM_WORLD, &r);
        req.push_back(r);
      }
    }
  }
  if (!req.empty())
    MPI_Waitall(static_cast<int>(req.size()), req.data(), MPI_STATUSES_IGNORE);

  // ---- deliver ----
  const int ni = Photon::PropertyCountInt();
  const int nr = Photon::PropertyCountReal();
  for (std::size_t p = 0; p < np; ++p) {
    std::size_t oi = 0, orr = 0, oc = 0;
    for (std::size_t h = 0; h + kHdrWords <= rhdr_[p].size(); h += kHdrWords) {
      const int lid = rhdr_[p][h];
      const int bufid = rhdr_[p][h+1];
      const int npar = rhdr_[p][h+2];
      Photon *pp = pmy_mc_->my_blocks(lid)->pphot;
      pp->AcceptPhotons(bufid, &rint_[p][oi], &rreal_[p][orr],
                        (nc > 0 && oc < rcplx_[p].size()) ? &rcplx_[p][oc] : nullptr,
                        npar);
      nrecv_ += npar;
      oi += npar*ni;
      orr += npar*nr;
      oc += npar*nc;
    }
  }
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void MCRankExchange::SendStaged()
//! \brief post this round's outgoing photons and return without waiting
//!
//! One message per stream per peer that actually has photons waiting, and nothing at all
//! for a peer that does not.  The int stream leads with the header, so a receiver holding
//! only that one message can read off how many photons are coming and post exact receives
//! for the real and complex streams; that is what removes the need for the size
//! handshake, and with it the need for an idle rank to say anything.

void MCRankExchange::SendStaged() {
#ifdef MPI_PARALLEL
  if (!active_) return;
  for (std::size_t p = 0; p < peers_.size(); ++p) {
    if (shdr_[p].empty()) continue;

    smsg_[p].clear();
    smsg_[p].reserve(1 + shdr_[p].size() + sint_[p].size());
    smsg_[p].push_back(static_cast<int>(shdr_[p].size()/kHdrWords));
    smsg_[p].insert(smsg_[p].end(), shdr_[p].begin(), shdr_[p].end());
    smsg_[p].insert(smsg_[p].end(), sint_[p].begin(), sint_[p].end());

    MPI_Request r;
    MPI_Isend(smsg_[p].data(), static_cast<int>(smsg_[p].size()), MPI_INT,
              peers_[p], kTagInt, MPI_COMM_WORLD, &r);
    sreq_.push_back(r);
    MPI_Isend(sreal_[p].data(), static_cast<int>(sreal_[p].size()), MPI_ATHENA_REAL,
              peers_[p], kTagReal, MPI_COMM_WORLD, &r);
    sreq_.push_back(r);
    if (!scplx_[p].empty()) {
      MPI_Isend(scplx_[p].data(), static_cast<int>(scplx_[p].size()),
                MPI_ATHENA_COMPLEX, peers_[p], kTagCplx, MPI_COMM_WORLD, &r);
      sreq_.push_back(r);
    }
  }
#endif
}

//----------------------------------------------------------------------------------------
//! \fn int MCRankExchange::DrainIncoming()
//! \brief take delivery of everything that has arrived from anyone, without blocking
//!
//! Probing on the int stream alone is enough to find a sender: a rank that sends anything
//! always sends that stream, and MPI keeps messages between a given pair of ranks in
//! order on a tag, so the real and complex streams waiting behind it belong to the header
//! just read and can be received with exact, already-known counts.
//!
//! Messages from a round ahead of this one are accepted here too, and that is harmless:
//! the photons are appended to the receiving block's buffer and transported whenever it
//! next runs.  Transport does not depend on the order photons arrive in.

int MCRankExchange::DrainIncoming() {
  int ndeliv = 0;
#ifdef MPI_PARALLEL
  if (!active_) return 0;
  const int ni = Photon::PropertyCountInt();
  const int nr = Photon::PropertyCountReal();
  const int nc = Photon::PropertyCountCplx();

  std::vector<int> msg;
  std::vector<Real> rb;
  std::vector<std::complex<Real> > cb;

  int flag = 0;
  MPI_Status st;
  while (true) {
    MPI_Iprobe(MPI_ANY_SOURCE, kTagInt, MPI_COMM_WORLD, &flag, &st);
    if (!flag) break;
    const int src = st.MPI_SOURCE;
    int nw = 0;
    MPI_Get_count(&st, MPI_INT, &nw);
    msg.resize(nw);
    MPI_Recv(msg.data(), nw, MPI_INT, src, kTagInt, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    const int ntrip = msg[0];
    int npar_tot = 0;
    for (int t = 0; t < ntrip; ++t) npar_tot += msg[1 + kHdrWords*t + 2];

    rb.resize(static_cast<std::size_t>(npar_tot)*nr);
    MPI_Recv(rb.data(), static_cast<int>(rb.size()), MPI_ATHENA_REAL, src, kTagReal,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (nc > 0) {
      cb.resize(static_cast<std::size_t>(npar_tot)*nc);
      MPI_Recv(cb.data(), static_cast<int>(cb.size()), MPI_ATHENA_COMPLEX, src, kTagCplx,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // The int payload starts after the count word and the header triples.
    const std::size_t ibase = 1 + static_cast<std::size_t>(kHdrWords)*ntrip;
    std::size_t oi = 0, orr = 0, oc = 0;
    for (int t = 0; t < ntrip; ++t) {
      const int lid = msg[1 + kHdrWords*t + 0];
      const int bufid = msg[1 + kHdrWords*t + 1];
      const int npar = msg[1 + kHdrWords*t + 2];
      Photon *pp = pmy_mc_->my_blocks(lid)->pphot;
      pp->AcceptPhotons(bufid, &msg[ibase + oi], &rb[orr],
                        (nc > 0) ? &cb[oc] : nullptr, npar);
      nrecv_ += npar;
      ndeliv += npar;
      oi += static_cast<std::size_t>(npar)*ni;
      orr += static_cast<std::size_t>(npar)*nr;
      oc += static_cast<std::size_t>(npar)*nc;
    }
  }
#endif
  return ndeliv;
}

//----------------------------------------------------------------------------------------
//! \fn void MCRankExchange::CompleteSends()
//! \brief release the posted send buffers so Reset can reuse them, taking delivery while
//!        waiting
//!
//! Draining here is not an optimization, it is what keeps this from deadlocking.  A send
//! large enough to go by rendezvous does not complete until the receiver has matched it,
//! so a rank that waits on its own sends without receiving anything is waiting on a peer
//! that may be doing exactly the same thing.  Every rank reaching this point at once is
//! the normal case, not a rare one -- it is what happens on the first round, when they
//! all have a full complement of photons to hand over.

void MCRankExchange::CompleteSends() {
#ifdef MPI_PARALLEL
  if (sreq_.empty()) return;
  while (true) {
    int flag = 0;
    MPI_Testall(static_cast<int>(sreq_.size()), sreq_.data(), &flag, MPI_STATUSES_IGNORE);
    if (flag) break;
    DrainIncoming();
  }
  sreq_.clear();
#endif
}
