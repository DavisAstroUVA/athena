//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file monte_carlo.cpp
//! \brief implementation of functions in class Photon

// C++ Standard Libraries
#include <vector>
#include <stdexcept>  // runtime_error

// Athena++ headers
#include "photon.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "mcexchange.hpp"

namespace {
//! Charges the time a hand-off function spends to its block's pending load-balancing
//! cost, which MonteCarlo folds into the block's cost with its sweeps.  Photons crossing
//! block boundaries are a large part of the transport time on a mesh of many blocks
//! and it falls on the blocks the photons leave and enter
struct HandoffTimer {
  MonteCarloBlock *pmcb;
  double t0;
  explicit HandoffTimer(MonteCarloBlock *b) : pmcb(b), t0(MonteCarlo::LoadBalanceClock()) {}
  ~HandoffTimer() { pmcb->lb_pending += MonteCarlo::LoadBalanceClock() - t0; }
};
} // namespace

// class variable initialization
bool Photon::initialized = false;
MCPolarization Photon::polarized = MCPOL_NONE;
bool Photon::general_pusher_flag = false;

int Photon::inscp = -1, Photon::istatp = -1, Photon::ityp = -1, Photon::inmvp = -1;
int Photon::ii1p = -1, Photon::ii2p = -1, Photon::ii3p = -1;
int Photon::ix0p = -1, Photon::ix1p = -1, Photon::ix2p = -1, Photon::ix3p = -1;
int Photon::ik0p = -1, Photon::ik1p = -1, Photon::ik2p = -1, Photon::ik3p = -1;
int Photon::idk0p = -1, Photon::idk1p = -1, Photon::idk2p = -1, Photon::idk3p = -1;
std::vector<Real> Photon::dk_scratch_;
int Photon::iep = -1, Photon::iwp = -1, Photon::iscp = -1, Photon::iacp = -1;
int Photon::isip = -1, Photon::isqp = -1, Photon::isup = -1, Photon::isvp = -1;
int Photon::iuserp = -1, Photon::ipolp = -1, Photon::idtp = -1;

// Local function prototypes
static int CheckSide(int xi, int xi1, int xi2);
static int nloc = 0;
static int nper = 0;
static int nbuf = 0;
static int nnper = 0;
static int nadj = 0;
static int nmpi = 0;

//----------------------------------------------------------------------------------------
//! Photon constructor

Photon::Photon(MonteCarloBlock *pmcb, ParameterInput *pin)
  : Particles(pmcb->pmy_block, pin),
    has_incoming_(false), has_offrank_neighbor_(false),
  // Allocate space for photon data via initialization list
    //user(new std::vector<Real> [pmcb->pmy_mc->nuser_var]),
    //polten(new std::vector<std::complex<Real>> [ncplx]),
    nphot(npar),nscp(intprop[inscp]), statp(intprop[istatp]),
    type(intprop[ityp]), nmvp(intprop[inmvp]),
    i1p(intprop[ii1p]), i2p(intprop[ii2p]), i3p(intprop[ii3p]),
    x0p(rp[ix0p]), x1p(rp[ix1p]), x2p(rp[ix2p]), x3p(rp[ix3p]),
    k0p(rp[ik0p]), k1p(rp[ik1p]), k2p(rp[ik2p]), k3p(rp[ik3p]),
#if MC_VERLET_DK
    dk0p(rp[idk0p]), dk1p(rp[idk1p]), dk2p(rp[idk2p]),
    dk3p(rp[idk3p]),
#else
    dk0p(dk_scratch_), dk1p(dk_scratch_), dk2p(dk_scratch_), dk3p(dk_scratch_),
#endif
    ep(rp[iep]), wp(rp[iwp]), scp(rp[iscp]), acp(rp[iacp]),
    sip(rp[isip]), sqp(rp[isqp]), sup(rp[isup]), svp(rp[isvp]),
    dtp(rp[idtp]) {

  pmy_mcb = pmcb;
  nphot_limit = pmcb->pmy_mc->max_phots_init;
  nuser_var = pmcb->pmy_mc->nuser_var;
  // SWD: should these be set or controlled by flags?
  user = &(rp[iuserp]);
  polten = &(rp[ipolp]);
  npar = 0;


}

//----------------------------------------------------------------------------------------
//! destructor

Photon::~Photon() {

}

//----------------------------------------------------------------------------------------
//! \fn void Photon::PrintPhoton(std::stringstream msg, int ip)
//! \brief print key photon properites with message

void Photon::PrintPhoton(const std::string &msg, int ip) const {
  std::cout << "----------------------------" << std::endl;
  std::cout << "** " << msg << " **" << std::endl;
  PrintPhoton(ip);
}

//----------------------------------------------------------------------------------------
//! \fn void Photon::PrintPhoton(int ip)
//! \brief print key photon properites, primarily for debugging

void Photon::PrintPhoton(int ip) const {

  Real keverg = 1.602176634e-9;
  std::cout << "----------------------------" << std::endl
            << "Energy [erg] [kev], weight: " << ep[ip] << " " << ep[ip]/keverg << " "
	    << wp[ip] << std::endl
            << "i: " << i1p[ip] << " " << i2p[ip] << " " << i3p[ip] <<std::endl
            << "x: " << x0p[ip] << " " << x1p[ip] << " " << x2p[ip] << " " << x3p[ip]
            << std::endl
            << "k: " << k0p[ip] << " " << k1p[ip] << " " << k2p[ip] << " " << k3p[ip]
            << std::endl;
#if MC_VERLET_DK
  if (general_pusher_flag) {
    std::cout << "dk: " << dk0p[ip] << " " << dk1p[ip] << " " << dk2p[ip] << " "
              << dk3p[ip] << std::endl;
  }
#endif
  if (IsPolarized(polarized)) {
    std:: cout << "stokes: " << sip[ip] << " " << sqp[ip] << " " << sup[ip]
              << " " << svp[ip] << std::endl;
    if (general_pusher_flag) {
      std:: cout << "pol tensor: ";
        for (int k = 0; k < 4; k++) {
          for (int l = 0; l < 4; l++) {
            std:: cout << Tensor(ip, k, l) << " ";
          }
          std::cout << std::endl;
        }
    }
  }
  std::cout << "opacity (cgs) [sc] [abs]: " << scp[ip] << " " << acp[ip] << std::endl;
  Real l_cgs = pmy_mcb->l_cgs;
  if (l_cgs != 1.0)
    std::cout << "opacity (dim) [sc] [abs]: " << scp[ip]*l_cgs << " " << acp[ip]*l_cgs << std::endl;
    std::cout << "dt: " << dtp[ip] << " ";
  if (statp[ip] == EVOLVING)
    std::cout << "EVOLVING" << std::endl;
  else if (statp[ip] == ESCAPED)
    std::cout << "ESCAPED" << std::endl;
  else if (statp[ip] == DESTROYED)
    std::cout << "DESTROYED" << std::endl;
  else if (statp[ip] == BUFFERED)
    std::cout << "BUFFERED" << std::endl;
  else
    std::cout << std::endl;
  std::cout << "nscat: " << nscp[ip] << std::endl;
  if (nuser_var > 0)
    std::cout << nuser_var << " user vars:";
  for (int i=0; i<nuser_var; i++)
    std::cout << " " << user[i][ip];
  std::cout << std::endl;
  std::cout << "----------------------------" << std::endl;

}

//----------------------------------------------------------------------------------------
//! \fn void Photon::IsNanPhoton(int ip)
//! \brief check for Nan in photon properties

bool Photon::IsNanPhoton(int ip) {

  if (std::isnan(wp[ip])) return true;
  if (std::isnan(ep[ip])) return true;
  if (std::isnan(x0p[ip])) return true;
  if (std::isnan(x1p[ip])) return true;
  if (std::isnan(x2p[ip])) return true;
  if (std::isnan(x3p[ip])) return true;
  if (std::isnan(k0p[ip])) return true;
  if (std::isnan(k1p[ip])) return true;
  if (std::isnan(k2p[ip])) return true;
  if (std::isnan(k3p[ip])) return true;
  if (IsPolarized(polarized)) {
    if (std::isnan(sip[ip])) return true;
    if (std::isnan(sqp[ip])) return true;
    if (std::isnan(sup[ip])) return true;
    // Only the general pusher carries the tensor.
    if (general_pusher_flag) {
      for (int i = 0; i < 16; ++i) {
        if (std::isnan(polten[i][ip]) || std::isinf(polten[i][ip])) return true;
      }
    }
  }
  if (std::isnan(scp[ip])) return true;
  if (std::isnan(acp[ip])) return true;

  return false;
}

//----------------------------------------------------------------------------------------
//! \fn void Photon::EnsureScratch()
//! \brief keep the shared dk scratch column at least as long as this block's arrays
//
// One column per process, shared by all four dk*p references of every block, so a
// problem generator's dk0p[ip] = 0 lands in bounds for any ip < npar.  Called wherever
// npar can grow: AllocatePhotons and the receive flush.  A no-op with MC_VERLET_DK on.

void Photon::EnsureScratch() {
#if !MC_VERLET_DK
  if (dk_scratch_.size() < static_cast<std::size_t>(npar)) dk_scratch_.resize(npar);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn bool Photon::IsNanTransport(int ip) const
//! \brief NaN in the weight, position or wavevector

bool Photon::IsNanTransport(int ip) const {
  return std::isnan(wp[ip]) || std::isnan(x0p[ip]) || std::isnan(x1p[ip]) ||
         std::isnan(x2p[ip]) || std::isnan(x3p[ip]) || std::isnan(k0p[ip]) ||
         std::isnan(k1p[ip]) || std::isnan(k2p[ip]) || std::isnan(k3p[ip]);
}

//--------------------------------------------------------------------------------------
//! \fn void Photon::PackAll(std::vector<int> &ib, std::vector<Real> &rb) const
//! \brief append every resident photon to the two streams, in ParticleBuffer's order

void Photon::PackAll(std::vector<int> &ib, std::vector<Real> &rb) const {
  if (ncplx > 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Photon::PackAll" << std::endl
        << "complex photon properties are not packed" << std::endl;
    ATHENA_ERROR(msg);
  }
  ib.reserve(ib.size() + static_cast<std::size_t>(npar)*nint);
  rb.reserve(rb.size() + static_cast<std::size_t>(npar)*(nreal + naux));
  for (int k=0; k<npar; ++k) {
    for (int j=0; j<nint; ++j) ib.push_back(intprop[j][k]);
    for (int j=0; j<nreal; ++j) rb.push_back(rp[j][k]);
    for (int j=0; j<naux; ++j) rb.push_back(aux[j][k]);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void Photon::UnpackAll(const int *ib, const Real *rb, int n)
//! \brief append n photons read from the two streams PackAll writes

void Photon::UnpackAll(const int *ib, const Real *rb, int n) {
  if (n <= 0) return;
  const int k0 = npar;
  Resize(k0 + n);
  for (int k=k0; k<npar; ++k) {
    for (int j=0; j<nint; ++j) intprop[j][k] = *ib++;
    for (int j=0; j<nreal; ++j) rp[j][k] = *rb++;
    for (int j=0; j<naux; ++j) aux[j][k] = *rb++;
  }
  EnsureScratch();
}

//--------------------------------------------------------------------------------------
//! \fn void Photon::AllocatePhotons(int nphot)
//! \brief Allocates photons

// SWD: Temporary --> converts protected function to public :(
void Photon::AllocatePhotons(int nphot) {
  const int nold = npar;
  // Call Resize function
  Resize(nphot);
  EnsureScratch();

  // Zero the free-flight step counter on the slots just claimed for new photons.  Resize
  // only value-initializes when the underlying vector actually grows, and it does not:
  // RemoveOneParticle swaps the last photon down into the freed slot and decrements the
  // count without shrinking the storage, so a slot handed out here has usually held a
  // photon before and still carries its count.  Left alone, a new photon would inherit
  // it and could be retired by capmove before travelling anywhere.
  for (int ip = nold; ip < npar; ++ip) nmvp[ip] = 0;
}

//----------------------------------------------------------------------------------------
//! \fn void Photon::PolarizationToTetrad(std::complex<Real> ttet[4][4], Real ecov[4][4],
//!                                       const int ip)
//!
//! \brief transform complex tensor from coordinate frame to tetrad frame

void Photon::PolarizationToTetrad(std::complex<Real> ttet[4][4], Real ecov[4][4],
                                  const int ip) {

  std::complex<Real> n[4][4];
  LoadTensor(ip, n);

  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      ttet[i][j] = std::complex<Real>(0.,0.);

  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 4; k++)
        for (int l = 0; l < 4; l++) {
          ttet[i][j] += n[k][l] * ecov[i][k] * ecov[j][l];
        }

}

//----------------------------------------------------------------------------------------
//! \fn void PolarizationToCoord(std::complex<Real> ttet[4][4], Real econ[4][4],
//!                              const int ip)
//!
//! \brief transform complex tensor from tetrad frame to coordinate frame

void Photon::PolarizationToCoord(std::complex<Real> ttet[4][4], Real econ[4][4],
                                 const int ip) {

  std::complex<Real> n[4][4];
  for(int i = 0; i < NCOORD; i++)
    for(int j = 0; j < NCOORD; j++)
      n[i][j] = std::complex<Real>(0.,0.);

  for(int i = 0; i < NCOORD; i++)
    for(int j = 0; j < NCOORD; j++)
      for(int k = 0; k < NCOORD; k++)
        for(int l = 0; l < NCOORD; l++) {
          n[i][j] += ttet[k][l] * econ[k][i] * econ[l][j];
        }

  StoreTensor(ip, n);
}

//----------------------------------------------------------------------------------------
//! Hermitian storage of the coherency tensor.  See the declaration in photon.hpp.
//
// The transport in GeneralPusher::AdvanceStep preserves Hermiticity exactly in floating
// point (each lower-triangle term is the conjugate of the matching upper-triangle term,
// operation for operation). The frame transforms above sum the same terms in a different
// order for (i,j) and (j,i)

namespace {
// index of the pair (i,j), i < j, in the order (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)
const int kPairIndex[4][4] = {{-1, 0, 1, 2}, {0, -1, 3, 4}, {1, 3, -1, 5}, {2, 4, 5, -1}};
} // namespace

int Photon::TensorSlot(int i, int j, bool imag) {
  if (i == j) return i;
  return 4 + 2*kPairIndex[i][j] + (imag ? 1 : 0);
}

void Photon::LoadTensor(int ip, std::complex<Real> n[4][4]) const {
  for (int i = 0; i < 4; i++) {
    n[i][i] = std::complex<Real>(polten[i][ip], 0.);
    for (int j = i+1; j < 4; j++) {
      const int p = 4 + 2*kPairIndex[i][j];
      n[i][j] = std::complex<Real>(polten[p][ip], polten[p+1][ip]);
      n[j][i] = std::conj(n[i][j]);
    }
  }
}

void Photon::StoreTensor(int ip, const std::complex<Real> n[4][4]) {
  for (int i = 0; i < 4; i++) {
    polten[i][ip] = n[i][i].real();
    for (int j = i+1; j < 4; j++) {
      const int p = 4 + 2*kPairIndex[i][j];
      polten[p][ip] = n[i][j].real();
      polten[p+1][ip] = n[i][j].imag();
    }
  }
}

std::complex<Real> Photon::Tensor(int ip, int i, int j) const {
  if (i == j) return std::complex<Real>(polten[i][ip], 0.);
  const int p = 4 + 2*kPairIndex[i][j];
  const std::complex<Real> upper(polten[p][ip], polten[p+1][ip]);
  return (i < j) ? upper : std::conj(upper);
}

//--------------------------------------------------------------------------------------
//! \fn Photon::Initialize(MonteCarloBlock *pmcb, ParameterInput *pin)
//! \brief initializes the Photon class.
// SWD: Change name to distinguish with InitializePhoton?

void Photon::Initialize(MonteCarlo *pmc, ParameterInput *pin) {

  // Initialize first the parent class.
  Particles::Initialize(pmc->pmy_mesh, pin);

  if (initialized) return;

  // Add particle ID and status flags, other int parameters.
  inscp = AddIntProperty("nscp");
  istatp = AddIntProperty("statp");
  ityp = AddIntProperty("type");
  inmvp = AddIntProperty("nmv");

  // Add photon position.
  ix0p = AddRealProperty("x0");
  ix1p = AddRealProperty("x1");
  ix2p = AddRealProperty("x2");
  ix3p = AddRealProperty("x3");

  // Add photon momentum.
  ik0p = AddRealProperty("k0");
  ik1p = AddRealProperty("k1");
  ik2p = AddRealProperty("k2");
  ik3p = AddRealProperty("k3");

  if (pmc->general_pusher_flag) {
    general_pusher_flag = true;
#if MC_VERLET_DK
    // Add change in photon momentum (Verlet only; see MC_VERLET_DK in photon.hpp).
    idk0p = AddRealProperty("dk0");
    idk1p = AddRealProperty("dk1");
    idk2p = AddRealProperty("dk2");
    idk3p = AddRealProperty("dk3");
#endif
  }

  // Add energy, weight, and opacities.
  //
  // k0p holds the photon energy in whatever frame the photon is currently expressed in,
  // which is exactly what ep means, so the two are the same column rather than two
  // arrays kept in step by hand.  ep survives as a name because the non-relativistic
  // physics reads far better in terms of an energy than a time component, but writing
  // either one now writes both.  See GetFourVector/SetFourVector in montecarloblock.cpp
  // for the storage convention this relies on.
  iep = ik0p;
  iwp = AddRealProperty("wp");
  iscp = AddRealProperty("scp");
  iacp = AddRealProperty("acp");

  // Add time remaining parameter
  idtp = AddRealProperty("dtp");

  if (IsPolarized(pmc->polarized)) {
    polarized = pmc->polarized;
    // Add stokes vectors
    isip = AddRealProperty("sip");
    isqp = AddRealProperty("sqp");
    isup = AddRealProperty("sup");
    isvp = AddRealProperty("svp");
    if (general_pusher_flag) {
      // The coherency tensor, as sixteen consecutive real columns (Hermitian storage; the
      // layout is TensorSlot's).  Consecutive because polten is a pointer to the first.
      for (int n = 0; n < 16; n++) {
        int idummy = AddRealProperty("pol"+std::to_string(n));
        if (n == 0) ipolp = idummy;
      }
    }
  }

  // Add particle position indices.
  ii1p = AddIntProperty("i1p");
  ii2p = AddIntProperty("i2p");
  ii3p = AddIntProperty("i3p");

  // Add nuser variables
  for (int i=0; i<pmc->nuser_var; i++) {
    int idummy = AddRealProperty("user"+std::to_string(i));
    if (i==0)
      iuserp = idummy;
  }

#ifdef MPI_PARALLEL
  // Get my MPI communicator.
  MPI_Comm_dup(MPI_COMM_WORLD, &my_comm);
#endif

  initialized = true;
}


//--------------------------------------------------------------------------------------
//! \fn void Photon::SendToNeighbors()
//! \brief sends photons outside boundary to the buffers of neighboring meshblocks.

void Photon::SendToNeighbors() {
  HandoffTimer timer(pmy_mcb);
  const int IS = pmy_block->is;
  const int IE = pmy_block->ie;
  const int JS = pmy_block->js;
  const int JE = pmy_block->je;
  const int KS = pmy_block->ks;
  const int KE = pmy_block->ke;

  nbuf = nloc = nadj = nper = nnper = 0, nmpi = 0;
  for (int k = npar-1; k >=0; ) {
    if (statp[k] != BUFFERED) {
      --k;
      continue;
    }
    nbuf++;
    // Find which boundary photon has passed beyond
    int ox1 = CheckSide(i1p[k], IS, IE),
        ox2 = CheckSide(i2p[k], JS, JE),
        ox3 = CheckSide(i3p[k], KS, KE);
    if (ox1 == 0 && ox2 == 0 && ox3 == 0) {
      std::cout << "Warning: photon status is BUFFERED but not outside of boundary,"
                << " photon marked destroyed" << std::endl;
      statp[k] = DESTROYED;
      --k;
      continue;
    }

    // Apply periodic boundary conditions and find the mesh coordinates.
    ApplyPeriodicBoundary(x1p[k], x2p[k], x3p[k], k);
    //printf("%d %d %g %g %g\n",Globals::my_rank,k,x1p[k],x2p[k],x3p[k]);
    // Find the neighbor block to send it to.
    if (!active1_) ox1 = 0;
    if (!active2_) ox2 = 0;
    if (!active3_) ox3 = 0;
    Neighbor *pn = FindTargetNeighbor(ox1, ox2, ox3, i1p[k], i2p[k], i3p[k]);
    NeighborBlock *pnb = pn->pnb;
    if (pnb == nullptr) {
      PrintPhoton("pnb == nullptr",k);
      std::cout << ox1 << " " << ox2 << " " << ox3 << " " << i1p[k] << " "
                << i2p[k] << " " << i3p[k] << std::endl;
      MCCoord *pco = pmy_mcb->pcoord;
      //printf("%d %d %g %g %g %g %g %g\n",Globals::my_rank,k,
      //       pco->x1f(i1p[k]),pco->x1f(i1p[k]+1),
      //       pco->x2f(i2p[k]),pco->x2f(i2p[k]+1),
      //       pco->x3f(i3p[k]),pco->x3f(i3p[k]+1));

      RemoveOneParticle(k);
      --k;
      std::cout << "[SendToNeighbors] Warning: pnb==nullptr." << std::endl;
      continue;
    }

    // Determine which particle buffer to use.
    ParticleBuffer *ppb = NULL;
    if (pnb->snb.rank == Globals::my_rank) {
      // No need to send if back to the same block.
      if (pnb->snb.gid == pmy_block->gid) {
        GetPositionIndices(k,k);
        --k;
        nloc++;
        continue;
      }
      // Use the target receive buffer.
      ppb = &pn->pmb->pmy_mcb->pphot->recv_[pnb->targetid];
      nadj++;
    } else {
#ifdef MPI_PARALLEL
      nmpi++;
      // Use the send buffer.
      ppb = &send_[pnb->bufid];
#endif
    }

    // Check the buffer size.
    if (ppb->npar >= ppb->nparmax)
      ppb->Reallocate((ppb->nparmax > 0) ? 2 * ppb->nparmax : 1);

    // Copy the properties of the particle to the buffer.
    int *pi = ppb->ibuf + ParticleBuffer::nint * ppb->npar;
    for (int j = 0; j < nint; ++j)
      *pi++ = intprop[j][k];
    Real *pr(ppb->rbuf + ParticleBuffer::nreal * ppb->npar);
    // rp1 deliberately not sent: it is the dust integrator's second register and no
    // Monte Carlo code path reads it.  See Particles::Resize.
    for (int j = 0; j < nreal; ++j)
      *pr++ = rp[j][k];
    for (int j = 0; j < naux; ++j)
      *pr++ = aux[j][k];
    // copy complex properties (none at present: the coherency tensor travels as reals)
    if (ParticleBuffer::ncplx > 0) {
      std::complex<Real> *pc(ppb->cbuf + ParticleBuffer::ncplx * ppb->npar);
      for (int j = 0; j < ncplx; ++j) {
        *pc++ = cplxprop[j][k];
      }
    }
    ++ppb->npar;
    // Pop the particle from the current MeshBlock.
    RemoveOneParticle(k);
    --k;
  }

  // Nothing was handed off and no neighbor is on another rank, so there is nobody to
  // notify: same-rank neighbors already sit at "completed" (see Photon::ClearBoundary),
  // and walking the neighbor list to tell each of them "nothing for you" is exactly the
  // per-block, per-round cost that dominates a run with many small blocks.
  const bool rank_exchange = (pmy_mcb->pmy_mc->pexch != nullptr
                              && pmy_mcb->pmy_mc->pexch->Active());
  if (nbuf == 0 && (rank_exchange || !has_offrank_neighbor_)) return;

  // Send to neighbor processes and update boundary status.
  for (int i = 0; i < pbval_->nneighbor; ++i) {
    NeighborBlock& nb = pbval_->neighbor[i];
    int dst = nb.snb.rank;
    if (dst == Globals::my_rank) {
      Photon *ppar = pmy_mesh->FindMeshBlock(nb.snb.gid)->pmy_mcb->pphot;
      if (ppar->recv_[nb.targetid].npar > 0) {
        ppar->bstatus_[nb.targetid] = BoundaryStatus::arrived;
        // tell the receive sweep this block has something waiting for it
        ppar->has_incoming_ = true;
      } else {
        ppar->bstatus_[nb.targetid] = BoundaryStatus::completed;
      }
    } else if (pmy_mcb->pmy_mc->pexch != nullptr
               && pmy_mcb->pmy_mc->pexch->Active()) {
#ifdef MPI_PARALLEL
      // Hand these to the rank exchange instead of opening a conversation with this one
      // neighbor block.  Everything leaving for that rank, from every block here, goes in
      // one message at the end of the round.  Nothing is sent when there is nothing to
      // send: the receiver no longer waits on a per-link count, so silence is the signal.
      ParticleBuffer& send = send_[nb.bufid];
      if (send.npar > 0) {
        pmy_mcb->pmy_mc->pexch->Stage(nb.snb.rank, nb.snb.lid, nb.targetid,
                                      send.ibuf, send.rbuf, send.cbuf, send.npar);
        // Handed over, so empty the slot: the send sweep runs more than once per round
        // now, and anything still sitting here would be staged a second time.
        send.npar = 0;
      }
#endif
    } else {
#ifdef MPI_PARALLEL
      ParticleBuffer& send = send_[nb.bufid];
      int npsend = send.npar;
      MPI_Send(&npsend, 1, MPI_INT, nb.snb.rank, send.tag, my_comm);
      //if (npsend > 0)
      //printf("send: %d %d %d %d %d %d %g %g\n",Globals::my_rank,nb.snb.rank,pmy_block->lid,nb.targetid,send.tag+1,npsend,send.rbuf[0],send.rbuf[(npsend-1)*ParticleBuffer::nreal]);
      if (npsend > 0) {
        MPI_Request req = MPI_REQUEST_NULL;
	/*
        MPI_Isend(send.ibuf, npsend * ParticleBuffer::nint, MPI_INT,
                  dst, send.tag + 1, my_comm, &req);
        MPI_Request_free(&req);
        MPI_Isend(send.rbuf, npsend * ParticleBuffer::nreal, MPI_ATHENA_REAL,
                  dst, send.tag + 2, my_comm, &req);
        MPI_Request_free(&req);
	*/
        MPI_Isend(send.rbuf, npsend * ParticleBuffer::nreal, MPI_ATHENA_REAL,
                  dst, send.tag + 1, my_comm, &req);
        MPI_Request_free(&req);
        MPI_Isend(send.ibuf, npsend * ParticleBuffer::nint, MPI_INT,
                  dst, send.tag + 2, my_comm, &req);
        MPI_Request_free(&req);
        // Send complex properties
        if (ParticleBuffer::ncplx > 0) {
          MPI_Isend(send.cbuf, npsend * ParticleBuffer::ncplx, MPI_ATHENA_COMPLEX,
                    dst, send.tag + 3, my_comm, &req);
          MPI_Request_free(&req);
        }
      }
#endif
    }
  }
  //if (nbuf > 0)
  //   printf("send %d %d %d %d %d %d %d %d\n",Globals::my_rank,pmy_block->gid,nbuf,nloc,nadj,
  //         nmpi,nper,nnper);
}

//--------------------------------------------------------------------------------------
//! \fn void Photons::ApplyPeriodicBoundary(Real &x1, Real &x2, Real &x3, int k)
//! \brief applies periodic boundary conditions to photon k and returns its updated mesh
//!        coordinates (x1,x2,x3).

void Photon::ApplyPeriodicBoundary(Real &x1, Real &x2, Real &x3, int k) {
  bool flag = false;
  RegionSize& mesh_size = pmy_mesh->mesh_size;
  //MCCoord *pcoord = pmy_mcb->pcoord;
  Real l1cgs = 1., l2cgs = 1., l3cgs = 1.;

  Real frac = 1.0e-8;

  // Apply periodic boundary conditions in X1.
  if (x1 <= mesh_size.x1min) {
    // Inner x1
    Real dx = mesh_size.x1min - x1;
    x1 = mesh_size.x1max - dx;
    //x1 = mesh_size.x1max * l1cgs * (1.-frac);
    flag = true;
  } else if (x1 >= mesh_size.x1max) {
    // Outer x1
    Real dx = x1 - mesh_size.x1max;
    x1 = mesh_size.x1min + dx;
    //x1 = mesh_size.x1min * l1cgs * (1.+frac);
    flag = true;
  }

  // Apply periodic boundary conditions in X2.
  if (x2 <= mesh_size.x2min) {
    // Inner x2
    Real dx = mesh_size.x2min - x2;
    x2 = mesh_size.x2max - dx;
    //x2 = mesh_size.x2max * l2cgs * (1.-frac);
    flag = true;
  } else if (x2 >= mesh_size.x2max) {
    // Outer x2
    Real dx = x2 - mesh_size.x2max;
    x2 = mesh_size.x2min + dx;
    //x2 = mesh_size.x2min * l2cgs * (1.+frac);
    flag = true;
  }

  // Apply periodic boundary conditions in X3.
  if (x3 <= mesh_size.x3min) {
    // Inner x3
    Real dx = mesh_size.x3min - x3;
    x3 = mesh_size.x3max - dx;
    //x3 = mesh_size.x3max * l3cgs * (1.-frac);
    flag = true;
  } else if (x3 >= mesh_size.x3max) {
    // Outer x3
    Real dx = x3 - mesh_size.x3max;
    x3 = mesh_size.x3min + dx;
    //x3 = mesh_size.x3min * l3cgs * (1.+frac);
    flag = true;
  }

  if (flag) {
    nper++;
  } else {
    nnper++;
  }
}

//--------------------------------------------------------------------------------------
//! \fn bool Photon::ReceiveFromNeighbors()
//! \brief receives particles from neighboring meshblocks and returns a flag indicating
//!        if all receives are completed.

bool Photon::ReceiveFromNeighbors() {
  HandoffTimer timer(pmy_mcb);
  bool flag = true;

  for (int i = 0; i < pbval_->nneighbor; ++i) {
    NeighborBlock& nb = pbval_->neighbor[i];
    enum BoundaryStatus& bstatus = bstatus_[nb.bufid];

#ifdef MPI_PARALLEL
    // Communicate with neighbor processes.
    int nb_rank = nb.snb.rank;

    if (nb_rank != Globals::my_rank && bstatus == BoundaryStatus::waiting) {

      ParticleBuffer& recv = recv_[nb.bufid];
      if (!recv.mpi_active) {
        // Get the number of incoming particles.
        MPI_Irecv(&recv.npar, 1, MPI_INT, nb_rank, recv.tag, my_comm, &recv.reqn);
	//printf("r: %d %d %d %d %d %d %d\n",Globals::my_rank,nb_rank,nb.snb.lid,nb.bufid,recv_[nb.bufid].tag,recv.npar,recv.flagn);
        recv.mpi_active = true;
      }
      if (!recv.flagn) {
        MPI_Test(&recv.reqn, &recv.flagn, MPI_STATUS_IGNORE);
        if (recv.flagn) {
          if (recv.npar > 0) {
            // Check the buffer size.
            int nprecv = recv.npar;
            if (nprecv > recv.nparmax) {
	      //printf("buf res: %d %d %d %d %d %d %d\n",Globals::my_rank,nb_rank,nb.snb.lid,nb.bufid,recv_[nb.bufid].tag+1,recv.npar,recv.nparmax);
              recv.npar = 0;
	      //recv.Reallocate(2 * nprecv - recv.nparmax +50);
              recv.Reallocate(2 * nprecv - recv.nparmax);
              recv.npar = nprecv;
            }
            // Receive data from the neighbor.
	    /*
            MPI_Irecv(recv.ibuf, recv.npar * ParticleBuffer::nint, MPI_INT,
                      nb_rank, recv.tag + 1, my_comm, &recv.reqi);
            MPI_Irecv(recv.rbuf, recv.npar * ParticleBuffer::nreal, MPI_ATHENA_REAL,
                      nb_rank, recv.tag + 2, my_comm, &recv.reqr);
	    */
	    int ierr;
	    ierr = MPI_Irecv(recv.rbuf, recv.npar * ParticleBuffer::nreal, MPI_ATHENA_REAL,
                      nb_rank, recv.tag + 1, my_comm, &recv.reqr);

	    /*char err_buffer[MPI_MAX_ERROR_STRING];
	    int resultlen;
	    MPI_Error_string(ierr,err_buffer,&resultlen);
	    printf(err_buffer);
	    printf("\n");*/
            MPI_Irecv(recv.ibuf, recv.npar * ParticleBuffer::nint, MPI_INT,
                      nb_rank, recv.tag + 2, my_comm, &recv.reqi);
	    //int test;
	    //MPI_Status stat;
	    //MPI_Request_get_status(recv.reqr,&test,&stat);
	    //printf("t2: %d %d %d %d %d %d %d %d %d %d\n",Globals::my_rank,nb_rank,nb.snb.lid,nb.bufid,recv.tag+1,pmy_block->lid,test,stat.MPI_SOURCE,stat.MPI_TAG,stat.MPI_ERROR);
            if (ParticleBuffer::ncplx > 0) {
              MPI_Irecv(recv.cbuf, recv.npar * ParticleBuffer::ncplx, MPI_ATHENA_COMPLEX,
                        nb_rank, recv.tag + 3, my_comm, &recv.reqc);
            }
          } else {
            // No incoming particles.
            bstatus = BoundaryStatus::completed;
          }
        }
      }

      if (recv.flagn && recv.npar > 0) {
        if (!recv.flagi)
	  MPI_Test(&recv.reqi, &recv.flagi, MPI_STATUS_IGNORE);
        if (!recv.flagr) {
          MPI_Test(&recv.reqr, &recv.flagr, MPI_STATUS_IGNORE);
	}
        if (general_pusher_flag && IsPolarized(polarized)) {
          if (!recv.flagc)
            MPI_Test(&recv.reqc, &recv.flagc, MPI_STATUS_IGNORE);
          if (recv.flagi && recv.flagr && recv.flagc) {
            bstatus = BoundaryStatus::arrived;
	  }
        } else {
          if (recv.flagi && recv.flagr) {
            bstatus = BoundaryStatus::arrived;
	    //printf("g: %d %d %d %d %d\n",Globals::my_rank,nb_rank,nb.snb.lid,nb.bufid,recv_[nb.bufid].tag+1);
	  } else {
	    // SWD debug
	    //printf("bad: %d %d %d\n",recv.flagn,recv.flagi,recv.flagr);
	    //printf("%g %g %d %d\n",recv_[nb.bufid].rbuf[0],recv_[nb.bufid].rbuf[(recv_[nb.bufid].npar-1)*ParticleBuffer::nreal],recv_[nb.bufid].npar,recv_[nb.bufid].nparmax);
	    //printf("b: %d %d %d %d %d\n",Globals::my_rank,nb_rank,nb.snb.lid,nb.bufid,recv_[nb.bufid].tag+1);
	    //printf("n: %d\n",recv.flagn);
	    MPI_Wait(&recv.reqr, MPI_STATUS_IGNORE);
	    //printf("wait %d\n",Globals::my_rank);
	  }
        }
      }
    }
#endif
    switch (bstatus) {
      case BoundaryStatus::completed:
        break;

      case BoundaryStatus::waiting:
	//printf("w: %d %d %d %d %d %d\n",Globals::my_rank,nb_rank,nb.snb.lid,nb.bufid,recv_[nb.bufid].tag+1,recv_[nb.bufid].npar);
        flag = false;
        break;

      case BoundaryStatus::arrived:
        ParticleBuffer& recv = recv_[nb.bufid];
        int nparold = npar;
        FlushReceiveBuffer(recv);
        EnsureScratch();
        // Update Photon position indices
        GetPositionIndices(nparold,npar-1);
        //        printf("recv %d %d %d\n",Globals::my_rank,nparold,npar-1);
        bstatus = BoundaryStatus::completed;
        break;
    }
  }

  return flag;
}

//--------------------------------------------------------------------------------------
//! \fn int Photon::PropertyCountInt()
//! \brief per-photon property counts, exposed for MCRankExchange's buffer arithmetic

int Photon::PropertyCountInt()  { return ParticleBuffer::nint; }
int Photon::PropertyCountReal() { return ParticleBuffer::nreal; }
int Photon::PropertyCountCplx() { return ParticleBuffer::ncplx; }

//--------------------------------------------------------------------------------------
//! \fn void Photon::CollectPeerRanks(std::vector<bool> &seen) const
//! \brief mark every rank this block has a neighbor on

void Photon::CollectPeerRanks(std::vector<bool> &seen) const {
  for (int i = 0; i < pbval_->nneighbor; ++i)
    seen[pbval_->neighbor[i].snb.rank] = true;
}

//--------------------------------------------------------------------------------------
//! \fn void Photon::AcceptPhotons(int bufid, ...)
//! \brief fill one receive slot from the rank exchange
//!
//! Leaves the buffer in exactly the state an off-rank MPI receive used to leave it in, so
//! FlushReceiveBuffer downstream cannot tell the two apart.

void Photon::AcceptPhotons(int bufid, const int *ib, const Real *rb,
                           const std::complex<Real> *cb, int npar) {
  HandoffTimer timer(pmy_mcb);
  if (npar <= 0) return;
  ParticleBuffer& recv = recv_[bufid];
  // Append rather than overwrite.  Several rounds of local transport can happen before
  // the ranks exchange, so the same neighbor link may contribute more than once to a
  // single message, and the second batch must not land on top of the first.
  const int nold = recv.npar;
  if (nold + npar > recv.nparmax) recv.Reallocate(nold + npar);
  const int ni = PropertyCountInt(), nr = PropertyCountReal(), nc = PropertyCountCplx();
  for (int n = 0; n < npar*ni; ++n) recv.ibuf[nold*ni + n] = ib[n];
  for (int n = 0; n < npar*nr; ++n) recv.rbuf[nold*nr + n] = rb[n];
  if (nc > 0 && cb != nullptr) {
    for (int n = 0; n < npar*nc; ++n) recv.cbuf[nold*nc + n] = cb[n];
  }
  recv.npar = nold + npar;
  bstatus_[bufid] = BoundaryStatus::arrived;
  has_incoming_ = true;
}

//--------------------------------------------------------------------------------------
//! \fn void Photon::SetOffRankNeighborFlag()
//! \brief record whether any neighbor of this block lives on another rank
//!
//! Fixed for the life of the mesh, so it is worked out once after LinkNeighbors rather
//! than rediscovered every transfer round.

void Photon::SetOffRankNeighborFlag() {
  has_offrank_neighbor_ = false;
#ifdef MPI_PARALLEL
  for (int i = 0; i < pbval_->nneighbor; ++i) {
    if (pbval_->neighbor[i].snb.rank != Globals::my_rank) {
      has_offrank_neighbor_ = true;
      return;
    }
  }
#endif
}

//--------------------------------------------------------------------------------------
//! \fn void Photon::ClearBoundary()
//! \brief reset the boundary state between transfer rounds
//!
//! See the note in photon.hpp: same-rank neighbors start "completed" rather than
//! "waiting", because a same-rank hand-off is a direct write during the send sweep and
//! there is nothing to poll for.  That is what lets a block with no photons and no
//! delivery be skipped entirely, instead of having to report "nothing for you" to each
//! of its neighbors every round.

void Photon::ClearBoundary() {
  // With the rank exchange running there is nothing to poll for on either kind of
  // neighbor: same-rank hand-offs are direct writes, and off-rank photons are delivered
  // by MCRankExchange before the receive sweep.  A slot only becomes "arrived" when
  // something was actually put in it.
  const bool rank_exchange = (pmy_mcb->pmy_mc->pexch != nullptr
                              && pmy_mcb->pmy_mc->pexch->Active());
  for (int i = 0; i < pbval_->nneighbor; ++i) {
    NeighborBlock& nb = pbval_->neighbor[i];
    if (nb.snb.rank == Globals::my_rank || rank_exchange) {
      bstatus_[nb.bufid] = BoundaryStatus::completed;
#ifdef MPI_PARALLEL
      if (nb.snb.rank != Globals::my_rank) send_[nb.bufid].npar = 0;
#endif
    } else {
      bstatus_[nb.bufid] = BoundaryStatus::waiting;
#ifdef MPI_PARALLEL
      ParticleBuffer& recv = recv_[nb.bufid];
      recv.mpi_active = false;
      recv.flagn = recv.flagi = recv.flagr = recv.flagc = 0;
      recv.reqn = recv.reqi = recv.reqr = recv.reqc = MPI_REQUEST_NULL;
      send_[nb.bufid].npar = 0;
#endif
    }
  }
  has_incoming_ = false;
}

//--------------------------------------------------------------------------------------
//! \fn void Photon::GetPositionIndices(int ibegin, int iend)
//! \brief finds the position indices of each particle with respect to the local grid.

void Photon::GetPositionIndices(int ibegin, int iend) {

  Real xi1, xi2, xi3;
  int is = pmy_mcb->is, ie = pmy_mcb->ie;
  int js = pmy_mcb->js, je = pmy_mcb->je;
  int ks = pmy_mcb->ks, ke = pmy_mcb->ke;

  // SWD: set these somewhere else.
  Real l1cgs = 1., l2cgs = 1., l3cgs = 1.;
  //l1cgs = pmy_mcb->l_cgs;
  //if ( (COORDINATE_SYSTEM == "cartesian") || (COORDINATE_SYSTEM == "minkowski") ||
  //     (COORDINATE_SYSTEM == "gr_user") ) {
  //  l2cgs *= pmy_mcb->l_cgs;
  //  l3cgs *= pmy_mcb->l_cgs;
  //}

  for (int k = ibegin; k <= iend; ++k) {
    // Convert to the index space.
    pmy_block->pcoord->MeshCoordsToIndices(x1p[k]/l1cgs, x2p[k]/l2cgs, x3p[k]/l3cgs,
                                           xi1, xi2, xi3);

    i1p[k] = static_cast<int>(xi1);
    i2p[k] = static_cast<int>(xi2);
    i3p[k] = static_cast<int>(xi3);
    if ((i1p[k] < is-1) || (i1p[k] > ie+1)) {
      //printf("1: %d %d %d %g %g %g %g\n",i1p[k],is,ie,xi1,x1p[k]/l1cgs,pmy_block->block_size.x1min,pmy_block->block_size.x1max);
      PrintPhoton("Warning: [GetPostionIndicies], Photon not on block, destroyed",k);
      statp[k] = DESTROYED;
      continue;
    }
    if ((i2p[k] < js-1) || (i2p[k] > je+1)) {
      //printf("2: %d %d %d %g %g %g %g\n",i2p[k],js,je,xi2,x2p[k]/l2cgs,pmy_block->block_size.x2min,pmy_block->block_size.x2max);
      PrintPhoton("Warning: [GetPostionIndicies], Photon not on block, destroyed",k);
      statp[k] = DESTROYED;
      continue;
    }
    if ((i3p[k] < ks-1) || (i3p[k] > ke+1)) {
      //printf("3: %d %d %d %g %g %g %g %g %g\n",i3p[k],ks,ke,xi3,x3p[k]/l3cgs,pmy_block->block_size.x3min,
      //       pmy_block->block_size.x3max,x2p[k],k3p[k]/ep[k]);
      PrintPhoton("Warning: [GetPostionIndicies], Photon not on block, destroyed",k);
      statp[k] = DESTROYED;
      continue;
    }
    // MeshCoordsToIndicies can fail for refined grids so we make some checks

    // First check to make cell index is correct
    MCCoord *pco = pmy_mcb->pcoord;
    while (x1p[k] > pco->x1f(i1p[k]+1)) {
      i1p[k]++;
      if (i1p[k] > ie+1) {
        break;
      }
    }
    while (x1p[k] < pco->x1f(i1p[k])) {
      i1p[k]--;
      if (i1p[k] < is-1) {
        break;
      }
    }
    while (x2p[k] > pco->x2f(i2p[k]+1)) {
      i2p[k]++;
      if (i2p[k] > je+1) {
        break;
      }
    }
    while (x2p[k] < pco->x2f(i2p[k])) {
      i2p[k]--;
      if (i2p[k] < js-1) {
        break;
      }
    }
    while (x3p[k] > pco->x3f(i3p[k]+1)) {
      i3p[k]++;
      if (i3p[k] > ke+1) {
        break;
      }
    }
    while (x3p[k] < pco->x3f(i3p[k])) {
      i3p[k]--;
      if (i3p[k] < ks-1) {
        break;
      }
    }
    bool on_block = true;
    // Next check to see if sample landed in active cell or adjacent
    if ( (i1p[k] < is-1) || (i1p[k] > ie+1) || (i2p[k] < js-1) || (i2p[k] > je+1)
         || (i3p[k] < ks-1) || (i3p[k] > ke+1) ) {
      on_block = false;
    } else {
      Real tol = 0.01;
      // if sample in adjacent ghost cell, set to active cell if sufficinetly
      // close and update position and position index accordingly
      if (i1p[k] == is-1) {
        Real dxr = (pco->x1f(is)-x1p[k])/(pco->x1f(is+1)-pco->x1f(is));
        if (fabs(dxr) < tol) {
          i1p[k] = is;
          x1p[k] = pco->x1f(is);
        } else {
          on_block = false;
        }
      } else if (i1p[k] == ie+1) {
        Real dxr = (x1p[k]-pco->x1f(ie+1))/(pco->x1f(ie+1)-pco->x1f(ie));
        if (fabs(dxr) < tol) {
          i1p[k] = ie;
          x1p[k] = pco->x1f(ie+1);
        } else {
          on_block = false;
        }
      } else if (i2p[k] == js-1) {
        Real dxr = (pco->x2f(js)-x2p[k])/(pco->x2f(js+1)-pco->x2f(js));
        if (fabs(dxr) < tol) {
          i2p[k] = js;
          x2p[k] = pco->x2f(js);
        } else {
          on_block = false;
        }
      } else if (i2p[k] == je+1) {
        Real dxr = (x2p[k]-pco->x2f(je+1))/(pco->x2f(je+1)-pco->x2f(je));
        if (fabs(dxr) < tol) {
          i2p[k] = je;
          x2p[k] = pco->x2f(je+1);
        } else {
          on_block = false;
        }
      } else if (i3p[k] == ks-1) {
        Real dxr = (pco->x3f(ks)-x3p[k])/(pco->x3f(ks+1)-pco->x3f(ks));
        if (fabs(dxr) < tol) {
          i3p[k] = ks;
          x3p[k] = pco->x3f(ks);
        } else {
          on_block = false;
        }
      } else if (i3p[k] == ke+1) {
        Real dxr = (x3p[k]-pco->x3f(ke+1))/(pco->x3f(ke+1)-pco->x3f(ke));
        if (fabs(dxr) < tol) {
          i3p[k] = ke;
          x3p[k] = pco->x3f(ke+1);
        } else {
          on_block = false;
        }
      }
    } // end of else
    if (on_block)
      statp[k] = EVOLVING;
    else {
      //PrintPhoton("Warning: [GetPostionIndicies], Photon not on block, destroyed",k);
      statp[k] = DESTROYED;
      continue;
    }

    MonteCarloBlock *pmcb = pmy_mcb;
    if (pmcb->boosts) {
      // Shift photon energy to comoving frame
      //Real shift = pmcb->LorentzTransformFrequencyShift(this,k);
      Real shift = pmcb->FrequencyShiftComoving(this,k);
      if (( std::isinf(shift)) || (std::isnan(shift)) ) {
        printf("shift: %d %d %d %g %g %g %g\n",i1p[k],i2p[k],i3p[k],shift,xi1,xi2,xi3);
      }
      ep[k] *= shift;
      // compute opacities in comoving frame
      acp[k] = pmcb->AbsorptionOpacity(pmcb,this,k);
      scp[k] = pmcb->ScatteringOpacity(pmcb,this,k);
      // Shift energy back to Eulerian frame
      ep[k] /= shift;
      // Shift opacities to Eulerian frame
      acp[k] *= shift;
      scp[k] *= shift;
    } else {
      // No distinction between comovinng frame and eulerian frame
      acp[k] = pmcb->AbsorptionOpacity(pmcb,this,k);
      scp[k] = pmcb->ScatteringOpacity(pmcb,this,k);
    }
    if (IsNanPhoton(k)) {
      PrintPhoton("Warning: Nan photon in GetPositionIndicies, destroying",k);
      statp[k] = DESTROYED;
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn int CheckSide(int xi, nx, int xi1, int xi2)
//! \brief returns -1 if xi < xi1, +1 if xi > xi2, or 0 otherwise.

inline int CheckSide(int xi, int xi1, int xi2) {
  if (xi < xi1) return -1;
  if (xi > xi2) return +1;
  return 0;
}

//----------------------------------------------------------------------------------------
//! \fn void Photon::GetFourVector(int ip, bool unit_spatial, Real k[4]) const
//! \brief pack the stored photon wavevector into a genuine contravariant four-vector.
//!
//! k0p always holds the photon energy in the frame the photon is currently expressed in.
//! The spatial components are stored one of two ways: as a dimensional k^i (the general
//! pusher in the coordinate frame), or as a unit propagation direction (everything in the
//! comoving frame, and the legacy pushers in either frame).  In the latter case the true
//! four-vector is (E, E*nhat), which is what these helpers reconstruct.  Pass
//! unit_spatial = true when the spatial components are a unit direction.

void Photon::GetFourVector(int ip, bool unit_spatial, Real k[4]) const {
#ifdef DEBUG
  // ep and k0p are still separate arrays until iep is aliased to ik0p; until then this
  // guards the invariant that they always agree.  Harmless (and always true) afterwards.
  Real eref = ep[ip];
  if (std::fabs(k0p[ip] - eref) > 1.0e-12*std::fabs(eref)) {
    PrintPhoton("ep/k0p invariant violated in GetFourVector", ip);
    std::stringstream msg;
    msg << "### FATAL ERROR in GetFourVector" << std::endl
        << "ep = " << eref << " but k0p = " << k0p[ip] << std::endl;
    ATHENA_ERROR(msg);
  }
#endif
  Real k0 = k0p[ip];
  k[IMC0] = k0;
  Real scale = unit_spatial ? k0 : 1.0;
  k[IMC1] = k1p[ip] * scale;
  k[IMC2] = k2p[ip] * scale;
  k[IMC3] = k3p[ip] * scale;
}

//----------------------------------------------------------------------------------------
//! \fn void Photon::SetFourVector(int ip, bool unit_spatial, const Real k[4])
//! \brief store a four-vector back into the photon, normalizing the spatial part if the
//! stored representation is a unit direction.
//!
//! Reads only from the caller's local array, never from the photon, so it is safe once ep
//! aliases k0p and the write to k0p is therefore also a write to ep.

void Photon::SetFourVector(int ip, bool unit_spatial, const Real k[4]) {
  Real inv = unit_spatial ? 1.0/k[IMC0] : 1.0;
  k0p[ip] = k[IMC0];

  ep[ip] = k[IMC0];
  k1p[ip] = k[IMC1] * inv;
  k2p[ip] = k[IMC2] * inv;
  k3p[ip] = k[IMC3] * inv;
}
