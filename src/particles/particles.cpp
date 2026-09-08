//======================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file particles.cpp
//! \brief implements functions in particle classes

// C++ Standard Libraries
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream> // <<
#include <limits>   // numeric_limits<T>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>  // swap()
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "particles.hpp"

// Class variable initialization
std::vector<std::string> Particles::ipname, Particles::rpname, Particles::cpname;
bool Particles::initialized = false;
int Particles::idmax = 0;
int Particles::nint = 0;
int Particles::nreal = 0;
int Particles::naux = 0;
int Particles::nwork = 0;
int Particles::ncplx = 0;
int Particles::ipid = -1;
int Particles::ixp = -1, Particles::iyp = -1, Particles::izp = -1;
int Particles::ivpx = -1, Particles::ivpy = -1, Particles::ivpz = -1;
int Particles::ixi1 = -1, Particles::ixi2 = -1, Particles::ixi3 = -1;
int Particles::imom1 = -1, Particles::imom2 = -1, Particles::imom3 = -1;
Real Particles::cfl_par = 1;
ParameterInput* Particles::pinput = NULL;
#ifdef MPI_PARALLEL
MPI_Comm Particles::my_comm = MPI_COMM_NULL;
#endif

// Local function prototypes
static int CheckSide(int xi, int xi1, int xi2);

//--------------------------------------------------------------------------------------
//! \fn void Particles::AMRCoarseToFine(MeshBlock* pmbc, MeshBlock* pmbf)
//! \brief load particles from a coarse meshblock to a fine meshblock.

void Particles::AMRCoarseToFine(MeshBlock* pmbc, MeshBlock* pmbf) {
  // Initialization
  Particles *pparc = pmbc->ppar, *pparf = pmbf->ppar;
  const Real x1min = pmbf->block_size.x1min, x1max = pmbf->block_size.x1max;
  const Real x2min = pmbf->block_size.x2min, x2max = pmbf->block_size.x2max;
  const Real x3min = pmbf->block_size.x3min, x3max = pmbf->block_size.x3max;
  const bool active1 = pparc->active1_,
             active2 = pparc->active2_,
             active3 = pparc->active3_;
  const std::vector<Real> &xp(pparc->xp), &yp(pparc->yp), &zp(pparc->zp);
  const Coordinates *pcoord = pmbf->pcoord;

  // Loop over particles in the coarse meshblock.
  int nparf(pparf->npar);
  for (int k = 0; k < pparc->npar; ++k) {
    Real x1, x2, x3;
    pcoord->CartesianToMeshCoords(xp[k], yp[k], zp[k], x1, x2, x3);
    if ((!active1 || (active1 && x1min <= x1 && x1 < x1max)) &&
        (!active2 || (active2 && x2min <= x2 && x2 < x2max)) &&
        (!active3 || (active3 && x3min <= x3 && x3 < x3max))) {
      // Load a particle to the fine meshblock.
      pparf->Resize(nparf + 1);
      for (int j = 0; j < nint; ++j)
        pparf->intprop[j][nparf] = pparc->intprop[j][k];
      for (int j = 0; j < nreal; ++j)
        pparf->rp[j][nparf] = pparc->rp[j][k];
      if (!MONTE_CARLO_ENABLED)
        for (int j = 0; j < nreal; ++j)
          pparf->rp1[j][nparf] = pparc->rp1[j][k];
      for (int j = 0; j < naux; ++j)
        pparf->aux[j][nparf] = pparc->aux[j][k];
      ++nparf;
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::AMRFineToCoarse(MeshBlock* pmbf, MeshBlock* pmbc)
//! \brief load particles from a fine meshblock to a coarse meshblock.

void Particles::AMRFineToCoarse(MeshBlock* pmbf, MeshBlock* pmbc) {
  // Check the capacity.
  Particles *pparf = pmbf->ppar, *pparc = pmbc->ppar;
  int nparf = pparf->npar, nparc = pparc->npar;

  // Load the particles.
  pparc->Resize(nparf + nparc);
  for (int j = 0; j < nint; ++j)
    for (int k = 0; k < nparf; ++k)
      pparc->intprop[j][nparc+k] = pparf->intprop[j][k];
  for (int j = 0; j < nreal; ++j)
    for (int k = 0; k < nparf; ++k)
      pparc->rp[j][nparc+k] = pparf->rp[j][k];
  if (!MONTE_CARLO_ENABLED)
    for (int j = 0; j < nreal; ++j)
      for (int k = 0; k < nparf; ++k)
        pparc->rp1[j][nparc+k] = pparf->rp1[j][k];
  for (int j = 0; j < naux; ++j)
    for (int k = 0; k < nparf; ++k)
      pparc->aux[j][nparc+k] = pparf->aux[j][k];
}

//--------------------------------------------------------------------------------------
//! \fn Particles::Initialize(Mesh *pm, ParameterInput *pin)
//! \brief initializes the class.

void Particles::Initialize(Mesh *pm, ParameterInput *pin) {
  if (initialized) return;

  // Add particle ID.
  ipid = AddIntProperty("id");

  if(!MONTE_CARLO_ENABLED) {
    // Add particle position.
    ixp = AddRealProperty("xp");
    iyp = AddRealProperty("yp");
    izp = AddRealProperty("zp");

    // Add particle velocity.
    ivpx = AddRealProperty("vpx");
    ivpy = AddRealProperty("vpy");
    ivpz = AddRealProperty("vpz");

    // Add particle position indices.
    ixi1 = AddWorkingArray();
    ixi2 = AddWorkingArray();
    ixi3 = AddWorkingArray();

    // Initiate ParticleMesh class.
    ParticleMesh::Initialize(pin);
    imom1 = ParticleMesh::AddMeshAux();
    imom2 = ParticleMesh::AddMeshAux();
    imom3 = ParticleMesh::AddMeshAux();

    // Get the CFL number for particles.
    cfl_par = pin->GetOrAddReal("particles", "cfl_par", pm->cfl_number);

    // Remember the pointer to input parameters.
  }
  pinput = pin;

#ifdef MPI_PARALLEL
  // Get my MPI communicator.
  MPI_Comm_dup(MPI_COMM_WORLD, &my_comm);
#endif

  initialized = true;
}

//--------------------------------------------------------------------------------------
//! \fn Particles::PostInitialize(Mesh *pm, ParameterInput *pin)
//! \brief preprocesses the class after problem generator and before the main loop.

void Particles::PostInitialize(Mesh *pm, ParameterInput *pin) {
  // Set particle IDs.
  ProcessNewParticles(pm);

  // Set position indices.
  for (int b = 0; b < pm->nblocal; ++b)
    pm->my_blocks(b)->ppar->SetPositionIndices();
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::FindDensityOnMesh(Mesh *pm, bool include_momentum)
//! \brief finds the number density of particles on the mesh.
//!
//!   If include_momentum is true, the momentum density field is also computed,
//!   assuming mass of each particle is unity.
//! \note
//!   Postcondition: ppm->weight becomes the density in each cell, and
//!   if include_momentum is true, ppm->meshaux(imom1:imom3,:,:,:)
//!   becomes the momentum density.

void Particles::FindDensityOnMesh(Mesh *pm, bool include_momentum) {
  // Assign particle properties to mesh and send boundary.
  int nblocks(pm->nblocal);
  for (int b = 0; b < nblocks; ++b) {
    const MeshBlock *pmb(pm->my_blocks(b));
    const Particles *ppar(pmb->ppar);
    ParticleMesh *ppm(ppar->ppm);
    ppm->StartReceiving();
    if (include_momentum) {
      std::vector<Real> vp[3] = {
          std::vector<Real> (ppar->npar),
          std::vector<Real> (ppar->npar),
          std::vector<Real> (ppar->npar)};
      std::vector<Real> &vp1(vp[0]), &vp2(vp[1]), &vp3(vp[2]);
      const Coordinates *pcoord = pmb->pcoord;
      for (int k = 0; k < ppar->npar; ++k)
        pcoord->CartesianToMeshCoordsVector(ppar->xp[k], ppar->yp[k], ppar->zp[k],
            ppar->vpx[k], ppar->vpy[k], ppar->vpz[k], vp1[k], vp2[k], vp3[k]);
      ppm->AssignParticlesToMeshAux(vp, 0, imom1, 3);
    } else {
      ppm->AssignParticlesToMeshAux(ppar->rp, 0, ppm->iweight, 0);
    }
    ppm->SendBoundary();
  }

  std::vector<bool> completed(nblocks, false);
  bool pending = true;
  while (pending) {
    pending = false;
    for (int i = 0; i < nblocks; ++i) {
      const MeshBlock *pmb(pm->my_blocks(i));
      Coordinates *pc(pmb->pcoord);
      ParticleMesh *ppm(pmb->ppar->ppm);
      if (!completed[i]) {
        // Finalize boundary communications.
        if ((completed[i] = ppm->ReceiveBoundary())) {
          // Convert to densities.
          const int is = ppm->is, ie = ppm->ie;
          const int js = ppm->js, je = ppm->je;
          const int ks = ppm->ks, ke = ppm->ke;
          if (include_momentum) {
            for (int k = ks; k <= ke; ++k)
              for (int j = js; j <= je; ++j)
                for (int i = is; i <= ie; ++i) {
                  Real vol(pc->GetCellVolume(k,j,i));
                  ppm->weight(k,j,i) /= vol;
                  ppm->meshaux(imom1,k,j,i) /= vol;
                  ppm->meshaux(imom2,k,j,i) /= vol;
                  ppm->meshaux(imom3,k,j,i) /= vol;
                }
          } else {
            for (int k = ks; k <= ke; ++k)
              for (int j = js; j <= je; ++j)
                for (int i = is; i <= ie; ++i)
                  ppm->weight(k,j,i) /= pc->GetCellVolume(k,j,i);
          }
          ppm->ClearBoundary();
        } else {
          pending = true;
        }
      }
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::FindHistoryOutput(Mesh *pm, Real data_sum[], int pos)
//! \brief finds the data sums of history output from particles in my process and assign
//!   them to data_sum beginning at index pos.

void Particles::FindHistoryOutput(Mesh *pm, Real data_sum[], int pos) {
  const int NSUM = NHISTORY - 1;

  // Initiate the summations.
  std::int64_t np = 0;
  std::vector<Real> sum(NSUM, 0.0);

  // Sum over each meshblock.
  Real vp1, vp2, vp3;
  for (int b = 0; b < pm->nblocal; ++b) {
    const MeshBlock *pmb(pm->my_blocks(b));
    const Particles *ppar(pmb->ppar);
    np += ppar->npar;
    const Coordinates *pcoord(pmb->pcoord);
    for (int k = 0; k < ppar->npar; ++k) {
      pcoord->CartesianToMeshCoordsVector(ppar->xp[k], ppar->yp[k], ppar->zp[k],
          ppar->vpx[k], ppar->vpy[k], ppar->vpz[k], vp1, vp2, vp3);
      sum[0] += vp1;
      sum[1] += vp2;
      sum[2] += vp3;
      sum[3] += vp1 * vp1;
      sum[4] += vp2 * vp2;
      sum[5] += vp3 * vp3;
    }
  }

  // Assign the values to output variables.
  data_sum[pos++] = static_cast<Real>(np);
  for (int i = 0; i < NSUM; ++i)
    data_sum[pos++] = sum[i];
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::GetHistoryOutputNames(std::string output_names[])
//! \brief gets the names of the history output variables in history_output_names[].

void Particles::GetHistoryOutputNames(std::string output_names[]) {
  output_names[0] = "np";
  output_names[1] = "vp1";
  output_names[2] = "vp2";
  output_names[3] = "vp3";
  output_names[4] = "vp1^2";
  output_names[5] = "vp2^2";
  output_names[6] = "vp3^2";
}

//--------------------------------------------------------------------------------------
//! \fn int Particles::GetTotalNumber(Mesh *pm)
//! \brief returns total number of particles (from all processes).

int Particles::GetTotalNumber(Mesh *pm) {
  int npartot(0);
  for (int b = 0; b < pm->nblocal; ++b)
    npartot += pm->my_blocks(b)->ppar->npar;
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &npartot, 1, MPI_INT, MPI_SUM, my_comm);
#endif
  return npartot;
}

//--------------------------------------------------------------------------------------
//! \fn Particles::Particles(MeshBlock *pmb, ParameterInput *pin)
//! \brief constructs a Particles instance.

Particles::Particles(MeshBlock *pmb, ParameterInput *pin)
  // Allocate space for particle data.
  : intprop(new std::vector<int> [nint]),
    rp(new std::vector<Real> [nreal]), rp1(new std::vector<Real> [nreal]),
    drp(new std::vector<Real> [nreal]),
    cplxprop(new std::vector<std::complex<Real>> [ncplx]),
    aux(new std::vector<Real> [naux]), work(new std::vector<Real> [nwork]),
    pid(intprop[ipid]),
    xp(rp[ixp]), yp(rp[iyp]), zp(rp[izp]), vpx(rp[ivpx]), vpy(rp[ivpy]), vpz(rp[ivpz]),
    dxp(drp[ixp]), dyp(drp[iyp]), dzp(drp[izp]),
    dvpx(drp[ivpx]), dvpy(drp[ivpy]), dvpz(drp[ivpz]),
    xi1(work[ixi1]), xi2(work[ixi2]), xi3(work[ixi3]) {
  // Point to the calling MeshBlock.
  pmy_block = pmb;

  pmy_mesh = pmb->pmy_mesh;
  pbval_ = pmb->pbval;
  npar = 0;

  // Check active dimensions.
  active1_ = pmy_mesh->mesh_size.nx1 > 1;
  active2_ = pmy_mesh->mesh_size.nx2 > 1;
  active3_ = pmy_mesh->mesh_size.nx3 > 1;

  if (!MONTE_CARLO_ENABLED) {
    // Allocate mesh auxiliaries.
    ppm = new ParticleMesh(this);
  }
  // Initiate ParticleBuffer class.
  // rp1 is the second register of the dust-particle time integrator
  // (Particles::Integrate).  Monte Carlo photons are pushed by PhotonPusher and never
  // read it, so in a Monte Carlo build it is not resized and not communicated.
  ParticleBuffer::SetNumberOfProperties(
      nint, (MONTE_CARLO_ENABLED ? nreal : 2 * nreal) + naux, ncplx);
}

//--------------------------------------------------------------------------------------
//! \fn Particles::~Particles()
//! \brief destroys a Particles instance.

Particles::~Particles() {
  // Free dynamically allocated space.
  delete [] intprop;
  delete [] rp;
  delete [] rp1;
  delete [] drp;
  delete [] aux;
  delete [] work;

  // Clear links to neighbors.
  ClearNeighbors();

  // Delete mesh auxiliaries.
  if (!MONTE_CARLO_ENABLED) {
    delete ppm;
  }
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::ClearBoundary()
//! \brief resets boundary for particle transportation.

void Particles::ClearBoundary() {
  for (int i = 0; i < pbval_->nneighbor; ++i) {
    NeighborBlock& nb = pbval_->neighbor[i];
    bstatus_[nb.bufid] = BoundaryStatus::waiting;
#ifdef MPI_PARALLEL
    if (nb.snb.rank != Globals::my_rank) {
      ParticleBuffer& recv = recv_[nb.bufid];
      recv.mpi_active = false;
      recv.flagn = recv.flagi = recv.flagr = recv.flagc = 0;
      recv.reqn = recv.reqi = recv.reqr = recv.reqc = MPI_REQUEST_NULL;
      send_[nb.bufid].npar = 0;
    }
#endif
  }

  if (!MONTE_CARLO_ENABLED)
    ppm->ClearBoundary();
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::ClearNeighbors()
//! \brief clears links to neighbors.

void Particles::ClearNeighbors() {
  delete neighbor_[1][1][1].pnb;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) {
        Neighbor *pn = &neighbor_[i][j][k];
        if (pn == NULL) continue;
        while (pn->next != NULL)
          pn = pn->next;
        while (pn->prev != NULL) {
          pn = pn->prev;
          delete pn->next;
          pn->next = NULL;
        }
        pn->pnb = NULL;
        pn->pmb = NULL;
      }
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::Integrate(int stage, Real t, Real dt, Real gamma[])
//! \brief updates all particle positions and velocities from t to t + dt.

void Particles::Integrate(int stage, Real t, Real dt, Real gamma[]) {
  // Compute the rates of change.
  for (int i = 0; i < nreal; ++i)
    drp[i].assign(npar, 0);
  SourceTerms(t, dt, pmy_block->phydro->w);
  UserSourceTerms(t, dt, pmy_block->phydro->w);
  ReactToMeshAux(t, dt, pmy_block->phydro->w);

  if (stage == 1)
    // Initiate multiple copies of particle data.
    RealPropCopy(rp1, rp);

  // Compute weighted averages of the two copies of particle data.
  if (gamma[2] == 0.0) {
    if (gamma[0] == 0.0 && gamma[1] == 1.0)
      RealPropSwap(rp, rp1);
    else
      WeightedAverage(rp, rp1, gamma);
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Particles::Integrate]" << std::endl
        << ">3-stage integrator is not implemented yet. " << std::endl;
    ATHENA_ERROR(msg);
  }

  // Evolve the particle properties.
  for (int i = 0; i < nreal; ++i) {
    std::vector<Real> &rpi(rp[i]), &drpi(drp[i]);
    for (int k = 0; k < npar; ++k)
      rpi[k] += dt * drpi[k];
  }

  // Update the position index.
  SetPositionIndices();
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::LinkNeighbors(MeshBlockTree &tree,
//!         int64_t nrbx1, int64_t nrbx2, int64_t nrbx3, int root_level)
//! \brief fetches neighbor information for later communication.

void Particles::LinkNeighbors(MeshBlockTree &tree,
    int64_t nrbx1, int64_t nrbx2, int64_t nrbx3, int root_level) {

#ifdef MPI_PARALLEL
  // SWD: Get tag limit on this machine
  void *max_tag;
  int flag;
  /* The address of a void pointer must be used! */
  MPI_Comm_get_attr( MPI_COMM_WORLD, MPI_TAG_UB, &max_tag, &flag);
  int maxtag = *static_cast<int*>(max_tag);
  //printf("max tag: %d\n",maxtag);
#endif
  // Set myself as one of the neighbors.
  Neighbor *pn = &neighbor_[1][1][1];
  pn->pmb = pmy_block;
  pn->pnb = new NeighborBlock;
  pn->pnb->SetNeighbor(Globals::my_rank, pmy_block->loc.level,
      pmy_block->gid, pmy_block->lid, 0, 0, 0, NeighborConnect::none,
      -1, -1, false, false, 0, 0);
  // Save pointer to each neighbor.
  for (int i = 0; i < pbval_->nneighbor; ++i) {
    NeighborBlock& nb = pbval_->neighbor[i];
    SimpleNeighborBlock& snb = nb.snb;
    NeighborIndexes& ni = nb.ni;
    Neighbor *pn = &neighbor_[ni.ox1+1][ni.ox2+1][ni.ox3+1];
    while (pn->next != NULL)
      pn = pn->next;
    if (pn->pnb != NULL) {
      pn->next = new Neighbor;
      pn->next->prev = pn;
      pn = pn->next;
    }
    pn->pnb = &nb;
    if (snb.rank == Globals::my_rank) {
      pn->pmb = pmy_mesh->FindMeshBlock(snb.gid);
    } else {
#ifdef MPI_PARALLEL
      send_[nb.bufid].tag = (snb.lid<<8) | (nb.targetid<<2);
      recv_[nb.bufid].tag = (pmy_block->lid<<8) | (nb.bufid<<2);
      if (send_[nb.bufid].tag > maxtag)
        printf("warning: send tag %d %d\n",send_[nb.bufid].tag,max_tag);
      if (recv_[nb.bufid].tag > maxtag)
        printf("warning: send tag %d %d\n",recv_[nb.bufid].tag,max_tag);
#endif
    }
  }

  // Collect missing directions from fine to coarse level.
  if (pmy_mesh->multilevel) {
    int my_level = pbval_->loc.level;
    for (int l = 0; l < 3; l++) {
      if (!active1_ && l != 1) continue;
      for (int m = 0; m < 3; m++) {
        if (!active2_ && m != 1) continue;
        for (int n = 0; n < 3; n++) {
          if (!active3_ && n != 1) continue;
          Neighbor *pn = &neighbor_[l][m][n];
          if (pn->pnb == NULL) {
            int nblevel = pbval_->nblevel[n][m][l];
            if (0 <= nblevel && nblevel < my_level) {
              int ngid = tree.FindNeighbor(pbval_->loc, l-1, m-1, n-1)->GetGid();
              for (int i = 0; i < pbval_->nneighbor; ++i) {
                NeighborBlock& nb = pbval_->neighbor[i];
                if (nb.snb.gid == ngid) {
                  pn->pnb = &nb;
                  if (nb.snb.rank == Globals::my_rank)
                    pn->pmb = pmy_mesh->FindMeshBlock(ngid);
                  break;
                }
              }
            }
          }
        }
      }
    }
  }

  if (!MONTE_CARLO_ENABLED) {
    // Initiate ParticleMesh boundary data.
    ppm->SetBoundaryAttributes();
    ppm->InitiateBoundaryData();
  }

  // Initiate boundary values.
  ClearBoundary();
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::RemoveOneParticle(int k)
//! \brief removes particle k in the block.

void Particles::RemoveOneParticle(int k) {
  if (0 <= k && k < npar) {
    if (--npar != k) {
      // Replace the k-th particle by the last particle.
      for (int j = 0; j < nint; ++j)
        intprop[j][k] = intprop[j].back();
      for (int j = 0; j < nreal; ++j)
        rp[j][k] = rp[j].back();
      if (!MONTE_CARLO_ENABLED)
        for (int j = 0; j < nreal; ++j)
          rp1[j][k] = rp1[j].back();
      for (int j = 0; j < naux; ++j)
        aux[j][k] = aux[j].back();
      for (int j = 0; j < nwork; ++j)
        work[j][k] = work[j].back();
      for (int j = 0; j < ncplx; ++j)
        cplxprop[j][k] = cplxprop[j].back();
    }
    // Remove the last particle.
    for (int j = 0; j < nint; ++j)
      intprop[j].pop_back();
    for (int j = 0; j < nreal; ++j)
      rp[j].pop_back();
    if (!MONTE_CARLO_ENABLED)
      for (int j = 0; j < nreal; ++j)
        rp1[j].pop_back();
    for (int j = 0; j < naux; ++j)
      aux[j].pop_back();
    for (int j = 0; j < nwork; ++j)
      work[j].pop_back();
    for (int j = 0; j < ncplx; ++j)
      cplxprop[j].pop_back();

  } else {
    // Throw error when index k is invalid.
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Particles::RemoveOneParticle]" << std::endl
        << "\tk = " << k << ", npar = " << npar << std::endl
        << "Index k is out of range. " << std::endl;
    ATHENA_ERROR(msg);
  }
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::SendParticleMesh()
//! \brief send ParticleMesh meshaux near boundaries to neighbors.

void Particles::SendParticleMesh() {
  if (ppm->nmeshaux > 0)
    ppm->SendBoundary();
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::SendToNeighbors()
//! \brief sends particles outside boundary to the buffers of neighboring meshblocks.

void Particles::SendToNeighbors() {
  const int IS = pmy_block->is;
  const int IE = pmy_block->ie;
  const int JS = pmy_block->js;
  const int JE = pmy_block->je;
  const int KS = pmy_block->ks;
  const int KE = pmy_block->ke;

  for (int k = 0; k < npar; ) {
    // Check if a particle is outside the boundary.
    int xi1i(static_cast<int>(xi1[k])),
        xi2i(static_cast<int>(xi2[k])),
        xi3i(static_cast<int>(xi3[k]));
    int ox1 = CheckSide(xi1i, IS, IE),
        ox2 = CheckSide(xi2i, JS, JE),
        ox3 = CheckSide(xi3i, KS, KE);
    if (ox1 == 0 && ox2 == 0 && ox3 == 0) {
      ++k;
      continue;
    }

    // Apply boundary conditions and find the mesh coordinates.
    Real x1, x2, x3;
    ApplyBoundaryConditions(k, x1, x2, x3);

    // Find the neighbor block to send it to.
    if (!active1_) ox1 = 0;
    if (!active2_) ox2 = 0;
    if (!active3_) ox3 = 0;
    Neighbor *pn = FindTargetNeighbor(ox1, ox2, ox3, xi1i, xi2i, xi3i);
    NeighborBlock *pnb = pn->pnb;
    if (pnb == NULL) {
      RemoveOneParticle(k);
      continue;
    }

    // Determine which particle buffer to use.
    ParticleBuffer *ppb = NULL;
    if (pnb->snb.rank == Globals::my_rank) {
      // No need to send if back to the same block.
      if (pnb->snb.gid == pmy_block->gid) {
        pmy_block->pcoord->MeshCoordsToIndices(x1, x2, x3, xi1[k], xi2[k], xi3[k]);
        ++k;
        continue;
      }
      // Use the target receive buffer.
      ppb = &pn->pmb->ppar->recv_[pnb->targetid];

    } else {
#ifdef MPI_PARALLEL
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
    for (int j = 0; j < nreal; ++j) {
      *pr++ = rp[j][k];
      if (!MONTE_CARLO_ENABLED) *pr++ = rp1[j][k];
    }
    for (int j = 0; j < naux; ++j)
      *pr++ = aux[j][k];
    ++ppb->npar;

    // Pop the particle from the current MeshBlock.
    RemoveOneParticle(k);
  }

  // Send to neighbor processes and update boundary status.
  for (int i = 0; i < pbval_->nneighbor; ++i) {
    NeighborBlock& nb = pbval_->neighbor[i];
    int dst = nb.snb.rank;
    if (dst == Globals::my_rank) {
      Particles *ppar = pmy_mesh->FindMeshBlock(nb.snb.gid)->ppar;
      ppar->bstatus_[nb.targetid] =
          (ppar->recv_[nb.targetid].npar > 0) ? BoundaryStatus::arrived
                                              : BoundaryStatus::completed;
    } else {
#ifdef MPI_PARALLEL
      ParticleBuffer& send = send_[nb.bufid];
      int npsend = send.npar;
      MPI_Send(&npsend, 1, MPI_INT, nb.snb.rank, send.tag, my_comm);
      if (npsend > 0) {
        MPI_Request req = MPI_REQUEST_NULL;
        MPI_Isend(send.ibuf, npsend * ParticleBuffer::nint, MPI_INT,
                  dst, send.tag + 1, my_comm, &req);
        MPI_Request_free(&req);
        MPI_Isend(send.rbuf, npsend * ParticleBuffer::nreal, MPI_ATHENA_REAL,
                  dst, send.tag + 2, my_comm, &req);
        MPI_Request_free(&req);
      }
#endif
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::SetPositionIndices()
//! \brief updates position indices of particles.

void Particles::SetPositionIndices() {
  GetPositionIndices(0, npar, xp, yp, zp, xi1, xi2, xi3);
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::StartReceiving()
//! \brief starts receiving ParticleMesh meshaux near boundary from neighbor processes.

void Particles::StartReceiving() {
  ppm->StartReceiving();
}

//--------------------------------------------------------------------------------------
//! \fn bool Particles::ReceiveFromNeighbors()
//! \brief receives particles from neighboring meshblocks and returns a flag indicating
//!        if all receives are completed.

bool Particles::ReceiveFromNeighbors() {
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
        MPI_Irecv(&recv.npar, 1, MPI_INT, nb_rank, recv.tag, my_comm, &recv.reqi);
        recv.mpi_active = true;
      }
      if (!recv.flagn) {
        MPI_Test(&recv.reqi, &recv.flagn, MPI_STATUS_IGNORE);
        if (recv.flagn) {
          if (recv.npar > 0) {
            // Check the buffer size.
            int nprecv = recv.npar;
            if (nprecv > recv.nparmax) {
              recv.npar = 0;
              recv.Reallocate(2 * nprecv - recv.nparmax);
              recv.npar = nprecv;
            }
            // Receive data from the neighbor.
            MPI_Irecv(recv.ibuf, recv.npar * ParticleBuffer::nint, MPI_INT,
                      nb_rank, recv.tag + 1, my_comm, &recv.reqi);
            MPI_Irecv(recv.rbuf, recv.npar * ParticleBuffer::nreal, MPI_ATHENA_REAL,
                      nb_rank, recv.tag + 2, my_comm, &recv.reqr);
          } else {
            // No incoming particles.
            bstatus = BoundaryStatus::completed;
          }
        }
      }
      if (recv.flagn && recv.npar > 0) {
        if (!recv.flagi)
          MPI_Test(&recv.reqi, &recv.flagi, MPI_STATUS_IGNORE);
        if (!recv.flagr)
          MPI_Test(&recv.reqr, &recv.flagr, MPI_STATUS_IGNORE);
        if (recv.flagi && recv.flagr)
          bstatus = BoundaryStatus::arrived;
      }
    }
#endif

    switch (bstatus) {
      case BoundaryStatus::completed:
        break;

      case BoundaryStatus::waiting:
        flag = false;
        break;

      case BoundaryStatus::arrived:
        ParticleBuffer& recv = recv_[nb.bufid];
        FlushReceiveBuffer(recv);
        bstatus = BoundaryStatus::completed;
        break;
    }
  }

  return flag;
}

//--------------------------------------------------------------------------------------
//! \fn bool Particles::ReceiveParticleMesh(Real t, Real dt)
//! \brief receives ParticleMesh meshaux near boundaries from neighbors and returns a
//!        flag indicating if all receives are completed.

bool Particles::ReceiveParticleMesh(Real t, Real dt) {
  if (ppm->nmeshaux <= 0) return true;

  // Flush ParticleMesh receive buffers.
  bool flag = ppm->ReceiveBoundary();

  if (flag) {
    // Deposit ParticleMesh meshaux to MeshBlock.
    Hydro *phydro(pmy_block->phydro);
    DepositToMesh(t, dt, phydro->w, phydro->u);
  }
  return flag;
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::ProcessNewParticles()
//! \brief searches for and books new particles.

void Particles::ProcessNewParticles(Mesh *pmesh) {
  // Count new particles.
  const int nbtotal(pmesh->nbtotal), nblocks(pmesh->nblocal);
  std::vector<int> nnewpar(nbtotal, 0);
  for (int b = 0; b < nblocks; ++b) {
    const MeshBlock *pmb(pmesh->my_blocks(b));
    nnewpar[pmb->gid] = pmb->ppar->CountNewParticles();
  }
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &nnewpar[0], nbtotal, MPI_INT, MPI_MAX, my_comm);
#endif

  // Make the counts cumulative.
  for (int i = 1; i < nbtotal; ++i)
    nnewpar[i] += nnewpar[i-1];

  // Set particle IDs.
  for (int b = 0; b < nblocks; ++b) {
    const MeshBlock *pmb(pmesh->my_blocks(b));
    pmb->ppar->SetNewParticleID(idmax + (pmb->gid > 0 ? nnewpar[pmb->gid - 1] : 0));
  }
  idmax += nnewpar[nbtotal - 1];
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::SourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc)
//! \brief computes and adds the rate of change for each dynamical variable.

void Particles::SourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc) {
  for (int k = 0; k < npar; ++k) {
    dxp[k] += vpx[k];
    dyp[k] += vpy[k];
    dzp[k] += vpz[k];
  }
}

//--------------------------------------------------------------------------------------
//! \fn int Particles::CountNewParticles()
//! \brief counts new particles in the block.

int Particles::CountNewParticles() const {
  int n = 0;
  for (int i = 0; i < npar; ++i)
    if (pid[i] <= 0) ++n;
  return n;
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::ApplyBoundaryConditions(int k, Real &x1, Real &x2, Real &x3)
//! \brief applies boundary conditions to particle k and returns its updated mesh
//!        coordinates (x1,x2,x3).
//! \todo (ccyang):
//! - implement nonperiodic boundary conditions.

void Particles::ApplyBoundaryConditions(int k, Real &x1, Real &x2, Real &x3) {
  bool flag = false;
  RegionSize& mesh_size = pmy_mesh->mesh_size;
  Coordinates *pcoord = pmy_block->pcoord;

  // Find the mesh coordinates.
  Real x10, x20, x30;
  pcoord->IndicesToMeshCoords(xi1[k], xi2[k], xi3[k], x1, x2, x3);
  pcoord->CartesianToMeshCoords(rp1[ixp][k], rp1[iyp][k], rp1[izp][k], x10, x20, x30);

  // Convert velocity vectors in mesh coordinates.
  Real vp1, vp2, vp3, vp10, vp20, vp30;
  pcoord->CartesianToMeshCoordsVector(xp[k], yp[k], zp[k],
      vpx[k], vpy[k], vpz[k], vp1, vp2, vp3);
  pcoord->CartesianToMeshCoordsVector(rp1[ixp][k], rp1[iyp][k], rp1[izp][k],
      rp1[ivpx][k], rp1[ivpy][k], rp1[ivpz][k], vp10, vp20, vp30);

  // Apply periodic boundary conditions in X1.
  if (x1 < mesh_size.x1min) {
    // Inner x1
    x1 += mesh_size.x1len;
    x10 += mesh_size.x1len;
    flag = true;
  } else if (x1 >= mesh_size.x1max) {
    // Outer x1
    x1 -= mesh_size.x1len;
    x10 -= mesh_size.x1len;
    flag = true;
  }

  // Apply periodic boundary conditions in X2.
  if (x2 < mesh_size.x2min) {
    // Inner x2
    x2 += mesh_size.x2len;
    x20 += mesh_size.x2len;
    flag = true;
  } else if (x2 >= mesh_size.x2max) {
    // Outer x2
    x2 -= mesh_size.x2len;
    x20 -= mesh_size.x2len;
    flag = true;
  }

  // Apply periodic boundary conditions in X3.
  if (x3 < mesh_size.x3min) {
    // Inner x3
    x3 += mesh_size.x3len;
    x30 += mesh_size.x3len;
    flag = true;
  } else if (x3 >= mesh_size.x3max) {
    // Outer x3
    x3 -= mesh_size.x3len;
    x30 -= mesh_size.x3len;
    flag = true;
  }

  if (flag) {
    // Convert positions and velocities back in Cartesian coordinates.
    pcoord->MeshCoordsToCartesian(x1, x2, x3, xp[k], yp[k], zp[k]);
    pcoord->MeshCoordsToCartesian(x10, x20, x30, rp1[ixp][k], rp1[iyp][k], rp1[izp][k]);
    pcoord->MeshCoordsToCartesianVector(x1, x2, x3,
        vp1, vp2, vp3, vpx[k], vpy[k], vpz[k]);
    pcoord->MeshCoordsToCartesianVector(x10, x20, x30,
        vp10, vp20, vp30, rp1[ivpx][k], rp1[ivpy][k], rp1[ivpz][k]);
  }
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::GetPositionIndices(int ibegin, int iend,
//!         const std::vector<Real>& xp,
//!         const std::vector<Real>& yp,
//!         const std::vector<Real>& zp,
//!         std::vector<Real>& xi1, std::vector<Real>& xi2, std::vector<Real>& xi3)
//! \brief finds the position indices of each particle with respect to the local grid.

void Particles::GetPositionIndices(int ibegin, int iend,
    const std::vector<Real>& xp, const std::vector<Real>& yp, const std::vector<Real>& zp,
    std::vector<Real>& xi1, std::vector<Real>& xi2, std::vector<Real>& xi3) {
  for (int k = ibegin; k < iend; ++k) {
    // Convert to the Mesh coordinates.
    Real x1, x2, x3;
    pmy_block->pcoord->CartesianToMeshCoords(xp[k], yp[k], zp[k], x1, x2, x3);

    // Convert to the index space.
    pmy_block->pcoord->MeshCoordsToIndices(x1, x2, x3, xi1[k], xi2[k], xi3[k]);
  }
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::RealPropCopy(std::vector<Real> *rp1, const std::vector<Real> *rp2)
//! \brief copies all properties in rp2 into rp1.

void Particles::RealPropCopy(std::vector<Real> *rp1, const std::vector<Real> *rp2) {
  for (int i = 0; i < nreal; ++i)
    rp1[i] = rp2[i];
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::RealPropSwap(std::vector<Real> *rp1, std::vector<Real> *rp2)
//! \brief swap rp1 and rp2.

void Particles::RealPropSwap(std::vector<Real> *rp1, std::vector<Real> *rp2) {
  for (int i = 0; i < nreal; ++i)
    std::swap(rp1[i], rp2[i]);
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::SetNewParticleID(int id0)
//! \brief searches for new particles and assigns ID, beginning at id + 1.

void Particles::SetNewParticleID(int id) {
  for (int i = 0; i < npar; ++i)
    if (pid[i] <= 0) pid[i] = ++id;
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::WeightedAverage(
//    std::vector<Real> *rp_out, const std::vector<Real> *rp_in1, const Real weights[])
//! \brief computes weighted averages of particle arrays.

void Particles::WeightedAverage(
    std::vector<Real> *rp_out, const std::vector<Real> *rp_in1, const Real weights[]) {
  const Real a(weights[0]), b(weights[1]);
  if (a == 0.0) { // rp_out = b * rp_in1;
    if (b == 1.0) {
      RealPropCopy(rp_out, rp_in1);
    } else {
      for (int i = 0; i < nreal; ++i)
        for (int k = 0; k < npar; ++k)
          rp_out[i][k] = b * rp_in1[i][k];
    }
  } else if (a == 1.0) { // rp_out += b * rp_in1;
    if (b == 1.0) {
      for (int i = 0; i < nreal; ++i)
        for (int k = 0; k < npar; ++k)
          rp_out[i][k] += rp_in1[i][k];
    } else if (b != 0.0) {
      for (int i = 0; i < nreal; ++i)
        for (int k = 0; k < npar; ++k)
          rp_out[i][k] += b * rp_in1[i][k];
    }
  } else { // rp_out = a * rp_out + b * rp_in1;
    if (b == 0.0) {
      for (int i = 0; i < nreal; ++i)
        for (int k = 0; k < npar; ++k)
          rp_out[i][k] *= a;
    } else if (b == 1.0) {
      for (int i = 0; i < nreal; ++i)
        for (int k = 0; k < npar; ++k)
          rp_out[i][k] = a * rp_out[i][k] + rp_in1[i][k];
    } else {
      for (int i = 0; i < nreal; ++i)
        for (int k = 0; k < npar; ++k)
          rp_out[i][k] = a * rp_out[i][k] + b * rp_in1[i][k];
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn MeshBlock* Particles::FindTargetNeighbor(
//!         int ox1, int ox2, int ox3, int xi1, int xi2, int xi3)
//! \brief finds the neighbor to send a particle to.

struct Neighbor* Particles::FindTargetNeighbor(
    int ox1, int ox2, int ox3, int xi1, int xi2, int xi3) {
  // Find the head of the linked list.
  Neighbor *pn = &neighbor_[ox1+1][ox2+1][ox3+1];

  // Search down the list if the neighbor is at a finer level.
  if (pmy_mesh->multilevel && pn->pnb != NULL &&
      pn->pnb->snb.level > pmy_block->loc.level) {
    RegionSize& bs = pmy_block->block_size;
    int fi[2] = {0, 0}, i = 0;
    if (active1_ && ox1 == 0) fi[i++] = 2 * (xi1 - pmy_block->is) / bs.nx1;
    if (active2_ && ox2 == 0) fi[i++] = 2 * (xi2 - pmy_block->js) / bs.nx2;
    if (active3_ && ox3 == 0) fi[i++] = 2 * (xi3 - pmy_block->ks) / bs.nx3;
    while (pn != NULL) {
      NeighborIndexes& ni = pn->pnb->ni;
      if (ni.fi1 == fi[0] && ni.fi2 == fi[1]) break;
      pn = pn->next;
    }
  }

  // Return the target neighbor.
  return pn;
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::FlushReceiveBuffer(ParticleBuffer& recv)
//! \brief adds particles from the receive buffer.

void Particles::FlushReceiveBuffer(ParticleBuffer& recv) {
  // Check the memory size.
  int nprecv(recv.npar), npar_old(npar);
  //printf("npar,nprecv: %d %d\n",npar,nprecv);
  Resize(npar + nprecv);

  // Flush the receive buffers.
  int *pi = recv.ibuf;
  Real *pr = recv.rbuf;
  std::complex<Real> *pc = recv.cbuf;
  for (int k = npar_old; k < npar; ++k) {
    for (int j = 0; j < nint; ++j)
      intprop[j][k] = *pi++;
    for (int j = 0; j < nreal; ++j) {
      rp[j][k] = *pr++;
      if (!MONTE_CARLO_ENABLED) rp1[j][k] = *pr++;
    }
    for (int j = 0; j < naux; ++j)
      aux[j][k] = *pr++;
    if (ncplx > 0) {
      for (int j = 0; j < ncplx; ++j) {
        cplxprop[j][k] = *pc++;
      }
    }
  }
  // Find their position indices.
  if (!MONTE_CARLO_ENABLED) {
    GetPositionIndices(npar_old, npar, xp, yp, zp, xi1, xi2, xi3);
  }
  // Clear the receive buffers.
  recv.npar = 0;
}

//--------------------------------------------------------------------------------------
//! \fn int Particles::AddIntProperty(const std::string& name)
//! \brief adds one integer property to the particles and returns the index.

int Particles::AddIntProperty(const std::string& name) {
  ipname.push_back(name);
  return nint++;
}

//--------------------------------------------------------------------------------------
//! \fn int Particles::AddRealProperty(const std::string& name)
//! \brief adds one real property to the particles and returns the index.

int Particles::AddRealProperty(const std::string& name) {
  rpname.push_back(name);
  return nreal++;
}

//--------------------------------------------------------------------------------------
//! \fn int Particles::AddAuxProperty()
//! \brief adds one auxiliary property to the particles and returns the index.

int Particles::AddAuxProperty() {
  return naux++;
}

//--------------------------------------------------------------------------------------
//! \fn int Particles::AddWorkingArray()
//! \brief adds one working array to the particles and returns the index.

int Particles::AddWorkingArray() {
  return nwork++;
}

//--------------------------------------------------------------------------------------
//! \fn int Particles::AddComplexProperty(const std::string& name)
//! \brief adds one complex property to the particle class and returns the index.
int Particles::AddComplexProperty(const std::string& name) {
  cpname.push_back(name);
  return ncplx++;
}



//--------------------------------------------------------------------------------------
//! \fn void Particles::Resize(int new_npar)
//! \brief changes number of particles.

void Particles::Resize(int new_npar) {
  // Resize the particle arrays.
  for (int i = 0; i < nint; ++i)
    intprop[i].resize(new_npar);
  for (int i = 0; i < nreal; ++i)
    rp[i].resize(new_npar);
  if (!MONTE_CARLO_ENABLED)
    for (int i = 0; i < nreal; ++i)
      rp1[i].resize(new_npar);
  for (int i = 0; i < naux; ++i)
    aux[i].resize(new_npar);
  for (int i = 0; i < nwork; ++i)
    work[i].resize(new_npar);
  for (int i = 0; i < ncplx; ++i)
    cplxprop[i].resize(new_npar);

  // Flag new particles.
  for (int k = npar; k < new_npar; ++k)
    pid[k] = -1;

  // Update number of particles.
  npar = new_npar;
}

//--------------------------------------------------------------------------------------
//! \fn Real Particles::NewBlockTimeStep();
//! \brief returns the time step required by particles in the block.

Real Particles::NewBlockTimeStep() {
  Coordinates *pc = pmy_block->pcoord;

  // Find the maximum coordinate speed.
  Real dt_inv2_max = 0.0;
  for (int k = 0; k < npar; ++k) {
    Real dt_inv2 = 0.0, vpx1, vpx2, vpx3;
    pc->CartesianToMeshCoordsVector(xp[k], yp[k], zp[k], vpx[k], vpy[k], vpz[k],
                                    vpx1, vpx2, vpx3);
    dt_inv2 += active1_ ? std::pow(vpx1 / pc->dx1f(static_cast<int>(xi1[k])), 2) : 0;
    dt_inv2 += active2_ ? std::pow(vpx2 / pc->dx2f(static_cast<int>(xi2[k])), 2) : 0;
    dt_inv2 += active3_ ? std::pow(vpx3 / pc->dx3f(static_cast<int>(xi3[k])), 2) : 0;
    dt_inv2_max = std::max(dt_inv2_max, dt_inv2);
  }

  // Return the time step constrained by the coordinate speed.
  return dt_inv2_max > 0.0 ? cfl_par / std::sqrt(dt_inv2_max)
                           : std::numeric_limits<Real>::max();
}

//--------------------------------------------------------------------------------------
//! \fn std::size_t Particles::GetSizeInBytes()
//! \brief returns the data size in bytes in the meshblock.

std::size_t Particles::GetSizeInBytes() {
  std::size_t size = sizeof(npar);
  if (npar > 0) size += npar * (nint * sizeof(int) + nreal * sizeof(Real));
  return size;
}

//--------------------------------------------------------------------------------------
//! \fn Particles::UnpackParticlesForRestart()
//! \brief reads the particle data from the restart file.

void Particles::UnpackParticlesForRestart(char *mbdata, std::size_t &os) {
  // Read number of particles.
  std::memcpy(&npar, &(mbdata[os]), sizeof(npar));
  os += sizeof(npar);
  Resize(npar);

  if (npar > 0) {
    // Read integer properties.
    std::size_t size = npar * sizeof(int);
    for (int k = 0; k < nint; ++k) {
      std::memcpy(intprop[k].data(), &(mbdata[os]), size);
      os += size;
    }

    // Read real properties.
    size = npar * sizeof(Real);
    for (int k = 0; k < nreal; ++k) {
      std::memcpy(rp[k].data(), &(mbdata[os]), size);
      os += size;
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn Particles::PackParticlesForRestart()
//! \brief pack the particle data for restart dump.

void Particles::PackParticlesForRestart(char *&pdata) {
  // Write number of particles.
  std::memcpy(pdata, &npar, sizeof(npar));
  pdata += sizeof(npar);

  if (npar > 0) {
    // Write integer properties.
    std::size_t size = npar * sizeof(int);
    for (int k = 0; k < nint; ++k) {
      std::memcpy(pdata, intprop[k].data(), size);
      pdata += size;
    }
    // Write real properties.
    size = npar * sizeof(Real);
    for (int k = 0; k < nreal; ++k) {
      std::memcpy(pdata, rp[k].data(), size);
      pdata += size;
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
