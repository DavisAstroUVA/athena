//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mcsnapshot.cpp
//! \brief reads one MeshBlock's cell data out of an athdf snapshot
//
// This replaces the block of near-identical reading code that used to be copied into each
// athdf-reading problem generator.  The copies had drifted: one passed the collective
// flag and another did not, one rescaled velocities and another did not, and the field
// read asked for a layout that neither an Athena++ dump nor an AthenaK-derived one
// actually has.

// C++ headers
#include <algorithm>  // max
#include <cctype>     // tolower
#include <iostream>   // cout
#include <sstream>
#include <string>
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../defs.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../inputs/hdf5_reader.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "mcgrid.hpp"
#include "mcsnapshot.hpp"

#ifdef HDF5OUTPUT
#include <hdf5.h>
#endif

namespace {

//----------------------------------------------------------------------------------------
//! \fn bool IEquals(const std::string &a, const std::string &b)
//! \brief case-insensitive comparison; Athena++ writes Bcc1 and AthenaK writes bcc1

bool IEquals(const std::string &a, const std::string &b) {
  if (a.size() != b.size()) return false;
  for (std::size_t n = 0; n < a.size(); ++n) {
    if (std::tolower(static_cast<unsigned char>(a[n]))
        != std::tolower(static_cast<unsigned char>(b[n]))) return false;
  }
  return true;
}

//----------------------------------------------------------------------------------------
//! \struct VarRef
//! \brief where one variable lives in the file

struct VarRef {
  std::string name;     // as spelled in the file
  std::string dataset;  // dataset holding it
  int index;            // its slot along that dataset's leading dimension
  bool found;
  VarRef() : index(-1), found(false) {}
};

//----------------------------------------------------------------------------------------
//! \struct Catalogue
//! \brief the file's variable list, flattened to name -> (dataset, index)

struct Catalogue {
  std::vector<std::string> name;
  std::vector<std::string> dataset;
  std::vector<int> index;

  //! first variable matching any of the candidate spellings
  VarRef Find(const char *const *candidates, int ncand) const {
    for (int c = 0; c < ncand; ++c) {
      for (std::size_t n = 0; n < name.size(); ++n) {
        if (IEquals(name[n], candidates[c])) {
          VarRef r;
          r.name = name[n];
          r.dataset = dataset[n];
          r.index = index[n];
          r.found = true;
          return r;
        }
      }
    }
    return VarRef();
  }

  //! same, but an explicit <problem> override wins when it names a variable
  VarRef Find(const char *const *candidates, int ncand, ParameterInput *pin,
              const char *override_key) const {
    if (pin->DoesParameterExist("problem", override_key)) {
      std::string want = pin->GetString("problem", override_key);
      if (!want.empty()) {
        const char *one[1] = {want.c_str()};
        VarRef r = Find(one, 1);
        if (!r.found) {
          std::stringstream msg;
          msg << "### FATAL ERROR in MCReadSnapshotBlock" << std::endl
              << "<problem>/" << override_key << " names variable '" << want
              << "', which the snapshot does not contain." << std::endl;
          ATHENA_ERROR(msg);
        }
        return r;
      }
    }
    return Find(candidates, ncand);
  }

  std::string Join() const {
    std::string s;
    for (std::size_t n = 0; n < name.size(); ++n) {
      if (n) s += ", ";
      s += name[n];
    }
    return s;
  }
};

#ifdef HDF5OUTPUT

//----------------------------------------------------------------------------------------
//! \fn std::vector<std::string> ReadStringArrayAttr(hid_t file, const char *name)
//! \brief read an array-of-fixed-length-strings attribute

std::vector<std::string> ReadStringArrayAttr(hid_t file, const char *name) {
  std::vector<std::string> out;
  if (H5Aexists(file, name) <= 0) return out;
  hid_t attr = H5Aopen(file, name, H5P_DEFAULT);
  hid_t type = H5Aget_type(attr);
  hid_t space = H5Aget_space(attr);
  hsize_t dims[1] = {0};
  H5Sget_simple_extent_dims(space, dims, NULL);
  std::size_t len = H5Tget_size(type);
  int n = static_cast<int>(dims[0]);
  std::vector<char> buf((n > 0 ? n : 1) * len);
  H5Aread(attr, type, buf.data());
  for (int i = 0; i < n; ++i) {
    const char *p = buf.data() + i * len;
    std::size_t l = 0;
    while (l < len && p[l] != '\0') ++l;
    out.push_back(std::string(p, l));
  }
  H5Sclose(space);
  H5Tclose(type);
  H5Aclose(attr);
  return out;
}

//----------------------------------------------------------------------------------------
//! \fn std::vector<int> ReadIntArrayAttr(hid_t file, const char *name)
//! \brief read an integer array attribute, whatever width the file stores it at

std::vector<int> ReadIntArrayAttr(hid_t file, const char *name) {
  std::vector<int> out;
  if (H5Aexists(file, name) <= 0) return out;
  hid_t attr = H5Aopen(file, name, H5P_DEFAULT);
  hid_t space = H5Aget_space(attr);
  hsize_t dims[1] = {0};
  H5Sget_simple_extent_dims(space, dims, NULL);
  out.resize(static_cast<std::size_t>(dims[0]));
  if (!out.empty()) H5Aread(attr, H5T_NATIVE_INT, out.data());
  H5Sclose(space);
  H5Aclose(attr);
  return out;
}

#endif  // HDF5OUTPUT

//----------------------------------------------------------------------------------------
//! \fn Catalogue ReadCatalogue(const std::string &filename)
//! \brief flatten DatasetNames / NumVariables / VariableNames into
//!        name -> (dataset, slot)

Catalogue ReadCatalogue(const std::string &filename) {
  Catalogue cat;
  std::stringstream msg;
#ifndef HDF5OUTPUT
  msg << "### FATAL ERROR in MCReadSnapshotBlock" << std::endl
      << "Reading an athdf snapshot requires HDF5.  Reconfigure with -hdf5."
      << std::endl;
  ATHENA_ERROR(msg);
#else
  hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) {
    msg << "### FATAL ERROR in MCReadSnapshotBlock" << std::endl
        << "Could not open " << filename << std::endl;
    ATHENA_ERROR(msg);
  }
  std::vector<std::string> dsets = ReadStringArrayAttr(file, "DatasetNames");
  std::vector<std::string> vars = ReadStringArrayAttr(file, "VariableNames");
  std::vector<int> nvar = ReadIntArrayAttr(file, "NumVariables");
  H5Fclose(file);

  if (dsets.empty() || vars.empty() || nvar.size() != dsets.size()) {
    msg << "### FATAL ERROR in MCReadSnapshotBlock" << std::endl
        << filename << " does not carry a usable DatasetNames / NumVariables / "
        << "VariableNames triple." << std::endl;
    ATHENA_ERROR(msg);
  }

  std::size_t v = 0;
  for (std::size_t d = 0; d < dsets.size(); ++d) {
    for (int i = 0; i < nvar[d]; ++i, ++v) {
      if (v >= vars.size()) break;
      cat.name.push_back(vars[v]);
      cat.dataset.push_back(dsets[d]);
      cat.index.push_back(i);
    }
  }
  if (cat.name.size() != vars.size()) {
    msg << "### FATAL ERROR in MCReadSnapshotBlock" << std::endl
        << filename << ": NumVariables sums to " << cat.name.size() << " but "
        << "VariableNames holds " << vars.size() << " entries." << std::endl;
    ATHENA_ERROR(msg);
  }
#endif  // HDF5OUTPUT
  return cat;
}

// Accepted spellings.  Athena++ first, then AthenaK.
const char *kDens[]  = {"rho", "dens"};
const char *kVel1[]  = {"vel1", "velx", "vx"};
const char *kVel2[]  = {"vel2", "vely", "vy"};
const char *kVel3[]  = {"vel3", "velz", "vz"};
const char *kPress[] = {"press", "pgas", "pres"};
const char *kEint[]  = {"eint", "egas", "e"};
const char *kBcc1[]  = {"Bcc1", "bcc1"};
const char *kBcc2[]  = {"Bcc2", "bcc2"};
const char *kBcc3[]  = {"Bcc3", "bcc3"};

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn MCSnapshotVars GetMCSnapshotVarsFlag(const std::string &name)
//! \brief parse <problem>/snapshot_vars

MCSnapshotVars GetMCSnapshotVarsFlag(const std::string &name) {
  if (name == "primitive" || name == "prim") return MCSNAP_PRIMITIVE;
  if (name == "conserved" || name == "cons") return MCSNAP_CONSERVED;
  std::stringstream msg;
  msg << "### FATAL ERROR in GetMCSnapshotVarsFlag" << std::endl
      << "Unrecognized <problem>/snapshot_vars = '" << name << "'." << std::endl
      << "Valid values are: primitive, conserved." << std::endl;
  ATHENA_ERROR(msg);
  return MCSNAP_PRIMITIVE;
}

const char *GetMCSnapshotVarsName(MCSnapshotVars v) {
  return (v == MCSNAP_CONSERVED) ? "conserved" : "primitive";
}

//----------------------------------------------------------------------------------------
//! \fn void MCReadSnapshotBlock(MeshBlock *pmb, ParameterInput *pin)
//! \brief fill this block's primitives, and the cell-centred field, from the snapshot

void MCReadSnapshotBlock(MeshBlock *pmb, ParameterInput *pin, int max_blocks_per_rank) {
  std::stringstream msg;

  // Which file.  <problem>/input_filename is the long-standing key; a run that already
  // names a snapshot for its grid need not repeat it.
  std::string filename = pin->GetOrAddString("problem", "input_filename", "");
  if (filename.empty() && MCGridFile::Loaded() != nullptr)
    filename = MCGridFile::Loaded()->filename;
  if (filename.empty()) {
    msg << "### FATAL ERROR in MCReadSnapshotBlock" << std::endl
        << "No snapshot named.  Set <problem>/input_filename or "
        << "<montecarlo>/grid_from_file." << std::endl;
    ATHENA_ERROR(msg);
  }

  const MCSnapshotVars vars =
      GetMCSnapshotVarsFlag(pin->GetOrAddString("problem", "snapshot_vars", "primitive"));
  if (vars != MCSNAP_PRIMITIVE) {
    msg << "### FATAL ERROR in MCReadSnapshotBlock" << std::endl
        << "<problem>/snapshot_vars = " << GetMCSnapshotVarsName(vars)
        << " is not implemented yet; only primitives can be read." << std::endl
        << "Reading conserved variables also needs a ConservedToPrimitive call and, in "
        << std::endl << "GR, the metric at the time the snapshot was written."
        << std::endl;
    ATHENA_ERROR(msg);
  }

  const bool collective = pin->GetOrAddBoolean("problem", "collective", false);
  const Catalogue cat = ReadCatalogue(filename);

  // Locate what we need.  Explicit <problem> keys override the name search, for a file
  // whose variables are spelled in some third way.
  VarRef dens  = cat.Find(kDens,  2, pin, "var_dens");
  VarRef vel1  = cat.Find(kVel1,  3, pin, "var_vel1");
  VarRef vel2  = cat.Find(kVel2,  3, pin, "var_vel2");
  VarRef vel3  = cat.Find(kVel3,  3, pin, "var_vel3");
  VarRef press = cat.Find(kPress, 3, pin, "var_press");
  VarRef eint  = cat.Find(kEint,  3, pin, "var_eint");

  if (!dens.found || !vel1.found || !vel2.found || !vel3.found) {
    msg << "### FATAL ERROR in MCReadSnapshotBlock" << std::endl
        << filename << " is missing density or velocity." << std::endl
        << "It contains: " << cat.Join() << std::endl
        << "Name them explicitly with <problem>/var_dens, var_vel1, var_vel2, var_vel3 "
        << "if" << std::endl << "they are spelled differently." << std::endl;
    ATHENA_ERROR(msg);
  }
  // A file carries the gas energy either as a pressure or as an internal energy density;
  // AthenaK-derived snapshots use the latter.  Prefer pressure when both are present.
  if (!press.found && !eint.found) {
    msg << "### FATAL ERROR in MCReadSnapshotBlock" << std::endl
        << filename << " has neither a pressure nor an internal energy density."
        << std::endl << "It contains: " << cat.Join() << std::endl;
    ATHENA_ERROR(msg);
  }
  const bool from_eint = !press.found;
  const VarRef energy = from_eint ? eint : press;

  if (Globals::my_rank == 0 && pmb->lid == 0) {
    std::cout << "Monte Carlo snapshot data from " << filename << std::endl
              << "  density  = " << dens.name << ", velocity = " << vel1.name << ", "
              << vel2.name << ", " << vel3.name << std::endl
              << "  energy   = " << energy.name << " ("
              << (from_eint ? "internal energy density, converted to pressure"
                            : "pressure") << ")" << std::endl;
  }

  Hydro *ph = pmb->phydro;
  const int nx1 = pmb->block_size.nx1;
  const int nx2 = pmb->block_size.nx2;
  const int nx3 = pmb->block_size.nx3;

  // Index along the file's block dimension.  When the grid was rebuilt from this
  // snapshot, MCGridFile matched the Mesh's gids to the file's block ordering; the two
  // need not coincide for a refined tree.  Otherwise the input file is expected to
  // describe a mesh that already matches, and gid is the index.
  const int file_block = (MCGridFile::Loaded() != nullptr)
                         ? MCGridFile::Loaded()->FileIndex(pmb->gid) : pmb->gid;

  int start_file[5] = {0, file_block, 0, 0, 0};
  int count_file[5] = {1, 1, nx3, nx2, nx1};
  int start_mem[4] = {0, pmb->ks, pmb->js, pmb->is};
  int count_mem[4] = {1, nx3, nx2, nx1};

  // Every rank must make the same number of collective reads, so count them.
  int nread = 0;
  const VarRef *src[5] = {&dens, &vel1, &vel2, &vel3, &energy};
  const int dst[5] = {IDN, IVX, IVY, IVZ, IPR};
  for (int n = 0; n < 5; ++n) {
    start_file[0] = src[n]->index;
    start_mem[0] = dst[n];
    HDF5ReadRealArray(filename.c_str(), src[n]->dataset.c_str(), 5, start_file,
                      count_file, 4, start_mem, count_mem, ph->w, collective);
    ++nread;
  }

  // Internal energy density -> pressure for an ideal gas.  Done after the read so the
  // conversion is visible here rather than hidden in each problem generator.
  if (from_eint) {
    const Real gm1 = pmb->peos->GetGamma() - 1.0;
    for (int k = pmb->ks; k <= pmb->ke; ++k) {
      for (int j = pmb->js; j <= pmb->je; ++j) {
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          ph->w(IPR,k,j,i) *= gm1;
        }
      }
    }
  }

  // Cell-centred magnetic field.  Both an Athena++ dump and an AthenaK-derived one store
  // Bcc, not the face-centred b, and a face-centred field cannot be recovered from it:
  // there is no unique divergence-free reconstruction.  This fills bcc, which is what the
  // Monte Carlo module reads (MonteCarloBlock::bcc is a shallow slice of it).  Note that
  // Field::CalculateCellCenteredField recomputes bcc from b in ghost zones during
  // boundary application, so values loaded here survive only in the active cells.
  if (MAGNETIC_FIELDS_ENABLED) {
    VarRef b1 = cat.Find(kBcc1, 2, pin, "var_bcc1");
    VarRef b2 = cat.Find(kBcc2, 2, pin, "var_bcc2");
    VarRef b3 = cat.Find(kBcc3, 2, pin, "var_bcc3");
    if (!b1.found || !b2.found || !b3.found) {
      msg << "### FATAL ERROR in MCReadSnapshotBlock" << std::endl
          << "This build has magnetic fields enabled but " << filename << std::endl
          << "does not carry a cell-centred field (Bcc1/Bcc2/Bcc3)." << std::endl
          << "It contains: " << cat.Join() << std::endl;
      ATHENA_ERROR(msg);
    }
    const VarRef *bsrc[3] = {&b1, &b2, &b3};
    for (int n = 0; n < 3; ++n) {
      start_file[0] = bsrc[n]->index;
      start_mem[0] = n;
      HDF5ReadRealArray(filename.c_str(), bsrc[n]->dataset.c_str(), 5, start_file,
                        count_file, 4, start_mem, count_mem, pmb->pfield->bcc,
                        collective);
      ++nread;
    }
  }

  // A collective read has to be issued the same number of times by every rank, so a rank
  // holding fewer blocks than the busiest one makes up the difference with reads that
  // select nothing.  Only meaningful when the reads above were collective.
#ifdef MPI_PARALLEL
  if (collective && max_blocks_per_rank > 0) {
    const int nb_here = pmb->pmy_mesh->nblocal;
    if (pmb->lid == nb_here - 1) {
      for (int block = 0; block < max_blocks_per_rank - nb_here; ++block) {
        for (int n = 0; n < nread; ++n) {
          start_file[0] = 0;
          start_mem[0] = 0;
          HDF5ReadRealArray(filename.c_str(), src[0]->dataset.c_str(), 5, start_file,
                            count_file, 4, start_mem, count_mem, ph->w, collective,
                            true);
        }
      }
    }
  }
#else
  (void)max_blocks_per_rank;
#endif
}
