//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mcgrid.cpp
//! \brief recovers the mesh structure of an athdf snapshot
//
// The HDF5 helpers here are local to monte_carlo rather than added to
// src/inputs/hdf5_reader.cpp: they read attributes and integer datasets, which nothing
// outside the Monte Carlo module needs

// C++ headers
#include <cmath>      // pow, fabs
#include <cstring>    // strncmp
#include <iomanip>    // setprecision
#include <iostream>
#include <limits>     // numeric_limits
#include <sstream>
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../defs.hpp"
#include "../globals.hpp"
#include "../parameter_input.hpp"
#include "mcgrid.hpp"

// Only the reading is conditional on HDF5; the class itself always compiles so that
// callers can be gated on MONTE_CARLO_ENABLED alone.
#ifdef HDF5OUTPUT
#include <hdf5.h>
#endif

MCGridFile *MCGridFile::ploaded_ = nullptr;

namespace {

// name of the input parameter holding the snapshot path
const char *kGridFileBlock = "montecarlo";
const char *kGridFileParam = "grid_from_file";

// tolerance on the agreement between regenerated and stored cell faces, relative to
// the root grid extent.  athdf is written in single precision unless the source run
// was configured with -h5double, so a few float epsilon is the floor; measured worst
// case on a 5-level spherical-polar AMR snapshot with a logarithmic radial grid is
// 2.1e-7.  The mesh generator evaluates rat^(x*nx) on a normalized fraction rather
// than multiplying iteratively, so the error does not grow with nx.
const Real kFaceTol = 1.0e-6;

#ifdef HDF5OUTPUT

//----------------------------------------------------------------------------------------
//! \fn bool HasAttribute(hid_t file, const char *name)
//! \brief does the file carry this attribute?

bool HasAttribute(hid_t file, const char *name) {
  return H5Aexists(file, name) > 0;
}

//----------------------------------------------------------------------------------------
//! \fn void ReadAttribute(hid_t file, const char *name, int count, T *out)
//! \brief read a numeric attribute, converting from whatever the file stores

void ReadAttribute(hid_t file, const char *name, int count, int *out) {
  hid_t attr = H5Aopen(file, name, H5P_DEFAULT);
  H5Aread(attr, H5T_NATIVE_INT, out);
  H5Aclose(attr);
  (void)count;
}

void ReadAttribute(hid_t file, const char *name, int count, double *out) {
  hid_t attr = H5Aopen(file, name, H5P_DEFAULT);
  H5Aread(attr, H5T_NATIVE_DOUBLE, out);
  H5Aclose(attr);
  (void)count;
}

//----------------------------------------------------------------------------------------
//! \fn std::string ReadStringAttribute(hid_t file, const char *name)
//! \brief read a scalar fixed-length string attribute

std::string ReadStringAttribute(hid_t file, const char *name) {
  hid_t attr = H5Aopen(file, name, H5P_DEFAULT);
  hid_t type = H5Aget_type(attr);
  size_t len = H5Tget_size(type);
  char *buf = new char[len + 1];
  buf[len] = '\0';
  H5Aread(attr, type, buf);
  std::string result(buf);
  delete[] buf;
  H5Tclose(type);
  H5Aclose(attr);
  return result;
}

//----------------------------------------------------------------------------------------
//! \fn bool CellDataIsFloatingPoint(hid_t file)
//! \brief reject integer-quantized cell data (upstream's u8/u16/... data_format), whose
//!        values are scaled by vmin/vmax and cannot be read as Reals

bool CellDataIsFloatingPoint(hid_t file) {
  if (!HasAttribute(file, "DatasetNames")) return true;  // nothing to check against
  hid_t attr = H5Aopen(file, "DatasetNames", H5P_DEFAULT);
  hid_t type = H5Aget_type(attr);
  hid_t space = H5Aget_space(attr);
  hsize_t dims[1] = {0};
  H5Sget_simple_extent_dims(space, dims, NULL);
  size_t len = H5Tget_size(type);
  int n = static_cast<int>(dims[0]);
  char *buf = new char[(n > 0 ? n : 1) * len];
  H5Aread(attr, type, buf);

  bool ok = true;
  for (int i = 0; i < n; ++i) {
    std::string name(buf + i * len, strnlen(buf + i * len, len));
    if (name.empty() || H5Lexists(file, name.c_str(), H5P_DEFAULT) <= 0) continue;
    hid_t dset = H5Dopen(file, name.c_str(), H5P_DEFAULT);
    hid_t dtype = H5Dget_type(dset);
    if (H5Tget_class(dtype) != H5T_FLOAT) ok = false;
    H5Tclose(dtype);
    H5Dclose(dset);
  }

  delete[] buf;
  H5Sclose(space);
  H5Tclose(type);
  H5Aclose(attr);
  return ok;
}

//----------------------------------------------------------------------------------------
//! \fn void ReadIntDataset(hid_t file, const char *name, T *out)
//! \brief read a whole integer dataset; HDF5 converts the big-endian on-disk type

void ReadIntDataset(hid_t file, const char *name, int *out) {
  hid_t dset = H5Dopen(file, name, H5P_DEFAULT);
  H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, out);
  H5Dclose(dset);
}

void ReadInt64Dataset(hid_t file, const char *name, std::int64_t *out) {
  hid_t dset = H5Dopen(file, name, H5P_DEFAULT);
  H5Dread(dset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, out);
  H5Dclose(dset);
}

void ReadRealDataset(hid_t file, const char *name, double *out) {
  hid_t dset = H5Dopen(file, name, H5P_DEFAULT);
  H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out);
  H5Dclose(dset);
}

//----------------------------------------------------------------------------------------
//! \fn int DatasetBlockCount(hid_t file, const char *name)
//! \brief length of a dataset's leading (block) dimension, or -1 if absent

int DatasetBlockCount(hid_t file, const char *name) {
  if (H5Lexists(file, name, H5P_DEFAULT) <= 0) return -1;
  hid_t dset = H5Dopen(file, name, H5P_DEFAULT);
  hid_t space = H5Dget_space(dset);
  hsize_t dims[8];
  int rank = H5Sget_simple_extent_ndims(space);
  H5Sget_simple_extent_dims(space, dims, NULL);
  int n = (rank > 0) ? static_cast<int>(dims[0]) : -1;
  H5Sclose(space);
  H5Dclose(dset);
  return n;
}

#endif  // HDF5OUTPUT

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn bool MCGridFile::Requested(ParameterInput *pin)
//! \brief is the grid to be taken from an athdf file?

bool MCGridFile::Requested(ParameterInput *pin) {
  if (!pin->DoesParameterExist(kGridFileBlock, kGridFileParam)) return false;
  return !pin->GetString(kGridFileBlock, kGridFileParam).empty();
}

//----------------------------------------------------------------------------------------
//! \fn MCGridFile *MCGridFile::Load(ParameterInput *pin)
//! \brief read and validate the snapshot named in the input file, caching the result

MCGridFile *MCGridFile::Load(ParameterInput *pin) {
  if (ploaded_ == nullptr)
    ploaded_ = new MCGridFile(pin->GetString(kGridFileBlock, kGridFileParam));
  return ploaded_;
}

MCGridFile *MCGridFile::Loaded() {
  return ploaded_;
}

void MCGridFile::Free() {
  delete ploaded_;
  ploaded_ = nullptr;
}

//----------------------------------------------------------------------------------------
//! \fn MCGridFile::MCGridFile(const std::string &fname)
//! \brief read the header of an athdf file and check that it describes a grid we can
//!        rebuild.  Every failure here is fatal: a grid that is silently wrong is far
//!        worse than a run that refuses to start.

MCGridFile::MCGridFile(const std::string &fname) :
    nbtotal(0), root_level(0), max_level(0),
    nrbx1(1), nrbx2(1), nrbx3(1), multilevel(false),
    filename(fname), loclist(nullptr), file_index_(nullptr) {
  ReadHeader();
  ValidateStructure();
  ValidateFaces();

  if (Globals::my_rank == 0) {
    std::cout << "Monte Carlo grid taken from " << filename << std::endl
              << "  root grid   = " << mesh_size.nx1 << " x " << mesh_size.nx2
              << " x " << mesh_size.nx3 << " cells, "
              << nrbx1 << " x " << nrbx2 << " x " << nrbx3 << " blocks" << std::endl
              << "  block size  = " << block_size.nx1 << " x " << block_size.nx2
              << " x " << block_size.nx3 << std::endl
              << "  blocks      = " << nbtotal << " on "
              << (max_level - root_level + 1) << " level(s)" << std::endl;
  }
}

MCGridFile::~MCGridFile() {
  delete[] loclist;
  delete[] file_index_;
}

//----------------------------------------------------------------------------------------
//! \fn void MCGridFile::ReadHeader()
//! \brief pull the mesh structure out of the file

void MCGridFile::ReadHeader() {
  std::stringstream msg;
#ifndef HDF5OUTPUT
  msg << "### FATAL ERROR in MCGridFile" << std::endl
      << "<" << kGridFileBlock << "> " << kGridFileParam << " requires HDF5."
      << std::endl << "Reconfigure with -hdf5." << std::endl;
  ATHENA_ERROR(msg);
#else
  hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) {
    msg << "### FATAL ERROR in MCGridFile" << std::endl
        << "Could not open " << filename << std::endl;
    ATHENA_ERROR(msg);
  }

  // The mesh metadata is optional in newer Athena++ writers (<output> mesh_data =
  // false).  Levels and LogicalLocations are always written, but without the root grid
  // extent there is nothing to rebuild the grid from.
  const char *required[] = {"RootGridX1", "RootGridX2", "RootGridX3", "RootGridSize",
                            "MeshBlockSize", "NumMeshBlocks", "Coordinates"};
  for (int n = 0; n < 7; ++n) {
    if (!HasAttribute(file, required[n])) {
      H5Fclose(file);
      msg << "### FATAL ERROR in MCGridFile" << std::endl
          << filename << " has no '" << required[n] << "' attribute." << std::endl
          << "It was probably written with <output> mesh_data = false, which omits the"
          << std::endl << "grid description.  Re-dump the snapshot with mesh_data = true."
          << std::endl;
      ATHENA_ERROR(msg);
    }
  }

  if (!CellDataIsFloatingPoint(file)) {
    H5Fclose(file);
    msg << "### FATAL ERROR in MCGridFile" << std::endl
        << filename << " stores cell data as integers." << std::endl
        << "It was written with an integer <output> data_format (u8/u16/u32/u64), whose"
        << std::endl << "values are scaled by vmin/vmax and cannot be read as Reals."
        << std::endl;
    ATHENA_ERROR(msg);
  }

  double rgx1[3], rgx2[3], rgx3[3];
  int rgsize[3], mbsize[3];
  ReadAttribute(file, "RootGridX1", 3, rgx1);
  ReadAttribute(file, "RootGridX2", 3, rgx2);
  ReadAttribute(file, "RootGridX3", 3, rgx3);
  ReadAttribute(file, "RootGridSize", 3, rgsize);
  ReadAttribute(file, "MeshBlockSize", 3, mbsize);
  ReadAttribute(file, "NumMeshBlocks", 1, &nbtotal);
  coordinates = ReadStringAttribute(file, "Coordinates");

  mesh_size.x1min = rgx1[0];  mesh_size.x1max = rgx1[1];  mesh_size.x1rat = rgx1[2];
  mesh_size.x2min = rgx2[0];  mesh_size.x2max = rgx2[1];  mesh_size.x2rat = rgx2[2];
  mesh_size.x3min = rgx3[0];  mesh_size.x3max = rgx3[1];  mesh_size.x3rat = rgx3[2];
  mesh_size.nx1 = rgsize[0];  mesh_size.nx2 = rgsize[1];  mesh_size.nx3 = rgsize[2];
  mesh_size.x1len = mesh_size.x1max - mesh_size.x1min;
  mesh_size.x2len = mesh_size.x2max - mesh_size.x2min;
  mesh_size.x3len = mesh_size.x3max - mesh_size.x3min;

  block_size = mesh_size;
  block_size.nx1 = mbsize[0];  block_size.nx2 = mbsize[1];  block_size.nx3 = mbsize[2];

  // block list; Levels holds the physical level, LogicalLocations the logical indices
  int nlev = DatasetBlockCount(file, "Levels");
  int nloc = DatasetBlockCount(file, "LogicalLocations");
  if (nlev != nbtotal || nloc != nbtotal) {
    H5Fclose(file);
    msg << "### FATAL ERROR in MCGridFile" << std::endl
        << filename << ": NumMeshBlocks = " << nbtotal << " but Levels has " << nlev
        << " and LogicalLocations has " << nloc << " entries." << std::endl;
    ATHENA_ERROR(msg);
  }

  int *levels = new int[nbtotal];
  std::int64_t *locs = new std::int64_t[3 * nbtotal];
  ReadIntDataset(file, "Levels", levels);
  ReadInt64Dataset(file, "LogicalLocations", locs);
  H5Fclose(file);

  // root_level is not stored; Mesh derives it from the root grid size in blocks, and so
  // must we, before the physical levels in the file can be made logical (mesh.cpp:341)
  nrbx1 = mesh_size.nx1 / block_size.nx1;
  nrbx2 = mesh_size.nx2 / block_size.nx2;
  nrbx3 = mesh_size.nx3 / block_size.nx3;
  if (nrbx1 * block_size.nx1 != mesh_size.nx1
      || nrbx2 * block_size.nx2 != mesh_size.nx2
      || nrbx3 * block_size.nx3 != mesh_size.nx3) {
    delete[] levels;
    delete[] locs;
    msg << "### FATAL ERROR in MCGridFile" << std::endl
        << filename << ": root grid (" << mesh_size.nx1 << ", " << mesh_size.nx2 << ", "
        << mesh_size.nx3 << ") is not divisible by the block size (" << block_size.nx1
        << ", " << block_size.nx2 << ", " << block_size.nx3 << ")." << std::endl
        << "The snapshot was probably written with <output> ghost_zones = true, which"
        << std::endl << "adds NGHOST to each reported block dimension." << std::endl;
    ATHENA_ERROR(msg);
  }

  std::int64_t nbmax = (nrbx1 > nrbx2) ? nrbx1 : nrbx2;
  nbmax = (nbmax > nrbx3) ? nbmax : nrbx3;
  for (root_level = 0; (1LL << root_level) < nbmax; root_level++) {}

  loclist = new LogicalLocation[nbtotal];
  max_level = root_level;
  for (int i = 0; i < nbtotal; ++i) {
    loclist[i].level = levels[i] + root_level;
    loclist[i].lx1 = locs[3*i + 0];
    loclist[i].lx2 = locs[3*i + 1];
    loclist[i].lx3 = locs[3*i + 2];
    if (loclist[i].level > max_level) max_level = loclist[i].level;
  }
  multilevel = (max_level > root_level);

  delete[] levels;
  delete[] locs;
#endif  // HDF5OUTPUT
}

//----------------------------------------------------------------------------------------
//! \fn void MCGridFile::ValidateStructure() const
//! \brief check that the block list describes a grid we can actually rebuild

void MCGridFile::ValidateStructure() const {
  std::stringstream msg;

  if (nbtotal < 1) {
    msg << "### FATAL ERROR in MCGridFile" << std::endl
        << filename << " contains no MeshBlocks." << std::endl;
    ATHENA_ERROR(msg);
  }

  // A sliced or summed output writes only the blocks intersecting the slice, and the
  // block index in the file then no longer matches the gid.  Both collapse one
  // dimension to a single cell while the root grid keeps its full size.
  const int nxb[3] = {block_size.nx1, block_size.nx2, block_size.nx3};
  const int nxm[3] = {mesh_size.nx1, mesh_size.nx2, mesh_size.nx3};
  for (int d = 0; d < 3; ++d) {
    if (nxb[d] == 1 && nxm[d] > 1) {
      msg << "### FATAL ERROR in MCGridFile" << std::endl
          << filename << ": block size is 1 in x" << (d+1) << " while the root grid has "
          << nxm[d] << " cells." << std::endl
          << "This is a sliced or summed output.  It holds only the blocks intersecting"
          << std::endl << "the slice, so the grid cannot be rebuilt from it.  Use a full"
          << " dump." << std::endl;
      ATHENA_ERROR(msg);
    }
  }

  if (coordinates != std::string(COORDINATE_SYSTEM)) {
    msg << "### FATAL ERROR in MCGridFile" << std::endl
        << filename << " was written with coordinates '" << coordinates
        << "' but this build is configured for '" << COORDINATE_SYSTEM << "'."
        << std::endl << "Reconfigure with --coord=" << coordinates << "." << std::endl;
    ATHENA_ERROR(msg);
  }

  // logical locations must lie inside the root grid at their own level
  for (int i = 0; i < nbtotal; ++i) {
    int shift = loclist[i].level - root_level;
    if (shift < 0) {
      msg << "### FATAL ERROR in MCGridFile" << std::endl
          << filename << ": block " << i << " has level below the root level."
          << std::endl;
      ATHENA_ERROR(msg);
    }
    const std::int64_t lim[3] = {nrbx1 << shift, nrbx2 << shift, nrbx3 << shift};
    const std::int64_t lx[3] = {loclist[i].lx1, loclist[i].lx2, loclist[i].lx3};
    for (int d = 0; d < 3; ++d) {
      if (lx[d] < 0 || lx[d] >= lim[d]) {
        msg << "### FATAL ERROR in MCGridFile" << std::endl
            << filename << ": block " << i << " has logical location lx" << (d+1)
            << " = " << lx[d] << " outside [0, " << lim[d] << ") at level "
            << loclist[i].level << "." << std::endl;
        ATHENA_ERROR(msg);
      }
    }
  }

  // the leaves must tile the root grid exactly: no gaps, no overlaps.  Count in units
  // of finest-level blocks, skipping the check if that count would overflow.
  int ndim = 1;
  if (mesh_size.nx2 > 1) ndim++;
  if (mesh_size.nx3 > 1) ndim++;
  const int maxshift = max_level - root_level;
  if (static_cast<std::int64_t>(ndim) * (maxshift + 3) < 62) {
    std::int64_t covered = 0;
    for (int i = 0; i < nbtotal; ++i) {
      const int s = maxshift - (loclist[i].level - root_level);
      std::int64_t v = 1;
      for (int d = 0; d < ndim; ++d) v <<= s;
      covered += v;
    }
    std::int64_t full = nrbx1 * nrbx2 * nrbx3;
    for (int d = 0; d < ndim; ++d) full <<= maxshift;
    if (covered != full) {
      msg << "### FATAL ERROR in MCGridFile" << std::endl
          << filename << ": the " << nbtotal << " blocks cover " << covered
          << " finest-level cells but the root grid holds " << full << "." << std::endl
          << "The block list has gaps or overlaps and does not describe a valid tree."
          << std::endl;
      ATHENA_ERROR(msg);
    }
  }

  // With more than one level the Mesh constructor imposes a floor on the block size
  // that depends on this build's NGHOST, not the source run's (mesh.cpp:300).
  if (multilevel) {
    const bool active[3] = {true, mesh_size.nx2 > 1, mesh_size.nx3 > 1};
    for (int d = 0; d < 3; ++d) {
      if (active[d] && nxb[d] < NGHOST) {
        msg << "### FATAL ERROR in MCGridFile" << std::endl
            << filename << " is refined and has block size " << nxb[d] << " in x"
            << (d+1) << ", below this build's NGHOST = " << NGHOST << "." << std::endl
            << "Reconfigure with a smaller --nghost to read this snapshot." << std::endl;
        ATHENA_ERROR(msg);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real MCGridFile::MeshPosition(int dir, std::int64_t index,
//!                                   std::int64_t nrange) const
//! \brief position of global logical face `index` of `nrange`, reproducing the mesh
//!        generator Mesh will use.  Mirrors ComputeMeshGeneratorX() composed with
//!        Uniform/DefaultMeshGeneratorX* in mesh.hpp.

Real MCGridFile::MeshPosition(int dir, std::int64_t index, std::int64_t nrange) const {
  Real xmin, xmax, rat;
  int nx;
  if (dir == 0) {
    xmin = mesh_size.x1min; xmax = mesh_size.x1max;
    rat = mesh_size.x1rat;  nx = mesh_size.nx1;
  } else if (dir == 1) {
    xmin = mesh_size.x2min; xmax = mesh_size.x2max;
    rat = mesh_size.x2rat;  nx = mesh_size.nx2;
  } else {
    xmin = mesh_size.x3min; xmax = mesh_size.x3max;
    rat = mesh_size.x3rat;  nx = mesh_size.nx3;
  }

  if (rat == 1.0) {
    // uniform generator, evaluated on the symmetric [-0.5, 0.5] interval
    std::int64_t noffset = index - nrange/2;
    std::int64_t noffset_ceil = index - (nrange + 1)/2;
    Real x = static_cast<Real>(noffset + noffset_ceil)/(2.0*nrange);
    return static_cast<Real>(0.5)*(xmin + xmax) + (x*xmax - x*xmin);
  }
  // ratioed generator, evaluated on [0, 1]
  Real x = static_cast<Real>(index)/static_cast<Real>(nrange);
  Real ratn = std::pow(rat, static_cast<Real>(nx));
  Real rnx = std::pow(rat, x*nx);
  Real lw = (rnx - ratn)/(1.0 - ratn);
  Real rw = 1.0 - lw;
  return xmin*lw + xmax*rw;
}

//----------------------------------------------------------------------------------------
//! \fn void MCGridFile::ValidateFaces() const
//! \brief compare the cell faces Mesh would regenerate against the ones stored in the
//!        file.  This is the check that catches a source run using a user-defined mesh
//!        generator, which the file records no trace of: the recovered x1min/x1max/x1rat
//!        would then produce a different grid with no other symptom.

void MCGridFile::ValidateFaces() const {
#ifdef HDF5OUTPUT
  std::stringstream msg;
  hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) return;  // already reported by ReadHeader

  const char *fname[3] = {"x1f", "x2f", "x3f"};
  const int nxb[3] = {block_size.nx1, block_size.nx2, block_size.nx3};
  const std::int64_t nrb[3] = {nrbx1, nrbx2, nrbx3};
  const Real len[3] = {mesh_size.x1len, mesh_size.x2len, mesh_size.x3len};

  for (int d = 0; d < 3; ++d) {
    if (DatasetBlockCount(file, fname[d]) != nbtotal) continue;  // mesh_data = false
    const int nf = nxb[d] + 1;
    double *stored = new double[static_cast<size_t>(nbtotal)*nf];
    ReadRealDataset(file, fname[d], stored);

    Real worst = 0.0;
    int worst_block = -1;
    for (int i = 0; i < nbtotal; ++i) {
      const int shift = loclist[i].level - root_level;
      const std::int64_t nrbx_ll = nrb[d] << shift;
      const std::int64_t lx = (d == 0) ? loclist[i].lx1
                            : (d == 1) ? loclist[i].lx2 : loclist[i].lx3;
      for (int k = 0; k < nf; ++k) {
        Real rec = MeshPosition(d, lx*nxb[d] + k, nrbx_ll*nxb[d]);
        Real err = std::fabs(rec - stored[static_cast<size_t>(i)*nf + k]);
        if (len[d] > 0.0) err /= len[d];
        if (err > worst) {
          worst = err;
          worst_block = i;
        }
      }
    }
    delete[] stored;

    if (worst > kFaceTol) {
      H5Fclose(file);
      msg << "### FATAL ERROR in MCGridFile" << std::endl
          << filename << ": regenerated x" << (d+1) << " faces disagree with the file by"
          << std::endl << "  " << worst << " (relative), worst at block " << worst_block
          << "; tolerance is " << kFaceTol << "." << std::endl
          << "The source run most likely used a user-defined mesh generator, which the"
          << std::endl << "athdf file does not record.  The grid cannot be rebuilt from"
          << " the file alone." << std::endl;
      ATHENA_ERROR(msg);
    }
  }
  H5Fclose(file);
#endif  // HDF5OUTPUT
}

//----------------------------------------------------------------------------------------
//! \fn void MCGridFile::SetRealExact(ParameterInput *pin, const char *block,
//!                                   const char *name, Real value)
//! \brief store a Real in the input deck without losing precision.
//!
//! ParameterInput::SetReal formats through a default ostringstream, which keeps only 6
//! significant digits.  That is harmless for the grid extent but not for x1rat: the mesh
//! generator evaluates rat^(x*nx), so a rounded ratio is amplified by nx in the exponent.
//! A logarithmic radial grid with rat = 1.0812277 and nx1 = 64 loses 2.1e-6 in the ratio
//! and about 1e-4 in the face positions, which is well outside the tolerance the faces
//! were just validated against.  Go through SetString instead, which stores the text
//! verbatim for GetReal to parse back.

void MCGridFile::SetRealExact(ParameterInput *pin, const char *block, const char *name,
                              Real value) {
  std::ostringstream ss;
  ss << std::setprecision(std::numeric_limits<Real>::max_digits10) << value;
  pin->SetString(block, name, ss.str());
}

//----------------------------------------------------------------------------------------
//! \fn void MCGridFile::InjectMeshParameters(ParameterInput *pin) const
//! \brief overwrite the <mesh> and <meshblock> grid description with the file's, so the
//!        Mesh constructor builds the snapshot's grid.  Must run before Mesh is built.
//!
//! ParameterInput::Set* creates the block and the parameters if they do not exist, so an
//! input file that names a snapshot needs no <mesh> block of its own.

void MCGridFile::InjectMeshParameters(ParameterInput *pin) const {
  SetRealExact(pin, "mesh", "x1min", mesh_size.x1min);
  SetRealExact(pin, "mesh", "x1max", mesh_size.x1max);
  SetRealExact(pin, "mesh", "x1rat", mesh_size.x1rat);
  SetRealExact(pin, "mesh", "x2min", mesh_size.x2min);
  SetRealExact(pin, "mesh", "x2max", mesh_size.x2max);
  SetRealExact(pin, "mesh", "x2rat", mesh_size.x2rat);
  SetRealExact(pin, "mesh", "x3min", mesh_size.x3min);
  SetRealExact(pin, "mesh", "x3max", mesh_size.x3max);
  SetRealExact(pin, "mesh", "x3rat", mesh_size.x3rat);
  pin->SetInteger("mesh", "nx1", mesh_size.nx1);
  pin->SetInteger("mesh", "nx2", mesh_size.nx2);
  pin->SetInteger("mesh", "nx3", mesh_size.nx3);

  pin->SetInteger("meshblock", "nx1", block_size.nx1);
  pin->SetInteger("meshblock", "nx2", block_size.nx2);
  pin->SetInteger("meshblock", "nx3", block_size.nx3);

  // A refined snapshot is replayed from the file's block list rather than refined during
  // the run, so the mesh is static.  The tree replay itself lands in a later phase; a
  // multilevel file cannot be built correctly until then.
  if (multilevel) {
    pin->SetString("mesh", "refinement", "static");
    std::stringstream msg;
    msg << "### FATAL ERROR in MCGridFile" << std::endl
        << filename << " holds " << (max_level - root_level + 1) << " refinement levels."
        << std::endl
        << "Rebuilding a refined tree from an athdf file is not implemented yet; only"
        << std::endl << "uniform snapshots can be read this way so far." << std::endl;
    ATHENA_ERROR(msg);
  }
  pin->SetString("mesh", "refinement", "none");
}

//----------------------------------------------------------------------------------------
//! \fn void MCGridFile::MapBlocks(const LogicalLocation *mesh_loclist, int nbtotal_mesh)
//! \brief build the gid -> file block index map by matching logical locations.  The two
//!        orderings agree whenever the tree is replayed in file order, but nothing in
//!        the file format guarantees that, so it is matched rather than assumed.

void MCGridFile::MapBlocks(const LogicalLocation *mesh_loclist, int nbtotal_mesh) {
  std::stringstream msg;
  if (nbtotal_mesh != nbtotal) {
    msg << "### FATAL ERROR in MCGridFile" << std::endl
        << "Mesh has " << nbtotal_mesh << " blocks but " << filename << " holds "
        << nbtotal << "." << std::endl;
    ATHENA_ERROR(msg);
  }

  delete[] file_index_;
  file_index_ = new int[nbtotal];
  for (int gid = 0; gid < nbtotal; ++gid) file_index_[gid] = -1;

  // nbtotal is at most a few 1e5 in practice; the linear scan runs once at setup.  If
  // that ever becomes a bottleneck, sort the file's locations and bisect instead.
  for (int gid = 0; gid < nbtotal; ++gid) {
    for (int i = 0; i < nbtotal; ++i) {
      if (loclist[i].level == mesh_loclist[gid].level
          && loclist[i].lx1 == mesh_loclist[gid].lx1
          && loclist[i].lx2 == mesh_loclist[gid].lx2
          && loclist[i].lx3 == mesh_loclist[gid].lx3) {
        file_index_[gid] = i;
        break;
      }
    }
    if (file_index_[gid] < 0) {
      msg << "### FATAL ERROR in MCGridFile" << std::endl
          << "Mesh block gid = " << gid << " (level " << mesh_loclist[gid].level
          << ", lx = " << mesh_loclist[gid].lx1 << ", " << mesh_loclist[gid].lx2 << ", "
          << mesh_loclist[gid].lx3 << ") has no counterpart in " << filename << "."
          << std::endl;
      ATHENA_ERROR(msg);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn int MCGridFile::FileIndex(int gid) const
//! \brief index along the file's block dimension for a Mesh gid

int MCGridFile::FileIndex(int gid) const {
  if (file_index_ == nullptr) return gid;  // not mapped; file order is gid order
  return file_index_[gid];
}
