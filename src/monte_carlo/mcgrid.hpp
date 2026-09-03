#ifndef MCGRID_HPP
#define MCGRID_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mcgrid.hpp
//! \brief definitions for MCGridFile, which recovers the mesh structure of an athdf
//!        snapshot so the Monte Carlo run can be built on the same grid
//
// An athdf file records everything needed to rebuild the block tree it was written
// from: the root grid extent and size, the block size, and the level and logical
// location of every block.  This is the same information the restart file carries in
// its ID list, so the tree can be replayed exactly as Mesh's restart constructor
// replays it.
//
// The class does not touch the Mesh.  It reads and validates, and hands back a plain
// description that the caller uses: ParameterInput injection before Mesh is built, and
// (later) the block list the Mesh constructor replays into the tree.

// C++ headers
#include <cstdint>
#include <string>

// Athena++ headers
#include "../athena.hpp"  // Real, RegionSize, LogicalLocation

// Forward declarations
class ParameterInput;

//----------------------------------------------------------------------------------------
//! \class MCGridFile
//! \brief mesh structure recovered from an athdf snapshot

class MCGridFile {
 public:
  explicit MCGridFile(const std::string &filename);
  ~MCGridFile();

  // is <montecarlo> grid_from_file set?  Safe to call in a non-MC build.
  static bool Requested(ParameterInput *pin);

  // read the file named by <montecarlo> grid_from_file, validate it, and cache it.
  // Repeated calls return the cached instance rather than re-reading.
  static MCGridFile *Load(ParameterInput *pin);

  // the cached instance, or nullptr if Load() has not been called
  static MCGridFile *Loaded();

  // release the cached instance
  static void Free();

  // overwrite the <mesh> and <meshblock> grid parameters with the values from the
  // file.  Must run before the Mesh constructor, which reads them in its member
  // initializer list.
  void InjectMeshParameters(ParameterInput *pin) const;

  // index into the file's block dimension for a given Mesh gid.  The mapping is the
  // identity whenever the tree is replayed in file order, but it is built by matching
  // logical locations rather than assumed.
  int FileIndex(int gid) const;

  // fill file_index_ by matching loclist against the Mesh's block list
  void MapBlocks(const LogicalLocation *mesh_loclist, int nbtotal_mesh);

  // recovered mesh structure
  RegionSize mesh_size;          // root grid extent and size
  RegionSize block_size;         // MeshBlock size (cell counts only)
  int nbtotal;                   // number of blocks in the file
  int root_level;                // logical level of the root grid
  int max_level;                 // logical level of the finest block
  std::int64_t nrbx1, nrbx2, nrbx3;  // root grid size in blocks
  bool multilevel;               // true if the file holds more than one level
  std::string coordinates;       // COORDINATE_SYSTEM the file was written with
  std::string filename;

  // block list, in file order, with levels converted to logical levels
  LogicalLocation *loclist;

 private:
  int *file_index_;              // gid -> index along the file's block dimension

  void ReadHeader();
  void ValidateStructure() const;
  void ValidateFaces() const;

  // reproduce Mesh's face position for one global logical face index
  Real MeshPosition(int dir, std::int64_t index, std::int64_t nrange) const;

  // store a Real in the input deck at full precision.  ParameterInput::SetReal keeps
  // only 6 significant digits, which is not enough for x1rat.
  static void SetRealExact(ParameterInput *pin, const char *block, const char *name,
                           Real value);

  static MCGridFile *ploaded_;
};

#endif  // MCGRID_HPP
