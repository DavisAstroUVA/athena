#ifndef MCSNAPSHOT_HPP
#define MCSNAPSHOT_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mcsnapshot.hpp
//! \brief reads one MeshBlock's cell data out of an athdf snapshot
//
// Companion to mcgrid.hpp, which recovers the grid: this recovers what sits on it.  The
// two are independent.  A run may rebuild its grid from the file and then read the data,
// rebuild the grid and fill it some other way, or describe the grid by hand in the input
// file and still read the data here -- which is what the problem generators did before
// mcgrid existed, and still do when <montecarlo>/grid_from_file is unset.
//
// Variables are located by name from the file's own VariableNames attribute rather than
// by hand-entered indices, so a snapshot written by Athena++ (rho, press, vel1) and one
// converted from an AthenaK run (dens, eint, velx) are both read without the input file
// having to describe the layout.

// C++ headers
#include <string>

// Athena++ headers
#include "../athena.hpp"

// Forward declarations
class MeshBlock;
class ParameterInput;

// How the snapshot stores the fluid state.  Only primitives are implemented; the
// conserved case is named here so that the input parameter, the dispatch and the error
// message already exist when a file that needs it turns up.
enum MCSnapshotVars {
  MCSNAP_PRIMITIVE = 0,
  MCSNAP_CONSERVED = 1
};

// spelling of a mode in <problem>/snapshot_vars, and its inverse for messages
MCSnapshotVars GetMCSnapshotVarsFlag(const std::string &name);
const char *GetMCSnapshotVarsName(MCSnapshotVars v);

// Fill this block's primitives from the snapshot named by <problem>/input_filename,
// falling back to <montecarlo>/grid_from_file.  Fills pfield->bcc as well when the build
// has magnetic fields and the file carries a cell-centred field.  Does not convert to
// conserved variables: the caller does that once it has finished with the primitives.
//
// max_blocks_per_rank is the largest number of MeshBlocks held by any rank.  It is only
// used to pad collective reads so that every rank issues the same number of them, and it
// has to be passed in because it comes from Mesh::nblist, which is private to everything
// except MeshBlock -- that is, to the problem generator calling this.  Pass 0 to skip the
// padding, which is safe whenever <problem>/collective is false.
void MCReadSnapshotBlock(MeshBlock *pmb, ParameterInput *pin, int max_blocks_per_rank);

#endif  // MCSNAPSHOT_HPP
