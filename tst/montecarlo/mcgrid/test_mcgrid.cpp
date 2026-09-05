//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file test_mcgrid.cpp
//! \brief standalone driver for MCGridFile
//!
//! Constructs an MCGridFile on the athdf file named on the command line and reports
//! either the recovered mesh structure or the error it was rejected with.  Built and run
//! by run_tests.py; not part of bin/athena.

#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include "../../../src/athena.hpp"
#include "../../../src/globals.hpp"
#include "../../../src/monte_carlo/mcgrid.hpp"

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cout << "usage: test_mcgrid <file.athdf>" << std::endl;
    return 2;
  }
  Globals::my_rank = 0;
  Globals::nranks = 1;

  try {
    MCGridFile grid(argv[1]);
    std::cout << "OK"
              << " nbtotal=" << grid.nbtotal
              << " root_level=" << grid.root_level
              << " max_level=" << grid.max_level
              << " nrbx=" << grid.nrbx1 << "," << grid.nrbx2 << "," << grid.nrbx3
              << " multilevel=" << (grid.multilevel ? 1 : 0)
              << " coord=" << grid.coordinates
              << std::setprecision(std::numeric_limits<Real>::max_digits10)
              << " x2min=" << grid.mesh_size.x2min
              << " x2max=" << grid.mesh_size.x2max
              << " x3min=" << grid.mesh_size.x3min
              << " x3max=" << grid.mesh_size.x3max
              << std::endl;
  } catch (std::exception const &ex) {
    std::cout << "REJECTED" << std::endl << ex.what() << std::endl;
    return 1;
  }
  return 0;
}
