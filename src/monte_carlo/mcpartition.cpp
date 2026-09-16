//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mcpartition.cpp
//! \brief optimal contiguous partition of a block cost list; see the header

// C++ headers
#include <algorithm>  // max
#include <sstream>
#include <stdexcept>
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "mcpartition.hpp"

namespace {

//! Sweep left to right opening a new range whenever the next block would push the
//! current one past cap.  Returns the number of ranges; fills starts if given.
int SweepAtCapacity(const double *cost, int nb, double cap, std::vector<int> *starts) {
  int nranges = 0;
  double running = 0.0;
  for (int i=0; i<nb; ++i) {
    if (nranges == 0 || running + cost[i] > cap) {
      ++nranges;
      running = 0.0;
      if (starts != nullptr) starts->push_back(i);
    }
    running += cost[i];
  }
  return nranges;
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn void OptimalContiguousPartition(...)

void OptimalContiguousPartition(const double *cost, int nb, int nranks, int *rlist,
                                int *slist, int *nlist) {
  if (nb < nranks || nranks < 1) {
    std::stringstream msg;
    msg << "### FATAL ERROR in OptimalContiguousPartition" << std::endl
        << nb << " blocks cannot give every one of " << nranks << " ranks a block"
        << std::endl;
    ATHENA_ERROR(msg);
  }

  // The capacity is bracketed by the largest single block (a range holds at least that)
  // and the total (one range holds everything).  Feasibility -- a sweep at that
  // capacity needs no more than nranks ranges -- is monotone in the capacity, so
  // bisect; the sweep is exact for a given capacity, so the answer is the smallest
  // capacity at which the sweep fits, up to the bisection's resolution.
  double lo = 0.0, hi = 0.0;
  for (int i=0; i<nb; ++i) {
    lo = std::max(lo, cost[i]);
    hi += cost[i];
  }
  for (int iter=0; iter<200 && hi - lo > 1.0e-12*hi; ++iter) {
    const double mid = 0.5*(lo + hi);
    if (SweepAtCapacity(cost, nb, mid, nullptr) <= nranks) hi = mid;
    else lo = mid;
  }

  std::vector<int> starts;
  SweepAtCapacity(cost, nb, hi, &starts);
  starts.push_back(nb);   // sentinel: end of the last range

  // The sweep may use fewer ranges than ranks.  Split the range with the most blocks
  // at its cost midpoint until every rank has one; a split never raises the maximum.
  while (static_cast<int>(starts.size()) - 1 < nranks) {
    int widest = -1, wid = 1;
    for (std::size_t r=0; r+1<starts.size(); ++r) {
      const int w = starts[r+1] - starts[r];
      if (w > wid) { wid = w; widest = static_cast<int>(r); }
    }
    if (widest < 0) break;   // cannot happen with nb >= nranks
    const int s = starts[widest], e = starts[widest+1];
    double total = 0.0;
    for (int i=s; i<e; ++i) total += cost[i];
    double running = 0.0;
    int cut = s + 1;
    for (int i=s; i<e-1; ++i) {
      running += cost[i];
      cut = i + 1;
      if (running >= 0.5*total) break;
    }
    starts.insert(starts.begin() + widest + 1, cut);
  }

  for (int r=0; r<nranks; ++r) {
    slist[r] = starts[r];
    nlist[r] = starts[r+1] - starts[r];
    for (int i=starts[r]; i<starts[r+1]; ++i) rlist[i] = r;
  }
}

//----------------------------------------------------------------------------------------
//! \fn double PartitionMaxCost(...)

double PartitionMaxCost(const double *cost, int nranks, const int *slist,
                        const int *nlist) {
  double worst = 0.0;
  for (int r=0; r<nranks; ++r) {
    double rc = 0.0;
    for (int n=slist[r]; n<slist[r] + nlist[r]; ++n) rc += cost[n];
    worst = std::max(worst, rc);
  }
  return worst;
}
