#ifndef MCPARTITION_HPP
#define MCPARTITION_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mcpartition.hpp
//! \brief optimal contiguous partition of a block cost list over ranks
//
// The mesh assigns blocks to ranks as consecutive ranges of the gid sequence (the
// Z-order curve), and Mesh::CalculateLoadBalance chooses those ranges greedily: it fills
// ranks from the last one down, closing a range as soon as it reaches the running fair
// share.  With few blocks per rank that overshoots, and the partition it proposes can be
// worse than the layout it replaces.
//
// This finds the contiguous partition that minimizes the largest range sum exactly.  The
// smallest feasible capacity is bracketed between the largest single cost and the total
// and found by bisection, each probe being one left-to-right sweep that opens a new
// range whenever the next block would overflow the capacity; a partition at that
// capacity is then laid out the same way, and ranges are split until every rank holds
// at least one block, which the mesh requires and which cannot raise the maximum.

#include "../athena.hpp"

//! rlist[i] = rank of block i; slist[r] and nlist[r] = first block and block count of
//! rank r.  Requires nb >= nranks and non-negative costs.
void OptimalContiguousPartition(const double *cost, int nb, int nranks, int *rlist,
                                int *slist, int *nlist);

//! the largest range sum of a partition given by slist/nlist, for reports and tests
double PartitionMaxCost(const double *cost, int nranks, const int *slist,
                        const int *nlist);

#endif // MCPARTITION_HPP
