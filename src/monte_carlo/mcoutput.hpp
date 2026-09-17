#ifndef MCOUTPUT_HPP
#define MCOUTPUT_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file photon.hpp
//! \brief definitions for MCOutput and output classes

// Athena++ classes headers
#include <cstdint>
#include <cstdio>
#include <string>

#include "../athena.hpp"
#include "montecarlo.hpp"
#include "polarization.hpp"  // MCPolarization; no includes of its own, so no cycle

class Photon;

//----------------------------------------------------------------------------------------
//! \struct MomentumRange
//! \brief physical ranges of photon momentum grid

typedef struct MomentumRange {
  int ne, ncth, nphi;
  Real emin, emax;
  Real phimin, phimax;
  Real cthmin, cthmax;

} MomentumRange;

//----------------------------------------------------------------------------------------
//! \class Spectrum
//! \brief spectral output

class Spectrum {
public:
  Spectrum(MomentumRange input_range, MCPolarization polarized, bool logarithmic);
  Spectrum(Spectrum *pspec);
  ~Spectrum();

  MonteCarlo *pmy_mc;
  std::int64_t nsrun;  // total number of photons samples run for this spectrum
  std::string base_name;
  MomentumRange range;
  MCPolarization polarized;
  bool polar_axis;
  bool coordinates;
  bool logarithmic;

  Spectrum *next;  // next spectrum
  enum BoundaryFace face;
  int id; // spectrum id -- maybe multiple spectra requested
  int output_number;// current output number
  Real x1min,x1max,x2min,x2max,x3min,x3max;
  Real dt; // targe integration time for this spectrum
  Real last_time;

  AthenaArray<Real> energies;
  AthenaArray<Real> intensity;
  AthenaArray<Real> intensity_sq;
  //! Stokes planes stored alongside the intensity, indexed by MCISQ/MCISU/MCISV.  Only
  //! the first NumStokesStored(polarized) of them are allocated.
  AthenaArray<Real> stokes[3];
  AthenaArray<Real> stokes_sq[3];

  //functions
  void BuildEnergyGrid(Real emin, Real emax, int nen, bool xlog);
  void UpdateSpectrum(Photon *pphot, int ip);
  int EnergyBin(Real energy);
  int EnergyBinUniform(Real energy, bool loge);
  bool AngleBinsCartesian(Real k[4], int &Phibin, int &mubin);
  bool AngleBinsSphericalPolar(Real k[4], int &Phibin, int &mubin);
  void SetSurface(std::string input_face);
  enum BoundaryFace GetPhotonFace(Photon *phot, int ip);
  bool ScreenCoordinates(Photon *pphot, int ip);
  void ResetSpectrum();
  void AddSpectrum(Spectrum *pspec);
  void WriteSpectrum(std::string filename, Real tint_out);

};

//----------------------------------------------------------------------------------------
//! \fn Real PhotonEnergyAtInfinity(Photon *pphot, int ip)
//! \brief conserved photon energy -k_t, valid for any stationary asymptotically flat
//! metric.  Used for the energy column of relativistic list output.

Real PhotonEnergyAtInfinity(Photon *pphot, int ip);

//----------------------------------------------------------------------------------------
//! \class PhotonList
//! \brief List of output Photon properties

class PhotonList {
public:
  PhotonList(int list_size_init, MCPolarization pol, int nuser, int max_res);
  ~PhotonList();

  MonteCarlo *pmy_mc;
  std::string base_name;

  std::int64_t nsrun;  // total number of photons samples run for this list
  int length; // number of occupied elements
  int nparams; // number of properties for each photon in list
  int output_number;// current output number
  int nuser_out;
  MCPolarization polarized;
  Real dt; // targe integration time for this spectrum
  Real last_time;
  AthenaArray<Real> photons;  // array of photon properies

  //functions
  void AddPhoton(Photon *pphot, int ip);
  void WriteList(std::string filename, Real tint_out);
  void ResetList();
  //! the file this list writes to, so the spill path and the final write cannot disagree
  std::string Filename() const;

private:
  int len_limit;  // number of photons allowed with current allocated memory
  void ResizeList(int new_size);

  //! Incremental output.  The list spills to the file whenever it reaches max_resident,
  //! and streams through a small fixed buffer. WriteList seeks back to correct the header
  //! fields that were notknown when the header was written.
  void OpenAndWriteHeader(const std::string &filename, Real tint_out);
  void StreamResident();
  void SpillToFile();

  FILE *fp_;                  //!> open between the first spill and WriteList
  std::int64_t nwritten_;     //!> photons already on disk
  long dt_pos_, length_pos_, ntot_pos_;  //!> offsets of the padded header fields
  int max_resident;           //!> photons held in memory before spilling

};

//----------------------------------------------------------------------------------------
//! \class PhotonTrajectoryList
//  \brief List of photon trajectories

class PhotonTrajectoryList {
public:
  PhotonTrajectoryList(int init_len_limit, int init_step_limit, int nuser);
  ~PhotonTrajectoryList();

  MonteCarlo *pmy_mc;
  std::string base_name;

  int length; // number of trajectories
  int maxstep;
  int nparams; // number of properties for each photon trajectory
  int output_number;// current output number
  int nuser_out;

  int *nsteps;  // step number for each trajectory
  AthenaArray<Real> trajectories;  // array of photon properies

  //functions
  void InitializeTrajectory(int itraj);
  void CompleteTrajectory(int itraj);
  void AddToTrajectory(Photon *pphot, int ip);
  void WriteList(std::string filename);
  //void ResetList();

private:
  int len_limit;  // number of trajectories allowed with current allocated memory
  int step_limit; // maximum number of steps
  void ResizeList(int new_len_limit, int new_step_limit);

};

//----------------------------------------------------------------------------------------
//! \class Image
//  \brief Image created using ray traced photons (SWD: in progress)

class Image {
public:
  Image(int list_mem_size, MCPolarization pol, bool rel, int nuser);
  ~Image();

  std::string base_name;

  int nx; // number of horizontal pixels
  int ny; // number of vertical pixels
  Real xcam[4]; // position of camera (tetrad)
  Real kcam[4]; // Camera direction
  Real xmin, xmax; // horizontal angular extent of image
  Real ymin, ymax; // vertical angular extent of image
  int nparams; // number of properties stored for each pixel
  int output_number;// current output number
  int nuser_out;
  MCPolarization polarized;
  bool relativistic;
  AthenaArray<Real> image;  // pixel array

  //functions
  void WriteImage(std::string filename);

  //private:

};

// SWD: Will need to be modifed with new parallelization scheme
//----------------------------------------------------------------------------------------
//! \class MCOutput
//! \brief class for managing monte carlo specific spectral outputs

class MCOutput {
public:
  MCOutput(MonteCarlo *pmc, ParameterInput *pin);
  ~MCOutput();

  MonteCarlo *pmy_mc;
  Spectrum *pspec;
  PhotonList *pphlist;
  PhotonTrajectoryList *ptraj;

  bool mom_flag_lab;
  bool mom_flag_com;
  bool mom_flag_coord;
  bool mom_flag_src;
  bool mom_flag_usr;
  bool mom_flag_scat;
  
  Real time_cgs;

  //functions
  void OutputSpectrum(bool wtflag);
  void SendMonteCarloSpectrum(Spectrum *spect, int dest);
  void ReceiveMonteCarloSpectrum(Spectrum *spect, bool add);
  void OutputPhotonList(bool wtflag);
  void OutputTrajectoryList();
  void UpdateOutputCount(std::int64_t nph);
  void MakeOutputs(bool wtflag);
};

#endif //MC_OUTPUT
