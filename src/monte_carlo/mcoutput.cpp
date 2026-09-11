//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mcoutput.cpp
//! \brief implementation of functions in class MCOutput

// C++ headers
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>    // int64_t
#include <vector>
#include <stdexcept>  // runtime_error
#include <iomanip>    // setfill(), setw()
#include <errno.h>

// Athena++ headers
#include "montecarlo.hpp"
#include "mccoord.hpp"
#include "mcoutput.hpp"
#include "photonpusher.hpp"
#include "tetrad.hpp"
#include "../globals.hpp"
#include "../outputs/io_wrapper.hpp"
#include "../utils/buffer_utils.hpp"

namespace mcoutput {
//----------------------------------------------------------------------------------------
// Functions to detect big endian machine, and to byte-swap 32-bit words.

  int IsBigEndian(void) {
    int32_t n = 1;
    // careful! although int -> char * -> int round-trip conversion is safe,
    // an arbitrary char* may not be converted to int*
    char *ep = reinterpret_cast<char *>(&n);
    return (*ep == 0); // Returns 1 (true) on a big endian machine
  }

  static inline void Swap4Bytes(void *vdat) {
    char tmp, *dat = static_cast<char *>(vdat);
    tmp = dat[0];  dat[0] = dat[3];  dat[3] = tmp;
    tmp = dat[1];  dat[1] = dat[2];  dat[2] = tmp;
  }
  static inline void Swap8Bytes(void *vdat) {
    char tmp, *dat = static_cast<char *>(vdat);
    tmp = dat[0];  dat[0] = dat[7];  dat[7] = tmp;
    tmp = dat[1];  dat[1] = dat[6];  dat[6] = tmp;
    tmp = dat[2];  dat[2] = dat[5];  dat[5] = tmp;
    tmp = dat[3];  dat[3] = dat[4];  dat[4] = tmp;
  }
}

//----------------------------------------------------------------------------------------
//! Spectrum constructor from input

Spectrum::Spectrum(MomentumRange input_range, MCPolarization pol, bool xlog) {

  // SWD some of this should be used to initialization outside constructor
  next = nullptr;
  face = BoundaryFace::undef;
  range = input_range;
  polarized = pol;
  logarithmic = xlog;
  coordinates = false;
  x1min = x2min = x3min = -HUGE_NUMBER;
  x1max = x2max = x3max = HUGE_NUMBER;
  // Allocate and intialize energy bins
  energies.NewAthenaArray(range.ne+1);
  BuildEnergyGrid(range.emin,range.emax,range.ne,logarithmic);

  // Allocate arrays for intensities
  intensity.NewAthenaArray(range.nphi,range.ncth,range.ne);
  intensity_sq.NewAthenaArray(range.nphi,range.ncth,range.ne);
  for (int m = 0; m < NumStokesStored(polarized); ++m) {
    stokes[m].NewAthenaArray(range.nphi,range.ncth,range.ne);
    stokes_sq[m].NewAthenaArray(range.nphi,range.ncth,range.ne);
  }
}

//----------------------------------------------------------------------------------------
//! Spectrum constructor from copy
//! Copies arrays and medata but leaves array empty

Spectrum::Spectrum(Spectrum *pspec) {

  // Set pointers
  pmy_mc = pspec->pmy_mc;
  next = nullptr; // copy is not part of linked list

  base_name.assign(pspec->base_name);
  range = pspec->range;
  polarized = pspec->polarized;
  logarithmic = pspec->logarithmic;
  polar_axis = pspec->polar_axis;
  coordinates = pspec->coordinates;
  face = pspec->face;
  id = pspec->id;
  output_number = pspec->output_number;
  x1min = pspec->x1min;
  x2min = pspec->x2min;
  x3min = pspec->x3min;
  x1max = pspec->x1max;
  x2max = pspec->x2max;
  x3max = pspec->x3max;
  last_time = pspec->last_time;
  dt = pspec->dt;

  nsrun = 0; // arrays are uninitialized
  // Allocate and intialize energy bins
  energies.NewAthenaArray(range.ne+1);
  BuildEnergyGrid(range.emin,range.emax,range.ne,logarithmic);

  // Allocate arrays for intensities
  intensity.NewAthenaArray(range.nphi,range.ncth,range.ne);
  intensity_sq.NewAthenaArray(range.nphi,range.ncth,range.ne);
  for (int m = 0; m < NumStokesStored(polarized); ++m) {
    stokes[m].NewAthenaArray(range.nphi,range.ncth,range.ne);
    stokes_sq[m].NewAthenaArray(range.nphi,range.ncth,range.ne);
  }
}

//----------------------------------------------------------------------------------------
// !destructor

Spectrum::~Spectrum() {

  energies.DeleteAthenaArray();
  intensity.DeleteAthenaArray();
  intensity_sq.DeleteAthenaArray();
  for (int m = 0; m < NumStokesStored(polarized); ++m) {
    stokes[m].DeleteAthenaArray();
    stokes_sq[m].DeleteAthenaArray();
  }
}

//----------------------------------------------------------------------------------------
//! \fn void Spectrum::BuildEnergyGrid(Real emin, Real emax, int nen, bool logarithmic)
//! \brief initialize energy bins

void Spectrum::BuildEnergyGrid(Real emin, Real emax, int nen, bool logarthmic) {

  if (logarithmic) {
    // Distribute energies logarithmically
    Real de = log10(emax/emin) / static_cast<Real>(nen);
    energies(0) = log10(emin);
    for(int i=0; i<nen; ++i) {
      energies(i+1) = energies(i) + de;
    }
    for(int i=0; i<=nen; ++i){
      energies(i) = exp(2.30258509299*energies(i));
    }
  } else {
    // Distribute frequencies linear.  Useful for lines or path length
    // distributions
    Real de = (emax-emin) / static_cast<Real>(nen);
    energies(0) = emin;
    for(int i=0; i<nen; ++i) {
      energies(i+1) = energies(i) + de;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void Spectrum::SetSurface(std::string input_face)
//! \brief set corresponding surface

void Spectrum::SetSurface(std::string input_face) {

  if (input_face == "inner_x1") {
    face = BoundaryFace::inner_x1;
  } else if (input_face == "outer_x1") {
    face = BoundaryFace::outer_x1;
  } else if (input_face == "inner_x2") {
    face = BoundaryFace::inner_x2;
  } else if (input_face == "outer_x2") {
    face = BoundaryFace::outer_x2;
  } else if (input_face == "inner_x3") {
    face = BoundaryFace::inner_x3;
  } else if (input_face == "outer_x3") {
    face = BoundaryFace::outer_x3;
  } else if (input_face == "none") {
    face = BoundaryFace::undef;
  } else {
    std::stringstream msg;
      msg << "### FATAL ERROR in function [Spectrum::SetSurface]" << std::endl
          << "Face not recognized in spectrum input." << std::endl;
      throw std::runtime_error(msg.str().c_str());
  }
}

//----------------------------------------------------------------------------------------
//! \fn bool Spectrum::AngleBinsCarteisan(Real kx[4], int &phibin, int &cthbin)
//! \brief set index of phi and cth bins relative to cartesian axis

bool Spectrum::AngleBinsCartesian(Real k[4], int &phibin, int &cthbin) {

  Real kx = k[IMC1];
  Real ky = k[IMC2];
  Real kz = k[IMC3];

  Real ctheta, phi, stheta;
  if (pmy_mc->topology == MCTOPO_CARTESIAN) {
    // Set ctheta, phi according to face
    switch(face) {
      case BoundaryFace::inner_x1:
        ctheta = -kx;
        stheta = sqrt(SQR(ky)+SQR(kz));
        phi = acos(ky/stheta);
        if(kz < 0.0)
          phi = 2 * PI - phi;
        break;
      case BoundaryFace::outer_x1:
        ctheta = kx;
        stheta = sqrt(SQR(ky)+SQR(kz));
        phi = acos(ky/stheta);
        if(kz < 0.0)
          phi = 2 * PI - phi;
        break;
      case BoundaryFace::inner_x2:
        ctheta = -ky;
        stheta = sqrt(SQR(kx)+SQR(kz));
        phi = acos(kx/stheta);
        if(kz < 0.0)
          phi = 2 * PI - phi;
        break;
      case BoundaryFace::outer_x2:
        ctheta = ky;
        stheta = sqrt(SQR(kx)+SQR(kz));
        phi = acos(kx/stheta);
        if(kz < 0.0)
          phi = 2 * PI - phi;
        break;
      case BoundaryFace::inner_x3:
        ctheta = -kz;
        stheta = sqrt(SQR(kx)+SQR(ky));
        phi = acos(kx/stheta);
        if(ky < 0.0)
          phi = 2 * PI - phi;
        break;
      case BoundaryFace::outer_x3:
        ctheta = kz;
        stheta = sqrt(SQR(kx)+SQR(ky));
        phi = acos(kx/stheta);
        if(ky < 0.0)
          phi = 2 * PI - phi;
        break;
      case BoundaryFace::undef:
        ctheta=fabs(kz);
        phi = 0.;
        break;
      default:
        std::stringstream msg;
        msg << "### FATAL ERROR in function [Spectrum::AngleBinsCartesian]" << std::endl
            << "Face not valid" << std::endl;
        throw std::runtime_error(msg.str().c_str());
        break;
    }
  } else if (pmy_mc->topology == MCTOPO_SPHERICAL) {
    if (kz >= 0.0) {
      ctheta = kz;
      stheta = sqrt(SQR(kx)+SQR(ky));
      phi = acos(kx/stheta);
      if(ky < 0.0)
        phi = 2 * PI - phi;
    } else {
      ctheta = -kz;
      stheta = sqrt(SQR(kx)+SQR(ky));
      phi = acos(kx/stheta);
      if(ky < 0.0)
        phi = 2 * PI - phi;
    }
  } else {
    // Previously fell through with ctheta, phi and stheta uninitialized.
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Spectrum::AngleBinsCartesian]" << std::endl
        << "No angle binning defined for the topology of coordinate system "
        << GetMCCoordSystemName(pmy_mc->coord_system) << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  // Get ctheta bin
  int ncth = range.ncth;
  cthbin = static_cast<int>(ctheta * static_cast<Real>(ncth) );
  if (cthbin >= ncth) {
    std::cout << "Warning: cthbin > ncth (cthbin, mu, k1, k2, k3): " << cthbin << ' '
              << ctheta << ' ' << kx << ' ' << ky << ' ' << kz << std::endl;
    return false;
  }
  if (cthbin < 0) {
    std::cout << "Warning: cthbin < 0 (cthbin, mu, k1, k2, k3): " << cthbin << ' '
              << ctheta << ' ' << kx << ' ' << ky << ' ' << kz << std::endl;
    return false;
  }

  // Get phi bin
  int nphi = range.nphi;
  if (ctheta == 1.0) {
    phibin = 0;
  } else {
    phibin = static_cast<int>(phi * static_cast<Real>(nphi) / (2.*PI));
  }
  if(phibin >= nphi) {
    std::cout << "Warning: phibin > nphi (phibin, phi, k1, k2, k3): " << phibin << ' '
              << phi << ' ' << kx << ' ' << ky << ' ' << kz << std::endl;
    return false;
  }
  if(phibin < 0) {
    std::cout << "Warning: phibin < 0 (phibin, phi, k1, k2, k3): " << phibin << ' '
              << phi << ' ' << kx << ' ' << ky << ' ' << kz << std::endl;
    return false;
  }

  return true;
}


//----------------------------------------------------------------------------------------
//! \fn bool Spectrum::AngleBinsSphericalPolar(Real k[4], int &phibin, int &cthbin)
//! \brief set index of phi and cth bins relative to spherical-polar axis

bool Spectrum::AngleBinsSphericalPolar(Real k[4], int &phibin, int &cthbin) {

  Real kr = k[IMC1];
  Real kth = k[IMC2];
  Real kph = k[IMC3];

  Real ctheta, phi, stheta;
  // Set ctheta, phi according to face
  switch(face) {
    case BoundaryFace::inner_x1:
      ctheta = -kr;
      stheta = sqrt(SQR(kth)+SQR(kph));
      phi = acos(kth/stheta);
      if(kph < 0.0)
        phi = 2 * PI - phi;
      break;
    case BoundaryFace::outer_x1:
      ctheta = kr;
      stheta = sqrt(SQR(kth)+SQR(kph));
      phi = acos(kth/stheta);
      if(kph < 0.0)
        phi = 2 * PI - phi;
      break;
    case BoundaryFace::inner_x2:
      ctheta = -kth;
      stheta = sqrt(SQR(kr)+SQR(kph));
      phi = acos(kr/stheta);
      if(kph < 0.0)
        phi = 2 * PI - phi;
      break;
    case BoundaryFace::outer_x2:
      ctheta = kth;
      stheta = sqrt(SQR(kr)+SQR(kph));
      phi = acos(kr/stheta);
      if(kph < 0.0)
        phi = 2 * PI - phi;
      break;
    case BoundaryFace::inner_x3:
      ctheta = -kph;
      stheta = sqrt(SQR(kr)+SQR(kth));
      phi = acos(kr/stheta);
      if(kth < 0.0)
        phi = 2 * PI - phi;
      break;
    case BoundaryFace::outer_x3:
      ctheta = kph;
      stheta = sqrt(SQR(kr)+SQR(kth));
      phi = acos(kr/stheta);
      if(kth < 0.0)
        phi = 2 * PI - phi;
      break;
    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in function [Spectrum::AngleBinsSphericalPolar]"
          << std::endl << "Face not valid" << std::endl;
      throw std::runtime_error(msg.str().c_str());
      break;
  }
  if (ctheta < 0.0) {
    std::cout << "Warning: ctheta < 0\n" << std::endl;
    return false;
  }

  // Get ctheta bin
  int ncth = range.ncth;
  cthbin = static_cast<int>(ctheta * static_cast<Real>(ncth) );
  if(cthbin >= ncth) {
    std::cout << "Warning: cthbin > ncth (cthbin, mu, k1, k2, k3): " << cthbin << ' '
              << ctheta << ' ' << kr << ' ' << kth << ' ' << kph << std::endl;
    return false;
  }

  // Get phi bin
  int nphi = range.nphi;
  if (ctheta == 1.0) {
    phibin = 0;
  } else {
    phibin = static_cast<int>(phi * static_cast<Real>(nphi) / (2.*PI));
  }
  if(phibin >= nphi) {
    std::cout << "Warning: phibin > nphi (phibin, phi, k1, k2, k3): " << phibin << ' '
              << phi << ' ' << kr << ' ' << kth << ' '
              << kph << std::endl;
    return false;
  }
  if(phibin < 0) {
    std::cout << "Warning: phibin < 0 (phibin, phi, k1, k2, k3): " << phibin << ' '
              << phi << ' ' << kr << ' ' << kth << ' '
              << kph << std::endl;
    return false;
  }
  return true;

}

//----------------------------------------------------------------------------------------
//! \fn bool Spectrum::ScreenCoordinates(Photon *pphot, int ip)
//! \brief Returns true if photon is not within specified coordinate ranges

bool Spectrum::ScreenCoordinates(Photon *pphot, int ip) {

  if (pphot->x1p[ip] < x1min)
    return true;
  else if (pphot->x1p[ip] > x1max)
    return true;
  else if (pphot->x2p[ip] < x2min)
    return true;
  else if (pphot->x2p[ip] > x2max)
    return true;
  else if (pphot->x3p[ip] < x3min)
    return true;
  else if (pphot->x3p[ip] > x3max)
    return true;
  else
    return false;
}

//----------------------------------------------------------------------------------------
//! \fn void Spectrum::UpdateSpectrum(Photon *pphot, int ip)
//! \brief add photon contribution to spectrum

void Spectrum::UpdateSpectrum(Photon *pphot, int ip) {

  MonteCarloBlock *pmcb = pphot->pmy_mcb;

  // if face is set, then determine if photon positions matches
  if (face != BoundaryFace::undef) {
    enum BoundaryFace photon_face = GetPhotonFace(pphot,ip);
    if (face != photon_face)
      return;
  }

  Real weight = pphot->wp[ip];
  if ((std::isinf(weight)) || (std::isnan(weight))) {
    std::cout << "Warning: weight is Nan or Inf: " << weight << std::endl;
  } else {

    if (coordinates) {
      //Apply coordinate cuts
      if (ScreenCoordinates(pphot,ip))
        return;
    }

    // For curved spacetimes, bin the conserved -k_t, matching PhotonList::AddPhoton.
    // use ep otherwise
    int ebin;
    Real ebin_energy = pmy_mc->relativistic_output
                       ? PhotonEnergyAtInfinity(pphot,ip) : pphot->ep[ip];
    ebin = EnergyBinUniform(ebin_energy,logarithmic);
    if (ebin < 0) return;

    // Get angle bins
    int phibin, mubin;
    if (polar_axis) {
      Real kcart[4];
      // Was keyed on cartesian || minkowski, which left kcart uninitialized for the
      // Cartesian Kerr-Schild case (gr_user); it shares this topology and belongs here.
      if (pmy_mc->topology == MCTOPO_CARTESIAN)  {
        if (pphot->pmy_mcb->pmy_mc->general_pusher_flag) {
          // Under GeneralPusher k1,k2,k3 are contravariant coordinate components, so they
          // are neither orthonormal nor a unit vector. Since projecting to infinity is
          // challenging, we insteady project onto local the normal observer's orthonormal
          // frame and normalize.
          //
          // The frame is the normal observer, and specifically not MCCoord::InverseTetrad,
          // because CoherencyToObserverStokes references the Stokes parameters to this
          // one.
          Real econ[4][4], ecov[4][4], ktet[4];
          if (!NormalFrameWavevector(pphot->pmy_mcb, pphot, ip, econ, ecov, ktet)) return;
          Real norm = sqrt(SQR(ktet[IMC1]) + SQR(ktet[IMC2]) + SQR(ktet[IMC3]));
          if (norm <= TINY_NUMBER) return;
          for (int a = IMC1; a < 4; ++a) kcart[a] = ktet[a]/norm;
        } else {
          kcart[IMC1] = pphot->k1p[ip];
          kcart[IMC2] = pphot->k2p[ip];
          kcart[IMC3] = pphot->k3p[ip];
        }
      } else  if (pmy_mc->topology == MCTOPO_SPHERICAL) {
        Real cth = cos(pphot->x2p[ip]);
        Real sth = sin(pphot->x2p[ip]);
        Real cph = cos(pphot->x3p[ip]);
        Real sph = sin(pphot->x3p[ip]);
        Real kr, kth, kph;
        // Flat orthonormalization.  Only valid because the MCOutput constructor refuses
        // spec output for a curved metric; see the note there.
        if (pphot->pmy_mcb->pmy_mc->general_pusher_flag) {
          kr = pphot->k1p[ip];
          kth = pphot->k2p[ip]*pphot->x1p[ip];
          kph = pphot->k3p[ip]*pphot->x1p[ip]*sth;
        } else {
          kr = pphot->k1p[ip];
          kth = pphot->k2p[ip];
          kph = pphot->k3p[ip];
        }
        // Compute cartesian
        kcart[IMC1] = kr*sth*cph + kth*cth*cph - kph*sph;
        kcart[IMC2] = kr*sth*sph + kth*cth*sph + kph*cph;
        kcart[IMC3] = kr*cth - kth*sth;
        Real knorm = std::sqrt(SQR(kcart[IMC1]) + SQR(kcart[IMC2]) + SQR(kcart[IMC3]));
        if (knorm <= TINY_NUMBER) return;
        for (int a = IMC1; a < 4; ++a) kcart[a] /= knorm;
      } else {
        // SWD: Add cylindrical
        std::stringstream msg;
        msg << "### FATAL ERROR in function [Spectrum::UpdateSpectrum]" << std::endl
            << "polar_axis spectra are not implemented for coordinate system "
            << GetMCCoordSystemName(pmy_mc->coord_system) << std::endl;
        throw std::runtime_error(msg.str().c_str());
      }
      if (!AngleBinsCartesian(kcart,phibin,mubin))
        return;
    } else {
      if (pmy_mc->topology == MCTOPO_SPHERICAL) {
        Real ksph[4];
        // Flat orthonormalization.  Only valid because the MCOutput constructor refuses
        // spec output for a curved metric; see the note there.
        if (pphot->pmy_mcb->pmy_mc->general_pusher_flag) {
          ksph[IMC1] = pphot->k1p[ip];
          ksph[IMC2] = pphot->k2p[ip]*pphot->x1p[ip];
          ksph[IMC3] = pphot->k3p[ip]*pphot->x1p[ip]*sin(pphot->x2p[ip]);
          Real norm = sqrt(SQR(ksph[IMC1])+SQR(ksph[IMC2])+SQR(ksph[IMC3]));
          // from IMC1: ksph[IMC0] is never set, and AngleBinsSphericalPolar reads only
          // the spatial components
          for (int i=IMC1; i<4; ++i)
            ksph[i] /= norm;
        } else {
          ksph[IMC1] = pphot->k1p[ip];
          ksph[IMC2] = pphot->k2p[ip];
          ksph[IMC3] = pphot->k3p[ip];
        }
        if(!AngleBinsSphericalPolar(ksph,phibin,mubin))
            return;
      } else {
        // Previously fell through and binned with phibin and mubin uninitialized.
        std::stringstream msg;
        msg << "### FATAL ERROR in function [Spectrum::UpdateSpectrum]" << std::endl
            << "Spectra with polar_axis = false require a spherical topology; coordinate "
            << "system " << GetMCCoordSystemName(pmy_mc->coord_system) << " has none."
            << std::endl;
        throw std::runtime_error(msg.str().c_str());
      }
    } // if (polar_axis) else

    intensity(phibin,mubin,ebin) += weight;
    intensity_sq(phibin,mubin,ebin) += weight * weight;
    //intensity(phibin,mubin,ebin) += pphot->sip[ip] * weight;
    //intensity_sq(phibin,mubin,ebin) += SQR(pphot->sip[ip] * weight);
    // The Stokes arrays on the photon only exist when polarization is tracked
    if (IsPolarized(polarized)) {
      const Real spol[3] = {pphot->sqp[ip], pphot->sup[ip], pphot->svp[ip]};
      for (int m = 0; m < NumStokesStored(polarized); ++m) {
        stokes[m](phibin,mubin,ebin) += spol[m] * weight;
        stokes_sq[m](phibin,mubin,ebin) += SQR(spol[m] * weight);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void Spectrum::EnergyBinUniform(Real energy, bool loge)
//! \brief return bin number corresponding to energy

int Spectrum::EnergyBinUniform(Real energy, bool loge)
{
  int ne = range.ne;
  Real elow = energies(0);
  Real ehigh = energies(ne);
  if ( (energy < elow) || (energy > ehigh) )
    return -1;
  if (loge) {
    elow = log10(elow);
    ehigh = log10(ehigh);
    return static_cast<int>((log10(energy)-elow)/(ehigh-elow)*static_cast<Real>(ne));
  } else {
    return static_cast<int>((energy-elow)/(ehigh-elow)*static_cast<Real>(ne));
  }

}

//----------------------------------------------------------------------------------------
//! \fn void Spectrum::EnergyBin(Real energy)
//! \brief return bin number corresponding to energy

int Spectrum::EnergyBin(Real energy)
{

  if ( (energy < energies(0)) || (energy > energies(range.ne)) )
    return -1;

  // Perform binary search
  int low = 0;
  int high = range.ne+1;
  int mid;
  while(low <= high) {
    mid = (low + high) / 2;
    if(energies(mid) <= energy) {
      if(energies(mid+1) > energy) {
        return mid;
      } else
        low = mid+1;
    } else
      high = mid-1;
  }
  if(mid == high) {
    return mid;
  }
  else {
    std::cout << "Warning: binary search failed in ebin: " << energy << ' '
              << mid <<  std::endl;
    return -1;
  }

}

//----------------------------------------------------------------------------------------
//! \fn void Spectrum::ResetSpectrum()
//! \brief zero elements of output spectrum

void Spectrum::ResetSpectrum() {

  nsrun = 0;
  for(int i=0; i<range.nphi; ++i) {
    for(int j=0; j<range.ncth; ++j) {
      for(int k=0; k<range.ne; ++k) {
        intensity(i,j,k) = 0.;
        intensity_sq(i,j,k) = 0.;
        for (int m = 0; m < NumStokesStored(polarized); ++m) {
          stokes[m](i,j,k) = 0.;
          stokes_sq[m](i,j,k) = 0.;
        }
      }
    }
  }

}

//----------------------------------------------------------------------------------------
//! \fn void Spectrum::AddSpectrum(Spectrum *pspec)
//! \brief return add contents of another spectrum

void Spectrum::AddSpectrum(Spectrum *pspec) {

  if (pspec->id != id) {
    std::stringstream msg;
    msg << "### FATAL ERROR in AddSpectrum" << std::endl
        << "Input spectrum id ="  << pspec->id << " but this spectrum id = "
        << id << std::endl;
    throw std::runtime_error(msg.str().c_str());
  } else {
    nsrun += pspec->nsrun;
    for(int i=0; i<range.nphi; ++i) {
      for(int j=0; j<range.ncth; ++j) {
        for(int k=0; k<range.ne; ++k) {
          intensity(i,j,k) += pspec->intensity(i,j,k);
          intensity_sq(i,j,k) += pspec->intensity_sq(i,j,k);
          for (int m = 0; m < NumStokesStored(pspec->polarized); ++m) {
            stokes[m](i,j,k) += pspec->stokes[m](i,j,k);
            stokes_sq[m](i,j,k) += pspec->stokes_sq[m](i,j,k);
          }
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn enum BoundaryFace Spectrum::GetPhotonFace(Photon *pphot, int ip)
//! \brief Determine which boundary photon crossed, if any

enum BoundaryFace Spectrum::GetPhotonFace(Photon *pphot, int ip) {

  MonteCarloBlock *pmcb = pphot->pmy_mcb;

  if(pphot->i1p[ip] > pmcb->ie)
    return BoundaryFace::outer_x1;
  else if(pphot->i1p[ip] < pmcb->is)
    return BoundaryFace::inner_x1;
  else if(pphot->i2p[ip] > pmcb->je)
    return BoundaryFace::outer_x2;
  else if(pphot->i2p[ip] < pmcb->js)
    return BoundaryFace::inner_x2;
  else if(pphot->i3p[ip] > pmcb->ke)
    return BoundaryFace::outer_x3;
  else if(pphot->i3p[ip] < pmcb->ks)
    return BoundaryFace::inner_x3;
  else
    return BoundaryFace::undef;
}

//----------------------------------------------------------------------------------------
//! PhotonList constructor from input

PhotonList::PhotonList(int list_size_init, MCPolarization pol, int nuser,
                       int max_res) {

  // Incremental output state; nothing is open until the list first spills or is written.
  fp_ = nullptr;
  nwritten_ = 0;
  dt_pos_ = length_pos_ = ntot_pos_ = 0;
  max_resident = max_res;

  // Allocate memory for photon list
  len_limit = list_size_init;
  nparams = 10;
  polarized = pol;
  nparams += NumStokesStored(polarized); // stokes q and u, plus v when circular
  nparams += nuser;
  nuser_out = nuser;
  photons.NewAthenaArray(len_limit,nparams);

}

//----------------------------------------------------------------------------------------
//! destructor
PhotonList::~PhotonList() {

  photons.DeleteAthenaArray();

}

//----------------------------------------------------------------------------------------
//! \fn Real PhotonEnergyAtInfinity(Photon *pphot, int ip)
//! \brief conserved photon energy -k_t = -g_{t mu} k^mu
//!
//! Every metric the Monte Carlo module supports is stationary, so k_t is exactly
//! conserved along the geodesic, and all of them are asymptotically flat with t
//! normalized so that g_tt -> -1.  Under those two conditions -k_t is the energy a
//! distant observer measures, and it inherits the units of ep (erg).
//!
//! A user-supplied metric that is not stationary would break the first condition, and
//! this would silently return a non-conserved quantity.  gr_user is currently always
//! built as MCKerrSchildCartesian, so there is no such case today.

Real PhotonEnergyAtInfinity(Photon *pphot, int ip) {

  MCCoord *pco = pphot->pmy_mcb->pcoord;

  Real x[4], kcon[4], gcov[4][4];
  x[IMC0] = pphot->x0p[ip];
  x[IMC1] = pphot->x1p[ip];
  x[IMC2] = pphot->x2p[ip];
  x[IMC3] = pphot->x3p[ip];
  kcon[IMC0] = pphot->k0p[ip];
  kcon[IMC1] = pphot->k1p[ip];
  kcon[IMC2] = pphot->k2p[ip];
  kcon[IMC3] = pphot->k3p[ip];

  pco->Metric(x, gcov);

  Real kt = 0.;
  for (int i = 0; i < 4; i++)
    kt += gcov[IMC0][i] * kcon[i];

  return -kt;
}

//----------------------------------------------------------------------------------------
//! \fn PhotonList::AddPhoton(Photon *pphot, int ip)
//! \brief add photon to list

void PhotonList::AddPhoton(Photon *pphot, int ip) {

  // Bound the resident list.  Past max_resident the photons already collected are
  // streamed to the file and the array is reused, so the memory held is bounded.
  if (max_resident > 0 && length >= max_resident) SpillToFile();
  if (length == len_limit) {
    // double array size when list is full
    ResizeList(2*len_limit);
  }
  int n = 0;
  photons(length,n++) = pphot->wp[ip];
  // The energy column carries whatever a distant observer would call the photon energy.
  // In flat space that is just ep.  Under a GR integration ep holds k^t, a coordinate
  // component that no observer measures -- in Kerr-Schild g_{tr} is non-zero, so k^t
  // mixes k_t and k_r and binning a spectrum on it produces an apparent blueshift.  The
  // conserved -k_t is written instead; k^t remains available in the k0 column below.
  photons(length,n++) = pmy_mc->relativistic_output
                        ? PhotonEnergyAtInfinity(pphot,ip) : pphot->ep[ip];
  photons(length,n++) = pphot->x1p[ip];
  photons(length,n++) = pphot->x2p[ip];
  photons(length,n++) = pphot->x3p[ip];
  photons(length,n++) = pphot->x0p[ip];
  // The wavevector goes out in the same normal frame the Stokes parameters are
  // referenced to
  if (pmy_mc->general_pusher_flag) {
    Real econ[4][4], ecov[4][4], ktet[4];
    if (NormalFrameWavevector(pphot->pmy_mcb, pphot, ip, econ, ecov, ktet)) {
      photons(length,n++) = ktet[IMC1];
      photons(length,n++) = ktet[IMC2];
      photons(length,n++) = ktet[IMC3];
      photons(length,n++) = ktet[IMC0];
    } else {
      // no normal observer here; fall back to the stored components rather than drop the
      // photon, and leave it detectable as a zero time component
      photons(length,n++) = pphot->k1p[ip];
      photons(length,n++) = pphot->k2p[ip];
      photons(length,n++) = pphot->k3p[ip];
      photons(length,n++) = pphot->k0p[ip];
    }
  } else {
    photons(length,n++) = pphot->k1p[ip];
    photons(length,n++) = pphot->k2p[ip];
    photons(length,n++) = pphot->k3p[ip];
    photons(length,n++) = pphot->k0p[ip];
  }
  if (IsPolarized(polarized)) {
    const Real spol[3] = {pphot->sqp[ip], pphot->sup[ip], pphot->svp[ip]};
    for (int m = 0; m < NumStokesStored(polarized); ++m) {
      photons(length,n++) = spol[m];
    }
  }
  for (int i=0; i<nuser_out; i++) {
    photons(length,n++) = pphot->user[i][ip];
  }
  length++;

}

//----------------------------------------------------------------------------------------
//! \fn std::string PhotonList::Filename() const
//! \brief name of the file this list writes to
//
// One definition, used by both the spill path and the final write.  They open the same
// file at different times and must not be able to disagree about which one it is.

std::string PhotonList::Filename() const {
  std::stringstream n;
  n << base_name << "." << std::setw(5) << std::setfill('0') << output_number << ".list";
  return n.str();
}

//----------------------------------------------------------------------------------------
//! \fn void PhotonList::OpenAndWriteHeader(const std::string &filename, Real tint_out)
//! \brief open the list file and write its header
//
// length and ntot are padded to a fixed width and corrected in place once the data has been
// written, because neither is known if/when the first photons spill: both depend on how many
// photons ultimately arrive.  dt is padded for the one case where it is not known either --
// a dynamic run accumulating several steps into one list, where the header records the
// elapsed time since the previous output rather than the integration time of any one step.
// A static run writes the final value at the outset and its correction changes nothing.
// The reader
// takes the first space-delimited token on each line (athena_mc.parse_line_value), so the
// trailing blanks are ignored and the format is unchanged as far as it is concerned.

void PhotonList::OpenAndWriteHeader(const std::string &filename, Real tint_out) {
  std::stringstream msg;
  if ((fp_ = fopen(filename.c_str(),"wb")) == nullptr) {
    msg << "### FATAL ERROR in function [PhotonList::WriteList]" << std::endl
        << "Output file '" << filename << "' could not be opened";
    throw std::runtime_error(msg.str().c_str());
  }
  dt_pos_ = ftell(fp_);
  fprintf(fp_,"dt=%-24.8e\n",tint_out);
  length_pos_ = ftell(fp_);
  fprintf(fp_,"length=%-20lld\n",static_cast<long long>(0));
  fprintf(fp_,"npars=%d\n",nparams);
  ntot_pos_ = ftell(fp_);
  fprintf(fp_,"ntot=%-20d\n",nsrun);
  fprintf(fp_,"polarized=%s\n",GetMCPolarizationName(polarized));
  fprintf(fp_,"coord=%s\n",pmy_mc->geometry_tag.c_str());
  // free parameters of the metric, so the file is self-describing; absent for
  // metrics that have none
  if (!pmy_mc->metric_params.empty())
    fprintf(fp_,"metric_params=%s\n",pmy_mc->metric_params.c_str());
  fprintf(fp_,"frame=%s\n",pmy_mc->frame_tag.c_str());
}

//----------------------------------------------------------------------------------------
//! \fn void PhotonList::StreamResident()
//! \brief convert and append the resident photons, a chunk at a time
//
// A fixed buffer rather than one the size of the list

void PhotonList::StreamResident() {
  if (length <= 0) return;
  const int kChunk = 65536;
  std::vector<double> buf(static_cast<std::size_t>(kChunk)*nparams);
  int i = 0;
  while (i < length) {
    const int nthis = (length - i < kChunk) ? length - i : kChunk;
    std::size_t n = 0;
    for (int m = 0; m < nthis; ++m)
      for (int j = 0; j < nparams; ++j)
        buf[n++] = static_cast<double>(photons(i+m,j));
    // write data in big endian order
    if (!(mcoutput::IsBigEndian()))
      for (std::size_t m = 0; m < n; ++m) mcoutput::Swap8Bytes(&buf[m]);
    fwrite(buf.data(),sizeof(double),n,fp_);
    i += nthis;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void PhotonList::SpillToFile()
//! \brief push the resident photons to disk and reuse the array

void PhotonList::SpillToFile() {
  if (length == 0) return;
  // pmy_mc->tint, not the member dt: tint is the integration time already folded into
  // every photon weight (ComputeEmissionArray multiplies the emissivity by it), whereas dt
  // is this output's cadence.  For a static run tint is what WriteList will finally record,
  // so the header goes out correct and the later correction is a no-op.  The file name has
  // to match the one WriteList will use, which is why both go through Filename().
  if (fp_ == nullptr) OpenAndWriteHeader(Filename(), pmy_mc->tint);
  StreamResident();
  nwritten_ += length;
  length = 0;
}

//----------------------------------------------------------------------------------------
//! \fn void PhotonList::WriteList(std::string filename, Real tint_out)
//! \brief write photon list to binary file

void PhotonList::WriteList(std::string filename, Real tint_out) {
  // Since list lengths are variable each process writes its own list
  if (fp_ == nullptr) OpenAndWriteHeader(filename, tint_out);
  StreamResident();
  nwritten_ += length;
  length = 0;

  // Correct the fields that were placeholders when the header went out.
  fseek(fp_, dt_pos_, SEEK_SET);
  fprintf(fp_,"dt=%-24.8e",tint_out);
  fseek(fp_, length_pos_, SEEK_SET);
  fprintf(fp_,"length=%-20lld",static_cast<long long>(nwritten_));
  fseek(fp_, ntot_pos_, SEEK_SET);
  fprintf(fp_,"ntot=%-20d",nsrun);
  fclose(fp_);
  fp_ = nullptr;
}

//----------------------------------------------------------------------------------------
//! \fn void PhotonList::ResetList()
//! \brief reset lists length and total

void PhotonList::ResetList() {
  length = 0;
  nwritten_ = 0;
  nsrun = 0;
}


//----------------------------------------------------------------------------------------
//! \fn void PhotonList::ResizeList(int new_len)
//!  \brief resize a photon list

void PhotonList::ResizeList(int new_len) {

  if (new_len < len_limit) {
    std::cout << "Warning: new list length " << new_len << " < len_limit "
              << len_limit << ".  Aborting ResizeList()" << std::endl;
    return;
  }
  AthenaArray<Real> temp_array(photons); // create deep copy
  photons.DeleteAthenaArray();
  photons.NewAthenaArray(new_len,nparams);
  for (int i=0; i<length; ++i) {
    for (int j=0; j<nparams; ++j) {
      photons(i,j) = temp_array(i,j);
    }}
  temp_array.DeleteAthenaArray();
  len_limit = new_len;
}

//----------------------------------------------------------------------------------------
//! PhotonTrajectoryList constructor from input

PhotonTrajectoryList::PhotonTrajectoryList(int init_len_limit, int init_step_limit,
                                           int nuser) {

  // Allocate memory for trajectory list
  len_limit = init_len_limit;
  step_limit = init_step_limit;
  nparams = 4;
  nparams += nuser;
  nuser_out = nuser;
  trajectories.NewAthenaArray(len_limit,step_limit,nparams);
  nsteps = new int[len_limit];

}

//----------------------------------------------------------------------------------------
//! destructor

PhotonTrajectoryList::~PhotonTrajectoryList() {

  trajectories.DeleteAthenaArray();
  delete [] nsteps;

}

//----------------------------------------------------------------------------------------
//! \fn PhotonTrajectoryList::InitializeTrajectory(int itraj)
//! \brief add new trajectory

void PhotonTrajectoryList::InitializeTrajectory(int itraj) {

  if (itraj >= len_limit) {
    // double array size when list is full
    ResizeList(2*len_limit,step_limit);
  }
  if (itraj >= length)
    length = itraj+1;

  nsteps[itraj] = 0;
}

//----------------------------------------------------------------------------------------
//! \fn PhotonTrajectoryList::CompleteTrajectory(int itraj)
//! \brief complete trajectory

void PhotonTrajectoryList::CompleteTrajectory(int itraj) {

  if (nsteps[itraj] > maxstep) maxstep = nsteps[itraj];

}

//----------------------------------------------------------------------------------------
//! \fn PhotonTrajectoryList::AddToTrajectory(Photon *pphot, int ip)
//! \brief add photon location to trajectory

void PhotonTrajectoryList::AddToTrajectory(Photon *pphot, int ip) {

  int itr = 0; //pphot->trp[ip]; SWD: removed trp
  int &step = nsteps[itr];
  if (step >= step_limit)
    return;

  int n = 0;
  trajectories(itr,step,n++) = pphot->x1p[ip];
  trajectories(itr,step,n++) = pphot->x2p[ip];
  trajectories(itr,step,n++) = pphot->x3p[ip];
  trajectories(itr,step,n++) = pphot->x0p[ip];
  for (int i=0; i<nuser_out; i++) {
    trajectories(itr,step,n++) = pphot->user[i][ip];
  }
  step++;
}

//----------------------------------------------------------------------------------------
//! \fn void PhotonTrajectoryList::WriteList(std::string filename)
//! \brief write photon trajectory list to binary file with header

void PhotonTrajectoryList::WriteList(std::string filename) {
  // Since list lengths are variable each process writes its own list

  // open file for output
  FILE *pfile;
  std::stringstream msg;

  //if ((pfile = fopen("temp.out","w")) == nullptr) {
  if ((pfile = fopen(filename.c_str(),"w")) == nullptr) {
    msg << "### FATAL ERROR in function [PhotonTrajectoryList::WriteList]" << std::endl
        << "Output file '" << filename << "' could not be opened";
    throw std::runtime_error(msg.str().c_str());
  }

  // write header information
  fprintf(pfile,"length=%d\n",length);
  fprintf(pfile,"maxstep=%d\n",maxstep);
  fprintf(pfile,"npars=%d\n",nparams);
  fprintf(pfile,"coord=%s\n",pmy_mc->geometry_tag.c_str());
  // free parameters of the metric, so the file is self-describing; absent for
  // metrics that have none
  if (!pmy_mc->metric_params.empty())
    fprintf(pfile,"metric_params=%s\n",pmy_mc->metric_params.c_str());
  fprintf(pfile,"frame=%s\n",pmy_mc->frame_tag.c_str());
  int *idata = new int[length];
  for (int i=0; i<length; ++i)
    idata[i] = nsteps[i];
  // write step numbers
  if (!(mcoutput::IsBigEndian()))
    for (int i=0; i<length; ++i) mcoutput::Swap4Bytes(&idata[i]);
  fwrite(idata,sizeof(int),static_cast<size_t>(length),pfile);
  // Get total length of array.  64-bit for the same reason as PhotonList::WriteList:
  // steps summed over trajectories, times nparams, outgrows a 32-bit int.
  std::int64_t ndata = 0;
  for (int i=0; i<length; ++i)
    ndata += nsteps[i];
  ndata *= nparams;
  double *data = new double[ndata];
  // write data
  std::int64_t n=0;
  for (int i=0; i<length; ++i) {
    for (int j=0; j<nsteps[i]; ++j) {
      for (int k=0; k<nparams; ++k) {
        data[n++] = static_cast<double>(trajectories(i,j,k));
      }}}
  // write data in big endian order
  if (!(mcoutput::IsBigEndian()))
    for (std::int64_t i=0; i<ndata; ++i) mcoutput::Swap8Bytes(&data[i]);
  fwrite(data,sizeof(double),static_cast<size_t>(ndata),pfile);
  fclose(pfile);
  delete [] data;
  delete [] idata;
}


//----------------------------------------------------------------------------------------
//! \fn void PhotonTrajectoryList::ResizeList(int new_len_limit, int new_step_limit)
//! \brief resize a photon trajectory list

void PhotonTrajectoryList::ResizeList(int new_len_limit, int new_step_limit) {

  if (new_len_limit < len_limit) {
    std::cout << "Warning: new list length " << new_len_limit << " < len_limit "
              << len_limit << ".  Aborting ResizeList()" << std::endl;
    return;
  }
  // Resize nsteps
  int *itemp_array = new int[length];
  for (int i=0; i<length; i++)
    itemp_array[i] = nsteps[i];
  delete [] nsteps;
  nsteps = new int[new_len_limit];
  for (int i=0; i<length; i++)
    nsteps[i] = itemp_array[i];
  delete [] itemp_array;
  // Resize trajectories
  AthenaArray<Real> temp_array(trajectories); // create deep copy
  trajectories.DeleteAthenaArray();
  trajectories.NewAthenaArray(new_len_limit,step_limit,nparams);
  for (int i=0; i<length; ++i) {
    for (int j=0; j<step_limit; ++j) {
      for (int k=0; k<nparams; ++k) {
        trajectories(i,j,k) = temp_array(i,j,k);
      }}}
  temp_array.DeleteAthenaArray();
  len_limit = new_len_limit;

}

//----------------------------------------------------------------------------------------
//! MCOutput constructor from ParameterInput and MonteCarlo
// SWD needs to be split into constructor + intializer
MCOutput::MCOutput(MonteCarlo *pmc, ParameterInput *pin) {

  pmy_mc = pmc;
  std::stringstream msg;
  InputBlock *pib = pin->pfirst_block;

  Real vel_cgs = pin->GetOrAddReal("problem","vel_cgs",1.);
  Real l_cgs = pin->GetOrAddReal("problem","l_cgs",1.);
  time_cgs = l_cgs / vel_cgs;
  mom_flag_lab = false;
  mom_flag_src = false;
  mom_flag_usr = false;
  mom_flag_com = false;
  mom_flag_coord = false;
  mom_flag_scat = false;
  pspec = nullptr;
  pphlist = nullptr;
  ptraj = nullptr;
  // loop over input block names.  Find those that start with "output", read parameters,
  // and construct linked list of spectra if present, set moments flag if moments output
  // present
  Spectrum *pfirst = nullptr, *plast;
  int id =0;
  while (pib != nullptr) {
    if (pib->block_name.compare(0,6,"output") == 0) {
      // Look for spectra
      std::string type = pin->GetString(pib->block_name,"file_type");

      if (type.compare("spec") == 0) {
        // Refused up front rather than per photon, so the run fails at setup instead of
        // after the transfer has started.
        //
        // Spectrum angle binning orthonormalizes the wavevector with the flat scale
        // factors (r, r sin(theta), and unity in the radial and cartesian directions) and
        // then treats the result as a unit direction.  That is exact only for a diagonal
        // metric with g_rr = 1.  In Kerr-Schild g_rr = 1 + 2M/r, g_tr does not vanish,
        // and the normal observer has n^r < 0, so the direction that comes out is off by
        // to ~20 degrees at r = 6M -- large enough that photons with mu < 0.35 are
        // classified as ingoing and silently dropped.  The energy axis is wrong in the
        // same way: it bins ep = k^t rather than the conserved -k_t.
        //
        // Doing this properly means projecting through the normal-observer tetrad, which
        // MonteCarloBlock::boost_lab already holds per cell.  Until then the photon list
        // is the supported route: it stores -k_t directly, and
        // vis/python/montecarlo/make_spectrum.py bins it.
        // Keyed on curvature, not on whether the flat scale factors happen to
        // orthonormalize the basis: a flat metric in a sheared chart, like snake, has a
        // covariantly constant orthonormal frame that InverseTetrad projects onto exactly,
        // and UpdateSpectrum uses it.  What breaks the binning is curvature, which is what
        // the paragraph above is about.
        if (IsMCMetricCurved(pmc->coord_system)) {
          std::stringstream msg;
          msg << "### FATAL ERROR in MCOutput constructor" << std::endl
              << "spec output is not supported for coordinate system "
              << GetMCCoordSystemName(pmc->coord_system) << "." << std::endl
              << "Angle and energy binning assume a flat spacetime, so that the "
              << "orthonormal frame the direction is measured in is globally defined."
              << std::endl
              << "Use file_type = phlist and bin it with "
              << "vis/python/montecarlo/make_spectrum.py instead." << std::endl;
          ATHENA_ERROR(msg);
        }
        // set momentum range and polarization, logarithmic flags for spectrum constructor
        MomentumRange range;
        range.ne = pin->GetInteger(pib->block_name,"ne");
        Real everg = 1.602176634e-12;
        range.emin = everg * pin->GetReal(pib->block_name,"emin");
        range.emax = everg * pin->GetReal(pib->block_name,"emax");
        range.nphi = pin->GetOrAddInteger(pib->block_name,"nphi",8);
        range.phimin = pin->GetOrAddReal(pib->block_name,"phimin",0.);
        range.phimax = pin->GetOrAddReal(pib->block_name,"phimax",2.*PI);
        range.ncth = pin->GetOrAddInteger(pib->block_name,"ncth",8);
        range.cthmin = pin->GetOrAddReal(pib->block_name,"cthmin",0.);
        range.cthmax = pin->GetOrAddReal(pib->block_name,"cthmax",1.);
        MCPolarization polarized = GetMCPolarizationFlag(pin->GetOrAddString(
            pib->block_name,"polarized",GetMCPolarizationName(pmc->polarized)));
        bool xlog = pin->GetOrAddBoolean(pib->block_name,"xlog",true);

        // Create spectrum
        pspec = new Spectrum(range,polarized,xlog);
        pspec->pmy_mc = pmc;
        pspec->id = id++;
        pspec->output_number = 0;
        if (pmc->dynamic) {
          pspec->dt = pin->GetReal(pib->block_name,"dt");
        } else {
          // input should be in code units
          pspec->dt = pin->GetOrAddReal("montecarlo","tint",1.);
        }
        pspec->last_time = pmy_mc->pmy_mesh->time;
        // Generate file name
        std::string outn = pib->block_name.substr(6); // 6 because counting starts at 0!
        int outid = atoi(outn.c_str());
        // set file name
        std::string basename = pin->GetString("job","problem_id");
        pspec->base_name.assign(basename);
        pspec->base_name.append(".");
        char define_id[10];
        sprintf(define_id,"out%d",outid);  // default id="outN"
        pspec->base_name.append(define_id);
        // set output face if specified
        std::string face = pin->GetOrAddString(pib->block_name,"face","none");
        pspec->SetSurface(face);
        // Check for coordinate ranges for spectrum
        if (pin->DoesParameterExist(pib->block_name,"x1min")) {
          pspec->x1min = pin->GetReal(pib->block_name,"x1min");
          pspec->coordinates = true;
        }
        if (pin->DoesParameterExist(pib->block_name,"x1max")) {
          pspec->x1max = pin->GetReal(pib->block_name,"x1max");
          pspec->coordinates = true;
        }
        if (pin->DoesParameterExist(pib->block_name,"x2min")) {
          pspec->x2min = pin->GetReal(pib->block_name,"x2min");
          pspec->coordinates = true;
        }
        if (pin->DoesParameterExist(pib->block_name,"x2max")) {
          pspec->x2max = pin->GetReal(pib->block_name,"x2max");
          pspec->coordinates = true;
        }
        if (pin->DoesParameterExist(pib->block_name,"x3min")) {
          pspec->x3min = pin->GetReal(pib->block_name,"x3min");
          pspec->coordinates = true;
        }
        if (pin->DoesParameterExist(pib->block_name,"x3max")) {
          pspec->x3max = pin->GetReal(pib->block_name,"x3max");
          pspec->coordinates = true;
        }
        // Set axis for determining output angles.  A Cartesian-topology grid has no
        // polar axis of its own, so escape angles have to be binned against the
        // Cartesian axes; the alternative branch of UpdateSpectrum has nothing to use.
        // Previously tested cartesian only, which left minkowski and gr_user defaulting
        // to polar_axis = false and then binning uninitialized angle indices.
        if (pmc->topology == MCTOPO_CARTESIAN)
          pspec->polar_axis = true;
        else
          pspec->polar_axis = pin->GetOrAddBoolean(pib->block_name,"polar_axis",false);
        pspec->nsrun = 0;
        // Check for coordinate range
        if (pfirst == nullptr)
          pfirst = pspec;
        else
          plast->next = pspec;
        plast = pspec;
      } else if (type.compare("phlist") == 0) {
        // Create photon list
        // Get number of user output variables and confirm it is less than
        // the number of user variables
        int nuser_out = pin->GetOrAddInteger(pib->block_name,"nuser",0);
        if (nuser_out > pmy_mc->nuser_var) {
          std::stringstream msg;
          msg << "### ERROR in MCOutput constructor" << std::endl
              << "User output variables: " << nuser_out
              << " greater than user variables: " << pmy_mc->nuser_var << std::endl;
          throw std::runtime_error(msg.str().c_str());
        }
        // Photons held in memory before the list spills to its file.  At ~13 columns
        // the default is a couple of hundred MB per rank, large enough that a modest run
        // never spills and small enough that a large one cannot grow without bound.
        int max_res = pin->GetOrAddInteger(pib->block_name,"max_resident",2000000);
        pphlist = new PhotonList(pmc->list_size_init,pmc->polarized,nuser_out,max_res);
        pphlist->pmy_mc = pmc;
        // Initialize photon list
        if (pmc->dynamic) {
          pphlist->dt = pin->GetReal(pib->block_name,"dt");
        } else {
          // input should be in code units
          pphlist->dt = pin->GetOrAddReal("montecarlo","tint",1.);
        }
        pphlist->last_time = pmy_mc->pmy_mesh->time;
        pphlist->nsrun = 0;
        pphlist->length = 0;
        pphlist->output_number = 0;
        // Generate file name
        std::string outn = pib->block_name.substr(6); // 6 because counting starts at 0!
        int outid = atoi(outn.c_str());
        // set file name
        std::string basename = pin->GetString("job","problem_id");
        pphlist->base_name.assign(basename);
        pphlist->base_name.append(".");
        char define_id[10];
        sprintf(define_id,"out%d",outid);  // default id="outN"
        pphlist->base_name.append(define_id);
        pphlist->base_name.append(".");
        char proc_id[11];
        sprintf(proc_id,"proc%d",Globals::my_rank);
        pphlist->base_name.append(proc_id);
      } else if (type.compare("traj") == 0) {
        // Create photon trajectory list
        // Get number of user output variables and confirm it is less than
        // the number of user variables
        int nuser_out = pin->GetOrAddInteger(pib->block_name,"nuser",0);
        if (nuser_out > pmy_mc->nuser_var) {
          std::stringstream msg;
          msg << "### ERROR in MCOutput constructor" << std::endl
              << "User output variables: " << nuser_out
              << " greater than user variables: " << pmy_mc->nuser_var << std::endl;
          throw std::runtime_error(msg.str().c_str());
        }
        int step_limit = pin->GetOrAddInteger(pib->block_name,"steplimit",10000);
        if (pmc->list_size_init <= 0) pmc->list_size_init = 1;
        ptraj = new PhotonTrajectoryList(pmc->list_size_init+1,step_limit,nuser_out);
        ptraj->pmy_mc = pmc;
        // Initialize photon list
        ptraj->length = 0;
        ptraj->maxstep = 0;
        ptraj->output_number = 0;
        // Generate file name
        std::string outn = pib->block_name.substr(6); // 6 because counting starts at 0!
        int outid = atoi(outn.c_str());
        // set file name
        std::string basename = pin->GetString("job","problem_id");
        ptraj->base_name.assign(basename);
        ptraj->base_name.append(".");
        char define_id[10];
        sprintf(define_id,"out%d",outid);  // default id="outN"
        ptraj->base_name.append(define_id);
        ptraj->base_name.append(".");
        char proc_id[11];
        sprintf(proc_id,"proc%d",Globals::my_rank);
        ptraj->base_name.append(proc_id);
      } else {
        // Look for moments
        std::string var = pin->GetOrAddString(pib->block_name,"variable","none");
        if (var.compare("mclab") == 0 || var.compare("Ermc") == 0 ||
            var.compare("Frmc") == 0 || var.compare("Prmc") == 0) {
          mom_flag_lab = true;
        } else if (var.compare("mccom") == 0 || var.compare("Ermc0") == 0 ||
                   var.compare("Frmc0") == 0 || var.compare("Prmc0") == 0) {
          mom_flag_com = true;
        } else if (var.compare("mccoord") == 0 || var.compare("Ermc_coord") == 0 ||
                   var.compare("Frmc_coord") == 0 || var.compare("Prmc_coord") == 0) {
          mom_flag_coord = true;
        } else if (var.compare("mcsrc") == 0) {
          mom_flag_src = true;
        } else if (var.compare("mcscat") == 0) {
          mom_flag_scat = true;
        } else if (var.compare("uom") == 0) {
          if (pmy_mc->nuser_mom > 0) {
            mom_flag_usr = true;
          } else {
            std::stringstream msg;
            msg << "### ERROR in MCOutput constructor" << std::endl
                << "user output moments requested bu nuser_mom = 0."
                << std::endl;
            ATHENA_ERROR(msg);
          }
        }
      }
    }
    pib = pib->pnext;  // move to next input block name
  }
  // set pspec to head node of specrum list
  pspec = pfirst;

}

// destructor
MCOutput::~MCOutput() {

}

//----------------------------------------------------------------------------------------
//! \fn void Spectrum::WriteSpectrum(std::string fname, Real tint_out)
//! \brief output intensity spectrum in original mcgrid format

void Spectrum::WriteSpectrum(std::string fname, Real tint_out) {

  // open file for output
  FILE *pfile;
  std::stringstream msg;
  if ((pfile = fopen(fname.c_str(),"w")) == nullptr) {
    msg << "### FATAL ERROR in function [Spectrum::WriteSpectrum]" << std::endl
        << "Output file '" << fname << "' could not be opened";
    throw std::runtime_error(msg.str().c_str());
  }
  // Write header information
  Real everg = 1.602176634e-12;
  Real emin = range.emin / everg; // output in eV
  Real emax = range.emax / everg; // output in eV
  int ne = range.ne;
  int nmu = range.ncth;
  int nphi = range.nphi;

  fprintf(pfile,"dt=%.8e\n",tint_out);
  fprintf(pfile,"nx=%d\n",ne);
  fprintf(pfile,"nmu=%d\n",nmu);
  fprintf(pfile,"nphi=%d\n",nphi);
  fprintf(pfile,"ntot=%d\n",nsrun);
  int nintens = 1 + NumStokesStored(polarized);
  fprintf(pfile,"nintens=%d\n",nintens);
  fprintf(pfile,"units=ev\n");
  fprintf(pfile,"polarized=%s\n",GetMCPolarizationName(polarized));
  fprintf(pfile,"yerror=true\n");
  // Coordinates and metric paramters
  fprintf(pfile,"coord=%s\n",pmy_mc->geometry_tag.c_str());
  if (!pmy_mc->metric_params.empty())
    fprintf(pfile,"metric_params=%s\n",pmy_mc->metric_params.c_str());
  fprintf(pfile,"frame=%s\n",pmy_mc->frame_tag.c_str());
  // Output bin faces with fwrite
  bool bigend = mcoutput::IsBigEndian();
  int nface = (ne+1 > nmu+1) ? ne+1 : nmu+1;
  nface = (nface > nphi+1) ? nface : nphi+1;
  double *faces;
  faces = new double[nface];
  everg = 1.602176634e-12;
  for (int i=0; i<ne+1; ++i)
    faces[i] = static_cast<double>(energies(i)/everg);
  if (!bigend) {for (int i=0; i<ne+1; ++i) mcoutput::Swap8Bytes(&faces[i]);}
  fwrite(faces,sizeof(double),static_cast<size_t>(ne+1),pfile);
  for (int i=0; i<nmu+1; ++i)
    faces[i] = static_cast<double>(i)/static_cast<double>(nmu);
  if (!bigend) {for (int i=0; i<nmu+1; ++i) mcoutput::Swap8Bytes(&faces[i]);}
  fwrite(faces,sizeof(double),static_cast<size_t>(nmu+1),pfile);
  for (int i=0; i<nphi+1; ++i)
    faces[i] = static_cast<double>(i)/static_cast<double>(nphi)*2.*PI;
  if (!bigend) {for (int i=0; i<nphi+1; ++i) mcoutput::Swap8Bytes(&faces[i]);}
  fwrite(faces,sizeof(double),static_cast<size_t>(nphi+1),pfile);
  delete [] faces;
  // Generate normalized intensties and errors
  Real *emid, *dnu;
  emid = new double[ne];
  dnu = new double[ne];
  Real h = 6.62607015e-27;
  for(int i=0; i<ne; ++i) {
    emid[i] = 0.5*(energies(i)+energies(i+1));
    dnu[i] = (energies(i+1)-energies(i))/h;
  }
  // First compute normalized intensities, stokes vectors, and errors
  Real norms;
  if (nsrun != pmy_mc->nsamp) {
    norms = static_cast<Real>(nsrun)/static_cast<Real>(pmy_mc->nsamp);
    // nsamp is 64-bit; %d on it is undefined and prints garbage.
    printf("nsrun != nsamp: %d %lld\n",nsrun,
           static_cast<long long>(pmy_mc->nsamp));
  } else {
    norms = 1.;
  }
  AthenaArray<Real> intens, errors;
  intens.NewAthenaArray(nintens,nphi,nmu,ne);
  errors.NewAthenaArray(nintens,nphi,nmu,ne);
  Real fac1 = norms*static_cast<Real>(nmu)*static_cast<Real>(nphi)/2./PI;
  // One normalization for every plane, intensity and Stokes alike, which is what
  // make_spectrum in athena_mc.py does: it builds a single factor and applies it to the
  // whole intensity array.  The Stokes planes used to be scaled in a separate loop that
  // omitted tint_out, so Q/U/V came out larger than I by the integration time and Q/I
  // read off a .spec was not the polarization fraction.  Invisible whenever tint_out
  // happens to be one, which it is in every test deck here.  Sharing one fac2 makes
  // the two impossible to get out of step again.
  for (int k = 0; k < nphi; ++k) {
    for (int j = 0; j < nmu; ++j) {
      Real mumid = (static_cast<Real>(j)+0.5)/static_cast<Real>(nmu);
      for (int i = 0; i < ne; ++i) {
        Real fac2 = fac1*emid[i]/(mumid*dnu[i]*tint_out);
        intens(0,k,j,i) = static_cast<double>(intensity(k,j,i)*fac2);
        errors(0,k,j,i) = sqrt(intensity_sq(k,j,i)*SQR(fac2));
        // Stokes planes follow the intensity in the order Q, U, V
        for (int m = 0; m < NumStokesStored(polarized); ++m) {
          intens(m+1,k,j,i) = static_cast<double>(stokes[m](k,j,i)*fac2);
          errors(m+1,k,j,i) = sqrt(stokes_sq[m](k,j,i)*SQR(fac2));
        }
      }
    }
  }
  delete [] emid;
  delete [] dnu;
  int ndata = nintens*ne*nmu*nphi;
  double *data;
  data = new double[ndata];
  // Output intensity and stokes parametres with fwrite
  int n = 0;
  for(int m=0; m<nintens; ++m) {
    for(int k=0; k<nphi; ++k) {
      for(int j=0; j<nmu; ++j) {
        for(int i=0; i<ne; ++i) {
          data[n++] = static_cast<double>(intens(m,k,j,i));
        }}}}
  if (!bigend) {for (int i=0; i<ndata; ++i) mcoutput::Swap8Bytes(&data[i]);}
  fwrite(data,sizeof(double),static_cast<size_t>(ndata),pfile);
  // Output normalized errors
  n = 0;
  for(int m=0; m<nintens; ++m) {
    for(int k=0; k<nphi; ++k) {
      for(int j=0; j<nmu; ++j) {
        for(int i=0; i<ne; ++i) {
          data[n++] = static_cast<double>(errors(m,k,j,i));
        }}}}
  if (!bigend) {for (int i=0; i<ndata; ++i) mcoutput::Swap8Bytes(&data[i]);}
  fwrite(data,sizeof(double),static_cast<size_t>(ndata),pfile);
  fclose(pfile);
  delete [] data;
  intens.DeleteAthenaArray();
  errors.DeleteAthenaArray();

}


//----------------------------------------------------------------------------------------
//! \fn void MCOutput::OutputSpectrum(bool wtflag)
//! \brief output all intensity spectra

void MCOutput::OutputSpectrum(bool wtflag) {

  if (pspec == nullptr) //no spectra requested
    return;

  Spectrum *pspect = pspec;
  Spectrum *pspecout = nullptr;
  // Check if any spectra are ready to be output
  Real time = pmy_mc->pmy_mesh->time;
  Real tstart = pmy_mc->pmy_mesh->start_time;
  Real tlim = pmy_mc->pmy_mesh->tlim;
  while (pspect != nullptr) {
    if ( (time >= pspect->last_time+pspect->dt) || (time == tstart) || (time >= tlim)
         || wtflag ) {
      if (Globals::my_rank == 0) {
        pspecout = new Spectrum(pspect);
      }
#ifdef MPI_PARALLEL
      if (Globals::my_rank == 0) {
        // Receive spectra from other processes and add to output spectrum
        for(int i=1; i<Globals::nranks; ++i)
          ReceiveMonteCarloSpectrum(pspecout,true);
      } else {
        SendMonteCarloSpectrum(pspect,0);
      }
#endif
      if (Globals::my_rank == 0) {
        // Add spectrum from this process.  This has to happen with or without MPI: the
        // copy constructor allocates pspecout empty and sets nsrun = 0, so a serial run
        // that skipped this wrote an all-zero spectrum -- WriteSpectrum normalizes by
        // nsrun/nsamp, which is zero, scaling every bin to zero.
        pspecout->AddSpectrum(pspect);

        std::string filename;
        filename.assign(pspecout->base_name);
        filename.append(".");
        std::stringstream file_number;
        file_number << std::setw(5) << std::setfill('0') << pspecout->output_number;
        filename.append(file_number.str());
        filename.append(".spec");
        // compute integration time in cgs
        Real tint_out;
        if (pmy_mc->dynamic) {
          tint_out = (time - pspect->last_time) * time_cgs;
        } else {
          tint_out = pmy_mc->tint;
        }
        pspecout->WriteSpectrum(filename,tint_out);
        if (pspecout != nullptr)
          delete pspecout;
      }
      // Update spectra on all blocks
      pspect->output_number++;
      pspect->ResetSpectrum();
      if (pmy_mc->dynamic) {
        pspect->last_time = time;
      }
    }
    pspect = pspect->next;
  } // end while loop

}

//----------------------------------------------------------------------------------------
//! \fn void MCOutput::SendMonteCarloSpectrum(Spectrum *pspect, int dest)
//! \brief send one monte carlo spectrum to another process

void MCOutput::SendMonteCarloSpectrum(Spectrum *pspect, int dest) {
#ifdef MPI_PARALLEL

  int ne = pspect->range.ne;
  int ncth = pspect->range.ncth;
  int nphi = pspect->range.nphi;
  int size = 2 + 2*NumStokesStored(pspect->polarized);
  size *= (ne*ncth*nphi);

  Real *send_buf;
  send_buf = new Real[size];
  MPI_Request send_rq;
  unsigned int tag = 100; //temporary

  int p=0;
  ne--; ncth--; nphi--;
  MPI_Isend(&pspect->nsrun,1,MPI_INT,dest,tag++,MPI_COMM_WORLD,&send_rq);
  MPI_Wait(&send_rq, MPI_STATUS_IGNORE);
  BufferUtility::PackData(pspect->intensity,send_buf,0,ne,0,ncth,0,nphi,p);
  BufferUtility::PackData(pspect->intensity_sq,send_buf,0,ne,0,ncth,0,nphi,p);
  for (int m = 0; m < NumStokesStored(pspect->polarized); ++m) {
    BufferUtility::PackData(pspect->stokes[m],send_buf,0,ne,0,ncth,0,nphi,p);
    BufferUtility::PackData(pspect->stokes_sq[m],send_buf,0,ne,0,ncth,0,nphi,p);
  }
  MPI_Isend(send_buf,size,MPI_ATHENA_REAL,dest,tag++,MPI_COMM_WORLD,&send_rq);
  MPI_Wait(&send_rq, MPI_STATUS_IGNORE);

  delete [] send_buf;
#endif
}


//----------------------------------------------------------------------------------------
//! \fn void MCOutput::ReceiveMonteCarloSpectrum(Spectrum *pspect, bool add)
//! \brief receive one monte carlo spectrum from another process

void MCOutput::ReceiveMonteCarloSpectrum(Spectrum *pspect, bool add) {
#ifdef MPI_PARALLEL
  // Calling function checks to see that send and receive spectra match
  int ne = pspect->range.ne;
  int ncth = pspect->range.ncth;
  int nphi = pspect->range.nphi;
  int size = 2 + 2*NumStokesStored(pspect->polarized);
  size *= (ne*ncth*nphi);

  Real *recv_buf;
  recv_buf = new Real[size];
  unsigned int tag = 100; // temporary

  ne--; ncth--; nphi--;
  int nsrun;
  MPI_Status st;
  // Two corrections to what this used to do.  The count is 1, not size: nsrun is a single
  // int, and posting a receive for size of them into it overruns the stack the moment any
  // sender puts more than one on this tag.  And the source is taken from the first
  // message rather than left as MPI_ANY_SOURCE on both, so the sample count and the
  // spectrum that follows it come from the same rank; with ANY_SOURCE twice, one rank's
  // count could be paired with another's spectrum.
  MPI_Recv(&nsrun,1,MPI_INT,MPI_ANY_SOURCE,tag++,MPI_COMM_WORLD,&st);
  MPI_Recv(recv_buf,size,MPI_ATHENA_REAL,st.MPI_SOURCE,tag++,MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  Spectrum *ptemp;
  if (add) {
    // Make temporary spectrum for copying, initalized empty
    ptemp = new Spectrum(pspec);
  } else {
    // Copy buffer directly into destination spectrum
    ptemp = pspect;
  }
  ptemp->nsrun += nsrun;

  int p=0;
  BufferUtility::UnpackData(recv_buf,ptemp->intensity,0,ne,0,ncth,0,nphi,p);
  BufferUtility::UnpackData(recv_buf,ptemp->intensity_sq,0,ne,0,ncth,0,nphi,p);
  for (int m = 0; m < NumStokesStored(pspect->polarized); ++m) {
    BufferUtility::UnpackData(recv_buf,ptemp->stokes[m],0,ne,0,ncth,0,nphi,p);
    BufferUtility::UnpackData(recv_buf,ptemp->stokes_sq[m],0,ne,0,ncth,0,nphi,p);
  }
  if (add) {
    pspect->AddSpectrum(ptemp);
    delete ptemp;
  }
  delete [] recv_buf;
#endif
}


//----------------------------------------------------------------------------------------
//! \fn void MCOutput::OutputPhotonList(bool wtflag)
//! \brief output list of photon properties

void MCOutput::OutputPhotonList(bool wtflag) {

  if (pphlist == nullptr)
    return;

  // Check if any spectra are ready to be output
  Real time = pmy_mc->pmy_mesh->time;
  Real tstart = pmy_mc->pmy_mesh->start_time;
  Real tlim = pmy_mc->pmy_mesh->tlim;
  if ( (time >= pphlist->last_time+pphlist->dt) || (time == tstart) || (time >= tlim)
       || wtflag ) {
    // Filename() rather than a second copy of this logic: the spill path opens the file
    // before this point and the two names have to agree.
    std::string filename = pphlist->Filename();
    // compute integration time in cgs
    Real tint_out;
    if (pmy_mc->dynamic) {
      tint_out = (time - pphlist->last_time) * time_cgs;
    } else {
      tint_out = pmy_mc->tint;
    }
    pphlist->WriteList(filename,tint_out);
    pphlist->output_number++;
    // Reset list length and ntot to 0
    // List outputs are not cumulative
    pphlist->last_time = time;
    // Photon lists are always reset to zero upon output
    pphlist->ResetList();
  }

}

//----------------------------------------------------------------------------------------
//! \fn void MCOutput::OutputTrajectoryList()
//! \brief output list of photon trajectories

void MCOutput::OutputTrajectoryList() {

  if (ptraj == nullptr)
    return;

  std::string filename;
  filename.assign(ptraj->base_name);
  filename.append(".");
  std::stringstream file_number;
  file_number << std::setw(5) << std::setfill('0') << ptraj->output_number;
  filename.append(file_number.str());
  filename.append(".traj");
  ptraj->WriteList(filename);
  ptraj->output_number++;
  // Reset list length to 0
  ptraj->length = 0;

}

//----------------------------------------------------------------------------------------
//! \fn void MCOutput::UpdateOutputCount(int nph)
//! \brief updates total numbers of photons run for output normalization

// SWD Add Trajectory, image?
void MCOutput::UpdateOutputCount(int nph) {

  if (pphlist != nullptr)
    pphlist->nsrun += nph;

  Spectrum *pspect = pspec;
  while (pspect != nullptr) {
    pspect->nsrun += nph;
    pspect = pspect->next;
  }

}

//----------------------------------------------------------------------------------------
//! \fn void MCOutputs::MakeOutputs(bool wtflag)
//! \brief write MonteCarlo outputs

void MCOutput::MakeOutputs(bool wtflag) {

  OutputSpectrum(wtflag);
  OutputPhotonList(wtflag);
  OutputTrajectoryList();

}
