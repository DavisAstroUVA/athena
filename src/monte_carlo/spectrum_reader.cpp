// spectrum_reader.cpp

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "spectrum_reader.hpp"
#include "montecarlo.hpp"
#include "../athena.hpp"

void ReadSpectrumToCDF(const std::string& filename, std::vector<Real>& wl, std::vector<Real>& cdf, Real& itot, Real& emean) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("ReadSpectrumToCDF: cannot open file " + filename);
  }

  // clear any values that might already be in wl and cdf
  wl.clear();
  cdf.clear();

  std::vector<Real> ilam;
  std::string line;
  int line_number = 0;
  while (std::getline(file, line)) {
    ++line_number;

    // skip empty lines and comments
    if (line.empty()) continue;
    if (line[0] == '#') continue;

    std::istringstream iss(line);
    Real wavelength, intensity;

    // check for lines that don't read nicely into two doubles
    if (!(iss >> wavelength >> intensity)) {
      throw std::runtime_error("ReadSpectrumToCDF: malformed line " + std::to_string(line_number) + " in input file '" + filename + "'");
    }

    wl.push_back(wavelength);
    ilam.push_back(intensity);

  }
  file.close();

  const int nrows = wl.size();
  if (nrows == 0) {
    throw std::runtime_error("ReadSpectrumToCDF: no valid data in file " + filename);
  }
  if (nrows < 2) {
    throw std::runtime_error("ReadSpectrumToCDF: need at least 2 data points for interpolation, found " + std::to_string(nrows));
  }

  // PDF = ilam / (int dlambda*ilam) = ilam / itot
  // compute CDF[i] as int dlambda*ilam from wl[0] to wl[i]
  // compute mean energy as int dlambda * (ilam/itot) * (h*c/lambda)
  // assumes wavelengths are uniformly spaced and increasing
  Real dlambda = wl[1] - wl[0];
  cdf.push_back(0.);
  emean = 0.;

  // integrate using trapezoid rule
  for (int i = 1; i < nrows; ++i) {
    cdf.push_back(cdf[i-1] + 0.5*dlambda*(ilam[i-1] + ilam[i]));
    emean += 0.5*dlambda*(ilam[i-1]/wl[i-1] + ilam[i]/wl[i]);
  }
  emean *= MCConstants::h_cgs * MCConstants::c_cgs;
  
  // normalize: CDF runs from 0 to 1
  itot = cdf[nrows-1];
  if (itot == 0.) {
    throw std::runtime_error("ReadSpectrumToCDF: cdf norm is zero");
  }
  for (int i=0; i<nrows; ++i) {
    cdf[i] = cdf[i] / itot;
  }
  emean /= itot;

} // end ReadSpectrumToCDF
