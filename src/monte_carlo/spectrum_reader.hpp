#ifndef SPECTRUM_READER_HPP_
#define SPECTRUM_READER_HPP_

#include <string>
#include <vector>
#include "../athena.hpp"

// Read two-column spectrum data from ASCII file
//   - Column 0: wavelength [cm]
//   - Column 1: intensity [erg/cm^2/s/cm]
// Modifies wavelength 'wl' and cumulative density function 'cdf' as vectors
// Also accumulates total intensity 'itot' and average energy 'emean'
void ReadSpectrumToCDF(const std::string& filename, std::vector<Real>& wl, std::vector<Real>& cdf, Real& itot, Real& emean);

#endif // SPECTRUM_READER_HPP_
