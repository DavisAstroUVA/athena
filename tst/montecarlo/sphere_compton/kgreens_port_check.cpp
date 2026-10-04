// Port check of KompaneetsTable against the Python Sampler.  Built and run by
// kgreens_port_check.py:
//   g++ -O2 -std=c++11 -I<athena root> kgreens_port_check.cpp <root>/src/monte_carlo/kgreens.cpp
// Reads (xi, y, lam, r) rows from stdin and prints xf and S per row.

#include <cstdio>
#include <string>

#include "src/monte_carlo/kgreens.hpp"

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s table.bin < cases\n", argv[0]);
    return 2;
  }
  KompaneetsTable tab;
  tab.Read(argv[1]);
  double xi, y, lam, r;
  while (std::scanf("%lf %lf %lf %lf", &xi, &y, &lam, &r) == 4) {
    std::printf("%.17g %.17g\n", tab.Quantile(xi, y, lam, r), tab.Survival(xi, y, lam));
  }
  return 0;
}
