# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

This is a fork of **Athena++**, a C++ GRMHD (General Relativistic Magnetohydrodynamics) code with AMR (Adaptive Mesh Refinement). This fork adds a **Monte Carlo radiation transport** module (`src/monte_carlo/`) for post-processing or coupled radiation-MHD simulations. All developer additions live on `main`; work happens on short-lived feature branches that are squash-merged through pull requests and then deleted, so start every new branch from a freshly pulled `main`, never from an already merged branch.

## Build system

Athena++ uses a two-step configure-then-make build, not CMake.

**Configure** (generates `Makefile` and `src/defs.hpp` from templates):
```bash
python configure.py --prob=<problem_generator> [options]
```

Key configure flags:
- `--prob=<name>` — selects `src/pgen/<name>.cpp` as the problem generator
- `--coord=<cartesian|cylindrical|spherical_polar|kerr-schild|...>` — coordinate system
- `-mc` — enable Monte Carlo radiation transport
- `-b` — enable magnetic fields
- `-g` — enable general relativity
- `-mpi` — enable MPI
- `-omp` — enable OpenMP
- `-hdf5` — enable HDF5 output
- `-debug` — enable debug flags (`-g -O0`)
- `-gsl` — use GSL for random numbers instead of `<random>`

**Build:**
```bash
make -j<N>
```

**IMPORTANT — after editing any header, run `make clean` before `make`.** The Makefile
tracks no header dependencies. This is a deliberate upstream Athena++ choice and is kept
here for consistency, so it is not a bug to fix; it is a rule to follow. A plain `make`
after a header edit rebuilds only the `.cpp` files that changed, and the result is one of:

- a **link error** naming an undefined symbol or a vtable, if the change altered a class's
  virtual functions — annoying but obvious; or
- a **silently mismatched binary**, if the change altered a type's size or layout. Half the
  objects then use the old layout and half the new, so reads of a member return another
  member's bytes. This presents as impossible behaviour — a flag that is true where it is
  set and false where it is read, through the same pointer — and is easy to spend a long
  time misdiagnosing as a logic error.

Anything that changes a layout counts, including a one-word edit like giving an enum a
different underlying type. If a build behaves inexplicably, check `ls -l obj/ src/**/*.hpp`
for objects older than a header before debugging further.

**Run:**
```bash
bin/athena -i inputs/mc/athinput.<problem>
```

Example full workflow for a Monte Carlo disk simulation:
```bash
python configure.py --prob=mc_disksim --coord=spherical_polar -mc -mpi -hdf5
make -j8
mpirun -n <N> bin/athena -i inputs/mc/athinput.mciso
```

## Testing

**Regression tests** (standard Athena++ test suite):
```bash
cd tst/regression
python run_tests.py                        # run all tests
python run_tests.py scripts/tests/hydro    # run a specific test directory
```

**Monte Carlo convergence tests** (custom, in `tst/montecarlo/`):
```bash
cd tst/montecarlo/thomson_polarized_spectrum
python convergence.py <iseed> <nmin> <nstep> <step> [--mcranks N] [--path /path/to/athena]

cd tst/montecarlo/boosts
python convergence.py <iseed> <nmin> <nstep> <step> [--vel 0.1] [--scat] [--path /path/to/athena]

# parallel transport of the polarization tensor; refines on step size, not photon number
cd tst/montecarlo/snake_polarization
python convergence_dd.py <iseed> <nstep> [--beta 0.3] [--polcirc 0.0] [--path /path/to/athena]

# single-photon polarized transport regression, PASS/FAIL
# needs: python configure.py --prob=mc_poltest --coord=gr_user -g -mc && make
cd tst/montecarlo/poltest
python poltest.py [--workdir DIR] [--path /path/to/athena]

# polarized Thomson atmosphere in snake coordinates vs Feautrier; --beta 0 is the control
# needs: python configure.py --prob=mc_snake_atm --coord=gr_user -g -mc && make
cd tst/montecarlo/snake_thomson_spectrum
python convergence_dd.py <iseed> <nmin> <nstep> <step> [--beta 0.3] [--mcranks N]

# single-photon polarization in spherical polar, PASS/FAIL: transport order, the comoving
# round trip at a scatter (radial, polar and azimuthal drift), output referencing, the
# photon list's wavevector basis, and transversality of the tensor against its wavevector
# needs: python configure.py --prob=mc_sphpol --coord=spherical_polar -mc && make
cd tst/montecarlo/spherical_polarization
python sphpol.py [--workdir DIR] [--path /path/to/athena]

# stratified disk atmosphere in spherical polar vs Feautrier, either pusher; the end-to-end
# gate.  Bins its own photon lists with a height mask (see the module docstring for why).
# needs: python configure.py --prob=mc_isoth --coord=spherical_polar -mc -mpi && make
cd tst/montecarlo/disk_atmosphere
python disk_atmosphere.py [--general-pusher] [--nphot N] [--nstep 3] [--mcranks N] [--path /path/to/athena]
```

These tests configure, compile, run Athena++, and compare results against analytic solutions (Feautrier, blackbody).

## Monte Carlo module architecture

The MC module lives entirely in `src/monte_carlo/` and is activated by the `-mc` configure flag which sets `MONTE_CARLO_ENABLED`.

**Key classes:**

| Class | File | Role |
|---|---|---|
| `MonteCarlo` | `montecarlo.cpp/.hpp` | Top-level MC object owned by `Mesh`; holds global config, boundary conditions, user function pointers |
| `MonteCarloBlock` | `montecarloblock.cpp` | Per-`MeshBlock` MC state; owns photons and pusher |
| `Photon` | `photon.cpp/.hpp` | Stores all photon data as parallel `std::vector` arrays (SOA layout); one `Photon` object per block holds all photons for that block |
| `PhotonPusher` | `photonpusher.cpp/.hpp` | Abstract base; `CartesianPusher`, `SphericalPolarPusher`, `GeneralPusher` are derived classes that implement coordinate-specific photon propagation |
| `MCCoord` | `mccoord.cpp/.hpp` | Coordinate utilities for MC (distinct from Athena's hydro `Coordinates`) |
| `MCBoundaryValues` | `mcbvals.cpp/.hpp` | Handles photon communication across MeshBlock boundaries via MPI |
| `MCOutput` | `mcoutput.cpp/.hpp` | Writes photon lists and radiation moment arrays |

**Physics:**
- `emission.cpp` — photon emission (free-free, blackbody, user-defined)
- `scattering.cpp` — scattering (isotropic, Thomson, Compton, resonance, dust)
- `opacity.cpp` — absorption/scattering opacities
- `tetrad.cpp` / `tetrad.hpp` — geometric primitives: Gram-Schmidt tetrad construction, vector algebra in a metric, coordinate/tetrad transforms, Stokes/tensor. Acts on bare four-vectors; knows nothing about photons or blocks
- `photon_frames.cpp` / `photon_frames.hpp` — the layer above: projects a photon into a named frame (`MCFRAME_LAB`, `MCFRAME_COMOVING`, `MCFRAME_COORD`) and transforms accumulated moments between frames. See `doc/monte_carlo/frames_and_moments.tex` for the mathematics
- `polarization.cpp` / `polarization.hpp` — Stokes/coherency-tensor conversions and the meridian basis they are referenced to; `GeneralPusher::AdvanceStep` transports the tensor with Heun's method (second order)

Planned work is written up rather than carried in anyone's head: see `doc/monte_carlo/faraday_rotation_plan.md` for Faraday rotation and conversion, and `doc/monte_carlo/polar_axis_notes.md` for the general pusher's known defects at the polar axis (a stub polar boundary and a runaway integrator near theta = 0) with both fixes specified and deferred. `doc/monte_carlo/memory_and_speed_plan.md` is the ordered plan for cutting the memory footprint and per-step cost of GR runs (memory items first, then speed), with the verification criterion for each item. `doc/monte_carlo/load_balancing_plan.md` assesses reusing the mesh's load-balancing machinery for Monte Carlo transport (measured per-block cost, redistribution mid-transport) and lays out the steps with their tests; its status block records what is built.

**Load balancing** (static and dynamic Monte Carlo runs): each block's transport sweep time is added to `MeshBlock::cost_`, and the mesh's balancer (`<loadbalancing> balancer = automatic`, `tolerance`, `interval`; set `interval = 1` to consider every output interval) redistributes blocks, with Monte Carlo hooks in `RedistributeAndRefineMeshBlocks` moving photons, moments and RNG state. Module keys live in `<montecarlo>`: `lb_report` (per-rank and per-block cost report), `lb_check_interval` and `lb_check_fraction` (mid-transport checks in the synchronous and asynchronous loops), `lb_max_per_transport`, `lb_min_window`, `lb_min_gain` (a redistribution is taken only if the predicted busiest rank improves by this), `lb_partition` (optimal contiguous by default), `lb_cost_decay`, `lb_initial`, and `<loadbalancing> cost_file` to carry costs between runs. `lb_test_repack` and `lb_test_costs` are test knobs. The `mc_readhdf*` generators refuse balancing with table opacities.

**Photon data layout:** Each photon is stored as a struct-of-arrays via `static int` index members (`ix1p`, `ik1p`, etc.) on the `Photon` class. Position is `(x0p,x1p,x2p,x3p)` (covariant coordinates), wavevector is `(k0p,k1p,k2p,k3p)`, and under the general pusher the coherency tensor is stored Hermitian-packed as sixteen real columns (`polten`, accessed only through `Photon::LoadTensor`/`StoreTensor`/`Tensor`). Status flags use `PhotonStatus` enum (`EVOLVING`, `ESCAPED`, `ABSORBED`, `DESTROYED`, `BUFFERED`).

**Monte Carlo radiation moments** are stored in `NMOM=15` arrays indexed by `MCIER`, `MCIFRx`, `MCIPRxy` enums (energy density, flux, pressure tensor).

## Problem generators

Problem generators live in `src/pgen/`. Monte Carlo problem generators have the naming convention `mc_*.cpp` and guard themselves with `#if !MONTE_CARLO_ENABLED`. The ones most often built:
- `mc_readhdf_gr.cpp` — X-ray binary in Cartesian Kerr-Schild (`gr_user`), initialized from an athdf snapshot; the production GR problem
- `mc_readhdf.cpp` — non-relativistic counterpart, reads an athdf snapshot
- `mc_disksim.cpp` — disk simulation reading from VTK hydro snapshots
- `mc_isoth.cpp`, `mc_isoth_gr.cpp` — isothermal atmosphere, flat and Kerr-Schild (convergence and disk atmosphere tests)
- `mc_snake.cpp`, `mc_snake_atm.cpp` — uniform box and stratified atmosphere in snake coordinates (flat spacetime, sheared chart)
- `mc_poltest.cpp`, `mc_sphpol.cpp` — single-photon polarized transport tests, `gr_user` and spherical polar
- `mc_disk.cpp` — in progress, untracked

## Visualization / post-processing

Python tools are in `vis/python/montecarlo/`. Add this directory to `PYTHONPATH` to use them.

- `athena_mc.py` — core library; `read_list_generator()` streams photon list files in chunks, `Photons` class wraps a single chunk, `make_spectrum()` bins photons into a spectrum
- `make_spectrum.py` — CLI wrapper around `athena_mc.make_spectrum()`
- `plot_spectrum.py` — plots spectra
- `make_spectrum_single.py` — single-file variant (untracked, in progress)
- `scattering_histogram.py` — distribution of the per-photon scattering count, read from whichever user variable a problem generator copied `nscp` into (`--nscat-col`, default 0; `mc_readhdf*` use 2). Reports how concentrated the scatterings are in the worst photons, which is the number that matters in a resonant-line run where the mean is set by a handful of photons

Photon list files use a custom binary format read by `athena_mc.read_list_generator()`.

## Code style

- Follows the [Athena++ Style Guide](https://github.com/PrincetonUniversity/athena/wiki/Style-Guide)
- `CPPLINT.cfg` enforces Google C++ style via `cpplint`
- Headers use `#ifndef FILENAME_HPP` guards
- Doxygen-style `//!` comments on classes and key functions
- User-customizable physics hooks are set via function pointers on `MonteCarlo` (e.g. `GetEmission`, `UserScattering`, `UserAbsorptionOpacity`)
