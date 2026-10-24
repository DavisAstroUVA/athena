# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

This is a fork of **Athena++**, a C++ GRMHD (General Relativistic Magnetohydrodynamics) code with AMR (Adaptive Mesh Refinement). This fork adds a **Monte Carlo radiation transport** module (`src/monte_carlo/`) for post-processing or coupled radiation-MHD simulations. The user documentation of the module is the GitHub wiki of this repository, maintained in the clone at `/home/swd8g/wiki/athena.wiki` (one Markdown file per page, pushed with git); it is the reference for keys, outputs, tools and methods, and this file keeps the rules for working on the code and pointers to the pages. `doc/monte_carlo/` holds the project records: plans with status blocks, measurements and decisions. Input keys live in the block of the feature they belong to (`<montecarlo>` for the run itself, `<sampling>`, `<loadbalancing>`, `<randomwalk>`, `<opacitytable>`, the `<output[n]>` block of the output they configure); a key read from a new place leaves its old location in the relocation table of `CheckRelocatedKeys` in `montecarlo.cpp`, which refuses an input file that still sets it there, and that table is dropped once the input files have moved. Any change to the input keys, the outputs, the tools or a method is a change to the wiki: say which pages need updating and propose the text for approval before editing them. All developer additions live on `main`; work happens on short-lived feature branches that are squash-merged through pull requests and then deleted, so start every new branch from a freshly pulled `main`, never from an already merged branch.

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

# Compton scattering in a uniform sphere vs the Kompaneets Green's function (point source;
# --absorption freefree adds the free-free survival; --accel turns the random walk on)
# needs: python configure.py --prob=mc_sphere_isoth -mc && make
cd tst/montecarlo/sphere_compton
python sphere_compton.py --run --workdir DIR [--tau 15 --x0 1 --theta 2e-3] [--template ../../../inputs/mc/athinput.sphere_compton_grid --nx 8] [--accel --kgreens TABLE.bin]

# stratified disk atmosphere in spherical polar vs Feautrier, either pusher; the end-to-end
# gate.  Bins its own photon lists with a height mask (see the module docstring for why).
# needs: python configure.py --prob=mc_isoth --coord=spherical_polar -mc -mpi && make
cd tst/montecarlo/disk_atmosphere
python disk_atmosphere.py [--general-pusher] [--nphot N] [--nstep 3] [--mcranks N] [--path /path/to/athena]

# table opacities: the module path against a binary built before it existed
# (byte-identical lists), table_storage = cell against global, and the table path against
# free-free on a free-free table (boxes over a grid of T and rho; a Thomson atmosphere)
# needs: python configure.py --prob=mc_readhdf -mc -hdf5 && make (snapshot in ~/mc_test/read_hdf5),
#        or --prob=mc_isoth_opac -mc -hdf5 for the free-free comparisons
cd tst/montecarlo/opacity_table
python port_equivalence.py --reference OLD/bin/athena --path /path/to/athena --workdir DIR
python storage_equivalence.py --path /path/to/athena --workdir DIR --pgen readhdf|readhdf_gr|isoth_opac
python ff_table.py --path /path/to/athena --workdir DIR [--emax-ext 100]
python ff_atmosphere.py --path /path/to/athena --workdir DIR
python ne_table.py --path /path/to/athena --workdir DIR --table TABLE.txt   # with TABLE_nfree.txt beside it
python twoband_box.py --path /path/to/athena --workdir DIR --table TABLE.txt
python rosseland_slab.py --path /path/to/athena --workdir DIR [--table TABLE.txt]
python repack_equivalence.py --path /path/to/athena --workdir DIR --table-dir DIR --table TABLE.txt   # <loadbalancing> test_repack, bitwise on one rank
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

Planned work is written up rather than carried in anyone's head: see `doc/monte_carlo/faraday_rotation_plan.md` for Faraday rotation and conversion, and `doc/monte_carlo/polar_axis_notes.md` for the general pusher's known defects at the polar axis (a stub polar boundary and a runaway integrator near theta = 0) with both fixes specified and deferred. `doc/monte_carlo/memory_and_speed_plan.md` is the ordered plan for cutting the memory footprint and per-step cost of GR runs (memory items first, then speed), with the verification criterion for each item. `doc/monte_carlo/compton_ddmc_plan.md` is the deferred outline of a discrete diffusion Monte Carlo for Compton scattering, the method that fits the XRB's regime of cells a few mean free paths thick where the in-cell random walk cannot reach. `doc/monte_carlo/load_balancing_plan.md` assesses reusing the mesh's load-balancing machinery for Monte Carlo transport (measured per-block cost, redistribution mid-transport) and lays out the steps with their tests; its status block records what is built.

**Load balancing**: user documentation on the wiki page Load Balancing (`/home/swd8g/wiki/athena.wiki/Load-Balancing.md`). For the code: each block's transport sweep time goes into `MeshBlock::cost_`, the mesh's balancer (`<loadbalancing>`) redistributes blocks, and Monte Carlo hooks in `RedistributeAndRefineMeshBlocks` move photons, moments and RNG state; the module's keys (`initial`, `check_interval`, `check_fraction`, `max_per_transport`, `min_window`, `min_gain`, `partition`, `cost_decay`, `report`) are in `<loadbalancing>` beside the mesh's own, with `test_repack` and `test_costs` as test knobs. Table opacities, escape tables and two-band arrays are rebuilt on arrival, so every generator can be balanced; user block data is not carried.

**Sample weighting** (`<sampling> weights = emission | equal | biased`, `bias_mix`, `bias_energy`, `bias_energy_mix`, `minweight`, `roulette`): wiki page Sampling and Weighting (`/home/swd8g/wiki/athena.wiki/Sampling-and-Weighting.md`); the derivations and the X-ray binary measurements are in `doc/monte_carlo/biased_sampling_guide.tex`. What an agent must not forget: all three schemes are unbiased; the escape-table importance rates photons by birth energy, so `bias_energy_mix = 1` lost the soft-born photons that Compton-upscatter into the hard band (8 to 24 percent low on the hires XRB, the emission run being right); effective counts and per-bin errors are biased low by heavy tails, so use `nout > 1` and the scatter between outputs as the error.

**Weight window** (`<sampling> window_top`, `window_bottom`, `window_max_split`, `window_energy`, `window_interp`; needs `weights = biased`): same wiki page. Copies share their history, so photon-level errors of a window run are meaningless (`family_stats.py`, `make_spectrum.py --family-cols`); on the hires XRB the window cost 15.7 times the emission run and beat it per CPU only at 0.1 to 0.3 keV, so `weights = emission` remains the production choice there.

**Random walk acceleration** (`<randomwalk> enable`, `tau`, `face_tau`, `domain`, `pmax`, `kgreens_file`, `report`; `src/monte_carlo/mrw.cpp`, `mrw.hpp`, `kgreens.cpp`): wiki page Random Walk Acceleration (`/home/swd8g/wiki/athena.wiki/Random-Walk-Acceleration.md`), the full account in `doc/monte_carlo/mrw_guide.tex`, the Green's function in `doc/monte_carlo/kompaneets/`, the plan in `doc/monte_carlo/compton_mrw_plan.md`. For the code: the step is a fixed-time step in the cell box, never crosses a cell, declines outside the Kompaneets table's validity and in two-band runs; `MonteCarloBlock::accel_mask` keeps it out of cells a generator excludes; the general pusher needs `varystep = true` in flat spacetime, stops at cell faces, and keeps a first-order `stepsize` dependence (about 2.5 percent in the escaping fraction between 0.1 and the zero-step limit on the hires XRB; the suspected cause is the start-of-step extinction charged over a step along which the comoving frequency changes, a trapezoid rule being the candidate fix; use 0.05 where a percent matters). On the hires XRB the step is right but saves nothing (9 percent of scatterings in cells thick enough); a region step over several cells (plan Section 6.7) is where a saving would come from. Tests in `tst/montecarlo/sphere_compton/`.

**Table opacities** (`<montecarlo> emission = table`, `absorption = table`; the table's own keys in `<opacitytable>`; `src/monte_carlo/opacity_table.cpp`, `.hpp`): wiki page Table Opacities (`/home/swd8g/wiki/athena.wiki/Table-Opacities.md`); design, tests and measurements in `doc/monte_carlo/opacity_table_plan.md`. For the code: `OpacityTable` is read once per rank into `MonteCarlo::optab`, and `MonteCarloBlock::BuildOpacityTables` (from `SetupBlockFromFluid`, after the number densities) locates every active cell, so dynamic runs, restarts and redistributions rebuild the per-cell tables; `TableAbsorptionOpacity`, `GetEmissionTable` and `PhotonEmitTable` are the entry points, called as the free-free ones are; `mc_readhdf`, `mc_readhdf_gr` and `mc_isoth_opac` refuse `user` opacities; `table_groups = edge` is the pre-module reading kept for reproducing old runs (`port_equivalence.py` holds it byte-identical to a pre-module binary for `mc_readhdf`, statistically for `mc_readhdf_gr`, whose excluded cells now carry zero species); `table_storage = cell | global` give byte-identical lists (`storage_equivalence.py`); a photon carries a band rank `ubp` and last group `grpp`; with two bands use `abs_method = prob`. The readers take the module's number densities: `DefaultNumberDensity` is public, and `mc_readhdf_gr`'s `ExcludeCells` calls it and zeroes the species below `dcut`, above `tcut` and inside the horizon, setting `tab_mask` above `tcut`. Findings that change practice: the edge reading shifts every strong group half a group down (17 times the mean reading's 1 to 10 eV emission on the TDE; `snapshot_emission.py` shows it without transport); `abs_method = prob` was 27 times cheaper at the same bands on the TDE; the regridded table's extrapolation below the TOPS density edge makes the TDE outflow absorbing where the old table's gray rows under `table_gray = freefree` left it transparent, consistent with the hydro's own opacities; the two-band split moves 30 to 100 eV up by a factor 2 to 3 against the Planck mean at the same total and decorrelation at every crossing takes a quarter back; the inner-boundary photons of the TDE are half its escaping luminosity and everything above 0.3 keV (screen or `absorb`); the within-cell expansion opacity under shear is deferred indefinitely.

**Path stretching** (`<sampling> stretch`, `stretch_taucell`, `stretch_bound`): same wiki page as the weighting. For the code: `Photon::strp` holds the flight's log weight factor and `Photon::taup` the optical depth left in the flight, carried across blocks so that a flight whose interaction point lies on a neighbouring block interacts there (this also holds for analog transport).

**Transport cadence** (dynamic runs, `cadence`, `cadence_frac`, and per emission type `cadence_type[etype]`, set by the generator, as `mc_hotjupiter` does from `<problem> ion_cadence` and `lya_cadence`): wiki page Dynamic Runs (`/home/swd8g/wiki/athena.wiki/Dynamic-Runs.md`); the design record is `doc/monte_carlo/per_type_cadence_plan.md`. For the code: `TransportDue` marks the types due; each type's emission carries `tint_type`, the time since its own last transport, so the outputs normalize by `MonteCarlo::ttransport` unchanged; the per-type moment frames are normalized by `TintOfType(m)`; the arrays summed over types (source terms, scattering and user moments) are converted to held rates per type at the end of its transport (`StoreTypeRates`) and the consumers sum them, so between transports the coupled source terms are applied at the last rates; `UserWorkAfterTransfer` runs only on transport cycles, so per-cycle physics driven by the transport holds a rate there (`mc_hotjupiter` keeps the photoionization rate in `ruser_meshblock_data[1]`, which load balancing does not carry). A type that runs every N cycles is credited to the output interval in which it runs, so output intervals should span the longest cadence.

**Restarts** (dynamic runs): same wiki page. For the code: `MCRandom::PackForRestart` writes a fixed 8 KB record per block after the user block data; the Monte Carlo outputs keep `file_number` in the input and are written before the standard outputs; one rank restarts bitwise, several ranks statistically; restart files from before the generator record cannot be read; static runs are not restarted.

**Photon data layout:** Each photon is stored as a struct-of-arrays via `static int` index members (`ix1p`, `ik1p`, etc.) on the `Photon` class. Position is `(x0p,x1p,x2p,x3p)` (covariant coordinates), wavevector is `(k0p,k1p,k2p,k3p)`, and under the general pusher the coherency tensor is stored Hermitian-packed as sixteen real columns (`polten`, accessed only through `Photon::LoadTensor`/`StoreTensor`/`Tensor`). Status flags use `PhotonStatus` enum (`EVOLVING`, `ESCAPED`, `ABSORBED`, `DESTROYED`, `BUFFERED`).

**Monte Carlo radiation moments** are stored in `NMOM=15` arrays indexed by `MCIER`, `MCIFRx`, `MCIPRxy` enums (energy density, flux, pressure tensor). The flux radiation force `sourceterms(MCRF1..3)` carries a statistical error, `sourceterms_error`, built like the scattering moments' `Jnu*_err`: each photon's contribution to a cell is held back until it leaves the cell and squared once, so correlated steps of one flight count as one sample; the `mcsrc` output writes it as the `RadforceF_err` vector, and `average_moments.py` combines it like any `_err` variable. On hot Jupiter the pull between two seeds has an rms near 1.

## Problem generators

Problem generators live in `src/pgen/`. Monte Carlo problem generators have the naming convention `mc_*.cpp` and guard themselves with `#if !MONTE_CARLO_ENABLED`. The ones most often built:
- `mc_readhdf_gr.cpp` — X-ray binary in Cartesian Kerr-Schild (`gr_user`), initialized from an athdf snapshot; the production GR problem
- `mc_readhdf.cpp` — non-relativistic counterpart, reads an athdf snapshot
- `mc_disksim.cpp` — disk simulation reading from VTK hydro snapshots
- `mc_isoth.cpp`, `mc_isoth_gr.cpp` — isothermal atmosphere, flat and Kerr-Schild (convergence and disk atmosphere tests)
- `mc_isoth_opac.cpp` — the same atmosphere or box with `emission = table`, `absorption = table` (the table-opacity test bed)
- `mc_sphere_isoth.cpp` — uniform isothermal sphere over any Cartesian grid, point source or free-free emission, escape surface at `radius` (the Compton sphere test)
- `mc_snake.cpp`, `mc_snake_atm.cpp` — uniform box and stratified atmosphere in snake coordinates (flat spacetime, sheared)
- `mc_poltest.cpp`, `mc_sphpol.cpp` — single-photon polarized transport tests, `gr_user` and spherical polar
- `mc_disk.cpp` — in progress, untracked

## Visualization / post-processing

Python tools are in `vis/python/montecarlo/`. Add this directory to `PYTHONPATH` to use them.

- `athena_mc/` — the core library, a package whose `__init__` re-exports every public name of its submodules, so `import athena_mc as athenamc` works as it did when this was one file: `lists` (`read_list_generator()` streams photon list files in chunks, `Photons` wraps a single chunk, the geometry tags), `binning` (bins and photon directions), `spectra` (`make_spectrum()`, the spectrum files, combining, family errors), `images` and `plotting`. `plotting` is the only submodule that imports matplotlib and is loaded on first use through a module `__getattr__`, so the binning tools and the forked `--nproc` workers start without it. When a function moves or is added, keep the dependency order lists → binning → spectra → images → plotting (nothing imports upward) and its name in the submodule's `__all__`
- `make_spectrum.py` — CLI wrapper around `athena_mc.make_spectrum()`. Takes any number of list files, sums the ranks of each output into one spectrum (`<base>.<output>.spec`) and with `--combine` averages the outputs weighted by integration time into `<base>.spec`. Same `parse_args()`/`main(args)` layout as `plot_spectrum.py`; replaced the separate `make_spectrum_multi.py`. `--family-cols T E` names the user columns of the emission temperature and emitted energy (0 and 1 for `mc_readhdf*`) and makes the errors family-level: the per-bin weight of each birth photon's copies is summed across chunks and ranks before squaring (`athena_mc.family_key`, `finalize_family_errors`, applied through `mc_cli.bin_outputs`'s `finish` hook), which is the correct estimator for a weight-window run and identical to the plain one when there are no copies
- `make_image.py` — the same for images, via `athena_mc.make_image()`: pixels are the impact parameter of each photon for a distant observer along its direction (no camera radius), binned in cos(inclination) and keV; surface brightness per steradian of direction, without the spectrum's 1/mu. Writes `<base>.<output>.img` or `<base>.img`
- `mc_cli.py` — what `make_spectrum.py` and `make_image.py` share: grouping list files by run and output, naming the outputs, the input checks, loading a `--screen` function, and `--nproc` (a forked process pool over the files of an output; the scripts' `bin_chunk` closures cannot be pickled, so the pool inherits them through the module global `_task`). `band_luminosity.py` has the same option over its own file loop
- `plot_spectrum.py`, `plot_image.py` — plot spectra and images. Both expose `parse_args(argv)`, `make_figure(args)` returning the figure, and `main(args)` which saves it
- `plot_polarization.py` — three-panel plot of a polarized spectrum: intensity over either polarization fraction and angle (`--panels fracangle`) or Q/I and U/I (`--panels qu`), linear by default. `--ffile`/`--fnorm`/`--fcompute` overlay the Feautrier Thomson-atmosphere solution, replacing `plot_feautrier.py` (copies remain in `tst/montecarlo/thomson_polarized_spectrum/` and `problem_specific/`)
- `analyze_spectra.ipynb`, `analyze_images.ipynb` — notebooks that make and plot in one place. Parameter dictionaries mirror the scripts' options; `mc_cli.argv_from()` turns them into a command line for the scripts' own `parse_args()`, so the notebooks add no second interface. `MAKE`/`PLOT`/`SAVE` flags let the plotting cells be re-run on spectra or images already on disk; a screen function can be defined in the notebook and passed to `main(args, screen_function=...)`
- `joinlists.py` — joins the per-rank list files of a run into one list per run or, with `--multiout`, per output; streams the data in chunks, sums `length` and `ntot`, counts each output's `dt` once, and corrects the header if an input was truncated. `make_spectrum.py` does not need joined lists
- `combine_spectra.py` — wrapper around `athena_mc.add_spectra()` for spectrum files that already exist: `-m statistical` sums shares of one interval (ranks), `-m time` averages complete spectra weighted by integration time (outputs, independent runs); prints each input's luminosity as the check on which was meant
- `make_spectrum_single.py` — single-file variant (untracked, in progress)
- `scattering_histogram.py` — distribution of the per-photon scattering count, read from whichever user variable a problem generator copied `nscp` into (`--nscat-col`, default 0; `mc_readhdf*` use 2). Reports how concentrated the scatterings are in the worst photons, which is the number that matters in a resonant-line run where the mean is set by a handful of photons
- `kompaneets_greens.py` — the Kompaneets Green's function solver and quantile tables (`python kompaneets_greens.py [--absorption --nproc 8] --out TABLE.npz`), with `problem_specific/sphere_compton/sphere_theory.py` (escape-path distributions, free-free parameter, the convolution spectrum) and `kgreens_export.py` (npz to the binary the code reads); the method is written up in `doc/monte_carlo/kompaneets/`
- `band_luminosity.py` — band luminosities of a run's escapers output by output, and the ratio of two runs with the error from both runs' output scatter; `--screen NAME` applies a function of a `screen.py` as `make_spectrum.py` does (on the TDE mesh, whose inner radial boundary is an escape boundary, a screen on the exit radius separates the light that reaches the observer from the light that falls in); its total column includes every photon, which is what catches a few samples with absurd energies that the bands never see
- `average_moments.py` — combines the moment athdf files a `nout > 1` run writes into one lower-noise estimate. Values are averaged, `*_err` variables combine in quadrature and are divided by M. It inherits the athdf format by copying the first input and overwriting the data, so there is no writer to keep in step with the output code. `--empirical` writes a second file whose error slots hold the scatter between intervals instead, which assumes nothing about correlations within a photon's path and is the only error estimate available for `mclab`, `mccom`, `mccoord` and `mcsrc`

Photon list files use a custom binary format read by `athena_mc.read_list_generator()`.

## Code style

- Follows the [Athena++ Style Guide](https://github.com/PrincetonUniversity/athena/wiki/Style-Guide)
- `CPPLINT.cfg` enforces Google C++ style via `cpplint`
- Headers use `#ifndef FILENAME_HPP` guards
- Doxygen-style `//!` comments on classes and key functions
- User-customizable physics hooks are set via function pointers on `MonteCarlo` (e.g. `GetEmission`, `UserScattering`, `UserAbsorptionOpacity`). The fluid-derived arrays follow one pattern: `MonteCarloBlock::GetDensity`, `GetTemperature` and `GetNumberDensity` call the user function enrolled for that quantity (`EnrollUserGetDensity` and so on), else the public `DefaultDensity`, `DefaultTemperature` or `DefaultNumberDensity`, which a user function can call and then modify (as `mc_readhdf_gr`'s `ExcludeCells` does); `GetVelocity`, `GetScalars` and `GetBField` have no user function
