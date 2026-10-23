# Updating the GitHub wiki: plan

**Status (2026-10-22): the pages are written; awaiting review and the first push.** Every
existing page is updated, the nine new pages and the sidebar are in the clone, the six
method write-ups are on the wiki as PDFs under `pdf/`, CLAUDE.md is reduced to rules and
pointers, and `opacity_table_guide.md` is retired in favour of the Table Opacities page.
The LaTeX sources stay in `doc/monte_carlo/` as the sources of the PDFs (pandoc is not
installed here, so they were not converted; the pages were written from them). Work went on
the `wiki_update` branch of the code repository and in the wiki clone at
`/home/swd8g/wiki/athena.wiki`.

## 1. Where things stand

The wiki exists: 21 pages, 820 lines, last touched 2025-09-01, in the Athena++ wiki
style with a sidebar of User Guide, Getting Started and an empty Programmer Guide. It
is a sound user guide for the module as it was a year ago. Since then the repository
has gained load balancing, the sampling schemes (`weights`, the weight window, path
stretching), the random-walk acceleration with the Compton Green's function, table
opacities with the two-band split, dynamic runs with cadence and restarts, the snapshot
readers with `grid_from_file`, the output frame and polarization conventions, images,
and a dozen analysis tools, and several keys the wiki documents no longer exist. The
reference for all of that has been accumulating in CLAUDE.md and in the guides under
`doc/monte_carlo/`.

The wiki clone's `origin` is now `DavisAstroUVA/athena.wiki`, the wiki of the group
repository the code lives in (changed 2026-10-22; the personal fork's wiki is kept as
the remote `swdavis`). The first push to `origin` creates that wiki's content; the
group repository must have its wiki enabled first.

The pages are retained and updated in place, and new pages added, editing the wiki
clone directly as has been the practice. The `wiki_update` branch of the code
repository carries the complementary changes: CLAUDE.md reduced to rules and pointers,
the LaTeX guides converted so that the wiki holds the one maintained copy, and this
plan. Mirroring from the repository by a workflow is not needed while one person edits
the wiki; it can be added later if that changes.

## 2. Page-by-page audit of what exists

| page | state | what to do |
|---|---|---|
| Home | one paragraph | keep; add two sentences on what the module does now and where the records (plans) live |
| Configuring | current | add `-omp`, `-gsl`, `--hdf5_path` note for parallel HDF5; keep |
| Compiling | current | add the header-dependency rule (`make clean` after editing any header, and why) |
| The Input File | stale | `equal_weight` gone (now `weights`); `raytrace` is retired; `minweight` default is 1e-20; add `dynamic`, `cadence`, `cadence_frac`, `general_pusher`, `varystep`, `stepsize`, `max_resident_photons`, `rank_exchange`, the `<opacitytable>` block, `<loadbalancing>`; split into a short format section and a per-feature key table, each linked to its page |
| Problem Generators | mostly current | add the hook list as built (`EnrollUserGetDensity/Temperature/NumberDensity`, `DefaultNumberDensity`, `UserWorkAfterRebalance`, `UserWorkAfterTransfer`), the snapshot readers (`mc_readhdf`, `mc_readhdf_gr`, `grid_from_file`), and a table of the generators that exist |
| Boundary Conditions | current | note `absorb` at an inner radial boundary as the way to drop light that falls in |
| Coordinate Systems | current | add `mc_coord` (`kerr_schild_cartesian`, `snake`) and `gr_user` |
| Running the Code | current | restarts now exist for dynamic runs (`-r`); the wall-time limit; `-t` |
| Outputs | partly stale | `nout` semantics (lists not cumulative, spectra cumulative) fine; add `mcscat`, `mccoord`, `mcsrc` variables with `RadforceF_err`; `nuser`; `file_number` on restart; images are made from lists, not by the code |
| Photon List Analysis | stale | `joinlists.py` usage changed (`--multiout`); `make_spectrum.py` takes any number of list files and sums ranks, so joining is rarely needed; the screen function; `band_luminosity.py`, `family_stats.py`, `scattering_histogram.py` |
| Spectrum Analysis and Plotting | stale | `makespectrum.py` and `athena_mc_spec.py` are now `make_spectrum.py` and `athena_mc.py`; `--combine`, `--yerror`, `--family-cols`; `plot_polarization.py`; `combine_spectra.py`; errors: the output-to-output scatter |
| Trajectory Analysis and Plotting | stub | dropped 2026-10-22 (file and sidebar line removed in the clone; the Outputs page's trajectory section no longer links to it) |
| Monte Carlo Visualization | thin | add `make_image.py`/`plot_image.py`, `average_moments.py` (and `--empirical`), `plot_spherical*.py` as is |
| Using MPI | stale | the "old scheme" section goes; domain decomposition is the only scheme; load balancing exists (link); memory per rank; photon exchange (`rank_exchange`, `max_resident_photons`) |
| General Relativity | thin | add the general pusher, the normal-observer output frame, `kerr_schild_cartesian`, the polar-axis caveat (link to the note) |
| Radiation Transfer | partly stale | `minweight` default; `abs_method = prob` as the production choice where weights die; Russian roulette (`roulette`); the reference-frame paragraph is good |
| Emission | stale | `equal_weight` paragraph becomes the `weights` schemes (emission, equal, biased) with a link to Sampling and Weighting; add `emission = table` with a link |
| Scattering | current | add `dust`; note the Compton energy change and the MRW's Green's function (link) |
| Opacity | stale | `n_e` is now the default mixture or the table's free electrons (`table_ne`); add `absorption = table` (link); the Compton table and `comptonio` fine |
| Transport | stale | the MRW section describes the sphere domain; now the cell box, `accel_*` keys, Compton Green's function; the general pusher's face stop; link the new page |
| Example: Lyman alpha | current but points at `swdavis/athena` and branch `monte_carlo` | update the clone line and branch; check the commands still run |
| sidebar entries without pages | `Pseudo Random Numbers`, `Ray Tracing` | write a short RNG page (streams per block, `iseed`, GSL option, restart state) ; drop Ray Tracing (the branch was retired) |

## 3. New pages and their sources

| new page | source | notes |
|---|---|---|
| Sampling and Weighting | CLAUDE.md "Sample weighting", "Weight window", "Path stretching"; `biased_sampling_guide.tex` | the schemes, when each is right, the effective count, the family-level errors, the XRB and TDE findings in one paragraph each |
| Random Walk Acceleration | CLAUDE.md "Random walk acceleration"; `mrw_guide.tex`; `kompaneets/KGREENS_TABLE_FORMAT.md` | keys, thresholds, the box domain, the Compton table and how to make it, what the report says; the Green's-function method linked as PDF |
| Table Opacities | `opacity_table_guide.md` | nearly verbatim; it was written as this page |
| Load Balancing | CLAUDE.md "Load balancing"; `load_balancing_guide.tex` | keys, what the report prints, setting the parameters |
| Dynamic Runs | CLAUDE.md "Transport cadence", "Restarts" | coupling, cadence, restart files and RNG state, output numbering |
| Frames and Polarization Conventions | `frames_and_moments.tex`, `polarization_conventions.tex` | the output frame, the meridian basis and the sign of Q, angle binning choices |
| Tests | CLAUDE.md "Testing" | the regression suite and every `tst/montecarlo/` test with its configure line and arguments |
| Developer Notes (Programmer Guide) | `photon_communication.tex` (PDF link), `transport_loop_performance.md`, the plans | one page: the module's class map (from CLAUDE.md "architecture"), and links to the records in `doc/monte_carlo/` by repository URL |

Sidebar after the change:

```
User Guide: Configuring, Compiling, The Input File, Problem Generators, Boundary
  Conditions, Coordinate Systems, Running the Code, Using MPI, Load Balancing,
  Dynamic Runs, Outputs (Photon List Analysis, Spectrum Analysis and Plotting,
  Monte Carlo Visualization), General Relativity
Physics: Radiation Transfer (Pseudo Random Numbers, Emission, Scattering, Opacity,
  Table Opacities, Transport, Random Walk Acceleration, Sampling and Weighting,
  Frames and Polarization Conventions)
Getting Started: Example: Lyman alpha, Tests
Programmer Guide: Developer Notes
```

## 4. Conversions

The five LaTeX guides become Markdown pages with `pandoc -f latex -t gfm --wrap=none`
and a pass by hand: equations stay as `$...$` and `$$...$$`, which the wiki renders;
tables come across; the figure of the meridian plane is exported to PNG and committed
in the wiki clone. After conversion the Markdown is the maintained copy and the `.tex`
files are removed from `doc/monte_carlo/` with their PDFs kept, so that a correction is
made once. The Kompaneets method write-up and `photon_communication.tex` stay as LaTeX,
their PDFs committed in the wiki clone and linked.

## 5. CLAUDE.md after the move

Keep: what the fork is, the branch workflow, the build steps and the header rule, the
test list with configure lines (an agent runs them), the naming and comment
conventions, the file map of the module. Replace each reference paragraph (load
balancing, sampling, weight window, acceleration, tables, stretching, cadence,
restarts, the tool list) by one line naming the wiki page and the wiki clone path, so
that an agent with the clone on disk reads the page. The findings that matter for
working on the code (the XRB biased deficit, the TDE inner-boundary share, the
`abs_method = prob` cost) go to the pages they belong to, with a one-line pointer here
where an agent would otherwise repeat a mistake.

## 6. Order of work

1. Done: the remote is the group wiki and the Trajectory page is dropped.
2. **Bring the stale pages up to date** (Section 2), one page per commit in the wiki
   clone, checking each key against `montecarlo.cpp` as it is written. This is the
   part users hit first and it needs nothing new.
3. **Add Table Opacities** (verbatim) and **Tests** (from CLAUDE.md), then the sidebar.
4. **Convert the five guides** and write Sampling and Weighting, Random Walk
   Acceleration, Load Balancing, Frames and Polarization Conventions from them and the
   CLAUDE.md paragraphs.
5. **Write Dynamic Runs, Pseudo Random Numbers, Developer Notes**; refresh Home.
6. **Reduce CLAUDE.md** on `wiki_update`, remove the converted `.tex` sources, move
   `opacity_table_guide.md` out (its content is now the page), update this plan's status,
   and merge.
7. **Push the wiki** and read every page rendered, checking links and equations.

A link check is a one-line script over the clone (`[[...]]` targets without a file);
run it before each push.
