# Commands and analysis section for the student guide

Text to paste into "Compiling and Running the Monte Carlo Code on Pella". Part A gives
every command of the existing sections as a block to paste in place of the `>` lines;
Part B replaces the "Python Analysis Tools for Lists and Spectra" section in full.

Formatting in Google Docs: put each block in a code block (Insert > Building blocks >
Code block, choose "Unformatted" or "Plain text"), or select the lines and set the font
to Courier New with a light grey paragraph shading. One command per line, with no prompt
character, so that a triple-click selects exactly one command and a drag selects the
block. Where a command spans two lines it ends in a backslash, which the shell accepts
as a continuation when pasted whole.

---

## Part A: commands of the existing sections

### Compiling and running on Pella

Downloading the code:

```
git clone https://github.com/DavisAstroUVA/athena.git
```

Configuring for the mc_readhdf problem generator:

```
python configure.py --prob=mc_readhdf --coord=spherical_polar -mc -mpi -hdf5 --hdf5_path=/usr/lib/x86_64-linux-gnu/hdf5/openmpi
```

Compiling (make clean is needed after any .hpp file changes; the 8 is the number of
compile processes):

```
make clean
make -j 8
```

Running with 32 MPI processes, output and errors to a file named out, in the background
(replace the path to the binary with your own):

```
mpirun -np 32 /home/swd8g/athena-swdavis/bin/athena -i athinput.mctde >& out &
```

Following the run:

```
top
tail -f out
```

### Compiling and running on Rivanna/Afton

Logging in (replace swd8g with your user name):

```
ssh -X swd8g@login.hpc.virginia.edu
```

Copying a code directory from Pella, either way:

```
rsync -av swd8g@pella-gb1.astro.virginia.edu:/home/swd8g/athena .
scp -r swd8g@pella-gb1.astro.virginia.edu:/home/swd8g/athena .
```

Configuring:

```
module load python
python configure.py --prob=mc_readhdf --coord=spherical_polar -mc -mpi -hdf5
```

Loading the compiler modules and compiling (no -j on the login nodes):

```
module purge
module load gcc
module load openmpi
module load hdf5/1.14.6
make clean
make
```

Submitting, watching and cancelling a job:

```
sbatch slurm.athena
squeue -u swd8g
scancel 2848426
```

An example slurm.athena:

```
#!/bin/bash
#SBATCH --ntasks-per-node=96
#SBATCH --partition=standard
#SBATCH --time=24:00:00
#SBATCH -J user_boosts
#SBATCH -A swdavis

module load gcc
module load openmpi
module load hdf5/1.14.6

mpirun /home/swd8g/athena-swdavis/bin/athena -i athinput.mctde
```

Copying results back to Pella (no trailing slash on test_01, so that the directory itself
is created on Pella):

```
cd /scratch/swd8g/tde
rsync -av test_01 swd8g@pella-gb1.astro.virginia.edu:/PellaShared/swd8g/mc_tde
```

Two paragraphs at the end of the Rivanna section are left over from an older Camber
version of the guide and no longer apply: the one beginning "But we will be mostly just
using cpu-queue for now" and the "Analysis of results" paragraph about the hpc-outputs
directory and jupyter hub. They can be deleted.

---

## Part B: Python analysis tools for lists, spectra and images

All of the tools live in the `vis/python/montecarlo` directory of the athena repository
and are kept up to date there; the copies in `/PellaShared/swd8g/monte_carlo` are older
and will drift. Everything is built on the `athena_mc.py` module in that directory. To use
the scripts from any directory, add it to your Python path once per session (replace the
path with your own copy of the code):

```
export PYTHONPATH=/home/swd8g/athena-swdavis/vis/python/montecarlo:$PYTHONPATH
```

Every script prints its full documentation with `--help`, including worked examples; what
follows is the usual workflow.

There are two ways to do the analysis. The scripts take their options from the command
line. The notebooks `analyze_spectra.ipynb` and `analyze_images.ipynb`, in the same
directory, do the same work from a set of parameter dictionaries and call the scripts'
own functions, so the two cannot disagree and the script help documents every notebook
entry. The notebooks are the convenient route for a large output, where reading the
lists takes a while: with `MAKE = True` the lists are read once and the spectra or
images are written to disk, and with `MAKE = False` and `PLOT = True` the plotting cells
can be re-run on the files already made without touching the lists again.

### 1. The photon lists

The primary output of a run is a photon list, one file per MPI process and per output
interval, named like `prob.out1.proc3.00000.list`: `proc3` is the process and `00000` the
output number. There is one output unless `nout` is set in the `<montecarlo>` block of
the input file, in which case the numbers run from 00000 to nout-1. A run on 32 processes
with nout = 10 writes 320 files.

You do not need to join the lists. `make_spectrum.py` and `make_image.py` take any number
of list files, sum the processes of each output and treat the outputs separately or
average them. Join lists only when a single file is simpler to keep or hand on:

```
joinlists.py prob.out1 32 0 0
joinlists.py prob.out1 32 0 9 --multiout
```

The arguments are the base name up to `.proc`, the number of processes, and the first
and last output numbers. The first command joins the 32 files of the single output into
`prob.out1.list`; the second joins outputs 0 to 9 into one list per output,
`prob.out1.00000.list` to `prob.out1.00009.list`. Without `--multiout` all ten outputs go
into one list whose integration time is the total of the ten. `--skip` passes over
processes that wrote no list (none of their photons escaped) and `--rm` deletes the inputs
once the join is written.

### 2. Making a spectrum

`make_spectrum.py` takes the number of bins, the lower and upper edge, and the list files:

```
make_spectrum.py 48 1 1000 prob.out1.proc*.00000.list --yerror
```

This bins the photons of output 0 into 48 logarithmic bins from 1 to 1000 in the x unit,
electron volts by default, and writes `prob.out1.00000.spec`. `--xunit kev`, `nu` or
`lambda` changes the unit of the binning and of the limits. `--yerror` computes the
statistical error of each bin, which the plotting needs for error bars; use it always.
`--nmu 8` gives eight bins in mu, the cosine of the polar angle, and `--nphi 4` four bins
in azimuth, over 0 to 1 and 0 to 2 pi unless `--mumin`, `--mumax`, `--phimin` or
`--phimax` say otherwise.

For a run with several outputs give all the files. One spectrum per output is written,
`prob.out1.00000.spec`, `prob.out1.00001.spec` and so on, and with `--combine` the outputs
are averaged, weighted by their integration times, into a single `prob.out1.spec`:

```
make_spectrum.py 48 1 1000 prob.out1.proc*.list --yerror
make_spectrum.py 48 1 1000 prob.out1.proc*.list --yerror --combine
```

Running with `nout = 4` or so and looking at the per-output spectra is the recommended
way to judge the real statistical error of a run. The quoted error of a bin is computed
from the photons that happened to be drawn, and when a few heavy photons dominate a bin,
as they can in a biased run, it comes out too small; the scatter between outputs does not
have this problem.

If the run used the weight window (`wwin_top` in the input file), add
`--family-cols 0 1`. The window splits photons into copies that share their history, and
the plain error counts the copies as independent; this option sums each birth photon's
copies before squaring, using the emission temperature and energy columns that the
`mc_readhdf` problem generators write as user variables 0 and 1. It changes nothing for
a run without copies.

```
make_spectrum.py 48 1 1000 prob.out1.proc*.list --yerror --family-cols 0 1 --combine
```

Photons can be left out with a screen function: a file `screen.py` in the working
directory with a function that takes a chunk of photons and returns True for those to
drop. An example is in `/PellaShared/swd8g/monte_carlo/screen.py`, whose `inner_radius`
drops photons with x1 below 1e13 cm, the ones that left a TDE run through the inner
radial boundary:

```
make_spectrum.py 48 1 1000 prob.out1.proc*.list --yerror --screen inner_radius
```

`--calclum` prints the luminosity of each output from its lists, a useful check that the
files are the ones you meant. `--outfile` overrides the output name.

In `analyze_spectra.ipynb` the same choices are the `LISTS` glob, the `NX, XMIN, XMAX`
line, the `make_params` dictionary (its keys are the script's options with the dashes
removed) and the optional `SCREEN` function defined in the notebook itself.

### 3. Plotting a spectrum

```
plot_spectrum.py prob.out1.spec --ploterr
```

plots nu L_nu against photon energy in keV, converting from the unit the spectrum was
binned in, and saves `prob.out1.png`; the luminosity of each spectrum is printed.
`--ploterr` draws the error bars. `--xunit`, `--yunit` (nulnu, lnu, counts, or for
polarized spectra polfrac, polangle, q, u, v), `--xscale`, `--yscale`, `--xmin`,
`--xmax`, `--ymin`, `--ymax` and `--rebinx N` adjust what is drawn.

For spectra binned in angle, `--imu` and `--iphi` choose the bins: an index, `sum` to
integrate over the angle, or for phi also `ave` to average over azimuth. The defaults,
`--imu sum --iphi sum`, give the angle-integrated spectrum. Several values are given with
spaces and draw one curve each, with `--mulegend` putting the mu of each in the legend:

```
plot_spectrum.py prob.out1.spec --ploterr --imu 0 4 7 --iphi ave --mulegend
```

Several spectra on one set of axes are given as separate arguments, with one label each:

```
plot_spectrum.py run1/prob.out1.spec run2/prob.out1.spec --ploterr --labels "1e8 photons" "1e9 photons"
```

`--bbtemp T` overlays a blackbody of temperature T in kelvin, scaled by `--bbnorm`, and
`--txtfile` also writes each curve to a text file next to the spectrum.
`plot_polarization.py` makes a three-panel plot of a polarized spectrum, intensity over
polarization fraction and angle or over Q/I and U/I.

In the notebook, the `PLOT_FILES` list and the `plot_params` dictionary hold the same
choices; `SAVE` writes the figure to the file named in `outfile`.

### 4. Combining spectra that already exist

`make_spectrum.py` sums processes and averages outputs itself, so this is for spectrum
files made separately, for example two independent runs of the same problem:

```
combine_spectra.py -m time -o prob.both.spec run1/prob.out1.spec run2/prob.out1.spec
```

`-m time` averages complete spectra weighted by their integration times; `-m statistical`,
the default, sums spectra that each hold part of the photons of one interval. The
luminosity of each input and of the result are printed as the check that the right one
was used.

### 5. Making and plotting images

Images complement the spectra. Each escaping photon is placed in the image plane of a
distant observer looking back along the photon's direction, at its impact parameter, so
there is no camera distance to choose. The pixel values are surface brightness, and the
photons are binned by the cosine of the observer's inclination. `make_image.py` takes
the half-width and half-height of the image in the units of the photon positions, code
units for most runs, and the list files:

```
make_image.py 20. 20. prob.out1.proc*.list --nx 64 --ny 64 --combine
```

This makes a 64 by 64 pixel image from -20 to 20 in both directions, averaged over all
outputs, into `prob.out1.img`, with the default 16 inclination bins from cos i = -1 to 1.
`--ninc`, `--imin` and `--imax` change the inclination bins, and `--nen`, `--emin` and
`--emax` add photon-energy bands in keV, one band over all energies by default. The
images follow the same output grouping and `--screen` option as the spectra. The physical
size worth imaging is a fraction of the simulation domain, read from `x1max` and the
like in the input file; if those are in code units and you want centimetres on the axes,
`--unit` only labels the axes, the limits stay in the list's units.

```
plot_image.py prob.out1.img --iinc 15 --logc
```

plots one plane: `--iinc` picks the inclination bin, with cos i increasing with the index
(15 is the most face-on of 16 bins over -1 to 1; 7 the most edge-on), `--ie` the energy
band, `--logc` a logarithmic colour scale, `--vmin` and `--vmax` its limits, and `--type`
the quantity (intensity by default, or q, u, v, polfrac and polangle for polarized runs,
with `-p` drawing polarization bars every `--step` pixels). `analyze_images.ipynb` does
both steps from its `make_params` and `plot_params` dictionaries.

### 6. The hdf5 moment outputs

If requested in the input file, the code writes the radiation moments averaged over each
cell, the energy density, flux and pressure tensor in the frame the output block asks
for, plus any user moments, as .athdf files. They are read with `athena_read.py` in
`vis/python`, which returns a dictionary of numpy arrays by variable name, and they can
be plotted with `plot_sim.py` from an input file of plot blocks, as before:

```
python plot_sim.py -pfile plot.in
```

Example plot files and the `athena_functions.py` that defines derived quantities such as
`trad_mc` are in `/PellaShared/swd8g/monte_carlo`.

A run with `nout > 1` writes one moment file per output. `average_moments.py` combines
them into one lower-noise estimate, and with `--empirical` writes a second file whose
error slots hold the scatter between the outputs instead of the propagated error, which
is the honest error for the moments that carry none of their own:

```
average_moments.py -o prob.out4.mean.athdf --empirical prob.out4.emp.athdf prob.out4.0000*.athdf
```

### 7. Two diagnostics of a run

`scattering_histogram.py` prints the distribution of the number of scatterings per
escaping photon and how concentrated the work is in the worst photons; the `mc_readhdf`
generators keep the count in user variable 2:

```
scattering_histogram.py prob.out1.proc*.list --nscat-col 2
```

`family_stats.py` is for weight-window runs: it groups the escaping photons by birth
photon and prints the effective number of samples at the photon and at the family level,
which is the number that says how much a window run is worth:

```
family_stats.py prob.out1.proc*.00000.list --emin 1000 --emax 3000
```
