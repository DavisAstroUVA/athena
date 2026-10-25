# Jackknife errors for spectra

Errors of quantities derived from a spectrum (the polarization fraction and angle, Q/I,
U/I, and the intensity itself) by the jackknife over random batches of photons, as an
option beside the per-bin variance formulas. Written and built 2026-10-25 on the branch
`error_est`.

## Status

Built: `make_spectrum.py --nbatch N` stores the per-plane sums of N random batches in
the spectrum (`spectrum['batches']`, an `nbatch=N` header line and a block after the
errors in the file); `athena_mc.batch_key` assigns the batches, `athena_mc.jackknife`
takes the errors, and `plotting._derive` switches every derived quantity to it when a
spectrum carries batches. Spectra without batches are byte-identical to before. The wiki
pages Spectrum Analysis and Plotting and Photon List Analysis were updated 2026-10-25.

## Why

The per-bin errors of the fraction and the angle are first-order propagation assuming
independent errors on I, Q and U. In a Monte Carlo spectrum the three are sums over the
same photons with the same weights, so their fluctuations are correlated and much of the
weight variance cancels in Q/I; the fraction's formula carries a sigma_I term that counts
it in full, the angle's has no I in it. Measured against a photon bootstrap on the TDE
example run (1e5 samples), the fraction's error was 1.7 times the bootstrap's and the
angle's matched it, which is the asymmetry a user notices on a plot. Storing the Q-I and
U-I covariances would fix the linear part only; P = sqrt(Q^2 + U^2) is biased and the
angle is non-Gaussian wherever P/sigma is below about 3, which is most of a TDE
polarization spectrum.

## Design

- **Batches.** Each photon gets an index in [0, N) from a hash of its own numbers (the
  bits of its weight, energy and exit radius, mixed with splitmix64), so the assignment
  is random with respect to rank, position and time, and the same whenever the tool is
  rerun; with `--family-cols` the family key is hashed instead, so a weight window's
  copies of one birth photon share a batch and the batches stay independent. Never by
  rank: a rank's list holds the escapers of one region of the mesh, so rank batches are
  stratified and their scatter includes real differences between regions. Never by
  output interval either, which needs `nout > 1` and has no control over N.
- **Accumulation.** `make_spectrum` takes `batch` and `nbatch`; after the bin indices
  are known it adds one flat bincount per Stokes plane with the batch as the leading
  index, giving `batches` of shape (N, nintens, nphi, nmu, nx), scaled like the
  intensity. Cost: a few percent to a third of a pass, less with `--nproc`; the file
  grows by N copies of the spectrum array (2.5 to 15 kB at N = 16 for 100 bins).
- **Combining.** `add_spectra` sums the batch arrays (and averages them in time with
  `method='time'`), so `--nproc`, `--combine` and `combine_spectra.py` carry them; a
  spectrum with batches combines only with one that has the same number.
- **Errors.** `jackknife(batches, func, period)`: with S the sum over batches, func is
  evaluated on S and on S less each batch; the error is sqrt((n-1)/n sum (f_k - fbar)^2)
  and the bias estimate (n-1)(fbar - f(S)); `period = 180` folds the angle about the
  full value first. The plotting functions stack the batches as extra leading planes so
  that the existing angle and energy selections apply to them unchanged, then call
  `_derive(func, intensity, errors, batches)`, which uses the jackknife with func as the
  point function when batches are present and func's own propagation otherwise. Nothing
  is linearized, so one estimator serves the intensity, the Stokes ratios, the fraction
  and the angle, and the formulas remain for spectra made without `--nbatch`.

## Measurements

TDE example run (1e5 samples, 12 bins 3 to 300 eV, all angles, P/sigma about 2), ratio to
a 400-sample photon bootstrap, median over bins with more than 50 photons: fraction,
formula 1.65 and jackknife 1.48; angle, formula 0.99 and jackknife 1.36; the nonlinear
regime, where the delete-one jackknife is conservative and its own noise at 16 batches is
about 18 percent.

`twoband_991_1e10` (TDE, 1e9 samples per output, nine complete outputs, 20 bins 1 to
1000 eV, all angles), ratio to the scatter between outputs, median over the 20 bins:

| quantity | formula | jackknife |
|---|---|---|
| L_nu (the formula is exact here) | 0.93 | 0.89 |
| polarization fraction | 1.45 | 1.22 |
| polarization angle | 0.72 | 0.82 |

The scatter between nine outputs is itself uncertain by about 25 percent per bin, so the
L_nu row says both estimators are calibrated. The polarization of this run is detected
(P/sigma above 4) only at 30 to 90 eV, where P is 1.6 to 3.5 percent; elsewhere P/sigma
is 1.5 to 3 and the bins above 250 eV, with P of 18 to 59 percent and errors as large,
are the Rice bias of a positive-definite quantity, which the jackknife's bias term
reports and no per-bin formula does. The binning of the 66 GB of lists with
`--nbatch 16 --nproc 32` took 198 s on a loaded machine.

## What it does not do

It cannot create signal: where the polarization is not detected the jackknife angle
error tends to the uniform limit (52 degrees) and the fraction stays biased upward. The
debiased estimator sqrt(P^2 - sigma_Q^2 - sigma_U^2), zero below the noise, is the next
step if that bias matters in practice. `make_image.py` and the in-run `spec` writer do
not carry batches; the image tool can take the same `batch` argument with no change to
the design, and the in-run writer would need the C++ side to hash and hold the sums.
