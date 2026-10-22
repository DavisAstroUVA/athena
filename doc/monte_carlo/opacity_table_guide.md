# Table opacities: a guide to running with them

How to run the Monte Carlo transport on a multigroup opacity table, with or without the
two-band split. The history, the tests and the measurements behind every choice are in
`opacity_table_plan.md`; the mathematics of the split is in the write-up
`two_band_method.tex` (outside the repository). This guide is what a run needs.

## 1. Switching the tables on

```
<montecarlo>
emission   = table
absorption = table

<opacitytable>
opacity_file  = tops_rossabs.txt
nfree_file    = tops_rossabs_nfree.txt
table_ne      = true
```

`emission = table` and `absorption = table` select the module's tabulated emission and
absorption; they sit in `<montecarlo>` beside the other choices (`freefree`, `user`).
Everything about the table itself is read from `<opacitytable>`. The table is read once
per rank into `MonteCarlo::optab`; each block then locates its cells on the (T, rho)
grid when it is built from the fluid, so dynamic runs, restarts and load-balancing
redistributions rebuild the per-cell tables by themselves.

The generators that use the path are `mc_readhdf`, `mc_readhdf_gr` and `mc_isoth_opac`.
They refuse `emission = user` or `absorption = user`, which they no longer provide. A
generator that wants the table needs nothing of its own: it calls `PhotonEmitTable`
from its `InitializePhoton` as it would call `PhotonEmitFreeFree`, and the number
densities come from the module's default (`MonteCarloBlock::DefaultNumberDensity`),
which a generator's hook can call and then modify, as `mc_readhdf_gr` does to exclude
cells above `tcut` and inside the horizon.

## 2. The files

Both files are plain text, whitespace separated, and `tops_regrid.py`
(`vis/python/montecarlo/problem_specific/opacity_table/`) writes them from a raw TOPS
results page onto a grid uniform in log T and log rho:

```
python tops_regrid.py regrid TOPS_n32_TDEephot_0.0001kev_100kev.txt -o tops_rossabs.txt \
    --rho-range 1e-18 1e-6 --n-rho 121 --T-range 5.8e3 1e9 --n-T 120 --rossabs --digits 5
```

**The opacity table.** `nfre ntem nrho`; the group lower bounds in keV (`nfre` of
them, uniform in log, the top group's upper edge continuing the spacing); the
temperatures in keV; the densities in g/cm^3; the grey Rosseland mean (`ntem` rows of
`nrho`); the grey Planck mean; the Rosseland group means (`nfre` blocks), which include
the scattering of the table maker's model; the Planck group means (absorption only);
and, with `--rossabs`, a fifth block of Rosseland absorption, the Rosseland group means
without that scattering. All opacities are per gram. The temperature and density axes
need not be uniform (the old 69-row tables read as well); the energy axis must be.

**The free-electron file** (`regrid` writes it beside the table as `_nfree.txt` unless
told `--no-nfree`):
comment lines, one of which is `# Abar = <A>  Z_full = <Z>`, then `nT nrho`, the same
two grids, and `nT` rows of `nrho` values of N_free, the free electrons per atom. It
is required by `table_ne` and by `two_band`.

Tables meant for two-band runs should carry the Rosseland absorption block: without it
the code forms `kappa_R - kappa_s` from five-digit means, which is ill-conditioned
where the two are close, and on the solar TOPS table that changes whether a group is
one band or two in 11% of the cells.

## 3. The keys

| key (`<opacitytable>`) | default | meaning |
|---|---|---|
| `opacity_file` | required | the table |
| `nfree_file` | none | free electrons per atom; `none` spells no file, so a deck can override it from the command line |
| `table_groups` | `mean` | how the group values are read, Section 4 |
| `table_above` | `freefree` under `mean`, `clamp` under `edge` | above the top group: analytic free-free, the top value, or nothing |
| `table_emax` | 0 | keV; with `table_above = freefree`, groups of the same spacing up to this energy carry free-free emission for gas hotter than the table |
| `table_offgrid` | `freefree` | cells off the (T, rho) grid: free-free, or clamp to the grid's edge |
| `table_gray` | `keep` | `freefree` replaces rows that are gray in every group (the old tables' clamped cells) by free-free |
| `table_fill_freefree` | false | the whole table replaced by free-free, for testing |
| `table_storage` | `cell` | `cell`: extinction per group and the cumulative emission stored per cell; `global`: recomputed from the rank's table at every call |
| `table_ne` | false | the electron density from N_free: rho N_free / (Abar m_u) |
| `two_band` | false | the split of Section 5; needs `table_groups = mean` and `table_ne` |
| `two_band_w` | 0.01 | the high band's frequency fraction (the closure) |
| `two_band_floor` | 0.003 | the Rosseland absorption is floored at this times the scattering |
| `two_band_closure` | `w` | `w` fixes the fraction; `gap0` takes no absorption between the lines |
| `two_band_thick` | 0 (off) | cell depth in the high band above which it is thermalized, Section 5 |
| `two_band_decorrelate` | 0 | probability of redrawing a photon's band at a cell crossing |

The run report prints the cells served by the table, those off its grid, the masked
ones and, with two bands, the split's parameters and the share of absorbed samples that
died in their first step.

## 4. The two readings of a group value

The k-th Planck value of a TOPS table is the mean over the group [E_k, E_k+1).
`table_groups = mean` uses it as the opacity throughout that group: emission in the
group is kappa_k times the photon Planck integral over the group, and a photon's energy
within the group follows B_nu / (h nu). This is the right reading and the default.

`table_groups = edge` is the reading the generators had before the module: the k-th
value taken as the opacity at the k-th lower bound and interpolated in log E to the
next, with the emission the trapezoid in ln nu between the bounds. It never emits in
the top group, and it moves every strong group half a group down, because the
trapezoid spreads the value of a strong group over the interval below it. On the TDE
snapshot that is 17 times the mean reading's emission in the 1 to 10 eV decade and a
third more absorption overall (twice the scatterings per sample). It is kept so that
older runs can be reproduced, with `table_gray = freefree` as they had it;
`snapshot_emission.py` shows the difference between the readings for a snapshot
without transport.

## 5. The two-band split

A line-dominated group has a Planck mean set by its line cores and a Rosseland mean set
by the gaps between them, ten to ten thousand times smaller below a few 1e5 K. A single
opacity per group cannot do both: the Planck mean emits correctly and traps radiation
far too strongly in thick gas. `two_band = true` represents each group of each cell by
two bands obeying Kirchhoff's law separately, a high band over a fraction w of the
frequencies with absorption kappa_h and a low band over the rest with kappa_l, chosen
so that the two reproduce both means with the cell's own scattering kappa_s (Thomson,
or thermal Compton at the group centre, on the electron density from N_free). The
means fix two of the three numbers; `two_band_w` is the third. The result converges as
w falls: on the TDE 0.01 and 0.001 agree within the sampling scatter and 0.1 is a
tenth lower, so the default is in the converged regime.

A photon carries a rank in [0, 1) (`Photon::ubp`): below the cell's w it is in the high
band, else the low one, so a band means the same frequencies in every cell. The rank
is redrawn after every scattering (the electron's Doppler shift is many line spacings),
whenever the photon's group changes, and at cell crossings with probability
`two_band_decorrelate`.

**Set `two_band_thick`.** Where the high band is thick across a cell, nearly all of a
group's emission is born in it and absorbed where it was born, and no sample ever
carries energy out: on the TOPS table the plain split loses every sample in its first
step. Above `two_band_thick` (cell depth rho kappa_h W, with W the cube root of the
cell volume) the high band is taken to be in equilibrium with the gas and is not
transported: its emission is dropped, the low band receives what the high band would
have passed to it by scattering, and a photon scattered into the high band there is
absorbed on the spot. Detailed balance holds and the tests reproduce both the Planck
field in a closed box and the Rosseland flux through a slab. Use 10; the TDE is
insensitive between 3 and 30.

**Use `abs_method = prob`.** Weight absorption follows a sample through hundreds of
scatterings after its weight has gone, and with two bands that is most of the cost.
On the TDE the analog method gave the same bands 27 times faster.

**What the correlated-k assumption carries.** Whether a gap in one cell is a gap in the
next depends on the ionization changing from cell to cell and on the velocity shear
shifting the lines. `two_band_decorrelate = 1` is the opposite limit, each cell's lines
unrelated to its neighbours'. On the TDE it took a quarter off the escaping light at
every energy from 10 to 300 eV; the factor two to three over the Planck mean at 30 to
300 eV is escape within single thick cells and survived it, and the 10 to 30 eV light,
which crosses many cells, returned to the Planck-mean value. Run both and quote the
pair.

**What the split does not do.** Inside a cell a photon in a gap is swept across lines
by the velocity gradient, the expansion-opacity regime, and the group means cannot
describe it. This is not going to be modelled: the two-band table is an approximation
to a treatment that would need the line list, and its results are to be read with
that understood. The split is also only ever applied to the table's own groups of cells
on the grid, never to the free-free extension groups or to off-grid cells.

## 6. Sampling and errors on a large snapshot

`weights = emission` draws cells uniformly and gives each sample its cell's emission as
weight, so the weights are heavy-tailed; `weights = equal` draws cells by emission and
on a disk puts nearly every sample where none escapes. Run with `nout` of 4 or more
and take the scatter between outputs as the error: on the TDE the quoted per-bin
errors agreed with it from 10 eV to 3 keV, and above that the bands were a handful of
samples. On a mesh whose inner radial boundary is an escape boundary, the photons that
leave through it are in the lists with the rest: `band_luminosity.py --screen` and
`make_spectrum.py --screen` with a screen on the exit radius separate the light that
reaches the observer from the light that falls in. On the TDE the inner photons were
half the escaping luminosity and everything above 0.3 keV.

## 7. Storage and memory

`table_storage = cell` stores the extinction per group and the cumulative emission
spectrum of every cell (five arrays per cell with two bands); `global` stores only the
cell's place on the grid and recomputes both from the rank's table at every call. The
two give byte-identical photon lists for the same seed. On the TDE at 48 ranks global
storage saved 70 MB of 620 per rank at the same speed; it earns its place on runs with
far more cells per rank.

## 8. Tests and tools

In `tst/montecarlo/opacity_table/`: `port_equivalence.py` (the module against a binary
from before it, byte-identical for `mc_readhdf`), `storage_equivalence.py` (cell against
global), `repack_equivalence.py` (across a load-balancing repack), `ff_table.py` and
`ff_atmosphere.py` (the table path against free-free on a free-free table),
`ne_table.py` (the electron density from N_free), `twoband_box.py` (the split's
emission and equilibrium) and `rosseland_slab.py` (its diffusive flux on a picket-fence
table). In `vis/python/montecarlo/problem_specific/opacity_table/`: `tops_regrid.py`,
`make_ff_table.py`, `make_picket_table.py` and `snapshot_emission.py`. CLAUDE.md has
the configure lines and the arguments of each.
