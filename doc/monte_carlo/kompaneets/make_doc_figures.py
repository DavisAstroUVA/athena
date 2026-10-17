
"""
Figures for kompaneets_greens_method.tex.  Run from this directory:
    python make_doc_figures.py [mc]
The solver is vis/python/montecarlo/kompaneets_greens.py and the direct Monte Carlo
(for the mc figure) tst/montecarlo/sphere_compton/mc_sphere_check.py.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      '..', '..', '..'))
sys.path.insert(0, os.path.join(_ROOT, 'vis', 'python', 'montecarlo'))
sys.path.insert(0, os.path.join(_ROOT, 'tst', 'montecarlo', 'sphere_compton'))
import kompaneets_greens as kg  # noqa: E402

# Becker (2003) eq. 33 at x0 = 1, from kg.becker_greens (mpmath, 25 digits),
# as P = x^3 f per unit ln x
BECKER_X0_1 = {0.1: [(0.3, 4.93784e-03), (1., 8.50102e-01), (3., 8.75827e-02),
                     (10., 5.51240e-07)],
               0.5: [(0.3, 2.60689e-02), (1., 3.29804e-01), (3., 5.34853e-01),
                     (10., 8.38552e-03)],
               2.0: [(0.3, 1.10912e-02), (1., 1.88257e-01), (3., 6.67264e-01),
                     (10., 2.22677e-02)]}


def fig_greens():
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    ys = [0.01, 0.1, 0.5, 2., 15.]
    ax = axs[0]
    res = kg.greens(1., ys)
    for k, (y, (s, p)) in enumerate(zip(ys, res)):
        ax.plot(np.exp(s), p, color="C{:d}".format(k),
                label="$y={:g}$".format(y))
        for xb, pb in BECKER_X0_1.get(y, []):
            ax.plot(xb, pb, "o", mfc="none", color="C{:d}".format(k))
    xw = np.logspace(-3, np.log10(40.), 300)
    ax.plot(xw, 0.5*xw**3*np.exp(-xw), "k:", label="Wien")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1.e-5, 5.)
    ax.set_xlim(1.e-2, 40.)
    ax.set_xlabel("$x_f$")
    ax.set_ylabel(r"$P = x_f^3 n$ (per unit $\ln x_f$)")
    ax.set_title(r"$x_i = 1$, no absorption (circles: Becker 2003)",
                 fontsize=9)
    ax.legend(fontsize=7)

    ax = axs[1]
    ff = kg.freefree_shape
    ys = [0.03, 0.3, 1., 3.]
    r0 = kg.greens(0.03, ys)
    r1 = kg.greens(0.03, ys, sink=ff, lam=2.e-3)
    for k, (y, (s0, p0), (s1, p1)) in enumerate(zip(ys, r0, r1)):
        c = "C{:d}".format(k)
        ax.plot(np.exp(s0), p0, color=c, lw=0.8, ls="--")
        ax.plot(np.exp(s1), p1, color=c,
                label="$y={:g}$, $S={:.2f}$".format(y, np.trapz(p1, s1)))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1.e-5, 5.)
    ax.set_xlim(1.e-3, 40.)
    ax.set_xlabel("$x_f$")
    ax.set_title(r"$x_i = 0.03$: $\Lambda = 2\times10^{-3}$ (solid), "
                 r"$\Lambda = 0$ (dashed)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("fig_greens.pdf")


def fig_quantiles():
    t = np.load("../kgreens_table_ff.npz")
    y = t["y"]
    lev = t["levels"]
    xi = t["xi"]
    j = np.argmin(np.abs(xi-1.))
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    want = [1.e-4, 0.01, 0.16, 0.5, 0.84, 0.99, 1.-1.e-4]
    ks = [int(np.argmin(np.abs(lev-p))) for p in want]
    for ax, il, ttl in [(axs[0], 0, r"$\Lambda = 0$"),
                        (axs[1], int(np.argmin(np.abs(t["lam"]-1.))),
                         r"$\Lambda = 1$")]:
        labels = ["10^{-4}", "0.01", "0.16", "0.5", "0.84", "0.99",
                  "1-10^{-4}"]
        for k, lab in zip(ks, labels):
            ax.plot(y, t["uq"][il, :, j, k], label="$p={}$".format(lab))
        ax.set_xscale("log")
        ax.set_xlabel("$y$")
        ax.set_title(r"$x_i = {:.3g}$, {}".format(xi[j], ttl), fontsize=9)
    axs[0].set_ylabel(r"$u_p = \ln(x_f/x_i)/\sqrt{2y}$")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("fig_quantiles.pdf")


if __name__ == "__main__":
    fig_greens()
    fig_quantiles()
    if len(sys.argv) > 1 and sys.argv[1] == "mc":
        import mc_sphere_check as mc
        mc.check(0.03, 15., 2.e-3, 2.e-3, 200000, plot="fig_mc.pdf")
