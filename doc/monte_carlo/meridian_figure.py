#! /usr/bin/env python

"""
Generate the meridian-plane figure for polarization_conventions.tex.

Kept as a script rather than a checked-in binary so the figure can be regenerated and
edited.  The local TeX installation has no tikz or pgf, hence matplotlib and
\\includegraphics rather than drawing it in the document.

The vectors are the real ones the code builds -- l = normalize(zref - (zref.n) n) and
r = n x l -- and their orthonormality and handedness are asserted at the end, so the
picture cannot drift from the convention it documents.  They are drawn through a fixed
axonometric projection rather than mplot3d: r is normal to the meridian plane, and the
whole point of the figure is that it visibly leaves the plane, which needs the projection
and the label placement chosen deliberately rather than by a 3d autoscaler.

    python meridian_figure.py        # writes meridian_plane.pdf beside this file
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
import matplotlib.patheffects as pe

# Axonometric basis.  x projects right, z up, and y down-left, so the meridian plane --
# spanned here by x and z -- occupies the upper right and a vector along -y unambiguously
# points out of it.
EX = np.array([1.00, 0.16])
EY = np.array([0.52, 0.32])
EZ = np.array([0.00, 1.00])

CLR_Z = "#333333"
CLR_K = "#c03020"
CLR_L = "#1a7a3a"
CLR_R = "#7a3aa8"
CLR_PLANE = "#4878a8"


def project(v):
    """3d vector -> 2d canvas coordinates."""
    return v[0]*EX + v[1]*EY + v[2]*EZ


def meridian(zref, nhat):
    """The pair the code builds: l = normalize(zref - (zref.n) n),  r = n x l."""
    lvec = zref - np.dot(zref, nhat)*nhat
    lhat = lvec/np.linalg.norm(lvec)
    rhat = np.cross(nhat, lhat)
    return lhat, rhat


def arrow(ax, vec, color, lw=2.4, scale=1.0, halo=False):
    tip = project(vec*scale)
    fx = [pe.withStroke(linewidth=lw + 3.2, foreground="white")] if halo else None
    ax.add_patch(FancyArrowPatch((0, 0), tip, color=color, linewidth=lw,
                                 arrowstyle="-|>", mutation_scale=17,
                                 shrinkA=0, shrinkB=0, zorder=5, path_effects=fx))
    return tip


def right_angle(ax, u, v, d=0.11, color="#666666", halo=False):
    """Small square marking that u and v are perpendicular, drawn at the origin."""
    pts = np.array([project(d*u), project(d*u + d*v), project(d*v)])
    fx = [pe.withStroke(linewidth=3.0, foreground="white")] if halo else None
    ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.0, zorder=6,
            path_effects=fx)


def main():
    theta = np.radians(52.0)
    nhat = np.array([np.sin(theta), 0.0, np.cos(theta)])
    zref = np.array([0.0, 0.0, 1.0])
    lhat, rhat = meridian(zref, nhat)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    # the meridian plane, drawn as the quadrilateral through the two in-plane unit
    # directions; kept inside the unit arrows so they read as leaving it
    s = 0.78
    corners = [project(a*nhat + b*lhat) for a, b in ((s, s), (s, -s), (-s, -s), (-s, s))]
    ax.add_patch(Polygon(corners, closed=True, facecolor=CLR_PLANE, alpha=0.15,
                         edgecolor=CLR_PLANE, linewidth=0.9, zorder=1))

    tz = arrow(ax, zref, CLR_Z, lw=1.8)
    tk = arrow(ax, nhat, CLR_K)
    tl = arrow(ax, lhat, CLR_L)
    tr = arrow(ax, rhat, CLR_R, halo=True)

    # Right angles.  k-l lies in the plane and needs no halo; the two involving r are
    # what carry "normal to the plane", since at any projection that shows the plane as a
    # patch r must be drawn across it.
    # Only the k-l angle is marked.  A right-angle square between r and either in-plane
    # vector degenerates into a sliver here, because r projects nearly anti-parallel to k:
    # that r is normal to the plane is carried by its label and by the white halo that puts
    # it in front, which is as much as a single 2d projection can show.
    right_angle(ax, nhat, lhat)

    ax.annotate(r"$\hat{z}_{\rm ref}$", tz, xytext=(4, 6), textcoords="offset points",
                color=CLR_Z, fontsize=14, ha="left", va="bottom")
    ax.annotate(r"$\hat{n}=\hat{k}$", tk, xytext=(7, 1), textcoords="offset points",
                color=CLR_K, fontsize=14, ha="left", va="center")
    ax.annotate(r"$\hat{l}$", tl, xytext=(-7, 4), textcoords="offset points",
                color=CLR_L, fontsize=14, ha="right", va="bottom")
    ax.annotate(r"$\hat{r}$", tr, xytext=(-6, -4), textcoords="offset points",
                color=CLR_R, fontsize=14, ha="right", va="top")

    ax.annotate("meridian plane:\n" + r"spanned by $\hat{z}_{\rm ref}$ and $\hat{k}$",
                project(0.70*nhat - 0.78*lhat), color="#2a5578", fontsize=10.5,
                ha="center", va="center")
    ax.annotate(r"$\hat{l}$ in the plane, $\perp\,\hat{k}$" + "\n" +
                r"$Q>0$: polarization along $\hat{l}$",
                project(-0.42*nhat + 1.16*lhat), color=CLR_L, fontsize=10,
                ha="center", va="center")
    ax.annotate(r"$\hat{r}=\hat{n}\times\hat{l}$, normal to the plane" + "\n" +
                r"$Q<0$: polarization along $\hat{r}$",
                tr + np.array([-0.02, -0.22]), color=CLR_R, fontsize=10,
                ha="center", va="top")

    ax.set_xlim(-1.75, 1.70)
    ax.set_ylim(-1.35, 1.55)
    ax.set_aspect("equal")
    ax.set_axis_off()

    fig.tight_layout(pad=0.2)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meridian_plane.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote " + out)

    # the identities the figure asserts, checked rather than trusted
    assert abs(np.dot(nhat, lhat)) < 1e-12,  "l must be perpendicular to n"
    assert abs(np.dot(nhat, rhat)) < 1e-12,  "r must be perpendicular to n"
    assert abs(np.dot(lhat, rhat)) < 1e-12,  "l and r must be perpendicular"
    assert np.allclose(np.cross(lhat, rhat), nhat), "(l, r, n) must be right handed"
    assert abs(np.dot(rhat, zref)) < 1e-12,  "r must be normal to the meridian plane"
    print("orthonormality and handedness verified")


if __name__ == "__main__":
    main()
