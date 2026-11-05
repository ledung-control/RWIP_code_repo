# -*- coding: utf-8 -*-
"""Fig. 5 - Nominal response of the three analytical configurations.

Regenerated as true vector art directly from the simulation result file that
produced the published figure and Table 4,

    Compare_v13_results.mat  ->  R.PSOFLC / R.SMCESO / R.HYBRID

whose stored metrics reproduce Table 4 exactly (IAE_theta 0.024463 / 0.13692 /
0.026624, ts_theta 2.777 / 5.707 / 3.621, ...).  Curves, axes, ranges, legend
and panel captions are unchanged; only the rendering is new.

Usage (from this folder):  python draw_fig5.py [--manuscript]
--manuscript: 11 pt text and no panel titles (style of the manuscript; the exact
edited copy used for the submitted figure was not archived, the data are identical).

Authored at the final published size: 17.0 cm (full text width), text 10 pt.
"""
import sys
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MANUSCRIPT = "--manuscript" in sys.argv   # manuscript style: 11 pt text, no panel titles
W_CM = 17.0
FS = 11.0 if MANUSCRIPT else 10.0
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
    "axes.linewidth": 0.8, "grid.linewidth": 0.6,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
})

BLUE, RED, GREEN = "#054D96", "#D90202", "#329221"
STY = [("PSO-FLC", BLUE, "-", 1.9),
       ("SMC-ESO", RED, (0, (5, 2.4)), 1.6),
       ("PSO-FLC + SMC-ESO", GREEN, (0, (7, 2, 1.4, 2)), 1.6)]

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
MAT = os.path.join(_HERE, "..", "..", "results", "archived_tables4-7", "Compare_v13_results.mat")
d = sio.loadmat(MAT, squeeze_me=True,
                struct_as_record=False)
R = d["R"]
runs = [R.PSOFLC, R.SMCESO, R.HYBRID]
TMAX = 20.0

# y-tick sets reproduced from the published figure
PANELS = [("theta",     r"$\theta$ (rad)",         r"(a) Pendulum angle $\theta(t)$",
           [-0.050, -0.025, 0.000, 0.025, 0.050, 0.075, 0.100], "%.3f"),
          ("theta_dot", r"$\dot{\theta}$ (rad/s)", r"(b) Angular velocity $\dot{\theta}(t)$",
           [-1.00, -0.75, -0.50, -0.25, 0.00, 0.25], "%.2f"),
          ("phi",       r"$\varphi$ (rad)",        r"(c) Flywheel angle $\phi(t)$",
           [-0.2, 0.0, 0.2, 0.4, 0.6], "%.1f"),
          ("u",         r"$u$ (V)",                r"(d) Control voltage $u(t)$",
           [-2.5, 0.0, 2.5, 5.0, 7.5, 10.0], "%.1f")]

w_in = W_CM / 2.54
fig, axes = plt.subplots(2, 2, figsize=(w_in, w_in / (1.549 if MANUSCRIPT else 1.179)))

for k, (fld, ylab, cap, yt, yfmt) in enumerate(PANELS):
    ax = axes[k // 2][k % 2]
    for (nm, col, ls, lw), run in zip(STY, runs):
        t = np.asarray(run.t, float)
        y = np.asarray(getattr(run, fld), float)
        m = t <= TMAX
        ax.plot(t[m], y[m], color=col, linestyle=ls, linewidth=lw, label=nm,
                solid_joinstyle="round", dash_capstyle="butt")
    ax.set_xlim(0, TMAX)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("t (s)")
    ax.set_ylabel(ylab)
    ax.set_yticks(yt)
    ax.set_yticklabels([yfmt % v for v in yt])
    ax.grid(True, color="#CCCCCC", linewidth=0.6)
    ax.set_axisbelow(True)
    if not MANUSCRIPT:
        ax.set_title(cap, y=-0.34, fontsize=FS)
    if k == 0:
        ax.legend(loc="upper right", frameon=True, framealpha=1.0,
                  edgecolor="#AAAAAA", borderpad=0.45, handlelength=2.4)

if MANUSCRIPT:
    fig.subplots_adjust(left=0.10, right=0.985, top=0.985, bottom=0.115, wspace=0.32, hspace=0.50)
else:
    fig.subplots_adjust(left=0.095, right=0.985, top=0.985, bottom=0.115,
                        wspace=0.30, hspace=0.62)
OUT = "Fig5_manuscript_style" if MANUSCRIPT else "Fig5_nominal_17cm"
fig.savefig(OUT + ".pdf", facecolor="white")
fig.savefig(OUT + ".svg", facecolor="white")
fig.savefig(OUT + "_600dpi.png", dpi=600, facecolor="white")
plt.close(fig)

print("Fig. 5 regenerated at %.1f cm wide, text %.0f pt" % (W_CM, FS))
for nm, run in zip(("PSO-FLC", "SMC-ESO", "HYBRID"), runs):
    print("   %-8s IAEtheta=%.6f  ts_theta=%.3f  IAEphi=%.4f"
          % (nm, run.IAEtheta, run.ts_theta, run.IAEphi))
