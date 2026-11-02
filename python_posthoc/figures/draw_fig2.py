# -*- coding: utf-8 -*-
"""Fig. 2 - Equilibrium-preserving PSO-FLC structure.

Regenerated as true vector art directly from the manuscript's own official
fuzzy inference system, fis_rwip_v13_best.fis, whose breakpoints reproduce the
tuned values of Table 2 exactly (s_e = 0.20969, s_de = 0.78364, s_u = 0.290554)
and whose rule block reproduces Eq. (7), u(i,j) = clamp(i + j - 3, 1, 5).

Panel (c) is evaluated with the inference operators declared in the same file:
AND = min, implication = min, aggregation = max, defuzzification = centroid.

Authored at the final published size: 17.0 cm (full text width), text 10 pt.
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

W_CM, FS = 17.0, 10.0
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
    "axes.linewidth": 0.8, "grid.linewidth": 0.6,
})

# ---------------------------------------------------------------- read FIS
TXT = open("fis_rwip_v13_best.fis", encoding="utf-8", errors="ignore").read()


def mfs(section):
    blk = re.search(r"\[%s\](.*?)(?=\n\[|\Z)" % section, TXT, re.S).group(1)
    out = []
    for m in re.finditer(r"MF\d+='(\w+)':'(\w+)',\[([^\]]+)\]", blk):
        out.append((m.group(1), m.group(2),
                    [float(v) for v in m.group(3).split()]))
    return out


def mu(kind, p, x):
    x = np.asarray(x, float)
    if kind == "trimf":
        a, b, c = p
        left = np.where(b > a, (x - a) / np.where(b > a, b - a, 1), 0.0)
        right = np.where(c > b, (c - x) / np.where(c > b, c - b, 1), 0.0)
        return np.clip(np.minimum(left, right), 0, 1)
    a, b, c, d = p                                        # trapmf
    left = np.where(b > a, (x - a) / np.where(b > a, b - a, 1), 1.0)
    right = np.where(d > c, (d - x) / np.where(d > c, d - c, 1), 1.0)
    return np.clip(np.minimum(np.minimum(left, right), 1.0), 0, 1)


IN1, IN2, OUT = mfs("Input1"), mfs("Input2"), mfs("Output1")
RULES = [(int(a), int(b), int(c)) for a, b, c in
         re.findall(r"^(\d+) (\d+), (\d+) \(1\) : 1$", TXT, re.M)]
assert len(RULES) == 25 and len(IN1) == len(IN2) == len(OUT) == 5

# rule table must equal Eq. (7):  u(i,j) = clamp(i + j - 3, 1, 5)
for i, j, k in RULES:
    assert k == min(max(i + j - 3, 1), 5), (i, j, k)

# ------------------------------------------------ Mamdani surface (panel c)
UG = np.linspace(-1, 1, 401)                 # output universe for the centroid
MU_OUT = np.array([mu(k, p, UG) for _, k, p in OUT])


def flc(e, de):
    """min / min / max / centroid, exactly as declared in the .fis header"""
    e = np.atleast_1d(e); de = np.atleast_1d(de)
    A = np.array([mu(k, p, e) for _, k, p in IN1])        # 5 x N
    B = np.array([mu(k, p, de) for _, k, p in IN2])
    agg = np.zeros((len(e), UG.size))
    for i, j, k in RULES:
        w = np.minimum(A[i - 1], B[j - 1])                # AND = min
        agg = np.maximum(agg, np.minimum(w[:, None],      # imp = min
                                        MU_OUT[k - 1][None, :]))
    num = (agg * UG[None, :]).sum(1)
    den = agg.sum(1)
    return np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)


print("FLC(0,0)          = %.3e   (manuscript: |FLC(0,0)| < 1e-16)" % flc(0, 0)[0])
_t = np.array([0.37, -0.62, 0.11, 0.94])
print("odd-symmetry dev  = %.3e" % np.abs(flc(_t, _t) + flc(-_t, -_t)).max())

N = 61
g = np.linspace(-1, 1, N)
E, DE = np.meshgrid(g, g)
Z = np.array([flc(E[r], DE[r]) for r in range(N)])

# --------------------------------------------------------------- the figure
COLS = [("NB", "#C00000", "-"), ("NS", "#E8862A", (0, (6, 2.5))),
        ("Z", "#4B7A2B", (0, (7, 2, 1.2, 2))), ("PS", "#5B9BD5", (0, (1.2, 2))),
        ("PB", "#1F4E79", (0, (8, 2, 1.2, 2)))]

w_in = W_CM / 2.54
fig = plt.figure(figsize=(w_in, w_in * 1582.0 / 1979.0))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.55],
                      left=0.085, right=0.985, top=0.975, bottom=0.045,
                      wspace=0.26, hspace=0.72)

x = np.linspace(-1, 1, 2001)
for col, (sec, xlab, cap) in enumerate(
        [(IN1, r"$e_\theta$ (normalized)",
          r"(a) Input MFs for $e_\theta$ ($s_e$ = 0.210)"),
         (OUT, r"$u_{FLC}$ (normalized)",
          r"(b) Output MFs for $u_{FLC}$ ($s_u$ = 0.291)")]):
    ax = fig.add_subplot(gs[0, col])
    for (nm, kind, p), (lab, c, ls) in zip(sec, COLS):
        ax.plot(x, mu(kind, p, x), color=c, linestyle=ls, linewidth=1.7,
                label=lab)
    ax.set_xlim(-1, 1); ax.set_ylim(-0.04, 1.06)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xlabel(xlab); ax.set_ylabel(r"Membership degree $\mu$")
    ax.grid(True, color="#CCCCCC", linewidth=0.6); ax.set_axisbelow(True)
    ax.set_title(cap, y=-0.42, fontsize=FS)
    if col == 0:
        handles, labels = ax.get_legend_handles_labels()

fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, 0.565),
           ncol=5, frameon=True, edgecolor="#AAAAAA", framealpha=1.0,
           handlelength=2.8, columnspacing=1.8, fontsize=FS)

ax3 = fig.add_subplot(gs[1, :], projection="3d")
ax3.plot_surface(E, DE, Z, cmap="coolwarm", linewidth=0.25,
                 edgecolors="white", rstride=1, cstride=1,
                 antialiased=True, shade=False)
ax3.set_xlabel(r"$e_\theta$", labelpad=-2)
ax3.set_ylabel(r"$\dot{e}_\theta$", labelpad=-2)
ax3.set_zlabel(r"$u_{FLC}$", labelpad=-4)
for a in (ax3.xaxis, ax3.yaxis):
    a.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax3.set_zticks([-0.5, 0.0, 0.5])
ax3.tick_params(pad=-1)
ax3.view_init(elev=26, azim=-122)
ax3.set_box_aspect((1, 1, 0.62), zoom=1.03)
ax3.set_title(r"(c) Fuzzy control surface (anti-symmetric about the origin)",
              y=-0.10, fontsize=FS)

fig.savefig("Fig2_PSOFLC_17cm.pdf", facecolor="white")
fig.savefig("Fig2_PSOFLC_17cm.svg", facecolor="white")
fig.savefig("Fig2_PSOFLC_17cm_600dpi.png", dpi=600, facecolor="white")
plt.close(fig)
print("Fig. 2 regenerated at %.1f cm wide, text %.0f pt" % (W_CM, FS))
