#!/usr/bin/env python3
"""Fig. 1 - Schematic of the reaction-wheel inverted pendulum (RWIP).

Drawn to scale in metres: Lp = 0.256 m, Lr = 0.385 m (both from the axis O).
Conventions match the simulation model:
    theta = 0  -> upright (unstable) equilibrium, measured from the vertical
    phi        -> flywheel angle RELATIVE to the pendulum body
    tau = (Kt/Ra)(V - Ke*phi_dot) on the wheel, reaction -tau on the body
    Ip about the pendulum centre of mass, Ir about the wheel spin axis
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")

# final published size and type size -------------------------------------
W_CM_FIG = 8.0     # single-column width, centimetres
FS_FIG = 10.0      # every label: exactly 10 pt at that size
import matplotlib.pyplot as plt
from matplotlib.patches import (Arc, Circle, Wedge, FancyArrowPatch,
                                Rectangle, Polygon)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "font.size": FS_FIG,
})

LANG = sys.argv[1] if len(sys.argv) > 1 else "en"
VI = (LANG == "vi")

T = dict(
    upright=r"($\theta = 0$: upright, unstable)",
    motor="DC motor:  $R_a,\\ K_e,\\ K_t$",
    react=r"$-\tau$",
    lp="$L_p = 0.256$ m",
    lr="$L_r = 0.385$ m",
    notes=[
        r"$\theta$ : tilt from the upward vertical ($+$ counter-clockwise); "
        r"$\theta = 0$ is upright, unstable.",
        r"$\varphi$ : flywheel angle RELATIVE to the pendulum body "
        r"(dashed line = body reference).",
        r"$I_p$ about $G_p$; $I_r$ about the spin axis through $o_2$; "
        r"$L_p$, $L_r$ measured from $O$.",
        r"$\tau$ on the wheel, reaction $-\tau$ on the body; "
        r"$b_p$, $b_r$ : viscous joint friction.",
        r"Rod drawn to scale ($L_p\!:\!L_r = 0.256\!:\!0.385$); flywheel "
        r"outline is schematic.",
    ],
)
if VI:
    T = dict(
        upright=r"($\theta = 0$: thẳng đứng, không ổn định)",
        motor="Động cơ DC:  $R_a,\\ K_e,\\ K_t$",
        react=r"$-\tau$  (phản lực)",
        lp="$L_p = 0{,}256$ m",
        lr="$L_r = 0{,}385$ m",
        notes=[
            r"$\theta$ : góc nghiêng của thanh lắc so với phương thẳng đứng "
            r"hướng lên ($+$ ngược kim đồng hồ);",
            r"     $\theta = 0$ là vị trí thẳng đứng — điểm cân bằng không ổn định.",
            r"$\varphi$ : góc bánh đà đo TƯƠNG ĐỐI so với thân thanh lắc "
            r"(nét đứt = mốc gắn thân).",
            r"$I_p$ lấy quanh $G_p$; $I_r$ quanh trục tự quay tại $o_2$; "
            r"$L_p$, $L_r$ đều đo từ trục quay $O$.",
            r"$\tau$ tác dụng lên bánh đà, phản lực $-\tau$ lên thân; "
            r"$b_p$, $b_r$ : ma sát nhớt hai khớp.",
            r"Thanh vẽ đúng tỉ lệ ($L_p\!:\!L_r = 0{,}256\!:\!0{,}385$); "
            r"bánh đà chỉ minh họa.",
        ],
    )

# ----------------------------------------------------------------- geometry
LP, LR = 0.256, 0.385
LINK_END = 0.435
TH = np.deg2rad(23.0)
PHI = np.deg2rad(38.0)

u = np.array([np.sin(TH), np.cos(TH)])
n = np.array([-np.cos(TH), np.sin(TH)])
O = np.array([0.0, 0.0])
GP = LP * u
O2 = LR * u
TIP = LINK_END * u
A = np.degrees(np.arctan2(u[1], u[0]))          # = 90 - theta

R_OUT, R_IN, R_HUB = 0.072, 0.045, 0.014

C_RULE = "#4a4a4a"
C_ACC = "#0b6e99"
C_TAU = "#b03a2e"
C_MASS = "#1a1a1a"

fig, ax = plt.subplots(figsize=(W_CM_FIG / 2.54,
                                W_CM_FIG / 2.54 / 0.95566))
ax.set_aspect("equal")
ax.axis("off")

# ------------------------------------------------------- ground and pivot
ax.add_patch(Rectangle((-0.150, -0.066), 0.30, 0.020, facecolor="#dcdcdc",
                       edgecolor=C_RULE, lw=0.9, hatch="////", zorder=1))
ax.plot([-0.150, 0.150], [-0.046, -0.046], color=C_RULE, lw=1.4, zorder=2)
ax.add_patch(Polygon([[0, 0], [-0.030, -0.046], [0.030, -0.046]],
                     closed=True, facecolor="white", edgecolor=C_RULE,
                     lw=1.4, zorder=3))
ax.add_patch(Circle(O, 0.0075, facecolor="white", edgecolor="black",
                    lw=1.3, zorder=6))

# ------------------------------- global frame; the y axis IS the theta = 0 line
ax.annotate("", xy=(0.205, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", lw=1.3, color="black",
                            mutation_scale=13))
ax.annotate("", xy=(0, 0.510), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", lw=1.3, color="black",
                            mutation_scale=13))
ax.text(0.212, -0.002, "$x$", ha="left", va="center", style="italic")
ax.text(-0.010, 0.516, "$y$", ha="right", va="bottom", style="italic")
ax.text(0.008, 0.516, T["upright"], color=C_ACC,
        ha="left", va="bottom", fontsize=FS_FIG)
ax.text(-0.030, -0.008, "$O$", ha="right", va="center")

# ---------------------------------------------------------------- pendulum
ax.plot([O[0], TIP[0]], [O[1], TIP[1]], color=C_MASS, lw=3.4,
        solid_capstyle="round", zorder=4)

# theta measured FROM THE VERTICAL, kept close to the pivot
r_th = 0.095
ax.add_patch(Arc(O, 2 * r_th, 2 * r_th, angle=0, theta1=A, theta2=90,
                 color=C_ACC, lw=1.6, zorder=5))
m = np.deg2rad(0.5 * (A + 90))
ax.text(0.121 * np.cos(m), 0.121 * np.sin(m), r"$\theta$", color=C_ACC,
        ha="center", va="center", fontsize=FS_FIG)
ax.annotate("", xy=(r_th * np.cos(np.deg2rad(A + 2.5)),
                    r_th * np.sin(np.deg2rad(A + 2.5))),
            xytext=(r_th * np.cos(np.deg2rad(A + 10)),
                    r_th * np.sin(np.deg2rad(A + 10))),
            arrowprops=dict(arrowstyle="-|>", color=C_ACC, lw=1.4,
                            mutation_scale=11))

# ------------------------------------------------- pendulum centre of mass
ax.add_patch(Wedge(GP, 0.0115, 0, 90, facecolor=C_MASS, ec="none", zorder=7))
ax.add_patch(Wedge(GP, 0.0115, 180, 270, facecolor=C_MASS, ec="none", zorder=7))
ax.add_patch(Circle(GP, 0.0115, facecolor="none", edgecolor=C_MASS,
                    lw=1.0, zorder=8))
ax.text(0.136, 0.228, "$G_p$\n($m_p,\\ I_p$)", ha="left", va="center",
        linespacing=1.35,
        fontsize=FS_FIG)
ax.plot([GP[0] + 0.013, 0.114], [GP[1] + 0.002, 0.222], color=C_RULE,
        lw=0.7, zorder=3)

# ------------------------------------------------------------- flywheel
ax.add_patch(Circle(O2, R_OUT, facecolor="#cfd8bf", edgecolor=C_MASS,
                    lw=1.4, zorder=5))
ax.add_patch(Circle(O2, R_IN, facecolor="white", edgecolor=C_MASS,
                    lw=1.4, zorder=6))
ax.add_patch(Circle(O2, R_HUB, facecolor="white", edgecolor=C_MASS,
                    lw=1.2, zorder=8))
ax.plot(O2[0], O2[1], marker="+", color=C_MASS, ms=7, mew=1.2, zorder=9)
ax.text(0.247, 0.470, "$o_2 \\equiv G_r$  ($m_r,\\ I_r$)", ha="left",
        va="center", fontsize=FS_FIG)
ax.plot([O2[0] + 0.036 * np.cos(np.deg2rad(62)) + 0.026,
         0.242], [O2[1] + 0.062, 0.466], color=C_RULE, lw=0.7, zorder=3)

# body-fixed reference through the wheel + spoke at phi
ra_, rb_ = O2 - 0.058 * u, O2 + 0.098 * u
ax.plot([ra_[0], rb_[0]], [ra_[1], rb_[1]], color=C_RULE, lw=1.0,
        ls=(0, (4, 3)), zorder=10)
sd = np.array([np.cos(np.deg2rad(A) + PHI), np.sin(np.deg2rad(A) + PHI)])
sp = O2 + 0.098 * sd
ax.plot([O2[0], sp[0]], [O2[1], sp[1]], color=C_TAU, lw=2.0, zorder=10)
ax.add_patch(Arc(O2, 2 * 0.062, 2 * 0.062, angle=0, theta1=A,
                 theta2=A + np.degrees(PHI), color=C_TAU, lw=1.5, zorder=11))
m2 = np.deg2rad(A) + 0.52 * PHI
ax.text(O2[0] + 0.086 * np.cos(m2), O2[1] + 0.086 * np.sin(m2),
        r"$\phi$", color=C_TAU, ha="center", va="center", fontsize=FS_FIG)

# tau on the wheel (upper right)
p1 = O2 + (R_OUT + 0.018) * np.array([np.cos(np.deg2rad(-18)),
                                      np.sin(np.deg2rad(-18))])
p2 = O2 + (R_OUT + 0.018) * np.array([np.cos(np.deg2rad(44)),
                                      np.sin(np.deg2rad(44))])
ax.add_patch(FancyArrowPatch(p1, p2, connectionstyle="arc3,rad=-0.40",
                             arrowstyle="-|>", mutation_scale=15,
                             color=C_TAU, lw=1.7, zorder=11))
ax.text(0.262, 0.360, r"$\tau$", color=C_TAU, fontsize=FS_FIG, ha="left",
        va="center")

# reaction -tau on the pendulum body (right-hand side of the rod)
q0 = 0.225 * u - 0.040 * n
q1 = 0.135 * u - 0.030 * n
ax.add_patch(FancyArrowPatch(q0, q1, connectionstyle="arc3,rad=0.38",
                             arrowstyle="-|>", mutation_scale=13,
                             color=C_TAU, lw=1.5, zorder=10))
ax.text(0.130, 0.118, T["react"], color=C_TAU, fontsize=FS_FIG,
        ha="left", va="center")

# ------------------------------------------------------ dimension lines
def dim(p1_, p2_, off, label, tpos=0.24, skip=(0.0, 0.0)):
    a_, b_ = p1_ + off * n, p2_ + off * n
    for p, q, sk in ((p1_, a_, skip[0]), (p2_, b_, skip[1])):
        st = p + sk * n
        ax.plot([st[0], q[0] + 0.012 * n[0]], [st[1], q[1] + 0.012 * n[1]],
                color=C_ACC, lw=0.7, ls=(0, (3, 3)), zorder=3)
    ax.annotate("", xy=b_, xytext=a_,
                arrowprops=dict(arrowstyle="<|-|>", color=C_ACC, lw=1.2,
                                mutation_scale=11, shrinkA=0, shrinkB=0))
    t = a_ + tpos * (b_ - a_) + 0.017 * n
    ax.text(t[0], t[1], label, color=C_ACC, ha="center", va="center",
            fontsize=FS_FIG, rotation=A - 90, rotation_mode="anchor", zorder=13,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.4))

dim(O, GP, 0.090, T["lp"], tpos=0.80)
dim(O, O2, 0.196, T["lr"], tpos=0.26, skip=(0.0, 0.078))

# ------------------------------------------------------- friction, gravity
ax.add_patch(Arc(O, 2 * 0.050, 2 * 0.050, angle=0, theta1=8, theta2=54,
                 color=C_RULE, lw=1.3, zorder=6))
ax.annotate("", xy=(0.050 * np.cos(np.deg2rad(54)), 0.050 * np.sin(np.deg2rad(54))),
            xytext=(0.050 * np.cos(np.deg2rad(46)), 0.050 * np.sin(np.deg2rad(46))),
            arrowprops=dict(arrowstyle="-|>", color=C_RULE, lw=1.2,
                            mutation_scale=10))
ax.text(0.058, 0.024, "$b_p$", color=C_RULE, fontsize=FS_FIG, ha="left",
        va="bottom")

ax.add_patch(Arc(O2, 2 * 0.030, 2 * 0.030, angle=0, theta1=A + 140,
                 theta2=A + 212, color=C_RULE, lw=1.3, zorder=12))
ax.annotate("", xy=(O2[0] + 0.030 * np.cos(np.deg2rad(A + 212)),
                    O2[1] + 0.030 * np.sin(np.deg2rad(A + 212))),
            xytext=(O2[0] + 0.030 * np.cos(np.deg2rad(A + 204)),
                    O2[1] + 0.030 * np.sin(np.deg2rad(A + 204))),
            arrowprops=dict(arrowstyle="-|>", color=C_RULE, lw=1.2,
                            mutation_scale=10), zorder=12)
ax.plot([O2[0] - 0.026, 0.048], [O2[1] - 0.020, 0.292], color=C_RULE,
        lw=0.7, zorder=12)
ax.text(0.044, 0.292, "$b_r$", color=C_RULE, fontsize=FS_FIG, ha="right",
        va="center", zorder=13)

ax.annotate("", xy=(-0.126, 0.372), xytext=(-0.126, 0.452),
            arrowprops=dict(arrowstyle="-|>", lw=1.4, color="black",
                            mutation_scale=13))
ax.text(-0.118, 0.410, "$g$", ha="left", va="center", fontsize=FS_FIG)

# --------------------------------------------------------------- actuator
ax.add_patch(Rectangle((0.248, 0.070), 0.244, 0.112, facecolor="white",
                       edgecolor=C_TAU, lw=1.1, zorder=6))
ax.text(0.370, 0.126, T["motor"].replace(":  ", ":\n"),
        ha="center", va="center", fontsize=FS_FIG, color=C_TAU, zorder=7,
        linespacing=1.35)
ax.annotate("", xy=(O2[0] + 0.034, O2[1] - 0.068), xytext=(0.336, 0.184),
            arrowprops=dict(arrowstyle="-|>", color=C_TAU, lw=1.0,
                            mutation_scale=10, ls=(0, (4, 2))), zorder=6)
ax.annotate("", xy=(0.370, 0.068), xytext=(0.370, 0.008),
            arrowprops=dict(arrowstyle="-|>", color=C_TAU, lw=1.3,
                            mutation_scale=12))
ax.text(0.382, 0.018, "$u$", color=C_TAU, ha="left",
        va="center", fontsize=FS_FIG)

# ------------------------------------------------------------------ notes
ax.set_xlim(-0.30, 0.62)
ax.set_ylim(-0.20, 0.62)
fig.subplots_adjust(left=0.004, right=0.996, top=0.997, bottom=0.003)

# ---- fit the view box to what was actually drawn --------------------------
from matplotlib.transforms import Bbox
fig.canvas.draw()
_r = fig.canvas.get_renderer()
_bbs = []
for _a in list(ax.texts) + list(ax.patches) + list(ax.lines) + list(ax.artists):
    try:
        _b = _a.get_window_extent(renderer=_r)
        if _b.width > 0 and _b.height > 0:
            _bbs.append(_b)
    except Exception:
        pass
_u = Bbox.union(_bbs).transformed(ax.transData.inverted())
_mx, _my = 0.022 * _u.width, 0.022 * _u.height
_x0, _x1 = _u.x0 - _mx, _u.x1 + _mx
_y0, _y1 = _u.y0 - _my, _u.y1 + _my
ax.set_xlim(_x0, _x1)
ax.set_ylim(_y0, _y1)
# canvas exactly 8.0 cm wide; height follows the content so aspect stays equal
fig.set_size_inches(W_CM_FIG / 2.54,
                    W_CM_FIG / 2.54 * (_y1 - _y0) / (_x1 - _x0))
print("Fig. 1  view %.3f x %.3f data units -> %.2f cm x %.2f cm"
      % (_x1 - _x0, _y1 - _y0, W_CM_FIG,
         W_CM_FIG * (_y1 - _y0) / (_x1 - _x0)))
stem = "/tmp/wrip/figsrc/Fig1_RWIP_8cm"
fig.savefig(stem + "_1250dpi.png", dpi=1250, facecolor="white")
fig.savefig(stem + ".svg", facecolor="white")
fig.savefig(stem + ".pdf", facecolor="white")
print("ok")
