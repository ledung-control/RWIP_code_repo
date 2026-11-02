# -*- coding: utf-8 -*-
"""Fig. 3 - Block structure of the cascade SMC-ESO.

Redrawn as true vector art at the final published size (8.0 cm single column,
all text exactly 10 pt).  Every block, signal, equation, coefficient and arrow
is transcribed from the manuscript's existing Fig. 3.  Two changes:
  * line breaking, because the same content must fit one column instead of two;
  * the control-voltage symbol in the SMC and DC-motor blocks is written "u",
    not "V", so the figure agrees with Eq. (1), Table 1 and Fig. 1.  It is the
    same signal (the arrow between the two blocks is u_SMC-ESO); only the
    symbol used to name it changes.
"""
from fig_common import (new_canvas, box, arrow, poly_arrow, label, save,
                        audit_fit, LINE_UNITS,
                        C_BLUE_E, C_BLUE_F, C_GOLD_E, C_GOLD_F)

L, R = 12.0, 95.0            # block band; left strip = theta feedback,
W = R - L                    # right strip = phi feedback
GAP = 7.0                    # vertical space taken by each signal arrow
TOP = 4.0

import sys
HEIGHT = float(sys.argv[1]) if len(sys.argv) > 1 else 196.0
fig, ax = new_canvas(HEIGHT)
Y = HEIGHT - TOP
boxes = []


def blk(y, lines, ec=C_BLUE_E, fc=C_BLUE_F, x=L, w=W, heights=None):
    y1 = box(ax, x, y, w, lines, ec, fc, heights=heights)
    boxes.append((x, y1, w, y - y1))
    return y1


# 1 - cascade outer loop ----------------------------------------------------
y = blk(Y, [r"Cascade outer loop",
            r"$\theta_{\mathrm{ref}} = \mathrm{sat}(-(K_{p\phi}\phi + "
            r"K_{v\phi}\dot{\phi}),$",
            r"$\pm\,\theta_{\mathrm{ref,max}})$"])
y_outer_bot = y
arrow(ax, (50, y), (50, y - GAP), C_BLUE_E)
label(ax, 52.5, y - GAP / 2.0, r"$\theta_{\mathrm{ref}}$", C_BLUE_E)
y -= GAP

# 2 - SMC + ESO -------------------------------------------------------------
y_smc_top = y
y = blk(y, [r"SMC + 3rd-order ESO $(e = \theta - \theta_{\mathrm{ref}})$",
            r"$s = ce + z_2,\quad u = \frac{1}{g_a}(-cz_2 - z_3$",
            r"$-\,ks - \eta\,\mathrm{sat}(s/\Phi))$",
            r"$c = 8,\; k = 5,\; \eta = 1.0,\; \Phi = 0.05,$",
            r"$\omega_0 = 20$ rad/s"])
y_smc_bot = y
arrow(ax, (50, y), (50, y - GAP), C_BLUE_E)
label(ax, 52.5, y - GAP / 2.0, r"$u_{\mathrm{SMC\text{-}ESO}}$", C_BLUE_E)
y -= GAP

# 3 - DC motor --------------------------------------------------------------
y = blk(y, [r"DC motor",
            r"$\tau = (K_t/R_a)(u - K_e\dot{\phi})$"], x=L + 12, w=W - 24)
arrow(ax, (50, y), (50, y - GAP), C_BLUE_E)
label(ax, 52.5, y - GAP / 2.0, r"$\tau$", C_BLUE_E)
y -= GAP

# 4 - plant -----------------------------------------------------------------
y_pl_top = y
y = blk(y, [r"RWIP dynamics (double integrator on $\phi$)",
            r"$I_{p,eq}\,\ddot{\theta} = -\tau + b_r\dot{\phi} + "
            r"mgl_{eq}\sin\theta - b_p\dot{\theta}$",
            r"$I_r\ddot{\phi} = \tau - b_r\dot{\phi} - "
            r"I_{p,eq}\ddot{\theta}$"])
y_pl_bot = y

# three signal arrows into the invariant panel
arrow(ax, (28, y), (28, y - GAP), C_BLUE_E)
label(ax, 25.5, y - GAP / 2.0, r"$\theta,\ \dot{\theta}$", C_BLUE_E, ha="right")
arrow(ax, (54, y), (54, y - GAP), C_GOLD_E)
arrow(ax, (80, y), (80, y - GAP), "#B41F2E")
label(ax, 82.5, y - GAP / 2.0, r"$\phi,\ \dot{\phi}$", "#B41F2E")
y -= GAP

# 5 - integral invariant ----------------------------------------------------
y = blk(y, [r"Integral invariant (Section 3.2):",
            r"$\int_0^T \sin\theta\, dt = \dfrac{I_r\dot{\phi}(T) - "
            r"b_p\theta_0}{mgl_{eq}}$",
            r"$\phi$: long-term integrating state (a structural",
            r"difference from a flywheel-free IP)"],
        ec=C_GOLD_E, fc=C_GOLD_F, x=2.0, w=96.0,
        heights=[1.0, 2.6, 1.15, 1.0])

# feedback: theta -> SMC (left strip) ---------------------------------------
xf = 5.5
poly_arrow(ax, [(L, (y_pl_top + y_pl_bot) / 2.0), (xf, (y_pl_top + y_pl_bot) / 2.0),
                (xf, (y_smc_top + y_smc_bot) / 2.0),
                (L, (y_smc_top + y_smc_bot) / 2.0)], C_BLUE_E)
label(ax, xf - 1.4, (y_pl_top + y_smc_bot) / 2.0 + 6, r"$\theta$", C_BLUE_E,
      ha="right")

# feedback: phi, phidot -> outer loop (right strip) -------------------------
xg = 98.0
poly_arrow(ax, [(R, y_pl_top - 6.0), (xg, y_pl_top - 6.0),
                (xg, y_outer_bot + 6.0), (R, y_outer_bot + 6.0)], "#B41F2E")
label(ax, xg - 1.2, (y_pl_top + y_outer_bot) / 2.0, r"$\phi,\ \dot{\phi}$",
      "#B41F2E", ha="right")

fig.canvas.draw()
bad = [b for b in audit_fit(fig, ax, boxes) if b[0] not in
       (r"$\theta$", r"$\phi,\ \dot{\phi}$")]
print("Fig. 3  overflowing labels:", bad if bad else "none")
print("Fig. 3  content bottom at y = %.1f (canvas %.1f)" % (y, HEIGHT))
print("Fig. 3  suggested height = %.1f" % (HEIGHT - y + TOP))
print("Fig. 3  canvas: 8.0 cm x %.2f cm" % (8.0 * HEIGHT / 100.0))
save(fig, "Fig3_SMCESO_8cm")
