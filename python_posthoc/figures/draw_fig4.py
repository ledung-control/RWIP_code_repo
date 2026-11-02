# -*- coding: utf-8 -*-
"""Fig. 4 - Residual control architecture.

Redrawn as true vector art at the final published size (8.0 cm single column,
all text exactly 10 pt).  Content is transcribed from the manuscript's existing
Fig. 4; the only wording change is the internal label "SMC-ESO@ZOH", which is
set in the journal-ready form "SMC-ESO (ZOH)" used by the caption.
"""
import sys
from fig_common import (new_canvas, box, arrow, poly_arrow, label, save,
                        audit_fit, summing_junction,
                        C_BLUE_E, C_BLUE_F, C_ORNG_E, C_ORNG_F,
                        C_RED_E, C_RED_F, C_GREY_E, C_GREY_F)

HEIGHT = float(sys.argv[1]) if len(sys.argv) > 1 else 260.0
fig, ax = new_canvas(HEIGHT)

XL, WL = 2.0, 27.0            # left branch: fixed PSO-FLC baseline
XR, WR = 32.0, 66.0           # right branch: residual actor + stages
XC_L = XL + WL / 2.0          # u_FLC lane
XC_R = XR + WR / 2.0
GAP = 6.5
boxes = []


def blk(y, lines, ec, fc, x, w, heights=None):
    y1 = box(ax, x, y, w, lines, ec, fc, heights=heights)
    boxes.append((x, y1, w, y - y1))
    return y1


# 0 - state vector ----------------------------------------------------------
y = blk(HEIGHT - 4.0, [r"RWIP state: $\theta,\ \dot{\theta},\ \phi,\ \dot{\phi}$"],
        C_GREY_E, C_GREY_F, 22.0, 56.0)
# split to the two branches
ysp = y - 4.0
ax.plot([50, 50], [y, ysp], color="black", lw=1.1, zorder=4)
ax.plot([XC_L, XC_R], [ysp, ysp], color="black", lw=1.1, zorder=4)
arrow(ax, (XC_L, ysp), (XC_L, ysp - 4.0), "black")
arrow(ax, (XC_R, ysp), (XC_R, ysp - 4.0), "black")
ytop = ysp - 4.0

# 1 - the two parallel branches --------------------------------------------
yl = blk(ytop, [r"PSO-FLC", r"baseline", r"(Section 2,", r"fixed)"],
         C_BLUE_E, C_BLUE_F, XL, WL)
yr = blk(ytop, [r"Residual actor $\pi(s)$", r"(neural network, 2 hidden",
                r"layers, 256/layer)"],
         C_BLUE_E, C_BLUE_F, XR, WR)

# right branch: Stage A -> B -> C ------------------------------------------
y = yr
arrow(ax, (XC_R, y), (XC_R, y - GAP), C_ORNG_E); y -= GAP
y = blk(y, [r"Stage A — Behavior Cloning",
            r"Regression on (obs, action)", r"from SMC-ESO (ZOH)"],
        C_ORNG_E, C_ORNG_F, XR, WR)
arrow(ax, (XC_R, y), (XC_R, y - GAP), C_ORNG_E); y -= GAP
y = blk(y, [r"Stage B — TD3 fine-tuning",
            r"18 episodes, actor LR $10^{-5}$",
            r"(displacement $\approx$ 0.36 units)"],
        C_ORNG_E, C_ORNG_F, XR, WR)
arrow(ax, (XC_R, y), (XC_R, y - GAP), C_ORNG_E); y -= GAP
y = blk(y, [r"Stage C — Bias correction", r"(post hoc)",
            r"$a_{\mathrm{bias}} = \pi(0)$",
            r"$\pi'(s) = \pi(s) - a_{\mathrm{bias}}$"],
        C_RED_E, C_RED_F, XR, WR)
arrow(ax, (XC_R, y), (XC_R, y - GAP), C_RED_E); y -= GAP
y = blk(y, [r"$u_{corr} = u_{corr,\mathrm{max}}\cdot a,\ \ a \in [-1,\,1]$",
            r"$u_{corr,\mathrm{max}} = 10$ V"],
        C_BLUE_E, C_BLUE_F, XR + 3.0, WR - 6.0)
y_ucorr_bot = y

# summing junction ----------------------------------------------------------
y_sum = y - 13.0
summing_junction(ax, XC_L, y_sum)
# u_FLC lane down the left
arrow(ax, (XC_L, yl), (XC_L, y_sum + 2.4), C_BLUE_E)
label(ax, XC_L + 1.8, (yl + y_sum) / 2.0, r"$u_{FLC}$", C_BLUE_E)
# u_corr across
poly_arrow(ax, [(XC_R, y_ucorr_bot), (XC_R, y_sum), (XC_L + 2.4, y_sum)],
           C_RED_E)
label(ax, XC_R + 2.0, y_sum + 6.5, r"$u_{corr}$", C_RED_E)

# total control -------------------------------------------------------------
y = y_sum - 2.4
arrow(ax, (XC_L, y), (XC_L, y - 7.0), C_BLUE_E)
y = blk(y - 7.0,
        [r"$u_{tot} = \mathrm{sat}(u_{FLC} + u_{corr},\ \pm 24$ V$)$"],
        C_BLUE_E, C_BLUE_F, 4.0, 84.0)

fig.canvas.draw()
bad = [b for b in audit_fit(fig, ax, boxes)
       if b[0] not in (r"$u_{FLC}$", r"$u_{corr}$", "+")]
print("Fig. 4  overflowing labels:", bad if bad else "none")
print("Fig. 4  content bottom y = %.1f  -> suggested height %.1f"
      % (y, HEIGHT - y + 4.0))
print("Fig. 4  canvas: 8.0 cm x %.2f cm" % (8.0 * HEIGHT / 100.0))
save(fig, "Fig4_residual_8cm")
