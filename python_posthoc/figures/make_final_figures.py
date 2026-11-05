# -*- coding: utf-8 -*-
"""Generate the submitted Figs. 1-4 (PNG) from the archived masters in masters_submitted/.

Usage (from this folder):  python make_final_figures.py      Needs NumPy and Pillow.
Writes FigN_regenerated.png (white margin trimmed; Figs. 1, 3, 4 at 8 cm and Fig. 2 at 17 cm, 600 dpi).
"""
import os
import numpy as np
from PIL import Image, ImageChops
Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(HERE, "masters_submitted")
JOBS = [(1, "Fig1_RWIP_8cm_1250dpi.png", "all", 1890), (2, "Fig2_nolabel_600dpi.png", "v", None),
        (3, "Fig3_SMCESO_8cm_1250dpi.png", "all", 1890), (4, "Fig4_residual_8cm_1250dpi.png", "all", 1890)]
for n, master, mode, width in JOBS:
    im = Image.open(os.path.join(M, master)).convert("RGB")
    x0, y0, x1, y1 = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
    if mode == "all":
        p = 18; box = (max(x0 - p, 0), max(y0 - p, 0), min(x1 + p, im.size[0]), min(y1 + p, im.size[1]))
    else:
        p = 20; box = (0, max(y0 - p, 0), im.size[0], min(y1 + p, im.size[1]))
    out = im.crop(box)
    if width:
        out = out.resize((width, round(out.size[1] * width / out.size[0])), Image.LANCZOS)
    name = os.path.join(HERE, "Fig%d_regenerated.png" % n); out.save(name, dpi=(600, 600))

