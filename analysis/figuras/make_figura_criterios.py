"""
make_figura_criterios.py
Annotation-criteria figure for the thesis. Two panels, same frame:
  (a) grayscale frame, no annotation
  (b) same frame + reference mask (semi-transparent fill ~45% + solid 1.5px contour)
Both cropped to the SAME cervical region / same exact size. No titles/axes/legend.
(a)/(b) labels below each panel. Minimal margins. Exports vector PDF + 300 dpi PNG.
READ-ONLY on source data.
"""
import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VF = "v138_f67"
SPLIT = "val"
BASE = r"G:/My Drive/UNM_vertebras_seg_v3/data/inca_dataset"
OUT_DIR = r"G:/My Drive/UNM_vertebras_seg_v3/resultados/inter_rater_inca/figuras_finales"
PAD_FRAC = 0.18            # crop padding around the mask bbox
FILL_ALPHA = 0.45
CONTOUR_LW = 1.5
GREEN = (0.10, 0.80, 0.20)  # reference color (matches thesis "green outline")

os.makedirs(OUT_DIR, exist_ok=True)
frame = np.array(Image.open(f"{BASE}/{SPLIT}/images/{VF}.png").convert("L"))
mask = np.array(Image.open(f"{BASE}/{SPLIT}/masks/{VF}.png").convert("L")) > 127

# crop to cervical region (mask bbox + padding), identical for both panels
ys, xs = np.where(mask)
y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
ph = int((y1 - y0 + 1) * PAD_FRAC); pw = int((x1 - x0 + 1) * PAD_FRAC)
H, W = frame.shape
Y0, Y1 = max(0, y0 - ph), min(H, y1 + ph + 1)
X0, X1 = max(0, x0 - pw), min(W, x1 + pw + 1)
fr = frame[Y0:Y1, X0:X1]
mk = mask[Y0:Y1, X0:X1]

# semi-transparent green fill as an RGBA overlay
overlay = np.zeros((*mk.shape, 4), float)
overlay[mk] = (*GREEN, FILL_ALPHA)

ch, cw = fr.shape
aspect = cw / ch
fig, axes = plt.subplots(1, 2, figsize=(2 * aspect * 3.2, 3.2))
for ax in axes:
    ax.imshow(fr, cmap="gray", interpolation="nearest")
    ax.set_axis_off()
# panel (b): fill + solid contour
axes[1].imshow(overlay, interpolation="nearest")
axes[1].contour(mk, levels=[0.5], colors=[GREEN], linewidths=CONTOUR_LW)
# (a)/(b) labels below each panel
axes[0].text(0.5, -0.03, "(a)", transform=axes[0].transAxes, ha="center", va="top", fontsize=12)
axes[1].text(0.5, -0.03, "(b)", transform=axes[1].transAxes, ha="center", va="top", fontsize=12)

plt.subplots_adjust(left=0, right=1, top=1, bottom=0.05, wspace=0.02)
for ext in ("pdf", "png"):
    p = os.path.join(OUT_DIR, f"figura_criterios.{ext}")
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("saved", p)
plt.close(fig)
print(f"crop size: {ch}x{cw} px  (from {VF}, split={SPLIT})")
