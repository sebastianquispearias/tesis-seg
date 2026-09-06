"""
make_figura_interanotador.py
Inter-annotator figure: 2 panels, full frame (no zoom), same pixel size.
  (a) v26_f135 (typical agreement)   (b) v20_f152 (extra vertebra)
Both annotations as CONTOUR only: annotator A cyan, annotator B orange, 1.5 px,
same colors/width in both. No burned-in text (no title/legend/Dice). No (a)/(b)
labels (LaTeX adds them). Minimal margins. Exports vector PDF + 300 dpi PNG.
READ-ONLY; masks read from local mirror.
"""
import csv, os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = r"G:/My Drive/UNM_vertebras_seg_v3"
LOCAL = r"C:/Users/User/temp_inter_rater/inca_fig"
METRICS = f"{PROJECT}/resultados/inter_rater_inca/figure_candidates/per_frame_metrics.csv"
OUT_DIR = f"{PROJECT}/resultados/inter_rater_inca/figuras_finales"
COLOR_A, COLOR_B = "#00E5FF", "#FF9800"     # A cyan, B orange
LW = 1.5
FRAMES = [("v26_f135", "val"), ("v20_f152", "train")]
os.makedirs(OUT_DIR, exist_ok=True)

metrics = {r["video_frame"]: r for r in csv.DictReader(open(METRICS, encoding="utf-8"))}
bof = {r["video_frame"]: str(r["batch"])
       for r in csv.DictReader(open(f"{LOCAL}/video_frame_metadata.csv", encoding="utf-8"))}

def mask(vf, lab):
    b = bof[vf]
    a = np.array(Image.open(f"{LOCAL}/rotulos/batch-{b}/{lab}/{vf}/Mask.tif").convert("L"))
    a = np.where(a > 127, 255, 0).astype(np.uint8)
    if (a == 255).sum() / a.size > 0.10:
        a = 255 - a
    return a > 127

def frame(vf, split):
    return np.array(Image.open(f"{LOCAL}/frames/{split}/{vf}.png").convert("L"))

# both frames are 1024x1024 (verified) -> identical pixel size, no rescaling
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for ax, (vf, split) in zip(axes, FRAMES):
    r = metrics[vf]
    fr = frame(vf, split)
    A = mask(vf, r["labeler_A"]); B = mask(vf, r["labeler_B"])
    ax.imshow(fr, cmap="gray", interpolation="nearest")
    ax.contour(A, levels=[0.5], colors=[COLOR_A], linewidths=LW)
    ax.contour(B, levels=[0.5], colors=[COLOR_B], linewidths=LW)
    ax.set_axis_off()
    ax.set_aspect("equal")

plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
for ext in ("pdf", "png"):
    p = os.path.join(OUT_DIR, f"figura_interanotador.{ext}")
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.01)
    print("saved", p)
plt.close(fig)
print(f"Dice: v26_f135={metrics['v26_f135']['dice']}  v20_f152={metrics['v20_f152']['dice']}")
