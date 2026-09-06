"""
make_figura_categorias.py
Final figure: 5 annotation-problem categories, ONE ROW PER CATEGORY (A-E).
Each row = [full frame | full frame + mask(s)] (no zoom). A,B,D show two
annotations (cyan/orange); C,E show one mask (cyan). No embedded text/labels
(LaTeX adds A-E). Minimal margins. Vector PDF + 300 dpi PNG. READ-ONLY.
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
C_A, C_B = "#00E5FF", "#FF9800"
LW = 1.5
os.makedirs(OUT_DIR, exist_ok=True)

metrics = {r["video_frame"]: r for r in csv.DictReader(open(METRICS, encoding="utf-8"))}
bof = {r["video_frame"]: str(r["batch"])
       for r in csv.DictReader(open(f"{LOCAL}/video_frame_metadata.csv", encoding="utf-8"))}

def _bin(a):
    a = np.where(a > 127, 255, 0).astype(np.uint8)
    if (a == 255).sum() / a.size > 0.10:
        a = 255 - a
    return a > 127

def load_mask(vf, lab):
    b = bof[vf]
    return _bin(np.array(Image.open(f"{LOCAL}/rotulos/batch-{b}/{lab}/{vf}/Mask.tif").convert("L")))

def load_frame(vf, split):
    return np.array(Image.open(f"{LOCAL}/frames/{split}/{vf}.png").convert("L"))

# one row per category. mode 'two' = both annotations, 'one' = labeler_A only.
ROWS = [
    ("v39_f4", "two"),     # A missing
    ("v207_f159", "two"),  # B extra
    ("v235_f214", "one"),  # C over-extended
    ("v128_f235", "two"),  # D edge error
    ("v12_f90", "one"),    # E limited visibility
]

nrows = len(ROWS)
fig, axes = plt.subplots(nrows, 2, figsize=(6, 3 * nrows))
for (vf, mode), axrow in zip(ROWS, axes):
    r = metrics[vf]; split = r["split"]
    fr = load_frame(vf, split)
    masks = [(load_mask(vf, r["labeler_A"]), C_A)]
    if mode == "two":
        masks.append((load_mask(vf, r["labeler_B"]), C_B))
    axrow[0].imshow(fr, cmap="gray", interpolation="nearest"); axrow[0].set_axis_off()
    axrow[1].imshow(fr, cmap="gray", interpolation="nearest")
    for m, col in masks:
        axrow[1].contour(m, levels=[0.5], colors=[col], linewidths=LW)
    axrow[1].set_axis_off()

fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005, wspace=0.02, hspace=0.03)
for ext in ("pdf", "png"):
    p = os.path.join(OUT_DIR, f"figura_categorias.{ext}")
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("saved", p)
plt.close(fig)
print("rows:", [(vf, metrics[vf]["labeler_A"], metrics[vf].get("labeler_B", "") if mode == "two" else "-")
                for vf, mode in ROWS])
