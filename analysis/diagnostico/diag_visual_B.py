"""
diag_visual_B.py
Visual verification with METHOD B (GT-guided partition), one image per PDF page,
seed_0, for the 3 main configs. The prediction is COLORED per vertebra so a merged
blob visibly splits into C3/C4 along the GT boundary. READ-ONLY.
Output: resultados/diagnostico_dice/verificacion_visual_B/{config}.pdf
"""
import os, numpy as np, cv2
from PIL import Image
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BASE = r"C:/Users/User/temp_inter_rater/diag"
PREDS = os.path.join(BASE, "preds")
OUT = r"G:/My Drive/UNM_vertebras_seg_v3/resultados/diagnostico_dice/verificacion_visual_B"
os.makedirs(OUT, exist_ok=True)
CONN = np.ones((3, 3), int)
DILATE_PX = 8
LAB = ["C2", "C3", "C4", "C5", "C6", "C7"]
# distinct colors per vertebra index (RGB 0-1)
VCOL = [(0.0, 0.9, 1.0), (1.0, 0.6, 0.0), (0.3, 1.0, 0.3), (1.0, 0.2, 0.8), (1.0, 1.0, 0.2), (0.6, 0.4, 1.0)]

CONFIGS = [("UNM__PL-r3", "unm/gt", "unm/frames"),
           ("UNM__MT-alllateral", "unm/gt", "unm/frames"),
           ("INCA__MT-r15", "inca/gt", "inca/frames")]

def pad_sq(a, fill=0):
    im = Image.fromarray(a); w, h = im.size; s = max(w, h)
    c = Image.new(im.mode, (s, s), fill); c.paste(im, ((s-w)//2, (s-h)//2)); return np.array(c)
def gt320(p):
    m = np.array(Image.open(p).convert("L")); m = np.where(m > 127, 255, 0).astype(np.uint8)
    return cv2.resize(pad_sq(m, 0), (320, 320), interpolation=cv2.INTER_NEAREST) > 127
def fr320(p):
    a = np.array(Image.open(p).convert("L"))
    return cv2.resize(pad_sq(a, 0), (320, 320), interpolation=cv2.INTER_LINEAR)
def predb(p): return np.array(Image.open(p).convert("L")) > 127
def dice(a, b):
    sa, sb = a.sum(), b.sum()
    return 1.0 if sa == 0 and sb == 0 else (0.0 if sa == 0 or sb == 0 else 2.0*(a & b).sum()/(sa+sb))

def gt_labels_topdown(gt):
    lbl, n = ndimage.label(gt, structure=CONN)
    order = sorted(range(1, n + 1), key=lambda k: np.where(lbl == k)[0].mean())
    gl = np.zeros_like(lbl)
    for new, old in enumerate(order, 1):
        gl[lbl == old] = new
    return gl, n

for key, gtrel, frel in CONFIGS:
    prd = os.path.join(PREDS, key, "seed_0")
    files = sorted(f for f in os.listdir(prd) if f.lower().endswith(".png"))
    pdf = PdfPages(os.path.join(OUT, f"{key}.pdf"))
    for f in files:
        gp = os.path.join(BASE, gtrel, f)
        if not os.path.isfile(gp): continue
        gt = gt320(gp); pr = predb(os.path.join(prd, f)); fr = fr320(os.path.join(BASE, frel, f))
        gl, n = gt_labels_topdown(gt)
        idx = ndimage.distance_transform_edt(gl == 0, return_distances=False, return_indices=True)
        nearest = gl[tuple(idx)]
        territory = ndimage.binary_dilation(gt, iterations=DILATE_PX)
        per = []
        for i in range(1, min(n, 3) + 1):
            per.append(dice(gl == i, pr & (nearest == i) & territory))
        extra_region = pr & ~territory
        n_extra = ndimage.label(extra_region, structure=CONN)[1]
        gd = dice(gt, pr)

        fig, ax = plt.subplots(1, 3, figsize=(13, 5))
        # 1) GT labeled
        ax[0].imshow(fr, cmap="gray")
        for i in range(1, n + 1):
            ax[0].contour(gl == i, [0.5], colors=[VCOL[(i-1) % len(VCOL)]], linewidths=1.4)
            ys, xs = np.where(gl == i)
            ax[0].text(xs.mean(), ys.mean(), LAB[i-1] if i <= len(LAB) else f"#{i}",
                       color="white", fontsize=10, ha="center", va="center", weight="bold")
        ax[0].set_title("GT — vertebras etiquetadas (por color)", fontsize=9); ax[0].set_axis_off()
        # 2) prediction colored per assigned vertebra (method B) + extra in gray
        ov = np.stack([fr]*3, -1).astype(float)/255
        for i in range(1, min(n, 6) + 1):
            reg = pr & (nearest == i) & territory
            ov[reg] = 0.35*ov[reg] + 0.65*np.array(VCOL[(i-1) % len(VCOL)])
        ov[extra_region] = [0.6, 0.6, 0.6]  # extra = gray
        ax[1].imshow(ov)
        ax[1].set_title("Prediccion PARTIDA por vertebra (metodo B)\ngris=extra", fontsize=9); ax[1].set_axis_off()
        # 3) numbers
        ax[2].axis("off")
        L = [f"{key}", f"{f}", "", "Dice POR VERTEBRA (metodo B):"]
        for i in range(min(3, n)):
            tag = "" if per[i] > 0 else "  <- no detectada"
            L.append(f"  {LAB[i]} = {per[i]:.3f}{tag}")
        L += ["", f"  promedio (missing-as-0) = {np.mean(per):.3f}" if per else "",
              "", f"Componentes GT={n}   EXTRA(fuera territorio)={n_extra}",
              "", f"Dice GLOBAL = {gd:.3f}"]
        ax[2].text(0.0, 0.98, "\n".join(L), va="top", ha="left", fontsize=11, family="monospace")
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    pdf.close()
    print("saved", os.path.join(OUT, f"{key}.pdf"), f"({len(files)} pages)")
print("DONE")
