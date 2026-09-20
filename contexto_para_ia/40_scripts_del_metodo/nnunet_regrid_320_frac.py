"""
nnunet_regrid_320.py — Recompute nnU-Net main-run metrics on the common 320x320
grid, using the SAME pipeline boundary functions and the SAME pad_to_square+resize
(NEAREST) as the U-Net++ pipeline, applied identically to prediction AND GT.
READ-ONLY on originals. Writes NEW files only (keeps native results intact).

Metrics: F1(Dice), ASSD, HD95, BF1@5px — computed with identical code at NATIVE
and at 320 so the only difference is the grid.
Outputs:
  nnunet_baseline_results/nnunet_regrid320_per_image.csv
  nnunet_baseline_results/nnunet_regrid320_summary.csv
"""
import os, sys, csv, glob
import numpy as np
from PIL import Image
import cv2

sys.path.insert(0, r"G:/My Drive/UNM_vertebras_seg_v3/tesis_seg")
from src.metrics import boundary_f1_score, assd_hd95   # exact pipeline definitions

ROOT = r"G:/My Drive/UNM_vertebras_seg_v3"
import sys as _s; FRAC=_s.argv[1]; PRED = os.path.join(ROOT, "nnunet_baseline_results", f"predictions_{FRAC}pct")
GTD = os.path.join(ROOT, "test", "masks")
OUT = os.path.join(ROOT, "nnunet_baseline_results")
R_TOL = 5  # BF1 tolerance in px, as requested

def pad_to_square(arr, fill=0):
    img = Image.fromarray(arr); w, h = img.size; side = max(w, h)
    canvas = Image.new(img.mode, (side, side), color=fill)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return np.array(canvas)

def to320(mask_bool):
    m = (mask_bool.astype(np.uint8) * 255)
    m = pad_to_square(m, 0)
    m = cv2.resize(m, (320, 320), interpolation=cv2.INTER_NEAREST)
    return m > 127

def dice(a, b):
    sa, sb = a.sum(), b.sum()
    return 1.0 if sa == 0 and sb == 0 else (0.0 if sa == 0 or sb == 0 else 2.0*(a & b).sum()/(sa+sb))

def metrics(pred, gt):
    f1 = dice(pred, gt)
    assd, hd95 = assd_hd95(pred, gt)
    bf1 = boundary_f1_score(pred, gt, r_tol_px=R_TOL)
    return f1, assd, hd95, bf1

rows = []
preds = sorted(glob.glob(os.path.join(PRED, "*.png")))
matched = 0
for pf in preds:
    stem = os.path.basename(pf)
    gf = os.path.join(GTD, stem)
    if not os.path.isfile(gf):
        print("NO GT for", stem); continue
    pr = np.array(Image.open(pf).convert("L")) > 0
    gt = np.array(Image.open(gf).convert("L")) > 127
    if pr.shape != gt.shape:
        print("SHAPE MISMATCH", stem, pr.shape, gt.shape); continue
    matched += 1
    # native grid
    f1n, assdn, hdn, bfn = metrics(pr, gt)
    # common 320 grid (same pad+resize NEAREST for both pred and GT)
    pr3, gt3 = to320(pr), to320(gt)
    f13, assd3, hd3, bf3 = metrics(pr3, gt3)
    rows.append({"stem": stem.replace(".png", ""),
                 "native_shape": f"{pr.shape[0]}x{pr.shape[1]}",
                 "f1_native": round(f1n, 4), "assd_native_px": round(assdn, 4),
                 "hd95_native_px": round(hdn, 4), "bf1_5_native": round(bfn, 4),
                 "f1_320": round(f13, 4), "assd_320_px": round(assd3, 4),
                 "hd95_320_px": round(hd3, 4), "bf1_5_320": round(bf3, 4)})

print(f"\nMatched pairs: {matched} / {len(preds)} predictions")
with open(os.path.join(OUT, f"nnunet_regrid320_{FRAC}pct_per_image.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def agg(k):
    v = [r[k] for r in rows]; return float(np.mean(v)), float(np.std(v))
summary = []
for grid, keys in [("native", ["f1_native", "assd_native_px", "hd95_native_px", "bf1_5_native"]),
                   ("320", ["f1_320", "assd_320_px", "hd95_320_px", "bf1_5_320"])]:
    m = {"grid": grid, "n": len(rows)}
    for k in keys:
        mean, std = agg(k)
        base = k.replace("_native", "").replace("_320", "")
        m[base] = f"{mean:.4f}"; m[base + "_std"] = f"{std:.4f}"
    summary.append(m)
with open(os.path.join(OUT, f"nnunet_regrid320_{FRAC}pct_summary.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

print("\n===== nnU-Net: NATIVO vs 320 (código idéntico, solo cambia la rejilla) =====")
print(f"{'metric':12s} {'NATIVO':>18s} {'320x320':>18s}")
for base, unit in [("f1", ""), ("assd", " px"), ("hd95", " px"), ("bf1_5", "")]:
    n = agg(base + "_native" if base != "bf1_5" else "bf1_5_native")
    t = agg(base + "_320" if base != "bf1_5" else "bf1_5_320")
    print(f"{base:12s} {n[0]:8.4f}±{n[1]:.4f}   {t[0]:8.4f}±{t[1]:.4f}{unit}")
print("\nsaved nnunet_regrid320_per_image.csv + nnunet_regrid320_summary.csv")
print("DONE")
