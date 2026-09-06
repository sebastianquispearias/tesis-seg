"""Utilidad como ROI: la caja de la prediccion, ¿contiene la region de referencia?
READ-ONLY sobre test_probs/*.npy y el GT local. Sin entrenamientos."""
import os, glob, numpy as np
from PIL import Image
import cv2

RUNS = "G:/My Drive/UNM_vertebras_seg_v3/runs_final_v1"
GT   = "C:/Users/User/temp_inter_rater/diag/unm/gt"
MARGINS = [0, 5, 10, 20]          # px de holgura sobre la caja predicha
CONFIGS = [("Supervised", "supervised"),
           ("MT all-lateral", "mean_teacher_all_lateral")]

def pad_sq(a, fill=0):
    im = Image.fromarray(a); w, h = im.size; s = max(w, h)
    c = Image.new(im.mode, (s, s), fill); c.paste(im, ((s-w)//2, (s-h)//2)); return np.array(c)
def gt320(p):
    m = np.array(Image.open(p).convert("L")); m = np.where(m > 127, 255, 0).astype(np.uint8)
    return cv2.resize(pad_sq(m, 0), (320, 320), interpolation=cv2.INTER_NEAREST) > 127

gts = {os.path.basename(p)[:-4]: gt320(p) for p in glob.glob(f"{GT}/*.png")}
print(f"frames de test con GT: {len(gts)}\n")
print(f"{'config':16s} {'margen':>7s}  {'ROI completo':>13s}  {'cobertura media':>16s}  {'peor frame':>11s}")
print("-"*74)

for name, run in CONFIGS:
    for m in MARGINS:
        per_seed_full, per_seed_cov, per_seed_worst = [], [], []
        for sd in sorted(glob.glob(f"{RUNS}/{run}/seed_*")):
            full, cov = [], []
            for f in glob.glob(f"{sd}/test_probs/*.npy"):
                stem = os.path.basename(f)[:-4]
                if stem not in gts: continue
                pred = np.load(f) >= 0.5
                gt = gts[stem]
                if not pred.any():                    # sin prediccion -> ROI imposible
                    full.append(0.0); cov.append(0.0); continue
                ys, xs = np.where(pred)
                y0 = max(ys.min()-m, 0); y1 = min(ys.max()+1+m, 320)
                x0 = max(xs.min()-m, 0); x1 = min(xs.max()+1+m, 320)
                box = np.zeros_like(gt); box[y0:y1, x0:x1] = True
                c = (gt & box).sum() / gt.sum()
                cov.append(c); full.append(1.0 if c >= 0.999 else 0.0)
            per_seed_full.append(np.mean(full)*100)
            per_seed_cov.append(np.mean(cov)*100)
            per_seed_worst.append(np.min(cov)*100)
        F, C, W = map(np.array, (per_seed_full, per_seed_cov, per_seed_worst))
        print(f"{name:16s} {m:5d}px  {F.mean():6.1f}% +/-{F.std(ddof=1):4.1f}  "
              f"{C.mean():7.2f}% +/-{C.std(ddof=1):5.2f}  {W.mean():9.1f}%")
    print()
