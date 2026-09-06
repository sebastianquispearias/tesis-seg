"""Entropia de banda vs tamano del pool no etiquetado. READ-ONLY.
Formulas identicas a tesis_seg/analysis/entropia/ent_rework.py."""
import os, glob, numpy as np
from PIL import Image
from scipy import ndimage
import cv2

RUNS = "G:/My Drive/UNM_vertebras_seg_v3/runs_final_v1"
GT   = "C:/Users/User/temp_inter_rater/diag/unm/gt"
EPS, BAND = 1e-7, 5

# (etiqueta, carpeta, frames no etiquetados)  -- tamanos de CLAUDE.md
CONFIGS = [("Supervised",  "supervised",               0),
           ("MT r=3",      "mean_teacher_r3",       1257),
           ("MT r=5",      "mean_teacher_r5",       2066),
           ("MT r=7",      "mean_teacher_r7",       2849),
           ("MT r=10",     "mean_teacher_r10",      3937),
           ("MT r=15",     "mean_teacher_r15",      5590),
           ("MT r=20",     "mean_teacher_r20",      7081),
           ("MT all-lat",  "mean_teacher_all_lateral", 74774)]

def pad_sq(a, fill=0):
    im = Image.fromarray(a); w, h = im.size; s = max(w, h)
    c = Image.new(im.mode, (s, s), fill); c.paste(im, ((s-w)//2, (s-h)//2))
    return np.array(c)

def gt320(p):
    m = np.array(Image.open(p).convert("L"))
    m = np.where(m > 127, 255, 0).astype(np.uint8)
    return cv2.resize(pad_sq(m, 0), (320, 320), interpolation=cv2.INTER_NEAREST) > 127

def H_map(p):                                   # entropia binaria, log natural
    p = np.clip(p.astype(np.float32), EPS, 1 - EPS)
    return -(p*np.log(p) + (1-p)*np.log(1-p))

def band_mask(gt, px):                          # IDENTICA a ent_rework.py (L1, no euclidea)
    return ndimage.binary_dilation(gt, iterations=px) & ~ndimage.binary_erosion(gt, iterations=px)

gts = {os.path.basename(p)[:-4]: gt320(p) for p in glob.glob(f"{GT}/*.png")}
bands = {k: band_mask(v, BAND) for k, v in gts.items()}
print(f"GT cargados: {len(gts)}   banda +/-{BAND} px\n")
print(f"{'config':12s} {'pool':>7s} {'semillas':>9s} {'H banda (media +/- sd entre semillas)':>40s}")
print("-"*74)

filas = []
for lab, run, pool in CONFIGS:
    per_seed = []
    for sd in sorted(glob.glob(f"{RUNS}/{run}/seed_*")):
        vals = []
        for f in glob.glob(f"{sd}/test_probs/*.npy"):
            stem = os.path.basename(f)[:-4]
            if stem not in bands:
                continue
            H = H_map(np.load(f))
            b = bands[stem]
            if b.sum():
                vals.append(float(H[b].mean()))
        if vals:
            per_seed.append(np.mean(vals))
    a = np.array(per_seed)
    sd_txt = f"{a.std():.4f}" if len(a) > 1 else "  -  "
    print(f"{lab:12s} {pool:7d} {len(a):9d}      {a.mean():.4f} +/- {sd_txt}")
    filas.append((lab, pool, len(a), a.mean(), a.std() if len(a) > 1 else np.nan))

np.save("ent_pool_filas.npy", np.array(filas, dtype=object), allow_pickle=True)
