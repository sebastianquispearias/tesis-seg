"""Seleccion por incertidumbre: top-20% mas incierto vs azar. READ-ONLY.
Replica el item (5) de tesis_seg/analysis/entropia/ent_rework.py."""
import os, glob, numpy as np
from PIL import Image
from scipy import ndimage
import cv2

LOCAL, GTLOC = "C:/Users/User/temp_inter_rater/ent", "C:/Users/User/temp_inter_rater/diag"
EPS, BAND, FRAC, NRAND = 1e-7, 5, 0.20, 1000
CONFIGS = [("UNM", "Supervised", "UNM__sup", "unm/gt", [0,1,2,3,4]),
           ("UNM", "MT all-lat", "UNM__mt",  "unm/gt", [0,1,2]),
           ("INCA-p10", "Supervised", "INCApat10__sup", "inca/gt", [0,1,2]),
           ("INCA-p10", "MT r10",     "INCApat10__mt",  "inca/gt", [0,1,2])]

def pad_sq(a, fill=0):
    im = Image.fromarray(a); w, h = im.size; s = max(w, h)
    c = Image.new(im.mode, (s, s), fill); c.paste(im, ((s-w)//2, (s-h)//2)); return np.array(c)
def gt320(p):
    m = np.array(Image.open(p).convert("L")); m = np.where(m > 127, 255, 0).astype(np.uint8)
    return cv2.resize(pad_sq(m, 0), (320, 320), interpolation=cv2.INTER_NEAREST) > 127
def H_map(p):
    p = np.clip(p.astype(np.float32), EPS, 1-EPS); return -(p*np.log(p) + (1-p)*np.log(1-p))
def band_mask(gt, px):
    return ndimage.binary_dilation(gt, iterations=px) & ~ndimage.binary_erosion(gt, iterations=px)
def dice(a, b):
    sa, sb = a.sum(), b.sum()
    return 1.0 if sa==0 and sb==0 else (0.0 if sa==0 or sb==0 else 2.0*(a & b).sum()/(sa+sb))

rng = np.random.default_rng(0)
print(f"top-{int(FRAC*100)}% mas incierto (entropia de banda) vs {NRAND} subconjuntos al azar\n")
print(f"{'dataset/config':24s} {'n':>4s} {'k':>3s}   {'Dice top-inc':>14s}   {'Dice azar':>10s}  {'IC95 azar':>16s}  fuera?")
print("-"*94)
for ds, name, key, gtd, seeds in CONFIGS:
    tops, rms, los, his = [], [], [], []
    for s in seeds:
        ent, dcs = [], []
        for f in sorted(glob.glob(os.path.join(LOCAL, key, f"seed_{s}", "*.npy"))):
            stem = os.path.basename(f)[:-4]
            gp = os.path.join(GTLOC, gtd, stem + ".png")
            if not os.path.isfile(gp): continue
            p = np.load(f); gt = gt320(gp); bm = band_mask(gt, BAND)
            if not bm.any(): continue
            ent.append(float(H_map(p)[bm].mean())); dcs.append(dice(p >= 0.5, gt))
        e, d = np.array(ent), np.array(dcs); n = len(d)
        k = max(1, int(round(FRAC*n)))
        tops.append(d[np.argsort(-e)[:k]].mean())
        r = np.array([d[rng.choice(n, k, replace=False)].mean() for _ in range(NRAND)])
        rms.append(r.mean()); los.append(np.percentile(r, 2.5)); his.append(np.percentile(r, 97.5))
    t, rm = np.array(tops), np.array(rms)
    lo, hi = float(np.mean(los)), float(np.mean(his))
    fuera = "SI" if (t.mean() < lo or t.mean() > hi) else "no"
    print(f"{ds+'/'+name:24s} {n:4d} {k:3d}   {t.mean():.3f} +/- {t.std(ddof=1):.3f}   "
          f"{rm.mean():>10.3f}  [{lo:.3f}, {hi:.3f}]   {fuera}   (ancho {hi-lo:.3f})")
