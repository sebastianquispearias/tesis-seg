"""Measure, on the frame the figure shows, how visible the anchor shift actually is."""

import numpy as np
from PIL import Image
import cv2

ENT = "C:/Users/User/temp_inter_rater/ent"
GT = "C:/Users/User/temp_inter_rater/diag/unm/gt"
FRAME, SEED, MIN_AREA = "v081_f134", 0, 200
K_R = 2.0


def pad_sq(a, fill=0):
    im = Image.fromarray(a)
    w, h = im.size
    s = max(w, h)
    c = Image.new(im.mode, (s, s), fill)
    c.paste(im, ((s - w) // 2, (s - h) // 2))
    return np.array(c)


def to320(path):
    m = np.array(Image.open(path).convert("L"))
    m = np.where(m > 127, 255, 0).astype(np.uint8)
    return cv2.resize(pad_sq(m, 0), (320, 320), interpolation=cv2.INTER_NEAREST) > 127


def clean(mask, min_area=MIN_AREA):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area:
            out |= lab == i
    return out


def centroid(m):
    ys, xs = np.where(m)
    return xs.mean(), ys.mean()


gt = to320(f"{GT}/{FRAME}.png")
gx, gy = centroid(gt)
ys, _ = np.where(gt)
alto = ys.max() - ys.min() + 1
R = K_R * alto

print("frame:", FRAME)
print("  altura de la columna de referencia : %d px" % alto)
print("  R = %.1f x altura                  : %.0f px  (%.0f%% del frame)"
      % (K_R, R, 100 * R / 320))
print("  centroide de referencia            : (%.1f, %.1f)" % (gx, gy))
print()
print("%-14s %7s %9s %9s %11s" % ("panel", "Dice", "dx", "dy", "offset"))
for nombre, run in [("Supervised", "UNM__sup"), ("Mean Teacher", "UNM__mt")]:
    pred = clean(np.load(f"{ENT}/{run}/seed_{SEED}/{FRAME}.npy") >= 0.5)
    px, py = centroid(pred)
    off = float(np.hypot(px - gx, py - gy))
    dice = float(2 * (pred & gt).sum() / (pred.sum() + gt.sum()))
    print("%-14s %6.0f%% %+9.1f %+9.1f %8.1f px   = %.1f%% de R"
          % (nombre, 100 * dice, px - gx, py - gy, off, 100 * off / R))

print()
print("Para referencia, la media sobre los 63 frames del test es 6.09 px (sup) y")
print("4.62 px (MT). Este frame se eligio por ser un caso visible, no el promedio.")
