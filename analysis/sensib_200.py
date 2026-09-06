"""Is the 200 px cleaning threshold doing any work, or would any value do?"""

import glob
import os
import numpy as np
from PIL import Image
import cv2

ENT = "C:/Users/User/temp_inter_rater/ent"
GT = "C:/Users/User/temp_inter_rater/diag/unm/gt"


def pad_sq(a, fill=0):
    im = Image.fromarray(a)
    w, h = im.size
    s = max(w, h)
    c = Image.new(im.mode, (s, s), fill)
    c.paste(im, ((s - w) // 2, (s - h) // 2))
    return np.array(c)


def to320(p):
    m = np.array(Image.open(p).convert("L"))
    m = np.where(m > 127, 255, 0).astype(np.uint8)
    return cv2.resize(pad_sq(m, 0), (320, 320), interpolation=cv2.INTER_NEAREST) > 127


def clean(m, ma):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= ma:
            out |= lab == i
    return out


gts = {os.path.basename(p)[:-4]: to320(p) for p in glob.glob(f"{GT}/*.png")}
print("frames de referencia:", len(gts))

print()
print("%5s | %-28s | %-28s" % ("umbral", "Supervised", "Mean Teacher"))
print("%5s | %8s %8s %9s | %8s %8s %9s"
      % ("", "sin ancla", "de", "offset", "sin ancla", "de", "offset"))
for ma in (50, 100, 150, 200, 250, 300):
    fila = []
    for run in ("UNM__sup", "UNM__mt"):
        sin, tot, offs = 0, 0, []
        for sd in sorted(glob.glob(f"{ENT}/{run}/seed_*")):
            for f in glob.glob(f"{sd}/*.npy"):
                stem = os.path.basename(f)[:-4]
                if stem not in gts:
                    continue
                tot += 1
                pr = clean(np.load(f) >= 0.5, ma)
                if not pr.any():
                    sin += 1
                    continue
                ys, xs = np.where(pr)
                gy, gx = np.where(gts[stem])
                offs.append(np.hypot(xs.mean() - gx.mean(), ys.mean() - gy.mean()))
        fila.append((sin, tot, float(np.mean(offs))))
    print("%5d | %8d %8d %8.2f | %8d %8d %8.2f"
          % (ma, fila[0][0], fila[0][1], fila[0][2],
             fila[1][0], fila[1][1], fila[1][2]))
