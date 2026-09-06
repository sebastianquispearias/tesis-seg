"""Score the same predictions on the 320 grid and on the source grid.

The pipeline pads each frame to a square by centring it and then resizes to 320.
Inverting that means resizing back to the padded side and cropping the centred
window, so the prediction lands where the annotation actually is.
"""
import glob
import os

import cv2
import numpy as np
from PIL import Image

BASE = r"G:\My Drive\UNM_vertebras_seg_v3"
PRED = os.path.join(BASE, "runs_nnunet_ablation", "A_baseline_fixnoise",
                    "seed_0", "test_preds")
MASK = os.path.join(BASE, "test", "masks")


def dice(a, b):
    a, b = a > 0, b > 0
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * (a & b).sum() / s


en320, nativa = [], []
for p in sorted(glob.glob(os.path.join(PRED, "*.png"))):
    rm = os.path.join(MASK, os.path.basename(p))
    if not os.path.isfile(rm):
        continue
    pred320 = np.array(Image.open(p).convert("L"))
    orig = np.array(Image.open(rm).convert("L"))
    h, w = orig.shape
    lado = max(h, w)
    x0, y0 = (lado - w) // 2, (lado - h) // 2

    lienzo = np.zeros((lado, lado), np.uint8)
    lienzo[y0:y0 + h, x0:x0 + w] = orig
    m320 = cv2.resize(lienzo, (320, 320), interpolation=cv2.INTER_NEAREST)
    en320.append(dice(pred320, m320))

    pn = cv2.resize(pred320, (lado, lado), interpolation=cv2.INTER_NEAREST)
    nativa.append(dice(pn[y0:y0 + h, x0:x0 + w], orig))

a, b = float(np.mean(en320)), float(np.mean(nativa))
print("fotogramas:", len(en320))
print("  Dice a 320x320, las dos mascaras encogidas : {:.4f}".format(a))
print("  Dice en la rejilla original de cada frame  : {:.4f}".format(b))
print("  diferencia                                 : {:+.4f}".format(b - a))
d = np.abs(np.array(en320) - np.array(nativa))
i = int(np.argmax(d))
print("  el frame donde mas cambia: {:.4f} -> {:.4f}  ({:+.4f})".format(
    en320[i], nativa[i], nativa[i] - en320[i]))
