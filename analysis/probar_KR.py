"""Render the region-of-interest figure at three window scales, side by side.

The scale is a free parameter: Lee's R is the input size of the network they feed the
crop to, not a measurement of anatomy, and their rule of running the box down to the
bottom of the image does not carry over to frames that were padded to a square. So the
only way to settle it is to look at the alternatives.

Writes to the scratchpad, never to figs/, so the figure in the manuscript is untouched.
"""

import importlib.util
import os
import shutil
import sys
from pathlib import Path

RAIZ = r"G:\My Drive\UNM_vertebras_seg_v3"
SCRIPT = os.path.join(RAIZ, "paper_figures", "fig_roi_lee.py")
SALIDA = (r"C:\Users\User\AppData\Local\Temp\claude\G--"
          r"\9356f667-f415-4b82-97bf-540838a4f8fd\scratchpad\roi")

os.makedirs(SALIDA, exist_ok=True)
sys.path.insert(0, os.path.join(RAIZ, "paper_figures"))

spec = importlib.util.spec_from_file_location("fig_roi_lee", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)          # el guard __main__ impide que dibuje al importar

mod.setup_matplotlib()
mod.OUT_DIRS = [SALIDA]

for k in (2.0, 1.5, 1.2):
    mod.K_R = k
    paths, R, stats = mod._plot()
    destino = os.path.join(SALIDA, "KR_%0.1f.pdf" % k)
    shutil.copy2(paths[0], destino)
    print("K_R = %.1f   R = %3.0f px  (%.0f%% del frame de 320)"
          % (k, R, 100 * R / 320))
    for nombre, off, dice in stats:
        print("     %-13s offset %5.2f px = %4.1f%% del ancho de la ventana"
              % (nombre, off, 100 * off / R))
    print("     ->", destino)
    print()
