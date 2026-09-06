"""Tighten the illustrative window of the region-of-interest figure.

The scale was free: Lee's R is the input size of the network the crop is fed to rather
than a measurement of anatomy, and their rule of running the box down to the bottom of
the frame does not survive the padding to a square. At twice the column height almost
nothing was left outside the window, so it read as a border rather than a selection;
at one and a half times it still contains the three vertebrae with margin and leaves
the jaw and the lower neck visible outside. At one and two tenths it clipped C4.

None of the measured numbers change. The anchor still moves 14.7 px under the
supervised model and 2.3 px under Mean Teacher; the tighter window only makes that
same shift a tenth of its width instead of a thirteenth.
"""

import io
import os
import re
import shutil
import subprocess
import sys

RAIZ = r"G:\My Drive\UNM_vertebras_seg_v3"
SCRIPT = os.path.join(RAIZ, "paper_figures", "fig_roi_lee.py")
TEX = os.path.join(RAIZ, "tesis_latex_overleaf", "tesis_final_v54.tex")

# ------------------------------------------------------------------ 1. el script
with io.open(SCRIPT, "r", encoding="utf-8", newline="") as fh:
    s = fh.read()
A = "K_R = 2.0                       # ventana = K_R x altura de la columna de referencia"
B = "K_R = 1.5                       # ventana = K_R x altura de la columna de referencia"
if s.count(A) != 1:
    sys.exit("ABORTO: el ancla de K_R aparece %d veces, no 1." % s.count(A))
if not os.path.exists(SCRIPT + ".ANTES_KR15"):
    shutil.copy2(SCRIPT, SCRIPT + ".ANTES_KR15")
with io.open(SCRIPT, "w", encoding="utf-8", newline="") as fh:
    fh.write(s.replace(A, B, 1))
print("K_R 2.0 -> 1.5 en el script")

# ---------------------------------------------------------------- 2. la figura
r = subprocess.run([sys.executable, SCRIPT], capture_output=True, text=True,
                   cwd=os.path.join(RAIZ, "paper_figures"))
print(r.stdout.strip() or r.stderr.strip()[-700:])

# ------------------------------------------------------------------ 3. el pie
with io.open(TEX, "r", encoding="utf-8", newline="") as fh:
    t = fh.read()
if t.count("\r"):
    sys.exit("ABORTO: el .tex tiene retornos de carro.")
A2 = "The scale $R$ is illustrative, twice the height of the reference column here"
B2 = ("The scale $R$ is illustrative, one and a half times the height of the "
      "reference column here")
if t.count(A2) != 1:
    sys.exit("ABORTO: el ancla del pie aparece %d veces, no 1." % t.count(A2))
if not os.path.exists(TEX + ".ANTES_KR15"):
    shutil.copy2(TEX, TEX + ".ANTES_KR15")
with io.open(TEX, "w", encoding="utf-8", newline="") as fh:
    fh.write(t.replace(A2, B2, 1))
with io.open(TEX, "r", encoding="utf-8", newline="") as fh:
    e = fh.read()
print()
print("pie del .tex actualizado")
print("  CR sueltos :", e.count("\r"))
print("  'twice the height' que quedan :", e.count("twice the height"))
print("  'one and a half times'        :", e.count("one and a half times"))
