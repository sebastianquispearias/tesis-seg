"""Say why the supervised figure is optimistic instead of asserting that it is.

Nine predictions had no component left after the cleaning step and so produced no
centre at all. They are therefore missing from the average reported two sentences
earlier, and they are the worst cases, which is the whole reason the average flatters
the supervised model. The sentence stated the conclusion and left the reader to
reconstruct that step.
"""

import io
import os
import re
import shutil
import sys

RUTA = r"G:\My Drive\UNM_vertebras_seg_v3\tesis_latex_overleaf\tesis_final_v54.tex"
COPIA = RUTA + ".ANTES_optimista"

APLICAR = "--aplicar" in sys.argv

ANTES = ("kept no component above 200~px and could not be anchored at all, so the "
         "supervised value is optimistic; Mean Teacher produced a usable mask in "
         "every frame.")
DESPUES = ("kept no component above that threshold and could not be anchored, so those "
           "nine are absent from the 6.09~px above. Mean Teacher produced a usable "
           "mask in every frame.")

with io.open(RUTA, "r", encoding="utf-8", newline="") as fh:
    original = fh.read()
if original.count("\r"):
    sys.exit("ABORTO: retornos de carro.")
if original.count(ANTES) != 1:
    sys.exit("ABORTO: el ancla aparece %d veces, no 1." % original.count(ANTES))

texto = original.replace(ANTES, DESPUES, 1)


def legible(s):
    s = re.sub(r"\\ref\{([^}]*)\}", r"[\1]", s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", s)
    s = s.replace("$\\pm$", "+/-").replace("~", " ")
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    return " ".join(s.replace("{", "").replace("}", "").replace("$", "").split())


i = texto.index("On the UNM test set")
par = texto[i:texto.index("\n", i)]
print("=" * 92)
print("EL PARRAFO B, DE CORRIDO")
print("=" * 92)
t = legible(par)
for k in range(0, len(t), 90):
    print("  " + t[k:k + 90])
print()
print("comprobaciones:")
print("  'is optimistic' que quedan :", texto.count("is optimistic"))
print("  ';' en el parrafo          :", legible(par).count(";"))
print("  '200 px' en el parrafo     :", legible(par).count("200 px"))
print()

if not APLICAR:
    print("MODO LECTURA. Anadir --aplicar para escribir.")
    sys.exit(0)

if not os.path.exists(COPIA):
    shutil.copy2(RUTA, COPIA)
with io.open(RUTA, "w", encoding="utf-8", newline="") as fh:
    fh.write(texto)
with io.open(RUTA, "r", encoding="utf-8", newline="") as fh:
    e = fh.read()
print("copia de seguridad :", os.path.basename(COPIA))
print("CR sueltos         :", e.count("\r"))
print("bytes              :", len(original), "->", len(e),
      "(%+d)" % (len(e) - len(original)))
