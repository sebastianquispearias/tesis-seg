"""Split the region-of-interest paragraph and repair its two small stumbles.

At two hundred and eleven words it was the longest paragraph of the chapter, eight
declarative sentences with no marker at any of its turns. The Limitations paragraph is
longer still and reads well, but it carries seven ordinals; this one carried nothing.
The break falls where the reasoning ends and the measurement begins, so the paragraph
break itself does the signposting, which is cheaper than adding headings inside a
block that already has three.

The window also arrived with a definite article although only the region of interest
had been named, and the colon between resizing and moving was doing nothing a comma
could not do.
"""

import io
import os
import re
import shutil
import sys

RUTA = r"G:\My Drive\UNM_vertebras_seg_v3\tesis_latex_overleaf\tesis_final_v54.tex"
COPIA = RUTA + ".ANTES_partir_roi"

APLICAR = "--aplicar" in sys.argv

CAMBIOS = [
    ("el articulo del window",
     "The window has a fixed size",
     "That window has a fixed size"),
    ("los dos puntos",
     "the segmentation does not resize it: it only moves it.",
     "the segmentation does not resize it, only moves it."),
    ("el punto y aparte",
     "only moves it. On the UNM test set",
     "only moves it.\n\nOn the UNM test set"),
]

with io.open(RUTA, "r", encoding="utf-8", newline="") as fh:
    original = fh.read()
if original.count("\r"):
    sys.exit("ABORTO: retornos de carro.")

texto = original
for etiqueta, a, b in CAMBIOS:
    if texto.count(a) != 1:
        sys.exit("ABORTO: el ancla de '%s' aparece %d veces, no 1."
                 % (etiqueta, texto.count(a)))
    texto = texto.replace(a, b, 1)
    print("  " + etiqueta)
print()


def palabras(s):
    s = re.sub(r"\\citep\{[^}]*\}|\\citet\{[^}]*\}", "", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    return len(re.sub(r"[{}$~\\^]", " ", s).split())


def legible(s):
    s = re.sub(r"\\citet\{ref_lee2020\}", "Lee et al.", s)
    s = re.sub(r"\\ref\{([^}]*)\}", r"[\1]", s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", s)
    s = s.replace("$\\pm$", "+/-").replace("~", " ")
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    return " ".join(s.replace("{", "").replace("}", "").replace("$", "").split())


i = texto.index("\\textbf{Placement of a region")
j = texto.index("\\begin{figure}", i)
bloque = texto[i:j].strip()
pars = [p.strip() for p in bloque.split("\n\n") if p.strip()]

print("=" * 92)
print("COMO QUEDA")
print("=" * 92)
for k, p in enumerate(pars, 1):
    print("--- PARRAFO %s  (%d palabras)" % (chr(64 + k), palabras(p)))
    txt = legible(p)
    for n in range(0, len(txt), 90):
        print("    " + txt[n:n + 90])
    print()

print("comprobaciones:")
print("  'The window' que quedan  :", texto.count("The window"))
print("  'That window'            :", texto.count("That window"))
print("  ':' en el bloque         :", legible(bloque).count(":"))
print("  parrafos                 :", len(pars))
print()

if not APLICAR:
    print("MODO LECTURA. No se escribio nada. Anadir --aplicar para escribir.")
    sys.exit(0)

if not os.path.exists(COPIA):
    shutil.copy2(RUTA, COPIA)
with io.open(RUTA, "w", encoding="utf-8", newline="") as fh:
    fh.write(texto)
with io.open(RUTA, "r", encoding="utf-8", newline="") as fh:
    escrito = fh.read()
print("copia de seguridad :", os.path.basename(COPIA))
print("CR sueltos         :", escrito.count("\r"))
print("bytes              :", len(original), "->", len(escrito),
      "(%+d)" % (len(escrito) - len(original)))
