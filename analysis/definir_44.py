"""Define, where they are used, the three things section 4.4 relies on.

The scalar was named in the introduction as a normalization reference and in the
appendix as something based on reference points, but never as a distance between two
corners, which is what makes the rest of the paragraph follow. The rule that finds
those corners was referred to with a definite article without ever having been
described, so the claim that it, and not the segmentation, is the bottleneck had
nothing to stand on. The pixel threshold appeared twice with no reason attached.

The section already carries its own short method notes, so these go here rather than
back in the introduction.
"""

import io
import os
import re
import shutil
import sys

RUTA = r"G:\My Drive\UNM_vertebras_seg_v3\tesis_latex_overleaf\tesis_final_v54.tex"
COPIA = RUTA + ".ANTES_definir_44"

APLICAR = "--aplicar" in sys.argv

CAMBIOS = [
    ("el escalar",
     "C4 is one of the two vertebrae the C2--C4 anatomical scalar depends "
     "on~\\citep{ref_molfenter}.",
     "The C2--C4 anatomical scalar~\\citep{ref_molfenter} measures the distance between "
     "a corner of C2 and a corner of C4, so C4 is one of the two vertebrae it depends "
     "on."),
    ("la regla",
     "The reason is the rule that locates the two corner landmarks.",
     "The reason is the rule that locates those corners, which splits the mask at the "
     "gaps between vertebrae."),
    ("el umbral",
     "after discarding connected components smaller than 200~px",
     "after discarding stray components smaller than 200~px"),
]

with io.open(RUTA, "r", encoding="utf-8", newline="") as fh:
    original = fh.read()
if original.count("\r"):
    sys.exit("ABORTO: retornos de carro.")

texto = original
for etiqueta, a, b in CAMBIOS:
    if texto.count(a) != 1:
        sys.exit("ABORTO: el ancla de %s aparece %d veces, no 1."
                 % (etiqueta, texto.count(a)))
    texto = texto.replace(a, b, 1)
    print("%-12s  %d -> %d palabras" % (etiqueta, len(a.split()), len(b.split())))

print()


def palabras(s):
    s = re.sub(r"\\citep\{[^}]*\}", "", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = re.sub(r"[{}$~\\^]", " ", s)
    return len(s.split())


def legible(s):
    s = re.sub(r"\\citep\{([^}]*)\}", r"[\1]", s)
    s = re.sub(r"\\ref\{([^}]*)\}", r"[\1]", s)
    s = s.replace("~=~", " = ").replace("$\\pm$", "+/-").replace("{,}", ",")
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    return " ".join(s.replace("{", "").replace("}", "").replace("$", "")
                    .replace("~", " ").split())


for marca in ["The C2--C4 anatomical scalar", "Losing a vertebra matters"]:
    i = texto.index(marca)
    par = texto[i:texto.index("\n", i)]
    print("=" * 92)
    print("PARRAFO QUE EMPIEZA EN: %s...  (%d palabras)" % (marca[:38], palabras(par)))
    print("=" * 92)
    for f in re.split(r"(?<=\.) (?=[A-Z*])", par):
        txt = legible(f)
        if txt:
            print("  (%2d) %s" % (palabras(f), txt[:86]))
            for k in range(86, len(txt), 86):
                print("       " + txt[k:k + 86])
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
