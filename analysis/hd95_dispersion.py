"""Give the HD95 figures of 4.4 the dispersion the F1 figures beside them already had.

The sentence reported F1 as a range and HD95 as three point values, and called the
change a doubling. Read from the run reports, one of those three rests on a single
seed and another has a seed below the supervised baseline it is being contrasted with,
so the doubling is not something three runs can carry. Stating each mean with its
standard deviation lets the reader see the overlap without being told about it, and
leaves the paragraph's actual point, that F1 alone would have shown nothing, untouched.
"""

import io
import os
import re
import shutil
import sys

RUTA = r"G:\My Drive\UNM_vertebras_seg_v3\tesis_latex_overleaf\tesis_final_v54.tex"
COPIA = RUTA + ".ANTES_hd95_dispersion"

APLICAR = "--aplicar" in sys.argv

ANTES = ("but HD95 roughly doubled: 29.0~px at $r{=}20$ and 26.5~px with the "
         "all-lateral pool, against 14.0~px for the supervised baseline.")
DESPUES = ("while HD95 rose to 29.0~$\\pm$~13.1~px at $r{=}20$ and "
           "26.5~$\\pm$~22.2~px with the all-lateral pool, against "
           "14.0~$\\pm$~7.1~px for the supervised baseline.")

with io.open(RUTA, "r", encoding="utf-8", newline="") as fh:
    original = fh.read()
if original.count("\r"):
    sys.exit("ABORTO: retornos de carro.")
if original.count(ANTES) != 1:
    sys.exit("ABORTO: el ancla aparece %d veces, no 1." % original.count(ANTES))

texto = original.replace(ANTES, DESPUES, 1)


def palabras(s):
    s = re.sub(r"\\citep\{[^}]*\}", "", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = re.sub(r"[{}$~\\^]", " ", s)
    return len(s.split())


def legible(s):
    s = re.sub(r"\\ref\{([^}]*)\}", r"[\1]", s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", s)
    s = s.replace("~=~", " = ").replace("$\\pm$", "+/-").replace("{,}", ",")
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    return " ".join(s.replace("{", "").replace("}", "").replace("$", "")
                    .replace("~", " ").split())


i = texto.index("**Boundary metrics" if "**Boundary" in texto
                else "\\textbf{Boundary metrics.}")
par = texto[i:texto.index("\n", i)]

print("=" * 92)
print("EL PARRAFO, DE CORRIDO  (%d palabras)" % palabras(par))
print("=" * 92)
seguido = legible(par)
for k in range(0, len(seguido), 90):
    print("  " + seguido[k:k + 90])
print()
print("comprobaciones:")
print("  queda 'roughly doubled' :", "roughly doubled" in texto)
print("  dos puntos en el parrafo:", legible(par).count(":"))
print("  los tres HD95 llevan +/-:",
      all(x in par for x in ["29.0~$\\pm$~13.1", "26.5~$\\pm$~22.2", "14.0~$\\pm$~7.1"]))
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
