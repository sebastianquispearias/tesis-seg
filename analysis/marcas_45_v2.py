"""Cut section 4.5 from five bold markers to three.

The opening paragraph loses its marker. It is the first thing in the section, so its
position already makes the definition findable, and a heading there mainly advertised
that the definition had been added on request. Section 4.2 already opens the same way,
with an unmarked lead-in followed by the marked analyses.

The markers over semi-supervised training and pool size become one. The prose already
joins those two paragraphs, the second of them opening on "either", so two headings
were splitting what the text was tying together. Merging them also removes the reason
the pool-size heading had to say "on entropy": it no longer collides with the heading
of the same name in 4.2.

No word of the running text changes.
"""

import io
import os
import re
import shutil
import sys

RUTA = r"G:\My Drive\UNM_vertebras_seg_v3\tesis_latex_overleaf\tesis_final_v54.tex"
COPIA = RUTA + ".ANTES_marcas_45_v2"

APLICAR = "--aplicar" in sys.argv

CAMBIOS = [
    ("quitar la marca de P1",
     "\\textbf{Definition of band entropy.} Uncertainty is measured per pixel",
     "Uncertainty is measured per pixel"),
    ("renombrar la marca de P3",
     "\\textbf{Effect of semi-supervised training.} Semi-supervised training did not",
     "\\textbf{Effect of semi-supervised training and pool size.} Semi-supervised "
     "training did not"),
    ("quitar la marca de P4",
     "\\textbf{Effect of pool size on entropy.} Pool size does not change it either",
     "Pool size does not change it either"),
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

# el texto corrido, sin las marcas, tiene que ser identico
def sin_marcas(s):
    return re.sub(r"\\textbf\{[^}]*\}\s*", "", s)


i, j = texto.index("\\section{Predictive uncertainty}"), \
    texto.index("\\section{Generality across SSL methods}")
a0, b0 = original.index("\\section{Predictive uncertainty}"), \
    original.index("\\section{Generality across SSL methods}")
print("el texto corrido de 4.5 es identico quitando las marcas:",
      sin_marcas(texto[i:j]) == sin_marcas(original[a0:b0]))

marcas = re.findall(r"\\textbf\{([^}]*)\}", texto[i:j])
print("marcas antes :", len(re.findall(r"\\textbf\{([^}]*)\}", original[a0:b0])))
print("marcas ahora :", len(marcas))
for m in marcas:
    print("    " + m)


def palabras(s):
    s = re.sub(r"\\citep\{[^}]*\}", "", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    return len(re.sub(r"[{}$~\\^]", " ", s).split())


sec = texto[i:j]
pars = [p.strip() for p in sec.split("\n\n")
        if p.strip() and palabras(p) > 20 and not p.strip().startswith("\\begin")]
print()
print("parrafos de 4.5 y su marca:")
for k, p in enumerate(pars, 1):
    m = re.match(r"\\textbf\{([^}]*)\}", p)
    print("  P%d (%3d palabras)  %s" % (k, palabras(p),
                                        m.group(1) if m else "(sin marca)"))
print("  total: %d palabras / %d marcas = una cada %d"
      % (palabras(sec), len(marcas), palabras(sec) // max(1, len(marcas))))

if not APLICAR:
    print()
    print("MODO LECTURA. No se escribio nada. Anadir --aplicar para escribir.")
    sys.exit(0)

if not os.path.exists(COPIA):
    shutil.copy2(RUTA, COPIA)
with io.open(RUTA, "w", encoding="utf-8", newline="") as fh:
    fh.write(texto)
with io.open(RUTA, "r", encoding="utf-8", newline="") as fh:
    escrito = fh.read()
print()
print("copia de seguridad :", os.path.basename(COPIA))
print("CR sueltos         :", escrito.count("\r"))
print("bytes              :", len(original), "->", len(escrito),
      "(%+d)" % (len(escrito) - len(original)))
