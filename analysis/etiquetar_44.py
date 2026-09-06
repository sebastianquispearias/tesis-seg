"""Name the configuration behind each set of numbers in 4.4, and close the scalar thread.

The paragraph reported three sets of per-vertebra numbers drawn from three different
runs and named only one of them. The sentence that did name its run, Mean Teacher at
r=10, sat immediately after one that did not, so a reader carried the first set of
gains over to the named configuration; those gains belong to the all-lateral pool and
are not the ones r=10 produced. The INCA values are Mean Teacher at r=15, and the
contrast they are there to draw only works once that is said.

The paragraph also ended where the scalar stopped improving, with nothing said about
where the remaining room is. The manuscript already argues that some downstream tasks
need masks rather than landmark coordinates, and already cites work that predicts the
corners directly, so pointing at the rule closes the thread without contradicting that
choice.
"""

import io
import os
import re
import shutil
import sys

RUTA = r"G:\My Drive\UNM_vertebras_seg_v3\tesis_latex_overleaf\tesis_final_v54.tex"
COPIA = RUTA + ".ANTES_etiquetar_44"

APLICAR = "--aplicar" in sys.argv

CAMBIOS = [
    ("B3, la configuracion de los +0.090 / +0.049",
     "was weakest: C4 gained 0.090 Dice and C2 only 0.049",
     "was weakest: with the all-lateral pool, C4 gained 0.090 Dice and C2 only 0.049"),
    ("B5, la configuracion de las cifras de INCA",
     "On INCA the values were high and uniform (C2: 0.914",
     "On INCA the values were high and uniform for Mean Teacher at $r{=}15$ "
     "(C2: 0.914"),
    ("el cierre de P3, hacia donde queda margen",
     "so better masks cannot move the scalar much further.",
     "so better masks cannot move the scalar much further. Further gains in the scalar "
     "would therefore have to come from the rule itself, or from predicting the corners "
     "directly as in \\citet{ref_zhang2021}, rather than from better masks."),
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
    print("  %-46s +%d palabras" % (etiqueta, len(b.split()) - len(a.split())))
print()


def palabras(s):
    s = re.sub(r"\\citep\{[^}]*\}|\\citet\{[^}]*\}", "", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = re.sub(r"[{}$~\\^]", " ", s)
    return len(s.split())


def legible(s):
    s = re.sub(r"\\citet\{ref_zhang2021\}", "Zhang et al.", s)
    s = re.sub(r"\\citep\{([^}]*)\}", r"[\1]", s)
    s = re.sub(r"\\ref\{([^}]*)\}", r"[\1]", s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = s.replace("{=}", "=").replace("~", " ")
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    return " ".join(s.replace("{", "").replace("}", "").replace("$", "").split())


for marca in ["Semi-supervised training did not benefit",
              "The C2--C4 anatomical scalar"]:
    i = texto.index(marca)
    par = texto[i:texto.index("\n", i)]
    print("=" * 92)
    print("PARRAFO (%d palabras)" % palabras(par))
    print("=" * 92)
    for f in re.split(r"(?<=\.) (?=[A-Z])", par):
        txt = legible(f)
        if txt:
            print("  (%2d) %s" % (palabras(f), txt[:84]))
            for k in range(84, len(txt), 84):
                print("       " + txt[k:k + 84])
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
