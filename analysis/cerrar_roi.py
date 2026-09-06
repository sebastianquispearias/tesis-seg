"""Let the region-of-interest paragraph end on what it set out to show.

Its two jobs are to show why segmentation quality matters beyond Dice, through a
published use of the mask, and to show that semi-supervised training moves that use.
The paragraph closed instead on adequacy, after two sentences of caution, so the
effect it had just measured was left in the middle. The scope caveat stays, since the
figure illustrates a clinical application that was not evaluated here, but it is now
stated as what was measured rather than as what was missing, and the paragraph ends on
the effect. The heading also had "downstream" attached to the region rather than to
the task, which is the one place in the manuscript where that adjective modifies
something that does not happen later.
"""

import io
import os
import re
import shutil
import sys

RUTA = r"G:\My Drive\UNM_vertebras_seg_v3\tesis_latex_overleaf\tesis_final_v54.tex"
COPIA = RUTA + ".ANTES_cerrar_roi"

APLICAR = "--aplicar" in sys.argv

CAMBIOS = [
    ("la entradilla",
     "\\textbf{Placement of a downstream region of interest.}",
     "\\textbf{Placement of a region of interest for a downstream task.}"),
    ("el cierre del parrafo",
     "Bolus annotations are not available here, so this is not an evaluation of "
     "airway-invasion detection. It shows that the predicted mask locates the column "
     "well enough to anchor such a region.",
     "This measures where the window lands, not whether invasion is detected. The "
     "predicted mask locates the column well enough to anchor the window, and a better "
     "mask places it closer."),
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
    print("  %-24s %d -> %d palabras" % (etiqueta, len(a.split()), len(b.split())))
print()


def palabras(s):
    s = re.sub(r"\\citep\{[^}]*\}|\\citet\{[^}]*\}", "", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = re.sub(r"[{}$~\\^]", " ", s)
    return len(s.split())


def legible(s):
    s = re.sub(r"\\citet\{ref_lee2020\}", "Lee et al.", s)
    s = re.sub(r"\\citep\{([^}]*)\}", r"[\1]", s)
    s = re.sub(r"\\ref\{([^}]*)\}", r"[\1]", s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", s)
    s = s.replace("$\\pm$", "+/-").replace("{=}", "=").replace("~", " ")
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    return " ".join(s.replace("{", "").replace("}", "").replace("$", "").split())


i = texto.index("**Placement" if "**Placement" in texto
                else "\\textbf{Placement of a region")
par = texto[i:texto.index("\n", i)]

print("=" * 92)
print("EL PARRAFO ENTERO  (%d palabras)" % palabras(par))
print("=" * 92)
for f in re.split(r"(?<=\.) (?=[A-Z*\\])", par):
    txt = legible(f)
    if txt:
        print("  (%2d) %s" % (palabras(f), txt[:84]))
        for k in range(84, len(txt), 84):
            print("       " + txt[k:k + 84])
print()
print("comprobaciones:")
print("  queda 'Bolus annotations' :", "Bolus annotations" in texto)
print("  queda 'downstream region' :", "downstream region" in texto)
print("  'downstream' en el doc    :", texto.count("downstream"))
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
