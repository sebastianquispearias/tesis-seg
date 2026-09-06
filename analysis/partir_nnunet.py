"""Split the nnU-Net paragraph of 4.3 where the text already says it splits.

At 279 words it was the longest paragraph of the chapter and it carried six blocks,
one of which is a different experiment under a different schedule. The paragraph had
to warn the reader about that itself: "This comparison is separate from the main
nnU-Net results above, which used a longer schedule." The break falls there. Not a
word changes: the edit is one blank line.
"""

import io
import os
import re
import shutil
import sys

RUTA = r"G:\My Drive\UNM_vertebras_seg_v3\tesis_latex_overleaf\tesis_final_v54.tex"
COPIA = RUTA + ".ANTES_particion_nnunet"

APLICAR = "--aplicar" in sys.argv

ANCLA = " The Mean Teacher comparison was also run on nnU-Net"
NUEVO = "\n\nThe Mean Teacher comparison was also run on nnU-Net"

with io.open(RUTA, "r", encoding="utf-8", newline="") as fh:
    original = fh.read()

if original.count("\r"):
    sys.exit("ABORTO: el .tex tiene retornos de carro y se esperaba solo LF.")
if original.count(ANCLA) != 1:
    sys.exit("ABORTO: el ancla aparece %d veces, no 1." % original.count(ANCLA))

texto = original.replace(ANCLA, NUEVO, 1)


def limpio(s):
    s = re.sub(r"\\citep\{[^}]*\}", "", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", s)
    s = re.sub(r"[{}$~\\^]", " ", s)
    return " ".join(s.split())


i = texto.index("The previous comparison kept the training pipeline fixed.")
a = texto[i:texto.index("\n\n", i)]
j = texto.index("The Mean Teacher comparison was also run on nnU-Net")
b = texto[j:texto.index("\n", j)]

print("=" * 78)
print("PARTICION DEL PARRAFO DE nnU-NET (4.3)")
print("=" * 78)
print("  parrafo A: %3d palabras  abre: The previous comparison kept the training..."
      % len(limpio(a).split()))
print("  parrafo B: %3d palabras  abre: The Mean Teacher comparison was also run..."
      % len(limpio(b).split()))
print()
print("  antes era un solo parrafo de %d palabras" % len(limpio(a + " " + b).split()))
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
print("lineas             :", original.count("\n"), "->", escrito.count("\n"))
print("bytes              :", len(original), "->", len(escrito),
      "(+%d)" % (len(escrito) - len(original)))
print("deshaciendo el salto, vuelve a ser identico al original:",
      escrito.replace(NUEVO, ANCLA, 1) == original)
