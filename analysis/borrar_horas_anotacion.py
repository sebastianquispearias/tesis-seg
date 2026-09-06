"""Drop the sentence that converts the INCA label reduction into annotation hours.

It was the only sentence of the results chapter that framed the outcome as a trade,
and it closed the paragraph on the weaker of the two comparisons the paragraph makes:
the loss against the full-label supervised result, rather than the gain against the
supervised baseline trained on the same budget. The annotation cost itself is stated
twice in the Methods chapter, with larger figures, so nothing is lost by removing it
here. No cross-reference points at it.
"""

import io
import os
import shutil
import sys

RUTA = r"G:\My Drive\UNM_vertebras_seg_v3\tesis_latex_overleaf\tesis_final_v54.tex"
COPIA = RUTA + ".ANTES_borrar_horas"

APLICAR = "--aplicar" in sys.argv

FRASE = (" In annotation terms the INCA figure means 578 masks that do not have to be "
         "drawn, roughly 19 to 29 hours at the 2--3 minutes per mask reported in "
         "Section~\\ref{sec:method-datasets}, for four points of F1.")

with io.open(RUTA, "r", encoding="utf-8", newline="") as fh:
    original = fh.read()

if original.count("\r"):
    sys.exit("ABORTO: el .tex tiene retornos de carro y se esperaba solo LF.")
if original.count(FRASE) != 1:
    sys.exit("ABORTO: la frase aparece %d veces, no 1." % original.count(FRASE))

texto = original.replace(FRASE, "", 1)

i = texto.index("A key practical observation")
fin = texto.index("\n", i)
print("=" * 78)
print("EL PARRAFO DESPUES DEL BORRADO")
print("=" * 78)
for linea in [" ".join(texto[i:fin].split())[k:k + 96]
              for k in range(0, len(" ".join(texto[i:fin].split())), 96)]:
    print("  " + linea)
print()
print("comprobaciones:")
print("  '578' que quedan en el documento      :", texto.count("578"))
print("  '19 to 29' que quedan                 :", texto.count("19 to 29"))
print("  el coste de anotacion sigue en cap. 3 :",
      texto.count("2--3 minutes"), "menciones de '2--3 minutes'")
print("  '32--48 hours' (INCA, cap. 3) sigue   :", texto.count("32--48 hours") == 1)
print("  '10--16 hours' (UNM, cap. 3) sigue    :", texto.count("10--16 hours") == 1)
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
      "(%+d)" % (len(escrito) - len(original)))
