"""Repair the sentence that closes the nnU-Net caveat in 4.3.

It concluded from the list of four configuration differences, and the two concrete
sentences added earlier today were inserted between the list and that conclusion, so
the link stopped being visible. It also claimed that the fixed pipeline stays within
supervised learning, which the rest of the chapter contradicts: that pipeline is where
every semi-supervised run lives. The replacement reaches back over the inserted detail
and states what was actually meant, which is that nnU-Net is a supervised system.
"""

import io
import os
import shutil
import sys

RUTA = r"G:\My Drive\UNM_vertebras_seg_v3\tesis_latex_overleaf\tesis_final_v54.tex"
COPIA = RUTA + ".ANTES_paradigma"

APLICAR = "--aplicar" in sys.argv

ANTES = ("Both nnU-Net and the fixed pipeline used here remain within supervised "
         "learning; what differs is how each pipeline is configured, not the learning "
         "paradigm.")

DESPUES = ("All of these are configuration differences rather than differences of "
           "learning paradigm, because nnU-Net is itself a supervised system.")

with io.open(RUTA, "r", encoding="utf-8", newline="") as fh:
    original = fh.read()

if original.count("\r"):
    sys.exit("ABORTO: el .tex tiene retornos de carro y se esperaba solo LF.")
if original.count(ANTES) != 1:
    sys.exit("ABORTO: el ancla aparece %d veces, no 1." % original.count(ANTES))

texto = original.replace(ANTES, DESPUES, 1)

print("ANTES  (%d palabras)" % len(ANTES.split()))
print("  " + ANTES)
print()
print("DESPUES (%d palabras)" % len(DESPUES.split()))
print("  " + DESPUES)
print()
print("comprobaciones:")
print("  dos puntos de anuncio en la frase nueva :", ":" in DESPUES)
print("  punto y coma                            :", ";" in DESPUES)
print("  sigue diciendo 'learning paradigm'      :", "learning paradigm" in DESPUES)
print("  ya no afirma que el pipeline fijo sea supervisado:",
      "fixed pipeline used here remain within supervised" not in texto)
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
