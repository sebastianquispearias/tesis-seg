"""Signpost the five paragraphs of section 4.5.

The section answers five separate questions in a row and carries no marker at all,
while the section before it marks almost every paragraph. Four of the five questions
are the ones the advisor asked about uncertainty, so the markers also make those
answers findable. Only bold labels are inserted; not a word of the text changes.
"""

import io
import os
import shutil
import sys

RUTA = r"G:\My Drive\UNM_vertebras_seg_v3\tesis_latex_overleaf\tesis_final_v54.tex"
COPIA = RUTA + ".ANTES_entradillas_45"

APLICAR = "--aplicar" in sys.argv

MARCAS = [
    ("Definition of band entropy.",
     "Uncertainty is measured per pixel"),
    ("Location of the uncertainty.",
     "Uncertainty was confined to the"),
    ("Effect of semi-supervised training.",
     "Semi-supervised training did not make"),
    ("Effect of pool size on entropy.",
     "Pool size does not change it either"),
    ("Use of uncertainty for frame selection.",
     "Whether that is enough to guide frame"),
]

with io.open(RUTA, "r", encoding="utf-8", newline="") as fh:
    original = fh.read()

if original.count("\r"):
    sys.exit("ABORTO: el .tex tiene retornos de carro y se esperaba solo LF.")

texto = original
for marca, ancla in MARCAS:
    if texto.count(ancla) != 1:
        sys.exit("ABORTO: el ancla %r aparece %d veces, no 1." % (ancla, texto.count(ancla)))
    i = texto.index(ancla)
    if texto[i - 2:i] != "\n\n":
        sys.exit("ABORTO: el ancla %r no abre parrafo." % ancla)
    texto = texto.replace(ancla, "\\textbf{" + marca + "} " + ancla, 1)

print("=" * 78)
print("SECCION 4.5 - las cinco entradillas")
print("=" * 78)
for marca, ancla in MARCAS:
    print("  \\textbf{%s}" % marca)
    print("      antes de: %s..." % ancla)
print()
print("No cambia ni una palabra del texto. Solo se anteponen cinco \\textbf{}.")
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

# comprobacion: el texto sin las marcas tiene que ser identico al original
sin_marcas = escrito
for marca, _ in MARCAS:
    sin_marcas = sin_marcas.replace("\\textbf{" + marca + "} ", "", 1)

print("copia de seguridad :", os.path.basename(COPIA))
print("CR sueltos         :", escrito.count("\r"))
print("lineas             :", original.count("\n"), "->", escrito.count("\n"))
print("bytes              :", len(original), "->", len(escrito))
print("quitando las 5 marcas, el archivo vuelve a ser identico al original:",
      sin_marcas == original)
