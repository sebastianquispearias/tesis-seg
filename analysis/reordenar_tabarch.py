"""Reorder the semi-supervised block of tab:arch to follow the supervised one.

The two blocks listed the same architectures in different orders, so comparing a
backbone against itself meant hunting for it twice. Reordering makes the rows line up
vertically. No number changes: the gate below checks that the set of rows is identical
before and after, character for character.
"""

import io
import os
import shutil
import sys

RUTA = r"G:\My Drive\UNM_vertebras_seg_v3\tesis_latex_overleaf\tesis_final_v54.tex"
COPIA = RUTA + ".ANTES_reorden_tabarch"

APLICAR = "--aplicar" in sys.argv

# el orden del bloque supervisado, leido del .tex
ORDEN = ["DeepLabV3+", "BiFPN-U-Net(T)", "U-Net++", "FPN", "U-Net", "TransUNet"]

BLOQUE_VIEJO = """\\quad U-Net++         & .830$\\pm$.038 & 1.92$\\pm$0.22 & 7.80$\\pm$0.93 \\\\
\\quad U-Net           & .851$\\pm$.005 & 1.64$\\pm$0.05 & 6.06$\\pm$0.74 \\\\
\\quad FPN             & .811$\\pm$.028 & 2.24$\\pm$0.13 & 7.74$\\pm$0.86 \\\\
\\quad DeepLabV3+      & .800$\\pm$.014 & 2.24$\\pm$0.25 & 8.17$\\pm$1.06 \\\\"""

with io.open(RUTA, "r", encoding="utf-8", newline="") as fh:
    original = fh.read()

if original.count("\r"):
    sys.exit("ABORTO: el .tex tiene retornos de carro y se esperaba solo LF.")
if original.count(BLOQUE_VIEJO) != 1:
    sys.exit("ABORTO: el bloque no aparece exactamente una vez (%d)."
             % original.count(BLOQUE_VIEJO))

filas = BLOQUE_VIEJO.split("\n")


def arq_de(fila):
    return fila.split("&")[0].replace("\\quad", "").strip()


presentes = {arq_de(f): f for f in filas}
nuevas = [presentes[a] for a in ORDEN if a in presentes]

# ------------------------------------------------------------------- la puerta
if sorted(nuevas) != sorted(filas):
    sys.exit("ABORTO: el conjunto de filas cambio. No es un reordenamiento.")
if len(nuevas) != len(filas):
    sys.exit("ABORTO: se perdio o duplico alguna fila.")

BLOQUE_NUEVO = "\n".join(nuevas)

print("PUERTA: el conjunto de filas es identico antes y despues.  OK")
print("        %d filas, ninguna cifra tocada." % len(nuevas))
print()
print("ANTES (orden actual del bloque SSL)")
for f in filas:
    print("  " + arq_de(f))
print()
print("DESPUES (sigue el orden del bloque supervisado)")
for f in nuevas:
    print("  " + arq_de(f))
print()
print("orden del bloque supervisado, para comparar:")
print("  " + ", ".join(ORDEN))
print()

if not APLICAR:
    print("MODO LECTURA. No se escribio nada. Anadir --aplicar para escribir.")
    sys.exit(0)

texto = original.replace(BLOQUE_VIEJO, BLOQUE_NUEVO, 1)

if not os.path.exists(COPIA):
    shutil.copy2(RUTA, COPIA)
with io.open(RUTA, "w", encoding="utf-8", newline="") as fh:
    fh.write(texto)
with io.open(RUTA, "r", encoding="utf-8", newline="") as fh:
    escrito = fh.read()

print("copia de seguridad :", os.path.basename(COPIA))
print("CR sueltos         :", escrito.count("\r"))
print("lineas             :", original.count("\n"), "->", escrito.count("\n"))
print("bytes              :", len(original), "->", len(escrito))
print("mismos bytes?      :", len(original) == len(escrito),
      "(tiene que ser True: solo cambia el orden)")
