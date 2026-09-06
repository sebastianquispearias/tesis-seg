"""Rewrite the opening of 4.2 so its scope matches what the section reports.

The opening announced two regimes, UNM and the INCA 10 per cent subset, while the
section also reports INCA at the full training set. It also closed on the reason for
the scope rather than opening with it, which left a near-tautology stranded after a
pointer to two tables. Reordering turns that sentence into the premise, and the INCA
full-set result stops being an exception to the stated scope and becomes the check
that the premise predicts. The pointer stays: it is the only citation those two
tables have.
"""

import io
import os
import shutil
import sys

RUTA = r"G:\My Drive\UNM_vertebras_seg_v3\tesis_latex_overleaf\tesis_final_v54.tex"
COPIA = RUTA + ".ANTES_intro_42"

APLICAR = "--aplicar" in sys.argv

ANTES = ("These analyses are reported for the two regimes where SSL provided gains: "
         "UNM and the INCA 10\\% subset. The corresponding pool-size results are "
         "reported in Tables~\\ref{tab:ssl_unm_pool} and~\\ref{tab:ssl_inca_pool}. "
         "Within those regimes, pool size and selection policy can affect performance.")

DESPUES = ("Pool size and selection policy can only affect performance where SSL "
           "provides gains, so these analyses focus on UNM and the INCA 10\\% subset. "
           "INCA with the full training set is reported as a check on that premise. "
           "The corresponding pool-size results are in "
           "Tables~\\ref{tab:ssl_unm_pool} and~\\ref{tab:ssl_inca_pool}.")

with io.open(RUTA, "r", encoding="utf-8", newline="") as fh:
    original = fh.read()

if original.count("\r"):
    sys.exit("ABORTO: el .tex tiene retornos de carro y se esperaba solo LF.")
if original.count(ANTES) != 1:
    sys.exit("ABORTO: el ancla aparece %d veces, no 1." % original.count(ANTES))

texto = original.replace(ANTES, DESPUES, 1)

print("=" * 78)
print("ANTES")
print("=" * 78)
for k in range(0, len(ANTES), 92):
    print("  " + ANTES[k:k + 92])
print()
print("=" * 78)
print("DESPUES")
print("=" * 78)
for k in range(0, len(DESPUES), 92):
    print("  " + DESPUES[k:k + 92])
print()
print("comprobaciones:")
print("  'the two regimes' que quedan          :", texto.count("the two regimes"))
print("  citas de tab:ssl_unm_pool             :", texto.count("ref{tab:ssl_unm_pool}"))
print("  citas de tab:ssl_inca_pool            :", texto.count("ref{tab:ssl_inca_pool}"))
print("  sigue reportando INCA full en 4.2     :",
      "On INCA with the full training set" in texto)
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
