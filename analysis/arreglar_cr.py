"""Restore the backslash that a heredoc turned into a carriage return.

The block appended to the state file mentioned the LaTeX label \\ref{fig:roilee}. It
went through a bash heredoc, which is exactly what the project's own rule forbids for
text carrying backslashes, and the \\r of \\ref became a real carriage return followed
by "ef". Deleting the stray return would leave "ef{fig:roilee}"; the repair is to put
the backslash back.
"""

import io
import os
import shutil
import sys

RUTA = r"G:\My Drive\UNM_vertebras_seg_v3\ESTADO_SESION.md"
COPIA = RUTA + ".ANTES_arreglar_cr"

APLICAR = "--aplicar" in sys.argv

ROTO = "\ref{fig:roilee}"        # CR + "ef{fig:roilee}"
SANO = "\\ref{fig:roilee}"       # barra + "ref{fig:roilee}"

with io.open(RUTA, "r", encoding="utf-8", newline="") as fh:
    t = fh.read()


def sueltos(s):
    return sum(1 for k, ch in enumerate(s) if ch == "\r" and s[k + 1:k + 2] != "\n")


print("antes:")
print("  CR sueltos            :", sueltos(t))
print("  apariciones del roto  :", t.count(ROTO))
print("  repr del trozo        :", repr(t[t.index(ROTO) - 22:t.index(ROTO) + 20]))

nuevo = t.replace(ROTO, SANO)

print()
print("despues:")
print("  CR sueltos            :", sueltos(nuevo))
print("  LF sueltos            :", nuevo.count("\n") - nuevo.count("\r\n"))
print("  lineas                :", t.count("\r\n"), "->", nuevo.count("\r\n"))
print("  repr del trozo        :",
      repr(nuevo[nuevo.index(SANO) - 22:nuevo.index(SANO) + 20]))

if not APLICAR:
    print()
    print("MODO LECTURA. Anadir --aplicar para escribir.")
    sys.exit(0)

if not os.path.exists(COPIA):
    shutil.copy2(RUTA, COPIA)
with io.open(RUTA, "w", encoding="utf-8", newline="") as fh:
    fh.write(nuevo)
with io.open(RUTA, "r", encoding="utf-8", newline="") as fh:
    e = fh.read()
print()
print("ESCRITO")
print("  CR sueltos :", sueltos(e))
print("  LF sueltos :", e.count("\n") - e.count("\r\n"))
print("  lineas     :", e.count("\r\n"))
