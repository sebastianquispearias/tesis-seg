"""Find where, if anywhere, the manuscript defines the terms 4.4 relies on."""

import io
import re

RUTA = r"G:\My Drive\UNM_vertebras_seg_v3\tesis_latex_overleaf\tesis_final_v54.tex"
t = io.open(RUTA, encoding="utf-8").read()

# capitulos, para situar cada aparicion
marcas = [(m.start(), m.group(2)) for m in
          re.finditer(r"\\(chapter|section)\{([^}]+)\}", t)]


def donde(pos):
    nombre = "?"
    for p, n in marcas:
        if p > pos:
            break
        nombre = n
    return nombre


TERMINOS = {
    "el escalar C2-C4": r"C2--C4 anatomical scalar",
    "la regla / landmarks": r"corner landmark|post-processing rule|the rule that locates",
    "el umbral de 200 px": r"200~px",
    "centroide": r"centroid",
    "componentes conexas": r"connected component",
}

for etiqueta, patron in TERMINOS.items():
    print("=" * 92)
    print(etiqueta.upper())
    print("=" * 92)
    for m in re.finditer(patron, t):
        ini = max(0, m.start() - 130)
        frag = " ".join(t[ini:m.end() + 130].split())
        linea = t[:m.start()].count("\n") + 1
        print("  l.%-5d [%s]" % (linea, donde(m.start())[:40]))
        print("      ..." + frag + "...")
        print()
