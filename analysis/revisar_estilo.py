"""Measure caption length, section labels and filler words in the manuscript."""

import io
import re

RUTA = r"G:\My Drive\UNM_vertebras_seg_v3\tesis_latex_overleaf\tesis_final_v54.tex"
t = io.open(RUTA, encoding="utf-8").read()


def cuerpo_llaves(texto, i):
    """Return the text inside the braces that open at index i."""
    d, j = 0, i
    while j < len(texto):
        if texto[j] == "{":
            d += 1
        elif texto[j] == "}":
            d -= 1
            if d == 0:
                return texto[i + 1:j], j
        j += 1
    return "", i


def palabras(s):
    return len(re.sub(r"\\[a-zA-Z]+|[{}$~\\]", " ", s).split())


print("=" * 78)
print("PIES DE FIGURA Y TABLA")
print("=" * 78)
largos = []
for m in re.finditer(r"\\caption(\[[^\]]*\])?\{", t):
    cuerpo, _ = cuerpo_llaves(t, m.end() - 1)
    corto = m.group(1)[1:-1] if m.group(1) else "(sin titulo corto)"
    n = palabras(cuerpo)
    marca = "  <-- LARGO" if n > 60 else ""
    print("  %3d palabras%-12s %s" % (n, marca, corto[:58]))
    if n > 60:
        largos.append((n, corto, " ".join(cuerpo.split())))

print()
print("=" * 78)
print("ENTRADILLAS EN NEGRITA DEL CAPITULO 4")
print("=" * 78)
ini = t.index("\\chapter{Results and Discussion}")
fin = t.index("\\chapter{Conclusion}")
cap = t[ini:fin]
for m in re.finditer(r"\\textbf\{([^}]{4,90})\}", cap):
    frag = cap[max(0, m.start() - 3):m.start()]
    if frag.endswith("\n\n") or frag.endswith("\n"):
        print("  ", m.group(1))

print()
print("=" * 78)
print("TITULOS DE SECCION DEL CAPITULO 4")
print("=" * 78)
for m in re.finditer(r"\\section\{([^}]+)\}", cap):
    print("  ", m.group(1))

print()
print("=" * 78)
print("MULETILLAS EN TODO EL DOCUMENTO")
print("=" * 78)
hay = False
for p in ["Moreover", "Furthermore", "It is worth noting", "Notably", "crucial",
          "robust", "leverage", "delve", "In order to", "It should be noted",
          "Importantly", "significantly", "novel", "comprehensive", "seamless",
          "state-of-the-art", "cutting-edge", "paradigm"]:
    c = len(re.findall(p, t))
    if c:
        print("  %-22s %d" % (p, c))
        hay = True
if not hay:
    print("  ninguna")

print()
print("=" * 78)
print("LOS PIES LARGOS, COMPLETOS")
print("=" * 78)
for n, corto, cuerpo in largos:
    print()
    print("--- %d palabras | %s" % (n, corto))
    print("    " + cuerpo)
