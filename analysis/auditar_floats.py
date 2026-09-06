"""Audit every figure and table: where it lives, how often it is cited, caption size."""

import io
import re

RUTA = r"G:\My Drive\UNM_vertebras_seg_v3\tesis_latex_overleaf\tesis_final_v54.tex"
t = io.open(RUTA, encoding="utf-8").read()


def palabras(s):
    return len(re.sub(r"\\[a-zA-Z]+|[{}$~\\]", " ", s).split())


def cuerpo_llaves(texto, i):
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


# donde empieza cada capitulo y seccion, para situar cada float
marcas = []
for m in re.finditer(r"\\(chapter|section)\{([^}]+)\}", t):
    marcas.append((m.start(), m.group(1), m.group(2)))


def situar(pos):
    cap, sec = "?", "?"
    for p, tipo, nombre in marcas:
        if p > pos:
            break
        if tipo == "chapter":
            cap, sec = nombre, "-"
        else:
            sec = nombre
    return cap, sec


floats = []
for m in re.finditer(r"\\begin\{(figure\*?|table\*?)\}", t):
    tipo = m.group(1)
    fin = t.find("\\end{" + tipo + "}", m.end())
    blo = t[m.start():fin]
    lab = re.search(r"\\label\{([^}]+)\}", blo)
    lab = lab.group(1) if lab else "(SIN LABEL)"
    cm = re.search(r"\\caption(\[[^\]]*\])?\{", blo)
    if cm:
        cuerpo, _ = cuerpo_llaves(blo, cm.end() - 1)
        corto = cm.group(1)[1:-1] if cm.group(1) else "(sin titulo corto)"
        n = palabras(cuerpo)
    else:
        corto, n = "(SIN CAPTION)", 0
    cap, sec = situar(m.start())
    citas = len(re.findall(r"\\ref\{" + re.escape(lab) + r"\}", t)) if lab != "(SIN LABEL)" else 0
    floats.append((tipo, lab, corto, n, citas, cap, sec))

print("=" * 100)
print("INVENTARIO DE TODOS LOS FLOTANTES")
print("=" * 100)
print("%-7s %-18s %5s %5s  %-28s %s" % ("tipo", "label", "palab", "citas", "seccion", "titulo corto"))
print("-" * 100)
for tipo, lab, corto, n, citas, cap, sec in floats:
    alerta = ""
    if citas == 0:
        alerta = "  <-- HUERFANO, nadie lo cita"
    elif n > 60:
        alerta = "  <-- pie largo"
    print("%-7s %-18s %5d %5d  %-28s %s%s"
          % (tipo.replace("*", "*"), lab, n, citas, sec[:28], corto[:40], alerta))

print()
print("=" * 100)
print("SOLO EL CAPITULO 4")
print("=" * 100)
c4 = [f for f in floats if f[5].startswith("Results")]
print("  %d flotantes en el capitulo 4" % len(c4))
por_sec = {}
for f in c4:
    por_sec.setdefault(f[6], []).append(f)
for sec, fs in por_sec.items():
    tabs = sum(1 for f in fs if f[0].startswith("table"))
    figs = sum(1 for f in fs if f[0].startswith("figure"))
    print("    %-45s %d tablas, %d figuras" % (sec[:45], tabs, figs))
    for f in fs:
        print("        %-7s %-18s %3d palab, %d citas" % (f[0], f[1], f[3], f[4]))

print()
print("=" * 100)
print("RESUMEN")
print("=" * 100)
print("  flotantes totales      :", len(floats))
print("  huerfanos (0 citas)    :", sum(1 for f in floats if f[4] == 0))
print("  pies de mas de 60 palab:", sum(1 for f in floats if f[3] > 60))
print("  pies de mas de 80 palab:", sum(1 for f in floats if f[3] > 80))
print("  sin label              :", sum(1 for f in floats if f[1] == "(SIN LABEL)"))
