"""Place each float relative to the section headings by position on the page.

Comparing page numbers alone is not enough, because a float flushed to the top of
the page on which the next section opens is printed before the heading and still
belongs to the section that referred to it. The vertical position of the caption
is therefore compared against the vertical position of the headings, and a float
counts as drifted only when it is printed after the heading of the section that
follows the one it was declared in.

Usage:  python donde_caen2.py <pdf> <tex>
"""
import io
import re
import sys

import fitz

PDF, TEX = sys.argv[1], sys.argv[2]
doc = fitz.open(PDF)

secciones = []
for lvl, titulo, pag in doc.get_toc():
    if lvl > 2 or pag < 1:
        continue
    # El ancla del marcador puede quedar al final de una pagina y el encabezado
    # imprimirse en la siguiente, asi que se busca tambien alli antes de rendirse.
    y, pagina_real = None, pag
    for cand in (pag, pag + 1):
        if cand > doc.page_count:
            break
        r = doc[cand - 1].search_for(titulo[:40])
        if r:
            y, pagina_real = r[0].y0, cand
            break
    if y is None:
        print("  aviso: no se encuentra el encabezado '{}'".format(titulo[:40]))
        y = 0.0
    secciones.append((pagina_real, y, titulo))
secciones.sort()


def seccion_en(pag, y):
    nombre = "(preliminares)"
    for p, sy, t in secciones:
        if (p, sy) <= (pag, y):
            nombre = t
    return nombre


primera = min(p for p, _, _ in secciones)
flotantes = []
for n in range(primera, doc.page_count + 1):
    for bloque in doc[n - 1].get_text("blocks"):
        y0, txt = bloque[1], bloque[4]
        m = re.match(r"^(Figure|Table)\s+(\d+\.\d+):\s*(.*)",
                     " ".join(txt.split()))
        if m:
            flotantes.append({"tipo": m.group(1), "num": m.group(2), "pag": n,
                              "y": y0, "sec": seccion_en(n, y0),
                              "txt": m.group(3)[:38]})
flotantes.sort(key=lambda f: (f["pag"], f["y"]))

with io.open(TEX, "r", encoding="utf-8", newline="") as fh:
    lineas = fh.read().split("\n")
sec, declara = "(preambulo)", {}
for i, l in enumerate(lineas, 1):
    m = re.search(r"\\(?:chapter|section)\{(.+?)\}", l)
    if m:
        sec = m.group(1)
    for lab in re.findall(r"\\label\{((?:fig|tab):[^}]+)\}", l):
        declara[lab] = sec

orden = []
i = 0
while i < len(lineas):
    if "\\caption" in lineas[i]:
        j, bloque = i, lineas[i]
        while bloque.count("{") > bloque.count("}") and j + 1 < len(lineas):
            j += 1
            bloque += " " + lineas[j]
        for k in range(j, min(j + 6, len(lineas))):
            m = re.search(r"\\label\{((?:fig|tab):[^}]+)\}", lineas[k])
            if m:
                orden.append(m.group(1))
                break
        i = j + 1
    else:
        i += 1

etiqueta_de = {}
for tipo, pref in (("Figure", "fig:"), ("Table", "tab:")):
    lista = [l for l in orden if l.startswith(pref)]
    nums = [f["num"] for f in flotantes if f["tipo"] == tipo]
    for lab, num in zip(lista, nums):
        etiqueta_de[(tipo, num)] = lab

print("=" * 100)
print("DONDE CAE CADA FLOTANTE, POR POSICION EN LA PAGINA")
print("=" * 100)
print("{:7s} {:5s} {:4s}  {:26s} {:26s}".format(
    "tipo", "num", "pag", "declarado en", "cae en"))
print("-" * 100)
derivan = 0
for f in flotantes:
    lab = etiqueta_de.get((f["tipo"], f["num"]), "?")
    decl = declara.get(lab, "?")
    mal = decl != "?" and f["sec"] != decl
    derivan += 1 if mal else 0
    print("{:7s} {:5s} {:4d}  {:26s} {:26s} {}".format(
        f["tipo"], f["num"], f["pag"], decl[:26], f["sec"][:26],
        "<<< DERIVA" if mal else ""))
print()
print("  flotantes fuera de su seccion: {} de {}".format(derivan, len(flotantes)))
