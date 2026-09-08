"""Run the mechanical part of a final review over one chapter.

The checks are the ones this project has actually been caught by: control
characters left by a shell escape, paragraphs far outside the length the rest of
the document uses, connectives the author never writes, references that point
nowhere, and numbers repeated between a table and the sentence that introduces
it. What the checks cannot judge is whether the prose follows, which is read by
hand afterwards.

Usage:  python revisar_cap.py <tex> <titulo del capitulo>
"""
import io
import re
import sys

B = chr(92)
NL = chr(10)
RUTA, TITULO = sys.argv[1], sys.argv[2]

with io.open(RUTA, "r", encoding="utf-8", newline="") as fh:
    texto = fh.read()
lineas = texto.split(NL)

ini = fin = None
for i, l in enumerate(lineas, 1):
    if l.strip().startswith(B + "chapter{"):
        if TITULO in l and ini is None:
            ini = i
        elif ini is not None:
            fin = i
            break
fin = fin or len(lineas)
cap = NL.join(lineas[ini - 1:fin - 1])
print("capitulo '{}': lineas {} a {}".format(TITULO, ini, fin))

print()
print("SECCIONES")
for m in re.finditer(re.escape(B) + r"section\{(.+?)\}", cap):
    print("   " + m.group(1))

# --- C1 longitud de parrafo
saltar = (B + "begin", B + "end", B + "label", B + "section", B + "caption",
          B + "centering", B + "footnotesize", B + "toprule", B + "midrule",
          B + "bottomrule", B + "resizebox", B + "includegraphics",
          B + "quad", B + "multicolumn", B + "textbf", B + "chapter", "%")
parr = [p.strip() for p in cap.split(NL + NL)
        if p.strip() and not p.strip().startswith(saltar)]
todos = [p.strip() for p in texto.split(NL + NL)
         if p.strip() and not p.strip().startswith(saltar)]
largos_doc = sorted(len(p) for p in todos)
mediana = largos_doc[len(largos_doc) // 2]
print()
print("C1  LONGITUD DE PARRAFO  (mediana del documento entero: {} chars)".format(mediana))
for p in sorted(parr, key=len, reverse=True):
    aviso = "  <<< el doble de la mediana" if len(p) > 2 * mediana else ""
    print("   {:5d} chars  {}...{}".format(
        len(p), " ".join(p.split())[:62], aviso))

# --- C2 marcas de estilo
print()
print("C2  MARCAS DE ESTILO")
for w in ("Furthermore", "Moreover", "Notably", "Importantly", "In addition",
          "It is worth noting", "It should be noted", "As mentioned",
          "In this section", "This section", "we can see", "As we can see"):
    n = cap.count(w)
    if n:
        print("   {:22s} {} vez/veces".format(w, n))
print("   guiones largos (em dash): {}".format(cap.count(chr(8212))))
dosp = len(re.findall(r"[a-z]\: [A-Za-z]", cap))
print("   dos puntos seguidos de texto: {}".format(dosp))

# --- D1 caracteres de control
print()
print("D1  CARACTERES DE CONTROL")
mal = 0
for c, nom in ((chr(9), "tabulador"), (chr(13), "retorno"), (chr(8), "backspace"),
               (chr(12), "form feed"), (chr(11), "tab vertical"), (chr(7), "campana")):
    n = cap.count(c)
    mal += n
    if n:
        print("   {:14s} {}".format(nom, n))
print("   total: {}".format(mal))

# --- B2 referencias
print()
print("B2  REFERENCIAS Y CITAS DEL CAPITULO")
etiquetas = set(re.findall(re.escape(B) + r"label\{([^}]+)\}", texto))
rotas = [r for r in re.findall(re.escape(B) + r"ref\{([^}]+)\}", cap)
         if r not in etiquetas]
print("   \\ref rotas: {}".format(rotas or "ninguna"))
claves = set()
for m in re.finditer(r"cite[tp]?\{([^}]+)\}", cap):
    claves.update(k.strip() for k in m.group(1).split(","))
try:
    with io.open(RUTA.rsplit("\\", 1)[0] + "\\tesis_sqa.bib", "r",
                 encoding="utf-8", errors="replace") as fh:
        bib = fh.read()
    faltan = [k for k in sorted(claves) if "{" + k + "," not in bib]
    print("   claves citadas: {}   sin entrada en el .bib: {}".format(
        len(claves), faltan or "ninguna"))
except Exception as e:
    print("   no se pudo leer el .bib: {}".format(e))
