"""Static check of notebook 22: every name a cell uses must be defined before it.

Cells are walked in order and the set of names each one binds is accumulated, so a
name used in the pool step but only imported in the training step would surface here
rather than three hours into a run.
"""

import ast
import builtins
import io
import json
import os
import sys

BASE = r"G:\My Drive\UNM_vertebras_seg_v3\tesis_seg\notebooks"
NB = os.path.join(BASE, "22_unm_pool_incertidumbre.ipynb")
REF = os.path.join(BASE, "19_unm_nnunet_ablation.ipynb")

with io.open(NB, encoding="utf-8") as fh:
    nb = json.load(fh)
with io.open(REF, encoding="utf-8") as fh:
    ref = json.load(fh)

print("=" * 76)
print("PROCEDENCIA DE CADA CELDA")
print("=" * 76)
ref_src = ["".join(c["source"]) for c in ref["cells"]]
for i, c in enumerate(nb["cells"]):
    s = "".join(c["source"])
    if c["cell_type"] == "markdown":
        origen = "markdown nuevo"
    elif s in ref_src:
        origen = "COPIADA VERBATIM del notebook 19 (celda %d)" % ref_src.index(s)
    else:
        origen = "codigo nuevo"
    cab = [l for l in s.strip().split("\n") if l.strip()][:1]
    print("  %2d %-10s %-46s %s" % (i, c["cell_type"][:8], origen,
                                    (cab[0][:34] if cab else "")))

print()
print("=" * 76)
print("NOMBRES USADOS SIN DEFINIR")
print("=" * 76)

definidos = set(dir(builtins))
problemas = []

for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    limpio = "\n".join(("pass  # " + l if l[:1] in ("!", "%") else l)
                       for l in src.split("\n"))
    arbol = ast.parse(limpio)

    usados, locales = set(), set()

    class V(ast.NodeVisitor):
        def visit_Name(self, n):
            (usados if isinstance(n.ctx, ast.Load) else locales).add(n.id)
        def visit_Import(self, n):
            for a in n.names:
                locales.add((a.asname or a.name).split(".")[0])
        def visit_ImportFrom(self, n):
            for a in n.names:
                locales.add(a.asname or a.name)
        def visit_FunctionDef(self, n):
            locales.add(n.name)
            for a in n.args.args:
                locales.add(a.arg)
            self.generic_visit(n)
        def visit_ClassDef(self, n):
            locales.add(n.name)
            self.generic_visit(n)
        def visit_comprehension(self, n):
            for t in ast.walk(n.target):
                if isinstance(t, ast.Name):
                    locales.add(t.id)
            self.generic_visit(n)
        def visit_ExceptHandler(self, n):
            if n.name:
                locales.add(n.name)
            self.generic_visit(n)

    V().visit(arbol)
    falta = sorted(usados - locales - definidos)
    if falta:
        problemas.append((i, falta))
        print("  celda %2d: %s" % (i, ", ".join(falta)))
    definidos |= locales

if not problemas:
    print("  ninguno")

print()
print("=" * 76)
print("LO QUE ESTE CHEQUEO NO PUEDE VER")
print("=" * 76)
print("  - que el checkpoint juez cargue de verdad (hace falta torch y el .pt)")
print("  - que la GPU aguante el batch")
print("  - la velocidad real de Drive en Colab")
print("  - que run_training converja")
print("  Todo eso lo cubre el Paso 1 del propio notebook, que no entrena.")
sys.exit(1 if problemas else 0)
