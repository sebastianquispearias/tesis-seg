"""Diff the config blocks of two run reports and print their test metrics.

Used to decide whether a run made before the augmentation fix is comparable
with the runs made after it: if the configurations differ only in fields that
the fix did not touch, the older run can be reused.
"""
import glob
import json
import os
import sys

A, B = sys.argv[1], sys.argv[2]


def cargar(carpeta):
    ps = glob.glob(os.path.join(carpeta, "*run_report.json"))
    if not ps:
        return None
    with open(ps[0], "r", encoding="utf-8") as fh:
        return json.load(fh)


ra, rb = cargar(A), cargar(B)
ca, cb = ra.get("config", {}), rb.get("config", {})

print("=" * 74)
print("A =", A)
print("B =", B)
print("=" * 74)
print()
print("DIFERENCIAS EN config")
claves = sorted(set(ca) | set(cb))
n = 0
for k in claves:
    va, vb = ca.get(k, "<ausente>"), cb.get(k, "<ausente>")
    if va != vb:
        n += 1
        print("  {:34s} A={!r:28} B={!r}".format(k, va, vb))
if n == 0:
    print("  ninguna: las dos configuraciones son identicas")
print()
print("  total de campos distintos: {} de {}".format(n, len(claves)))

print()
print("TODOS LOS CAMPOS DE AUMENTO, lado a lado")
for k in claves:
    if not k.startswith("aug_"):
        continue
    va, vb = ca.get(k, "<ausente>"), cb.get(k, "<ausente>")
    marca = "   <-- DISTINTO" if va != vb else ""
    print("  {:34s} A={!r:28} B={!r}{}".format(k, va, vb, marca))

print()
print("METRICAS DE TEST")
for etiqueta, r in (("A", ra), ("B", rb)):
    tm, bm = r.get("test_metrics", {}), r.get("test_boundary_metrics", {})
    print("  {}  F1 {:.4f}   ASSD {:.3f}   HD95 {:.3f}".format(
        etiqueta, tm.get("sample_mean_f1"), bm.get("assd_mean_px"),
        bm.get("hd95_mean_px")))

print()
print("HUELLA DE REPRODUCIBILIDAD")
for etiqueta, r in (("A", ra), ("B", rb)):
    fp = r.get("reproducibility_fingerprint", {}) or {}
    print("  {}: {}".format(etiqueta, json.dumps(fp)[:400]))
