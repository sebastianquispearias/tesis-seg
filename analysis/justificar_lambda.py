"""State how the unsupervised weight was chosen, with the runs that chose it.

The manuscript gives the value of the unsupervised weight and justifies the ramp
schedule, but not the value itself, which is the point the proposal committee
pressed on. The comparison exists: twenty-four runs on UNM, two weights under
both methods and two pool sizes with three seeds each, finished on 15 April at
02:48, thirty hours before the first run of the main grid on 16 April at 08:27,
so the claim that it preceded the grid is verifiable rather than asserted.

The four figures written into the sentence are recomputed from the run reports
before anything is written, so a number that drifts stops the edit.
"""

import glob
import json
import math
import os
import shutil
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEX = os.path.join(BASE, "tesis_latex_overleaf", "tesis_final_v54.tex")
COPIA = TEX + ".ANTES_lambda"

BRAZOS = {
    "MT all-lateral 0.05": "runs_lambda_ablation/mean_teacher_all_lateral_lam005_s15p40",
    "MT all-lateral 0.10": "runs_lambda_ablation/mean_teacher_all_lateral_lam010_s15p40",
    "PL r10 0.05": "runs_lambda_ablation/semi_r10_lam005_s15p40",
    "PL r10 0.10": "runs_lambda_ablation/semi_r10_lam010_s15p40",
    "MT r10 0.05": "runs_lambda_ablation/mean_teacher_r10_lam005_s15p40",
    "MT r10 0.10": "runs_lambda_ablation/mean_teacher_r10_lam010_s15p40",
    "PL all-lateral 0.05": "runs_lambda_ablation/semi_all_lateral_lam005_s15p40",
    "PL all-lateral 0.10": "runs_lambda_ablation/semi_all_lateral_lam010_s15p40",
}

ANTES = (r"where $\lambda_U = 0.05$. The unsupervised weight is ramped")
DESPUES = (
    r"where $\lambda_U = 0.05$. This value was compared against "
    r"$\lambda_U = 0.10$ on UNM before the main experimental grid, with both "
    r"methods, two pool sizes and three seeds. The lower weight was better with "
    r"Mean Teacher on the all-lateral pool, F1~=~0.860~$\pm$~0.006 against "
    r"0.820~$\pm$~0.046, and with pseudo-labeling at $r=10$, "
    r"0.839~$\pm$~0.008 against 0.811~$\pm$~0.010. In the remaining two settings "
    r"the gap between them was smaller than the spread across seeds. "
    r"The unsupervised weight is ramped"
)


def stats(carpeta):
    vals = []
    for d in sorted(glob.glob(os.path.join(BASE, carpeta, "seed_*"))):
        ps = glob.glob(os.path.join(d, "*run_report.json"))
        if not ps:
            continue
        with open(ps[0], "r", encoding="utf-8") as fh:
            v = (json.load(fh).get("test_metrics") or {}).get("sample_mean_f1")
        if v is not None:
            vals.append(v)
    m = sum(vals) / len(vals)
    s = math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))
    return m, s, len(vals)


print("=" * 74)
print("PUERTA 1: las cuatro cifras de la frase salen de los informes")
print("=" * 74)
med = {}
for etiqueta, carpeta in BRAZOS.items():
    m, s, n = stats(carpeta)
    med[etiqueta] = (m, s, n)
    print("  {:22s} {:.4f} +/- {:.4f}   n={}".format(etiqueta, m, s, n))

cerrada = []
ESPERADO = [("MT all-lateral 0.05", "0.860", "0.006"),
            ("MT all-lateral 0.10", "0.820", "0.046"),
            ("PL r10 0.05", "0.839", "0.008"),
            ("PL r10 0.10", "0.811", "0.010")]
print()
for etiqueta, mm, ss in ESPERADO:
    m, s, _n = med[etiqueta]
    ok = ("%.3f" % m) == mm and ("%.3f" % s) == ss
    print("  {:22s} escrito {} +/- {}   calculado {:.3f} +/- {:.3f}   {}".format(
        etiqueta, mm, ss, m, s, "OK" if ok else "NO COINCIDE"))
    if not ok:
        cerrada.append(etiqueta)

print()
print("=" * 74)
print("PUERTA 2: los empates son de verdad empates")
print("=" * 74)
for a, b in (("MT r10 0.05", "MT r10 0.10"),
             ("PL all-lateral 0.05", "PL all-lateral 0.10")):
    ma, sa, _ = med[a]
    mb, sb, _ = med[b]
    hueco = abs(ma - mb)
    disp = max(sa, sb)
    ok = hueco < disp
    print("  {:22s} hueco {:.4f}   dispersion {:.4f}   {}".format(
        a.rsplit(" ", 1)[0], hueco, disp, "OK" if ok else "NO ES EMPATE"))
    if not ok:
        cerrada.append("{} no es empate".format(a))

if cerrada:
    print()
    print("PUERTA CERRADA. No se toca el .tex.")
    for c in cerrada:
        print("  -", c)
    sys.exit(1)

print()
print("PUERTAS ABIERTAS.")

with open(TEX, "rb") as fh:
    crudo = fh.read()
texto = crudo.decode("utf-8")

n = texto.count(ANTES)
print("  ancla:", n, "aparicion(es)")
if n != 1:
    print("ANCLA NO UNICA. No se escribe nada.")
    sys.exit(1)

if not os.path.exists(COPIA):
    shutil.copy2(TEX, COPIA)
nuevo = texto.replace(ANTES, DESPUES)
with open(TEX, "wb") as fh:
    fh.write(nuevo.encode("utf-8"))

with open(TEX, "rb") as fh:
    escrito = fh.read()
d = escrito.decode("utf-8")
cr = sum(1 for i, b in enumerate(escrito)
         if b == 0x0D and (i + 1 >= len(escrito) or escrito[i + 1] != 0x0A))
lf = sum(1 for i, b in enumerate(escrito)
         if b == 0x0A and (i == 0 or escrito[i - 1] != 0x0D))

print()
print("copia          :", os.path.basename(COPIA))
print("bytes anadidos :", len(escrito) - len(crudo))
print("reversible     :", "SI" if d.replace(DESPUES, ANTES).encode("utf-8") == crudo
      else "NO")
print("CR sueltos     :", cr)
print("LF sueltos     :", lf)
