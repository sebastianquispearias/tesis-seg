"""Find run folders whose reports disagree with the folder name.

The summary scripts group runs by the `experiment_name` field of the report,
not by the folder that holds them. When a run was launched under a slightly
different name, its report is filed under that other name and the arm appears
to have fewer seeds than it really has. This script lists every folder where
the two disagree, and then recomputes the F1 mean and standard deviation
grouping strictly by folder, which is what the arm actually contains.

It also reports which folders are missing a checkpoint, to separate an
interrupted run from an old run whose weights were deleted to save space.
"""
import glob
import json
import math
import os

BASE = r"G:\My Drive\UNM_vertebras_seg_v3"


def sd(xs):
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


raices = sorted(d for d in os.listdir(BASE)
                if d.startswith("runs_") and os.path.isdir(os.path.join(BASE, d)))

desajustes = []
sin_peso = {}
por_carpeta = {}

for raiz in raices:
    rp = os.path.join(BASE, raiz)
    for brazo in sorted(os.listdir(rp)):
        bp = os.path.join(rp, brazo)
        if not os.path.isdir(bp):
            continue
        for d in sorted(glob.glob(os.path.join(bp, "seed_*"))):
            ps = glob.glob(os.path.join(d, "*run_report.json"))
            if not ps:
                continue
            with open(ps[0], "r", encoding="utf-8") as fh:
                r = json.load(fh)
            nombre = r.get("run_identity", {}).get("experiment_name", "")
            f1 = (r.get("test_metrics") or {}).get("sample_mean_f1")
            clave = raiz + "/" + brazo
            if nombre != brazo:
                desajustes.append((clave, os.path.basename(d), nombre))
            if f1 is not None:
                por_carpeta.setdefault(clave, []).append((os.path.basename(d), f1))
            if not os.path.exists(os.path.join(d, "best_model.pt")):
                sin_peso.setdefault(clave, []).append(os.path.basename(d))

print("=" * 92)
print("CARPETAS DONDE experiment_name NO COINCIDE CON EL NOMBRE DE LA CARPETA")
print("=" * 92)
if not desajustes:
    print("  ninguna")
for clave, seed, nombre in desajustes:
    print("  {:58s} {:8s} el informe dice '{}'".format(clave, seed, nombre))

print()
print("=" * 92)
print("MEDIAS RECALCULADAS AGRUPANDO POR CARPETA, solo los brazos afectados")
print("=" * 92)
afectadas = sorted({c for c, _s, _n in desajustes})
for clave in afectadas:
    vals = sorted(por_carpeta.get(clave, []))
    f1s = [v for _s, v in vals]
    print("  {}".format(clave))
    for s, v in vals:
        print("      {:8s} {:.4f}".format(s, v))
    print("      F1 media {:.4f} +/- {:.4f}   n={}".format(
        sum(f1s) / len(f1s), sd(f1s), len(f1s)))

print()
print("=" * 92)
print("BRAZOS SIN best_model.pt EN ALGUNA SEMILLA")
print("=" * 92)
for clave in sorted(sin_peso):
    print("  {:62s} {}".format(clave, ", ".join(sin_peso[clave])))
print("  total de brazos afectados: {}".format(len(sin_peso)))
