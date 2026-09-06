"""Compare the augmentation profiles using only the arms whose noise was intact.

The base profile lost its Gaussian noise under albumentations 2.x until the fix
of 3 September, so A_baseline is not comparable with anything and is reported
apart, as the measure of what the defect cost. The nnU-Net profiles were built
with the 2.x signature from the start and were never affected, so every arm
that uses them stays in.
"""
import glob
import json
import math
import os

BASE = r"G:\My Drive\UNM_vertebras_seg_v3"

GRUPOS = [
    ("base, ruido ARREGLADO", ["runs_nnunet_ablation/A_baseline_fixnoise",
                               "runs_nnunet_ablation/X_sup_base"]),
    ("nnunet_moderate", ["runs_nnunet_ablation/C1_aug_moderate",
                         "runs_nnunet_ablation/X_sup_moderate"]),
    ("nnunet_full", ["runs_nnunet_ablation/C2_aug_full"]),
    ("base, ruido ROTO (excluido)", ["runs_nnunet_ablation/A_baseline"]),
]


def sd(xs):
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def f1s(carpeta):
    vals = []
    for d in sorted(glob.glob(os.path.join(BASE, carpeta, "seed_*"))):
        ps = glob.glob(os.path.join(d, "*run_report.json"))
        if not ps:
            continue
        with open(ps[0], "r", encoding="utf-8") as fh:
            r = json.load(fh)
        v = (r.get("test_metrics") or {}).get("sample_mean_f1")
        if v is not None:
            vals.append(v)
    return vals


tot = {}
for nombre, carpetas in GRUPOS:
    todos = []
    print(nombre)
    for c in carpetas:
        v = f1s(c)
        todos.extend(v)
        print("   {:26s} {:.4f} +/- {:.4f}  n={}   {}".format(
            os.path.basename(c), sum(v) / len(v), sd(v), len(v),
            " ".join("{:.4f}".format(x) for x in v)))
    tot[nombre] = todos
    if len(carpetas) > 1:
        print("   {:26s} {:.4f} +/- {:.4f}  n={}".format(
            ">>> JUNTAS", sum(todos) / len(todos), sd(todos), len(todos)))
    print()

base = tot["base, ruido ARREGLADO"]
mb = sum(base) / len(base)
print("=" * 74)
print("LO QUE SE PUEDE DECIR")
print("=" * 74)
for nombre in ("nnunet_moderate", "nnunet_full"):
    v = tot[nombre]
    m = sum(v) / len(v)
    print("  {:22s} {:.4f} (n={})  contra base {:.4f} (n={})  =  {:+.4f}".format(
        nombre, m, len(v), mb, len(base), m - mb))
roto = tot["base, ruido ROTO (excluido)"]
mr = sum(roto) / len(roto)
print()
print("  LO QUE COSTO EL BUG DEL RUIDO:")
print("  base con ruido roto {:.4f} (n={})  contra base arreglada {:.4f} (n={})  =  {:+.4f}".format(
    mr, len(roto), mb, len(base), mr - mb))
