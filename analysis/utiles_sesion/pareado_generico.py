"""Seed-paired comparison between two arbitrary run folders, plus the tail of
the unsupervised loss.

Usage:  python pareado_generico.py <carpeta_A> <carpeta_B>

The arms are matched seed by seed. For every seed present in both arms the
script prints the F1 of each side and their difference, then the mean
difference, the sample standard deviation and the sign agreement. It also
prints the mean unsupervised loss over the last ten epochs of each run, which
is the quantity used to judge whether the consistency term carried signal.
"""
import glob
import json
import math
import os
import sys

BASE = r"G:\My Drive\UNM_vertebras_seg_v3"


def leer(carpeta):
    salida = {}
    for d in sorted(glob.glob(os.path.join(BASE, carpeta, "seed_*"))):
        ps = glob.glob(os.path.join(d, "*run_report.json"))
        if not ps:
            continue
        with open(ps[0], "r", encoding="utf-8") as fh:
            r = json.load(fh)
        seed = r.get("run_identity", {}).get("seed", -1)
        met = r.get("test_metrics") or r.get("metrics") or {}
        f1 = met.get("sample_mean_f1") or met.get("f1_mean")
        hist = r.get("epoch_history") or []
        cola = [e.get("unsup_loss") or 0.0 for e in hist[-10:]]
        salida[seed] = {
            "f1": f1,
            "unsup_cola": sum(cola) / len(cola) if cola else 0.0,
            "epocas": len(hist),
        }
    return salida


def sd(xs):
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


a_nom, b_nom = sys.argv[1], sys.argv[2]
A, B = leer(a_nom), leer(b_nom)
comunes = sorted(set(A) & set(B))

print("=" * 74)
print("A = {}".format(a_nom))
print("B = {}".format(b_nom))
print("-" * 74)
print("  semilla     F1(A)     F1(B)     delta B-A     unsup_cola A   unsup_cola B")
deltas = []
for s in comunes:
    d = B[s]["f1"] - A[s]["f1"]
    deltas.append(d)
    print("     {}       {:.4f}    {:.4f}    {:+.4f}         {:.6f}       {:.6f}".format(
        s, A[s]["f1"], B[s]["f1"], d, A[s]["unsup_cola"], B[s]["unsup_cola"]))

if deltas:
    m = sum(deltas) / len(deltas)
    print("-" * 74)
    print("  delta medio {:+.4f}   SD {:.4f}   n={}".format(m, sd(deltas), len(deltas)))
    pos = sum(1 for d in deltas if d > 0)
    print("  B gana en {} de {} semillas".format(pos, len(deltas)))

for nom, arm in ((a_nom, A), (b_nom, B)):
    f1s = [v["f1"] for v in arm.values()]
    colas = [v["unsup_cola"] for v in arm.values()]
    print("-" * 74)
    print("  {}".format(nom))
    print("    F1          media {:.4f} +/- {:.4f}   n={}".format(
        sum(f1s) / len(f1s), sd(f1s), len(f1s)))
    print("    unsup_cola  media {:.6f} +/- {:.6f}   valores {}".format(
        sum(colas) / len(colas), sd(colas),
        ", ".join("{:.6f}".format(c) for c in colas)))
