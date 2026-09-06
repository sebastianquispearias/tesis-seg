"""Check whether a run actually consumed unlabeled frames.

A Mean Teacher run that received a pool shows a non-zero unsupervised loss and a
non-empty block of pseudo-label statistics. A run whose unlabeled loader was
never built shows zeros throughout, whatever its configuration claims.
"""
import glob
import json
import os
import sys

BASE = r"G:\My Drive\UNM_vertebras_seg_v3"


def mirar(carpeta):
    for d in sorted(glob.glob(os.path.join(BASE, carpeta, "seed_*"))):
        ps = glob.glob(os.path.join(d, "*run_report.json"))
        if not ps:
            continue
        with open(ps[0], "r", encoding="utf-8") as fh:
            r = json.load(fh)
        cfg = r.get("config", {})
        hist = r.get("epoch_history") or []
        claves_unsup = [k for k in (hist[0] if hist else {})
                        if "unsup" in k.lower() or "lambda_u" in k.lower()]
        maximo = {}
        for k in claves_unsup:
            vals = [e.get(k) or 0.0 for e in hist]
            maximo[k] = max(vals) if vals else 0.0
        pl = r.get("pl_stats_at_best_epoch")
        print("  {:10s} use_semi={!s:5s} lambda_u={:<6} epocas={:<4} pl_stats={}".format(
            os.path.basename(d), cfg.get("use_semi"), cfg.get("lambda_u"),
            len(hist), "vacio" if not pl else "presente"))
        for k, v in maximo.items():
            print("             max({}) = {}".format(k, v))


for carpeta in sys.argv[1:]:
    print("=" * 70)
    print(carpeta)
    mirar(carpeta)
