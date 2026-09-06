"""Compare the epoch-history keys of two runs and show what they actually hold.

Reading a key that a newer training loop renamed would make a live quantity look
like a constant zero, so the keys themselves are printed before any value is
believed, together with how many epochs carry a non-null value.
"""
import glob
import json
import os
import sys

BASE = r"G:\My Drive\UNM_vertebras_seg_v3"

for carpeta in sys.argv[1:]:
    ps = glob.glob(os.path.join(BASE, carpeta, "*run_report.json"))
    if not ps:
        print(carpeta, "sin informe")
        continue
    with open(ps[0], "r", encoding="utf-8") as fh:
        r = json.load(fh)
    hist = r.get("epoch_history") or []
    print("=" * 74)
    print(carpeta)
    print("  epocas:", len(hist))
    if not hist:
        continue
    print("  claves:", ", ".join(sorted(hist[0].keys())))
    print()
    for k in sorted(hist[0].keys()):
        vals = [e.get(k) for e in hist]
        nn = [v for v in vals if v is not None]
        nz = [v for v in nn if isinstance(v, (int, float)) and v != 0]
        if "unsup" in k or "lambda" in k or "semi" in k or "pl_" in k:
            print("   {:24s} no nulos {:>4}/{:<4} no cero {:>4}   ejemplo {}".format(
                k, len(nn), len(vals), len(nz), nn[-1] if nn else None))
