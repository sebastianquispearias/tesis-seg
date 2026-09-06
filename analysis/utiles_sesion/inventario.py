"""Inventory of every seed folder under the given run roots.

A finished run leaves three artefacts: best_model.pt, test_metrics.csv and
run_summary.txt, plus the run_report.json the analysis reads. A run that was
interrupted leaves only some of them, and the notebook redoes it. This script
lists, for every seed folder, which artefacts are present, how many epochs the
history holds, and when the folder was last written, so that an incomplete arm
is distinguished from an arm that is still training.
"""
import glob
import json
import os
import sys
import time

BASE = r"G:\My Drive\UNM_vertebras_seg_v3"
ESPERADOS = ["best_model.pt", "test_metrics.csv", "run_summary.txt"]


def ultima_escritura(carpeta):
    t = 0.0
    for dp, _dn, fns in os.walk(carpeta):
        for fn in fns:
            try:
                t = max(t, os.path.getmtime(os.path.join(dp, fn)))
            except OSError:
                pass
    return t


ahora = time.time()
for raiz in sys.argv[1:]:
    ruta = os.path.join(BASE, raiz)
    print("=" * 92)
    print(raiz)
    print("=" * 92)
    brazos = sorted(d for d in os.listdir(ruta)
                    if os.path.isdir(os.path.join(ruta, d)))
    for brazo in brazos:
        pb = os.path.join(ruta, brazo)
        semillas = sorted(glob.glob(os.path.join(pb, "seed_*")))
        if not semillas:
            continue
        print("-" * 92)
        print("{}   {} carpetas de semilla".format(brazo, len(semillas)))
        for d in semillas:
            faltan = [f for f in ESPERADOS
                      if not os.path.exists(os.path.join(d, f))]
            ps = glob.glob(os.path.join(d, "*run_report.json"))
            if ps:
                with open(ps[0], "r", encoding="utf-8") as fh:
                    r = json.load(fh)
                ep = len(r.get("epoch_history") or [])
                f1 = (r.get("test_metrics") or {}).get("sample_mean_f1")
                f1s = "{:.4f}".format(f1) if f1 is not None else "  --  "
            else:
                ep, f1s = 0, "  --  "
            edad = (ahora - ultima_escritura(d)) / 60.0
            estado = "COMPLETO" if (ps and not faltan) else "INCOMPLETO"
            print("   {:8s} {:10s} F1 {}  ep {:<4d} informe {}  falta: {:28s} ultima escritura hace {:.0f} min".format(
                os.path.basename(d), estado, f1s, ep,
                "si" if ps else "NO",
                ", ".join(faltan) if faltan else "nada",
                edad))
