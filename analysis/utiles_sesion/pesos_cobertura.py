"""Check that every run whose folder has no checkpoint has one stored apart.

The checkpoints of the older runs were moved out of the run folders into
PESOS_UNM and PESOS_INCA, where each file is named after the arm and the seed.
This script pairs every seed folder that lacks best_model.pt with the file that
should hold its weights, and reports the ones that are genuinely absent.
"""
import glob
import os

BASE = r"G:\My Drive\UNM_vertebras_seg_v3"
ALMACENES = ["PESOS_UNM", "PESOS_INCA"]

guardados = set()
for a in ALMACENES:
    p = os.path.join(BASE, a)
    if os.path.isdir(p):
        for fn in os.listdir(p):
            if fn.endswith(".pt"):
                guardados.add(fn[:-3])

raices = sorted(d for d in os.listdir(BASE)
                if d.startswith("runs_") and os.path.isdir(os.path.join(BASE, d)))

sin_nada = []
rescatados = 0
total_sin_peso = 0
for raiz in raices:
    rp = os.path.join(BASE, raiz)
    for brazo in sorted(os.listdir(rp)):
        bp = os.path.join(rp, brazo)
        if not os.path.isdir(bp):
            continue
        for d in sorted(glob.glob(os.path.join(bp, "seed_*"))):
            if os.path.exists(os.path.join(d, "best_model.pt")):
                continue
            if not glob.glob(os.path.join(d, "*run_report.json")):
                continue
            total_sin_peso += 1
            clave = brazo + "_" + os.path.basename(d)
            if clave in guardados:
                rescatados += 1
            else:
                sin_nada.append(raiz + "/" + clave)

print("ficheros .pt en PESOS_UNM + PESOS_INCA          {}".format(len(guardados)))
print("carpetas de semilla sin best_model.pt dentro    {}".format(total_sin_peso))
print("de esas, con su peso guardado aparte            {}".format(rescatados))
print("de esas, SIN peso en ningun sitio               {}".format(len(sin_nada)))
if sin_nada:
    print()
    print("LAS QUE NO APARECEN:")
    for s in sin_nada:
        print("   " + s)
