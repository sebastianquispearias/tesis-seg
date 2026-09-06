"""Dry-run the pool selection of notebook 22 on the real file names.

Entropy is synthetic here; what is being tested is the selection itself: that the two
pools land on the target size, that they match the temporal pool video by video, that
they do not overlap, and that the cheap-candidate switch behaves the same way.
No image is read.
"""

import collections
import os
import re

import numpy as np
import pandas as pd

BASE = r"G:\My Drive\UNM_vertebras_seg_v3"
POOL_TEMPORAL = os.path.join(BASE, "unlabeling_r10_max0", "images")
POOL_TODOS = os.path.join(BASE, "unlabeling_all_lateral", "images")


def por_video(nombres):
    c = collections.Counter()
    for f in nombres:
        m = re.match(r"(v\d+)_f\d+\.png$", f)
        if m:
            c[m.group(1)] += 1
    return c


print("listando nombres (sin leer imagenes)...")
temporales = [f for f in os.listdir(POOL_TEMPORAL) if f.endswith(".png")]
todos = [f for f in os.listdir(POOL_TODOS) if f.endswith(".png")]
K_OBJETIVO = por_video(temporales)
print("  pool temporal   : %d frames, %d videos" % (len(temporales), len(K_OBJETIVO)))
print("  pool all lateral: %d frames" % len(todos))
print()

# entropia sintetica, solo para ejercitar la seleccion
rs = np.random.RandomState(0)
ENT = pd.DataFrame({"stem": todos, "H_mean": rs.rand(len(todos)) * 0.2})
ENT["video"] = ENT["stem"].str.extract(r"^(v\d+)_")


def construir(mayor, candidatos_por_objetivo):
    rng = np.random.RandomState(42)
    elegidos = []
    for v, k in K_OBJETIVO.items():
        sub = ENT[ENT.video == v]
        if candidatos_por_objetivo:
            n = min(len(sub), k * candidatos_por_objetivo)
            sub = sub.iloc[rng.choice(len(sub), n, replace=False)]
        sub = sub.sort_values("H_mean", ascending=not mayor)
        assert len(sub) >= k, "FALLO: %s tiene %d candidatos y hacen falta %d" % (
            v, len(sub), k)
        elegidos += list(sub.head(k).stem)
    return elegidos


for modo, m in [("pool entero", None), ("candidato 6x", 6)]:
    print("=" * 70)
    print("MODO:", modo)
    print("=" * 70)
    top = construir(True, m)
    bot = construir(False, m)
    ct, cb = por_video(top), por_video(bot)
    mal_t = [v for v in K_OBJETIVO if K_OBJETIVO[v] != ct.get(v, 0)]
    mal_b = [v for v in K_OBJETIVO if K_OBJETIVO[v] != cb.get(v, 0)]
    solap = set(top) & set(bot)
    fuera = (set(top) | set(bot)) - set(todos)

    print("  tamano objetivo                 : %d" % sum(K_OBJETIVO.values()))
    print("  mas inciertos                   : %d  (videos mal igualados: %d)"
          % (len(top), len(mal_t)))
    print("  menos inciertos                 : %d  (videos mal igualados: %d)"
          % (len(bot), len(mal_b)))
    print("  duplicados dentro de un pool    : %d / %d"
          % (len(top) - len(set(top)), len(bot) - len(set(bot))))
    print("  solapamiento entre los dos      : %d" % len(solap))
    print("  frames que no estan en lateral  : %d" % len(fuera))

    ok = (len(top) == len(bot) == sum(K_OBJETIVO.values())
          and not mal_t and not mal_b and not solap and not fuera
          and len(set(top)) == len(top) and len(set(bot)) == len(bot))
    print("  VEREDICTO:", "OK" if ok else "FALLA")

    hs_top = ENT.set_index("stem").loc[top, "H_mean"].mean()
    hs_bot = ENT.set_index("stem").loc[bot, "H_mean"].mean()
    print("  H media del pool incierto  : %.4f" % hs_top)
    print("  H media del pool cierto    : %.4f" % hs_bot)
    print("  separacion                 : %.4f  (con entropia sintetica; solo confirma"
          " que el orden se aplica)" % (hs_top - hs_bot))
    print()
