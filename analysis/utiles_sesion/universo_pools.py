"""Check that the four pools were drawn from the same universe of frames.

The two uncertainty pools are built by notebook 22 out of the all-lateral set,
while the random and temporal pools were built earlier by other scripts. If those
earlier pools came from a narrower universe, their disadvantage could come from
where they were drawn rather than from the absence of a selection criterion,
which would make the comparison say something different from what it seems to.
"""
import collections
import os
import re

BASE = r"G:\My Drive\UNM_vertebras_seg_v3"
MANIFIESTO = os.path.join(BASE, "unlabeling_all_lateral", "manifest_all_lateral.txt")

POOLS = ["unlabeling_std_matched_r10", "unlabeling_r10_max0",
         "unlabeling_uncertainty_top_r10", "unlabeling_uncertainty_bottom_r10"]


def nombres(pool):
    d = os.path.join(BASE, pool, "images")
    return {e.name for e in os.scandir(d)
            if e.is_file() and e.name.lower().endswith(".png")}


def por_video(ns):
    c = collections.Counter()
    for f in ns:
        m = re.match(r"(v\d+)_f\d+\.png$", f)
        if m:
            c[m.group(1)] += 1
    return c


with open(MANIFIESTO, "r", encoding="utf-8") as fh:
    todos = {l.strip() for l in fh if l.strip().endswith(".png")}
print("universo all-lateral (manifiesto): {} frames".format(len(todos)))
print()

conj = {}
print("{:36s} {:>7} {:>22} {:>8}".format(
    "pool", "frames", "dentro de all-lateral", "videos"))
print("-" * 78)
for p in POOLS:
    ns = nombres(p)
    conj[p] = ns
    dentro = len(ns & todos)
    print("{:36s} {:7d} {:>18d} {:>3.0f}% {:8d}".format(
        p, len(ns), dentro, 100.0 * dentro / max(1, len(ns)), len(por_video(ns))))

print()
print("COMPOSICION POR VIDEO: los cuatro tienen que coincidir")
ref = por_video(conj["unlabeling_r10_max0"])
for p in POOLS:
    c = por_video(conj[p])
    mal = [v for v in ref if ref[v] != c.get(v, 0)]
    print("  {:36s} videos que no cuadran con temporal: {}".format(p, len(mal)))

print()
print("SOLAPAMIENTO ENTRE POOLS")
for i, a in enumerate(POOLS):
    for b in POOLS[i + 1:]:
        inter = len(conj[a] & conj[b])
        print("  {:34s} n {:34s} = {:5d}".format(a, b, inter))
