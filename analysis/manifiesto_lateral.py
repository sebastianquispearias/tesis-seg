"""Write the file list of the lateral pool, so Colab never has to list it.

The Drive FUSE mount fails with Errno 5 when asked to enumerate a directory holding
tens of thousands of entries, which is what killed the second gate of notebook 22.
Reading the names from a file sidesteps every readdir on that folder, and tar can then
be pointed at the list instead of at the directory.
"""

import io
import json
import os
import re
import collections

BASE = r"G:\My Drive\UNM_vertebras_seg_v3"
POOL = os.path.join(BASE, "unlabeling_all_lateral", "images")
SALIDA_TXT = os.path.join(BASE, "unlabeling_all_lateral", "manifest_all_lateral.txt")
SALIDA_JSON = os.path.join(BASE, "unlabeling_all_lateral", "manifest_all_lateral.json")

print("listando (esto en local si funciona)...")
nombres = sorted(f for f in os.listdir(POOL) if f.lower().endswith(".png"))
print("  archivos:", len(nombres))

por_video = collections.Counter()
malos = []
for f in nombres:
    m = re.match(r"(v\d+)_f(\d+)\.png$", f)
    if m:
        por_video[m.group(1)] += 1
    else:
        malos.append(f)

print("  videos  :", len(por_video))
print("  nombres que no encajan en v<id>_f<n>.png:", len(malos))
if malos:
    print("   ", malos[:5])

with io.open(SALIDA_TXT, "w", encoding="utf-8", newline="\n") as fh:
    fh.write("\n".join(nombres) + "\n")

with io.open(SALIDA_JSON, "w", encoding="utf-8", newline="\n") as fh:
    json.dump({"n": len(nombres),
               "por_video": dict(sorted(por_video.items())),
               "archivos": nombres}, fh)

print()
print("escrito:")
for p in (SALIDA_TXT, SALIDA_JSON):
    print("  %-70s %.1f MB" % (os.path.basename(p), os.path.getsize(p) / 1024 ** 2))

# relectura, para no fiarse
with io.open(SALIDA_TXT, encoding="utf-8") as fh:
    vuelta = [l.strip() for l in fh if l.strip()]
print()
print("relectura del txt: %d nombres, identicos al listado: %s"
      % (len(vuelta), vuelta == nombres))
print("primeros:", vuelta[:3])
print("ultimos :", vuelta[-3:])
