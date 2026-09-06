"""Print one code cell of a notebook, or a line range of it, as plain text.

Writes through a UTF-8 stream so that arrows and accents in the source do not
crash on a console using a legacy code page.
"""
import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

path = sys.argv[1]
celda = int(sys.argv[2])
desde = int(sys.argv[3]) if len(sys.argv) > 3 else 0
hasta = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9

with open(path, "r", encoding="utf-8") as fh:
    nb = json.load(fh)

src = "".join(nb["cells"][celda].get("source", []))
for j, line in enumerate(src.split("\n")):
    if desde <= j <= hasta:
        print("{:>3}: {}".format(j, line.rstrip()))
