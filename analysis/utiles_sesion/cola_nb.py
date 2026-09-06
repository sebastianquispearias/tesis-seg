"""Print the code lines of a notebook that define the run queue.

Scans every code cell and prints the lines that mention seeds, the run list or
the experiment names, so the planned queue can be compared with what is on disk.
"""
import json
import re
import sys

path = sys.argv[1]
pattern = re.compile(sys.argv[2], re.IGNORECASE)

with open(path, "r", encoding="utf-8") as fh:
    nb = json.load(fh)

for i, cell in enumerate(nb.get("cells", [])):
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    for j, line in enumerate(src.split("\n")):
        if pattern.search(line):
            print("celda {:>3} linea {:>3}: {}".format(i, j, line.rstrip()))
