"""Break the architecture-comparison paragraph of 4.3 into sentences and size them."""

import io
import re

RUTA = r"G:\My Drive\UNM_vertebras_seg_v3\tesis_latex_overleaf\tesis_final_v54.tex"
t = io.open(RUTA, encoding="utf-8").read()

i = t.index("Performance ranged from BiFPN-U-Net(T)")
par = t[i:t.index("\n", i)]


def palabras(s):
    s = re.sub(r"\\citep\{[^}]*\}", "", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = re.sub(r"[{}$~\\^]", " ", s)
    return len(s.split())


frases = re.split(r"(?<=\.) (?=[A-Z])", par)
for k, f in enumerate(frases, 1):
    print("  F%d  (%d palabras)" % (k, palabras(f)))
    print("      " + " ".join(f.split()))
    print()
print("TOTAL: %d palabras en %d frases" % (palabras(par), len(frases)))
