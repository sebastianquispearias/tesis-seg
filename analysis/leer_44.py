"""Print section 4.4 sentence by sentence, in readable form."""

import io
import re

RUTA = r"G:\My Drive\UNM_vertebras_seg_v3\tesis_latex_overleaf\tesis_final_v54.tex"
t = io.open(RUTA, encoding="utf-8").read()

i = t.index("\\section{Complementary analyses}")
j = t.index("\\section{Predictive uncertainty}")
sec = t[i:j]


def legible(s):
    s = re.sub(r"\\citep\{([^}]*)\}", r"[\1]", s)
    s = re.sub(r"\\ref\{([^}]*)\}", r"[\1]", s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", s)
    s = re.sub(r"\\emph\{([^}]*)\}", r"*\1*", s)
    s = s.replace("~=~", " = ").replace("$\\pm$", "+/-").replace("{,}", ",")
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = s.replace("{", "").replace("}", "").replace("$", "").replace("~", " ")
    return " ".join(s.split())


def palabras(s):
    return len(legible(s).split())


n = 0
for par in [p.strip() for p in sec.split("\n\n") if p.strip()]:
    if par.startswith(("\\section", "\\label", "\\begin", "\\end", "\\clearpage")):
        continue
    if palabras(par) < 8:
        continue
    n += 1
    print("=" * 92)
    print("PARRAFO %d   (%d palabras)" % (n, palabras(par)))
    print("=" * 92)
    for f in re.split(r"(?<=\.) (?=[A-Z*])", par):
        txt = legible(f)
        if not txt:
            continue
        print("  (%2d) %s" % (palabras(f), txt[:88]))
        for k in range(88, len(txt), 88):
            print("       " + txt[k:k + 88])
    print()
