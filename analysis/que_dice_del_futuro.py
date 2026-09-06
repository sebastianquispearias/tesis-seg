"""What Limitations, the Conclusion and appendix A.2 already say about the rule."""

import io
import re

RUTA = r"G:\My Drive\UNM_vertebras_seg_v3\tesis_latex_overleaf\tesis_final_v54.tex"
t = io.open(RUTA, encoding="utf-8").read()


def legible(s):
    s = re.sub(r"\\citep\{([^}]*)\}", r"[\1]", s)
    s = re.sub(r"\\ref\{([^}]*)\}", r"[\1]", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    return " ".join(s.replace("{", "").replace("}", "").replace("$", "")
                    .replace("~", " ").split())


TROZOS = [
    ("LIMITATIONS", "\\section{Limitations}", "\\chapter{Conclusion}"),
    ("CONCLUSION", "\\chapter{Conclusion}", "\\appendix" if "\\appendix" in t
     else "\\end{document}"),
]

for nombre, ini, fin in TROZOS:
    try:
        a, b = t.index(ini), t.index(fin)
    except ValueError:
        continue
    bloque = t[a:b]
    print("=" * 92)
    print(nombre, "  (%d palabras)" % len(legible(bloque).split()))
    print("=" * 92)
    encontrado = False
    for par in [p.strip() for p in bloque.split("\n\n") if p.strip()]:
        txt = legible(par)
        if any(k in txt.lower() for k in
               ["landmark", "post-process", "scalar", "rule", "future"]):
            encontrado = True
            print("  ---")
            for k in range(0, len(txt), 90):
                print("  " + txt[k:k + 90])
    if not encontrado:
        print("  (ninguna mencion a landmark / post-process / scalar / rule / future)")
    print()

print("=" * 92)
print("BUSQUEDA DE 'FUTURE' EN TODO EL DOCUMENTO")
print("=" * 92)
for m in re.finditer(r"[Ff]uture", t):
    ini = max(0, m.start() - 170)
    print("  l.%-5d ...%s..." % (t[:m.start()].count("\n") + 1,
                                 legible(t[ini:m.end() + 170])))
    print()
