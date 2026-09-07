"""Look for LaTeX commands that a shell escape turned into control characters.

Passing a command through a here-document or a quoted shell string can replace
its leading backslash with the character the escape denotes, so that \\times
becomes a tab followed by imes. The result still compiles, because the remaining
letters typeset as text, so the damage is invisible in the log and only shows in
the page. This script counts the control characters that survive in the file and
prints their surroundings, then checks the commands most exposed to the problem.
"""
import io

RUTA = r"G:\My Drive\UNM_vertebras_seg_v3\tesis_latex_overleaf\tesis_final_v54.tex"

with io.open(RUTA, "r", encoding="utf-8", newline="") as fh:
    texto = fh.read()

SOSPECHOSOS = {
    "\t": "tabulador   (de \\t, como en \\times o \\textbf)",
    "\n": "salto       (de \\n, como en \\newline)",
    "\r": "retorno     (de \\r, como en \\ref o \\rho)",
    "\x08": "backspace   (de \\b, como en \\begin o \\bottomrule)",
    "\x0c": "form feed   (de \\f, como en \\footnotesize o \\frac)",
    "\v": "tabulador v (de \\v)",
    "\x07": "campana     (de \\a)",
}

lineas = texto.split("\n")
print("CARACTERES DE CONTROL EN EL FICHERO")
print("=" * 78)
for ch, desc in SOSPECHOSOS.items():
    if ch == "\n":
        continue
    n = texto.count(ch)
    print("  {:11s} {:5d}   {}".format(repr(ch), n, desc))
    if 0 < n <= 12:
        for i, l in enumerate(lineas, 1):
            j = l.find(ch)
            while j != -1:
                print("      linea {}: ...{}...".format(
                    i, repr(l[max(0, j - 45):j + 45])))
                j = l.find(ch, j + 1)

print()
print("COMANDOS QUE DEBERIAN APARECER Y SU CUENTA")
print("=" * 78)
for cmd in ("times", "textbf", "ref", "begin", "footnotesize", "frac", "toprule",
            "bottomrule", "midrule", "pm", "citep", "label", "caption"):
    bueno = texto.count("\\" + cmd)
    print("  \\{:14s} {:4d}".format(cmd, bueno))

print()
print("Un comando con cuenta 0 que si aparece en el texto es la senal de alarma.")
