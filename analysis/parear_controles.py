"""Tell the two ImageNet controls of 4.3 in the same shape.

They are parallel experiments and were written differently: the first took a
twenty-nine word setup before its fifteen word result, the second gave setup and
result in one sentence of twenty. Reading the paragraph straight through, the pattern
the first control sets up is broken by the second. Both now open the same way and give
their number in one sentence, and the colon of the second goes with it. The verbs stay
different because the outcomes were different, one moved nothing and the other dropped.
"""

import io
import os
import re
import shutil
import sys

RUTA = r"G:\My Drive\UNM_vertebras_seg_v3\tesis_latex_overleaf\tesis_final_v54.tex"
COPIA = RUTA + ".ANTES_parear_controles"

APLICAR = "--aplicar" in sys.argv

A1 = ("To assess whether the BiFPN-U-Net(T) result was explained by the absence of "
      "ImageNet initialization, an additional control was run using the same "
      "BiFPN-U-Net(T) architecture with an ImageNet-pretrained VGG16 encoder. This "
      "variant achieved F1~=~0.762~$\\pm$~0.014 across five seeds, similar to the "
      "random-initialized baseline.")
B1 = ("As a control, BiFPN-U-Net(T) was rerun with an ImageNet-pretrained VGG16 "
      "encoder, which left F1 at 0.762~$\\pm$~0.014 across five seeds.")

A2 = ("As a complementary control, U-Net++ was trained from scratch: removing ImageNet "
      "initialization lowered F1 from 0.801~$\\pm$~0.014 to 0.777~$\\pm$~0.024.")
B2 = ("As a complementary control, U-Net++ was trained from scratch, which lowered F1 "
      "from 0.801~$\\pm$~0.014 to 0.777~$\\pm$~0.024.")

with io.open(RUTA, "r", encoding="utf-8", newline="") as fh:
    original = fh.read()
if original.count("\r"):
    sys.exit("ABORTO: retornos de carro.")

texto = original
for n, (a, b) in enumerate([(A1, B1), (A2, B2)], 1):
    if texto.count(a) != 1:
        sys.exit("ABORTO: ancla %d aparece %d veces." % (n, texto.count(a)))
    texto = texto.replace(a, b, 1)


def palabras(s):
    s = re.sub(r"\\citep\{[^}]*\}", "", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = re.sub(r"[{}$~\\^]", " ", s)
    return len(s.split())


def legible(s):
    s = re.sub(r"\\citep\{[^}]*\}", "", s)
    s = s.replace("~=~", " = ").replace("$\\pm$", "+/-").replace("~", " ")
    s = s.replace("{,}", ",").replace("\\", "")
    return " ".join(s.split())


i = texto.index("Performance ranged from BiFPN-U-Net(T)")
par = texto[i:texto.index("\n", i)]
frases = re.split(r"(?<=\.) (?=[A-Z])", par)

print("=" * 94)
print("COMO QUEDA, PARA LEERLO DE CORRIDO")
print("=" * 94)
seguido = legible(par)
for k in range(0, len(seguido), 92):
    print("  " + seguido[k:k + 92])

print()
print("=" * 94)
print("FRASE A FRASE")
print("=" * 94)
for k, f in enumerate(frases, 1):
    print("  F%d (%2d palabras)  %s" % (k, palabras(f), legible(f)[:78]))
print()
print("TOTAL: %d palabras en %d frases  (antes: 164 en 6)" % (palabras(par), len(frases)))

print()
print("=" * 94)
print("CHEQUEO DE ESTILO")
print("=" * 94)
llano = " ".join(par.split())
print("  dos puntos               :", llano.count(":"), "(el de F5, explicativo)")
print("  punto y coma             :", llano.count(";"))
print("  guiones largos (em dash) :", llano.count("\u2014"))
muletillas = ["Moreover", "Furthermore", "It is worth noting", "Notably", "crucial",
              "robust", "leverage", "delve", "In order to", "Importantly",
              "significantly", "comprehensive", "It should be noted"]
print("  muletillas de IA         :", [m for m in muletillas if m in llano] or "ninguna")
print("  longitudes               :", [palabras(f) for f in frases])
print("  aperturas de los controles:")
for f in frases:
    if f.strip().startswith("As "):
        print("      " + legible(f)[:86])

if not APLICAR:
    print()
    print("MODO LECTURA. No se escribio nada. Anadir --aplicar para escribir.")
    sys.exit(0)

if not os.path.exists(COPIA):
    shutil.copy2(RUTA, COPIA)
with io.open(RUTA, "w", encoding="utf-8", newline="") as fh:
    fh.write(texto)
with io.open(RUTA, "r", encoding="utf-8", newline="") as fh:
    escrito = fh.read()
print()
print("copia de seguridad :", os.path.basename(COPIA))
print("CR sueltos         :", escrito.count("\r"))
print("bytes              :", len(original), "->", len(escrito),
      "(%+d)" % (len(escrito) - len(original)))
