"""Trim two captions that repeat what the figure or the running text already says.

The teacher-student caption narrates arrows the diagram already labels (weak aug,
strong aug, EMA, match loss) and restates the method that Section 3 develops in full;
only the soft/hard label needs decoding, so only that sentence survives. The scalar
caption ends on an interpretation that the surrounding paragraph states almost word
for word.
"""

import io
import os
import shutil
import sys

RUTA = r"G:\My Drive\UNM_vertebras_seg_v3\tesis_latex_overleaf\tesis_final_v54.tex"
COPIA = RUTA + ".ANTES_recorte_pies"

APLICAR = "--aplicar" in sys.argv

# ------------------------------------------------------------------ recorte 1
A1 = ("{Teacher--student framework shared by Pseudo-Labeling and Mean Teacher. "
      "An unlabeled frame is fed through the teacher (with a weak augmentation) and "
      "through the student (with a strong augmentation). The teacher's output "
      "supervises the student's output via a matching loss; the teacher is an EMA of "
      "the student's weights. The two methods differ in whether the teacher's output "
      "is binarized (PL) or used as a continuous probability map (MT).}")

B1 = ("{Teacher--student framework shared by Pseudo-Labeling and Mean Teacher. "
      "The two methods differ only at the \\emph{soft / hard} step: PL binarizes the "
      "teacher's output, MT keeps it as a probability map.}")

# ------------------------------------------------------------------ recorte 2
A2 = (" the median. The reference masks are correct by construction, so the error "
      "they produce\n  belongs to the rule and not to the segmentation.}")

B2 = " the median.}"

with io.open(RUTA, "r", encoding="utf-8", newline="") as fh:
    original = fh.read()

if original.count("\r"):
    sys.exit("ABORTO: el .tex tiene retornos de carro y se esperaba solo LF.")

texto = original
for n, (a, b) in enumerate([(A1, B1), (A2, B2)], start=1):
    if texto.count(a) != 1:
        sys.exit("ABORTO: el ancla %d aparece %d veces, no 1." % (n, texto.count(a)))
    texto = texto.replace(a, b, 1)


def palabras(s):
    import re
    return len(re.sub(r"\\[a-zA-Z]+|[{}$~\\]", " ", s).split())


print("=" * 78)
print("RECORTE 1 - pie del diagrama profesor-alumno")
print("=" * 78)
print("ANTES  (%d palabras)" % palabras(A1))
print("  " + " ".join(A1[1:-1].split()))
print()
print("DESPUES (%d palabras)" % palabras(B1))
print("  " + " ".join(B1[1:-1].split()))
print()
print("Se van las frases que describen flechas que el propio dibujo ya rotula:")
print("  Teacher | Student | weak aug | strong aug | EMA | match (loss) | soft / hard")
print("Se queda la que descifra 'soft / hard', la unica etiqueta que no se explica sola.")
print()
print("=" * 78)
print("RECORTE 2 - ultima frase del pie de fig:c2c4floor")
print("=" * 78)
print("SE QUITA:")
print("  The reference masks are correct by construction, so the error they produce")
print("  belongs to the rule and not to the segmentation.")
print()
print("MOTIVO: el cuerpo lo dice casi literal ->")
print("  \"Applying that rule to the reference masks, where the segmentation is")
print("   correct by construction, still leaves a median error of 6.9~px\"")
print()

if not APLICAR:
    print("MODO LECTURA. No se escribio nada. Anadir --aplicar para escribir.")
    sys.exit(0)

if not os.path.exists(COPIA):
    shutil.copy2(RUTA, COPIA)
with io.open(RUTA, "w", encoding="utf-8", newline="") as fh:
    fh.write(texto)
with io.open(RUTA, "r", encoding="utf-8", newline="") as fh:
    escrito = fh.read()

print("copia de seguridad :", os.path.basename(COPIA))
print("CR sueltos         :", escrito.count("\r"))
print("lineas             :", original.count("\n"), "->", escrito.count("\n"))
print("bytes              :", len(original), "->", len(escrito))
