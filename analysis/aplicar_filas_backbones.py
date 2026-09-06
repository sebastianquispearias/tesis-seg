"""Add the two missing backbones to tab:arch and update the paragraph that reads it.

The semi-supervised block of the table listed four architectures while the
supervised block listed six. The runs for the two that were missing, TransUNet
and BiFPN-U-Net(T), finished on 5 September, so the block can be completed and
the six architectures can be read as pairs.

Two gates run before anything is written. The rows about to be inserted are
recomputed from the run reports and compared against the literal text, so a
number that drifted stops the edit; and every run behind them has to show a
non-zero unsupervised loss, because a run that never received its unlabeled pool
is not a semi-supervised result whatever its configuration claims.

The paragraph is updated in the same pass. Leaving the table ahead of the prose
would have the text count four backbones while the table shows six.
"""

import glob
import json
import math
import os
import shutil
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEX = os.path.join(BASE, "tesis_latex_overleaf", "tesis_final_v54.tex")
COPIA = TEX + ".ANTES_filas_backbones"

NUEVOS = {
    "TransUNet": "runs_ssl_backbones/mean_teacher_transunet_std_matched_r15",
    "BiFPN-U-Net(T)": "runs_ssl_backbones/mean_teacher_bifpn_unet_std_matched_r15",
}

# ------------------------------------------------------------------ la tabla
TABLA_ANTES = (
    r"\quad DeepLabV3+      & .800$\pm$.014 & 2.24$\pm$0.25 & 8.17$\pm$1.06 \\" "\n"
    r"\quad U-Net++         & .830$\pm$.038 & 1.92$\pm$0.22 & 7.80$\pm$0.93 \\" "\n"
    r"\quad FPN             & .811$\pm$.028 & 2.24$\pm$0.13 & 7.74$\pm$0.86 \\" "\n"
    r"\quad U-Net           & .851$\pm$.005 & 1.64$\pm$0.05 & 6.06$\pm$0.74 \\" "\n"
)
TABLA_DESPUES = (
    r"\quad DeepLabV3+      & .800$\pm$.014 & 2.24$\pm$0.25 & 8.17$\pm$1.06 \\" "\n"
    r"\quad BiFPN-U-Net(T)  & .757$\pm$.029 & 5.26$\pm$2.60 & 12.59$\pm$4.24 \\" "\n"
    r"\quad U-Net++         & .830$\pm$.038 & 1.92$\pm$0.22 & 7.80$\pm$0.93 \\" "\n"
    r"\quad FPN             & .811$\pm$.028 & 2.24$\pm$0.13 & 7.74$\pm$0.86 \\" "\n"
    r"\quad U-Net           & .851$\pm$.005 & 1.64$\pm$0.05 & 6.06$\pm$0.74 \\" "\n"
    r"\quad TransUNet       & .831$\pm$.005 & 2.09$\pm$0.15 & 8.72$\pm$0.88 \\" "\n"
)

# --------------------------------------------------------------- el parrafo
PARRAFO_ANTES = (
    r"was rerun with three additional backbones. U-Net improved from F1~=~0.824 to "
    r"0.851~$\pm$~0.005, FPN changed only slightly from 0.799 to 0.811~$\pm$~0.028, and "
    r"DeepLabV3+ from 0.776 to 0.800~$\pm$~0.014. Under that same setting U-Net++ itself "
    r"improved from 0.801 to 0.830~$\pm$~0.038 (Table~\ref{tab:arch}), so all four "
    r"backbones can be compared under one configuration. These additional runs suggest "
    r"that the use of unlabeled frames is not tied only to U-Net++, although the detailed "
    r"analysis of pool size and selection policy remains based on the U-Net++ pipeline."
)
PARRAFO_DESPUES = (
    r"was rerun with five additional backbones. U-Net improved from F1~=~0.824 to "
    r"0.851~$\pm$~0.005, DeepLabV3+ from 0.776 to 0.800~$\pm$~0.014, and FPN changed only "
    r"slightly from 0.799 to 0.811~$\pm$~0.028. BiFPN-U-Net(T) did not move, from 0.759 to "
    r"0.757~$\pm$~0.029, and TransUNet fell from 0.851 to 0.831~$\pm$~0.005. Under that "
    r"same setting U-Net++ itself improved from 0.801 to 0.830~$\pm$~0.038 "
    r"(Table~\ref{tab:arch}), so all six backbones can be compared under one "
    r"configuration. These additional runs suggest that the use of unlabeled frames is "
    r"not tied only to U-Net++, though it does not reach every backbone. Four of the six "
    r"gain between 0.012 and 0.029 F1; BiFPN-U-Net(T) does not move, and TransUNet, the "
    r"strongest supervised architecture in this pipeline, loses 0.020. The detailed "
    r"analysis of pool size and selection policy remains based on the U-Net++ pipeline."
)


def metricas(carpeta):
    """Mean and SD across seeds, and the largest unsupervised loss of each run."""
    f1s, assds, hd95s, unsup = [], [], [], []
    for d in sorted(glob.glob(os.path.join(BASE, carpeta, "seed_*"))):
        ps = glob.glob(os.path.join(d, "*run_report.json"))
        if not ps:
            continue
        with open(ps[0], "r", encoding="utf-8") as fh:
            r = json.load(fh)
        tm, bm = r.get("test_metrics", {}), r.get("test_boundary_metrics", {})
        if "sample_mean_f1" not in tm:
            continue
        f1s.append(tm["sample_mean_f1"])
        assds.append(bm.get("assd_mean_px"))
        hd95s.append(bm.get("hd95_mean_px"))
        hist = r.get("epoch_history") or []
        unsup.append(max([(e.get("unsup_loss") or 0.0) for e in hist] or [0.0]))
    return f1s, assds, hd95s, unsup


def ms(v):
    m = sum(v) / len(v)
    s = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
    return m, s


def celda_f1(v):
    m, s = ms(v)
    return ("%.3f" % m).lstrip("0") + r"$\pm$" + ("%.3f" % s).lstrip("0")


def celda_px(v):
    m, s = ms(v)
    return "%.2f$\\pm$%.2f" % (m, s)


print("=" * 74)
print("PUERTA 1: los runs consumieron datos sin etiquetar")
print("=" * 74)
cerrada = []
filas = {}
for arq, carpeta in NUEVOS.items():
    f1s, assds, hd95s, unsup = metricas(carpeta)
    if len(f1s) < 3:
        cerrada.append("{}: solo {} semillas".format(arq, len(f1s)))
        continue
    for i, u in enumerate(unsup):
        estado = "OK" if u > 0 else "SIN POOL"
        print("  {:16s} seed_{}  max(unsup_loss) = {:.6f}   {}".format(
            arq, i, u, estado))
        if u <= 0:
            cerrada.append("{}/seed_{}: unsup_loss cero".format(arq, i))
    filas[arq] = (celda_f1(f1s), celda_px(assds), celda_px(hd95s))

print()
print("=" * 74)
print("PUERTA 2: las filas del texto coinciden con los informes")
print("=" * 74)
for arq, celdas in filas.items():
    # Se comparan las tres celdas, no la linea entera: el relleno de columna es
    # cosmetico y sigue el de la tabla, mientras que un numero distinto no puede
    # pasar.
    linea = [l for l in TABLA_DESPUES.splitlines()
             if l.startswith(r"\quad " + arq + " ")]
    if len(linea) != 1:
        cerrada.append("{}: no hay una unica fila suya en el texto".format(arq))
        print("  {:16s} NO HAY FILA".format(arq))
        continue
    faltan = [c for c in celdas if c not in linea[0]]
    print("  {:16s} {}   {}".format(
        arq, "OK" if not faltan else "NO COINCIDE", "  ".join(celdas)))
    if faltan:
        print("      en el texto : " + linea[0].strip())
        print("      no aparece  : " + ", ".join(faltan))
        cerrada.append("{}: cifra distinta entre informes y texto".format(arq))

if cerrada:
    print()
    print("PUERTA CERRADA. No se toca el .tex.")
    for c in cerrada:
        print("  -", c)
    sys.exit(1)

print()
print("PUERTAS ABIERTAS.")

# ------------------------------------------------------------------ escribir
with open(TEX, "rb") as fh:
    crudo = fh.read()
texto = crudo.decode("utf-8")

for etiqueta, antes, despues in (("tabla", TABLA_ANTES, TABLA_DESPUES),
                                 ("parrafo", PARRAFO_ANTES, PARRAFO_DESPUES)):
    n = texto.count(antes)
    print("  ancla '{}': {} aparicion(es)".format(etiqueta, n))
    if n != 1:
        print("ANCLA NO UNICA. No se escribe nada.")
        sys.exit(1)

if not os.path.exists(COPIA):
    shutil.copy2(TEX, COPIA)

nuevo = texto.replace(TABLA_ANTES, TABLA_DESPUES).replace(
    PARRAFO_ANTES, PARRAFO_DESPUES)
with open(TEX, "wb") as fh:
    fh.write(nuevo.encode("utf-8"))

with open(TEX, "rb") as fh:
    escrito = fh.read()
d = escrito.decode("utf-8")
vuelta = d.replace(TABLA_DESPUES, TABLA_ANTES).replace(
    PARRAFO_DESPUES, PARRAFO_ANTES)

cr = sum(1 for i, b in enumerate(escrito)
         if b == 0x0D and (i + 1 >= len(escrito) or escrito[i + 1] != 0x0A))
lf = sum(1 for i, b in enumerate(escrito)
         if b == 0x0A and (i == 0 or escrito[i - 1] != 0x0D))

print()
print("copia            :", os.path.basename(COPIA))
print("bytes antes      :", len(crudo))
print("bytes despues    :", len(escrito))
print("reversible       :", "SI" if vuelta.encode("utf-8") == crudo else "NO")
print("CR sueltos       :", cr)
print("LF sueltos       :", lf)
print("filas del bloque :", d.count(r"\quad TransUNet"), "TransUNet,",
      d.count(r"\quad BiFPN-U-Net(T)"), "BiFPN (2 de cada: supervisada y SSL)")
print("'all six'        :", "all six backbones" in d)
print("'three additional' fuera:", "three additional backbones" not in d)
