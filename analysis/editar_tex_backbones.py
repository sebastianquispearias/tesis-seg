"""Add the two missing semi-supervised backbone rows to the manuscript.

Runs the same gate as filas_backbones.py first: unless the four rows already in
tab:arch can be rebuilt from the reports on disk, the arithmetic behind the two new
ones cannot be trusted and nothing is written.

Without --aplicar it only prints the before and after. The file is stored with LF
endings and is rewritten with LF endings.
"""

import glob
import io
import json
import os
import shutil
import statistics as st
import sys

BASE = r"G:\My Drive\UNM_vertebras_seg_v3"
TEX = os.path.join(BASE, "tesis_latex_overleaf", "tesis_final_v54.tex")
COPIA = TEX + ".ANTES_backbones_6de6"

APLICAR = "--aplicar" in sys.argv

IMPRESO = {
    "U-Net++":    (".830$\\pm$.038", "1.92$\\pm$0.22", "7.80$\\pm$0.93"),
    "U-Net":      (".851$\\pm$.005", "1.64$\\pm$0.05", "6.06$\\pm$0.74"),
    "FPN":        (".811$\\pm$.028", "2.24$\\pm$0.13", "7.74$\\pm$0.86"),
    "DeepLabV3+": (".800$\\pm$.014", "2.24$\\pm$0.25", "8.17$\\pm$1.06"),
}
DIRS_IMPRESOS = {
    "U-Net++":    "runs_final_v1/mean_teacher_std_matched_r15",
    "U-Net":      "runs_final_v1/mean_teacher_unet_std_matched_r15",
    "FPN":        "runs_final_v1/mean_teacher_fpn_std_matched_r15",
    "DeepLabV3+": "runs_final_v1/mean_teacher_deeplabv3plus_std_matched_r15",
}
NUEVOS = [
    ("TransUNet",      "runs_ssl_backbones/mean_teacher_transunet_std_matched_r15", 0.851),
    ("BiFPN-U-Net(T)", "runs_ssl_backbones/mean_teacher_bifpn_unet_std_matched_r15", 0.759),
]


def metricas(carpeta):
    f1s, assds, hd95s = [], [], []
    for d in sorted(glob.glob(os.path.join(BASE, carpeta, "seed_*"))):
        ps = glob.glob(os.path.join(d, "*run_report.json"))
        if not ps:
            continue
        r = json.load(open(ps[0]))
        tm, bm = r.get("test_metrics", {}), r.get("test_boundary_metrics", {})
        if "sample_mean_f1" not in tm:
            continue
        f1s.append(tm["sample_mean_f1"])
        assds.append(bm.get("assd_mean_px"))
        hd95s.append(bm.get("hd95_mean_px"))
    return f1s, assds, hd95s


def celda_f1(v):
    m, s = st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)
    return ("%.3f" % m).lstrip("0") + "$\\pm$" + ("%.3f" % s).lstrip("0")


def celda_px(v):
    m, s = st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)
    return "%.2f$\\pm$%.2f" % (m, s)


def prosa_f1(v):
    """0.830~$\\pm$~0.038, the form the running text uses."""
    m, s = st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)
    return "%.3f~$\\pm$~%.3f" % (m, s)


# --------------------------------------------------------------- puerta
print("PUERTA: reproducir las cuatro filas impresas desde disco")
for arq, carpeta in DIRS_IMPRESOS.items():
    f1s, assds, hd95s = metricas(carpeta)
    calc = (celda_f1(f1s), celda_px(assds), celda_px(hd95s)) if f1s else None
    if calc != IMPRESO[arq]:
        print(f"  {arq}: FALLA. impreso {IMPRESO[arq]} calculado {calc}")
        sys.exit("PUERTA CERRADA. No se toca el .tex.")
    print(f"  {arq:16s} OK")
print("PUERTA ABIERTA.")
print()

# ------------------------------------------------------- las filas nuevas
filas, frases = [], []
for arq, carpeta, sup in NUEVOS:
    f1s, assds, hd95s = metricas(carpeta)
    if len(f1s) < 3:
        sys.exit(f"AUN NO: {arq} tiene {len(f1s)} de 3 semillas.")
    filas.append("\\quad %-16s & %s & %s & %s \\\\"
                 % (arq, celda_f1(f1s), celda_px(assds), celda_px(hd95s)))
    frases.append((arq, sup, prosa_f1(f1s)))

with io.open(TEX, "r", encoding="utf-8", newline="") as fh:
    original = fh.read()
if original.count("\r"):
    sys.exit("ABORTO: el .tex tiene retornos de carro y se esperaba solo LF.")
texto = original

# ------------------------------------------------------- 1. la tabla
ANCLA_FILA = ("\\quad DeepLabV3+      & .800$\\pm$.014 & 2.24$\\pm$0.25 "
              "& 8.17$\\pm$1.06 \\\\")
assert texto.count(ANCLA_FILA) == 1, "ancla de la fila no es unica"
texto = texto.replace(ANCLA_FILA, ANCLA_FILA + "\n" + "\n".join(filas), 1)

# --------------------------------------------------- 2. tres / cinco
A2 = "was rerun with three additional backbones"
B2 = "was rerun with five additional backbones"
assert texto.count(A2) == 1, "ancla de 'three additional' no es unica"
texto = texto.replace(A2, B2, 1)

# ------------------------------------------- 3. la frase con los numeros
A3 = "and DeepLabV3+ from 0.776 to 0.800~$\\pm$~0.014."
B3 = (A3 + " TransUNet, the strongest supervised architecture in this pipeline, "
      "went from 0.851 to %s, and BiFPN-U-Net(T), the only one without ImageNet "
      "initialization, from 0.759 to %s." % (frases[0][2], frases[1][2]))
assert texto.count(A3) == 1, "ancla de la frase no es unica"
texto = texto.replace(A3, B3, 1)

# ----------------------------------------------------- 4. cuatro / seis
A4 = "so all four backbones can be compared under one configuration"
B4 = "so all six backbones can be compared under one configuration"
assert texto.count(A4) == 1, "ancla de 'all four' no es unica"
texto = texto.replace(A4, B4, 1)

# ------------------------------------------------------------- informe
print("=" * 78)
print("FILAS NUEVAS EN tab:arch, detras de DeepLabV3+")
print("=" * 78)
for f in filas:
    print("  " + f)
print()
print("=" * 78)
print("PROSA: ANTES")
print("=" * 78)
print("  ...", A2, "...")
print("  ...", A3)
print("  ...", A4, "...")
print()
print("PROSA: DESPUES")
print("=" * 78)
print("  ...", B2, "...")
print("  ...", B3)
print("  ...", B4, "...")
print()

if not APLICAR:
    print("MODO LECTURA. No se escribio nada. Anadir --aplicar para escribir.")
    sys.exit(0)

if not os.path.exists(COPIA):
    shutil.copy2(TEX, COPIA)
with io.open(TEX, "w", encoding="utf-8", newline="") as fh:
    fh.write(texto)
with io.open(TEX, "r", encoding="utf-8", newline="") as fh:
    escrito = fh.read()

print("copia de seguridad :", os.path.basename(COPIA))
print("CR sueltos         :", escrito.count("\r"))
print("lineas             :", original.count("\n"), "->", escrito.count("\n"))
print("bytes              :", len(original), "->", len(escrito))
print()
print("AHORA: bash tesis_latex_overleaf/compilar.sh")
print("y comprobar en ~/tesis_build/p4.log: paginas, 0 errores, 0 refs sin resolver.")
