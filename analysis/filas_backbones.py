"""Build the two missing rows of tab:arch out of the notebook 20 runs.

The formatting routine is checked against the four semi-supervised rows already in
the manuscript before it is trusted with the two new ones: if it cannot reproduce
what is printed, from the reports on disk, it is wrong and the script stops.

Also reports whether the September stack moved the supervised baseline, which is
what decides whether the two new rows can sit next to the four July ones.

Read only. Applies nothing.
"""

import glob
import json
import os
import statistics as st
import sys

BASE = r"G:\My Drive\UNM_vertebras_seg_v3"

# lo que ya esta impreso, copiado del .tex, para usarlo como puerta
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

NUEVOS = {
    "TransUNet":        "runs_ssl_backbones/mean_teacher_transunet_std_matched_r15",
    "BiFPN-U-Net(T)":   "runs_ssl_backbones/mean_teacher_bifpn_unet_std_matched_r15",
}

CONTROLES = {
    "TransUNet":      ("runs_ssl_backbones/supervised_transunet_std_matched_r15", 0.851),
    "BiFPN-U-Net(T)": ("runs_ssl_backbones/supervised_bifpn_unet_std_matched_r15", 0.759),
}


def metricas(carpeta):
    """Mean and SD across seeds of the three metrics the table reports."""
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


def uso_pool(carpeta):
    """Per seed, the largest unsupervised loss the run ever recorded.

    A Mean Teacher run that received its unlabeled pool drives this above zero.
    A run whose unlabeled loader was never built leaves it at zero for every
    epoch while its configuration still claims use_semi and a non-zero lambda_u,
    so the configuration cannot be trusted and the history has to be read.
    """
    out = {}
    for d in sorted(glob.glob(os.path.join(BASE, carpeta, "seed_*"))):
        ps = glob.glob(os.path.join(d, "*run_report.json"))
        if not ps:
            continue
        r = json.load(open(ps[0]))
        hist = r.get("epoch_history") or []
        out[os.path.basename(d)] = max(
            [e.get("unsup_loss") or 0.0 for e in hist] or [0.0])
    return out


def celda_f1(vals):
    m = st.mean(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    return ("%.3f" % m).lstrip("0") + "$\\pm$" + ("%.3f" % s).lstrip("0")


def celda_px(vals):
    m = st.mean(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    return "%.2f$\\pm$%.2f" % (m, s)


# ------------------------------------------------- 1. la puerta: reproducir lo impreso
print("=" * 78)
print("PUERTA: reproducir desde disco las CUATRO filas que ya estan en la tesis")
print("=" * 78)
fallos = []
for arq, carpeta in DIRS_IMPRESOS.items():
    f1s, assds, hd95s = metricas(carpeta)
    if not f1s:
        fallos.append(f"{arq}: sin reportes en {carpeta}")
        print(f"  {arq:16s} SIN DATOS en {carpeta}")
        continue
    calc = (celda_f1(f1s), celda_px(assds), celda_px(hd95s))
    esp = IMPRESO[arq]
    ok = calc == esp
    if not ok:
        fallos.append(f"{arq}: calculado != impreso")
    print(f"  {arq:16s} n={len(f1s)}  {'OK' if ok else 'FALLA'}")
    if not ok:
        print(f"      impreso   : {esp}")
        print(f"      calculado : {calc}")

if fallos:
    print()
    print("PUERTA CERRADA. La forma de calcular no reproduce la tabla, asi que no se")
    print("puede confiar en ella para las filas nuevas. No editar el .tex.")
    for f in fallos:
        print("  -", f)
    sys.exit(1)

print()
print("PUERTA ABIERTA: la misma cuenta reproduce las cuatro filas impresas.")

# ------------------------------------- 2. el stack movio el supervisado o no
print()
print("=" * 78)
print("CONTROL DE STACK: el supervisado de septiembre contra el de la tesis")
print("=" * 78)
print("Si estos coinciden, las filas nuevas pueden ponerse al lado de las viejas.")
print()
for arq, (carpeta, tesis) in CONTROLES.items():
    f1s, _, _ = metricas(carpeta)
    if not f1s:
        print(f"  {arq:16s} todavia sin resultados")
        continue
    m = st.mean(f1s)
    print(f"  {arq:16s} tesis {tesis:.3f}   septiembre {m:.4f} (n={len(f1s)})   "
          f"diferencia {m - tesis:+.4f}")

# ------------------------------- 2b. la puerta del pool: se uso o no se uso
print()
print("=" * 78)
print("PUERTA DEL POOL: comprobar que los runs SSL consumieron datos sin etiquetar")
print("=" * 78)
print("El cfg dice use_semi y lambda_u, pero eso es la INTENCION. Lo que decide es")
print("si unsup_loss llego a moverse alguna epoca.")
print()
sin_pool = []
for arq, carpeta in list(DIRS_IMPRESOS.items()) + list(NUEVOS.items()):
    for seed, mx in sorted(uso_pool(carpeta).items()):
        marca = "OK" if mx > 0 else "SIN POOL"
        if mx <= 0:
            sin_pool.append(f"{arq}/{seed}")
        print(f"  {arq:16s} {seed:8s} max(unsup_loss) = {mx:.6f}   {marca}")

if sin_pool:
    print()
    print("PUERTA CERRADA. Estos runs no consumieron ni un fotograma sin etiquetar,")
    print("asi que no son semi-supervisados y no pueden ir en el bloque SSL de la")
    print("tabla. Hay que arreglar el notebook y relanzarlos. No editar el .tex.")
    for r in sin_pool:
        print("  -", r)
    sys.exit(1)

print()
print("PUERTA ABIERTA: todos los runs SSL movieron la perdida no supervisada.")

# --------------------------------------------------- 3. las dos filas nuevas
print()
print("=" * 78)
print("LAS DOS FILAS NUEVAS")
print("=" * 78)
listas = True
for arq, carpeta in NUEVOS.items():
    f1s, assds, hd95s = metricas(carpeta)
    if len(f1s) < 3:
        print(f"  {arq:16s} todavia {len(f1s)} de 3 semillas")
        listas = False
        continue
    fila = "\\quad %-16s & %s & %s & %s \\\\" % (
        arq, celda_f1(f1s), celda_px(assds), celda_px(hd95s))
    print("  " + fila)
    print("      crudos F1:", [round(f, 4) for f in f1s])

print()
print("LISTAS PARA EDITAR EL .tex" if listas else
      "AUN NO. Faltan semillas por terminar.")
