"""Entropia de banda vs NUMERO DE ETIQUETAS (el otro eje de la pregunta de Paulo).

READ-ONLY. Formulas COPIADAS TAL CUAL de ent_pool.py / ent_rework.py: misma
entropia binaria en log natural, misma banda de +/-5 px (dilatacion L1 menos
erosion L1), mismo promediado (pixeles de banda -> frames -> semillas).
Lo unico que cambia es que el eje x pasa de ser el pool NO etiquetado a ser el
numero de frames ETIQUETADOS.

Validacion: la fila "Supervised 218" tiene que dar 0.049, que es el valor
publicado en tesis_final_v54.tex l.984 (SD publicada 0.012).

Uso:
    python tesis_seg/analysis/entropia/ent_labelfrac.py
"""

import os
import glob
import numpy as np
from PIL import Image
from scipy import ndimage
import cv2

RUNS = "G:/My Drive/UNM_vertebras_seg_v3/runs_final_v1"
GT_CANDIDATOS = [
    "C:/Users/User/temp_inter_rater/diag/unm/gt",
    "G:/My Drive/UNM_vertebras_seg_v3/test/masks",
]
EPS, BAND = 1e-7, 5

# (etiqueta, carpeta, frames etiquetados de entrenamiento)
# los conteos salen de label_fractions/frac_*/stems.txt y del train completo
CONFIGS = [
    ("Supervised", [("25%", "supervised_frac25", 65),
                    ("50%", "supervised_frac50", 110),
                    ("75%", "supervised_frac75", 173),
                    ("100%", "supervised", 218)]),
    ("Pseudo-Label r=10", [("25%", "semi_r10_frac25", 65),
                           ("50%", "semi_r10_frac50", 110),
                           ("75%", "semi_r10_frac75", 173),
                           ("100%", "semi_r10", 218)]),
    ("Mean Teacher r=10", [("25%", "mean_teacher_r10_frac25", 65),
                           ("50%", "mean_teacher_r10_frac50", 110),
                           ("75%", "mean_teacher_r10_frac75", 173),
                           ("100%", "mean_teacher_r10", 218)]),
]


def pad_sq(a, fill=0):
    im = Image.fromarray(a)
    w, h = im.size
    s = max(w, h)
    c = Image.new(im.mode, (s, s), fill)
    c.paste(im, ((s - w) // 2, (s - h) // 2))
    return np.array(c)


def gt320(p):
    m = np.array(Image.open(p).convert("L"))
    m = np.where(m > 127, 255, 0).astype(np.uint8)
    return cv2.resize(pad_sq(m, 0), (320, 320), interpolation=cv2.INTER_NEAREST) > 127


def H_map(p):
    """Entropia binaria, log natural. Maximo ln 2 = 0.693."""
    p = np.clip(p.astype(np.float32), EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def band_mask(gt, px):
    """Banda alrededor del contorno: dilatacion menos erosion (L1, 4-conectividad)."""
    return ndimage.binary_dilation(gt, iterations=px) & ~ndimage.binary_erosion(gt, iterations=px)


def spearman(x, y):
    """Rho de Spearman sin depender de scipy.stats."""
    def rangos(v):
        orden = np.argsort(np.argsort(np.asarray(v, dtype=float)))
        return orden.astype(float)
    rx, ry = rangos(x), rangos(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den else float("nan")


def main():
    gt_dir = next((d for d in GT_CANDIDATOS if os.path.isdir(d)), None)
    if gt_dir is None:
        raise SystemExit("no encuentro las mascaras de referencia")
    gts = {os.path.basename(p)[:-4]: gt320(p) for p in glob.glob(f"{gt_dir}/*.png")}
    bands = {k: band_mask(v, BAND) for k, v in gts.items()}
    print(f"referencia: {gt_dir}")
    print(f"GT cargados: {len(gts)}   banda +/-{BAND} px\n")

    for familia, filas in CONFIGS:
        print("=" * 70)
        print(f"{familia}")
        print("=" * 70)
        print(f"{'etiquetas':>10s} {'frames':>7s} {'semillas':>9s}   H banda (media +/- sd)")
        print("-" * 70)
        xs, ys = [], []
        for etiqueta, run, n_lab in filas:
            per_seed = []
            for sd in sorted(glob.glob(f"{RUNS}/{run}/seed_*")):
                vals = []
                for f in glob.glob(f"{sd}/test_probs/*.npy"):
                    stem = os.path.basename(f)[:-4]
                    if stem not in bands:
                        continue
                    H = H_map(np.load(f))
                    b = bands[stem]
                    if b.sum():
                        vals.append(float(H[b].mean()))
                if vals:
                    per_seed.append(float(np.mean(vals)))
            a = np.array(per_seed)
            sd_txt = f"{a.std():.4f}" if len(a) > 1 else "  -   "
            print(f"{etiqueta:>10s} {n_lab:7d} {len(a):9d}        {a.mean():.4f} +/- {sd_txt}")
            xs.append(n_lab)
            ys.append(a.mean())
        rho = spearman(xs, ys)
        rango = max(ys) - min(ys)
        print(f"\n   Spearman(etiquetas, entropia) = {rho:+.2f}")
        print(f"   recorrido de la entropia      = {rango:.4f}"
              f"   ({min(ys):.4f} a {max(ys):.4f})\n")


if __name__ == "__main__":
    main()
