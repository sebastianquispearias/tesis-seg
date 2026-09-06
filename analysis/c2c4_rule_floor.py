"""c2c4_rule_floor.py — ¿cuanto error del escalar C2-C4 pone la REGLA, y cuanto la
segmentacion?

Control de Sebastian: aplicar la misma regla de post-procesado a la mascara de
REFERENCIA. Lo que quede de error ahi no lo puede causar la segmentacion, porque la
mascara es la correcta. Es el suelo del metodo.

  error_suelo    = |d(regla sobre la mascara de referencia) - d(linea manual)|
  error_modelo   = |d(regla sobre la prediccion)            - d(linea manual)|   (abs_err_px del CSV)

Si error_suelo se parece a error_modelo, la mejora de segmentacion no puede llegar
al escalar, y esa es la explicacion.

READ-ONLY.  Uso:  python tesis_seg/analysis/c2c4_rule_floor.py
"""
import glob, os, sys
import numpy as np
import cv2
from PIL import Image

ROOT = "G:/My Drive/UNM_vertebras_seg_v3"
sys.path.insert(0, f"{ROOT}/tesis_seg")
from src.ruler_eval import c2_c4_from_mask

MASKS = f"{ROOT}/test/masks"
RUNS = f"{ROOT}/runs_final_v1"


def pad_sq(a, fill=0):
    im = Image.fromarray(a); w, h = im.size; s = max(w, h)
    c = Image.new(im.mode, (s, s), fill); c.paste(im, ((s - w) // 2, (s - h) // 2))
    return np.array(c)


def gt320(path):
    m = np.array(Image.open(path).convert("L"))
    m = np.where(m > 127, 255, 0).astype(np.uint8)
    return cv2.resize(pad_sq(m, 0), (320, 320), interpolation=cv2.INTER_NEAREST)


def manual_dist():
    """d_gt_px es la distancia de la linea anotada a mano; es identica en todos los runs."""
    csv = f"{RUNS}/mean_teacher_all_lateral/seed_0/c2c4_comparison.csv"
    L = open(csv, encoding="utf-8").read().splitlines()
    h = L[0].split(","); i_s, i_d = h.index("stem"), h.index("d_gt_px")
    out = {}
    for l in L[1:]:
        p = l.split(",")
        if len(p) == len(h):
            out[p[i_s]] = float(p[i_d])
    return out


def model_errors(exp):
    e = []
    for sd in sorted(glob.glob(f"{RUNS}/{exp}/seed_*")):
        csv = f"{sd}/c2c4_comparison.csv"
        if not os.path.exists(csv):
            continue
        L = open(csv, encoding="utf-8").read().splitlines()
        h = L[0].split(","); i = h.index("abs_err_px")
        for l in L[1:]:
            p = l.split(",")
            if len(p) == len(h):
                e.append(float(p[i]))
    return np.array(e)


def main():
    man = manual_dist()
    floor, failed = [], []
    for p in sorted(glob.glob(f"{MASKS}/*.png")):
        stem = os.path.basename(p)[:-4]
        if stem not in man:
            continue
        try:
            _, _, d = c2_c4_from_mask(gt320(p) > 0)
        except Exception:
            failed.append(stem); continue
        if d is None or not np.isfinite(d):
            failed.append(stem); continue
        floor.append(abs(d - man[stem]))
    floor = np.array(floor)

    print(f"frames evaluados: {len(floor)}   (la regla fallo en {len(failed)}: {failed})")
    print()
    print("ERROR DEL ESCALAR C2-C4, en px (todos los pares frame-semilla agrupados)")
    print(f"{'entrada de la regla':34s} {'n':>5s} {'mediana':>9s} {'media':>8s} {'p90':>8s} {'max':>8s}")
    rows = [("mascara de REFERENCIA (suelo)", floor)]
    for lab, exp in (("prediccion supervisada", "supervised"),
                     ("prediccion Mean Teacher", "mean_teacher_all_lateral")):
        rows.append((lab, model_errors(exp)))
    for lab, a in rows:
        print(f"{lab:34s} {len(a):5d} {np.median(a):9.2f} {a.mean():8.2f} "
              f"{np.percentile(a,90):8.2f} {a.max():8.2f}")
    print()
    f_med = np.median(floor)
    for lab, a in rows[1:]:
        m = np.median(a)
        print(f"  del error de {lab:26s} {100*f_med/m:5.1f}% ya lo pone la regla sola")


if __name__ == "__main__":
    main()
