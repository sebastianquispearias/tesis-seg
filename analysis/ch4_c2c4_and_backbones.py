"""ch4_c2c4_and_backbones.py — cifras para dos afirmaciones del Capitulo 4 que
ahora mismo estan solo en prosa.

A) El escalar C2-C4 (seccion 4.4). El texto dice tres cosas:
     "its median error dropped by only 15%"
     "its largest errors did not change"
     "because they come from the rule that locates the two corner landmarks
      rather than from the segmentation"
   Aqui se comprueban las tres por separado.

B) SSL sobre todos los backbones (seccion 4.3, y peticion explicita de Paulo).
   OJO: en la Tabla `tab:arch` U-Net++ aparece con MT all-lateral y los otros
   tres con MT random r=15. No es la misma condicion. Aqui se leen los cuatro
   bajo MT random r=15, que si es comparable.

Las predicciones NO se limpian de componentes pequenas: el CSV de c2c4 se calculo
sobre las predicciones crudas y el F1 del capitulo tambien. (La limpieza <200 px
es especifica del argumento del ROI, no de aqui.)

READ-ONLY. No entrena ni reescribe ningun run.

Uso:  python tesis_seg/analysis/ch4_c2c4_and_backbones.py
"""
import glob
import json
import os

import numpy as np
import cv2
from PIL import Image
from scipy.stats import spearmanr

ROOT = "G:/My Drive/UNM_vertebras_seg_v3"
RUNS = f"{ROOT}/runs_final_v1"
MASKS = f"{ROOT}/test/masks"

SUP = "supervised"
MT = "mean_teacher_all_lateral"

BACKBONES = [
    ("U-Net++",     "supervised",              "mean_teacher_std_matched_r15"),
    ("U-Net",       "supervised_unet",         "mean_teacher_unet_std_matched_r15"),
    ("FPN",         "supervised_fpn",          "mean_teacher_fpn_std_matched_r15"),
    ("DeepLabV3+",  "supervised_deeplabv3plus", "mean_teacher_deeplabv3plus_std_matched_r15"),
]


# ---------------------------------------------------------------- utilidades
def pad_sq(a, fill=0):
    im = Image.fromarray(a); w, h = im.size; s = max(w, h)
    c = Image.new(im.mode, (s, s), fill); c.paste(im, ((s - w) // 2, (s - h) // 2))
    return np.array(c)


def gt320(path):
    m = np.array(Image.open(path).convert("L"))
    m = np.where(m > 127, 255, 0).astype(np.uint8)
    return cv2.resize(pad_sq(m, 0), (320, 320), interpolation=cv2.INTER_NEAREST) > 127


def read_csv(path):
    """Lee el c2c4_comparison.csv sin pandas. Devuelve dict de columnas."""
    with open(path, encoding="utf-8") as f:
        head = f.readline().strip().split(",")
        cols = {k: [] for k in head}
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) != len(head):
                continue
            for k, v in zip(head, parts):
                cols[k].append(v)
    out = {}
    for k, v in cols.items():
        try:
            out[k] = np.array([float(x) for x in v])
        except ValueError:
            out[k] = np.array(v)
    return out


def seed_dirs(exp):
    return sorted(glob.glob(f"{RUNS}/{exp}/seed_*"))


def f1_from_report(seed_dir):
    rep = sorted(glob.glob(f"{seed_dir}/*_run_report.json"))
    if not rep:
        return None
    with open(rep[0], encoding="utf-8") as f:
        d = json.load(f)
    return d.get("test_metrics", {}).get("sample_mean_f1")


def dice_per_frame(seed_dir, gts):
    """Dice por frame desde test_probs/, umbral 0.5, sin limpiar."""
    out = {}
    for p in sorted(glob.glob(f"{seed_dir}/test_probs/*.npy")):
        stem = os.path.basename(p)[:-4]
        if stem not in gts:
            continue
        pr = np.load(p) >= 0.5
        g = gts[stem]
        denom = pr.sum() + g.sum()
        out[stem] = float(2 * (pr & g).sum() / denom) if denom else 0.0
    return out


def pct(a, q):
    return float(np.percentile(a, q))


# ---------------------------------------------------------------- A) escalar
def part_a():
    print("=" * 78)
    print("A) EL ESCALAR C2-C4  (seccion 4.4)")
    print("=" * 78)

    gts = {os.path.basename(p)[:-4]: gt320(p) for p in sorted(glob.glob(f"{MASKS}/*.png"))}
    print(f"frames de test con mascara de referencia: {len(gts)}\n")

    data = {}
    for tag, exp in (("Supervised", SUP), ("Mean Teacher", MT)):
        rows = []
        for sd in seed_dirs(exp):
            csv = f"{sd}/c2c4_comparison.csv"
            if not os.path.exists(csv):
                print(f"  [SKIP] sin c2c4_comparison.csv: {sd}")
                continue
            c = read_csv(csv)
            dice = dice_per_frame(sd, gts)
            c["dice"] = np.array([dice.get(s, np.nan) for s in c["stem"]])
            rows.append((os.path.basename(sd), c))
        data[tag] = rows
        print(f"{tag:13s} {len(rows)} semillas, {len(rows[0][1]['stem'])} frames cada una")
    print()

    # --- (a) y (b): distribucion del error del escalar
    print("-" * 78)
    print("(a) mediana   y   (b) cola:  abs_err_px del escalar C2-C4")
    print("-" * 78)
    print(f"{'config':14s} {'mediana':>9s} {'media':>9s} {'p75':>8s} {'p90':>8s} {'p95':>8s} {'max':>8s}")
    summ = {}
    for tag in ("Supervised", "Mean Teacher"):
        per = {k: [] for k in ("med", "mean", "p75", "p90", "p95", "max")}
        for _, c in data[tag]:
            e = c["abs_err_px"]
            per["med"].append(np.median(e)); per["mean"].append(e.mean())
            per["p75"].append(pct(e, 75)); per["p90"].append(pct(e, 90))
            per["p95"].append(pct(e, 95)); per["max"].append(e.max())
        summ[tag] = {k: np.array(v) for k, v in per.items()}
        s = summ[tag]
        print(f"{tag:14s} {s['med'].mean():9.2f} {s['mean'].mean():9.2f} "
              f"{s['p75'].mean():8.2f} {s['p90'].mean():8.2f} "
              f"{s['p95'].mean():8.2f} {s['max'].mean():8.2f}")
    a, b = summ["Supervised"], summ["Mean Teacher"]
    print()
    for k, label in (("med", "mediana"), ("p90", "p90"), ("p95", "p95"), ("max", "maximo")):
        d = 100 * (b[k].mean() - a[k].mean()) / a[k].mean()
        print(f"  cambio en {label:8s}: {a[k].mean():6.2f} -> {b[k].mean():6.2f} px "
              f"({d:+5.1f}%)")

    # --- (c) la causa
    print()
    print("-" * 78)
    print("(c) causa: el error del escalar, ¿va con el LANDMARK o con la SEGMENTACION?")
    print("-" * 78)
    print(f"{'config':14s} {'rho(err, landmark)':>20s} {'rho(err, Dice)':>17s}")
    for tag in ("Supervised", "Mean Teacher"):
        rl, rd = [], []
        for _, c in data[tag]:
            ok = ~np.isnan(c["dice"])
            rl.append(spearmanr(c["abs_err_px"][ok], c["err_landmark_max_px"][ok]).statistic)
            rd.append(spearmanr(c["abs_err_px"][ok], c["dice"][ok]).statistic)
        rl, rd = np.array(rl), np.array(rd)
        print(f"{tag:14s} {rl.mean():12.3f} +/-{rl.std(ddof=1):5.3f} "
              f"{rd.mean():10.3f} +/-{rd.std(ddof=1):5.3f}")

    print()
    print("decil superior del error del escalar, ¿como estan sus dos posibles causas?")
    print(f"{'config':14s} {'grupo':>10s} {'err escalar':>12s} {'err landmark':>13s} {'Dice':>8s}")
    for tag in ("Supervised", "Mean Teacher"):
        for grp in ("resto", "decil alto"):
            E, L, D = [], [], []
            for _, c in data[tag]:
                ok = ~np.isnan(c["dice"])
                e, l, d = c["abs_err_px"][ok], c["err_landmark_max_px"][ok], c["dice"][ok]
                thr = pct(e, 90)
                m = e >= thr if grp == "decil alto" else e < thr
                E.append(e[m].mean()); L.append(l[m].mean()); D.append(d[m].mean())
            print(f"{tag:14s} {grp:>10s} {np.mean(E):12.2f} {np.mean(L):13.2f} {np.mean(D):8.3f}")

    # --- swaps, por si acaso
    print()
    for tag in ("Supervised", "Mean Teacher"):
        sw = [c["assignment_swapped"].sum() for _, c in data[tag]]
        print(f"{tag:14s} landmarks intercambiados: {np.mean(sw):.1f} de "
              f"{len(data[tag][0][1]['stem'])} frames por semilla")
    return data


# ---------------------------------------------------------------- B) backbones
def part_b():
    print()
    print("=" * 78)
    print("B) SSL SOBRE TODOS LOS BACKBONES  (misma condicion: MT random r=15)")
    print("=" * 78)
    print(f"{'backbone':13s} {'supervisado':>20s} {'MT random r=15':>22s} {'delta':>9s}")
    rows = []
    for name, sup_exp, mt_exp in BACKBONES:
        vals = {}
        for key, exp in (("sup", sup_exp), ("mt", mt_exp)):
            f1 = [f1_from_report(sd) for sd in seed_dirs(exp)]
            f1 = np.array([x for x in f1 if x is not None])
            vals[key] = f1
        s, m = vals["sup"], vals["mt"]
        print(f"{name:13s} {s.mean():8.3f} +/-{s.std(ddof=1):.3f} (n={len(s)}) "
              f"{m.mean():10.3f} +/-{m.std(ddof=1):.3f} (n={len(m)}) "
              f"{m.mean()-s.mean():+9.3f}")
        rows.append((name, s, m))
    print()
    print("control: lo que dice la Tabla tab:arch del .tex")
    print("  U-Net .824 -> .851 | FPN .799 -> .811 | DeepLabV3+ .776 -> .800")
    print("  U-Net++ .801, pero alli comparado con MT all-lateral (.860), NO con r=15")
    return rows


if __name__ == "__main__":
    part_a()
    part_b()
