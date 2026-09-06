"""Full audit of every run report in the project.

Walks all run roots and prints, for each experiment arm, the number of seeds,
the mean and sample standard deviation of the test F1, and the fields that
decide whether two arms are comparable: the dataset tag, the label directory,
the number of test images, the semi-supervised flag and the unlabeled pool.
The point is to catch arms that look comparable by name but were evaluated on
different data or under a different regime.
"""
import glob
import json
import math
import os

BASE = r"G:\My Drive\UNM_vertebras_seg_v3"


def sd(xs):
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


raices = sorted(d for d in os.listdir(BASE)
                if d.startswith("runs_") and os.path.isdir(os.path.join(BASE, d)))

filas = []
for raiz in raices:
    for dp, _dn, fns in os.walk(os.path.join(BASE, raiz)):
        for fn in fns:
            if not fn.endswith("run_report.json"):
                continue
            with open(os.path.join(dp, fn), "r", encoding="utf-8") as fh:
                r = json.load(fh)
            ident = r.get("run_identity", {})
            cfg = r.get("config", {})
            met = r.get("test_metrics") or {}
            filas.append({
                "raiz": raiz,
                "brazo": ident.get("experiment_name", "?"),
                "seed": ident.get("seed", -1),
                "f1": met.get("sample_mean_f1"),
                "n_img": met.get("n_images"),
                "dataset": cfg.get("dataset"),
                "rotulos": os.path.basename(str(cfg.get("rotulos_dir", ""))),
                "use_semi": cfg.get("use_semi"),
                "lambda_u": cfg.get("lambda_u"),
                "pool": os.path.basename(os.path.dirname(str(cfg.get("unlabeled_dir", "")))),
                "img_size": cfg.get("img_size"),
            })

brazos = {}
for f in filas:
    brazos.setdefault((f["raiz"], f["brazo"]), []).append(f)

print("{:34s} {:38s} {:>3s} {:>8s} {:>8s}  {:6s} {:>6s} {:>7s} {:>5s} {:>7s} {}".format(
    "raiz", "brazo", "n", "F1", "SD", "datos", "n_img", "semi", "lam", "size", "pool"))
print("-" * 155)
for (raiz, brazo) in sorted(brazos):
    fs = brazos[(raiz, brazo)]
    f1s = [f["f1"] for f in fs if f["f1"] is not None]
    if not f1s:
        continue
    ds = sorted({str(f["dataset"]) for f in fs})
    ni = sorted({str(f["n_img"]) for f in fs})
    sm = sorted({str(f["use_semi"]) for f in fs})
    lu = sorted({str(f["lambda_u"]) for f in fs})
    sz = sorted({str(f["img_size"]) for f in fs})
    po = sorted({str(f["pool"]) for f in fs})
    print("{:34s} {:38s} {:>3d} {:8.4f} {:8.4f}  {:6s} {:>6s} {:>7s} {:>5s} {:>7s} {}".format(
        raiz[:34], brazo[:38], len(f1s), sum(f1s) / len(f1s), sd(f1s),
        "/".join(ds)[:6], "/".join(ni)[:6], "/".join(sm)[:7],
        "/".join(lu)[:5], "/".join(sz)[:7], "/".join(po)[:30]))
