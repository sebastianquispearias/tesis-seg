"""Check every printed row of tab:arch against the run reports.

The values in the manuscript are recomputed from the JSON reports, grouping the
seeds by the folder that holds them rather than by the experiment_name field,
because two folders carry a report whose name field does not match the folder.
Each row prints the recomputed value next to the printed one and a verdict.
"""
import glob
import json
import math
import os

BASE = r"G:\My Drive\UNM_vertebras_seg_v3"

# fila impresa: (etiqueta, carpeta, F1, ASSD, HD95) tal como aparecen en el .tex
FILAS = [
    ("SUP DeepLabV3+",     "runs_final_v1/supervised_deeplabv3plus",              ".776+-.041", "2.08+-0.23", "7.07+-0.66"),
    ("SUP BiFPN-U-Net(T)", "runs_final_v1/supervised_bifpn_unet",                 ".759+-.012", "4.86+-1.59", "12.39+-2.78"),
    ("SUP U-Net++",        "runs_final_v1/supervised",                            ".801+-.014", "2.41+-0.22", "9.86+-2.06"),
    ("SUP FPN",            "runs_final_v1/supervised_fpn",                        ".799+-.025", "2.08+-0.20", "6.55+-0.87"),
    ("SUP U-Net",          "runs_final_v1/supervised_unet",                       ".824+-.032", "2.09+-0.31", "8.39+-2.22"),
    ("SUP TransUNet",      "runs_final_v1/supervised_transunet",                  ".851+-.011", "1.92+-0.20", "8.16+-1.49"),
    ("SSL PL temporal r3", "runs_final_v1/semi_r3",                               ".852+-.019", "1.76+-0.28", "6.72+-1.02"),
    ("SSL MT all lateral", "runs_final_v1/mean_teacher_all_lateral",              ".860+-.006", "1.62+-0.13", "5.68+-0.37"),
    ("MT DeepLabV3+",      "runs_final_v1/mean_teacher_deeplabv3plus_std_matched_r15", ".800+-.014", "2.24+-0.25", "8.17+-1.06"),
    ("MT BiFPN-U-Net(T)",  "runs_ssl_backbones/mean_teacher_bifpn_unet_std_matched_r15", ".757+-.029", "5.26+-2.60", "12.59+-4.24"),
    ("MT U-Net++",         "runs_final_v1/mean_teacher_std_matched_r15",          ".830+-.038", "1.92+-0.22", "7.80+-0.93"),
    ("MT FPN",             "runs_final_v1/mean_teacher_fpn_std_matched_r15",      ".811+-.028", "2.24+-0.13", "7.74+-0.86"),
    ("MT U-Net",           "runs_final_v1/mean_teacher_unet_std_matched_r15",     ".851+-.005", "1.64+-0.05", "6.06+-0.74"),
    ("MT TransUNet",       "runs_ssl_backbones/mean_teacher_transunet_std_matched_r15", ".831+-.005", "2.09+-0.15", "8.72+-0.88"),
]


def sd(xs):
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def stats(carpeta):
    f1, assd, hd95 = [], [], []
    for d in sorted(glob.glob(os.path.join(BASE, carpeta, "seed_*"))):
        ps = glob.glob(os.path.join(d, "*run_report.json"))
        if not ps:
            continue
        with open(ps[0], "r", encoding="utf-8") as fh:
            r = json.load(fh)
        m = r.get("test_metrics") or {}
        b = r.get("test_boundary_metrics") or {}
        if m.get("sample_mean_f1") is not None:
            f1.append(m["sample_mean_f1"])
        if b.get("assd_mean_px") is not None:
            assd.append(b["assd_mean_px"])
        if b.get("hd95_mean_px") is not None:
            hd95.append(b["hd95_mean_px"])
    return f1, assd, hd95


print("{:20s} {:>3s}  {:24s} {:24s} {:26s} {}".format(
    "fila impresa", "n", "F1  informes / .tex", "ASSD  informes / .tex",
    "HD95  informes / .tex", "veredicto"))
print("-" * 126)
todo_ok = True
for etiqueta, carpeta, tf1, tassd, thd in FILAS:
    f1, assd, hd95 = stats(carpeta)
    if not f1:
        print("{:20s}  SIN INFORMES en {}".format(etiqueta, carpeta))
        todo_ok = False
        continue
    cf1 = "{:.3f}+-{:.3f}".format(sum(f1) / len(f1), sd(f1)).replace("0.", ".")
    cassd = "{:.2f}+-{:.2f}".format(sum(assd) / len(assd), sd(assd))
    chd = "{:.2f}+-{:.2f}".format(sum(hd95) / len(hd95), sd(hd95))
    ok = (cf1 == tf1) and (cassd == tassd) and (chd == thd)
    todo_ok = todo_ok and ok
    print("{:20s} {:>3d}  {:11s} {:11s}  {:11s} {:11s}  {:12s} {:12s}  {}".format(
        etiqueta, len(f1), cf1, tf1, cassd, tassd, chd, thd,
        "OK" if ok else "<<< NO COINCIDE"))
print("-" * 126)
print("TODAS LAS FILAS COINCIDEN" if todo_ok else "HAY FILAS QUE NO COINCIDEN")
