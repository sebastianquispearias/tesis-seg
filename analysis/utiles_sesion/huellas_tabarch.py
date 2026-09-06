"""Print the library fingerprint of every seed behind every row of tab:arch.

The Gaussian noise augmentation was silently disabled by albumentations 2.x, so
the version recorded in each report decides whether that run was trained with
the intended noise or without it. The rows of the manuscript are only
comparable among themselves if their seeds share that version.
"""
import glob
import json
import os

BASE = r"G:\My Drive\UNM_vertebras_seg_v3"

FILAS = [
    ("SUP DeepLabV3+",     "runs_final_v1/supervised_deeplabv3plus"),
    ("SUP BiFPN-U-Net(T)", "runs_final_v1/supervised_bifpn_unet"),
    ("SUP U-Net++",        "runs_final_v1/supervised"),
    ("SUP FPN",            "runs_final_v1/supervised_fpn"),
    ("SUP U-Net",          "runs_final_v1/supervised_unet"),
    ("SUP TransUNet",      "runs_final_v1/supervised_transunet"),
    ("SSL PL temporal r3", "runs_final_v1/semi_r3"),
    ("SSL MT all lateral", "runs_final_v1/mean_teacher_all_lateral"),
    ("MT DeepLabV3+",      "runs_final_v1/mean_teacher_deeplabv3plus_std_matched_r15"),
    ("MT BiFPN-U-Net(T)",  "runs_ssl_backbones/mean_teacher_bifpn_unet_std_matched_r15"),
    ("MT U-Net++",         "runs_final_v1/mean_teacher_std_matched_r15"),
    ("MT FPN",             "runs_final_v1/mean_teacher_fpn_std_matched_r15"),
    ("MT U-Net",           "runs_final_v1/mean_teacher_unet_std_matched_r15"),
    ("MT TransUNet",       "runs_ssl_backbones/mean_teacher_transunet_std_matched_r15"),
]

CLAVES = ["albumentations", "torch", "segmentation_models_pytorch", "python"]

for etiqueta, carpeta in FILAS:
    print("-" * 96)
    versiones = set()
    for d in sorted(glob.glob(os.path.join(BASE, carpeta, "seed_*"))):
        ps = glob.glob(os.path.join(d, "*run_report.json"))
        if not ps:
            continue
        with open(ps[0], "r", encoding="utf-8") as fh:
            r = json.load(fh)
        h = r.get("reproducibility_fingerprint") or {}
        f1 = (r.get("test_metrics") or {}).get("sample_mean_f1")
        vals = tuple(str(h.get(k, "?")) for k in CLAVES)
        versiones.add(vals)
        print("  {:20s} {:8s} F1 {:.4f}   {}".format(
            etiqueta, os.path.basename(d), f1,
            "  ".join("{}={}".format(k, v) for k, v in zip(CLAVES, vals))))
    if len(versiones) > 1:
        print("  {:20s} >>> FILA MIXTA: {} combinaciones de versiones".format("", len(versiones)))
    else:
        print("  {:20s} fila homogenea".format(""))
