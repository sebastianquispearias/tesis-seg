"""Seed-paired comparison between two experiment folders.

Both arms share the seed grid, so the difference is taken seed by seed. Prints
the per-seed deltas, their mean, the sample SD and a paired t statistic with a
normal-approximation two-sided p value, plus the sign agreement across seeds.
"""
import glob
import json
import math
import os
import sys

BASE = r"G:\My Drive\UNM_vertebras_seg_v3"


def por_semilla(carpeta, clave="sample_mean_f1", bloque="test_metrics"):
    out = {}
    for d in sorted(glob.glob(os.path.join(BASE, carpeta, "seed_*"))):
        ps = glob.glob(os.path.join(d, "*run_report.json"))
        if not ps:
            continue
        with open(ps[0], "r", encoding="utf-8") as fh:
            r = json.load(fh)
        v = (r.get(bloque) or {}).get(clave)
        if v is not None:
            out[int(os.path.basename(d).split("_")[1])] = v
    return out


def erf_p(t):
    """Two-sided p under a normal approximation."""
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0))))


def comparar(etiqueta, a, b, clave="sample_mean_f1", bloque="test_metrics"):
    ma, mb = por_semilla(a, clave, bloque), por_semilla(b, clave, bloque)
    comunes = sorted(set(ma) & set(mb))
    print("-" * 74)
    print(etiqueta)
    if not comunes:
        print("  sin semillas en comun: A={} B={}".format(sorted(ma), sorted(mb)))
        return
    difs = []
    for s in comunes:
        d = mb[s] - ma[s]
        difs.append(d)
        print("  semilla {}:  {:.4f} -> {:.4f}   delta {:+.4f}".format(
            s, ma[s], mb[s], d))
    n = len(difs)
    m = sum(difs) / n
    if n > 1:
        sd = math.sqrt(sum((d - m) ** 2 for d in difs) / (n - 1))
        t = m / (sd / math.sqrt(n)) if sd > 0 else float("inf")
        print("  delta medio {:+.4f}   SD {:.4f}   t={:.2f}   p~{:.3f}   n={}".format(
            m, sd, t, erf_p(t), n))
    else:
        print("  delta medio {:+.4f}   n={}".format(m, n))
    positivos = sum(1 for d in difs if d > 0)
    print("  el signo coincide en {} de {} semillas".format(
        max(positivos, n - positivos), n))


print("=" * 74)
print("nb19: el arreglo del ruido sobre el brazo A de la ablacion nnU-Net")
print("=" * 74)
comparar("A_baseline  ->  A_baseline_fixnoise   (F1)",
         "runs_nnunet_ablation/A_baseline",
         "runs_nnunet_ablation/A_baseline_fixnoise")
comparar("A_baseline  ->  A_baseline_fixnoise   (HD95 px)",
         "runs_nnunet_ablation/A_baseline",
         "runs_nnunet_ablation/A_baseline_fixnoise",
         "hd95_mean_px", "test_boundary_metrics")

print()
print("=" * 74)
print("nb20: Mean Teacher contra su propio control supervisado, misma semilla")
print("=" * 74)
comparar("TransUNet   supervisado -> MT r=15   (F1)",
         "runs_ssl_backbones/supervised_transunet_std_matched_r15",
         "runs_ssl_backbones/mean_teacher_transunet_std_matched_r15")
comparar("BiFPN-U-Net supervisado -> MT r=15   (F1)",
         "runs_ssl_backbones/supervised_bifpn_unet_std_matched_r15",
         "runs_ssl_backbones/mean_teacher_bifpn_unet_std_matched_r15")

print()
print("=" * 74)
print("nb22: cada politica de pool contra el pool aleatorio, misma semilla")
print("=" * 74)
for etiqueta, carpeta in (
        ("aleatorio -> temporal      ", "runs_pool_incertidumbre/MT_temporal_r10"),
        ("aleatorio -> mas inciertos ", "runs_pool_incertidumbre/MT_mas_inciertos"),
        ("aleatorio -> menos inciertos", "runs_pool_incertidumbre/MT_menos_inciertos")):
    comparar(etiqueta + "   (F1)",
             "runs_pool_incertidumbre/MT_aleatorio_r10", carpeta)
