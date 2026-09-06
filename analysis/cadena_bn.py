"""Isolate where the semi-supervised gain comes from, one condition at a time.

Four conditions are read from disk: the supervised baseline, Mean Teacher, the
same regime with the consistency weight set to zero, and that control again with
the normalization statistics frozen while the unlabeled frames pass through.
Comparing them says whether the gain travels through the semi-supervised loss or
through the normalization buffers.

The script also prints the software stack recorded in each report, because the
conditions were not all trained in the same session and a comparison across
stacks carries a confound that no amount of seeds removes.

All figures are read from the run reports. Read only; applies nothing.
"""

import glob
import json
import math
import os

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONDICIONES = [
    ("supervisado U-Net++", "runs_final_v1/supervised"),
    ("MT r15 (perdida SSL activa)", "runs_final_v1/mean_teacher_std_matched_r15"),
    ("lambda0 r10 (perdida SSL apagada)", "runs_control_lambda0/control_lambda0_r10"),
    ("lambda0 r10 + BN congelada",
     "runs_control_lambda0_bnfrozen/bnfrozen_lambda0_r10"),
    ("lambda0 all-lateral (perdida SSL apagada)",
     "runs_control_lambda0/control_lambda0_all_lateral"),
    ("lambda0 all-lateral + BN congelada",
     "runs_control_lambda0_bnfrozen/bnfrozen_lambda0_all_lateral"),
]

PAREJAS = [
    ("r10", "lambda0 r10 (perdida SSL apagada)", "lambda0 r10 + BN congelada"),
    ("all-lateral", "lambda0 all-lateral (perdida SSL apagada)",
     "lambda0 all-lateral + BN congelada"),
]


def leer(carpeta):
    """seed -> (F1, software stack) for every finished run of one condition."""
    out = {}
    for d in sorted(glob.glob(os.path.join(BASE, carpeta, "seed_*"))):
        ps = glob.glob(os.path.join(d, "*run_report.json"))
        if not ps:
            continue
        with open(ps[0], "r", encoding="utf-8") as fh:
            r = json.load(fh)
        f1 = (r.get("test_metrics") or {}).get("sample_mean_f1")
        fp = r.get("reproducibility_fingerprint") or {}
        stack = "{} / albumentations {}".format(
            fp.get("torch", "?"), fp.get("albumentations", "?"))
        if f1 is not None:
            out[int(os.path.basename(d).split("_")[1])] = (f1, stack)
    return out


def ms(vals):
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    return m, math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def main():
    datos = {}
    print("MEDIAS Y STACK DE CADA CONDICION")
    for etiqueta, carpeta in CONDICIONES:
        d = leer(carpeta)
        datos[etiqueta] = d
        if not d:
            print("  {:44s} sin resultados".format(etiqueta))
            continue
        m, s = ms([v[0] for v in d.values()])
        stacks = sorted({v[1] for v in d.values()})
        print("  {:44s} {:.4f} +/- {:.4f}  n={}".format(etiqueta, m, s, len(d)))
        for st in stacks:
            print("  {:44s}   stack: {}".format("", st))

    print()
    print("PAREADO POR SEMILLA: congelar la BatchNorm, todo lo demas igual")
    for pool, a, b in PAREJAS:
        ma, mb = datos.get(a, {}), datos.get(b, {})
        comunes = sorted(set(ma) & set(mb))
        if not comunes:
            continue
        print()
        print("  pool {}".format(pool))
        difs = []
        for s in comunes:
            d = mb[s][0] - ma[s][0]
            difs.append(d)
            print("    semilla {}:  {:.4f} -> {:.4f}   delta {:+.4f}".format(
                s, ma[s][0], mb[s][0], d))
        m, sd = ms(difs)
        neg = sum(1 for d in difs if d < 0)
        print("    delta medio {:+.4f}   SD {:.4f}   n={}   baja en {} de {}".format(
            m, sd, len(difs), neg, len(difs)))

    print()
    print("AVISO: si los stacks impresos arriba no coinciden, la cadena mezcla")
    print("versiones de software y el residuo no se puede atribuir a nada.")


if __name__ == "__main__":
    main()
