"""Re-derive the recorded conclusions and check the assumptions behind them.

A number that matches is not enough. Each conclusion rests on two arms being
comparable, and the mistakes these checks exist to catch were mistakes of
comparability rather than of arithmetic: a run trained with a defect averaged
together with the run that fixed it, and one seed counted twice because it
appears under two folder names. This script therefore checks three things for
every conclusion.

First, that each side of a comparison holds the seeds it claims and holds each
of them once. Second, that the two sides differ only in the fields the
conclusion allows them to differ in, read from the stored configurations rather
than assumed. Third, that the recomputed difference still matches the value that
was written down.

Every figure comes from the run reports. Nothing is read from the manuscript or
from the session notes. Run it with no arguments; it exits non-zero if any
conclusion fails.
"""
import glob
import json
import math
import os
import sys

BASE = r"G:\My Drive\UNM_vertebras_seg_v3"

# Campos que siempre difieren y nunca cambian lo que se entrena.
IGNORAR_SIEMPRE = {"exp_dir", "seed"}

TOLERANCIA = 0.0002


def leer_runs(carpeta):
    """Every seed folder of an arm, as (seed, F1, config)."""
    salida = []
    for d in sorted(glob.glob(os.path.join(BASE, carpeta, "seed_*"))):
        ps = glob.glob(os.path.join(d, "*run_report.json"))
        if not ps:
            continue
        with open(ps[0], "r", encoding="utf-8") as fh:
            r = json.load(fh)
        f1 = (r.get("test_metrics") or {}).get("sample_mean_f1")
        if f1 is None:
            continue
        salida.append((r.get("run_identity", {}).get("seed"), f1, r.get("config", {})))
    return salida


def media(xs):
    return sum(xs) / len(xs)


def desv(xs):
    if len(xs) < 2:
        return 0.0
    m = media(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def por_semilla(carpetas):
    """Collapse several folders into one value per seed.

    A seed present in more than one folder is averaged, so that a configuration
    repeated across sessions counts once per seed and not once per run. This is
    the check that the TransUNet count failed on 6 September.
    """
    bruto = {}
    for c in carpetas:
        for s, f1, _cfg in leer_runs(c):
            bruto.setdefault(s, []).append(f1)
    return {s: media(v) for s, v in bruto.items()}, bruto


def diferencias_cfg(cfg_a, cfg_b):
    campos = set(cfg_a) | set(cfg_b)
    return {k for k in campos
            if cfg_a.get(k, "<ausente>") != cfg_b.get(k, "<ausente>")}


CONCLUSIONES = [
    {
        "nombre": "1. Alinear las dos vistas no mueve el F1",
        "a": ["runs_ssl_backbones/mean_teacher_unetpp_std_matched_r15"],
        "b": ["runs_ssl_backbones/alineada_unetpp_std_matched_r15"],
        "etiqueta_a": "vistas independientes",
        "etiqueta_b": "vistas alineadas",
        "permitidos": {"aug_alineada"},
        "semillas": {0, 1, 2},
        "delta": -0.0144,
        "dice": "Alinear no sube el F1. Las ganancias de la tesis no eran un suelo.",
    },
    {
        "nombre": "2a. TransUNet: apagar la perdida devuelve el valor supervisado",
        "a": ["runs_ssl_backbones/mean_teacher_transunet_std_matched_r15"],
        "b": ["runs_ssl_backbones/lambda0_transunet_std_matched_r15"],
        "etiqueta_a": "mean teacher completo",
        "etiqueta_b": "lambda_u = 0",
        # aug_alineada esta ausente en el brazo viejo y a False en el nuevo: el
        # commit f89064d solo anade codigo y la rama apagada es la de siempre.
        "permitidos": {"lambda_u", "aug_alineada"},
        "semillas": {0, 1, 2},
        "delta": +0.0202,
        "dice": "La culpa es de la perdida de consistencia, no del regimen.",
    },
    {
        "nombre": "3. El SSL aporta sobre la augmentation moderada de nnU-Net",
        "a": ["runs_nnunet_ablation/X_sup_moderate"],
        "b": ["runs_nnunet_ablation/X_mt_moderate"],
        "etiqueta_a": "supervisado",
        "etiqueta_b": "mean teacher",
        # Los cinco describen la configuracion SSL. Con use_semi False ninguno
        # llega a ejecutarse: semi_start_epoch se lee en train.py l.295, dentro
        # del bloque "if use_semi:" que abre en la l.215, y ssl_method en la
        # l.525, dentro de "if cfg['use_semi']". Ademas is_cps es False en los
        # dos brazos, asi que las ramas CPS tampoco se separan.
        "permitidos": {"use_semi", "lambda_u", "unlabeled_dir", "unlabeled_subdir",
                       "pool_dir", "unlabeling_dir", "ssl_method",
                       "semi_start_epoch"},
        "semillas": {0, 1, 2},
        "delta": +0.0323,
        "dice": "Es la comparacion mas limpia de la tanda: 3 de 3 semillas.",
    },
    {
        "nombre": "4. La augmentation moderada no cambia el resultado supervisado",
        "a": ["runs_nnunet_ablation/A_baseline_fixnoise",
              "runs_nnunet_ablation/X_sup_base"],
        "b": ["runs_nnunet_ablation/C1_aug_moderate",
              "runs_nnunet_ablation/X_sup_moderate"],
        "etiqueta_a": "augmentation nuestra",
        "etiqueta_b": "nnunet_moderate",
        "permitidos": {"aug_profile"},
        "semillas": {0, 1, 2},
        "delta": -0.0033,
        "dice": ("A_baseline queda FUERA: entreno con el ruido gaussiano roto. "
                 "Las dos sesiones se contradicen en el signo y ambas son diminutas."),
    },
]

fallos = []
print("=" * 84)
print("VERIFICACION DE LAS CONCLUSIONES DEL 6/9")
print("todo se recalcula de los run_report.json; nada se lee del .tex ni de las notas")
print("=" * 84)

for c in CONCLUSIONES:
    print()
    print(c["nombre"])
    print("-" * 84)
    ok = True

    val_a, bruto_a = por_semilla(c["a"])
    val_b, bruto_b = por_semilla(c["b"])

    # 1. las semillas son las esperadas y ninguna falta
    for lado, val in (("A", val_a), ("B", val_b)):
        if set(val) != c["semillas"]:
            print("   FALLO  lado {}: semillas {} y se esperaban {}".format(
                lado, sorted(val), sorted(c["semillas"])))
            ok = False

    # 2. cuantos runs hay por semilla, para que una repetida se vea
    reps_a = {s: len(v) for s, v in bruto_a.items()}
    reps_b = {s: len(v) for s, v in bruto_b.items()}
    print("   runs por semilla   A {}   B {}".format(
        dict(sorted(reps_a.items())), dict(sorted(reps_b.items()))))
    if len(set(reps_a.values())) > 1 or len(set(reps_b.values())) > 1:
        print("   AVISO  el diseno NO esta equilibrado: hay semillas con mas runs "
              "que otras, y la media queda sesgada hacia ellas")
        ok = False

    # 3. las dos configuraciones difieren solo en lo permitido
    cfg_a = leer_runs(c["a"][0])[0][2]
    cfg_b = leer_runs(c["b"][0])[0][2]
    distintos = diferencias_cfg(cfg_a, cfg_b) - IGNORAR_SIEMPRE
    sobra = distintos - c["permitidos"]
    print("   config: difieren en {}".format(sorted(distintos) or "nada"))
    if sobra:
        print("   FALLO  campos NO previstos por la conclusion: {}".format(sorted(sobra)))
        ok = False

    # 4. el numero
    ma, mb = media(list(val_a.values())), media(list(val_b.values()))
    d = mb - ma
    print("   {:24s} {:.4f} +/- {:.4f}".format(
        c["etiqueta_a"], ma, desv(list(val_a.values()))))
    print("   {:24s} {:.4f} +/- {:.4f}".format(
        c["etiqueta_b"], mb, desv(list(val_b.values()))))
    signos = sum(1 for s in sorted(c["semillas"]) if (val_b[s] - val_a[s]) > 0)
    print("   delta {:+.4f}   escrito {:+.4f}   B gana en {} de {} semillas".format(
        d, c["delta"], signos, len(c["semillas"])))
    if abs(d - c["delta"]) > TOLERANCIA:
        print("   FALLO  el delta se movio mas de {}".format(TOLERANCIA))
        ok = False

    print("   {}".format(c["dice"]))
    print("   >>> {}".format("OK" if ok else "REVISAR"))
    if not ok:
        fallos.append(c["nombre"])

# El supervisado de TransUNet de septiembre no es un brazo: son dos carpetas que
# comparten la semilla 0, asi que se comprueba aparte y explicitamente.
print()
print("2b. TransUNet supervisado de septiembre, sin ningun termino de consistencia")
print("-" * 84)
val, bruto = por_semilla([
    "runs_ssl_backbones/supervised_transunet_std_matched_r15",
    "runs_ssl_backbones_SIN_POOL/mean_teacher_transunet_std_matched_r15",
])
for s in sorted(bruto):
    print("   semilla {}   {}   ->  {:.4f}".format(
        s, " ".join("{:.4f}".format(v) for v in bruto[s]), val[s]))
m_sept = media(list(val.values()))
print("   supervisado septiembre  {:.4f} +/- {:.4f}   {} semillas, {} runs".format(
    m_sept, desv(list(val.values())), len(val), sum(len(v) for v in bruto.values())))
print("   promediando los RUNS en vez de las semillas saldria {:.4f}, que esta MAL:"
      .format(media([v for vs in bruto.values() for v in vs])))
print("   cuenta dos veces la semilla 0, que es la mas alta.")

abril = [f1 for _s, f1, _c in leer_runs("runs_final_v1/supervised_transunet")]
lam0 = [f1 for _s, f1, _c in leer_runs(
    "runs_ssl_backbones/lambda0_transunet_std_matched_r15")]
tres = [media(abril), m_sept, media(lam0)]
print()
print("   supervisado ABRIL       {:.4f}   {} semillas".format(media(abril), len(abril)))
print("   supervisado SEPTIEMBRE  {:.4f}   {} semillas".format(m_sept, len(val)))
print("   lambda0    SEPTIEMBRE   {:.4f}   {} semillas".format(media(lam0), len(lam0)))
print("   recorrido entre las tres: {:.4f}".format(max(tres) - min(tres)))
if max(tres) - min(tres) > 0.005:
    print("   FALLO  las tres estimaciones ya no coinciden")
    fallos.append("2b. TransUNet supervisado de septiembre")
else:
    print("   >>> OK")

# La tabla de descomposicion se comprueba aparte porque cada eje lleva su propio
# control: el arreglo del ruido gaussiano parte los brazos en dos generaciones y
# un eje que cruce las dos no mide lo que dice medir.
GENERACION = {
    "runs_nnunet_ablation/A_baseline": "ruido roto",
    "runs_nnunet_ablation/B_zscore": "ruido roto",
    "runs_nnunet_ablation/D_bs1": "ruido roto",
    "runs_nnunet_ablation/A_baseline_fixnoise": "ruido sano",
    "runs_nnunet_ablation/X_sup_base": "ruido sano",
    "runs_nnunet_ablation/X_sup_moderate": "ruido sano",
    "runs_nnunet_ablation/C1_aug_moderate": "ruido sano",
    "runs_nnunet_ablation/C2_aug_full": "ruido sano",
    "runs_resolucion/E_320_6img": "ruido sano",
    "runs_resolucion/F_1024_6img": "ruido sano",
}

EJES = [
    ("resolucion", ["runs_resolucion/E_320_6img"],
     ["runs_resolucion/F_1024_6img"], +0.0007),
    ("augmentation moderada",
     ["runs_nnunet_ablation/A_baseline_fixnoise", "runs_nnunet_ablation/X_sup_base"],
     ["runs_nnunet_ablation/C1_aug_moderate", "runs_nnunet_ablation/X_sup_moderate"],
     -0.0033),
    ("augmentation completa",
     ["runs_nnunet_ablation/A_baseline_fixnoise", "runs_nnunet_ablation/X_sup_base"],
     ["runs_nnunet_ablation/C2_aug_full"], -0.0165),
    ("normalizacion z-score", ["runs_nnunet_ablation/A_baseline"],
     ["runs_nnunet_ablation/B_zscore"], -0.0121),
    ("batch size 1", ["runs_nnunet_ablation/A_baseline"],
     ["runs_nnunet_ablation/D_bs1"], -0.0458),
]

print()
print("TABLA DE DESCOMPOSICION DE nnU-NET: cada eje contra un control de su misma")
print("generacion de ruido. Es la columna delta lo que se publica, no los absolutos.")
print("-" * 84)
for nombre, ctrl, var, esperado in EJES:
    gen_c = {GENERACION[c] for c in ctrl}
    gen_v = {GENERACION[v] for v in var}
    val_c, _ = por_semilla(ctrl)
    val_v, _ = por_semilla(var)
    d = media(list(val_v.values())) - media(list(val_c.values()))
    cruza = bool(gen_c | gen_v) and len(gen_c | gen_v) > 1
    print("   {:24s} control {:.4f} [{}]  variante {:.4f} [{}]  delta {:+.4f}".format(
        nombre, media(list(val_c.values())), "/".join(sorted(gen_c)),
        media(list(val_v.values())), "/".join(sorted(gen_v)), d))
    if cruza:
        print("      FALLO  el eje CRUZA las dos generaciones de ruido: no mide lo que dice")
        fallos.append("eje " + nombre)
    elif abs(d - esperado) > TOLERANCIA:
        print("      FALLO  el delta se movio: escrito {:+.4f}".format(esperado))
        fallos.append("eje " + nombre)

print()
print("=" * 84)
if fallos:
    print("HAY CONCLUSIONES QUE REVISAR:")
    for f in fallos:
        print("   " + f)
    sys.exit(1)
print("TODAS LAS CONCLUSIONES SE SOSTIENEN")
