"""Donde esta el error residual: distancia al contorno de referencia y asimetria FP/FN.

READ-ONLY. Solo lee artefactos ya guardados (test_preds/*.png y test/masks/*.png).
No entrena, no escribe nada dentro de runs_*.

Espacio de comparacion: el mismo del entrenamiento -- la mascara de referencia se
rellena a cuadrado con ceros y se reescala a 320x320 con vecino mas proximo. Ese
emparejamiento reproduce exactamente el sample_mean_f1 de los run_report.json.

Uso:
    python tesis_seg/analysis/error_localization.py
"""

import os
import numpy as np
import cv2

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GT_DIR = os.path.join(RAIZ, "test", "masks")

CONFIGS = [
    ("Supervised", "runs_final_v1/supervised", [0, 1, 2, 3, 4]),
    ("Mean Teacher all-lateral", "runs_final_v1/mean_teacher_all_lateral", [0, 1, 2]),
]

UMBRALES_PX = (1, 2, 3, 5, 10)


def _carga_binaria(ruta):
    m = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(ruta)
    return (m > 127).astype(np.uint8)


def _pad_cuadrado(m):
    h, w = m.shape
    lado = max(h, w)
    out = np.zeros((lado, lado), np.uint8)
    y = (lado - h) // 2
    x = (lado - w) // 2
    out[y:y + h, x:x + w] = m
    return out


def referencia_320(nombre):
    m = _carga_binaria(os.path.join(GT_DIR, nombre))
    return cv2.resize(_pad_cuadrado(m), (320, 320), interpolation=cv2.INTER_NEAREST)


def analiza_semilla(pred_dir, nombres):
    """Devuelve un dict con los agregados de una semilla (63 frames agrupados)."""
    area_gt = area_pred = fp_tot = fn_tot = 0
    dist_error = []
    dist_fn = []
    dist_fp = []
    f1_por_frame = []
    error_por_video = {}

    for n in nombres:
        g = referencia_320(n)
        p = _carga_binaria(os.path.join(pred_dir, n))

        inter = int((p & g).sum())
        suma = int(p.sum() + g.sum())
        f1_por_frame.append(2 * inter / suma if suma else 1.0)

        fp = (p == 1) & (g == 0)
        fn = (p == 0) & (g == 1)
        err = g != p

        area_gt += int(g.sum())
        area_pred += int(p.sum())
        fp_tot += int(fp.sum())
        fn_tot += int(fn.sum())

        video = n.split("_")[0]
        error_por_video[video] = error_por_video.get(video, 0) + int(err.sum())

        contorno = cv2.Canny(g * 255, 50, 150) > 0
        if contorno.sum() == 0 or err.sum() == 0:
            continue
        d = cv2.distanceTransform((~contorno).astype(np.uint8), cv2.DIST_L2, 5)
        dist_error.append(d[err])
        if fn.any():
            dist_fn.append(d[fn])
        if fp.any():
            dist_fp.append(d[fp])

    dist_error = np.concatenate(dist_error)
    res = {
        "f1": float(np.mean(f1_por_frame)),
        "area_gt": area_gt,
        "area_pred": area_pred,
        "pct_area": 100.0 * area_pred / area_gt,
        "fp": fp_tot,
        "fn": fn_tot,
        "ratio_fn_fp": fn_tot / fp_tot if fp_tot else float("nan"),
        "mediana_dist": float(np.median(dist_error)),
        "mediana_dist_fn": float(np.median(np.concatenate(dist_fn))),
        "mediana_dist_fp": float(np.median(np.concatenate(dist_fp))),
        "error_por_video": error_por_video,
    }
    for k in UMBRALES_PX:
        res[f"pct_<={k}px"] = 100.0 * float(np.mean(dist_error <= k))
    return res


def media_sd(valores):
    """Media y desviacion MUESTRAL (ddof=1), que es la convencion del capitulo 4.

    Comprobado: con ddof=1 el supervisado sale 0.8015 +/- 0.0140, identico a
    RESULTS_FINAL.md. Con ddof=0 saldria +/- 0.0125.
    """
    a = np.asarray(valores, dtype=float)
    return float(a.mean()), float(a.std(ddof=1))


def main():
    nombres = sorted(f for f in os.listdir(GT_DIR) if f.endswith(".png"))
    print(f"frames de test: {len(nombres)}\n")

    resumen = {}
    for etiqueta, subdir, semillas in CONFIGS:
        print("=" * 74)
        print(f"{etiqueta}   ({len(semillas)} semillas)")
        print("=" * 74)
        porseed = []
        for s in semillas:
            pred_dir = os.path.join(RAIZ, subdir, f"seed_{s}", "test_preds")
            r = analiza_semilla(pred_dir, nombres)
            porseed.append(r)
            print(f"  seed_{s}: F1={r['f1']:.4f}  area={r['pct_area']:.1f}%  "
                  f"FN={r['fn']:,}  FP={r['fp']:,}  "
                  f"<=5px={r['pct_<=5px']:.1f}%  mediana={r['mediana_dist']:.2f}px")

        print("\n  --- media +/- SD entre semillas ---")
        campos = [
            ("F1 (sample mean)", "f1", "{:.4f}"),
            ("area predicha / referencia (%)", "pct_area", "{:.1f}"),
            ("falsos negativos (px)", "fn", "{:,.0f}"),
            ("falsos positivos (px)", "fp", "{:,.0f}"),
            ("razon FN/FP", "ratio_fn_fp", "{:.2f}"),
            ("mediana dist. al contorno (px)", "mediana_dist", "{:.2f}"),
            ("  idem, solo FN (px)", "mediana_dist_fn", "{:.2f}"),
            ("  idem, solo FP (px)", "mediana_dist_fp", "{:.2f}"),
        ]
        for k in UMBRALES_PX:
            campos.append((f"error a <= {k} px del contorno (%)", f"pct_<={k}px", "{:.1f}"))

        agregados = {}
        for nombre_campo, clave, fmt in campos:
            m, sd = media_sd([r[clave] for r in porseed])
            agregados[clave] = (m, sd)
            print(f"  {nombre_campo:34s} {fmt.format(m)} +/- {fmt.format(sd)}")

        vids = sorted({v for r in porseed for v in r["error_por_video"]})
        print("\n  reparto del error por video (media entre semillas, % del error total):")
        reparto = []
        for v in vids:
            pcts = []
            for r in porseed:
                tot = sum(r["error_por_video"].values())
                pcts.append(100.0 * r["error_por_video"].get(v, 0) / tot)
            reparto.append((v, float(np.mean(pcts)), float(np.std(pcts))))
        for v, m, sd in sorted(reparto, key=lambda x: -x[1]):
            print(f"     {v}: {m:5.1f} % +/- {sd:.1f}")
        print()
        resumen[etiqueta] = agregados

    print("=" * 74)
    print("CIFRAS PARA EL PARRAFO")
    print("=" * 74)
    sup = resumen["Supervised"]
    mt = resumen["Mean Teacher all-lateral"]
    print(f"  supervisado: {sup['pct_<=5px'][0]:.0f} % del error a <=5 px, "
          f"mediana {sup['mediana_dist'][0]:.1f} px")
    print(f"  supervisado asigna {sup['pct_area'][0]:.1f} % del area de referencia; "
          f"FN/FP = {sup['ratio_fn_fp'][0]:.1f} a 1")
    print(f"  MT asigna {mt['pct_area'][0]:.1f} %; "
          f"FN caen {100 * (1 - mt['fn'][0] / sup['fn'][0]):.0f} %, "
          f"FP varian {100 * (mt['fp'][0] / sup['fp'][0] - 1):+.0f} %")


if __name__ == "__main__":
    main()
