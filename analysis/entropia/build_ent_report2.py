"""Consolidated PDF for the REWORKED entropy analysis. READ-ONLY. Reads local CSVs."""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

R = r"G:/My Drive/UNM_vertebras_seg_v3/resultados/diagnostico_dice/entropy_rework"
FIG = r"G:/My Drive/UNM_vertebras_seg_v3/resultados/diagnostico_dice/entropy_figs"
OUT = os.path.join(R, "REPORTE_entropia_v2.pdf")

def rd(fn): return list(csv.DictReader(open(os.path.join(R, fn), encoding="utf-8")))

def text_page(pdf, title, lines, fs=10.5):
    fig = plt.figure(figsize=(8.3, 11.7))
    fig.text(0.06, 0.955, title, fontsize=15, weight="bold", va="top")
    fig.text(0.06, 0.905, "\n".join(lines), fontsize=fs, va="top", family="monospace")
    pdf.savefig(fig); plt.close(fig)

def image_page(pdf, png, title):
    fig = plt.figure(figsize=(8.3, 11.7))
    fig.text(0.5, 0.97, title, fontsize=12, weight="bold", ha="center", va="top")
    ax = fig.add_axes([0.03, 0.04, 0.94, 0.89]); ax.imshow(mpimg.imread(png)); ax.set_axis_off()
    pdf.savefig(fig); plt.close(fig)

ent = rd("entropy_bands.csv"); sp = rd("spearman.csv")
conf = rd("confounder.csv"); al = rd("active_learning.csv"); cal = rd("calibration.csv")
pim = rd("per_image_allseeds.csv")

pdf = PdfPages(OUT)

# Page 1
text_page(pdf, "Incertidumbre (entropía) — reporte v2 (corregido)", [
    "Todo se calcula desde los mapas de probabilidad ya guardados (test_probs/",
    "*.npy), SIN reentrenar. Entropía binaria por píxel:",
    "    H = -p·log(p) - (1-p)·log(1-p)     (log natural, máx = ln2 = 0.693)",
    "",
    "CORRECCIONES aplicadas respecto a la versión anterior:",
    " 1) TODAS las semillas (no solo seed_0). Media ± std ENTRE semillas.",
    " 2) Métrica PRINCIPAL = entropía en la BANDA del borde (±5 px). La global",
    "    se deja como referencia: casi todos los píxeles son fondo, así que la",
    "    media global depende del tamaño del fondo, no de la incertidumbre real.",
    " 3) Correlación también por CLÚSTER (vídeo en UNM, paciente en INCA),",
    "    porque frames del mismo clúster no son independientes.",
    " 4) Se controla el CONFUSOR de área (¿las vértebras pequeñas son más",
    "    difíciles y explican la relación?).",
    " 5) Simulación de ACTIVE LEARNING (20% más incierto vs azar).",
    " 6) Bandas ±3, ±5, ±10 px.",
    " 7) CALIBRACIÓN (¿la red está sobreconfiada?).",
    "",
    "Configuraciones: UNM Supervisado (5 seeds) vs MT all-lateral (3);",
    "INCA patient10 (bajo etiquetado) Supervisado (3) vs MT r10 (3).",
    "Validación: el Dice desde estas probabilidades (umbral 0.5) reproduce lo",
    "reportado en la tesis.",
])

# Page 2: entropy bands
lines = ["Entropía media (±std entre semillas). PRINCIPAL = banda del borde.", "",
         f"{'Config':22s} {'banda±3':>13s} {'banda±5':>13s} {'banda±10':>13s} {'global':>10s}", "-"*74]
for r in ent:
    lines.append(f"{r['dataset']+' '+r['config']:22s} "
                 f"{r['Hband3']+'±'+r['Hband3_std']:>13s} {r['Hband5']+'±'+r['Hband5_std']:>13s} "
                 f"{r['Hband10']+'±'+r['Hband10_std']:>13s} {r['Hglobal']:>10s}")
lines += ["",
          "CÓMO LEERLO:",
          " - Las 3 anchuras de banda dan la MISMA conclusión (robusto).",
          " - La entropía del borde es ~70-115x la del resto de la imagen",
          "   => la incertidumbre está CONCENTRADA en el contorno, no repartida.",
          " - Supervisado vs MT: en UNM casi igual (.049 vs .048); en INCA-p10 el",
          "   MT es MAYOR (.054 -> .067).",
          "",
          " CONCLUSIÓN: el SSL NO reduce la incertidumbre; en INCA-p10 la AUMENTA.",
          " (Resultado honesto, contrario a la hipótesis 'SSL reduce incertidumbre'.)"]
text_page(pdf, "1-2) Entropía del borde (métrica principal)", lines)

# Page 3: Spearman
lines = ["Spearman entre entropía-banda(±5) y Dice. Por imagen y por CLÚSTER.", "",
         f"{'Config':22s} {'n_clús':>7s} {'ρ imagen':>14s} {'ρ clúster':>14s}", "-"*62]
for r in sp:
    lines.append(f"{r['dataset']+' '+r['config']:22s} {r['n_clusters']:>7s} "
                 f"{r['rho_img']+'±'+r['rho_img_std']:>14s} {r['rho_cluster']+'±'+r['rho_cluster_std']:>14s}")
lines += ["",
          "CORRECCIÓN IMPORTANTE:",
          " - Antes reporté ρ≈-0.59 para UNM Supervisado, pero usaba la entropía",
          "   GLOBAL (contaminada por área de fondo). Con la entropía de BANDA",
          "   (correcta), UNM Supervisado baja a ρ=-0.25 (DÉBIL). La correlación",
          "   fuerte era en parte un artefacto de la métrica global.",
          " - MT y INCA mantienen ρ≈-0.5.",
          "",
          "¿Se cae al agregar por clúster? NO: se mantiene o refuerza.",
          " PERO: UNM tiene solo 7 clústeres (7 vídeos de test) -> poca potencia,",
          " std alta. INCA con 23 pacientes es más fiable.",
          "",
          "LECTURA: 'imágenes más inciertas = peor segmentadas' se sostiene con",
          "fuerza SOLO en INCA y en UNM-MT; en UNM-Supervisado es débil."]
text_page(pdf, "3) Correlación entropía ↔ Dice (por imagen y clúster)", lines)

# Page 4: scatter band-entropy vs dice (aggregate across seeds)
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
pos = {("UNM", "Supervised"): (0, 0), ("UNM", "MT all-lateral"): (0, 1),
       ("INCA-p10", "Supervised"): (1, 0), ("INCA-p10", "MT r10"): (1, 1)}
agg = {}
for r in pim:
    k = (r["dataset"], r["config"], r["stem"])
    agg.setdefault(k, []).append((float(r["H_b5"]), float(r["dice"])))
by_cfg = {}
for (ds, cfg, stem), v in agg.items():
    e = np.nanmean([x[0] for x in v]); d = np.mean([x[1] for x in v])
    by_cfg.setdefault((ds, cfg), []).append((e, d))
sp_map = {(r["dataset"], r["config"]): r for r in sp}
for (ds, cfg), pts in by_cfg.items():
    if (ds, cfg) not in pos: continue
    r, c = pos[(ds, cfg)]; e = [p[0] for p in pts]; d = [p[1] for p in pts]
    axes[r, c].scatter(e, d, s=14, alpha=0.6)
    rr = sp_map[(ds, cfg)]
    axes[r, c].set_title(f"{ds}/{cfg}   ρ_img={rr['rho_img']}±{rr['rho_img_std']}", fontsize=10)
    axes[r, c].set_xlabel("entropía banda±5 (media entre seeds)"); axes[r, c].set_ylabel("Dice")
fig.suptitle("Dispersión: entropía del borde vs Dice (1 punto por imagen)", fontsize=12)
fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

# Page 5: confounder
lines = ["¿La relación entropía-Dice se explica solo porque las vértebras",
         "pequeñas/grandes son más difíciles? Correlación con el ÁREA del GT:", "",
         f"{'Config':22s} {'área↔Dice':>16s} {'área↔entropía':>16s}", "-"*58]
for r in conf:
    lines.append(f"{r['dataset']+' '+r['config']:22s} {r['rho_area_dice']:>16s} {r['rho_area_entropy']:>16s}")
lines += ["",
          "LECTURA:",
          " - UNM: el área SÍ es un confusor. Correlaciona con Dice (-0.47) y con",
          "   la entropía (+0.36). Parte de la relación entropía-Dice en UNM se",
          "   explica por el tamaño de la máscara, no por la incertidumbre en sí.",
          " - INCA-p10: NO hay confusor de área (todo ≈0). Ahí la relación",
          "   entropía-calidad es genuina, no un efecto de tamaño.",
          "",
          " CONCLUSIÓN: la señal de incertidumbre es limpia en INCA; en UNM está",
          " parcialmente confundida con el área."]
text_page(pdf, "4) Confusor de área", lines)

# Page 6: active learning
lines = ["Simulación de active learning con el modelo SUPERVISADO: se ordenan los",
         "frames de test por entropía y se compara el Dice medio del 20% más",
         "incierto contra el 20% elegido al azar (1000 repeticiones).", "",
         f"{'Config':22s} {'Dice 20% incierto':>20s} {'Dice 20% azar (95%CI)':>28s}", "-"*72]
for r in al:
    lines.append(f"{r['dataset']+' '+r['config']:22s} {r['dice_top20_uncertain']:>20s} "
                 f"{r['dice_random20']+' '+r['random20_95CI']:>28s}")
lines += ["",
          "LECTURA:",
          " - INCA-p10: el 20% más incierto (Dice 0.779) queda POR DEBAJO del",
          "   límite inferior del azar (0.823) => seleccionar por entropía",
          "   encuentra frames genuinamente peores. Active learning ÚTIL.",
          " - UNM: el 20% más incierto (0.769) queda DENTRO del intervalo del azar",
          "   (0.703-0.883) => apenas separa. Active learning MARGINAL en UNM.",
          "",
          " (Coherente con la correlación: fuerte en INCA, débil en UNM.)"]
text_page(pdf, "5) Active learning simulado", lines)

# Page 7: calibration
lines = ["¿La red está sobreconfiada? Calibración a nivel de píxel (ECE).",
         "(conf-acc) > 0 = sobreconfiado; < 0 = subconfiado.", "",
         f"{'Config':22s} {'ECE':>8s} {'(conf-acc)':>12s} {'veredicto':>14s}", "-"*60]
for r in cal:
    v = "sobreconfiado" if float(r["over_minus_under"]) > 0 else "subconfiado"
    lines.append(f"{r['dataset']+' '+r['config']:22s} {r['ECE']:>8s} {r['over_minus_under']:>12s} {v:>14s}")
lines += ["",
          "LECTURA (contradice lo esperado):",
          " - Los modelos NO están sobreconfiados aquí: ECE bajo (.004-.006) y",
          "   signo levemente subconfiado / calibrado.",
          "",
          "CAVEAT HONESTO (declararlo como limitación):",
          " - El ECE por píxel está DOMINADO por el fondo (millones de píxeles",
          "   fáciles p≈0, gt=0, perfectamente calibrados) => baja el ECE de forma",
          "   artificial. La calibración EN EL BORDE (donde vive la incertidumbre)",
          "   podría ser distinta y esta métrica global no la aísla.",
          " - Conclusión prudente: no afirmar 'bien calibrado' sin más; reportar",
          "   ECE bajo global + la limitación del dominio del fondo."]
text_page(pdf, "6) Calibración", lines)

# Page 8: uncertainty maps (existing, show boundary concentration)
mp = os.path.join(FIG, "uncertainty_maps_UNM.png")
if os.path.isfile(mp):
    image_page(pdf, mp, "Mapas de incertidumbre — UNM (la incertidumbre está en el borde)")

# Page 9: conclusions
text_page(pdf, "Conclusiones honestas (qué cambió)", [
    "1) La incertidumbre vive en el BORDE. Robusto en ±3/±5/±10 px. (se mantiene)",
    "",
    "2) El SSL NO reduce la incertidumbre. En INCA-p10 incluso la AUMENTA.",
    "   (se mantiene y se refuerza)",
    "",
    "3) La correlación entropía↔calidad es MÁS DÉBIL de lo que parecía:",
    "   - UNM Supervisado: ρ=-0.25 (no -0.59). El valor alto anterior venía de la",
    "     entropía global, contaminada por el área.",
    "   - Además, en UNM el ÁREA es un confusor real (parte de la relación se",
    "     explica por el tamaño). En INCA la señal es limpia (ρ≈-0.5, sin confusor).",
    "",
    "4) Active learning por entropía: ÚTIL en INCA (por debajo del azar),",
    "   MARGINAL en UNM (dentro del azar).",
    "",
    "5) Los modelos NO están sobreconfiados aquí (ECE bajo), pero el ECE global",
    "   está sesgado por el fondo -> declararlo como limitación.",
    "",
    "6) Agregar por clúster no destruye la correlación, pero UNM tiene solo 7",
    "   clústeres (poca potencia).",
    "",
    "MENSAJE REPORTABLE:",
    "  La incertidumbre es un fenómeno de BORDE; el SSL mejora el Dice pero no",
    "  reduce la incertidumbre; la entropía predice la calidad de forma fiable en",
    "  INCA (no confundida con área) y débilmente en UNM. La utilidad para",
    "  active learning es dataset-dependiente.",
    "",
    "Fuentes: resultados/diagnostico_dice/entropy_rework/*.csv",
    "Scripts: ent_rework.py (+ ent_copy_all.py, entropy_analysis.py, entropy_extra.py)",
])
pdf.close()
print("saved", OUT)
