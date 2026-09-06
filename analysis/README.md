# Análisis para la tesis (inter-rater, figuras de anotación, diagnóstico de Dice)

Scripts reproducibles usados para el apéndice de anotación y el análisis diagnóstico.
Todos son **READ-ONLY** sobre los datos originales (no reentrenan, no modifican máscaras/frames).
Rutas absolutas al workspace `G:/My Drive/UNM_vertebras_seg_v3`.

## inter_rater/
- **compute_inter_rater_inca.py** — Empareja las 2 anotaciones de cada frame INCA
  (batch-{N}/{anotador}/{vf}/Mask.tif, anotador por `video_frame_metadata.csv`),
  calcula Dice inter-anotador. Resultado: **Dice = 0.874 ± 0.070, mediana 0.899**
  sobre **898** frames doblemente anotados (67 con 1 solo anotador + 3 por resolución = 70 excluidos de 968).
  Salidas: `resultados/inter_rater_inca/inter_rater_per_frame.csv`, `_report.md`, `excluded_single_annotator.csv`.
- **build_annotation_figure_candidates.py** — Métricas por frame (Dice, área, componentes,
  XOR, centroide) + hojas de contacto de candidatos + histograma. Verifica que la media reproduce 0.874.
  Salidas: `resultados/inter_rater_inca/figure_candidates/`.

## figuras/  (figuras finales del apéndice de anotación)
- **make_figura_criterios.py** — Figura de 2 paneles (frame gris | frame + máscara de referencia),
  ejemplo `v138_f67`. → `figuras_finales/figura_criterios.{pdf,png}`.
- **make_figura_interanotador.py** — 2 paneles con las dos anotaciones (cian A / naranja B):
  `v26_f135` (acuerdo, Dice 0.900) y `v20_f152` (vértebra extra, Dice 0.738).
  → `figuras_finales/figura_interanotador.{pdf,png}`.
- **make_figura_categorias.py** — 5 categorías de problemas de anotación (A faltante, B extra,
  C sobreextendida, D error de borde, E visibilidad limitada), una fila por categoría.
  Frames elegidos: A=v39_f4, B=v207_f159, C=v235_f214, D=v128_f235, E=v12_f90.
  → `figuras_finales/figura_categorias.{pdf,png,svg}`.
- **anotar_categorias.py** — Herramienta interactiva (`annotate`) para colocar flechas/círculos
  a mano sobre los paneles, y `render` para exportar. Alternativa recomendada: editar el SVG en Inkscape.

## diagnostico/  (por qué el Dice global baja vs per-vértebra) — MÉTODO B
Configuraciones principales verificadas: UNM/PL-r3, UNM/MT-all-lateral, INCA/MT-r15 (todos los seeds).

- **diag_pervertebra_B.py** — PRINCIPAL. Diagnóstico component-level reproducible, todos los seeds.
  Evaluación idéntica al pipeline de la tesis: GT → `pad_to_square` → resize 320×320 (NEAREST).
  Valida que el Dice global reproduce los valores reportados (UNM/PL-r3 .852, UNM/MT-all-lateral .860,
  INCA/MT-r15 .907). Método B (asignación guiada por referencia): cada píxel predicho se asigna a la
  vértebra GT más cercana dentro de un territorio dilatado, de modo que una predicción con dos
  vértebras pegadas se **parte** correctamente (no las confunde con una faltante). Dice por vértebra
  en variante **missing-as-zero** (0 si una vértebra no se detecta). Reporta %extra, %missing, FP/FN,
  correlaciones, media ± std **entre seeds**.
  Salidas: `resultados/diagnostico_dice/pervertebra_B_per_image.csv`, `pervertebra_B_summary.csv`.
- **diag_visual_B.py** — Verificación visual, una imagen por página (seed_0): GT etiquetado por color,
  predicción **partida por vértebra** (cada píxel coloreado según la vértebra asignada), y las cuentas.
  Salidas: `resultados/diagnostico_dice/verificacion_visual_B/{config}.pdf`.
- **diag_copy.py / diag_copy_seeds.py** — Helpers que copian GT/preds/frames a disco local
  (`C:/Users/User/temp_inter_rater/diag`) para lectura rápida. Regenerables; la copia local es temporal.

### Por qué el método B (justificación, para la banca)
Las máscaras son binarias (no distinguen C2/C3/C4). La identidad se infiere separando la máscara en
componentes conexas y ordenándolas de arriba a abajo. Un método ingenuo (componentes conexas de la
PREDICCIÓN) cuenta dos vértebras predichas pegadas como una y marca la vecina como falsa "no detectada".
El método B evita eso usando la referencia como plantilla. Efecto medido: en UNM la tasa de "faltante"
baja de ~12% a ~2% (eran fusiones mal contadas); en INCA casi no cambia (allí el problema son vértebras
extra, no fusión). El Dice global **no** se ve afectado por la elección de método.

### Limitación a declarar en la tesis
La identidad C2/C3/C4 se infiere del orden supero-inferior de los componentes conexos de la máscara de
**referencia binaria**. En una minoría de frames donde vértebras adyacentes de la referencia aparecen en
contacto, esta asignación puede ser imprecisa (no existe una referencia etiquetada por vértebra). Por eso
los valores por vértebra se interpretan como **indicativos**, no exactos. El Dice global no depende de esto.

### Hallazgo clave
- **INCA → componentes EXTRA (falsos positivos fuera de C2–C4).** ~24% de imágenes con vértebra extra;
  FP≈FN bajos; correlación global↔per-vértebra baja (~0.19–0.57). El per-vértebra es optimista porque
  ignora las vértebras extra que sí bajan el global.
- **UNM → falsos negativos / sub-segmentación, peor en C4.** FN ≈ 3× FP; C4 la peor vértebra
  (.846–.849); correlación global↔per-vértebra alta (0.63–0.93).

## entropia/  (análisis de incertidumbre desde los mapas de probabilidad)
Usa `test_probs/*.npy` (float32 320x320) ya guardados; NO reentrena. Entropía binaria por píxel
H = -p·log(p) - (1-p)·log(1-p). Configs: UNM Supervisado vs MT all-lateral; INCA patient10
Supervisado vs MT r10. Todas las semillas; métrica principal = entropía en la BANDA del borde (±5px).

- **ent_copy_all.py** — copia los `.npy` necesarios a disco local (`temp_inter_rater/ent`) para lectura
  rápida (Drive es lento). Regenerable; la copia local es temporal.
- **ent_rework.py** — PRINCIPAL. Todas las semillas (media±std ENTRE seeds). Calcula: entropía por
  banda (±3/±5/±10) y global; Spearman entropía↔Dice por imagen y por CLÚSTER (vídeo en UNM, paciente
  en INCA); confusor de área (área↔Dice, área↔entropía); active learning (20% más incierto vs azar×1000);
  calibración (ECE por píxel). Salidas: `resultados/diagnostico_dice/entropy_rework/*.csv`.
- **build_ent_report2.py** — arma el PDF consolidado `entropy_rework/REPORTE_entropia_v2.pdf`.
- **entropy_extra.py** — scatter + mapas de incertidumbre (seed_0). Salidas: `entropy_figs/`.
- **entropy_analysis.py** — versión previa (banda vs resto, INCA full). Histórica.

### Hallazgos (honestos, algunos corrigen la versión ingenua)
- La incertidumbre está CONCENTRADA en el borde (banda ~70-115x el resto). Robusto en ±3/±5/±10.
- El SSL NO reduce la incertidumbre; en INCA-p10 la AUMENTA. Contradice la hipótesis.
- Correlación entropía↔Dice: usar la entropía de BANDA (no la global, sesgada por el fondo). Con la
  banda: fuerte y limpia en INCA (ρ≈-0.5), DÉBIL en UNM (ρ=-0.25) y parcialmente explicada por el ÁREA
  (confusor real en UNM, ausente en INCA). Agregar por clúster no la destruye (UNM solo 7 clústeres → poca potencia).
- Active learning por entropía: útil en INCA (20% incierto por debajo del azar), marginal en UNM.
- Calibración: los modelos NO están sobreconfiados aquí (ECE bajo), PERO el ECE global está dominado por
  el fondo → declararlo como limitación (la calibración en el borde no se aísla).

## Nota sobre `per_vertebra_dice_v3_overlap.csv`
Archivo histórico (generador no localizado, seed/pipeline desconocido). NO usar; usar `diag_pervertebra_B.py`.
