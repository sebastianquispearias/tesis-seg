# tesis_seg

Segmentacion semi-supervisada de vertebras cervicales en videofluoroscopia (VFSS).

Proyecto de mestrado. El objetivo es estudiar como la aprendizaje semi-supervisada (SSL) reduce la necesidad de anotaciones manuales para segmentar vertebras C2-C4, y como la construccion del pool de datos no rotulados afecta el resultado.

---

## Paper target

- **Venue:** CBM (Computers in Biology and Medicine) como primero alvo. CMPB como backup.
- **Titulo tentativo:** Semi-Supervised Segmentation of Cervical Vertebrae in Videofluoroscopy: the Role of Unlabeled-Pool Size and Selection Policy
- **Contribucion:** Aplicar SSL (teacher-student) a segmentacion de vertebras en VFSS por primera vez. Estudiar empiricamente como el tamano y la composicion del pool unlabeled afectan performance y estabilidad. Mostrar que SSL reduce la necesidad de rotulos (label-efficiency).

---

## Estructura del repositorio

```
tesis_seg/
    notebooks/01_train_eval_colab.ipynb   # Entry point: ejecutar en Colab
    src/
        defaults.py        # Configuracion base (DEFAULT_CONFIG)
        preprocessing.py   # Preprocesamiento: base, clahe_soft, denoise, he, ad
        augmentations.py   # Augmentations supervisadas + weak/strong para SSL
        datasets.py        # SegmentationDataset, UnlabeledFramesDataset
        models.py          # U-Net++ con segmentation_models_pytorch
        losses.py          # BCE + Dice
        metrics.py         # F1, IoU, BF1, ASSD, HD95
        train.py           # Training loop (supervisado + SSL teacher-student)
        evaluate.py        # Evaluacion, export de predicciones, run_report.json
        visualization.py   # Visualizacion de predicciones y curvas
        ruler_eval.py      # Comparacion C2-C4 manual vs automatica
        utils.py           # Seed, EMA, guardado JSON, etc.
    requirements.txt

# Scripts en workspace root (fuera de tesis_seg/):
prep_label_fractions.py        # Genera subsets anidados para label-fraction ablation
plot_perseed_f1.py             # Figura per-seed F1 dot-plot
refresh_run_reports.py         # Regenera run_report.json desde artefactos existentes
summarize_runs.py              # Consolida boundary metrics de todos los runs
paper_figures/                 # Pipeline de generacion de figuras para el paper
```

---

## Dataset

- **Fuente:** UNM TalkBank Dysphagia database
- **Split:** 23 train / 5 val / 7 test videos (sin overlap de pacientes)
- **Frames etiquetados:** 218 (train), corresponden al pico de elevacion del hioides
- **Anotador:** Un solo anotador entrenado
- **Mascaras:** Binarias, region C2-C4
- **Resolucion de trabajo:** 320x320 (grayscale, zero-padded, normalizado [0,1])

---

## Metodos SSL implementados

### Pseudo-labeling (default, ssl_method="pseudo_label")
- Teacher = EMA del student (decay=0.99)
- Teacher recibe augmentacion debil, student recibe augmentacion fuerte
- Pseudo-labels: threshold 0.5 sobre probabilidades del teacher
- Confidence mask: solo pixeles con confianza >= tau (0.95)
- Loss: BCE-with-logits del student contra pseudo-labels, filtrada por confidence mask

### Mean Teacher (ssl_method="mean_teacher")
- Mismo teacher EMA, mismas augmentaciones
- Loss: MSE entre probabilidades del student y del teacher
- Sin threshold, sin confidence mask, todos los pixeles contribuyen
- Implementado como branch condicional en train.py

Ambos metodos comparten: EMA update, lambda_u ramp, semi_start_epoch, weak/strong augmentations.

---

## Experimentos completados

### Principales (5 condiciones x 3 seeds = 15 runs)
| Condicion | Pool | Frames unlabeled |
|---|---|---|
| supervised | ninguno | 0 |
| semi_std_matched_r3 (Random-r3) | random matched | 1,257 |
| semi_r3 (Temporal-r3) | temporal r=3 | 1,257 |
| semi_std_matched_r10 (Random-r10) | random matched | 3,937 |
| semi_r10 (Temporal-r10) | temporal r=10 | 3,937 |

### Label-fraction ablation (supervised)
- supervised_frac25: 66 frames (3 seeds)
- supervised_frac50: 111 frames (3 seeds)
- supervised_frac75: 174 frames (3 seeds)
- Subsets anidados (frac25 ⊂ frac50 ⊂ frac75 ⊂ full)

### Label-fraction ablation (SSL)
- semi_r10_frac25: 66 frames labeled + 3,937 unlabeled (seed 0)
- semi_r10_frac50: 111 frames labeled + 3,937 unlabeled (seed 0)
- semi_r10_frac75: 174 frames labeled + 3,937 unlabeled (seed 0)

### Mean Teacher (pendiente)
- mean_teacher_r3, mean_teacher_std_matched_r3, mean_teacher_r10, mean_teacher_std_matched_r10
- 4 condiciones x 3 seeds = 12 runs

### Estadisticas
- Wilcoxon signed-rank test (Supervised vs Temporal-r10): p < 10^-6

---

## Resultados principales (pseudo-labeling)

| Condicion | F1 (sample) | IoU (sample) | Delta F1 |
|---|---|---|---|
| Supervised | .821 +/- .014 | .716 +/- .016 | --- |
| Random-r3 | .835 +/- .012 | .744 +/- .014 | +.014 |
| Temporal-r3 | .838 +/- .016 | .746 +/- .017 | +.018 |
| Random-r10 | .846 +/- .013 | .755 +/- .014 | +.026 |
| Temporal-r10 | .855 +/- .002 | .762 +/- .004 | +.035 |

Label-fraction: supervised satura en ~0.82 con 50%+ de labels. SSL excede ese techo.

---

## Pools unlabeled

Generados fuera de tesis_seg/ por scripts de preprocessing:
- `unlabeling_r3_max0/images` (temporal, 1,257 frames)
- `unlabeling_r10_max0/images` (temporal, 3,937 frames)
- `unlabeling_std_matched_r3/images` (random matched, 1,257 frames)
- `unlabeling_std_matched_r10/images` (random matched, 3,937 frames)

Matched controls: mismo numero de frames por video que el temporal correspondiente.

---

## Label fractions

Generados por `prep_label_fractions.py`:
- `label_fractions/frac_25/stems.txt` (66 stems)
- `label_fractions/frac_50/stems.txt` (111 stems)
- `label_fractions/frac_75/stems.txt` (174 stems)
- Subsets anidados con SUBSET_SEED=0
- Filtrado en datasets.py via cfg["labeled_subset_file"] (solo afecta split="train")

---

## Salidas por experimento

Dentro de cada `runs/{exp_name}/seed_{N}/`:
- `config.json`
- `best_model.pt`
- `diagnostics_epoch.csv`
- `test_metrics.csv`
- `test_boundary_metrics.csv`
- `val_boundary_metrics.csv`
- `c2c4_comparison.csv`
- `*_run_report.json` (reporte portable con todas las metricas)
- `test_preds/` (mascaras predichas)
- `test_probs/` (probability maps)
- `preds_vis/` (figuras de visualizacion)

---

## Config keys importantes

```python
# Arquitectura
"arch": "unetpp"
"backbone": "efficientnet-b3"

# SSL
"use_semi": True/False
"ssl_method": "pseudo_label" | "mean_teacher"
"lambda_u": 0.05
"tau": 0.95
"ema_decay": 0.99
"semi_start_epoch": 30
"semi_warmup_epochs": 20

# Preprocesamiento (probado en preliminares, "base" fue el mejor)
"image_preproc": "base"
"mask_smoothing": "none"

# Label fraction (opcional, solo filtra train)
"labeled_subset_file": None | path/to/stems.txt

# Pool unlabeled
"unlabeled_subdir": "unlabeling_r10_max0/images"
```

---

## Decisiones de diseno ya tomadas (con justificacion)

- **Arquitectura:** U-Net++ seleccionado de comparacion preliminar (U-Net, U-Net++, FPN, PSPNet, DeepLabV3+)
- **Preprocesamiento:** "base" (sin CLAHE). CLAHE probado y descartado en preliminares.
- **Radios r=3 y r=10:** Pool conservador (~0.2s a 30fps) vs pool amplio (~0.7s). No son optimos; son dos puntos contrastantes para estudiar el efecto del tamano.
- **Consistency temporal:** Implementada pero deshabilitada (use_temp_consistency=False). No es parte de la contribucion actual.

---

## Convenciones de nombres

| Nombre interno | Nombre en el paper |
|---|---|
| supervised | Supervised |
| semi_std_matched_r3 | Random-matched-r3 |
| semi_r3 | Temporal-r3 |
| semi_std_matched_r10 | Random-matched-r10 |
| semi_r10 | Temporal-r10 |
| mean_teacher_r3 | MT-Temporal-r3 |
| mean_teacher_std_matched_r3 | MT-Random-matched-r3 |
| mean_teacher_r10 | MT-Temporal-r10 |
| mean_teacher_std_matched_r10 | MT-Random-matched-r10 |

---

## Evaluacion C2-C4 (preliminar)

- Extraccion geometrica de puntos C2/C4 desde mascaras predichas (PCA + deteccion de valles intervertebrales)
- Metricas: distance MAE, landmark mean error, % both landmarks < 5px
- Resultados todavia lejos de metodos dedicados de landmarks (~14px vs ~2-4px)
- Considerado analisis complementario, no contribucion principal
