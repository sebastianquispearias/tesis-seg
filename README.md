# tesis_seg

Proyecto modular para segmentación de vértebras cervicales en videofluoroscopia.

La idea del repo es mantener un notebook principal en Colab para ejecutar el experimento, mostrar imágenes, métricas y resultados, mientras que la lógica real vive en módulos Python dentro de `src/`.

---

## Estructura del repositorio

- `notebooks/01_train_eval_colab.ipynb`  
  Notebook principal para correr en Google Colab.

- `src/defaults.py`  
  Configuración base del experimento.

- `src/preprocessing.py`  
  Preprocesamiento de imagen y máscara.

- `src/augmentations.py`  
  Augmentations supervisadas y weak/strong para semi-supervisión.

- `src/datasets.py`  
  Datasets supervisados, unlabeled y temporal unlabeled, además de dataloaders.

- `src/models.py`  
  Construcción del modelo con `segmentation_models_pytorch`.

- `src/losses.py`  
  Loss supervisada BCE + Dice.

- `src/metrics.py`  
  Métricas de segmentación y métricas de borde:
  F1, IoU, BF1, ASSD y HD95.

- `src/train.py`  
  Entrenamiento supervisado / semi-supervisado teacher-student y consistencia temporal.

- `src/evaluate.py`  
  Evaluación final del checkpoint, exportación de predicciones y métricas.

- `src/visualization.py`  
  Visualización de ejemplos de dataset, predicciones y curvas de entrenamiento.

- `src/sanity_checks.py`  
  Sanity checks de métricas de borde.

- `src/ruler_eval.py`  
  Comparación entre régua manual y régua automática usando `Results.csv`.

- `src/utils.py`  
  Utilidades generales: seed, guardado de JSON, EMA, conteo de parámetros, etc.

- `requirements.txt`  
  Dependencias del proyecto.

---

## Qué hace el pipeline

1. Monta Drive en Colab.
2. Clona el repositorio.
3. Instala dependencias.
4. Construye la configuración del experimento.
5. Define augmentations.
6. Construye datasets y dataloaders.
7. Corre sanity checks.
8. Entrena el modelo.
9. Evalúa el mejor checkpoint en validación y test.
10. Exporta métricas, predicciones y comparaciones de régua.

---

## Componentes implementados

- Preprocesamiento modular.
- Augmentations supervisadas.
- Weak / strong augmentations para unlabeled.
- Dataset supervisado.
- Dataset unlabeled.
- Dataset temporal unlabeled.
- Modelo con `segmentation_models_pytorch`.
- Loss BCE + Dice.
- Entrenamiento teacher/student semi-supervisado.
- Consistencia temporal.
- Métricas F1 / IoU / BF1 / ASSD / HD95.
- Sanity checks.
- Export de `test_preds` y `test_metrics.csv`.
- Evaluación de régua manual vs automática.

---

## Salidas esperadas por experimento

Dentro de `exp_dir` se generan típicamente:

- `config.json`
- `best_model.pt`
- `train_log.csv`
- `run_summary.txt`
- `test_metrics.csv`
- `val_boundary_metrics.csv`
- `test_boundary_metrics.csv`
- `test_preds/`
- `preds_vis/`
- `ruler_compare.csv` (si `run_ruler_eval=True`)

---

## Uso esperado en Colab

El notebook `notebooks/01_train_eval_colab.ipynb` es el punto principal de ejecución.  
Ahí se configuran:

- rutas del dataset
- rutas de máscaras
- rutas de rótulos
- carpeta de salida del experimento
- arquitectura y backbone
- parámetros de semi-supervisión
- parámetros de consistencia temporal
- hiperparámetros de entrenamiento

---

## Objetivo del repositorio

Dejar el proyecto ordenado y fácil de revisar, mostrando claramente:

- qué preprocesamientos se usan
- qué augmentations se usan
- qué arquitectura y loss se usan
- cómo funciona la parte semi-supervisada
- qué métricas se reportan
- cómo se evalúa la régua automática vs manual