# tesis_seg

Proyecto modular para segmentación de vértebras en Colab.

## Estructura
- `notebooks/01_train_eval_colab.ipynb`: notebook principal
- `src/defaults.py`: configuración base
- `src/preprocessing.py`: preprocesamiento de imagen y máscara
- `src/augmentations.py`: augmentations supervisadas y weak/strong
- `src/datasets.py`: datasets y dataloaders
- `src/models.py`: modelos de segmentación
- `src/losses.py`: losses
- `src/train.py`: entrenamiento
- `src/evaluate.py`: evaluación final
- `src/metrics.py`: métricas
- `src/sanity_checks.py`: sanity checks
- `src/ruler_eval.py`: comparación régua manual vs automática