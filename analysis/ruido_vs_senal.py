"""Mide el ruido que la augmentacion aplica de verdad contra el contraste
real de una vertebra en un frame de VFSS del dataset de train."""

import glob
import os
import warnings

import cv2
import numpy as np
import albumentations as A

warnings.filterwarnings("ignore")

TRAIN = r"G:\My Drive\UNM_vertebras_seg_v3\data\inca_dataset\train"

nombres = sorted(
    os.path.basename(p) for p in glob.glob(os.path.join(TRAIN, "masks", "*.png"))
)

print("albumentations instalada:", A.__version__)
print("frames de train disponibles:", len(nombres))
print()

# ---------------------------------------------------------------- 1. la senal
print("=" * 72)
print("1. LA SENAL: cuanto contraste hay entre la vertebra y lo que la rodea")
print("=" * 72)

contrastes = []
for n in nombres[:40]:
    img = cv2.imread(os.path.join(TRAIN, "images", n), cv2.IMREAD_GRAYSCALE)
    msk = cv2.imread(os.path.join(TRAIN, "masks", n), cv2.IMREAD_GRAYSCALE)
    if img is None or msk is None:
        continue
    dentro = msk > 127
    if dentro.sum() < 50:
        continue
    # anillo de 5 px alrededor de la vertebra, que es donde cae el 80-90% del error
    fuera = (cv2.dilate(dentro.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0) & ~dentro
    contrastes.append(abs(img[dentro].mean() - img[fuera].mean()))

contrastes = np.array(contrastes)
print(f"  frames medidos                          : {len(contrastes)}")
print(f"  contraste vertebra vs anillo de 5 px    : {contrastes.mean():.1f} niveles "
      f"(mediana {np.median(contrastes):.1f}, min {contrastes.min():.1f}, max {contrastes.max():.1f})")
print()

# ---------------------------------------------------------------- 2. el ruido
print("=" * 72)
print("2. EL RUIDO: lo que el codigo PIDE contra lo que la libreria APLICA")
print("=" * 72)

base = cv2.imread(os.path.join(TRAIN, "images", nombres[0]), cv2.IMREAD_GRAYSCALE)
base3 = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)

# tal cual esta escrito en src/augmentations.py, linea 189
pedido = A.GaussNoise(var_limit=(4.0, 16.0), p=1.0)

stds = []
for _ in range(200):
    out = pedido(image=base3)["image"].astype(np.float32)
    stds.append((out - base3.astype(np.float32)).std())
stds = np.array(stds)

print("  linea del repo : A.GaussNoise(var_limit=(4.0, 16.0), p=0.25)")
print(f"  PIDE           : varianza 4 a 16  ->  std de {np.sqrt(4.0):.1f} a {np.sqrt(16.0):.1f} niveles")
print(f"  APLICA (2.0.8) : std medida de {stds.min():.1f} a {stds.max():.1f} niveles "
      f"(media {stds.mean():.1f})")
print(f"  parametros reales del objeto: {pedido.to_dict()['transform']}")
print()

# ------------------------------------------------------- 3. senal contra ruido
print("=" * 72)
print("3. SENAL CONTRA RUIDO")
print("=" * 72)
print(f"  contraste real de la vertebra          : {contrastes.mean():.1f} niveles")
print(f"  ruido PEDIDO      (albumentations 1.x) : {np.sqrt(4.0):.1f} a {np.sqrt(16.0):.1f} niveles"
      f"   ->  {contrastes.mean()/3.0:.1f} veces menor que la senal")
print(f"  ruido APLICADO    (albumentations 2.x) : {stds.mean():.1f} niveles"
      f"          ->  {stds.mean()/contrastes.mean():.1f} veces MAYOR que la senal")
print()

# --------------------------------------- 4. se sigue viendo la vertebra o no
print("=" * 72)
print("4. LA PRUEBA: se puede seguir distinguiendo la vertebra?")
print("=" * 72)

msk = cv2.imread(os.path.join(TRAIN, "masks", nombres[0]), cv2.IMREAD_GRAYSCALE) > 127
anillo = (cv2.dilate(msk.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0) & ~msk


def separabilidad(im):
    """d' de deteccion: cuantas desviaciones separan dentro de fuera."""
    a, b = im[msk].astype(np.float32), im[anillo].astype(np.float32)
    return abs(a.mean() - b.mean()) / np.sqrt(0.5 * (a.var() + b.var()))


d_limpio = separabilidad(base)

suave = np.clip(
    base.astype(np.float32) + np.random.RandomState(0).normal(0, 3.0, base.shape), 0, 255
).astype(np.uint8)
d_suave = separabilidad(suave)

ruidoso = pedido(image=base3)["image"][:, :, 0]
d_ruidoso = separabilidad(ruidoso)

print(f"  frame usado: {nombres[0]}")
print(f"  d' frame limpio                        : {d_limpio:.3f}")
print(f"  d' con el ruido PEDIDO   (std 3)       : {d_suave:.3f}   "
      f"({100*(d_suave/d_limpio-1):+.1f} %)")
print(f"  d' con el ruido APLICADO (std ~71)     : {d_ruidoso:.3f}   "
      f"({100*(d_ruidoso/d_limpio-1):+.1f} %)")
print()
print("  d' es la separabilidad: cuantas desviaciones tipicas separan el interior")
print("  de la vertebra del anillo que la rodea. Es lo que la red tiene que ver.")
