"""Equivalence gate for the augmentation fixes.

Three checks have to pass before anything is launched: no argument is silently
dropped any more, the noise that is actually applied sits in the range the code
asks for, and nothing else in any of the five augmentation pipelines changed.

Both sides of the comparison are passed through JSON so that a tuple and a list
holding the same numbers compare equal, which they do not in memory.
"""

import copy
import io
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, r"G:\My Drive\UNM_vertebras_seg_v3\tesis_seg")

import albumentations as A  # noqa: E402
from src.augmentations import (  # noqa: E402
    get_strong_augmentation,
    get_supervised_train_augmentation,
    get_weak_augmentation,
)

ANTES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aug_ANTES.json")

PERFILES = {
    "sup_base": lambda: get_supervised_train_augmentation({}),
    "weak": lambda: get_weak_augmentation({}),
    "strong": lambda: get_strong_augmentation({}),
    "nnunet_moderate": lambda: get_supervised_train_augmentation(
        {"aug_profile": "nnunet_moderate"}
    ),
    "nnunet_full": lambda: get_supervised_train_augmentation(
        {"aug_profile": "nnunet_full"}
    ),
}


def normalizar(obj):
    """JSON round trip, so that tuples and lists of equal numbers compare equal."""
    return json.loads(json.dumps(obj))


print("albumentations:", A.__version__)
fallos = []

# ------------------------------------------------ 1. ningun argumento descartado
print()
print("=" * 74)
print("CHECK 1 - ningun argumento se descarta en silencio")
print("=" * 74)
construidos = {}
for nombre, fabrica in PERFILES.items():
    with warnings.catch_warnings(record=True) as capturados:
        warnings.simplefilter("always")
        construidos[nombre] = fabrica()
    descartados = [str(w.message) for w in capturados if "not valid" in str(w.message)]
    if descartados:
        fallos.append(f"{nombre}: argumentos descartados")
    print(f"  {nombre:18s} {'OK' if not descartados else 'FALLA'}")
    for d in descartados:
        print(f"      -> {d}")

# ------------------------------------------------------ 2. el ruido que se aplica
print()
print("=" * 74)
print("CHECK 2 - el ruido aplicado cae en el rango que el codigo pide")
print("=" * 74)
base = np.full((512, 512, 3), 128, np.uint8)
ESPERADO = {
    "sup_base": (3.0, 12.0),   # var_limit que pide el codigo
    "strong": (4.0, 16.0),
}
for nombre, (vlo, vhi) in ESPERADO.items():
    ruido = [t for t in construidos[nombre].transforms
             if t.__class__.__name__ == "GaussNoise"][0]
    params = ruido.to_dict()["transform"]

    # 2a. comprobacion EXACTA del parametro, sin ruido de medicion
    quiere = (vlo ** 0.5 / 255.0, vhi ** 0.5 / 255.0)
    tiene = tuple(params.get("std_range", (None, None)))
    exacto = all(abs(a - b) < 1e-12 for a, b in zip(quiere, tiene))
    if not exacto:
        fallos.append(f"{nombre}: std_range no es el pedido")
    print(f"  {nombre:12s} std_range pedido {quiere[0]:.8f}-{quiere[1]:.8f}  "
          f"puesto {tiene[0]:.8f}-{tiene[1]:.8f}   {'OK' if exacto else 'FALLA'}")

    # 2b. control de magnitud. La std medida queda por DEBAJO de la pedida porque
    # el resultado se cuantiza a uint8, y a uno o dos niveles el redondeo se come
    # parte del ruido. Lo que se comprueba es el orden de magnitud, no la igualdad.
    forzado = copy.deepcopy(ruido)
    forzado.p = 1.0
    solo = A.Compose([forzado])
    stds = np.array([(solo(image=base)["image"].astype(np.float32) - 128.0).std()
                     for _ in range(200)])
    razonable = stds.max() <= vhi ** 0.5 * 1.10
    if not razonable:
        fallos.append(f"{nombre}: la magnitud del ruido se disparo")
    print(f"  {nombre:12s} std medida {stds.min():.2f}-{stds.max():.2f} "
          f"(tope {vhi ** 0.5:.2f}, la rota daba 63)   {'OK' if razonable else 'FALLA'}")

# --------------------------------------------- 3. nada mas cambio en los pipelines
print()
print("=" * 74)
print("CHECK 3 - lo unico que cambio es el GaussNoise")
print("=" * 74)
antes = json.load(io.open(ANTES, encoding="utf-8"))
for nombre, compose in construidos.items():
    viejo = normalizar(antes[nombre]["transform"]["transforms"])
    nuevo = normalizar(compose.to_dict()["transform"]["transforms"])
    nv = [t["__class_fullname__"] for t in viejo]
    nn = [t["__class_fullname__"] for t in nuevo]
    if nv != nn:
        fallos.append(f"{nombre}: cambio la lista de transformaciones")
        print(f"  {nombre:18s} FALLA  {nv}  ->  {nn}")
        continue
    difs = [n for a, b, n in zip(viejo, nuevo, nn) if a != b]
    admitido = all(d == "GaussNoise" for d in difs)
    if not admitido:
        fallos.append(f"{nombre}: cambio algo que no es GaussNoise")
    print(f"  {nombre:18s} {'OK' if admitido else 'FALLA'}  "
          f"{len(nn)} transformaciones, cambiadas: {difs if difs else 'ninguna'}")

# ------------------------------------------------------------------- veredicto
print()
print("=" * 74)
if fallos:
    print("PUERTA CERRADA. No lanzar nada.")
    for f in fallos:
        print("  -", f)
    sys.exit(1)
print("PUERTA ABIERTA. Los tres checks pasan.")
