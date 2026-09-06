"""Show what a single unlabeled frame does to a network that cannot learn from it.

One real frame from the unlabeled pool is passed forward in training mode with no
label, no loss and no backward pass, inside torch.no_grad. Learning is therefore
switched off and nothing that moves can be attributed to it. The script reports
how much the learnable parameters moved, which must be exactly zero, and how much
the normalization buffers moved, which is the channel through which an unlabeled
frame reaches the model.

The same frame is run through U-Net++, whose encoder is built on batch
normalization, and through TransUNet, whose encoder uses group and layer
normalization and keeps almost no running statistics.

Read only. Trains nothing and writes nothing.
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from contar_normalizacion import transunet  # noqa: E402  installs the shim too

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.models import create_model  # noqa: E402

BASE = os.path.dirname(REPO)
FRAME = os.path.join(BASE, "unlabeling_std_matched_r15", "images", "v025_f1005.png")


def cargar_frame(ruta):
    """Load one unlabeled frame as the three-channel tensor the models expect."""
    im = Image.open(ruta).convert("L").resize((320, 320))
    x = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(x)[None, None].repeat(1, 3, 1, 1)


def instantanea(modelo):
    """Copy every learnable parameter and every normalization buffer."""
    pesos = {k: v.detach().clone() for k, v in modelo.named_parameters()}
    bufs = {k: v.detach().clone().float() for k, v in modelo.named_buffers()
            if k.endswith("running_mean") or k.endswith("running_var")}
    return pesos, bufs


def cambio(antes, despues):
    """Total absolute change and how many tensors moved at all."""
    total, movidos = 0.0, 0
    for k in antes:
        d = (despues[k] - antes[k]).abs().sum().item()
        total += d
        if d > 0:
            movidos += 1
    return total, movidos, len(antes)


def probar(etiqueta, modelo, x):
    modelo.train()                    # training mode, as in the Mean Teacher pass
    p0, b0 = instantanea(modelo)
    with torch.no_grad():             # no gradient: nothing can be learned
        modelo(x)                     # no label and no loss
    p1, b1 = instantanea(modelo)

    tp, mp, np_ = cambio(p0, p1)
    tb, mb, nb = cambio(b0, b1)
    print("-" * 68)
    print(etiqueta)
    print("  pesos entrenables       : cambio total {:.6f}   movidos {} de {}".format(
        tp, mp, np_))
    print("  buffers de normalizacion: cambio total {:.6f}   movidos {} de {}".format(
        tb, mb, nb))


def main():
    if not os.path.isfile(FRAME):
        raise SystemExit("no se encuentra el fotograma: {}".format(FRAME))
    x = cargar_frame(FRAME)
    print("fotograma sin etiquetar:", FRAME)
    print("tensor de entrada:", tuple(x.shape))
    print()
    probar("U-Net++ / EfficientNet-B3",
           create_model("unetpp", "efficientnet-b3", n_classes=1, pretrained=False), x)
    probar("TransUNet / R50-ViT-B_16", transunet(), x)


if __name__ == "__main__":
    main()
