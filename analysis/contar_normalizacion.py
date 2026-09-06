"""Count the normalization layers of the six architectures of tab:arch.

Batch normalization keeps running statistics, held in PyTorch buffers rather than
parameters, that are updated by a moving average on every forward pass in
training mode. Layer normalization and group normalization compute their
statistics from the sample at hand and store nothing. The distinction matters for
the semi-supervised experiments, because an unlabeled frame carries no gradient
and can therefore only reach the network through those buffers.

This script builds each architecture and reports how many layers of each kind it
has, together with the number of running-statistic buffers, which is the surface
an unlabeled frame can move.

Read only. Trains nothing and writes nothing.
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

try:
    import ml_collections  # noqa: F401
except ImportError:
    # The TransUNet configuration files only need attribute access over a dict.
    # A stand-in is registered so the count can be produced on a machine that
    # does not carry the training dependencies.
    import types

    class ConfigDict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = value

    _mod = types.ModuleType("ml_collections")
    _mod.ConfigDict = ConfigDict
    sys.modules["ml_collections"] = _mod
    print("[aviso] ml_collections no instalado: se usa un sustituto minimo")

import torch.nn as nn

from src.models import create_model

BN = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)

SMP_ARCHS = [
    ("U-Net++", "unetpp", "efficientnet-b3"),
    ("U-Net", "unet", "efficientnet-b3"),
    ("FPN", "fpn", "efficientnet-b3"),
    ("DeepLabV3+", "deeplabv3plus", "efficientnet-b3"),
    ("BiFPN-U-Net(T)", "bifpn_unet", "vgg16"),
]


def contar(modelo):
    """Return the layer counts and the number of running-statistic buffers."""
    nbn = sum(1 for m in modelo.modules() if isinstance(m, BN))
    nln = sum(1 for m in modelo.modules() if isinstance(m, nn.LayerNorm))
    ngn = sum(1 for m in modelo.modules() if isinstance(m, nn.GroupNorm))
    nbuf = sum(1 for nombre, _b in modelo.named_buffers()
               if nombre.endswith("running_mean") or nombre.endswith("running_var"))
    return nbn, nln, ngn, nbuf


def transunet():
    """Build TransUNet exactly as create_model does, minus the weight loading.

    The checkpoint reader joins archive keys with the host path separator and
    fails on Windows. Pretrained weights cannot change how many layers a network
    has, so they are skipped here and the architecture is identical to the one
    trained on Colab.
    """
    from src.transunet.vit_seg_configs import get_r50_b16_config
    from src.transunet.vit_seg_modeling import VisionTransformer

    config = get_r50_b16_config()
    config.n_classes = 1
    config.n_skip = 3
    config.patches.grid = (20, 20)
    return VisionTransformer(config=config, img_size=320, num_classes=1,
                             zero_head=False, vis=False)


def main():
    print("{:16s} {:>8} {:>8} {:>8} {:>16}".format(
        "arquitectura", "BatchN", "LayerN", "GroupN", "buffers running"))
    print("-" * 62)

    for etiqueta, arch, backbone in SMP_ARCHS:
        modelo = create_model(arch, backbone, n_classes=1, pretrained=False)
        nbn, nln, ngn, nbuf = contar(modelo)
        print("{:16s} {:8d} {:8d} {:8d} {:16d}".format(
            etiqueta, nbn, nln, ngn, nbuf))

    modelo = transunet()
    nbn, nln, ngn, nbuf = contar(modelo)
    print("{:16s} {:8d} {:8d} {:8d} {:16d}".format(
        "TransUNet", nbn, nln, ngn, nbuf))

    print()
    print("las BatchNorm de TransUNet, una por una:")
    for nombre, mod in modelo.named_modules():
        if isinstance(mod, BN):
            print("   ", nombre)


if __name__ == "__main__":
    main()
