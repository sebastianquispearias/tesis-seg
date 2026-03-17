import torch
import segmentation_models_pytorch as smp


def create_model(arch: str, backbone: str, n_classes: int = 1):
    arch = arch.lower()

    if arch == "unet":
        return smp.Unet(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=3,
            classes=n_classes,
        )
    elif arch == "unetpp":
        return smp.UnetPlusPlus(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=3,
            classes=n_classes,
        )
    elif arch == "fpn":
        return smp.FPN(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=3,
            classes=n_classes,
        )
    elif arch == "pspnet":
        return smp.PSPNet(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=3,
            classes=n_classes,
        )
    else:
        raise ValueError(f"Arquitectura no soportada: {arch}")