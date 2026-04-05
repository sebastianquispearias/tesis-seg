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
    elif arch in ["deeplabv3+", "deeplabv3plus", "deeplabv3"]:
        return smp.DeepLabV3Plus(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=3,
            classes=n_classes,
        )
    elif arch == "bifpn_unet":
        from src.bifpn_unet import BiFPNUNet
        return BiFPNUNet(in_channels=3, n_classes=n_classes)
    elif arch == "transunet":
        import os
        import numpy as np
        from src.transunet.vit_seg_configs import get_r50_b16_config
        from src.transunet.vit_seg_modeling import VisionTransformer

        print(f"[TransUNet] variant: {backbone}")
        print("[TransUNet] img_size: 320")
        print(f"[TransUNet] n_classes: {n_classes}")

        config = get_r50_b16_config()
        config.n_classes = n_classes    # force 1 for binary segmentation
        config.n_skip = 3
        config.patches.grid = (20, 20)  # 320 / 16 = 20 patches per side

        model = VisionTransformer(
            config=config, img_size=320,
            num_classes=n_classes, zero_head=False, vis=False,
        )

        ckpt_path = os.environ.get("TRANSUNET_PRETRAINED_PATH", "")
        if not ckpt_path or not os.path.isfile(ckpt_path):
            raise RuntimeError(
                "TransUNet pretrained weights not found.\n"
                f"Expected R50+ViT-B_16.npz at: {ckpt_path!r}\n"
                "Set the TRANSUNET_PRETRAINED_PATH environment variable to the .npz file path.\n"
                "Download from: https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz"
            )
        print(f"[TransUNet] loading pretrained weights from: {ckpt_path}")
        weights = np.load(ckpt_path)
        model.load_from(weights)
        print("[TransUNet] pretrained weights loaded successfully")
        return model
    else:
        raise ValueError(f"Arquitectura no soportada: {arch}")