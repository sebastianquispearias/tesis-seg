from typing import Optional

import cv2
import numpy as np

try:
    import albumentations as A
except ImportError:
    A = None


def _require_albumentations():
    if A is None:
        raise ImportError(
            "albumentations no está instalado. Instálalo con: pip install albumentations"
        )


def get_supervised_train_augmentation(cfg: dict):
    _require_albumentations()

    transforms = []

    if cfg.get("aug_train_enable", True):
        transforms.extend([
            A.HorizontalFlip(p=cfg.get("aug_horizontal_flip_p", 0.5)),
            A.ShiftScaleRotate(
                shift_limit=cfg.get("aug_shift_limit", 0.03),
                scale_limit=cfg.get("aug_scale_limit", 0.05),
                rotate_limit=cfg.get("aug_rotate_limit", 10),
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.10,
                contrast_limit=0.10,
                p=cfg.get("aug_brightness_contrast_p", 0.2),
            ),
        ])

        # Ruido gaussiano: dejar activable porque te dio señal positiva
        gn_p = cfg.get("aug_gaussian_noise_p", 0.30)
        if gn_p > 0:
            transforms.append(
                A.GaussNoise(
                    std_range=(0.02, 0.08),
                    mean_range=(0.0, 0.0),
                    per_channel=True,
                    noise_scale_factor=1.0,
                    p=gn_p,
                )
            )

        # CLAHE ocasional: dejarlo opcional
        clahe_p = cfg.get("aug_clahe_p", 0.10)
        if clahe_p > 0:
            transforms.append(
                A.CLAHE(
                    clip_limit=cfg.get("aug_clahe_clip_limit", 2.0),
                    tile_grid_size=cfg.get("aug_clahe_tile_grid_size", (8, 8)),
                    p=clahe_p,
                )
            )

    return A.Compose(transforms)


def get_weak_augmentation(cfg: dict):
    _require_albumentations()

    if not cfg.get("weak_aug_enable", True):
        return A.Compose([])

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.02,
            scale_limit=0.02,
            rotate_limit=5,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.4,
        ),
    ])


def get_strong_augmentation(cfg: dict):
    _require_albumentations()

    if not cfg.get("strong_aug_enable", True):
        return A.Compose([])

    transforms = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.08,
            rotate_limit=12,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.6,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.3,
        ),
        A.GaussNoise(
            std_range=(0.03, 0.10),
            mean_range=(0.0, 0.0),
            per_channel=True,
            noise_scale_factor=1.0,
            p=0.4,
        ),
    ]

    clahe_p = cfg.get("aug_clahe_p", 0.10)
    if clahe_p > 0:
        transforms.append(
            A.CLAHE(
                clip_limit=cfg.get("aug_clahe_clip_limit", 2.0),
                tile_grid_size=cfg.get("aug_clahe_tile_grid_size", (8, 8)),
                p=clahe_p,
            )
        )

    return A.Compose(transforms)


def apply_aug_to_image_mask(image: np.ndarray, mask: np.ndarray, aug) -> tuple[np.ndarray, np.ndarray]:
    """
    image: HWC
    mask:  HW o HWC(1)
    """
    if aug is None:
        return image, mask

    transformed = aug(image=image, mask=mask)
    return transformed["image"], transformed["mask"]


def apply_aug_to_image_only(image: np.ndarray, aug) -> np.ndarray:
    if aug is None:
        return image
    transformed = aug(image=image)
    return transformed["image"]