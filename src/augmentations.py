import cv2

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

    if not cfg.get("aug_train_enable", True):
        return A.Compose([])

    return A.Compose([
        A.ShiftScaleRotate(
            shift_limit=cfg.get("aug_shift_limit", 0.01),
            scale_limit=cfg.get("aug_scale_limit", 0.03),
            rotate_limit=cfg.get("aug_rotate_limit", 5),
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.5,
        ),
        # En el notebook original el flip estaba comentado.
        # Lo dejamos controlado por cfg pero apagado por defecto.
        A.HorizontalFlip(p=cfg.get("aug_horizontal_flip_p", 0.0)),
        A.RandomBrightnessContrast(
            brightness_limit=0.10,
            contrast_limit=0.10,
            p=cfg.get("aug_brightness_contrast_p", 0.4),
        ),
        A.RandomGamma(
            gamma_limit=cfg.get("aug_random_gamma_limit", (90, 110)),
            p=cfg.get("aug_random_gamma_p", 0.2),
        ),
        A.GaussNoise(
            var_limit=cfg.get("aug_gaussian_noise_var_limit", (3.0, 12.0)),
            p=cfg.get("aug_gaussian_noise_p", 0.15),
        ),
    ])


def get_weak_augmentation(cfg: dict):
    _require_albumentations()

    if not cfg.get("weak_aug_enable", True):
        return A.Compose([])

    return A.Compose([
        A.ShiftScaleRotate(
            shift_limit=0.01,
            scale_limit=0.02,
            rotate_limit=3,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.5,
        ),
        A.HorizontalFlip(p=0.1),
    ])


def get_strong_augmentation(cfg: dict):
    _require_albumentations()

    if not cfg.get("strong_aug_enable", True):
        return A.Compose([])

    return A.Compose([
        A.ShiftScaleRotate(
            shift_limit=0.015,
            scale_limit=0.04,
            rotate_limit=6,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.6,
        ),
        A.HorizontalFlip(p=0.15),
        A.RandomBrightnessContrast(
            brightness_limit=0.12,
            contrast_limit=0.12,
            p=0.6,
        ),
        A.RandomGamma(
            gamma_limit=(88, 112),
            p=0.3,
        ),
        A.GaussNoise(
            var_limit=(4.0, 16.0),
            p=0.25,
        ),
    ])


def apply_aug_to_image_mask(image, mask, aug):
    if aug is None:
        return image, mask
    transformed = aug(image=image, mask=mask)
    return transformed["image"], transformed["mask"]


def apply_aug_to_image_only(image, aug):
    if aug is None:
        return image
    transformed = aug(image=image)
    return transformed["image"]