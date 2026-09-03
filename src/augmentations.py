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


def _build_optional(factory_variants):
    """First transform that the installed albumentations accepts.

    Several transforms were renamed or given new argument names between
    albumentations 1.x and 2.x. Rather than pinning a version, which the current
    Colab image no longer supports, each candidate signature is attempted in turn
    and the first one that constructs is used. A transform that no variant can
    build returns None and is dropped from the pipeline by the caller, which is
    reported rather than silently ignored.
    """
    for factory in factory_variants:
        try:
            return factory()
        except Exception:
            continue
    return None


def get_nnunet_style_augmentation(cfg: dict, full: bool):
    """Augmentation in the style of nnU-Net's default two-dimensional pipeline.

    nnU-Net perturbs far more aggressively than the pipeline used here: rotations
    reach half a turn, scaling spans 0.7 to 1.4, and it adds blur and a simulated
    loss of resolution that this pipeline has no equivalent of. Two intensities are
    offered because the strongest settings are not obviously valid for this data: a
    lateral videofluoroscopic view has a fixed anatomical orientation, so a mirrored
    or inverted frame is one the model will never meet at test time.

    With ``full`` false the geometry stays inside what the acquisition can plausibly
    produce, and mirroring is off. With ``full`` true the settings are nnU-Net's own,
    so that a drop can be attributed to implausible geometry rather than to intensity
    of augmentation in general.
    """
    _require_albumentations()

    rotate = 180 if full else 30
    scale_low, scale_high = (0.7, 1.4) if full else (0.85, 1.15)

    transforms = [
        A.ShiftScaleRotate(
            shift_limit=0.0,
            scale_limit=(scale_low - 1.0, scale_high - 1.0),
            rotate_limit=rotate,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.2,
        ),
    ]

    if full:
        transforms += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)]

    transforms += [
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.0, p=0.15),
        A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.25, p=0.15),
        A.RandomGamma(gamma_limit=(70, 150), p=0.3),
    ]

    blur = _build_optional([
        lambda: A.GaussianBlur(blur_limit=0, sigma_limit=(0.5, 1.0), p=0.2),
        lambda: A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ])
    lowres = _build_optional([
        lambda: A.Downscale(scale_range=(0.5, 0.9), p=0.25),
        lambda: A.Downscale(scale_min=0.5, scale_max=0.9, p=0.25),
    ])
    noise = _build_optional([
        lambda: A.GaussNoise(std_range=(0.0, 0.1), p=0.1),
        lambda: A.GaussNoise(var_limit=(0.0, 25.0), p=0.1),
    ])

    dropped = []
    for name, tf in (("GaussianBlur", blur), ("Downscale", lowres), ("GaussNoise", noise)):
        if tf is None:
            dropped.append(name)
        else:
            transforms.append(tf)

    if dropped:
        raise RuntimeError(
            "La albumentations instalada no acepta ninguna firma conocida de: "
            + ", ".join(dropped)
            + ". Abortando en vez de entrenar con una augmentation incompleta."
        )

    return A.Compose(transforms)


def get_supervised_train_augmentation(cfg: dict):
    _require_albumentations()

    if not cfg.get("aug_train_enable", True):
        return A.Compose([])

    profile = str(cfg.get("aug_profile", "base")).lower()
    if profile in ("nnunet_moderate", "nnunet_full"):
        return get_nnunet_style_augmentation(cfg, full=(profile == "nnunet_full"))
    if profile != "base":
        raise ValueError(f"aug_profile no soportado: {profile}")

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