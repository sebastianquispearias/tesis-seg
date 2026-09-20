from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image


def pad_to_square(img: Image.Image, fill_color=(0, 0, 0)) -> Image.Image:
    w, h = img.size
    side = max(w, h)
    canvas = Image.new(img.mode, (side, side), color=fill_color)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image


def apply_image_preproc(image_uint8: np.ndarray, mode: str = "base") -> np.ndarray:
    image_uint8 = ensure_rgb(image_uint8)

    if mode == "base":
        return image_uint8

    gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)

    if mode == "denoise":
        out = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)

    if mode == "he":
        out = cv2.equalizeHist(gray)
        return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)

    if mode == "clahe_soft":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out = clahe.apply(gray)
        return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)

    if mode == "ad":
        # difusión anisotrópica simple estilo Perona-Malik
        gray_f = gray.astype(np.float32)
        n_iters = 10
        kappa = 20.0
        gamma = 0.15

        for _ in range(n_iters):
            north = np.roll(gray_f, -1, axis=0) - gray_f
            south = np.roll(gray_f, 1, axis=0) - gray_f
            east = np.roll(gray_f, -1, axis=1) - gray_f
            west = np.roll(gray_f, 1, axis=1) - gray_f

            c_n = np.exp(-(north / kappa) ** 2)
            c_s = np.exp(-(south / kappa) ** 2)
            c_e = np.exp(-(east / kappa) ** 2)
            c_w = np.exp(-(west / kappa) ** 2)

            gray_f = gray_f + gamma * (
                c_n * north + c_s * south + c_e * east + c_w * west
            )

        out = np.clip(gray_f, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)

    raise ValueError(f"Modo de preprocesamiento no soportado: {mode}")


def smooth_mask(mask_uint8: np.ndarray, mode: str = "none") -> np.ndarray:
    mask = np.squeeze(mask_uint8).astype(np.uint8)
    mask = (mask > 127).astype(np.uint8) * 255

    if mode == "none":
        return mask

    if mode == "morph":
        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        return closed

    if mode == "gaussian":
        blur = cv2.GaussianBlur(mask, (5, 5), 0)
        out = (blur > 127).astype(np.uint8) * 255
        return out

    raise ValueError(f"Modo de suavizado de máscara no soportado: {mode}")


def preprocess_image_and_mask(
    image_uint8: np.ndarray,
    mask_uint8: np.ndarray,
    target_size: Tuple[int, int] = (320, 320),
    use_pad: bool = True,
    imagenet_norm: bool = False,
    image_preproc: str = "base",
    mask_smoothing: str = "none",
    image_norm: str = "unit",
    debug: bool = False,
) -> Dict[str, np.ndarray]:
    image_uint8 = ensure_rgb(image_uint8)
    mask_uint8 = np.squeeze(mask_uint8)

    if use_pad:
        image = np.array(pad_to_square(Image.fromarray(image_uint8), fill_color=(0, 0, 0)))
        mask = np.array(pad_to_square(Image.fromarray(mask_uint8), fill_color=0))
    else:
        image = image_uint8.copy()
        mask = mask_uint8.copy()

    w, h = target_size
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    if debug:
        print("[DEBUG] BEFORE apply_image_preproc")
        print(" image dtype:", image.dtype, "shape:", image.shape)
        print(" image min/max:", float(image.min()), float(image.max()))

    image = apply_image_preproc(image, mode=image_preproc)

    if debug:
        print("[DEBUG] AFTER apply_image_preproc")
        print(" image dtype:", image.dtype, "shape:", image.shape)
        print(" image min/max:", float(image.min()), float(image.max()))

    mask = smooth_mask(mask, mode=mask_smoothing)
    mask = (mask > 127).astype(np.float32)

    image = image.astype(np.float32) / 255.0

    if imagenet_norm:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
    elif image_norm == "zscore":
        # nnU-Net normalises every image to zero mean and unit variance rather than
        # rescaling it to the unit interval, and its plan for this dataset selects
        # ZScoreNormalization. Statistics are taken over the whole padded frame, the
        # only region available at inference time. The guard keeps a constant image,
        # which the padding can produce, from dividing by zero.
        mean = float(image.mean())
        std = float(image.std())
        image = (image - mean) / (std if std > 1e-8 else 1.0)
    elif image_norm != "unit":
        raise ValueError(f"image_norm no soportado: {image_norm}")

    # CHW para PyTorch
    image = np.transpose(image, (2, 0, 1))
    mask = np.expand_dims(mask, axis=0)

    return {
        "image": image.astype(np.float32),
        "mask": mask.astype(np.float32),
    }