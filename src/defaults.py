from copy import deepcopy


DEFAULT_CONFIG = {
    # =========================
    # Dataset / entorno
    # =========================
    "dataset": "inca",
    "mode": "colab",   # "local" o "colab"
    "img_root": "",
    "msk_root": "",
    "exp_dir": "./outputs/default_run",
    "boundary_tol_px": 2,
    "max_show_preds": 6,
    "run_ruler_eval": False,
    "rotulos_dir": "",

    # =========================
    # Modelo
    # =========================
    "arch": "unetpp",              # "unet", "unetpp", "fpn", "pspnet", ...
    "backbone": "efficientnet-b3",
    "n_classes": 1,

    # =========================
    # Entrenamiento
    # =========================
    "batch_size": 4,
    "num_workers": 2,
    "epochs": 100,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "seed": 42,
    "device": "cuda",

    # =========================
    # Loss / evaluación
    # =========================
    "loss_name": "bce_dice",
    "eval_threshold": 0.5,

    # =========================
    # Semi-supervisión
    # =========================
    "use_semi": True,
    "lambda_u": 0.05,
    "tau": 0.95,
    "ema_decay": 0.99,
    "semi_start_epoch": 30,
    "semi_warmup_epochs": 20,

    # =========================
    # Consistencia temporal
    # =========================
    "use_temp_consistency": False,
    "lambda_t": 0.00,
    "max_temp_delta": 2,
    "temp_start_epoch": 4000,
    "temp_warmup_epochs": 5,
    "tau_temp": 0.7,

    # =========================
    # Preprocesamiento
    # =========================
    "target_size": (320, 320),
    "use_pad": True,
    "imagenet_norm": False,
    "image_preproc": "base",       # "base", "denoise", "he", "clahe_soft", "ad"
    "mask_smoothing": "none",      # "none", "morph", "gaussian"
    "use_fixed_crop": False,

    # =========================
    # Augmentations supervisadas
    # =========================
    "aug_train_enable": True,
    "aug_horizontal_flip_p": 0.5,
    "aug_vertical_flip_p": 0.0,
    "aug_rotate_limit": 10,
    "aug_shift_limit": 0.03,
    "aug_scale_limit": 0.05,
    "aug_brightness_contrast_p": 0.2,
    "aug_gaussian_noise_p": 0.30,
    "aug_gaussian_noise_var_limit": (5.0, 25.0),
    "aug_clahe_p": 0.10,
    "aug_clahe_clip_limit": 2.0,
    "aug_clahe_tile_grid_size": (8, 8),

    # =========================
    # Weak / strong para unlabeled
    # =========================
    "weak_aug_enable": True,
    "strong_aug_enable": True,

    # =========================
    # Debug / visualización
    # =========================
    "debug": False,
    "save_outputs": True,
    "show_examples": True,
}


def get_default_config() -> dict:
    return deepcopy(DEFAULT_CONFIG)


def update_config(base_cfg: dict, **kwargs) -> dict:
    cfg = deepcopy(base_cfg)
    for k, v in kwargs.items():
        cfg[k] = v
    return cfg


def summarize_config(cfg: dict) -> str:
    keys = [
        "dataset",
        "mode",
        "arch",
        "backbone",
        "loss_name",
        "use_semi",
        "image_preproc",
        "mask_smoothing",
        "use_fixed_crop",
        "lambda_u",
        "tau",
        "ema_decay",
        "semi_start_epoch",
        "semi_warmup_epochs",
        "use_temp_consistency",
        "lambda_t",
        "temp_start_epoch",
        "temp_warmup_epochs",
        "tau_temp",
        "max_temp_delta",
        "batch_size",
        "epochs",
        "lr",
        "eval_threshold",
        "run_ruler_eval",
    ]
    lines = ["===== CONFIG RESUMIDA ====="]
    for k in keys:
        lines.append(f"{k}: {cfg.get(k)}")
    return "\n".join(lines)