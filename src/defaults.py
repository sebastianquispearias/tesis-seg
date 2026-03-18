from copy import deepcopy


DEFAULT_CONFIG = {
    # =========================
    # Dataset / entorno
    # =========================
    "dataset": "inca",
    "mode": "colab",
    "img_root": "",
    "msk_root": "",
    "exp_dir": "./outputs/default_run",
    "rotulos_dir": "",
    "unlabeled_subdir": "unlabeling_r10_max0/images",

    # =========================
    # Modelo
    # =========================
    "arch": "unetpp",
    "backbone": "efficientnet-b3",
    "n_classes": 1,

    # =========================
    # Entrenamiento
    # =========================
    "target_size": (320, 320),
    "use_pad": True,
    "imagenet_norm": False,

    "batch_size": 5,
    "num_workers": 4,
    "drop_last": True,
    "num_augmented": 5,

    "epochs": 2000,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "warmup_epochs": 10,
    "patience_es": 20,

    "seed": 0,
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
    "lambda_t": 0.01,
    "max_temp_delta": 2,
    "temp_start_epoch": 4000,
    "temp_warmup_epochs": 5,
    "tau_temp": 0.7,

    # =========================
    # Preprocesamiento
    # =========================
    "image_preproc": "base",
    "mask_smoothing": "none",
    "use_fixed_crop": False,

    # =========================
    # Augmentations supervisadas
    # =========================
    "aug_train_enable": True,

    # supervisado suave igual al notebook original
    "aug_horizontal_flip_p": 0.0,
    "aug_shift_limit": 0.01,
    "aug_scale_limit": 0.03,
    "aug_rotate_limit": 5,
    "aug_brightness_contrast_p": 0.4,
    "aug_random_gamma_p": 0.2,
    "aug_random_gamma_limit": (90, 110),
    "aug_gaussian_noise_p": 0.15,
    "aug_gaussian_noise_var_limit": (3.0, 12.0),
    "aug_clahe_p": 0.0,

    # weak / strong semi
    "weak_aug_enable": True,
    "strong_aug_enable": True,

    # visualización / evaluación
    "boundary_tol_px": 5,
    "max_show_preds": 6000,
    "run_ruler_eval": True,
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
        "seed",
        "batch_size",
        "num_workers",
        "drop_last",
        "num_augmented",
        "epochs",
        "lr",
        "weight_decay",
        "warmup_epochs",
        "patience_es",
        "use_semi",
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
        "image_preproc",
        "mask_smoothing",
        "use_fixed_crop",
        "unlabeled_subdir",
        "eval_threshold",
        "run_ruler_eval",
    ]
    lines = ["===== CONFIG RESUMIDA ====="]
    for k in keys:
        lines.append(f"{k}: {cfg.get(k)}")
    return "\n".join(lines)