"""Record the environment a run was trained in, next to the run itself.

Every experiment writes a ``debug_fingerprint.json`` inside its experiment
directory, and ``evaluate.py`` lifts a compact subset of it into the run report.
Without that file a report carries ``has_debug_fingerprint: false`` and the run
cannot be placed on a software stack afterwards, which makes it unusable in any
comparison that spans months.

The collector used to live inside each training notebook and was copied by hand,
so notebooks written later lost it. Keeping it here means a notebook enables the
whole thing with a single import.

Nothing in this module is allowed to interrupt a training run: every section is
guarded, and a section that fails records its error instead of raising. The model
fingerprint saves and restores the random number generator state, so building the
throwaway model leaves the run bit-identical to one that never called this.

Typical use, from a notebook, around the existing training call::

    from src.fingerprint import collect_pre, collect_post, save

    pre = collect_pre(cfg, loaders, train_ds, val_ds, test_ds, unlabeled_ds)
    artifacts = run_training(...)
    results = evaluate_checkpoint(...)
    save(pre, collect_post(artifacts, results), cfg["exp_dir"])
"""

import importlib.metadata
import json
import os
import platform
import subprocess
import time


def _version(pkg):
    """Installed version of a distribution, or None when it is absent."""
    try:
        return importlib.metadata.version(pkg)
    except Exception:
        return None


def _git_hash_from_src(src_train_file):
    """One-line git description of the checkout that provided src/train.py."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(src_train_file)))
    try:
        return subprocess.check_output(
            ["git", "log", "--oneline", "-1"], cwd=repo_root,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _environment():
    """Interpreter, framework and augmentation library versions, plus the GPU."""
    import sys

    import torch

    try:
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a"
    except Exception:
        gpu = "n/a"
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": _version("torchvision"),
        "cuda": torch.version.cuda,
        "cudnn": (str(torch.backends.cudnn.version())
                  if torch.cuda.is_available() else "n/a"),
        "segmentation_models_pytorch": _version("segmentation-models-pytorch"),
        "albumentations": _version("albumentations"),
        "platform": platform.platform(),
        "gpu_name": gpu,
    }


def _provenance():
    """Where the source modules were imported from, and at which commit."""
    import src.datasets
    import src.defaults
    import src.evaluate
    import src.train

    return {
        "src_train": src.train.__file__,
        "src_datasets": src.datasets.__file__,
        "src_defaults": src.defaults.__file__,
        "src_evaluate": src.evaluate.__file__,
        "git_hash": _git_hash_from_src(src.train.__file__),
    }


CFG_KEYS = [
    "seed", "arch", "backbone", "n_classes",
    "image_preproc", "mask_smoothing", "target_size", "use_pad", "imagenet_norm",
    "batch_size", "num_workers", "drop_last", "num_augmented",
    "lr", "weight_decay", "epochs", "warmup_epochs", "patience_es",
    "eval_threshold", "use_semi", "use_temp_consistency",
    "lambda_u", "tau", "ema_decay", "semi_start_epoch", "semi_warmup_epochs",
    "lambda_t", "freeze_bn_on_unlabeled", "ssl_method",
    "unlabeled_subdir", "exp_dir",
]


def _model_fingerprint(cfg):
    """Parameter counts of a throwaway model, with the RNG state left untouched."""
    import random

    import numpy as np
    import torch

    from src.models import create_model

    estado = {
        "py": random.getstate(),
        "np": np.random.get_state(),
        "th": torch.get_rng_state(),
        "cuda": (torch.cuda.get_rng_state_all()
                 if torch.cuda.is_available() else None),
    }
    try:
        m = create_model(cfg["arch"], cfg["backbone"], cfg["n_classes"])
        fp = {
            "total_params": sum(p.numel() for p in m.parameters()),
            "trainable_params": sum(p.numel() for p in m.parameters()
                                    if p.requires_grad),
            "first_state_dict_keys": list(m.state_dict().keys())[:8],
        }
        del m
    finally:
        random.setstate(estado["py"])
        np.random.set_state(estado["np"])
        torch.set_rng_state(estado["th"])
        if estado["cuda"] is not None:
            torch.cuda.set_rng_state_all(estado["cuda"])
    return fp


def collect_pre(cfg, loaders, train_ds, val_ds, test_ds,
                unlabeled_ds=None, temporal_unlab_ds=None):
    """Everything worth knowing about a run before it starts."""
    unlab = loaders.get("unlabeled_loader") if isinstance(loaders, dict) else None

    def guardado(fn, *a):
        try:
            return fn(*a)
        except Exception as e:  # noqa: BLE001  a fingerprint must never stop a run
            return {"error": str(e)}

    eff_cfg = {k: cfg.get(k) for k in CFG_KEYS}
    eff_cfg["batch_size_unlab"] = unlab.batch_size if unlab is not None else None

    try:
        ds_facts = {
            "len_train_ds": len(train_ds),
            "len_val_ds": len(val_ds),
            "len_test_ds": len(test_ds),
            "len_unlabeled_ds": len(unlabeled_ds) if unlabeled_ds is not None else None,
            "len_temporal_unlab_ds": (len(temporal_unlab_ds)
                                      if temporal_unlab_ds is not None else None),
            "train_loader_batch_size": loaders["train_loader"].batch_size,
            "train_loader_num_workers": loaders["train_loader"].num_workers,
            "train_loader_drop_last": loaders["train_loader"].drop_last,
            "unlabeled_loader_batch_size": unlab.batch_size if unlab else None,
            "unlabeled_loader_num_workers": unlab.num_workers if unlab else None,
            "unlabeled_loader_drop_last": unlab.drop_last if unlab else None,
            "batch_size": cfg.get("batch_size"),
            "batch_size_unlab": unlab.batch_size if unlab is not None else None,
        }
    except Exception as e:  # noqa: BLE001
        ds_facts = {"error": str(e)}

    try:
        sup_ids = train_ds.files[:5]
    except Exception as e:  # noqa: BLE001
        sup_ids = "unavailable: {}".format(e)
    try:
        unl_ids = unlabeled_ds.files[:5] if unlabeled_ds is not None else None
    except Exception as e:  # noqa: BLE001
        unl_ids = "unavailable: {}".format(e)

    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": guardado(_environment),
        "provenance": guardado(_provenance),
        "effective_cfg": eff_cfg,
        "dataset_facts": ds_facts,
        "sample_ids": {"first5_train": sup_ids, "first5_unlabeled": unl_ids},
        "model_fingerprint": guardado(_model_fingerprint, cfg),
    }


def collect_post(artifacts, results):
    """The little that is worth keeping once the run has finished."""
    history = (artifacts or {}).get("history") or []
    best = max(history, key=lambda r: r.get("val_iou_global", 0.0)) if history else None
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "best_path": (artifacts or {}).get("best_path"),
        "best_epoch_info": {
            "epoch": best.get("epoch") if best else None,
            "val_iou_global": best.get("val_iou_global") if best else None,
            "val_loss": best.get("val_loss") if best else None,
        },
        "val_metrics": (results or {}).get("val_metrics", {}),
        "test_metrics": (results or {}).get("test_metrics", {}),
    }


def save(pre, post, exp_dir):
    """Write debug_fingerprint.json where evaluate.py will look for it."""
    path = os.path.join(exp_dir, "debug_fingerprint.json")
    os.makedirs(exp_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"pre_run": pre, "post_run": post}, fh, indent=2, default=str)
    return path
