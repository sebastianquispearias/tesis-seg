"""Skeleton of the run_arm cell that every training notebook should start from.

Not meant to be imported or executed: it is the shape to copy into a new
notebook. It exists because the three notebooks written in September each lost a
piece of this by being copied from the previous one by hand, and nineteen runs
were trained without their unlabeled pool while their configuration claimed
otherwise.

The rule the guards encode: check the effect, never the intention. A
configuration records what was asked for. Only the loaders and the training
history record what happened.
"""

import os
import time

from src.datasets import (build_supervised_datasets, build_dataloaders,
                          build_unlabeled_datasets)
from src.augmentations import get_supervised_train_augmentation
from src.defaults import get_default_config
from src.evaluate import evaluate_checkpoint
from src.fingerprint import collect_post, collect_pre, save as save_fingerprint
from src.tee import tee_output
from src.train import run_training


def run_arm(nombre, semilla, semi, out_root, base, rotulos):
    """Train one arm and return its experiment directory."""
    cfg = get_default_config()
    cfg["img_root"] = base
    cfg["msk_root"] = base
    cfg["rotulos_dir"] = rotulos
    cfg["exp_dir"] = f"{out_root}/{nombre}/seed_{semilla}"
    cfg["seed"] = semilla
    cfg["use_semi"] = bool(semi)
    # ... el resto de la configuracion del experimento ...

    _exp = cfg["exp_dir"]
    _best = os.path.isfile(os.path.join(_exp, "best_model.pt"))
    _metrics = os.path.isfile(os.path.join(_exp, "test_metrics.csv"))
    _summary = os.path.isfile(os.path.join(_exp, "run_summary.txt"))
    if _best and _metrics and _summary:
        print(f"Skipping {nombre}/seed_{semilla}: run completo detectado")
        return _exp

    train_tf = get_supervised_train_augmentation(cfg)
    train_ds, val_ds, test_ds = build_supervised_datasets(cfg, train_tf=train_tf)

    # 1. EL POOL. build_dataloaders no lo crea, solo envuelve el que se le pase.
    #    Omitir estas tres lineas deja unlabeled_loader en None y la rama SSL no
    #    se ejecuta, sin error y sin aviso.
    unlabeled_ds, temporal_unlab_ds = None, None
    if cfg.get("use_semi", False) or cfg.get("use_temp_consistency", False):
        unlabeled_ds, temporal_unlab_ds = build_unlabeled_datasets(cfg)
    loaders = build_dataloaders(cfg, train_ds=train_ds, val_ds=val_ds,
                                test_ds=test_ds, unlabeled_ds=unlabeled_ds,
                                temporal_unlab_ds=temporal_unlab_ds)

    # 2. GUARDIANES ANTES DE ENTRENAR. Comprueban el efecto, no el cfg.
    if cfg.get("use_semi", False):
        assert loaders.get("unlabeled_loader") is not None, (
            "FALLO: use_semi=True pero no se construyo el unlabeled_loader")
        assert len(loaders["unlabeled_loader"].dataset) > 0, (
            "FALLO: el pool sin etiquetar esta vacio")
        print(f"pool OK: {len(loaders['unlabeled_loader'].dataset)} frames sin "
              f"etiquetar de {cfg['unlabeled_subdir']}")

    _bx = next(iter(loaders["train_loader"]))["image"]
    _esperado = cfg["batch_size"] * (1 + cfg["num_augmented"])
    assert _bx.shape[0] == _esperado, (
        f"FALLO: el loader entrega {_bx.shape[0]} imagenes, no {_esperado}")
    assert tuple(_bx.shape[2:]) == tuple(cfg["target_size"]), (
        "FALLO: resolucion inesperada")

    # 3. LA HUELLA, para poder situar el run en un stack de software despues.
    pre = collect_pre(cfg, loaders, train_ds, val_ds, test_ds, unlabeled_ds)

    # 4. LA TRANSCRIPCION. Se copia a stdout.log y se sigue viendo en pantalla.
    #    Cada linea se vuelca al vuelo, asi que un corte de Colab deja el log
    #    completo hasta donde llego, con su traceback si murio.
    with tee_output(os.path.join(_exp, "stdout.log")):
        _t0 = time.time()
        art = run_training(cfg, loaders)
        results = evaluate_checkpoint(cfg, art["model"], loaders,
                                      art["best_path"], art["history"])

        # 5. GUARDIAN DESPUES DE ENTRENAR. La perdida no supervisada tiene que
        #    haberse movido. Con lambda_u=0 sigue siendo distinta de cero, asi
        #    que un cero solo puede significar que la rama no corrio.
        if cfg.get("use_semi", False):
            _mu = max([(e.get("unsup_loss") or 0.0)
                       for e in (art["history"] or [])] or [0.0])
            assert _mu > 0, (
                "FALLO: unsup_loss se quedo en cero en todas las epocas; "
                "el run NO fue semi-supervisado")

        save_fingerprint(pre, collect_post(art, results), _exp)
        print(f"[{nombre}/seed_{semilla}] {(time.time()-_t0)/60:.1f} min")
        print(results)

    return _exp
