import csv
import os
import time

import torch
import torch.nn.functional as F

from src.losses import build_criterion
from src.metrics import eval_imagewise_and_global
from src.models import create_model
from src.utils import clone_model_weights, count_parameters_m, ensure_dir, save_json, seed_everything, update_ema


def run_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    train: bool = True,
    epoch: int = 0,
    cfg: dict | None = None,
    teacher=None,
    unl_loader=None,
):
    cfg = cfg or {}
    model.train(train)

    total_loss = 0.0
    n_samples = 0

    use_semi = bool(train and cfg.get("use_semi", False) and (unl_loader is not None) and (teacher is not None))
    unl_iter = iter(unl_loader) if use_semi else None
    if use_semi:
        teacher.eval()

    for batch in loader:
        xb = batch["image"].to(device, non_blocking=True).float()
        yb = batch["mask"].to(device, non_blocking=True).float()
        bs = xb.size(0)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(xb)
            sup_loss, components = criterion(logits, yb, epoch=epoch)
            loss = sup_loss

            if use_semi:
                try:
                    ubatch = next(unl_iter)
                except StopIteration:
                    unl_iter = iter(unl_loader)
                    ubatch = next(unl_iter)

                xw_u = ubatch["weak_image"].to(device, non_blocking=True).float()
                xs_u = ubatch["strong_image"].to(device, non_blocking=True).float()

                with torch.no_grad():
                    logits_teacher = teacher(xw_u)
                    probs_teacher = torch.sigmoid(logits_teacher)

                    pseudo = (probs_teacher >= 0.5).float()
                    tau = cfg.get("tau", 0.95)
                    conf_mask = (probs_teacher >= tau) | (probs_teacher <= (1.0 - tau))

                logits_u = model(xs_u)
                unsup_all = F.binary_cross_entropy_with_logits(logits_u, pseudo, reduction="none")

                if conf_mask.any():
                    unsup_loss = (unsup_all * conf_mask.float()).sum() / conf_mask.float().sum()
                else:
                    unsup_loss = torch.zeros((), device=device)

                lambda_u = cfg.get("lambda_u", 0.0)
                semi_start = cfg.get("semi_start_epoch", 0)
                semi_warmup = cfg.get("semi_warmup_epochs", 0)

                lambda_u_t = 0.0
                if epoch >= semi_start:
                    if semi_warmup > 0:
                        w = min(1.0, (epoch - semi_start + 1) / semi_warmup)
                        lambda_u_t = lambda_u * w
                    else:
                        lambda_u_t = lambda_u

                loss = loss + lambda_u_t * unsup_loss
                components["unsup"] = float(unsup_loss.detach())
                components["lambda_u_t"] = float(lambda_u_t)

            if train:
                loss.backward()
                optimizer.step()

                if use_semi:
                    update_ema(model, teacher, ema_decay=cfg.get("ema_decay", 0.99))

        total_loss += float(loss.detach()) * bs
        n_samples += bs

    avg_loss = total_loss / max(n_samples, 1)
    return {"loss": avg_loss}


def save_history_csv(history: list[dict], out_csv: str):
    ensure_dir(os.path.dirname(out_csv))
    if not history:
        return

    fieldnames = list(history[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(history)


def run_training(cfg: dict, loaders: dict):
    seed_everything(cfg.get("seed", 42))

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = cfg["exp_dir"]
    ensure_dir(exp_dir)
    ensure_dir(os.path.join(exp_dir, "preds_vis"))
    ensure_dir(os.path.join(exp_dir, "test_preds"))

    save_json(cfg, os.path.join(exp_dir, "config.json"))

    model = create_model(cfg["arch"], cfg["backbone"], cfg["n_classes"]).to(device)
    print("Modelo:", cfg["arch"], cfg["backbone"], "| params ≈", count_parameters_m(model), "M")

    criterion = build_criterion(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 1e-4),
    )

    teacher = None
    if cfg.get("use_semi", False):
        teacher = create_model(cfg["arch"], cfg["backbone"], cfg["n_classes"]).to(device)
        clone_model_weights(model, teacher)
        teacher.eval()

    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]
    unlabeled_loader = loaders.get("unlabeled_loader", None)

    history = []
    best_val_loss = float("inf")
    best_path = os.path.join(exp_dir, "best_model.pt")

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        train_stats = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=True,
            epoch=epoch,
            cfg=cfg,
            teacher=teacher,
            unl_loader=unlabeled_loader,
        )

        val_stats = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=False,
            epoch=epoch,
            cfg=cfg,
            teacher=None,
            unl_loader=None,
        )

        train_eval = eval_imagewise_and_global(
            model, train_loader, device=device, thr=cfg["eval_threshold"], logits=True, split_name="TRAIN"
        )
        val_eval = eval_imagewise_and_global(
            model, val_loader, device=device, thr=cfg["eval_threshold"], logits=True, split_name="VAL"
        )

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "train_f1": train_eval["f1_mean"],
            "train_iou": train_eval["iou_mean"],
            "val_f1": val_eval["f1_mean"],
            "val_iou": val_eval["iou_mean"],
            "elapsed_sec": time.time() - t0,
        }
        history.append(row)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={row['train_loss']:.6f} | val_loss={row['val_loss']:.6f} | "
            f"train_f1={row['train_f1']:.4f} | val_f1={row['val_f1']:.4f}"
        )

        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            torch.save(model.state_dict(), best_path)
            print(">> Mejor checkpoint guardado en:", best_path)

        save_history_csv(history, os.path.join(exp_dir, "train_log.csv"))

    return {
        "model": model,
        "teacher": teacher,
        "history": history,
        "best_path": best_path,
        "exp_dir": exp_dir,
    }