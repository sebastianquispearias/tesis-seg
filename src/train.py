import csv
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from src.losses import build_criterion
from src.metrics import eval_imagewise_and_global, compute_boundary_metrics_epoch
from src.models import create_model
from src.utils import (
    clone_model_weights,
    count_parameters_m,
    ensure_dir,
    save_json,
    seed_everything,
    update_ema,
)


def _fmt_td(seconds: float) -> str:
    s = int(seconds)
    m, s = divmod(s, 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def _cuda_max_mem_mb() -> str:
    if torch.cuda.is_available():
        return f"{torch.cuda.max_memory_allocated() / 1e6:.0f} MB"
    return "n/a"


def _ramp_weight(epoch: int, start_epoch: int, warmup_epochs: int, max_weight: float) -> float:
    if epoch < start_epoch:
        return 0.0
    if warmup_epochs <= 0:
        return float(max_weight)
    alpha = min(1.0, (epoch - start_epoch + 1) / warmup_epochs)
    return float(max_weight) * alpha


def _temporal_consistency_loss(
    probs_t: torch.Tensor,
    probs_tp: torch.Tensor,
    tau_temp: float = 0.7,
) -> torch.Tensor:
    """
    Consistencia temporal simple:
    - solo penaliza píxeles relativamente confiables en ambos frames
    - usa MSE entre probabilidades
    """
    conf_t = (probs_t >= tau_temp) | (probs_t <= (1.0 - tau_temp))
    conf_tp = (probs_tp >= tau_temp) | (probs_tp <= (1.0 - tau_temp))
    conf = conf_t & conf_tp

    mse_all = (probs_t - probs_tp) ** 2

    if conf.any():
        return (mse_all * conf.float()).sum() / conf.float().sum()

    return torch.zeros((), device=probs_t.device)


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
    temp_loader=None,
    log_every: int = 10,
):
    cfg = cfg or {}
    model.train(train)

    total_loss = 0.0
    total_sup = 0.0
    total_unsup = 0.0
    total_temp = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n_samples = 0
    n_unsup = 0
    n_temp = 0
    total_lambda_u = 0.0
    total_lambda_t = 0.0

    total_pl_conf_all      = 0.0
    total_pl_conf_sq_all   = 0.0
    total_pl_conf_selected = 0.0
    total_pl_coverage      = 0.0
    total_pl_pos_frac      = 0.0
    n_pl_batches           = 0
    n_pl_selected_batches  = 0

    use_semi = bool(
        train and cfg.get("use_semi", False)
        and (unl_loader is not None) and (teacher is not None)
    )
    use_temp = bool(
        train and cfg.get("use_temp_consistency", False) and (temp_loader is not None)
    )

    unl_iter = iter(unl_loader) if use_semi else None
    temp_iter = iter(temp_loader) if use_temp else None

    if use_semi:
        teacher.eval()

    if train:
        print(f"\n=== Epoch {epoch}/{cfg.get('epochs', '?')} ===")

    num_iter = len(loader)
    batch_time = 0.0
    data_time = 0.0
    end = time.perf_counter()

    for it, batch in enumerate(loader):
        t0 = time.perf_counter()
        data_time += t0 - end

        xb = batch["image"].to(device, non_blocking=True).float()
        yb = batch["mask"].to(device, non_blocking=True).float()
        bs = xb.size(0)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(xb)
            sup_loss, components = criterion(logits, yb, epoch=epoch)
            loss = sup_loss

            unsup_loss = torch.zeros((), device=device)
            temp_loss = torch.zeros((), device=device)

            # =========================
            # Semi-supervisión teacher/student
            # =========================
            if use_semi:
                try:
                    ubatch = next(unl_iter)
                except StopIteration:
                    unl_iter = iter(unl_loader)
                    ubatch = next(unl_iter)

                xw_u = ubatch["weak_image"].to(device, non_blocking=True).float()
                xs_u = ubatch["strong_image"].to(device, non_blocking=True).float()
                bs_u = xs_u.size(0)

                ssl_method = cfg.get("ssl_method", "pseudo_label")

                with torch.no_grad():
                    logits_teacher = teacher(xw_u)
                    probs_teacher = torch.sigmoid(logits_teacher)

                    if ssl_method == "pseudo_label":
                        pseudo = (probs_teacher >= 0.5).float()
                        tau = cfg.get("tau", 0.95)
                        conf_mask = (probs_teacher >= tau) | (probs_teacher <= (1.0 - tau))

                        # per-pixel confidence: max(p, 1-p) — unaffected by class prevalence
                        conf_pixel = torch.maximum(probs_teacher, 1.0 - probs_teacher)
                        total_pl_conf_all      += float(conf_pixel.mean())
                        total_pl_conf_sq_all   += float((conf_pixel ** 2).mean())
                        total_pl_coverage      += float(conf_mask.float().mean())
                        total_pl_pos_frac      += float(pseudo.mean())
                        n_pl_batches           += 1
                        if conf_mask.any():
                            total_pl_conf_selected += float(conf_pixel[conf_mask].mean())
                            n_pl_selected_batches  += 1

                logits_u = model(xs_u)
                if ssl_method == "mean_teacher":
                    probs_student = torch.sigmoid(logits_u)
                    unsup_loss = F.mse_loss(probs_student, probs_teacher.detach())
                else:  # pseudo_label (default)
                    unsup_all = F.binary_cross_entropy_with_logits(logits_u, pseudo, reduction="none")
                    if conf_mask.any():
                        unsup_loss = (unsup_all * conf_mask.float()).sum() / conf_mask.float().sum()

                lambda_u_t = _ramp_weight(
                    epoch=epoch,
                    start_epoch=cfg.get("semi_start_epoch", 0),
                    warmup_epochs=cfg.get("semi_warmup_epochs", 0),
                    max_weight=cfg.get("lambda_u", 0.0),
                )
                loss = loss + lambda_u_t * unsup_loss
                components["unsup"] = float(unsup_loss.detach())
                components["lambda_u_t"] = float(lambda_u_t)

                total_unsup += float(unsup_loss.detach()) * bs_u
                total_lambda_u += float(lambda_u_t) * bs_u
                n_unsup += bs_u

            # =========================
            # Consistencia temporal
            # =========================
            if use_temp:
                try:
                    tbatch = next(temp_iter)
                except StopIteration:
                    temp_iter = iter(temp_loader)
                    tbatch = next(temp_iter)

                xt = tbatch["image_t"].to(device, non_blocking=True).float()
                xtp = tbatch["image_tp"].to(device, non_blocking=True).float()
                bs_t = xt.size(0)

                logits_t = model(xt)
                logits_tp = model(xtp)
                probs_t = torch.sigmoid(logits_t)
                probs_tp = torch.sigmoid(logits_tp)

                temp_loss = _temporal_consistency_loss(
                    probs_t,
                    probs_tp,
                    tau_temp=cfg.get("tau_temp", 0.7),
                )

                lambda_t_t = _ramp_weight(
                    epoch=epoch,
                    start_epoch=cfg.get("temp_start_epoch", 0),
                    warmup_epochs=cfg.get("temp_warmup_epochs", 0),
                    max_weight=cfg.get("lambda_t", 0.0),
                )
                loss = loss + lambda_t_t * temp_loss
                components["temp"] = float(temp_loss.detach())
                components["lambda_t_t"] = float(lambda_t_t)

                total_temp += float(temp_loss.detach()) * bs_t
                total_lambda_t += float(lambda_t_t) * bs_t
                n_temp += bs_t

            if train:
                loss.backward()
                optimizer.step()

                if use_semi:
                    update_ema(model, teacher, ema_decay=cfg.get("ema_decay", 0.99))

        # Inline dice / IoU from supervised predictions
        with torch.no_grad():
            probs_s = torch.sigmoid(logits)
            preds_s = (probs_s >= cfg.get("eval_threshold", 0.5)).float()
            inter = (preds_s * yb).sum(dim=(1, 2, 3))
            union = preds_s.sum(dim=(1, 2, 3)) + yb.sum(dim=(1, 2, 3)) - inter
            dice_b = (2 * inter + 1e-7) / (preds_s.sum(dim=(1, 2, 3)) + yb.sum(dim=(1, 2, 3)) + 1e-7)
            iou_b = (inter + 1e-7) / (union + 1e-7)
            total_dice += float(dice_b.mean()) * bs
            total_iou += float(iou_b.mean()) * bs

        total_loss += float(loss.detach()) * bs
        total_sup += float(sup_loss.detach()) * bs
        n_samples += bs

        t1 = time.perf_counter()
        batch_time += t1 - t0
        it_done = it + 1

        if train and (it_done % log_every == 0 or it_done == num_iter):
            sec_per_it = batch_time / max(it_done, 1)
            eta = sec_per_it * (num_iter - it_done)
            lr_cur = optimizer.param_groups[0].get("lr", 0.0)
            avg_so_far = total_loss / max(1, n_samples)
            print(
                f"Epoch: [{epoch:02d}]\t[{it_done:3d}/{num_iter:3d}]"
                f"\teta: {_fmt_td(eta)}"
                f"\tloss: {float(loss.detach()):.4f} ({avg_so_far:.4f})"
                f"\tlr: {lr_cur:.6f}"
                f"\ttime: {batch_time/it_done:.4f}"
                f"\tdata: {data_time/it_done:.4f}"
                f"\tmax mem: {_cuda_max_mem_mb()}"
            )

        end = time.perf_counter()

    avg_loss = total_loss / max(n_samples, 1)
    avg_sup = total_sup / max(n_samples, 1)
    avg_dice = total_dice / max(n_samples, 1)
    avg_iou = total_iou / max(n_samples, 1)
    avg_unsup = total_unsup / max(n_unsup, 1) if n_unsup > 0 else 0.0
    avg_lambda_u = total_lambda_u / max(n_unsup, 1) if n_unsup > 0 else 0.0
    avg_temp = total_temp / max(n_temp, 1) if n_temp > 0 else 0.0
    avg_lambda_t = total_lambda_t / max(n_temp, 1) if n_temp > 0 else 0.0

    if n_pl_batches > 0:
        pl_conf_mean_all = total_pl_conf_all / n_pl_batches
        pl_conf_std_all  = max(0.0, total_pl_conf_sq_all / n_pl_batches - pl_conf_mean_all ** 2) ** 0.5
        pl_conf_coverage = total_pl_coverage / n_pl_batches
        pl_pos_frac      = total_pl_pos_frac / n_pl_batches
    else:
        pl_conf_mean_all = pl_conf_std_all = pl_conf_coverage = pl_pos_frac = 0.0
    pl_conf_mean_selected = (
        total_pl_conf_selected / n_pl_selected_batches if n_pl_selected_batches > 0 else 0.0
    )

    if train:
        lr_cur = optimizer.param_groups[0].get("lr", 0.0)
        print(
            f"Epoch: [{epoch:02d}] Total time: {_fmt_td(batch_time)} "
            f"({batch_time/max(num_iter,1):.4f} s/it)"
        )
        print(
            f"Averaged stats: loss: {avg_loss:.4f}\t"
            f"dice: {avg_dice:.4f}\tiou: {avg_iou:.4f}\t"
            f"lr: {lr_cur:.6f}"
        )
        print(
            f"semi: unsup={avg_unsup:.6f}  lambda_u_t={avg_lambda_u:.4f}  "
            f"temp={avg_temp:.6f}  lambda_t_t={avg_lambda_t:.4f}"
        )
        print(
            f"USE_SEMI: {cfg.get('use_semi', False)}"
            f"  has_unl: {unl_loader is not None}"
            f"  has_teacher: {teacher is not None}"
            f"  use_semi: {use_semi}"
        )

    return {
        "loss": avg_loss,
        "sup_loss": avg_sup,
        "unsup_loss": avg_unsup,
        "temp_loss": avg_temp,
        "train_dice": avg_dice,
        "train_iou": avg_iou,
        "lambda_u_t": avg_lambda_u,
        "lambda_t_t": avg_lambda_t,
        "pl_conf_mean_all": pl_conf_mean_all,
        "pl_conf_std_all": pl_conf_std_all,
        "pl_conf_mean_selected": pl_conf_mean_selected,
        "pl_conf_coverage": pl_conf_coverage,
        "pl_pos_frac": pl_pos_frac,
    }


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
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 1e-4),
    )

    warmup_epochs = int(cfg.get("warmup_epochs", 10))
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(
                optimizer,
                start_factor=1e-3,
                end_factor=1.0,
                total_iters=warmup_epochs,
            ),
            CosineAnnealingLR(
                optimizer,
                T_max=max(1, cfg["epochs"] - warmup_epochs),
                eta_min=0.0,
            ),
        ],
        milestones=[warmup_epochs],
    )

    teacher = None
    if cfg.get("use_semi", False):
        teacher = create_model(cfg["arch"], cfg["backbone"], cfg["n_classes"]).to(device)
        clone_model_weights(model, teacher)
        teacher.eval()

    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]
    unlabeled_loader = loaders.get("unlabeled_loader", None)
    temporal_unlab_loader = loaders.get("temporal_unlab_loader", None)

    tol_px = int(cfg.get("boundary_tol_px", 5))

    # ── CSV headers ────────────────────────────────────────────────────
    train_log_csv = os.path.join(exp_dir, "train_log.csv")
    with open(train_log_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_dice", "train_iou", "val_loss"])

    bf_log_csv = os.path.join(exp_dir, "val_boundary_log.csv")
    with open(bf_log_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "bf1_tol", "assd", "hd95"])

    detailed_log_csv = os.path.join(exp_dir, "train_log_detailed.csv")
    with open(detailed_log_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "epoch",
            "train_loss", "train_bce_loss", "train_dice_loss",
            "train_morph_loss", "train_boundary_loss", "train_hd_loss", "train_lambda_value",
            "train_dice_metric", "train_iou_metric",
            "val_loss", "val_bce_loss", "val_dice_loss",
            "val_morph_loss", "val_boundary_loss", "val_hd_loss", "val_lambda_value",
            "train_morph_raw", "train_boundary_raw", "train_hd_raw",
            "val_morph_raw", "val_boundary_raw", "val_hd_raw",
            "train_unsup_loss", "train_lambda_u_t",
            "train_temp_loss", "train_lambda_t_t",
        ])
    # ──────────────────────────────────────────────────────────────────

    diag_csv = os.path.join(exp_dir, "diagnostics_epoch.csv")
    with open(diag_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "epoch",
            "train_loss", "train_dice", "train_iou",
            "val_loss", "val_iou_global", "val_iou_sample", "val_f1_sample",
            "val_f1_std", "val_iou_std",
            "bf1_val", "assd_val", "hd95_val",
            "unsup_loss", "lambda_u_t",
            "temp_loss", "lambda_t_t",
            "pl_conf_mean_all", "pl_conf_std_all", "pl_conf_mean_selected",
            "pl_conf_coverage", "pl_pos_frac",
            "elapsed_sec",
        ])

    history = []
    best_ckpt_score = -float("inf")
    best_val_loss_ref = float("inf")
    best_path = os.path.join(exp_dir, "best_model.pt")
    patience_es = int(cfg.get("patience_es", 20))
    epochs_without_improve = 0

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

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
            temp_loader=temporal_unlab_loader,
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
            temp_loader=None,
        )

        val_eval = eval_imagewise_and_global(
            model, val_loader, device=device,
            thr=cfg["eval_threshold"], logits=True,
            split_name=f"VAL@{epoch:03d}",
        )

        bf1_mean, assd_mean, hd95_mean = compute_boundary_metrics_epoch(
            model, val_loader, device=device,
            thr=cfg["eval_threshold"], r_tol_px=tol_px,
        )
        print(f"[VAL boundary] BF1@{tol_px}px={bf1_mean:.4f} | ASSD={assd_mean:.3f} | HD95={hd95_mean:.3f}")

        val_iou_global = float(val_eval["global_iou"])
        bf1_for_score = float(bf1_mean) if math.isfinite(bf1_mean) else 0.0
        ckpt_score = val_iou_global
        print(f"[VAL ckpt-score] IoU_global={val_iou_global:.4f} | BF1={bf1_for_score:.4f} | score=1*IoU={ckpt_score:.6f}")

        if ckpt_score > best_ckpt_score + 1e-6:
            best_ckpt_score = ckpt_score
            best_val_loss_ref = min(best_val_loss_ref, val_stats["loss"])
            torch.save(model.state_dict(), best_path)
            print(f"  ↳ nuevo mejor checkpoint (score={best_ckpt_score:.6f}, val_loss={val_stats['loss']:.6f}) guardado en {best_path}")
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        scheduler.step()

        dt = time.time() - t0
        print(
            f"[{epoch:03d}] {dt:.1f}s | train: loss={train_stats['loss']:.4f} "
            f"dice={train_stats['train_dice']:.4f} iou={train_stats['train_iou']:.4f} | "
            f"val_loss={val_stats['loss']:.4f}"
        )

        # ── Append to CSVs ─────────────────────────────────────────────
        with open(train_log_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch,
                train_stats["loss"],
                train_stats["train_dice"],
                train_stats["train_iou"],
                val_stats["loss"],
            ])

        with open(bf_log_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, bf1_mean, assd_mean, hd95_mean])

        with open(detailed_log_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch,
                train_stats["loss"], 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                train_stats["train_dice"], train_stats["train_iou"],
                val_stats["loss"], 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                train_stats["unsup_loss"], train_stats["lambda_u_t"],
                train_stats["temp_loss"], train_stats["lambda_t_t"],
            ])
        with open(diag_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch,
                train_stats["loss"], train_stats["train_dice"], train_stats["train_iou"],
                val_stats["loss"], val_iou_global,
                val_eval["sample_mean_iou"], val_eval["sample_mean_f1"],
                val_eval["f1_std"], val_eval["iou_std"],
                bf1_mean, assd_mean, hd95_mean,
                train_stats["unsup_loss"], train_stats["lambda_u_t"],
                train_stats["temp_loss"], train_stats["lambda_t_t"],
                train_stats["pl_conf_mean_all"], train_stats["pl_conf_std_all"],
                train_stats["pl_conf_mean_selected"],
                train_stats["pl_conf_coverage"], train_stats["pl_pos_frac"],
                dt,
            ])
        # ──────────────────────────────────────────────────────────────

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_dice": train_stats["train_dice"],
            "train_iou": train_stats["train_iou"],
            "val_loss": val_stats["loss"],
            "val_f1": val_eval["sample_mean_f1"],
            "val_iou": val_eval["sample_mean_iou"],
            "val_iou_global": val_iou_global,
            "bf1_val": bf1_mean,
            "elapsed_sec": dt,
        }
        history.append(row)

        if epochs_without_improve >= patience_es:
            print(f">> Early stopping activado en epoch {epoch} (patience={patience_es})")
            break

    print("Mejor score de checkpoint:", best_ckpt_score, "| ckpt:", best_path)
    print("Mejor val_loss observado (solo referencia):", best_val_loss_ref)

    if history:
        best_row_sum = max(history, key=lambda r: r["val_iou_global"])
        save_json({
            "exp_dir": exp_dir,
            "seed": cfg.get("seed"),
            "use_semi": cfg.get("use_semi", False),
            "tau": cfg.get("tau"),
            "semi_start_epoch": cfg.get("semi_start_epoch"),
            "semi_warmup_epochs": cfg.get("semi_warmup_epochs"),
            "lambda_u": cfg.get("lambda_u"),
            "best_epoch": best_row_sum["epoch"],
            "best_val_iou_global": best_row_sum["val_iou_global"],
            "total_epochs_run": history[-1]["epoch"],
            "early_stopped": epochs_without_improve >= patience_es,
        }, os.path.join(exp_dir, "diagnostic_summary.json"))

    return {
        "model": model,
        "teacher": teacher,
        "history": history,
        "best_path": best_path,
        "exp_dir": exp_dir,
    }
