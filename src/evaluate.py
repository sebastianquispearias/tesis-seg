import csv
import json
import math
import os

import cv2
import numpy as np
import torch

from src.metrics import (
    boundary_metrics_per_image_to_csv,
    compute_boundary_metrics_epoch,
    eval_imagewise_and_global,
)

# ruler_eval and visualization import matplotlib at module load, which breaks
# any environment where matplotlib is unavailable or incompatible. Those
# functions are only needed inside evaluate_checkpoint(); write_run_report()
# and friends do not touch them. Import lazily so refresh_run_reports.py and
# any headless consumer of write_run_report() can run without matplotlib.


def write_simple_metrics_csv(path: str, metrics: dict):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, v])


def write_run_summary(cfg: dict, history: list[dict], exp_dir: str, test_metrics: dict | None = None):
    summary_path = os.path.join(exp_dir, "run_summary.txt")
    best_row = None

    if history:
        best_row = min(history, key=lambda x: x["val_loss"])

    train_csv = os.path.join(exp_dir, "train_log.csv")
    test_csv = os.path.join(exp_dir, "test_metrics.csv")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== Vertebra Segmentation — Run Summary ===\n")
        f.write(f"EXP_DIR: {exp_dir}\n")
        f.write(f"BEST_PATH: {os.path.join(exp_dir, 'best_model.pt')}\n\n")

        f.write("[Config]\n")
        for k, v in cfg.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n")

        if best_row is not None:
            f.write("[Training]\n")
            f.write(f"- Best val_loss: {best_row['val_loss']:.6f} @ epoch {best_row['epoch']}\n")
            f.write(f"- train_loss: {best_row.get('train_loss', float('nan')):.6f} | ")
            f.write(f"train_dice: {best_row.get('train_dice', float('nan')):.4f} | ")
            f.write(f"train_iou: {best_row.get('train_iou', float('nan')):.4f}\n\n")

        if test_metrics is not None:
            f.write("[TEST metrics]\n")
            f.write(f"- Sample-wise F1:  {test_metrics['f1_mean']:.6f} ± {test_metrics['f1_std']:.6f}\n")
            f.write(f"- Sample-wise IoU: {test_metrics['iou_mean']:.6f} ± {test_metrics['iou_std']:.6f}\n")
            f.write(f"- Global F1:       {test_metrics['f1_global']:.6f}\n")
            f.write(f"- Global IoU:      {test_metrics['iou_global']:.6f}\n\n")

        preds_vis = os.path.join(exp_dir, "preds_vis")
        preds_bin = os.path.join(exp_dir, "test_preds")
        f.write("[Artefactos]\n")
        f.write(f"- preds_vis/:  {preds_vis}\n")
        f.write(f"- test_preds/: {preds_bin}\n")
        f.write(f"- train_log.csv: {train_csv}\n")
        f.write(f"- test_metrics.csv: {test_csv}\n")

    print("Resumen guardado en:", summary_path)
    return summary_path


def _read_boundary_agg(csv_path: str) -> dict | None:
    """Aggregate per-image boundary CSV (bf1, assd, hd95) into mean±std dict."""
    if not os.path.isfile(csv_path):
        return None
    try:
        bf1s, assds, hd95s = [], [], []
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                bf1s.append(float(row["bf1"]))
                a, h = float(row["assd"]), float(row["hd95"])
                if math.isfinite(a):
                    assds.append(a)
                if math.isfinite(h):
                    hd95s.append(h)
        if not bf1s:
            return None

        def _ms(lst):
            if not lst:
                return None, None
            m = sum(lst) / len(lst)
            s = math.sqrt(sum((x - m) ** 2 for x in lst) / len(lst))
            return round(m, 4), round(s, 4)

        bf1_m, bf1_s = _ms(bf1s)
        assd_m, assd_s = _ms(assds)
        hd95_m, hd95_s = _ms(hd95s)
        return {
            "n_images": len(bf1s),
            "bf1_mean": bf1_m, "bf1_std": bf1_s,
            "assd_mean_px": assd_m, "assd_std_px": assd_s,
            "hd95_mean_px": hd95_m, "hd95_std_px": hd95_s,
        }
    except Exception:
        return None


def _read_best_epoch_stats(exp_dir: str) -> tuple[dict | None, dict | None]:
    """
    Read diagnostics_epoch.csv, find the row with the highest val_iou_global,
    and return (pl_stats_at_best_epoch, val_boundary_at_best_epoch).
    Both are None if the file is absent, unreadable, or columns are missing.
    """
    path = os.path.join(exp_dir, "diagnostics_epoch.csv")
    if not os.path.isfile(path):
        return None, None
    try:
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None, None
        best = max(rows, key=lambda r: float(r.get("val_iou_global", 0)))

        pl_keys = ["pl_conf_mean_all", "pl_conf_std_all",
                   "pl_conf_mean_selected", "pl_conf_coverage", "pl_pos_frac"]
        pl_stats = (
            {k: float(best[k]) for k in pl_keys}
            if all(k in best for k in pl_keys) else None
        )

        bnd_keys = ["bf1_val", "assd_val", "hd95_val"]
        bnd_stats = (
            {k: float(best[k]) for k in bnd_keys if k in best}
            if any(k in best for k in bnd_keys) else None
        )
        return pl_stats, bnd_stats
    except Exception:
        return None, None


def _safe_float(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def _safe_int(v):
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def _read_epoch_history(exp_dir: str, use_semi: bool | None) -> list[dict]:
    """
    Read diagnostics_epoch.csv into a list of dicts (one per epoch).
    Numeric cols cast to float/int, NaN -> None. Adds an `ssl_active` field
    per row:
      - None if use_semi is False (SSL doesn't apply to this run at all)
      - (lambda_u_t > 0) if use_semi is True and lambda_u_t is a finite number
      - None otherwise (e.g., lambda_u_t missing or NaN)
    Returns [] if the file is absent or unreadable.
    """
    path = os.path.join(exp_dir, "diagnostics_epoch.csv")
    if not os.path.isfile(path):
        return []
    int_cols = {"epoch"}
    try:
        out: list[dict] = []
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry: dict = {}
                for k, v in row.items():
                    if k is None:
                        continue
                    if k in int_cols:
                        entry[k] = _safe_int(v)
                    else:
                        entry[k] = _safe_float(v)

                lam = entry.get("lambda_u_t")
                if use_semi is False:
                    entry["ssl_active"] = None
                elif use_semi is True and lam is not None:
                    entry["ssl_active"] = bool(lam > 0)
                else:
                    entry["ssl_active"] = None
                out.append(entry)
        return out
    except Exception:
        return []


def _read_debug_fingerprint(exp_dir: str) -> dict:
    """
    Read debug_fingerprint.json and lift a compact subset of the pre_run
    block into a standalone dict for inclusion in run_report.json.
    Never recollects anything — only reads what the notebook helper already wrote.
    Missing file or unreadable JSON -> {"has_debug_fingerprint": False}.
    """
    path = os.path.join(exp_dir, "debug_fingerprint.json")
    if not os.path.isfile(path):
        return {"has_debug_fingerprint": False}
    try:
        with open(path, encoding="utf-8") as f:
            fp = json.load(f)
    except Exception:
        return {"has_debug_fingerprint": False}

    pre = (fp or {}).get("pre_run") or {}
    env = pre.get("environment") or {}
    prov = pre.get("provenance") or {}
    df = pre.get("dataset_facts") or {}
    mf = pre.get("model_fingerprint") or {}

    def _g(d, k):
        return d.get(k) if isinstance(d, dict) else None

    return {
        "git_hash": _g(prov, "git_hash"),
        "timestamp_utc": pre.get("timestamp_utc"),
        "python": _g(env, "python"),
        "torch": _g(env, "torch"),
        "torchvision": _g(env, "torchvision"),
        "cuda": _g(env, "cuda"),
        "cudnn": _g(env, "cudnn"),
        "albumentations": _g(env, "albumentations"),
        "segmentation_models_pytorch": _g(env, "segmentation_models_pytorch"),
        "platform": _g(env, "platform"),
        "gpu_name": _g(env, "gpu_name"),
        "model_total_params": _g(mf, "total_params"),
        "model_trainable_params": _g(mf, "trainable_params"),
        "dataset_facts": {
            "len_train_ds": _g(df, "len_train_ds"),
            "len_val_ds": _g(df, "len_val_ds"),
            "len_test_ds": _g(df, "len_test_ds"),
            "len_unlabeled_ds": _g(df, "len_unlabeled_ds"),
            "len_temporal_unlab_ds": _g(df, "len_temporal_unlab_ds"),
            "batch_size": _g(df, "batch_size"),
            "batch_size_unlab": _g(df, "batch_size_unlab"),
        },
        "has_debug_fingerprint": True,
    }


def _compute_ssl_interpretation(
    cfg: dict,
    training_summary: dict | None,
    epoch_history: list[dict],
) -> dict:
    """
    Derived interpretation of the SSL schedule for this run. Never reads files.
    Fields are None where undefined (e.g., no training_summary.best_epoch).
    """
    use_semi = bool(cfg.get("use_semi", False))
    semi_start = cfg.get("semi_start_epoch")

    best_epoch = None
    total_epochs = None
    early_stopped = None
    if isinstance(training_summary, dict):
        best_epoch = training_summary.get("best_epoch")
        total_epochs = training_summary.get("total_epochs_run")
        early_stopped = training_summary.get("early_stopped")

    def _num(x):
        return x if isinstance(x, (int, float)) else None

    best_e = _num(best_epoch)
    total_e = _num(total_epochs)
    semi_s = _num(semi_start)

    if best_e is not None and semi_s is not None:
        best_ge = bool(best_e >= semi_s)
        best_lt = bool(best_e < semi_s)
        best_minus_semi = int(best_e - semi_s)
        best_post_ssl = best_ge if use_semi else None
    else:
        best_ge = None
        best_lt = None
        best_minus_semi = None
        best_post_ssl = None

    # --- Two distinct epoch counts, intentionally kept separate ---
    #
    # n_epochs_at_or_after_semi_start (schedule-based):
    #   max(0, total_epochs_run - semi_start_epoch). Arithmetic over the
    #   schedule only; does NOT look at lambda_u_t. Example: total=62,
    #   semi_start=30 -> 32.
    #
    # n_epochs_ssl_active (activity-based):
    #   count of rows in epoch_history where ssl_active is True, i.e. where
    #   lambda_u_t > 0. Example: same run, epochs 30..62 inclusive all have
    #   lambda_u_t > 0 -> 33 rows.
    #
    # The off-by-one (32 vs 33) is expected and correct: the schedule-based
    # count is an exclusive difference (epochs strictly after the threshold),
    # while the activity-based count includes the boundary epoch itself
    # (epoch == semi_start_epoch, where lambda_u_t first becomes > 0).
    # Both values are useful:
    #   - schedule-based answers "how long past the threshold did we train?"
    #   - activity-based answers "on how many epochs was the SSL branch
    #     actually contributing to the loss?"
    # For analysis, prefer n_epochs_ssl_active — it is the ground truth.
    if total_e is not None and semi_s is not None:
        n_at_or_after_semi = max(0, int(total_e - semi_s))
        had_opportunity = bool(total_e > semi_s)
    else:
        n_at_or_after_semi = None
        had_opportunity = None

    if total_e is not None and best_e is not None:
        n_after_best = max(0, int(total_e - best_e))
    else:
        n_after_best = None

    if not use_semi:
        n_ssl_active = None
    elif epoch_history:
        any_known = any(e.get("ssl_active") is not None for e in epoch_history)
        if any_known:
            n_ssl_active = sum(1 for e in epoch_history if e.get("ssl_active") is True)
        else:
            n_ssl_active = None
    else:
        n_ssl_active = None

    return {
        "use_semi": use_semi,
        "ssl_method": cfg.get("ssl_method") if use_semi else None,
        "semi_start_epoch": semi_start,
        "semi_warmup_epochs": cfg.get("semi_warmup_epochs"),
        "lambda_u": cfg.get("lambda_u"),
        "best_epoch": best_epoch,
        "total_epochs_run": total_epochs,
        "best_epoch_geq_semi_start": best_ge,
        "best_epoch_lt_semi_start": best_lt,
        "best_checkpoint_is_post_ssl": best_post_ssl,
        "best_epoch_minus_semi_start": best_minus_semi,
        "n_epochs_at_or_after_semi_start": n_at_or_after_semi,
        "n_epochs_ssl_active": n_ssl_active,
        "n_epochs_after_best_epoch": n_after_best,
        "ssl_had_opportunity_to_improve": had_opportunity,
        "early_stopped": early_stopped,
    }


def _compute_training_diagnostics_summary(
    epoch_history: list[dict],
    cfg: dict,
) -> dict:
    """
    Derived summary of training dynamics, computed purely from epoch_history.
    Returns a minimal stub if epoch_history is empty.
    """
    if not epoch_history:
        return {"n_epochs_logged": 0, "has_epoch_history": False}

    def _f(row, key):
        v = row.get(key)
        return v if isinstance(v, (int, float)) else None

    epochs = [_f(r, "epoch") for r in epoch_history]
    iou_g = [_f(r, "val_iou_global") for r in epoch_history]
    val_losses = [_f(r, "val_loss") for r in epoch_history]
    elapsed = [_f(r, "elapsed_sec") for r in epoch_history]

    valid_iou_pairs = [(e, v) for e, v in zip(epochs, iou_g) if e is not None and v is not None]
    if valid_iou_pairs:
        best_e, best_iou = max(valid_iou_pairs, key=lambda p: p[1])
    else:
        best_e, best_iou = None, None

    valid_vl = [v for v in val_losses if v is not None]
    best_val_loss = min(valid_vl) if valid_vl else None

    final_iou = next((v for v in reversed(iou_g) if v is not None), None)
    final_vl = next((v for v in reversed(val_losses) if v is not None), None)

    first_epoch = next((e for e in epochs if e is not None), None)
    last_epoch = next((e for e in reversed(epochs) if e is not None), None)

    semi_start = cfg.get("semi_start_epoch")
    if isinstance(semi_start, (int, float)):
        before = sum(1 for e in epochs if e is not None and e < semi_start)
        at_or_after = sum(1 for e in epochs if e is not None and e >= semi_start)
        iou_at_semi = None
        for r in epoch_history:
            e = r.get("epoch")
            v = r.get("val_iou_global")
            if (
                isinstance(e, (int, float))
                and e >= semi_start
                and isinstance(v, (int, float))
            ):
                iou_at_semi = v
                break
    else:
        before = None
        at_or_after = None
        iou_at_semi = None

    if best_iou is not None and final_iou is not None:
        delta = round(best_iou - final_iou, 6)
    else:
        delta = None

    valid_elapsed = [v for v in elapsed if v is not None]
    total_elapsed = round(sum(valid_elapsed), 3) if valid_elapsed else None
    mean_elapsed = round(sum(valid_elapsed) / len(valid_elapsed), 3) if valid_elapsed else None

    return {
        "n_epochs_logged": len(epoch_history),
        "first_epoch": first_epoch,
        "last_epoch": last_epoch,
        "best_epoch_by_val_iou_global": best_e,
        "best_val_iou_global": best_iou,
        "best_val_loss": best_val_loss,
        "final_val_iou_global": final_iou,
        "final_val_loss": final_vl,
        "val_iou_global_at_semi_start": iou_at_semi,
        "val_iou_global_delta_best_minus_final": delta,
        "epochs_before_semi_start": before,
        "epochs_at_or_after_semi_start": at_or_after,
        "total_elapsed_sec": total_elapsed,
        "mean_epoch_sec": mean_elapsed,
        "has_epoch_history": True,
    }


def _compute_artifact_status(exp_dir: str, val_metrics: dict | None) -> dict:
    """
    Existence-check layer for expected run artifacts. Complements
    `file_references` (which only stores string paths).
    """
    def _file(name):
        return os.path.isfile(os.path.join(exp_dir, name))

    def _dir(name):
        p = os.path.join(exp_dir, name)
        try:
            return os.path.isdir(p) and len(os.listdir(p)) > 0
        except Exception:
            return False

    has_test_metrics = _file("test_metrics.csv")
    has_epoch_history = _file("diagnostics_epoch.csv")
    has_debug_fp = _file("debug_fingerprint.json")

    return {
        "best_model.pt": _file("best_model.pt"),
        "config.json": _file("config.json"),
        "diagnostics_epoch.csv": has_epoch_history,
        "diagnostic_summary.json": _file("diagnostic_summary.json"),
        "train_log.csv": _file("train_log.csv"),
        "train_log_detailed.csv": _file("train_log_detailed.csv"),
        "val_boundary_log.csv": _file("val_boundary_log.csv"),
        "test_metrics.csv": has_test_metrics,
        "test_boundary_metrics.csv": _file("test_boundary_metrics.csv"),
        "val_boundary_metrics.csv": _file("val_boundary_metrics.csv"),
        "c2c4_comparison.csv": _file("c2c4_comparison.csv"),
        "debug_fingerprint.json": has_debug_fp,
        "test_preds": _dir("test_preds"),
        "test_probs": _dir("test_probs"),
        "preds_vis": _dir("preds_vis"),
        "has_test_metrics": has_test_metrics,
        "has_val_metrics": val_metrics is not None,
        "has_epoch_history": has_epoch_history,
        "has_debug_fingerprint": has_debug_fp,
    }


def write_run_report(
    cfg: dict,
    test_metrics: dict,
    exp_dir: str,
    history: list | None = None,
    val_metrics: dict | None = None,
) -> str:
    """
    Writes a single portable JSON report for one completed run.
    Aggregates: config, training summary, test metrics, C2-C4 summary,
    plus five derived/read-from-disk sections (ssl_interpretation,
    epoch_history, training_diagnostics_summary, reproducibility_fingerprint,
    artifact_status).
    Always succeeds even if optional artifacts are absent.
    """
    # --- Run identity ---
    exp_name = cfg.get("experiment_name", os.path.basename(os.path.dirname(exp_dir)))
    seed = cfg.get("seed")

    # --- Training summary (prefer diagnostic_summary.json, fallback to history) ---
    training_summary = None
    diag_path = os.path.join(exp_dir, "diagnostic_summary.json")
    if os.path.isfile(diag_path):
        try:
            with open(diag_path, encoding="utf-8") as f:
                training_summary = json.load(f)
        except Exception:
            training_summary = None
    if training_summary is None and history:
        best_row = max(history, key=lambda r: r.get("val_iou_global", 0.0))
        training_summary = {
            "best_epoch": best_row["epoch"],
            "best_val_iou_global": best_row.get("val_iou_global"),
            "total_epochs_run": history[-1]["epoch"],
        }

    # --- C2-C4 summary (null if CSV absent or unreadable) ---
    c2c4_summary = None
    c2c4_csv = os.path.join(exp_dir, "c2c4_comparison.csv")
    if os.path.isfile(c2c4_csv):
        try:
            rows_c24 = []
            with open(c2c4_csv, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames_c24 = reader.fieldnames or []
                has_landmark = "err_c2_px" in fieldnames_c24
                for row in reader:
                    entry = {"abs_err_px": float(row["abs_err_px"])}
                    if has_landmark:
                        entry["err_c2_px"] = float(row["err_c2_px"])
                        entry["err_c4_px"] = float(row["err_c4_px"])
                        entry["err_landmark_mean_px"] = float(row["err_landmark_mean_px"])
                        entry["err_landmark_max_px"] = float(row["err_landmark_max_px"])
                        entry["assignment_swapped"] = int(row.get("assignment_swapped", 0))
                    rows_c24.append(entry)
            if rows_c24:
                n = len(rows_c24)
                mean_abs = sum(r["abs_err_px"] for r in rows_c24) / n
                std_abs = math.sqrt(sum((r["abs_err_px"] - mean_abs) ** 2 for r in rows_c24) / n)
                c2c4_summary = {
                    "n_valid": n,
                    "mean_abs_err_px": round(mean_abs, 3),
                    "std_abs_err_px": round(std_abs, 3),
                }
                if has_landmark:
                    thr = 5.0
                    c2c4_summary.update({
                        "mean_err_c2_px": round(sum(r["err_c2_px"] for r in rows_c24) / n, 3),
                        "mean_err_c4_px": round(sum(r["err_c4_px"] for r in rows_c24) / n, 3),
                        "mean_err_landmark_mean_px": round(sum(r["err_landmark_mean_px"] for r in rows_c24) / n, 3),
                        "mean_err_landmark_max_px": round(sum(r["err_landmark_max_px"] for r in rows_c24) / n, 3),
                        "pct_c2_lt5px": round(100.0 * sum(1 for r in rows_c24 if r["err_c2_px"] < thr) / n, 1),
                        "pct_c4_lt5px": round(100.0 * sum(1 for r in rows_c24 if r["err_c4_px"] < thr) / n, 1),
                        "pct_both_lt5px": round(
                            100.0 * sum(1 for r in rows_c24 if r["err_c2_px"] < thr and r["err_c4_px"] < thr) / n, 1
                        ),
                        "n_assignment_swapped": sum(r["assignment_swapped"] for r in rows_c24),
                    })
        except Exception:
            c2c4_summary = None

    # --- Boundary metrics from per-image CSVs ---
    test_boundary = _read_boundary_agg(os.path.join(exp_dir, "test_boundary_metrics.csv"))
    val_boundary  = _read_boundary_agg(os.path.join(exp_dir, "val_boundary_metrics.csv"))

    # --- Best-epoch pseudo-label stats and val boundary from diagnostics_epoch.csv ---
    pl_stats_at_best, val_boundary_at_best = _read_best_epoch_stats(exp_dir)

    # --- Epoch history (read once, reused by ssl_interpretation and diag summary) ---
    epoch_history = _read_epoch_history(exp_dir, cfg.get("use_semi"))

    report = {
        "run_identity": {
            "experiment_name": exp_name,
            "seed": seed,
            "exp_dir": exp_dir,
        },
        "config": cfg,
        "training_summary": training_summary,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_boundary_metrics": test_boundary,
        "val_boundary_metrics": val_boundary,
        "val_boundary_at_best_epoch": val_boundary_at_best,
        "pl_stats_at_best_epoch": pl_stats_at_best,
        "c2c4_summary": c2c4_summary,
        "file_references": {
            "diagnostics_epoch.csv": "diagnostics_epoch.csv",
            "test_metrics.csv": "test_metrics.csv",
            "test_boundary_metrics.csv": "test_boundary_metrics.csv",
            "val_boundary_metrics.csv": "val_boundary_metrics.csv",
            "c2c4_comparison.csv": "c2c4_comparison.csv" if c2c4_summary is not None else None,
            "test_preds": "test_preds/",
            "test_probs": "test_probs/",
        },
        # --- Derived / lifted sections (additive upgrade) ---
        "ssl_interpretation": _compute_ssl_interpretation(cfg, training_summary, epoch_history),
        "epoch_history": epoch_history,
        "training_diagnostics_summary": _compute_training_diagnostics_summary(epoch_history, cfg),
        "reproducibility_fingerprint": _read_debug_fingerprint(exp_dir),
        "artifact_status": _compute_artifact_status(exp_dir, val_metrics),
    }

    report_name = f"{exp_name}_seed_{seed}_run_report.json"
    report_path = os.path.join(exp_dir, report_name)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Run report saved: {report_path}")
    return report_path


@torch.no_grad()
def export_test_predictions(model, loader, out_dir, device="cuda", thr=0.5):
    os.makedirs(out_dir, exist_ok=True)
    probs_dir = os.path.join(os.path.dirname(out_dir), "test_probs")
    os.makedirs(probs_dir, exist_ok=True)
    model.eval()

    for batch in loader:
        xb = batch["image"].to(device, non_blocking=True).float()
        names = batch.get("name", [f"sample_{i}" for i in range(xb.shape[0])])

        logits = model(xb)
        probs = torch.sigmoid(logits)
        preds = (probs >= thr).float()

        for i in range(preds.shape[0]):
            pred = preds[i, 0].detach().cpu().numpy().astype(np.uint8) * 255
            save_path = os.path.join(out_dir, names[i])
            cv2.imwrite(save_path, pred)

            prob_map = probs[i, 0].detach().cpu().numpy().astype(np.float32)
            stem = os.path.splitext(names[i])[0]
            np.save(os.path.join(probs_dir, f"{stem}.npy"), prob_map)


@torch.no_grad()
def evaluate_checkpoint(cfg: dict, model, loaders: dict, best_path: str, history: list[dict] | None = None):
    # Lazy imports: keep matplotlib/visualization out of the top-level import
    # graph so write_run_report() and refresh_run_reports.py can run headless.
    from src.ruler_eval import compare_c2c4_manual_vs_auto
    from src.visualization import show_predictions

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = cfg["exp_dir"]

    if not os.path.isfile(best_path):
        raise FileNotFoundError(f"Checkpoint no encontrado: {best_path}")

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.to(device).eval()

    val_loader = loaders["val_loader"]
    test_loader = loaders["test_loader"]

    print("\n== Evaluación final ==")
    val_metrics = eval_imagewise_and_global(
        model, val_loader, device=device, thr=cfg["eval_threshold"], logits=True, split_name="VAL"
    )
    test_metrics = eval_imagewise_and_global(
        model, test_loader, device=device, thr=cfg["eval_threshold"], logits=True, split_name="TEST"
    )

    print("\n-- Métricas de borde (BF1, ASSD, HD95) --")
    tol_px = int(cfg.get("boundary_tol_px", 5))

    for split_name, loader in [("VAL", val_loader), ("TEST", test_loader)]:
        bf1_mean, assd_mean, hd95_mean = compute_boundary_metrics_epoch(
            model, loader, device=device, thr=cfg["eval_threshold"], r_tol_px=tol_px
        )
        print(
            f"[{split_name}] BF1@r={tol_px}px: {bf1_mean:.6f} | "
            f"ASSD: {assd_mean:.6f}px | HD95: {hd95_mean:.6f}px"
        )

    boundary_metrics_per_image_to_csv(
        model, val_loader, "val", device, cfg["eval_threshold"], tol_px, exp_dir
    )
    boundary_metrics_per_image_to_csv(
        model, test_loader, "test", device, cfg["eval_threshold"], tol_px, exp_dir
    )

    test_preds_dir = os.path.join(exp_dir, "test_preds")
    export_test_predictions(
        model,
        test_loader,
        out_dir=test_preds_dir,
        device=device,
        thr=cfg["eval_threshold"],
    )

    test_metrics_csv = os.path.join(exp_dir, "test_metrics.csv")
    write_simple_metrics_csv(test_metrics_csv, test_metrics)

    if cfg.get("run_ruler_eval", False):
        rotulos_dir = cfg.get("rotulos_dir", "")
        if rotulos_dir:
            compare_c2c4_manual_vs_auto(
                rotulos_dir=rotulos_dir,
                pred_masks_dir=test_preds_dir,
                out_csv=os.path.join(exp_dir, "c2c4_comparison.csv"),
                target_size=cfg.get("target_size", (320, 320)),
            )
            # visualize_c2c4_comparison() is intentionally NOT called here.
            # The C2-C4 overlay is now integrated into the 2x3 prediction figure
            # (show_predictions panel 5). To generate the standalone 1x3 figures,
            # call visualize_c2c4_comparison() manually from a notebook cell.

    if cfg.get("save_preds_vis", True):
        show_predictions(
            model,
            test_loader,
            device=device,
            thr=cfg["eval_threshold"],
            max_show=cfg.get("max_show_preds", 6),
            out_dir=os.path.join(exp_dir, "preds_vis"),
            tol_px=tol_px,
            c2c4_csv=os.path.join(exp_dir, "c2c4_comparison.csv"),
        )

    write_run_summary(cfg, history or [], exp_dir, test_metrics=test_metrics)
    write_run_report(cfg, test_metrics, exp_dir, history=history or [], val_metrics=val_metrics)

    return {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }