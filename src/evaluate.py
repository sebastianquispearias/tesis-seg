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
from src.ruler_eval import compare_c2c4_manual_vs_auto, visualize_c2c4_comparison
from src.visualization import show_predictions


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


def write_run_report(
    cfg: dict,
    test_metrics: dict,
    exp_dir: str,
    history: list | None = None,
    val_metrics: dict | None = None,
) -> str:
    """
    Writes a single portable JSON report for one completed run.
    Aggregates: config, training summary, test metrics, C2-C4 summary.
    Always succeeds even if C2-C4 or diagnostic_summary.json are absent.
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