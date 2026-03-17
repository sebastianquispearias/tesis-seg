import csv
import os

import numpy as np
import torch

from src.metrics import (
    boundary_metrics_per_image_to_csv,
    compute_boundary_metrics_epoch,
    eval_imagewise_and_global,
    per_image_metrics,
)
from src.visualization import show_predictions


def write_run_summary(cfg: dict, history: list[dict], exp_dir: str, test_metrics: dict | None = None):
    summary_path = os.path.join(exp_dir, "run_summary.txt")
    best_row = None

    if history:
        best_row = min(history, key=lambda x: x["val_loss"])

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
            f.write(f"- train_loss: {best_row.get('train_loss', float('nan')):.6f}\n")
            f.write(f"- train_f1: {best_row.get('train_f1', float('nan')):.6f}\n")
            f.write(f"- train_iou: {best_row.get('train_iou', float('nan')):.6f}\n\n")

        if test_metrics is not None:
            f.write("[TEST metrics]\n")
            f.write(f"- Sample-wise F1:  {test_metrics['f1_mean']:.6f} ± {test_metrics['f1_std']:.6f}\n")
            f.write(f"- Sample-wise IoU: {test_metrics['iou_mean']:.6f} ± {test_metrics['iou_std']:.6f}\n\n")

        f.write("[Artefactos]\n")
        f.write(f"- preds_vis/: {os.path.join(exp_dir, 'preds_vis')}\n")
        f.write(f"- train_log.csv: {os.path.join(exp_dir, 'train_log.csv')}\n")

    print("Resumen guardado en:", summary_path)
    return summary_path


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
    tol_px = int(cfg.get("boundary_tol_px", 2))

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

    show_predictions(
        model,
        test_loader,
        device=device,
        thr=cfg["eval_threshold"],
        max_show=cfg.get("max_show_preds", 6),
        out_dir=os.path.join(exp_dir, "preds_vis"),
    )

    write_run_summary(cfg, history or [], exp_dir, test_metrics=test_metrics)

    return {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }