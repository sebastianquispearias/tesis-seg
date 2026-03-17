import csv
import os

import cv2
import numpy as np
import torch

from src.metrics import (
    boundary_metrics_per_image_to_csv,
    compute_boundary_metrics_epoch,
    eval_imagewise_and_global,
)
from src.ruler_eval import compare_ruler_manual_vs_auto
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
            f.write(f"train_f1: {best_row.get('train_f1', float('nan')):.4f} | ")
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


@torch.no_grad()
def export_test_predictions(model, loader, out_dir, device="cuda", thr=0.5):
    os.makedirs(out_dir, exist_ok=True)
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
            compare_ruler_manual_vs_auto(
                rotulos_dir=rotulos_dir,
                pred_masks_dir=test_preds_dir,
                out_csv=os.path.join(exp_dir, "ruler_compare.csv"),
                target_size=cfg.get("target_size", (320, 320)),
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