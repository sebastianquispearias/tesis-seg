import csv
import math
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch


def to_numpy_bool(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.squeeze(x)
    return x > 0.5


def iou_f1_from_counts(tp: int, fp: int, fn: int, eps: float = 1e-7) -> Tuple[float, float]:
    iou = tp / (tp + fp + fn + eps)
    f1 = (2 * tp) / (2 * tp + fp + fn + eps)
    return float(iou), float(f1)


def per_image_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray) -> Tuple[float, float]:
    pred_bin = pred_bin.astype(bool)
    gt_bin = gt_bin.astype(bool)

    tp = int((pred_bin & gt_bin).sum())
    fp = int((pred_bin & (~gt_bin)).sum())
    fn = int(((~pred_bin) & gt_bin).sum())

    if gt_bin.sum() == 0 and pred_bin.sum() == 0:
        return 1.0, 1.0

    return iou_f1_from_counts(tp, fp, fn)[::-1]  # retorna f1, iou


@torch.no_grad()
def eval_imagewise_and_global(model, loader, device="cuda", thr=0.5, logits=True, split_name="VAL"):
    model.eval()

    ious, f1s = [], []
    TP_all = FP_all = FN_all = 0.0

    for batch in loader:
        xb = batch["image"].to(device, non_blocking=True).float()
        yb = batch["mask"].to(device, non_blocking=True).float()

        pb = model(xb)
        if logits:
            pb = torch.sigmoid(pb)
        pb = (pb >= thr).float()

        B = yb.shape[0]
        for i in range(B):
            y = to_numpy_bool(yb[i, 0])
            p = to_numpy_bool(pb[i, 0])

            if y.sum() == 0 and p.sum() == 0:
                iou_i, f1_i = 1.0, 1.0
            else:
                tp = int((p & y).sum())
                fp = int((p & (~y)).sum())
                fn = int(((~p) & y).sum())
                iou_i, f1_i = iou_f1_from_counts(tp, fp, fn)

                TP_all += tp
                FP_all += fp
                FN_all += fn

            ious.append(iou_i)
            f1s.append(f1_i)

    ious = np.array(ious, dtype=np.float32)
    f1s = np.array(f1s, dtype=np.float32)

    iou_g, f1_g = iou_f1_from_counts(TP_all, FP_all, FN_all)

    print(f"[{split_name}] [amostra]  F1: {f1s.mean():.6f} ± {f1s.std():.6f} | IoU: {ious.mean():.6f} ± {ious.std():.6f}")
    print(f"[{split_name}] [global]   F1: {f1_g:.6f} | IoU: {iou_g:.6f}")

    return {
        "sample_mean_iou": float(ious.mean()),
        "sample_mean_f1": float(f1s.mean()),
        "global_iou": float(iou_g),
        "global_f1": float(f1_g),
        "n_images": int(len(ious)),
        # backward-compat aliases
        "f1_mean": float(f1s.mean()),
        "f1_std": float(f1s.std()),
        "iou_mean": float(ious.mean()),
        "iou_std": float(ious.std()),
        "f1_global": float(f1_g),
        "iou_global": float(iou_g),
    }


def mask_to_boundary(mask_bool: np.ndarray) -> np.ndarray:
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boundary = np.zeros_like(mask_u8)
    cv2.drawContours(boundary, contours, -1, 255, 1)
    return boundary > 0


def boundary_f1_score(pred_bool: np.ndarray, gt_bool: np.ndarray, r_tol_px: int = 2) -> float:
    pred_b = mask_to_boundary(pred_bool)
    gt_b = mask_to_boundary(gt_bool)

    if pred_b.sum() == 0 and gt_b.sum() == 0:
        return 1.0
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return 0.0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_tol_px + 1, 2 * r_tol_px + 1))
    pred_d = cv2.dilate(pred_b.astype(np.uint8), kernel) > 0
    gt_d = cv2.dilate(gt_b.astype(np.uint8), kernel) > 0

    precision = (pred_b & gt_d).sum() / (pred_b.sum() + 1e-7)
    recall = (gt_b & pred_d).sum() / (gt_b.sum() + 1e-7)

    return float((2 * precision * recall) / (precision + recall + 1e-7))


def _surface_points(mask_bool: np.ndarray) -> np.ndarray:
    b = mask_to_boundary(mask_bool)
    pts = np.column_stack(np.where(b))
    return pts.astype(np.float32)


def assd_hd95(pred_bool: np.ndarray, gt_bool: np.ndarray) -> Tuple[float, float]:
    pred_pts = _surface_points(pred_bool)
    gt_pts = _surface_points(gt_bool)

    if len(pred_pts) == 0 and len(gt_pts) == 0:
        return 0.0, 0.0
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("inf"), float("inf")

    d_pred_to_gt = np.sqrt(((pred_pts[:, None, :] - gt_pts[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    d_gt_to_pred = np.sqrt(((gt_pts[:, None, :] - pred_pts[None, :, :]) ** 2).sum(axis=2)).min(axis=1)

    assd = float((d_pred_to_gt.mean() + d_gt_to_pred.mean()) / 2.0)
    hd95 = float(np.percentile(np.concatenate([d_pred_to_gt, d_gt_to_pred]), 95))

    return assd, hd95


@torch.no_grad()
def compute_boundary_metrics_epoch(model, loader, device="cuda", thr=0.5, r_tol_px: int = 2):
    model.eval()
    bf1s, assds, hd95s = [], [], []

    for batch in loader:
        xb = batch["image"].to(device, non_blocking=True).float()
        yb = batch["mask"].to(device, non_blocking=True).float()

        logits = model(xb)
        probs = torch.sigmoid(logits)
        preds = (probs >= thr).float()

        for i in range(preds.shape[0]):
            pred = to_numpy_bool(preds[i, 0])
            gt = to_numpy_bool(yb[i, 0])

            if not gt.any():
                continue  # GT vacío → no contribuye a métricas de borde

            bf1 = boundary_f1_score(pred, gt, r_tol_px=r_tol_px)
            assd, hd95 = assd_hd95(pred, gt)

            bf1s.append(bf1)
            if math.isfinite(assd):
                assds.append(assd)
            if math.isfinite(hd95):
                hd95s.append(hd95)

    bf1_mean = float(np.mean(bf1s)) if bf1s else float("nan")
    assd_mean = float(np.mean(assds)) if assds else float("nan")
    hd95_mean = float(np.mean(hd95s)) if hd95s else float("nan")
    return bf1_mean, assd_mean, hd95_mean


@torch.no_grad()
def boundary_metrics_per_image_to_csv(model, loader, split_name, device, thr, r_tol_px, out_dir):
    model.eval()
    rows = []

    idx = 0
    for batch in loader:
        xb = batch["image"].to(device, non_blocking=True).float()
        yb = batch["mask"].to(device, non_blocking=True).float()
        names = batch.get("name", [f"{split_name}_{i}" for i in range(xb.shape[0])])

        logits = model(xb)
        probs = torch.sigmoid(logits)
        preds = (probs >= thr).float()

        for i in range(preds.shape[0]):
            pred = to_numpy_bool(preds[i, 0])
            gt = to_numpy_bool(yb[i, 0])

            bf1 = boundary_f1_score(pred, gt, r_tol_px=r_tol_px)
            assd, hd95 = assd_hd95(pred, gt)

            rows.append({
                "idx": idx,
                "name": names[i],
                "bf1": bf1,
                "assd": assd,
                "hd95": hd95,
            })
            idx += 1

    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{split_name.lower()}_boundary_metrics.csv")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "name", "bf1", "assd", "hd95"])
        w.writeheader()
        w.writerows(rows)

    print("guardado:", out_csv)
    return out_csv