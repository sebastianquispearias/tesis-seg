import os

import cv2
import numpy as np

from src.metrics import boundary_f1_score, assd_hd95


def make_square(h: int = 64, w: int = 64, x0: int = 20, y0: int = 20, size: int = 20):
    m = np.zeros((h, w), dtype=np.uint8)
    m[y0:y0 + size, x0:x0 + size] = 1
    return m


def run_boundary_sanity_checks(train_masks_dir: str, tol_px: int = 2):
    print("=== SANITY CHECK MÉTRICAS DE BORDE ===")

    gt = make_square()
    pred_equal = gt.copy()
    pred_shift = np.roll(gt, shift=3, axis=1)
    pred_empty = np.zeros_like(gt)
    pred_full = np.ones_like(gt)

    tests = [
        ("pred = GT", pred_equal),
        ("pred desplazada 3px", pred_shift),
        ("pred vacía", pred_empty),
        ("pred llena", pred_full),
    ]

    for name, p in tests:
        bf1 = boundary_f1_score(p.astype(bool), gt.astype(bool), r_tol_px=int(tol_px))
        assd, hd95 = assd_hd95(p.astype(bool), gt.astype(bool))
        print(f"{name} | BF1={bf1:.6f} | ASSD={assd} | HD95={hd95}")

    if not os.path.isdir(train_masks_dir):
        print("No existe train_masks_dir:", train_masks_dir)
        return

    mask_files = [f for f in os.listdir(train_masks_dir) if f.lower().endswith(".png")]
    if not mask_files:
        print("No encontré máscaras en:", train_masks_dir)
        return

    mask_path = os.path.join(train_masks_dir, mask_files[0])
    print("Usando máscara real:", mask_path)

    gt_real = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    gt_real = (gt_real > 0).astype(np.uint8)

    pred_equal_real = gt_real.copy()
    kernel = np.ones((3, 3), np.uint8)
    pred_smooth_real = cv2.morphologyEx(gt_real, cv2.MORPH_OPEN, kernel)
    pred_shift_real = np.roll(gt_real, shift=2, axis=1)

    tests_real = [
        ("real = GT", pred_equal_real),
        ("real suavizada", pred_smooth_real),
        ("real desplazada 2px", pred_shift_real),
    ]

    for name, p in tests_real:
        bf1 = boundary_f1_score(p.astype(bool), gt_real.astype(bool), r_tol_px=int(tol_px))
        assd, hd95 = assd_hd95(p.astype(bool), gt_real.astype(bool))
        print(f"{name} | BF1={bf1:.6f} | ASSD={assd} | HD95={hd95}")