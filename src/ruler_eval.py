import csv
import math
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


def get_corners_from_angle(x: float, y: float, w: float, h: float, angle_degrees: float):
    corners = {
        "top_left": (x, y),
        "top_right": (x + w, y),
        "bottom_right": (x + w, y + h),
        "bottom_left": (x, y + h),
    }
    q1 = 0 < angle_degrees < 90
    q3 = -180 < angle_degrees < -90
    if q1 or q3:
        p1 = corners["top_right"]
        p2 = corners["bottom_left"]
    else:
        p1 = corners["top_left"]
        p2 = corners["bottom_right"]
    return p1, p2


def extract_manual_line_raw(rotulos_dir: Path, stem: str):
    csv_path = rotulos_dir / stem / "Results.csv"
    if not csv_path.exists():
        return None, None

    df = pd.read_csv(csv_path)
    if df is None or df.empty:
        return None, None

    row = df.iloc[0]
    p1, p2 = get_corners_from_angle(
        row["BX"], row["BY"], row["Width"], row["Height"], row["Angle"]
    )
    return p1, p2


def warp_points_like_preprocess(rotulos_dir: Path, stem: str, p1_raw, p2_raw, target_w: int, target_h: int):
    if p1_raw is None or p2_raw is None:
        return None, None

    mask_path = rotulos_dir / stem / "Mask.tif"
    if not mask_path.exists():
        return None, None

    m = Image.open(mask_path)
    w0, h0 = m.size

    side = max(w0, h0)
    pad_x = (side - w0) // 2
    pad_y = (side - h0) // 2

    def warp_one(p):
        x, y = p
        x_pad = x + pad_x
        y_pad = y + pad_y
        x_new = x_pad * (target_w / side)
        y_new = y_pad * (target_h / side)
        return float(x_new), float(y_new)

    return warp_one(p1_raw), warp_one(p2_raw)


def line_from_mask(mask_bool: np.ndarray):
    ys, xs = np.where(mask_bool > 0)
    if len(xs) < 2:
        return None, None

    pts = np.column_stack([xs, ys]).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

    x_min, x_max = xs.min(), xs.max()

    if abs(vx) < 1e-8:
        p1 = (float(x0), float(ys.min()))
        p2 = (float(x0), float(ys.max()))
    else:
        y1 = y0 + (x_min - x0) * (vy / vx)
        y2 = y0 + (x_max - x0) * (vy / vx)
        p1 = (float(x_min), float(y1))
        p2 = (float(x_max), float(y2))

    return p1, p2


def line_angle_deg(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return float(np.degrees(np.arctan2(dy, dx)))


def line_length(p1, p2):
    return float(np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2))


def compare_ruler_manual_vs_auto(
    rotulos_dir: str,
    pred_masks_dir: str,
    out_csv: str,
    target_size=(320, 320),
):
    rotulos_dir = Path(rotulos_dir)
    pred_masks_dir = Path(pred_masks_dir)
    target_w, target_h = target_size

    pred_files = sorted(pred_masks_dir.glob("*.png"))
    rows = []

    for pred_path in pred_files:
        stem = pred_path.stem

        p1_raw, p2_raw = extract_manual_line_raw(rotulos_dir, stem)
        p1_man, p2_man = warp_points_like_preprocess(rotulos_dir, stem, p1_raw, p2_raw, target_w, target_h)
        if p1_man is None:
            continue

        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        if pred is None:
            continue
        pred_bool = pred > 0

        p1_auto, p2_auto = line_from_mask(pred_bool)
        if p1_auto is None:
            continue

        angle_man = line_angle_deg(p1_man, p2_man)
        angle_auto = line_angle_deg(p1_auto, p2_auto)
        angle_abs_err = abs(angle_auto - angle_man)
        angle_abs_err = min(angle_abs_err, 180.0 - angle_abs_err)

        len_man = line_length(p1_man, p2_man)
        len_auto = line_length(p1_auto, p2_auto)

        rows.append({
            "stem": stem,
            "manual_angle_deg": angle_man,
            "auto_angle_deg": angle_auto,
            "angle_abs_err_deg": angle_abs_err,
            "manual_len_px": len_man,
            "auto_len_px": len_auto,
            "len_abs_err_px": abs(len_auto - len_man),
        })

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "stem",
                "manual_angle_deg",
                "auto_angle_deg",
                "angle_abs_err_deg",
                "manual_len_px",
                "auto_len_px",
                "len_abs_err_px",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print("guardado:", out_csv)
    return out_csv