import csv
import math
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


# ─── Manual annotation helpers (shared) ──────────────────────────────────────

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


def line_length(p1, p2):
    return float(np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2))


def _euc(a, b) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def c2c4_landmark_summary(df) -> dict:
    """
    Compute aggregate landmark error statistics from a c2c4_comparison DataFrame.
    Returns None if df is None or empty.
    """
    if df is None or len(df) == 0:
        return None
    n = len(df)
    thr = 5.0
    summary = {
        "n_valid": n,
        "mean_abs_err_px": round(float(df["abs_err_px"].mean()), 3),
        "std_abs_err_px": round(float(df["abs_err_px"].std()), 3),
    }
    if "err_c2_px" in df.columns:
        summary.update({
            "mean_err_c2_px": round(float(df["err_c2_px"].mean()), 3),
            "mean_err_c4_px": round(float(df["err_c4_px"].mean()), 3),
            "mean_err_landmark_mean_px": round(float(df["err_landmark_mean_px"].mean()), 3),
            "mean_err_landmark_max_px": round(float(df["err_landmark_max_px"].mean()), 3),
            "pct_c2_lt5px": round(100.0 * float((df["err_c2_px"] < thr).sum()) / n, 1),
            "pct_c4_lt5px": round(100.0 * float((df["err_c4_px"] < thr).sum()) / n, 1),
            "pct_both_lt5px": round(
                100.0 * float(((df["err_c2_px"] < thr) & (df["err_c4_px"] < thr)).sum()) / n, 1
            ),
            "n_assignment_swapped": int(df["assignment_swapped"].sum()) if "assignment_swapped" in df.columns else None,
        })
    return summary


# ─── Anatomical C2-C4 helpers ─────────────────────────────────────────────────

def _corner_inferior_anterior(mask_u8: np.ndarray):
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return None
    y_min, y_max = ys.min(), ys.max()
    y_thr = y_min + 0.66 * (y_max - y_min)
    idx = np.where(ys >= y_thr)[0]
    if len(idx) == 0:
        idx = np.arange(len(xs))
    xs2, ys2 = xs[idx], ys[idx]
    best = None
    best_score = None
    for x, y in zip(xs2, ys2):
        score = x - 0.5 * y  # small = leftward and inferior
        if best_score is None or score < best_score:
            best_score = score
            best = (int(x), int(y))
    return best


def c2_c4_from_mask_legacy(
    mask_bin: np.ndarray,
    min_pixels: int = 80,
    n_samples: int = 120,
    slab_half_thickness: float = 2.0,
    valley_alpha: float = 0.3,
    n_bands_fallback: int = 4,
):
    """
    Anatomical C2-C4 approximation via PCA + inter-vertebral gap detection.
    Kept for reference/debugging. Use c2_c4_from_mask() for new code.

    1) PCA on all mask pixels → principal axis.
    2) Build 1D profile f(t): pixel count in a slab perpendicular to the axis.
    3) Detect deep valleys in f(t) → inter-vertebral gaps.
    4) Assign segments to C2, C3, C4.
    5) In each segment find the infero-anterior corner.
    6) Return (p_C2, p_C4, distance_px). Falls back to 4-band split if no
       clear valleys are found.
    """
    mask_u8 = (mask_bin > 0).astype(np.uint8)
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) < min_pixels:
        return None, None, None

    # PCA
    coords = np.stack([xs, ys], axis=1).astype(float)
    center = coords.mean(axis=0)
    coords_centered = coords - center

    cov = coords_centered.T @ coords_centered / (coords_centered.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_dir = eigvecs[:, np.argmax(eigvals)]
    norm = np.linalg.norm(principal_dir)
    if norm < 1e-8:
        return None, None, None
    principal_dir = principal_dir / norm

    t = coords_centered @ principal_dir
    t_min, t_max = t.min(), t.max()
    if (t_max - t_min) < 1e-3:
        return None, None, None

    # 1D profile
    t_grid = np.linspace(t_min, t_max, n_samples)
    f = np.array([np.sum(np.abs(t - t0) <= slab_half_thickness) for t0 in t_grid], dtype=float)
    if f.max() < 1e-3:
        return None, None, None

    # Smooth and find valleys
    kernel = np.array([1., 2., 1.]) / 4.0
    f_smooth = np.convolve(f, kernel, mode="same")

    thr_valley = max(2.0, valley_alpha * f_smooth.max())
    valleys_idx = [
        i for i in range(1, len(f_smooth) - 1)
        if f_smooth[i] < f_smooth[i - 1]
        and f_smooth[i] < f_smooth[i + 1]
        and f_smooth[i] <= thr_valley
    ]

    def segment_mask(a, b):
        sel = (t >= a) & (t <= b)
        if sel.sum() < max(10, min_pixels // 6):
            return None
        m = np.zeros_like(mask_u8, dtype=np.uint8)
        m[ys[sel], xs[sel]] = 1
        return m

    if len(valleys_idx) >= 2:
        cuts = [t_grid[i] for i in sorted(valleys_idx)]
        bounds = [t_min] + cuts + [t_max]
        m_c2 = segment_mask(bounds[0], bounds[1])
        m_c4 = segment_mask(bounds[2], bounds[3]) if len(bounds) >= 4 else segment_mask(bounds[-2], bounds[-1])
        if m_c2 is not None and m_c4 is not None:
            p2 = _corner_inferior_anterior(m_c2)
            p4 = _corner_inferior_anterior(m_c4)
            if p2 is not None and p4 is not None:
                return p2, p4, float(math.hypot(p4[0] - p2[0], p4[1] - p2[1]))

    # Fallback: 4 equal bands along the axis
    L = t_max - t_min

    def band_mask(band_idx):
        a = t_min + L * (band_idx / n_bands_fallback)
        b = t_min + L * ((band_idx + 1) / n_bands_fallback)
        sel = (t >= a) & (t <= b)
        if sel.sum() < max(10, min_pixels // n_bands_fallback):
            return None
        m = np.zeros_like(mask_u8, dtype=np.uint8)
        m[ys[sel], xs[sel]] = 1
        return m

    m_c2 = band_mask(0)
    m_c4 = band_mask(2 if n_bands_fallback >= 3 else n_bands_fallback - 1)
    if m_c2 is None or m_c4 is None:
        return None, None, None
    p2 = _corner_inferior_anterior(m_c2)
    p4 = _corner_inferior_anterior(m_c4)
    if p2 is None or p4 is None:
        return None, None, None
    return p2, p4, float(math.hypot(p4[0] - p2[0], p4[1] - p2[1]))


def c2_c4_from_mask(
    mask_bin: np.ndarray,
    min_pixels: int = 80,
    n_samples: int = 120,           # unused; kept for signature compatibility
    slab_half_thickness: float = 2.0,  # unused
    valley_alpha: float = 0.3,         # unused
    n_bands_fallback: int = 4,         # unused
):
    """
    Direct-geometry C2-C4 landmark detection.

    1) Binarize mask; keep largest connected component.
    2) Extract external contour points.
    3) Split into C2 zone (top 33%) and C4 zone (bottom 33%) by y-coordinate.
    4) Select infero-anterior corner in each zone via score = x - 0.5*y (minimum).
    5) Anatomical sanity: ensure C2 is above C4 (smaller y).

    Returns (p_c2, p_c4, distance_px) on success, (None, None, None) on failure.
    Point coordinates are integer (x, y) tuples.
    """
    mask_u8 = (mask_bin > 0).astype(np.uint8)

    # Largest connected component
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8)
    if n_labels < 2:
        return None, None, None
    fg_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(fg_areas)) + 1  # +1: skip background (label 0)

    if stats[largest_label, cv2.CC_STAT_AREA] < min_pixels:
        return None, None, None

    comp_mask = (labels == largest_label).astype(np.uint8)

    # External contour
    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None, None

    # All contour points as (x, y), shape (N, 2)
    pts = np.concatenate([c.reshape(-1, 2) for c in contours], axis=0)

    y_min = int(pts[:, 1].min())
    y_max = int(pts[:, 1].max())
    span = y_max - y_min
    if span < 1:
        return None, None, None

    # Zone masks
    c2_zone = pts[:, 1] <= y_min + 0.33 * span
    c4_zone = pts[:, 1] >= y_min + 0.67 * span
    if not c2_zone.any() or not c4_zone.any():
        return None, None, None

    # Infero-anterior corner: minimum of score = x - 0.5*y
    scores = pts[:, 0].astype(float) - 0.5 * pts[:, 1].astype(float)
    c2_pts = pts[c2_zone]
    c4_pts = pts[c4_zone]
    idx_c2 = int(np.argmin(scores[c2_zone]))
    idx_c4 = int(np.argmin(scores[c4_zone]))
    p_c2 = (int(c2_pts[idx_c2, 0]), int(c2_pts[idx_c2, 1]))
    p_c4 = (int(c4_pts[idx_c4, 0]), int(c4_pts[idx_c4, 1]))

    # Anatomical sanity: C2 must be above C4 (smaller y)
    if p_c2[1] > p_c4[1]:
        p_c2, p_c4 = p_c4, p_c2

    dist = float(math.hypot(p_c4[0] - p_c2[0], p_c4[1] - p_c2[1]))
    return p_c2, p_c4, dist


# ─── C2-C4 comparison and visualization ──────────────────────────────────────

def compare_c2c4_manual_vs_auto(
    rotulos_dir: str,
    pred_masks_dir: str,
    out_csv: str,
    target_size=(320, 320),
):
    """
    For every predicted mask in pred_masks_dir, computes:
      - d_gt:   manual reference distance from Results.csv (warped bounding box diagonal)
      - d_pred: anatomical C2-C4 distance from c2_c4_from_mask() on predicted mask

    Saves out_csv and returns a DataFrame.
    """
    rotulos_dir = Path(rotulos_dir)
    pred_masks_dir = Path(pred_masks_dir)
    target_w, target_h = target_size

    pred_files = sorted(pred_masks_dir.glob("*.png"))
    rows = []
    n_skipped = 0

    for pred_path in pred_files:
        stem = pred_path.stem

        # Manual reference
        p1_raw, p2_raw = extract_manual_line_raw(rotulos_dir, stem)
        p1_man, p2_man = warp_points_like_preprocess(rotulos_dir, stem, p1_raw, p2_raw, target_w, target_h)
        if p1_man is None:
            n_skipped += 1
            continue

        d_gt = line_length(p1_man, p2_man)

        # Automatic C2-C4
        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        if pred is None:
            n_skipped += 1
            continue
        p_c2, p_c4, d_pred = c2_c4_from_mask(pred)
        if p_c2 is None:
            n_skipped += 1
            continue

        # Direct anatomical matching: p_c2 ↔ p1_man, p_c4 ↔ p2_man
        err_c2 = _euc(p_c2, p1_man)
        err_c4 = _euc(p_c4, p2_man)
        err_lm_mean = (err_c2 + err_c4) / 2.0
        err_lm_max = max(err_c2, err_c4)
        # Diagnostic: flag if reversed assignment would have lower total error
        err_c2_alt = _euc(p_c2, p2_man)
        err_c4_alt = _euc(p_c4, p1_man)
        assignment_swapped = 1 if (err_c2_alt + err_c4_alt) < (err_c2 + err_c4) else 0

        rows.append({
            "stem": stem,
            "d_gt_px": round(d_gt, 3),
            "d_pred_px": round(d_pred, 3),
            "abs_err_px": round(abs(d_pred - d_gt), 3),
            "p1_man_x": round(p1_man[0], 3),
            "p1_man_y": round(p1_man[1], 3),
            "p2_man_x": round(p2_man[0], 3),
            "p2_man_y": round(p2_man[1], 3),
            "p_c2_x": p_c2[0],
            "p_c2_y": p_c2[1],
            "p_c4_x": p_c4[0],
            "p_c4_y": p_c4[1],
            "err_c2_px": round(err_c2, 3),
            "err_c4_px": round(err_c4, 3),
            "err_landmark_mean_px": round(err_lm_mean, 3),
            "err_landmark_max_px": round(err_lm_max, 3),
            "assignment_swapped": assignment_swapped,
        })

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    fieldnames = [
        "stem", "d_gt_px", "d_pred_px", "abs_err_px",
        "p1_man_x", "p1_man_y", "p2_man_x", "p2_man_y",
        "p_c2_x", "p_c2_y", "p_c4_x", "p_c4_y",
        "err_c2_px", "err_c4_px", "err_landmark_mean_px", "err_landmark_max_px",
        "assignment_swapped",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"c2c4_comparison saved: {out_csv}  ({len(rows)} matched, {n_skipped} skipped)")
    return pd.DataFrame(rows)


def visualize_c2c4_comparison(
    rotulos_dir: str,
    pred_masks_dir: str,
    test_images_dir: str,
    out_dir: str,
    target_size=(320, 320),
):
    """
    For every predicted mask with a valid manual annotation and automatic C2-C4
    estimate, saves a 1×3 figure to out_dir/{stem}.png and displays it inline.

    Panel 0: original grayscale test image
    Panel 1: predicted mask with C2 (cyan dot) and C4 (magenta dot) + yellow line
    Panel 2: image + mask tint (blue) + manual reference line (red) + auto C2-C4 line (yellow)
    """
    rotulos_dir = Path(rotulos_dir)
    pred_masks_dir = Path(pred_masks_dir)
    test_images_dir = Path(test_images_dir)
    os.makedirs(out_dir, exist_ok=True)
    target_w, target_h = target_size

    pred_files = sorted(pred_masks_dir.glob("*.png"))
    n_saved = 0
    n_skipped = 0

    for pred_path in pred_files:
        stem = pred_path.stem

        # Manual reference
        p1_raw, p2_raw = extract_manual_line_raw(rotulos_dir, stem)
        p1_man, p2_man = warp_points_like_preprocess(rotulos_dir, stem, p1_raw, p2_raw, target_w, target_h)
        if p1_man is None:
            n_skipped += 1
            continue
        d_gt = line_length(p1_man, p2_man)

        # Predicted mask
        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        if pred is None:
            n_skipped += 1
            continue
        p_c2, p_c4, d_pred = c2_c4_from_mask(pred)
        if p_c2 is None:
            n_skipped += 1
            continue
        abs_err = abs(d_pred - d_gt)

        # Original test image — resized to target_size to match mask coordinate space
        img_path = test_images_dir / f"{stem}.png"
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            n_skipped += 1
            continue
        img = cv2.resize(img, (target_w, target_h))

        # --- Panel 1: mask with C2/C4 dots and connecting line ---
        mask_rgb = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)
        cv2.line(mask_rgb, p_c2, p_c4, (255, 255, 0), 2)       # yellow line
        cv2.circle(mask_rgb, p_c2, 5, (0, 255, 255), -1)        # cyan = C2
        cv2.circle(mask_rgb, p_c4, 5, (255, 0, 255), -1)        # magenta = C4

        # --- Panel 2: image overlay ---
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # tint mask region blue
        overlay = img_rgb.copy()
        mask_bool = pred > 0
        overlay[mask_bool, 0] = np.clip(overlay[mask_bool, 0].astype(int) // 2, 0, 255)
        overlay[mask_bool, 2] = np.clip(overlay[mask_bool, 2].astype(int) // 2 + 127, 0, 255)
        # manual line in red
        cv2.line(overlay,
                 (int(p1_man[0]), int(p1_man[1])),
                 (int(p2_man[0]), int(p2_man[1])),
                 (255, 0, 0), 2)
        # auto C2-C4 line in yellow
        cv2.line(overlay, p_c2, p_c4, (255, 255, 0), 2)
        cv2.circle(overlay, p_c2, 5, (0, 255, 255), -1)
        cv2.circle(overlay, p_c4, 5, (255, 0, 255), -1)

        fig, axs = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)

        axs[0].imshow(img, cmap="gray")
        axs[0].set_title("Image")
        axs[0].axis("off")

        axs[1].imshow(mask_rgb)
        axs[1].set_title("Mask  C2●(cyan) C4●(magenta)")
        axs[1].axis("off")

        axs[2].imshow(overlay)
        axs[2].set_title("Overlay  manual(red)  auto(yellow)")
        axs[2].axis("off")

        fig.suptitle(
            f"{stem}  |  d_gt={d_gt:.1f}px  d_pred={d_pred:.1f}px  err={abs_err:.1f}px",
            fontsize=10,
        )

        save_path = os.path.join(out_dir, f"{stem}.png")
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        n_saved += 1

    print(f"c2c4_vis: {n_saved} figures saved to {out_dir}  ({n_skipped} skipped)")
