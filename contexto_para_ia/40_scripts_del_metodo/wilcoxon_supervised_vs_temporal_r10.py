"""
wilcoxon_supervised_vs_temporal_r10.py
=======================================
Paired Wilcoxon signed-rank test comparing per-image overlap F1 (Dice) between
the Supervised baseline and the Temporal-r10 semi-supervised condition.

Methodology:
- Per-image F1 is computed from saved test_probs/*.npy (float32 probability
  maps, thresholded at 0.5) against GT masks in test/masks/*.png.
- For each condition (supervised, semi_r10), per-image F1 is averaged across
  the 3 available seeds (seed_0, seed_1, seed_2).
- Pairing is by image stem (filename without extension), using the intersection
  of stems present across all 6 seed directories.
- Two Wilcoxon tests are run:
    (1) Two-sided: tests whether the distributions differ.
    (2) One-sided:  wilcoxon(A, B, alternative="less")
                   = tests whether Temporal-r10 > Supervised (B > A).

Usage (local Windows):
    python wilcoxon_supervised_vs_temporal_r10.py \
        --workspace "G:/My Drive/UNM_vertebras_seg_v3"

Usage (Colab):
    !python wilcoxon_supervised_vs_temporal_r10.py \
        --workspace /content/drive/MyDrive/UNM_vertebras_seg_v3

Outputs:
    stats/wilcoxon_supervised_vs_temporal_r10.json
    stats/wilcoxon_per_image_f1.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_gt_mask(mask_path: Path) -> np.ndarray:
    """Load a grayscale GT mask as a binary bool array."""
    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read GT mask: {mask_path}")
    return img > 127


def load_pred_binary(npy_path: Path, threshold: float = 0.5) -> np.ndarray:
    """Load a float32 probability .npy and threshold to binary bool array."""
    prob = np.load(str(npy_path))
    # Shape may be (H, W) or (1, H, W) — squeeze to (H, W)
    if prob.ndim == 3:
        prob = prob[0]
    return prob >= threshold


def compute_f1(pred: np.ndarray, gt: np.ndarray) -> float:
    """Binary overlap F1 (Dice coefficient)."""
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def compute_per_image_f1(run_dir: Path, gt_dir: Path, threshold: float = 0.5):
    """
    Return {stem: f1} for all .npy files found in run_dir/test_probs/.
    Raises RuntimeError if test_probs/ is missing.
    """
    probs_dir = run_dir / "test_probs"
    if not probs_dir.is_dir():
        raise RuntimeError(f"test_probs/ not found in {run_dir}")

    result = {}
    for npy_path in sorted(probs_dir.glob("*.npy")):
        stem = npy_path.stem
        gt_path = gt_dir / f"{stem}.png"
        if not gt_path.is_file():
            print(f"  [WARN] GT mask not found for stem {stem!r}, skipping")
            continue
        pred = load_pred_binary(npy_path, threshold)
        gt   = load_gt_mask(gt_path)
        # Resize pred to GT size if they differ (shouldn't happen, but be safe)
        if pred.shape != gt.shape:
            pred_u8 = pred.astype(np.uint8)
            pred_u8 = cv2.resize(pred_u8, (gt.shape[1], gt.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
            pred = pred_u8.astype(bool)
        result[stem] = compute_f1(pred, gt)
    return result


def mean_median_std(values):
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), float(np.median(arr)), float(arr.std())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Paired Wilcoxon test: Supervised vs Temporal-r10 per-image F1."
    )
    parser.add_argument(
        "--workspace", type=str, default=None,
        help="Path to UNM_vertebras_seg_v3 root. "
             "Defaults to the parent of this script.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Binarisation threshold for probability maps (default: 0.5).",
    )
    args = parser.parse_args()

    ws = Path(args.workspace) if args.workspace else Path(__file__).resolve().parent
    runs_root = ws / "runs"
    gt_dir    = ws / "test" / "masks"
    out_dir   = ws / "stats"

    if not gt_dir.is_dir():
        print(f"[ERROR] GT mask dir not found: {gt_dir}")
        sys.exit(1)

    # ── Define run directories ─────────────────────────────────────────────
    CONDITIONS = {
        "supervised": [
            runs_root / "supervised" / "seed_0",
            runs_root / "supervised" / "seed_1",
            runs_root / "supervised" / "seed_2",
        ],
        "semi_r10": [
            runs_root / "semi_r10" / "seed_0",
            runs_root / "semi_r10" / "seed_1",
            runs_root / "semi_r10" / "seed_2",
        ],
    }

    for cond, dirs in CONDITIONS.items():
        for d in dirs:
            if not d.is_dir():
                print(f"[ERROR] Run dir not found: {d}")
                sys.exit(1)

    # ── Compute per-image F1 for every run ─────────────────────────────────
    print("Computing per-image F1 from test_probs/*.npy ...")
    all_f1 = {}  # {condition: [{stem: f1}, ...]}  one dict per seed
    for cond, dirs in CONDITIONS.items():
        all_f1[cond] = []
        for d in dirs:
            f1_dict = compute_per_image_f1(d, gt_dir, threshold=args.threshold)
            print(f"  {cond} / {d.name}: {len(f1_dict)} images")
            all_f1[cond].append(f1_dict)

    # ── Find paired stems ──────────────────────────────────────────────────
    stem_sets = []
    for cond in CONDITIONS:
        for f1_dict in all_f1[cond]:
            stem_sets.append(set(f1_dict.keys()))
    paired_stems = sorted(set.intersection(*stem_sets))
    print(f"\nPaired stems (present in all 6 runs): {len(paired_stems)}")

    if len(paired_stems) < 10:
        print("[WARN] Very few paired stems — check run directories.")

    # ── Average F1 across seeds per condition ─────────────────────────────
    sup_mean = {}
    r10_mean = {}
    for stem in paired_stems:
        sup_mean[stem] = float(np.mean([all_f1["supervised"][s][stem] for s in range(3)]))
        r10_mean[stem] = float(np.mean([all_f1["semi_r10"][s][stem]   for s in range(3)]))

    # Aligned vectors (sorted by stem for reproducibility)
    A = np.array([sup_mean[s] for s in paired_stems])  # supervised
    B = np.array([r10_mean[s] for s in paired_stems])  # semi_r10
    diff = B - A  # Temporal-r10 minus Supervised

    # ── Wilcoxon tests ─────────────────────────────────────────────────────
    from scipy.stats import wilcoxon

    stat_2s, p_2s = wilcoxon(A, B, alternative="two-sided")
    stat_1s, p_1s = wilcoxon(A, B, alternative="less")   # tests B > A

    diff_mean, diff_median, diff_std = mean_median_std(diff)

    # ── Console summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Paired Wilcoxon signed-rank test")
    print("  Supervised  vs  Temporal-r10")
    print("  Metric: per-image overlap F1 (Dice, threshold=0.5)")
    print(f"  n paired images : {len(paired_stems)}")
    print(f"\nPer-image diff (Temporal-r10 - Supervised):")
    print(f"  mean   = {diff_mean:+.4f}")
    print(f"  median = {diff_median:+.4f}")
    print(f"  std    =  {diff_std:.4f}")
    print(f"\nTwo-sided Wilcoxon (H1: distributions differ):")
    print(f"  W = {stat_2s:.1f},  p = {p_2s:.4g}")
    print(f"\nOne-sided Wilcoxon (H1: Temporal-r10 > Supervised):")
    print(f"  W = {stat_1s:.1f},  p = {p_1s:.4g}")
    print("  [wilcoxon(A, B, alternative='less') where A=supervised, B=semi_r10]")
    print("=" * 60)

    # ── Save JSON ──────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "description": "Paired Wilcoxon signed-rank test: Supervised vs Temporal-r10",
        "metric": f"per-image overlap F1 (Dice), threshold={args.threshold}",
        "source": "test_probs/*.npy thresholded vs test/masks/*.png",
        "n_paired_images": len(paired_stems),
        "conditions": {
            "supervised": {
                "seeds": [0, 1, 2],
                "run_dirs": [str(d) for d in CONDITIONS["supervised"]],
            },
            "semi_r10": {
                "seeds": [0, 1, 2],
                "run_dirs": [str(d) for d in CONDITIONS["semi_r10"]],
            },
        },
        "mean_f1": {
            "supervised": float(A.mean()),
            "semi_r10":   float(B.mean()),
        },
        "per_image_diff_r10_minus_sup": {
            "mean":   round(diff_mean,   6),
            "median": round(diff_median, 6),
            "std":    round(diff_std,    6),
        },
        "wilcoxon_two_sided": {
            "statistic": float(stat_2s),
            "p_value":   float(p_2s),
            "alternative": "two-sided",
        },
        "wilcoxon_one_sided_r10_gt_sup": {
            "statistic":  float(stat_1s),
            "p_value":    float(p_1s),
            "alternative": "less (A < B, i.e. supervised < semi_r10)",
            "call":        "wilcoxon(A=supervised, B=semi_r10, alternative='less')",
        },
    }

    json_path = out_dir / "wilcoxon_supervised_vs_temporal_r10.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nJSON saved -> {json_path}")

    # ── Save per-image CSV ─────────────────────────────────────────────────
    csv_path = out_dir / "wilcoxon_per_image_f1.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "stem", "supervised_mean_f1", "semi_r10_mean_f1", "diff_r10_minus_sup"
        ])
        w.writeheader()
        for stem in paired_stems:
            w.writerow({
                "stem":                stem,
                "supervised_mean_f1":  round(sup_mean[stem], 6),
                "semi_r10_mean_f1":    round(r10_mean[stem], 6),
                "diff_r10_minus_sup":  round(r10_mean[stem] - sup_mean[stem], 6),
            })
    print(f"CSV  saved -> {csv_path}")


if __name__ == "__main__":
    main()
