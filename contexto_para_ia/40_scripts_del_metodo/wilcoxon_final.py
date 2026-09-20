"""
wilcoxon_final.py
=================
Paired Wilcoxon signed-rank tests on the final rerun results.

7 comparisons x 3 seeds = 21 independent tests.
Per-image F1 (Dice) computed on-the-fly from test_preds/ PNGs vs GT masks.

Output: resultados/stats_final/
  - wilcoxon_per_seed_results.csv
  - per_image_deltas_long.csv
  - wilcoxon_summary.md

Usage:
    python wilcoxon_final.py
"""

import csv
import math
import sys
import io
from pathlib import Path

import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PROJECT = Path(r"G:/My Drive/UNM_vertebras_seg_v3")
UNM_RUNS = PROJECT / "runs_final_v1"
INCA_RUNS = PROJECT / "runs_inca_final_v1"
UNM_GT = PROJECT / "test" / "masks"
INCA_GT = PROJECT / "data" / "inca_dataset" / "test" / "masks"
OUT_DIR = PROJECT / "resultados" / "stats_final"

SEEDS = [0, 1, 2]

COMPARISONS = [
    # Eje 1: SSL vs supervised
    ("supervised",                "mean_teacher_all_lateral",            "UNM_sup_vs_MT_all_lateral",       "UNM"),
    ("supervised",                "semi_r3",                             "UNM_sup_vs_PL_r3",                "UNM"),
    ("supervised_inca",           "semi_inca_r20",                       "INCA_full_sup_vs_PL_r20",         "INCA"),
    ("supervised_inca_patient10", "semi_inca_r10_patient10",             "INCA_p10_sup_vs_PL_r10",          "INCA"),
    # Eje 3: temporal vs random
    ("semi_r10",                  "semi_std_matched_r10",                "UNM_PL_r10_temp_vs_rand",         "UNM"),
    ("mean_teacher_r10",          "mean_teacher_std_matched_r10",        "UNM_MT_r10_temp_vs_rand",         "UNM"),
    ("semi_inca_r10_patient10",   "semi_inca_std_matched_r10_patient10", "INCA_p10_PL_r10_temp_vs_rand",   "INCA"),
]


def _rankdata(a):
    """Rank data with average ranks for ties (equivalent to scipy.stats.rankdata)."""
    arr = np.asarray(a, dtype=float)
    sorter = np.argsort(arr, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(arr))

    arr_sorted = arr[sorter]
    obs = np.concatenate(([True], arr_sorted[1:] != arr_sorted[:-1]))
    dense = np.cumsum(obs)[inv]

    # Average ranks for ties
    count = np.bincount(dense)
    cumcount = np.cumsum(count)
    ranks = np.empty_like(arr)
    for i in range(len(count)):
        mask = dense == i
        start = cumcount[i] - count[i]
        end = cumcount[i]
        ranks[mask] = (start + end + 1) / 2.0  # average rank (1-based)
    return ranks


def wilcoxon_test(a, b):
    """Two-sided Wilcoxon signed-rank test with Pratt zero method.

    Returns (statistic, p_value). Uses normal approximation with tie
    correction (accurate for n > 25; our datasets have n=63 and n=194).
    """
    d = np.asarray(b) - np.asarray(a)
    n = len(d)

    # Pratt method: rank ALL |d| including zeros
    abs_d = np.abs(d)
    ranks = _rankdata(abs_d)

    # Only consider non-zero differences for the test statistic
    nonzero = d != 0
    n_r = int(nonzero.sum())

    if n_r < 2:
        return 0.0, 1.0

    # W+ = sum of ranks where d > 0 (non-zero only)
    r_plus = ranks[(d > 0)].sum()
    r_minus = ranks[(d < 0)].sum()
    T = min(r_plus, r_minus)

    # Normal approximation
    mu = n_r * (n_r + 1.0) / 4.0

    # Tie correction for variance
    # For each group of t tied values, subtract t*(t^2-1)/48
    unique_ranks, tie_counts = np.unique(ranks[nonzero], return_counts=True)
    tie_correction = sum(t * (t * t - 1) for t in tie_counts if t > 1) / 48.0

    sigma_sq = n_r * (n_r + 1.0) * (2.0 * n_r + 1.0) / 24.0 - tie_correction
    if sigma_sq <= 0:
        return T, 1.0
    sigma = math.sqrt(sigma_sq)

    z = (T - mu) / sigma
    # Two-sided p-value: p = erfc(|z| / sqrt(2))
    p = math.erfc(abs(z) / math.sqrt(2.0))

    return float(T), p


def compute_dice(pred_bin, gt_bin):
    gt_sum = gt_bin.sum()
    pred_sum = pred_bin.sum()
    if gt_sum == 0 and pred_sum == 0:
        return 1.0
    if gt_sum == 0 or pred_sum == 0:
        return 0.0
    intersection = (pred_bin * gt_bin).sum()
    return 2.0 * intersection / (pred_sum + gt_sum)


def find_test_preds(runs_root, exp_name, seed):
    primary = runs_root / exp_name / f"seed_{seed}" / "test_preds"
    if primary.is_dir():
        return primary
    for suffix in range(1, 5):
        alt = runs_root / f"{exp_name} ({suffix})" / f"seed_{seed}" / "test_preds"
        if alt.is_dir():
            return alt
    return None


def compute_per_image_f1(gt_dir, preds_dir):
    results = {}
    for gt_path in sorted(gt_dir.glob("*.png")):
        stem = gt_path.stem
        pred_path = preds_dir / f"{stem}.png"
        if not pred_path.is_file():
            continue
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        if gt is None or pred is None:
            continue
        # Resize GT to match prediction size (preds are at target_size=320x320)
        if gt.shape != pred.shape:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
        gt_bin = (gt > 127).astype(np.uint8)
        pred_bin = (pred > 0).astype(np.uint8)
        results[stem] = compute_dice(pred_bin, gt_bin)
    return results


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seed_rows = []
    delta_rows = []
    md_lines = []

    md_lines.append("# Wilcoxon Signed-Rank Test Results - Final Runs\n\n")
    md_lines.append("Per-image F1 comparison, paired by test image within each seed.\n\n")

    for condA, condB, label, dataset in COMPARISONS:
        runs_root = UNM_RUNS if dataset == "UNM" else INCA_RUNS
        gt_dir = UNM_GT if dataset == "UNM" else INCA_GT
        n_gt = len(list(gt_dir.glob("*.png")))

        print(f"{label}:")
        md_lines.append(f"## {label}\n\n")
        md_lines.append(f"- **A**: `{condA}`\n")
        md_lines.append(f"- **B**: `{condB}`\n")
        md_lines.append(f"- **Dataset**: {dataset} ({n_gt} test images)\n\n")
        md_lines.append("| Seed | N pairs | Mean A | Mean B | Mean delta | Median delta | Wins B | Wins A | Ties | p-value | Sig. |\n")
        md_lines.append("|------|---------|--------|--------|------------|--------------|--------|--------|------|---------|------|\n")

        directions = []
        significants = 0
        skipped = False

        for seed in SEEDS:
            preds_a = find_test_preds(runs_root, condA, seed)
            preds_b = find_test_preds(runs_root, condB, seed)

            if preds_a is None or preds_b is None:
                missing = []
                if preds_a is None:
                    missing.append(f"{condA}/seed_{seed}")
                if preds_b is None:
                    missing.append(f"{condB}/seed_{seed}")
                print(f"  SKIP seed {seed}: test_preds missing for {missing}")
                md_lines.append(f"| {seed} | SKIP | - | - | - | - | - | - | - | - | - |\n")
                skipped = True
                continue

            f1_a = compute_per_image_f1(gt_dir, preds_a)
            f1_b = compute_per_image_f1(gt_dir, preds_b)

            common = sorted(set(f1_a.keys()) & set(f1_b.keys()))
            n_pairs = len(common)

            a_vals = np.array([f1_a[s] for s in common])
            b_vals = np.array([f1_b[s] for s in common])
            diffs = b_vals - a_vals

            mean_a = float(a_vals.mean())
            mean_b = float(b_vals.mean())
            mean_delta = float(diffs.mean())
            median_delta = float(np.median(diffs))
            wins_b = int((diffs > 0).sum())
            wins_a = int((diffs < 0).sum())
            ties = int((diffs == 0).sum())

            nonzero = diffs[diffs != 0]
            if len(nonzero) < 2:
                p_val = 1.0
            else:
                _, p_val = wilcoxon_test(a_vals, b_vals)

            sig = p_val < 0.05
            if sig:
                significants += 1
            direction = "B > A" if mean_delta > 0 else ("A > B" if mean_delta < 0 else "equal")
            directions.append(direction)

            seed_rows.append({
                "dataset": dataset,
                "comparison_label": label,
                "run_a": condA,
                "run_b": condB,
                "seed": seed,
                "n_pairs_aligned": n_pairs,
                "n_zero_diffs": ties,
                "mean_a": f"{mean_a:.6f}",
                "mean_b": f"{mean_b:.6f}",
                "mean_delta": f"{mean_delta:.6f}",
                "median_delta": f"{median_delta:.6f}",
                "wins_a": wins_a,
                "wins_b": wins_b,
                "ties": ties,
                "p_value": f"{p_val:.6f}",
                "significant_0_05": sig,
                "direction": direction,
            })

            for stem in common:
                delta_rows.append({
                    "dataset": dataset,
                    "comparison_label": label,
                    "seed": seed,
                    "filename": stem,
                    "f1_a": f"{f1_a[stem]:.6f}",
                    "f1_b": f"{f1_b[stem]:.6f}",
                    "delta": f"{f1_b[stem] - f1_a[stem]:.6f}",
                })

            sig_str = "yes" if sig else "no"
            md_lines.append(
                f"| {seed} | {n_pairs} | {mean_a:.6f} | {mean_b:.6f} | "
                f"{mean_delta:+.6f} | {median_delta:+.6f} | "
                f"{wins_b} | {wins_a} | {ties} | {p_val:.6f} | {sig_str} |\n"
            )

            print(f"  seed {seed}: p={p_val:.4f}, {direction}, wins {wins_b}/{wins_a}/{ties}, delta={mean_delta:+.4f}")

        if not skipped and directions:
            consistent = len(set(directions)) == 1
            dir_str = directions[0] if consistent else "mixed"
            summary = f"Significant in {significants}/3 seeds, direction {'consistent' if consistent else 'MIXED'} ({dir_str})"
        else:
            summary = "Incomplete (some seeds skipped)"

        md_lines.append(f"\n**{summary}**\n\n")
        print(f"  -> {summary}\n")

    # Write outputs
    csv_path = OUT_DIR / "wilcoxon_per_seed_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(seed_rows[0].keys()))
        w.writeheader()
        w.writerows(seed_rows)
    print(f"Saved: {csv_path}")

    deltas_path = OUT_DIR / "per_image_deltas_long.csv"
    with open(deltas_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(delta_rows[0].keys()))
        w.writeheader()
        w.writerows(delta_rows)
    print(f"Saved: {deltas_path} ({len(delta_rows)} rows)")

    md_path = OUT_DIR / "wilcoxon_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.writelines(md_lines)
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
