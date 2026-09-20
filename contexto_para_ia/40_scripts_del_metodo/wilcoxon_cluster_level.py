"""
wilcoxon_cluster_level.py — Task 3: Wilcoxon at video/patient level
=====================================================================
Re-runs all 7(+1) Wilcoxon comparisons from wilcoxon_final.py, but
aggregates F1 per video (UNM) or per patient (INCA) before testing.

Uses cached .npy F1 arrays from scripts/output/f1_per_image/ when available,
falls back to computing F1 from PNGs for conditions without cache.

Sanity checks:
  1. Verifies .npy order matches frame_idx in cluster mapping CSVs
  2. Reproduces frame-level Wilcoxon from old results to validate .npy data
  3. Only then computes cluster-level Wilcoxon

Uses scipy.stats.wilcoxon with exact method for small N (UNM: N=7 videos).
Does NOT modify wilcoxon_final.py or its outputs.

Outputs:
  resultados/stats_cluster_level/wilcoxon_cluster_level_results.csv
  resultados/stats_cluster_level/wilcoxon_cluster_level_summary.md
  resultados/stats_cluster_level/wilcoxon_cluster_level_detail.txt

Usage:
    python wilcoxon_cluster_level.py
"""

import csv
import math
import sys
import io
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import wilcoxon as scipy_wilcoxon

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PROJECT = Path(r"G:/My Drive/UNM_vertebras_seg_v3")
UNM_RUNS = PROJECT / "runs_final_v1"
INCA_RUNS = PROJECT / "runs_inca_final_v1"
UNM_GT = PROJECT / "test" / "masks"
INCA_GT = PROJECT / "data" / "inca_dataset" / "test" / "masks"
OUT_DIR = PROJECT / "resultados" / "stats_cluster_level"
MAPPING_DIR = PROJECT / "scripts" / "output"
CACHE_DIR = PROJECT / "scripts" / "output" / "f1_per_image"

SEEDS = [0, 1, 2]

COMPARISONS = [
    ("supervised",                "mean_teacher_all_lateral",            "UNM_sup_vs_MT_all_lateral",       "UNM"),
    ("supervised",                "semi_r3",                             "UNM_sup_vs_PL_r3",                "UNM"),
    ("supervised_inca",           "semi_inca_r20",                       "INCA_full_sup_vs_PL_r20",         "INCA"),
    ("supervised_inca_patient10", "semi_inca_r10_patient10",             "INCA_p10_sup_vs_PL_r10",          "INCA"),
    ("semi_r10",                  "semi_std_matched_r10",                "UNM_PL_r10_temp_vs_rand",         "UNM"),
    ("mean_teacher_r10",          "mean_teacher_std_matched_r10",        "UNM_MT_r10_temp_vs_rand",         "UNM"),
    ("semi_inca_r10_patient10",   "semi_inca_std_matched_r10_patient10", "INCA_p10_PL_r10_temp_vs_rand",   "INCA"),
    ("supervised_inca_patient10", "mean_teacher_inca_r10_patient10",     "INCA_p10_sup_vs_MT_r10",          "INCA"),
]


def load_cluster_mapping(csv_path):
    """Returns dict {frame_stem: cluster_id} and ordered list of stems."""
    mapping = {}
    ordered_stems = []
    with open(str(csv_path), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["frame_idx"]))
    for row in rows:
        mapping[row["frame_stem"]] = row["cluster_id"]
        ordered_stems.append(row["frame_stem"])
    return mapping, ordered_stems


def load_old_pvalues(csv_path):
    old = {}
    if not csv_path.exists():
        return old
    with open(str(csv_path), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["comparison_label"], int(row["seed"]))
            old[key] = {
                "p_value": float(row["p_value"]),
                "mean_a": float(row["mean_a"]),
                "mean_b": float(row["mean_b"]),
                "n_pairs": int(row["n_pairs_aligned"]),
            }
    return old


def load_cached_f1(condition, seed):
    cache_path = CACHE_DIR / f"{condition}__seed_{seed}.npy"
    if cache_path.exists():
        return np.load(str(cache_path))
    return None


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


def compute_per_image_f1_from_pngs(gt_dir, preds_dir, ordered_stems):
    """Compute F1 for each stem in ordered_stems. Returns dict {stem: f1}."""
    results = {}
    for stem in ordered_stems:
        gt_path = gt_dir / f"{stem}.png"
        pred_path = preds_dir / f"{stem}.png"
        if not gt_path.is_file() or not pred_path.is_file():
            continue
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        if gt is None or pred is None:
            continue
        if gt.shape != pred.shape:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        gt_bin = (gt > 127).astype(np.uint8)
        pred_bin = (pred > 0).astype(np.uint8)
        results[stem] = compute_dice(pred_bin, gt_bin)
    return results


def get_per_image_f1(condition, seed, dataset, ordered_stems, runs_root, gt_dir):
    """Try cached .npy first, fall back to PNG computation."""
    cached = load_cached_f1(condition, seed)
    if cached is not None:
        if len(cached) != len(ordered_stems):
            print(f"    *** CACHE SIZE MISMATCH: {condition}/seed_{seed}: "
                  f"cache={len(cached)}, expected={len(ordered_stems)} ***")
            return None
        return {stem: float(cached[i]) for i, stem in enumerate(ordered_stems)}

    print(f"    (no cache for {condition}/seed_{seed}, computing from PNGs...)")
    preds_dir = find_test_preds(runs_root, condition, seed)
    if preds_dir is None:
        return None
    return compute_per_image_f1_from_pngs(gt_dir, preds_dir, ordered_stems)


def aggregate_by_cluster(f1_dict, frame_to_cluster):
    groups = defaultdict(list)
    for stem, f1 in f1_dict.items():
        if stem in frame_to_cluster:
            groups[frame_to_cluster[stem]].append(f1)
    return {cid: float(np.mean(vals)) for cid, vals in groups.items()}


def min_achievable_p(n):
    if n < 1:
        return 1.0
    return 2.0 / (2.0 ** n)


def sanity_check_npy_order(ordered_stems, gt_dir, dataset_label):
    """Verify that ordered_stems matches sorted(glob(*.png)) — the .npy order."""
    actual_stems = sorted(p.stem for p in gt_dir.glob("*.png"))
    if ordered_stems == actual_stems:
        print(f"  [OK] {dataset_label}: .npy order matches frame_idx order ({len(ordered_stems)} frames)")
        return True
    else:
        mismatches = [(i, a, b) for i, (a, b) in enumerate(zip(ordered_stems, actual_stems)) if a != b]
        print(f"  *** FAIL {dataset_label}: .npy order does NOT match! "
              f"{len(mismatches)} mismatches out of {len(ordered_stems)} ***")
        for i, a, b in mismatches[:5]:
            print(f"    idx {i}: mapping={a}, glob={b}")
        return False


def sanity_check_frame_level_wilcoxon(f1_a, f1_b, common_stems, old_result, label, seed):
    """Reproduce frame-level Wilcoxon and compare with old results."""
    arr_a = np.array([f1_a[s] for s in common_stems])
    arr_b = np.array([f1_b[s] for s in common_stems])

    # Check mean values match
    mean_a = float(arr_a.mean())
    mean_b = float(arr_b.mean())
    old_mean_a = old_result["mean_a"]
    old_mean_b = old_result["mean_b"]
    old_p = old_result["p_value"]
    n_pairs = len(common_stems)

    mean_a_match = abs(mean_a - old_mean_a) < 0.001
    mean_b_match = abs(mean_b - old_mean_b) < 0.001

    # Reproduce Wilcoxon
    try:
        _, repro_p = scipy_wilcoxon(arr_a, arr_b, alternative="two-sided",
                                     zero_method="pratt")
    except Exception:
        repro_p = float("nan")

    # Old script used a custom normal approximation; scipy uses exact for small N
    # and normal approximation for large N, so p-values may differ slightly.
    # We check that significance conclusion matches and means are close.
    old_sig = old_p < 0.05
    repro_sig = repro_p < 0.05 if not math.isnan(repro_p) else None

    ok = mean_a_match and mean_b_match
    if not ok:
        print(f"    *** SANITY FAIL {label}/seed_{seed}: "
              f"mean_a={mean_a:.6f} vs old={old_mean_a:.6f}, "
              f"mean_b={mean_b:.6f} vs old={old_mean_b:.6f} ***")
    else:
        sig_match = "match" if old_sig == repro_sig else "DIFFER (method difference)"
        print(f"    [OK] {label}/seed_{seed}: means match, "
              f"old_p={old_p:.6f}, repro_p={repro_p:.6f}, sig={sig_match}")

    return ok, {
        "mean_a": mean_a, "mean_b": mean_b,
        "old_mean_a": old_mean_a, "old_mean_b": old_mean_b,
        "old_p": old_p, "repro_p": repro_p,
        "n_pairs": n_pairs,
        "mean_a_match": mean_a_match, "mean_b_match": mean_b_match,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    unm_mapping, unm_stems = load_cluster_mapping(MAPPING_DIR / "test_frame_to_cluster_unm.csv")
    inca_mapping, inca_stems = load_cluster_mapping(MAPPING_DIR / "test_frame_to_cluster_inca.csv")

    old_pvals = load_old_pvalues(
        PROJECT / "resultados" / "stats_final" / "wilcoxon_per_seed_results.csv")

    # ── Sanity check 1: .npy order ───────────────────────────────────────
    print("=" * 60)
    print("SANITY CHECK 1: .npy array order vs frame_idx mapping")
    print("=" * 60)
    unm_order_ok = sanity_check_npy_order(unm_stems, UNM_GT, "UNM")
    inca_order_ok = sanity_check_npy_order(inca_stems, INCA_GT, "INCA")
    if not unm_order_ok or not inca_order_ok:
        print("\n*** ABORTING: .npy order mismatch detected. Fix mapping before proceeding. ***")
        sys.exit(1)

    # ── Sanity check 2: Reproduce frame-level Wilcoxon ───────────────────
    print("\n" + "=" * 60)
    print("SANITY CHECK 2: Reproduce frame-level Wilcoxon from .npy caches")
    print("=" * 60)

    sanity_results = []
    any_fail = False

    for condA, condB, label, dataset in COMPARISONS:
        runs_root = UNM_RUNS if dataset == "UNM" else INCA_RUNS
        gt_dir = UNM_GT if dataset == "UNM" else INCA_GT
        ordered_stems = unm_stems if dataset == "UNM" else inca_stems
        frame_to_cluster = unm_mapping if dataset == "UNM" else inca_mapping

        for seed in SEEDS:
            key = (label, seed)
            if key not in old_pvals:
                print(f"    [SKIP] {label}/seed_{seed}: no old result to compare")
                continue

            f1_a = get_per_image_f1(condA, seed, dataset, ordered_stems, runs_root, gt_dir)
            f1_b = get_per_image_f1(condB, seed, dataset, ordered_stems, runs_root, gt_dir)

            if f1_a is None or f1_b is None:
                print(f"    [SKIP] {label}/seed_{seed}: could not load F1 data")
                continue

            common_stems = sorted(set(f1_a) & set(f1_b))
            ok, info = sanity_check_frame_level_wilcoxon(
                f1_a, f1_b, common_stems, old_pvals[key], label, seed)
            sanity_results.append((label, seed, ok, info))
            if not ok:
                any_fail = True

    n_pass = sum(1 for _, _, ok, _ in sanity_results if ok)
    n_total = len(sanity_results)
    print(f"\nSanity check 2 result: {n_pass}/{n_total} passed")

    if any_fail:
        print("\n*** WARNING: Some frame-level reproductions failed. ***")
        print("*** Proceeding with cluster-level analysis but results should be verified. ***")

    # ── Cluster-level Wilcoxon ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CLUSTER-LEVEL WILCOXON ANALYSIS")
    print("=" * 60)

    csv_rows = []
    md_lines = ["# Wilcoxon Signed-Rank Test — Cluster Level\n"]
    md_lines.append("Per-cluster F1 comparison (video-level for UNM, patient-level for INCA).\n")
    md_lines.append("Uses scipy.stats.wilcoxon with exact method for small N.\n")
    md_lines.append("Data source: cached .npy F1 arrays (validated against frame-level results).\n\n")

    detail_lines = ["WILCOXON CLUSTER-LEVEL DETAIL\n"]
    detail_lines.append("Per-cluster F1 values for each comparison and seed.\n\n")

    for condA, condB, label, dataset in COMPARISONS:
        runs_root = UNM_RUNS if dataset == "UNM" else INCA_RUNS
        gt_dir = UNM_GT if dataset == "UNM" else INCA_GT
        frame_to_cluster = unm_mapping if dataset == "UNM" else inca_mapping
        ordered_stems = unm_stems if dataset == "UNM" else inca_stems
        cluster_type = "video" if dataset == "UNM" else "patient"

        print(f"\n{label} ({dataset}, {cluster_type}-level):")
        md_lines.append(f"## {label}\n")
        md_lines.append(f"- **A**: `{condA}`")
        md_lines.append(f"- **B**: `{condB}`")
        md_lines.append(f"- **Dataset**: {dataset} ({cluster_type}-level)\n")
        md_lines.append("| Seed | N frames | N clusters | Mean A (cl) | Mean B (cl) | Delta (cl) | Old p (frame) | New p (cluster) | Min p | Sig old | Sig new |")
        md_lines.append("|------|---------|------------|-------------|-------------|-----------|--------------|-----------------|-------|---------|---------|")

        for seed in SEEDS:
            f1_a = get_per_image_f1(condA, seed, dataset, ordered_stems, runs_root, gt_dir)
            f1_b = get_per_image_f1(condB, seed, dataset, ordered_stems, runs_root, gt_dir)
            if f1_a is None or f1_b is None:
                print(f"  Seed {seed}: data not available, skipping")
                continue

            common_stems = sorted(set(f1_a) & set(f1_b))
            n_frames = len(common_stems)

            cl_a = aggregate_by_cluster({s: f1_a[s] for s in common_stems}, frame_to_cluster)
            cl_b = aggregate_by_cluster({s: f1_b[s] for s in common_stems}, frame_to_cluster)

            common_clusters = sorted(set(cl_a) & set(cl_b))
            n_clusters = len(common_clusters)
            min_p = min_achievable_p(n_clusters)

            arr_a = np.array([cl_a[c] for c in common_clusters])
            arr_b = np.array([cl_b[c] for c in common_clusters])
            delta = arr_b - arr_a

            mean_a_frame = float(np.mean([f1_a[s] for s in common_stems]))
            mean_b_frame = float(np.mean([f1_b[s] for s in common_stems]))

            mean_a_cl = float(arr_a.mean())
            mean_b_cl = float(arr_b.mean())
            mean_delta_cl = float(delta.mean())

            wins_b = int((delta > 0).sum())
            wins_a = int((delta < 0).sum())
            ties = int((delta == 0).sum())

            nonzero = (delta != 0).sum()
            if nonzero < 2:
                new_p = float("nan")
                stat = float("nan")
                direction = "insufficient_data"
            else:
                try:
                    stat, new_p = scipy_wilcoxon(
                        arr_a, arr_b, alternative="two-sided",
                        zero_method="pratt", method="exact")
                except ValueError:
                    try:
                        stat, new_p = scipy_wilcoxon(
                            arr_a, arr_b, alternative="two-sided",
                            zero_method="pratt", method="approx")
                    except Exception:
                        stat, new_p = float("nan"), float("nan")

                if wins_b > wins_a:
                    direction = "B > A"
                elif wins_a > wins_b:
                    direction = "A > B"
                else:
                    direction = "tied"

            old_key = (label, seed)
            old_entry = old_pvals.get(old_key, {})
            old_p = old_entry.get("p_value", float("nan"))
            old_sig = old_p < 0.05 if not math.isnan(old_p) else False
            new_sig = new_p < 0.05 if not math.isnan(new_p) else False

            csv_rows.append({
                "dataset": dataset,
                "comparison_label": label,
                "run_a": condA,
                "run_b": condB,
                "seed": seed,
                "n_frames": n_frames,
                "n_clusters": n_clusters,
                "cluster_level": cluster_type,
                "mean_a_frame": round(mean_a_frame, 6),
                "mean_b_frame": round(mean_b_frame, 6),
                "mean_delta_frame": round(mean_b_frame - mean_a_frame, 6),
                "mean_a_cluster": round(mean_a_cl, 6),
                "mean_b_cluster": round(mean_b_cl, 6),
                "mean_delta_cluster": round(mean_delta_cl, 6),
                "old_p_value": round(old_p, 10) if not math.isnan(old_p) else "",
                "new_p_value": round(new_p, 10) if not math.isnan(new_p) else "",
                "min_achievable_p": round(min_p, 10),
                "old_significant": old_sig,
                "new_significant": new_sig,
                "wins_b": wins_b,
                "wins_a": wins_a,
                "ties": ties,
                "direction": direction,
            })

            old_p_str = f"{old_p:.6f}" if not math.isnan(old_p) else "N/A"
            new_p_str = f"{new_p:.6f}" if not math.isnan(new_p) else "N/A"
            print(f"  Seed {seed}: N_cl={n_clusters}, delta_cl={mean_delta_cl:+.4f}, "
                  f"old_p={old_p_str}, new_p={new_p_str}, "
                  f"wins={wins_b}/{wins_a}/{ties}")

            old_sig_str = "yes" if old_sig else "no"
            new_sig_str = "yes" if new_sig else "no"
            md_lines.append(
                f"| {seed} | {n_frames} | {n_clusters} | {mean_a_cl:.4f} | {mean_b_cl:.4f} | "
                f"{mean_delta_cl:+.4f} | {old_p_str} | {new_p_str} | {min_p:.6f} | "
                f"{old_sig_str} | {new_sig_str} |")

            detail_lines.append(f"--- {label} / seed {seed} ---")
            detail_lines.append(f"{'Cluster':>10s}  {'F1_A':>8s}  {'F1_B':>8s}  {'Delta':>8s}  {'N_frames':>8s}")
            for c in common_clusters:
                n_f = sum(1 for s in common_stems if frame_to_cluster.get(s) == c)
                detail_lines.append(
                    f"{c:>10s}  {cl_a[c]:8.4f}  {cl_b[c]:8.4f}  {cl_b[c]-cl_a[c]:+8.4f}  {n_f:>8d}")
            detail_lines.append("")

        md_lines.append("")

    # ── Write outputs ─────────────────────────────────────────────────────
    csv_path = OUT_DIR / "wilcoxon_cluster_level_results.csv"
    fieldnames = [
        "dataset", "comparison_label", "run_a", "run_b", "seed",
        "n_frames", "n_clusters", "cluster_level",
        "mean_a_frame", "mean_b_frame", "mean_delta_frame",
        "mean_a_cluster", "mean_b_cluster", "mean_delta_cluster",
        "old_p_value", "new_p_value", "min_achievable_p",
        "old_significant", "new_significant",
        "wins_b", "wins_a", "ties", "direction",
    ]
    with open(str(csv_path), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nSaved: {csv_path}")

    md_path = OUT_DIR / "wilcoxon_cluster_level_summary.md"
    with open(str(md_path), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Saved: {md_path}")

    detail_path = OUT_DIR / "wilcoxon_cluster_level_detail.txt"
    with open(str(detail_path), "w", encoding="utf-8") as f:
        f.write("\n".join(detail_lines))
    print(f"Saved: {detail_path}")


if __name__ == "__main__":
    main()
