"""
compute_inter_rater_inca.py
============================
Computes inter-rater agreement (Dice coefficient) between the two independent
annotations of each INCA frame.

For each of the 968 frames in video_frame_metadata.csv, identifies which 2 of
5 annotators (AM, BB, CS, VC, VR) produced Mask.tif files. Loads both TIFs
with symmetric processing (PIL convert L, binarize >127, detect inversion),
computes Dice, and reports results globally, per annotator pair, and per split.

Runs 3 verification steps before batch processing:
  4a: compare process_mask(selected TIF) vs dataset PNG for 10 frames
  4b: visual check of 3 frames (normal, inverted, palette)
  4c: anomaly scan of all second-annotator TIFs

Output: resultados/inter_rater_inca/
  - inter_rater_per_frame.csv
  - inter_rater_summary.csv
  - inter_rater_report.md
  - excluded_single_annotator.csv

Usage:
    python compute_inter_rater_inca.py
"""

import csv
import glob
import math
import sys
import io
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PROJECT = Path(r"G:/My Drive/UNM_vertebras_seg_v3")
METADATA_CSV = PROJECT / "data" / "video_frame_metadata.csv"
INCA_DATASET = PROJECT / "data" / "inca_dataset"
OUT_DIR = PROJECT / "resultados" / "inter_rater_inca"

# Find rotulos dir (accented name — glob from relative path to avoid encoding issues)
import os as _os
_cwd = _os.getcwd()
_os.chdir(str(PROJECT))
_rotulos_candidates = glob.glob("data/rotulos/Anota*/")
_os.chdir(_cwd)
ROTULOS = (PROJECT / _rotulos_candidates[0]) if _rotulos_candidates else None

LABELERS = ["AM", "BB", "CS", "VC", "VR"]
SPLITS = ["train", "val", "test"]


# =============================================================================
# Core functions
# =============================================================================

def process_mask(tif_path):
    """Same processing as notebook 03: convert to grayscale, binarize, detect inversion."""
    mask = Image.open(str(tif_path)).convert("L")
    arr = np.array(mask)
    arr = np.where(arr > 127, 255, 0).astype(np.uint8)

    # Detect inversion: vertebrae occupy 1-5% of image.
    # If foreground > 10%, the mask is inverted.
    fg_pct = (arr == 255).sum() / arr.size
    if fg_pct > 0.10:
        arr = 255 - arr
    return arr


def compute_dice(mask_a, mask_b):
    """Dice between two binary masks (0/255 uint8)."""
    a = (mask_a > 127).astype(np.float32)
    b = (mask_b > 127).astype(np.float32)
    if a.sum() == 0 and b.sum() == 0:
        return 1.0
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    intersection = (a * b).sum()
    return float(2.0 * intersection / (a.sum() + b.sum()))


def get_split(vf):
    """Determine which split a video_frame belongs to by checking file existence."""
    for split in SPLITS:
        if (INCA_DATASET / split / "masks" / f"{vf}.png").is_file():
            return split
    return "unknown"


def get_tif_path(batch, labeler, vf):
    return ROTULOS / f"batch-{batch}" / labeler / vf / "Mask.tif"


# =============================================================================
# Step 1: Identify annotators for each frame
# =============================================================================

def step1_identify_annotators():
    print("=" * 70)
    print("STEP 1: Identify annotators for each frame")
    print("=" * 70)

    with open(METADATA_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    frames_2ann = []  # (vf, batch, selected, other, split)
    frames_1ann = []  # (vf, batch, labeler, split)

    for idx, row in enumerate(rows):
        batch = row["batch"]
        vf = row["video_frame"]
        selected = row["selected_labeler"]
        split = get_split(vf)

        found = [lab for lab in LABELERS if get_tif_path(batch, lab, vf).is_file()]

        if len(found) == 2:
            other = [l for l in found if l != selected][0]
            frames_2ann.append((vf, batch, selected, other, split))
        elif len(found) == 1:
            frames_1ann.append((vf, batch, found[0], split))
        else:
            print(f"  WARNING: {vf} batch {batch} has {len(found)} annotators: {found}")

        if (idx + 1) % 100 == 0:
            print(f"  step1 progress: {idx + 1}/{len(rows)} frames scanned")

    print(f"  Frames with 2 annotators: {len(frames_2ann)}")
    print(f"  Frames with 1 annotator:  {len(frames_1ann)}")
    return frames_2ann, frames_1ann


# =============================================================================
# Step 4a: Validate process_mask vs dataset PNGs
# =============================================================================

def step4a_validate(frames_2ann):
    print("\n" + "=" * 70)
    print("VERIFICATION 4a: process_mask() vs dataset PNGs (10 frames)")
    print("=" * 70)

    # Known special frames
    known_inverted = {"v10_f22", "v11_f41", "v27_f190", "v4_f46"}
    known_antialiased = {"v57_f66"}

    # Pick 10 frames: include known problematic ones + random normal
    test_frames = []
    for vf, batch, selected, other, split in frames_2ann:
        if vf in known_inverted and len([f for f in test_frames if f[0] in known_inverted]) < 2:
            test_frames.append((vf, batch, selected, split))
        elif vf in known_antialiased:
            test_frames.append((vf, batch, selected, split))
        elif len(test_frames) < 10 and vf not in known_inverted and vf not in known_antialiased:
            test_frames.append((vf, batch, selected, split))
        if len(test_frames) >= 10:
            break

    all_ok = True
    for vf, batch, selected, split in test_frames:
        tif_path = get_tif_path(batch, selected, vf)
        png_path = INCA_DATASET / split / "masks" / f"{vf}.png"

        processed = process_mask(tif_path)
        dataset_png = np.array(Image.open(str(png_path)).convert("L"))
        dataset_bin = np.where(dataset_png > 127, 255, 0).astype(np.uint8)

        match = np.array_equal(processed, dataset_bin)
        special = ""
        if vf in known_inverted:
            special = " [INVERTED]"
        elif vf in known_antialiased:
            special = " [ANTI-ALIASED]"

        status = "OK" if match else "MISMATCH"
        if not match:
            diff_count = (processed != dataset_bin).sum()
            print(f"  {status}: {vf}{special} — {diff_count} pixels differ")
            all_ok = False
        else:
            print(f"  {status}: {vf}{special}")

    if not all_ok:
        print("\n  *** VERIFICATION 4a FAILED — process_mask() does not match dataset PNGs ***")
        print("  Stopping. Fix process_mask() before proceeding.")
        sys.exit(1)

    print("  All 10 frames match. process_mask() is correct.")
    return True


# =============================================================================
# Step 4b: Visual check of 3 frames
# =============================================================================

def step4b_visual(frames_2ann):
    print("\n" + "=" * 70)
    print("VERIFICATION 4b: Visual check of 3 frames")
    print("=" * 70)

    targets = {"v100_f10": "normal", "v10_f22": "inverted"}
    # Find a palette frame by checking TIF mode
    palette_frame = None

    checks = []
    for vf, batch, selected, other, split in frames_2ann:
        if vf in targets:
            checks.append((vf, batch, selected, other, split, targets[vf]))
        if palette_frame is None and len(checks) < 3:
            tif_b = get_tif_path(batch, other, vf)
            try:
                mode = Image.open(str(tif_b)).mode
                if mode == "P":
                    checks.append((vf, batch, selected, other, split, "palette"))
                    palette_frame = vf
            except Exception:
                pass

    # If no palette found, just use a third normal frame
    if len(checks) < 3:
        for vf, batch, selected, other, split in frames_2ann:
            if vf not in [c[0] for c in checks]:
                checks.append((vf, batch, selected, other, split, "normal(fallback)"))
                break

    for vf, batch, selected, other, split, label in checks[:3]:
        tif_a = get_tif_path(batch, selected, vf)
        tif_b = get_tif_path(batch, other, vf)
        mask_a = process_mask(tif_a)
        mask_b = process_mask(tif_b)
        dice = compute_dice(mask_a, mask_b)
        fg_a = (mask_a == 255).sum() / mask_a.size * 100
        fg_b = (mask_b == 255).sum() / mask_b.size * 100

        print(f"\n  {vf} [{label}] ({selected} vs {other}):")
        print(f"    Mask A: shape={mask_a.shape}, unique={np.unique(mask_a).tolist()}, fg={fg_a:.2f}%")
        print(f"    Mask B: shape={mask_b.shape}, unique={np.unique(mask_b).tolist()}, fg={fg_b:.2f}%")
        print(f"    Same resolution: {mask_a.shape == mask_b.shape}")
        print(f"    Dice: {dice:.4f}")


# =============================================================================
# Step 4c: Anomaly scan on all second-annotator TIFs
# =============================================================================

def step4c_anomalies(frames_2ann):
    print("\n" + "=" * 70)
    print("VERIFICATION 4c: Anomaly scan (all 901 second-annotator TIFs)")
    print("=" * 70)

    n_palette = 0
    n_nonbinary = 0
    n_inverted = 0
    anomaly_list = []

    for idx_4c, (vf, batch, selected, other, split) in enumerate(frames_2ann):
        tif_path = get_tif_path(batch, other, vf)
        if (idx_4c + 1) % 100 == 0:
            print(f"  4c progress: {idx_4c + 1}/{len(frames_2ann)}")
        try:
            img = Image.open(str(tif_path))
            if img.mode == "P":
                n_palette += 1
                anomaly_list.append((vf, other, "palette"))

            arr = np.array(img.convert("L"))
            unique_vals = np.unique(arr)
            if not set(unique_vals.tolist()).issubset({0, 255}):
                n_nonbinary += 1
                anomaly_list.append((vf, other, f"non-binary values: {unique_vals[:10]}"))

            bin_arr = np.where(arr > 127, 255, 0).astype(np.uint8)
            fg_pct = (bin_arr == 255).sum() / bin_arr.size
            if fg_pct > 0.10:
                n_inverted += 1
                anomaly_list.append((vf, other, f"inverted (fg={fg_pct:.1%})"))
        except Exception as e:
            anomaly_list.append((vf, other, f"ERROR: {e}"))

    print(f"  Mode P (palette): {n_palette}")
    print(f"  Non-binary values: {n_nonbinary}")
    print(f"  Inverted (fg>10%): {n_inverted}")
    if anomaly_list:
        print(f"\n  Anomalies ({len(anomaly_list)}):")
        for vf, lab, desc in anomaly_list[:20]:
            print(f"    {vf} ({lab}): {desc}")
        if len(anomaly_list) > 20:
            print(f"    ... and {len(anomaly_list) - 20} more")
    else:
        print("  No anomalies found.")

    print("\n  NOTE: all anomalies are handled by process_mask() (convert L, binarize, invert).")
    return anomaly_list


# =============================================================================
# Step 5-6: Batch compute + report
# =============================================================================

def step5_batch_compute(frames_2ann):
    print("\n" + "=" * 70)
    print("STEP 5: Computing Dice for 901 frames")
    print("=" * 70)

    results = []
    n_resolution_mismatch = 0

    for i, (vf, batch, selected, other, split) in enumerate(frames_2ann):
        tif_a = get_tif_path(batch, selected, vf)
        tif_b = get_tif_path(batch, other, vf)

        mask_a = process_mask(tif_a)
        mask_b = process_mask(tif_b)

        if mask_a.shape != mask_b.shape:
            n_resolution_mismatch += 1
            print(f"  SKIP {vf}: resolution mismatch {mask_a.shape} vs {mask_b.shape}")
            continue

        dice = compute_dice(mask_a, mask_b)
        results.append({
            "video_frame": vf,
            "split": split,
            "labeler_A": selected,
            "labeler_B": other,
            "resolution_A": f"{mask_a.shape[0]}x{mask_a.shape[1]}",
            "resolution_B": f"{mask_b.shape[0]}x{mask_b.shape[1]}",
            "dice": dice,
        })

        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{len(frames_2ann)}")

    print(f"  Done. {len(results)} frames computed, {n_resolution_mismatch} skipped (resolution mismatch).")
    return results


def step6_report(results, frames_1ann):
    print("\n" + "=" * 70)
    print("STEP 6: Generating reports")
    print("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Per-frame CSV
    csv_path = OUT_DIR / "inter_rater_per_frame.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_frame", "split", "labeler_A", "labeler_B",
                                          "resolution_A", "resolution_B", "dice"])
        w.writeheader()
        w.writerows(results)
    print(f"  Saved: {csv_path} ({len(results)} rows)")

    # Excluded CSV
    excl_path = OUT_DIR / "excluded_single_annotator.csv"
    with open(excl_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_frame", "batch", "labeler", "split"])
        w.writeheader()
        for vf, batch, lab, split in frames_1ann:
            w.writerow({"video_frame": vf, "batch": batch, "labeler": lab, "split": split})
    print(f"  Saved: {excl_path} ({len(frames_1ann)} rows)")

    # Compute statistics
    dices = [r["dice"] for r in results]
    arr = np.array(dices)

    def stats(values, label):
        a = np.array(values)
        n = len(a)
        if n == 0:
            return {"group": label, "n_frames": 0, "dice_mean": None, "dice_std": None,
                    "dice_median": None, "dice_p5": None, "dice_p95": None}
        return {
            "group": label,
            "n_frames": n,
            "dice_mean": f"{a.mean():.4f}",
            "dice_std": f"{a.std():.4f}",
            "dice_median": f"{np.median(a):.4f}",
            "dice_p5": f"{np.percentile(a, 5):.4f}",
            "dice_p95": f"{np.percentile(a, 95):.4f}",
        }

    summary_rows = []

    # Global
    summary_rows.append(stats(dices, "global"))

    # Per split
    for split in SPLITS:
        split_dices = [r["dice"] for r in results if r["split"] == split]
        summary_rows.append(stats(split_dices, f"split_{split}"))

    # Per pair
    pair_dices = defaultdict(list)
    for r in results:
        pair = tuple(sorted([r["labeler_A"], r["labeler_B"]]))
        pair_dices[pair].append(r["dice"])
    for pair in sorted(pair_dices):
        summary_rows.append(stats(pair_dices[pair], f"pair_{pair[0]}-{pair[1]}"))

    # Summary CSV
    sum_path = OUT_DIR / "inter_rater_summary.csv"
    with open(sum_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["group", "n_frames", "dice_mean", "dice_std",
                                          "dice_median", "dice_p5", "dice_p95"])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"  Saved: {sum_path}")

    # Anomalies (Dice < 0.5)
    low_dice = [(r["video_frame"], r["split"], r["labeler_A"], r["labeler_B"], r["dice"])
                for r in results if r["dice"] < 0.5]

    # Markdown report
    md_lines = []
    md_lines.append("# Inter-Rater Agreement: INCA Dataset\n\n")

    g = summary_rows[0]
    md_lines.append(f"## Global\n\n")
    md_lines.append(f"- **N frames**: {g['n_frames']} (of 968 total; {len(frames_1ann)} excluded with only 1 annotator)\n")
    md_lines.append(f"- **Dice mean +/- std**: {g['dice_mean']} +/- {g['dice_std']}\n")
    md_lines.append(f"- **Median**: {g['dice_median']}\n")
    md_lines.append(f"- **p5 / p95**: {g['dice_p5']} / {g['dice_p95']}\n\n")

    md_lines.append("## Per split\n\n")
    md_lines.append("| Split | N | Dice mean | Dice std | Median |\n")
    md_lines.append("|-------|---|-----------|----------|--------|\n")
    for row in summary_rows[1:4]:
        md_lines.append(f"| {row['group'].replace('split_','')} | {row['n_frames']} | {row['dice_mean']} | {row['dice_std']} | {row['dice_median']} |\n")

    md_lines.append("\n## Per annotator pair\n\n")
    md_lines.append("| Pair | N | Dice mean | Dice std | Median |\n")
    md_lines.append("|------|---|-----------|----------|--------|\n")
    for row in summary_rows[4:]:
        md_lines.append(f"| {row['group'].replace('pair_','')} | {row['n_frames']} | {row['dice_mean']} | {row['dice_std']} | {row['dice_median']} |\n")

    md_lines.append(f"\n## Percentiles (global)\n\n")
    md_lines.append(f"| p5 | p25 | p50 | p75 | p95 |\n")
    md_lines.append(f"|-----|-----|-----|-----|-----|\n")
    p = np.percentile(arr, [5, 25, 50, 75, 95])
    md_lines.append(f"| {p[0]:.4f} | {p[1]:.4f} | {p[2]:.4f} | {p[3]:.4f} | {p[4]:.4f} |\n")

    if low_dice:
        md_lines.append(f"\n## Anomalies (Dice < 0.5): {len(low_dice)} frames\n\n")
        md_lines.append("| Frame | Split | A | B | Dice |\n")
        md_lines.append("|-------|-------|---|---|------|\n")
        for vf, split, la, lb, d in sorted(low_dice, key=lambda x: x[4]):
            md_lines.append(f"| {vf} | {split} | {la} | {lb} | {d:.4f} |\n")
    else:
        md_lines.append("\n## Anomalies (Dice < 0.5): none\n")

    md_lines.append(f"\n## Excluded frames (1 annotator only): {len(frames_1ann)}\n\n")
    md_lines.append(f"See `excluded_single_annotator.csv` for full list.\n")

    md_path = OUT_DIR / "inter_rater_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.writelines(md_lines)
    print(f"  Saved: {md_path}")

    # Console summary
    print(f"\n  === RESULTS ===")
    print(f"  Global Dice: {g['dice_mean']} +/- {g['dice_std']} (n={g['n_frames']})")
    print(f"  Median: {g['dice_median']}, p5={g['dice_p5']}, p95={g['dice_p95']}")
    if low_dice:
        print(f"  Frames with Dice < 0.5: {len(low_dice)}")
    print(f"  Excluded (1 annotator): {len(frames_1ann)}")


# =============================================================================
# Main
# =============================================================================

def main():
    if ROTULOS is None or not ROTULOS.is_dir():
        print(f"ERROR: rotulos dir not found. Searched in {PROJECT / 'data' / 'rotulos'}")
        sys.exit(1)
    print(f"Rotulos dir: {ROTULOS}")

    # Step 1
    frames_2ann, frames_1ann = step1_identify_annotators()

    # Verifications
    step4a_validate(frames_2ann)
    step4b_visual(frames_2ann)
    step4c_anomalies(frames_2ann)

    # Batch compute
    results = step5_batch_compute(frames_2ann)

    # Report
    step6_report(results, frames_1ann)

    print("\nDone.")


if __name__ == "__main__":
    main()
