"""
prep_inca_label_fractions.py
============================
Creates nested labeled-subset stem files for the INCA label-fraction ablation.

Adapted from prep_label_fractions.py (UNM). The algorithm is IDENTICAL to the
UNM version. Only input/output paths differ. See the methodology note in
  resultados/contexto/label_fractions_methodology.md
for the full rationale and caveats.

For each fraction in [25%, 50%, 75%], samples that fraction of training frames
from EACH video (stratified), using a SINGLE random permutation per video so
that subsets are nested:

    frac_25 subset of frac_50 subset of frac_75 subset of full

Algorithm (per video):
  1. Sort that video's stems lexicographically.
  2. Shuffle using a single RNG (SUBSET_SEED=0), advancing state per video.
  3. Take prefix slices: k = max(1, ceil(n * frac)).
  4. frac_25 = first k25 stems, frac_50 = first k50, frac_75 = first k75.

Because ALL prefix slices come from the same permutation, smaller sets are
always subsets of larger ones.

Stratification is per VIDEO, matching UNM. INCA has 15 multi-video patients,
but each patient retains a proportional share of their annotation in every
fraction — we stratify by video for consistency with the UNM protocol.

Outputs:
    {img_root}/label_fractions/frac_25/stems.txt
    {img_root}/label_fractions/frac_50/stems.txt
    {img_root}/label_fractions/frac_75/stems.txt
    {img_root}/label_fractions/summary.csv

Usage (local Windows):
    python prep_inca_label_fractions.py \
        --img_root "G:/My Drive/UNM_vertebras_seg_v3/data/inca_dataset"

Usage (Colab):
    !python prep_inca_label_fractions.py \
        --img_root /content/drive/MyDrive/UNM_vertebras_seg_v3/data/inca_dataset
"""

import argparse
import csv
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path


FRACS       = [("25", 0.25), ("50", 0.50), ("75", 0.75)]
SUBSET_SEED = 0
DEFAULT_IMG_ROOT = r"G:/My Drive/UNM_vertebras_seg_v3/data/inca_dataset"


def build_by_video(train_images_dir: Path):
    """Return {video_id: [stem, ...]} sorted lexicographically within each video.

    Regex works for both padded (v025_f603) and unpadded (v1_f80) video IDs
    because `\\d+` matches any number of digits.
    """
    pattern = re.compile(r"^v(\d+)_f(\d+)\.png$")
    by_video = defaultdict(list)
    for p in sorted(train_images_dir.glob("*.png")):
        m = pattern.match(p.name)
        if not m:
            print(f"  [WARN] skipping unexpected filename: {p.name}")
            continue
        vid = m.group(1)
        by_video[vid].append(p.stem)
    for vid in by_video:
        by_video[vid].sort()
    return dict(by_video)


def build_permutations(by_video):
    """
    Return {video_id: shuffled_list} using ONE RNG that advances its state
    across videos (sorted order). Each video gets a distinct permutation
    because the RNG state after shuffling video i feeds into video i+1.
    """
    rng = random.Random(SUBSET_SEED)
    perms = {}
    for vid in sorted(by_video):
        perm = by_video[vid][:]
        rng.shuffle(perm)
        perms[vid] = perm
    return perms


def select_fraction(perms, frac):
    """Return sorted list of stems selected for a given fraction."""
    chosen = []
    for vid in sorted(perms):
        k = max(1, math.ceil(len(perms[vid]) * frac))
        chosen.extend(perms[vid][:k])
    return sorted(chosen)


def verify_nesting(sets, full_set):
    """Assert nesting and print confirmation."""
    s25, s50, s75 = sets["25"], sets["50"], sets["75"]

    ok_25_50 = s25 <= s50
    ok_50_75 = s50 <= s75
    ok_75_full = s75 <= full_set
    ok_25_full = s25 <= full_set

    print("\n-- Nesting verification ------------------------------")
    print(f"  frac_25 subset of frac_50  : {'OK' if ok_25_50  else 'FAIL  <-- CHECK'}")
    print(f"  frac_50 subset of frac_75  : {'OK' if ok_50_75  else 'FAIL  <-- CHECK'}")
    print(f"  frac_75 subset of full     : {'OK' if ok_75_full else 'FAIL  <-- CHECK'}")
    print(f"  frac_25 subset of full     : {'OK' if ok_25_full else 'FAIL  <-- CHECK'}")

    if not (ok_25_50 and ok_50_75 and ok_75_full and ok_25_full):
        print("\n[ERROR] Nesting check FAILED — do not use these files.")
        sys.exit(1)
    else:
        print("\n  All nesting checks passed.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate nested labeled-subset stem files for INCA label-fraction ablation."
    )
    parser.add_argument(
        "--img_root",
        type=str,
        default=DEFAULT_IMG_ROOT,
        help=f"Path to INCA dataset root (contains train/images/). Default: {DEFAULT_IMG_ROOT}",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for label_fractions/. Defaults to img_root.",
    )
    args = parser.parse_args()

    img_root = Path(args.img_root)

    out_root = Path(args.out_dir) if args.out_dir else img_root
    label_fractions_dir = out_root / "label_fractions"

    train_images_dir = img_root / "train" / "images"
    if not train_images_dir.is_dir():
        print(f"[ERROR] train images dir not found: {train_images_dir}")
        sys.exit(1)

    print(f"Scanning: {train_images_dir}")
    by_video = build_by_video(train_images_dir)
    full_stems = set(stem for stems in by_video.values() for stem in stems)
    n_total_full = len(full_stems)
    print(f"  {len(by_video)} videos, {n_total_full} labeled frames total")

    perms = build_permutations(by_video)

    selected = {}
    for tag, frac in FRACS:
        selected[tag] = select_fraction(perms, frac)

    for tag, _ in FRACS:
        out_dir = label_fractions_dir / f"frac_{tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        stems_path = out_dir / "stems.txt"
        stems_path.write_text("\n".join(selected[tag]), encoding="utf-8")
        print(f"  frac_{tag}: {len(selected[tag])} stems -> {stems_path}")

    summary_path = label_fractions_dir / "summary.csv"
    rows = []
    for tag, frac in FRACS:
        sel_set = set(selected[tag])
        for vid in sorted(by_video):
            n_avail = len(by_video[vid])
            n_sel   = sum(1 for s in by_video[vid] if s in sel_set)
            rows.append({
                "fraction":    f"frac_{tag}",
                "frac_value":  frac,
                "n_total":     len(selected[tag]),
                "video_id":    vid,
                "n_available": n_avail,
                "n_selected":  n_sel,
            })

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "fraction", "frac_value", "n_total",
            "video_id", "n_available", "n_selected",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"  Summary CSV -> {summary_path}")

    verify_nesting(
        {tag: set(selected[tag]) for tag, _ in FRACS},
        full_stems,
    )

    print("\n-- Counts --------------------------------------------")
    print(f"  Full train set : {n_total_full}")
    for tag, frac in FRACS:
        print(f"  frac_{tag} ({int(frac*100):3d}%)  : {len(selected[tag])}")
    print("------------------------------------------------------")
    print("Done.")


if __name__ == "__main__":
    main()
