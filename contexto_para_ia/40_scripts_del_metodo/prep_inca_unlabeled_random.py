"""
prep_inca_unlabeled_random.py

Builds random matched unlabeled pools as controls for the temporal-neighbor pools
in the INCA dataset.

For each reference radius, samples the same number of unlabeled frames per video
as the temporal pool, drawn uniformly at random from unlabeled_all/.

Unlike the UNM version:
- No AP cutoff (INCA has no projection change)
- No video decoding; copies PNGs directly from unlabeled_all/
- Frame IDs in filenames are 1-based

Usage:
    python prep_inca_unlabeled_random.py            # default seed=42
    python prep_inca_unlabeled_random.py --seed 7   # custom seed
    python prep_inca_unlabeled_random.py --root /content/drive/MyDrive/UNM_vertebras_seg_v3/data/inca_dataset
"""

import argparse
import glob
import json
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


# ─────────────────────────────────────────────
# TOP-LEVEL CONFIG
# ─────────────────────────────────────────────

OUT_ROOT       = r"G:\My Drive\UNM_vertebras_seg_v3\data\inca_dataset"
DEFAULT_SEED   = 42

REFERENCE_TAGS = ["r3", "r5", "r7", "r10", "r15", "r20"]


# ─────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────

def dual_print(*args, **kwargs):
    print(*args, **kwargs)


def ensure_dir(path: str):
    if path is None or path == "":
        raise ValueError(f"Invalid path: {path}")
    os.makedirs(path, exist_ok=True)


def load_labeled_from_folders(out_root: str):
    """Discover labeled frames from train/val/test mask folders."""
    split_map = {
        "train": os.path.join(out_root, "train", "masks"),
        "val":   os.path.join(out_root, "val",   "masks"),
        "test":  os.path.join(out_root, "test",  "masks"),
    }

    labeled_by_video = defaultdict(set)
    labeled_keys     = set()
    train_videos     = set()
    val_videos       = set()
    test_videos      = set()

    for split, mdir in split_map.items():
        if not os.path.isdir(mdir):
            dual_print(f"[WARN] Mask folder missing for '{split}': {mdir}")
            continue

        for fname in os.listdir(mdir):
            if not fname.lower().endswith(".png"):
                continue
            stem = Path(fname).stem
            if "_f" not in stem:
                continue

            try:
                vpart, fpart = stem.split("_f")
                vid = vpart.replace("v", "")
                fid = int(fpart)
            except Exception:
                dual_print(f"[WARN] Unexpected filename (skipped): {fname}")
                continue

            labeled_keys.add(stem)
            labeled_by_video[vid].add(fid)

            if split == "train":
                train_videos.add(vid)
            elif split == "val":
                val_videos.add(vid)
            elif split == "test":
                test_videos.add(vid)

    dual_print("=== Splits detected from folders ===")
    dual_print(f"TRAIN videos : {sorted(train_videos)}")
    dual_print(f"VAL videos   : {sorted(val_videos)}")
    dual_print(f"TEST videos  : {sorted(test_videos)}")
    dual_print(f"Total labeled frames (all splits): {len(labeled_keys)}")

    return labeled_keys, labeled_by_video, train_videos, val_videos, test_videos


# ─────────────────────────────────────────────
# CORE LOGIC
# ─────────────────────────────────────────────

def _count_ref_frames_per_video(ref_dir: str, train_videos: set) -> dict:
    """Count frames per video in the reference temporal pool."""
    counts = {}
    pattern = re.compile(r"^v(\d+)_f\d+\.png$", re.IGNORECASE)
    for fname in os.listdir(ref_dir):
        m = pattern.match(fname)
        if not m:
            continue
        vid = m.group(1)
        if vid in train_videos:
            counts[vid] = counts.get(vid, 0) + 1
    return counts


def _index_unlabeled_all(unlabeled_all_dir: str) -> dict:
    """Index all available frames in unlabeled_all/ by video ID.
    Returns: {vid: {stem: filepath, ...}, ...}
    """
    by_video = defaultdict(dict)
    pattern = re.compile(r"^v(\d+)_f(\d+)\.png$", re.IGNORECASE)
    for fname in os.listdir(unlabeled_all_dir):
        m = pattern.match(fname)
        if not m:
            continue
        vid = m.group(1)
        stem = Path(fname).stem  # e.g. v1_f80
        by_video[vid][stem] = os.path.join(unlabeled_all_dir, fname)
    return dict(by_video)


def build_random_pool(ref_tag: str, seed: int, out_root: str):
    """Build one random matched pool for a given reference tag."""

    ref_dir = os.path.join(out_root, f"unlabeled_{ref_tag}")
    out_dir = os.path.join(out_root, f"unlabeled_random_{ref_tag}")
    manifest_path = os.path.join(out_root, f"unlabeled_random_{ref_tag}_manifest.json")
    unlabeled_all_dir = os.path.join(out_root, "unlabeled_all")

    dual_print(f"\n{'='*60}")
    dual_print(f"Building random matched pool for: {ref_tag}")
    dual_print(f"  Reference dir    : {ref_dir}")
    dual_print(f"  Unlabeled source : {unlabeled_all_dir}")
    dual_print(f"  Output dir       : {out_dir}")
    dual_print(f"  Random seed      : {seed}")
    dual_print(f"  AP cutoffs       : N/A (INCA)")

    if not os.path.isdir(ref_dir):
        raise FileNotFoundError(
            f"Reference pool not found: {ref_dir}\n"
            f"Run the temporal pool generation script first."
        )

    if not os.path.isdir(unlabeled_all_dir):
        raise FileNotFoundError(
            f"Unlabeled source not found: {unlabeled_all_dir}"
        )

    # Safety: fail if output already contains PNG files
    if os.path.isdir(out_dir):
        existing_pngs = glob.glob(os.path.join(out_dir, "*.png"))
        if existing_pngs:
            raise RuntimeError(
                f"Output folder already contains {len(existing_pngs)} PNG file(s):\n"
                f"  {out_dir}\n"
                f"Remove the folder manually before re-running."
            )

    ensure_dir(out_dir)

    labeled_keys, labeled_by_video, train_videos, _, _ = load_labeled_from_folders(out_root)
    ref_counts = _count_ref_frames_per_video(ref_dir, train_videos)
    dual_print(f"\nReference pool per-video counts: {dict(sorted(ref_counts.items()))}")

    # Fresh RNG per tag — each radius starts independently from the same seed
    rng = random.Random(seed)

    # Index all available unlabeled frames
    all_frames_by_video = _index_unlabeled_all(unlabeled_all_dir)

    total_saved         = 0
    total_overlap       = 0
    per_video_manifest  = {}

    for vid in sorted(train_videos):
        ref_count = ref_counts.get(vid, 0)
        if ref_count == 0:
            dual_print(f"v{vid}: no reference frames - skipping")
            continue

        # All frames in unlabeled_all for this video that are NOT labeled
        vid_all = all_frames_by_video.get(vid, {})
        available_stems = sorted([
            stem for stem in vid_all
            if stem not in labeled_keys
        ])

        if len(available_stems) < ref_count:
            raise RuntimeError(
                f"v{vid}: need {ref_count} unlabeled frames "
                f"but only {len(available_stems)} available in unlabeled_all/. "
                f"Cannot build matched pool."
            )

        sampled_stems = sorted(rng.sample(available_stems, ref_count))

        # Overlap sanity check
        overlap = [s for s in sampled_stems if s in labeled_keys]
        if overlap:
            total_overlap += len(overlap)
            dual_print(f"[ERROR] v{vid}: {len(overlap)} sampled frames overlap with labeled!")

        # Copy PNGs from unlabeled_all/ to output dir
        saved_for_video = 0
        for stem in sampled_stems:
            src_path = vid_all[stem]
            if not os.path.isfile(src_path):
                raise FileNotFoundError(
                    f"Expected PNG not found: {src_path}"
                )
            dst_path = os.path.join(out_dir, f"{stem}.png")
            shutil.copy2(src_path, dst_path)
            saved_for_video += 1
            total_saved += 1

        per_video_manifest[vid] = {
            "reference_count":       ref_count,
            "total_unlabeled_all":   len(vid_all),
            "available_non_labeled": len(available_stems),
            "sampled_count":         saved_for_video,
            "sampled_stems":         sampled_stems,
        }

        assert saved_for_video == ref_count, (
            f"v{vid}: saved {saved_for_video} != reference {ref_count}"
        )

        dual_print(f"v{vid}: ref={ref_count}, "
                   f"unlabeled_all={len(vid_all)}, "
                   f"available={len(available_stems)}, saved={saved_for_video}")

    # Validate total
    expected_total = sum(v["reference_count"] for v in per_video_manifest.values())
    if total_saved != expected_total:
        dual_print(f"[WARN] total_saved ({total_saved}) != sum of reference_counts ({expected_total})")

    dual_print(f"\n=== SUMMARY for {ref_tag} ===")
    dual_print(f"Videos processed          : {len(per_video_manifest)}")
    dual_print(f"Total frames saved        : {total_saved}")
    dual_print(f"Overlap with labeled      : {total_overlap}")

    if total_overlap != 0:
        raise RuntimeError(
            f"BUG: {total_overlap} sampled frame(s) overlap with labeled frames."
        )

    manifest = {
        "reference_pool":             f"unlabeled_{ref_tag}",
        "output_pool":                f"unlabeled_random_{ref_tag}",
        "random_seed":                seed,
        "ap_cutoff_applied":          False,
        "ap_cutoff_frames":           {},
        "total_ap_frames_excluded":   0,
        "total_frames_saved":         total_saved,
        "total_overlap_with_labeled": total_overlap,
        "per_video":                  per_video_manifest,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    dual_print(f"Manifest saved: {manifest_path}")
    dual_print(f"Output dir    : {out_dir}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build random matched unlabeled pools for INCA dataset."
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed for frame sampling (default: {DEFAULT_SEED})"
    )
    parser.add_argument(
        "--root", type=str, default=OUT_ROOT,
        help=f"INCA dataset root directory (default: {OUT_ROOT})"
    )
    args = parser.parse_args()

    out_root = args.root
    if not os.path.exists(out_root):
        raise FileNotFoundError(f"Root does not exist: {out_root}")

    for ref_tag in REFERENCE_TAGS:
        build_random_pool(ref_tag, seed=args.seed, out_root=out_root)

    print("\nDone. All random matched pools built successfully.")


if __name__ == "__main__":
    main()
