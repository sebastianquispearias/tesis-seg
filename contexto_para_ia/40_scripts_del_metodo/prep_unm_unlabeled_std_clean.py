"""
prep_unm_unlabeled_std_clean.py

Builds random matched unlabeled pools as controls for the temporal-neighbor pools,
EXCLUDING frames after the AP projection change in each video.

Usage:
    python prep_unm_unlabeled_std_clean.py            # default seed=42
    python prep_unm_unlabeled_std_clean.py --seed 7   # custom seed
"""

import argparse
import glob
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2

# ─────────────────────────────────────────────
# TOP-LEVEL CONFIG
# ─────────────────────────────────────────────

ROOT           = r"G:\My Drive\UNM TalkBank Dysphagia"
OUT_ROOT       = r"G:\My Drive\UNM_vertebras_seg_v3"

VIDEOS_DIR     = os.path.join(ROOT, "videos")
VIDEOS_SEL_DIR = os.path.join(ROOT, "videos-selecionados")

DEFAULT_SEED   = 42

# Pools to generate
REFERENCE_TAGS = ["r20_max0"]

# Frame de corte AP por video. Frames >= este valor se EXCLUYEN.
# N/A significa que no hay cambio de toma (se usan todos los frames).
AP_CUTOFF = {
    "025": 3030,
    "029": 1190,
    "032": 3678,
    "033": 4280,
    "034": None,   # N/A
    "035": 1643,
    "036": None,   # N/A
    "037": 1549,
    "038": 3457,
    "039": 4078,
    "044": 3262,
    "045": 1056,
    "059": None,   # N/A
    "063": 3698,
    "064": None,   # N/A
    "067": 2497,
    "070": None,   # N/A
    "072": 5627,
    "078": 4426,
    "079": 3951,
    "080": 2357,
    "088": 1962,
    "089": None,   # N/A
}


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


def find_video_file(vid: str):
    vid_int = int(vid)
    candidates = [
        os.path.join(VIDEOS_SEL_DIR, f"v{vid_int:03d}.avi"),
        os.path.join(VIDEOS_SEL_DIR, f"{vid_int:03d}.avi"),
        os.path.join(VIDEOS_DIR,     f"v{vid_int:03d}.avi"),
        os.path.join(VIDEOS_DIR,     f"{vid_int}.avi"),
        os.path.join(VIDEOS_SEL_DIR, f"v{vid_int:03d}.mp4"),
        os.path.join(VIDEOS_DIR,     f"v{vid_int:03d}.mp4"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def count_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def read_frame(video_path: str, idx: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame


def save_gray(frame, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(out_path, frame)


def get_max_frame(vid: str, n_frames: int) -> int:
    """Returns the maximum usable frame index for a video (exclusive).
    If the video has an AP cutoff, returns the cutoff frame.
    Otherwise returns the total number of frames."""
    vid_normalized = f"{int(vid):03d}"
    cutoff = AP_CUTOFF.get(vid_normalized)
    if cutoff is not None:
        return cutoff
    return n_frames


# ─────────────────────────────────────────────
# CORE LOGIC
# ─────────────────────────────────────────────

def _count_ref_frames_per_video(ref_dir: str, train_videos: set) -> dict:
    counts = {}
    pattern = re.compile(r"^v(\d+)_f\d+\.png$", re.IGNORECASE)
    for fname in os.listdir(ref_dir):
        m = pattern.match(fname)
        if not m:
            continue
        vid = f"{int(m.group(1)):03d}"
        if vid in train_videos:
            counts[vid] = counts.get(vid, 0) + 1
    return counts


def build_std_pool(ref_tag: str, seed: int):
    short_label = ref_tag.split("_")[0]

    ref_dir = os.path.join(OUT_ROOT, f"unlabeling_{ref_tag}", "images")
    out_dir = os.path.join(OUT_ROOT, f"unlabeling_std_matched_{short_label}", "images")
    manifest_path = os.path.join(OUT_ROOT, f"unlabeling_std_matched_{short_label}_manifest.json")

    dual_print(f"\n{'='*60}")
    dual_print(f"Building CLEAN std pool matched to: {ref_tag}")
    dual_print(f"  Reference dir : {ref_dir}")
    dual_print(f"  Output dir    : {out_dir}")
    dual_print(f"  Random seed   : {seed}")
    dual_print(f"  AP cutoffs    : ENABLED")

    if not os.path.isdir(ref_dir):
        raise FileNotFoundError(
            f"Reference pool not found: {ref_dir}\n"
            f"Run prep_unm_unlabeled_new.py with matching NEIGHBOR_RADIUS first."
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

    labeled_keys, labeled_by_video, train_videos, _, _ = load_labeled_from_folders(OUT_ROOT)
    ref_counts = _count_ref_frames_per_video(ref_dir, train_videos)
    dual_print(f"\nReference pool per-video counts: {dict(sorted(ref_counts.items()))}")

    rng = random.Random(seed)

    total_saved         = 0
    total_overlap       = 0
    total_ap_excluded   = 0
    missing_video       = 0
    per_video_manifest  = {}

    for vid in sorted(train_videos):
        ref_count = ref_counts.get(vid, 0)
        if ref_count == 0:
            dual_print(f"v{int(vid):03d}: no reference frames - skipping")
            continue

        video_path = find_video_file(vid)
        if video_path is None:
            dual_print(f"[WARN] v{int(vid):03d}: video file not found - skipping")
            missing_video += 1
            continue

        n_frames = count_frames(video_path)
        if n_frames <= 0:
            dual_print(f"[WARN] v{int(vid):03d}: no readable frames - skipping")
            continue

        max_frame = get_max_frame(vid, n_frames)
        ap_frames_excluded = n_frames - max_frame

        # All frames before AP cutoff that are NOT labeled
        available = [
            k for k in range(max_frame)
            if f"v{int(vid):03d}_f{k}" not in labeled_keys
        ]

        if len(available) < ref_count:
            dual_print(
                f"[WARN] v{int(vid):03d}: need {ref_count} unlabeled frames "
                f"but only {len(available)} available (after AP cutoff at {max_frame}). "
                f"Sampling all {len(available)} available frames."
            )
            sampled = sorted(available)
        else:
            sampled = sorted(rng.sample(available, ref_count))

        # Overlap sanity check
        overlap = [k for k in sampled if f"v{int(vid):03d}_f{k}" in labeled_keys]
        if overlap:
            total_overlap += len(overlap)
            dual_print(f"[ERROR] v{int(vid):03d}: {len(overlap)} sampled frames overlap with labeled!")

        saved_for_video = 0
        for k in sampled:
            frame = read_frame(video_path, k)
            if frame is None:
                dual_print(f"[WARN] v{int(vid):03d} f{k}: could not read frame")
                continue
            out_name = f"v{int(vid):03d}_f{k}.png"
            out_path = os.path.join(out_dir, out_name)
            save_gray(frame, out_path)
            saved_for_video += 1
            total_saved += 1

        total_ap_excluded += ap_frames_excluded

        per_video_manifest[vid] = {
            "reference_count":       ref_count,
            "max_frame_lateral":     max_frame,
            "total_video_frames":    n_frames,
            "ap_frames_excluded":    ap_frames_excluded,
            "available_non_labeled": len(available),
            "sampled_count":         saved_for_video,
            "sampled_frames":        sampled,
        }

        dual_print(f"v{int(vid):03d}: ref={ref_count}, max_lateral={max_frame}, "
                   f"available={len(available)}, saved={saved_for_video}, "
                   f"ap_excluded={ap_frames_excluded}")

    dual_print(f"\n=== SUMMARY for {short_label} (CLEAN) ===")
    dual_print(f"Videos processed          : {len(per_video_manifest)}")
    dual_print(f"Videos missing video file : {missing_video}")
    dual_print(f"Total frames saved        : {total_saved}")
    dual_print(f"Total AP frames excluded  : {total_ap_excluded}")
    dual_print(f"Overlap with labeled      : {total_overlap}")

    if total_overlap != 0:
        raise RuntimeError(
            f"BUG: {total_overlap} sampled frame(s) overlap with labeled frames."
        )

    manifest = {
        "reference_pool":             f"unlabeling_{ref_tag}",
        "output_pool":                f"unlabeling_std_matched_{short_label}",
        "random_seed":                seed,
        "ap_cutoff_applied":          True,
        "ap_cutoff_frames":           {k: v for k, v in AP_CUTOFF.items() if v is not None},
        "total_frames_saved":         total_saved,
        "total_ap_frames_excluded":   total_ap_excluded,
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
        description="Build CLEAN random matched std unlabeled pools (excluding AP frames)."
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed for frame sampling (default: {DEFAULT_SEED})"
    )
    args = parser.parse_args()

    if not os.path.exists(OUT_ROOT):
        raise FileNotFoundError(f"OUT_ROOT does not exist: {OUT_ROOT}")

    for ref_tag in REFERENCE_TAGS:
        build_std_pool(ref_tag, seed=args.seed)

    print("\nDone. All clean std pools built successfully.")


if __name__ == "__main__":
    main()
