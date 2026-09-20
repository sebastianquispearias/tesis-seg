"""
prep_unm_unlabeled_all_lateral.py

Extracts ALL lateral frames (before AP cutoff) from training videos,
excluding labeled frames. This creates the "100%" unlabeled pool.

Usage:
    python prep_unm_unlabeled_all_lateral.py

Output: unlabeling_all_lateral/images/
"""

import os
import sys
import re
from collections import defaultdict
from pathlib import Path

import cv2

# ─────────────────────────────────────────────────
# CONFIG - adjust these paths to your setup
# ─────────────────────────────────────────────────

ROOT           = r"G:\My Drive\UNM TalkBank Dysphagia"
OUT_ROOT       = r"G:\My Drive\UNM_vertebras_seg_v3"

VIDEOS_DIR     = os.path.join(ROOT, "videos")
VIDEOS_SEL_DIR = os.path.join(ROOT, "videos-selecionados")

#OUT_DIR        = os.path.join(OUT_ROOT, "unlabeling_all_lateral", "images")
OUT_DIR = r"C:\temp\unlabeling_all_lateral\images"

# AP cutoff per video (frames >= this are excluded)
AP_CUTOFF = {
    "025": 3030, "029": 1190, "032": 3678, "033": 4280, "034": None,
    "035": 1643, "036": None, "037": 1549, "038": 3457, "039": 4078,
    "044": 3262, "045": 1056, "059": None, "063": 3698, "064": None,
    "067": 2497, "070": None, "072": 5627, "078": 4426, "079": 3951,
    "080": 2357, "088": 1962, "089": None,
}

# ─────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────

def load_labeled_keys(out_root):
    """Load all labeled frame keys from train/val/test mask folders."""
    labeled_keys = set()
    train_videos = set()

    for split in ["train", "val", "test"]:
        mdir = os.path.join(out_root, split, "masks")
        if not os.path.isdir(mdir):
            continue
        for fname in os.listdir(mdir):
            if not fname.lower().endswith(".png"):
                continue
            stem = Path(fname).stem
            if "_f" not in stem:
                continue
            labeled_keys.add(stem)
            if split == "train":
                vpart = stem.split("_f")[0]
                vid = vpart.replace("v", "")
                train_videos.add(f"{int(vid):03d}")

    return labeled_keys, train_videos


def find_video_file(vid):
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


# ─────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────

def main():
    print("=== prep_unm_unlabeled_all_lateral.py ===")
    print(f"Output: {OUT_DIR}")

    # Safety check
    if os.path.isdir(OUT_DIR):
        existing = [f for f in os.listdir(OUT_DIR) if f.endswith(".png")]
        if existing:
            print(f"ERROR: Output folder already contains {len(existing)} PNGs.")
            print(f"  {OUT_DIR}")
            print("Delete it manually before re-running.")
            sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    labeled_keys, train_videos = load_labeled_keys(OUT_ROOT)
    print(f"Train videos: {sorted(train_videos)}")
    print(f"Total labeled keys: {len(labeled_keys)}")

    total_saved = 0
    total_skipped_labeled = 0
    total_skipped_ap = 0

    for vid in sorted(train_videos):
        video_path = find_video_file(vid)
        if video_path is None:
            print(f"[WARN] v{vid}: video file not found - skipping")
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[WARN] v{vid}: could not open video - skipping")
            continue

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cutoff = AP_CUTOFF.get(vid)
        max_frame = cutoff if cutoff is not None else n_frames

        saved_this_video = 0
        skipped_labeled = 0
        skipped_ap = n_frames - max_frame

        # Read sequentially (much faster than seeking each frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for idx in range(max_frame):
            ok, frame = cap.read()
            if not ok:
                break

            key = f"v{int(vid):03d}_f{idx}"
            if key in labeled_keys:
                skipped_labeled += 1
                continue

            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            out_path = os.path.join(OUT_DIR, f"{key}.png")
            cv2.imwrite(out_path, frame)
            saved_this_video += 1

        cap.release()

        total_saved += saved_this_video
        total_skipped_labeled += skipped_labeled
        total_skipped_ap += skipped_ap

        print(f"v{vid}: max_lateral={max_frame}, saved={saved_this_video}, "
              f"skipped_labeled={skipped_labeled}, skipped_ap={skipped_ap}")

    print(f"\n=== SUMMARY ===")
    print(f"Videos processed:     {len(train_videos)}")
    print(f"Total frames saved:   {total_saved}")
    print(f"Skipped (labeled):    {total_skipped_labeled}")
    print(f"Skipped (AP):         {total_skipped_ap}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
