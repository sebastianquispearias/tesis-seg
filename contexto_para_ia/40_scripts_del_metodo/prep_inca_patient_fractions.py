"""
prep_inca_patient_fractions.py
===============================
Creates nested labeled-subset stem files for the INCA per-PATIENT fraction
experiments (notebook 09).

Unlike prep_inca_label_fractions.py (which samples frames per-video and
can't go below 156 frames = 24.6% of train because max(1, ceil(n*frac))
gives 1 frame × 156 videos), this script samples entire PATIENTS and
includes ALL their frames. This allows going as low as ~56 frames (8.8%).

Algorithm:
  1. Read patient→video→frames mapping from video_frame_metadata.csv
  2. Filter to train patients only (from split_info.json)
  3. Classify patients into multi-video (>1 video) and single-video (1 video)
  4. Shuffle each group independently with random.Random(0)
  5. For each fraction target (10%, 25%, 50%):
     - Take max(1, round(N_group × target)) patients from each group
     - Collect ALL frames from selected patients
     - Write sorted stems to stems.txt
  6. Verify nesting: frac10 subset_of frac25 subset_of frac50
  7. Verify all stems exist as files in train/images/
  8. Verify no val/test patients are included
  9. Write summary CSV

Stratification rationale:
  INCA has 10 multi-video patients (374 frames, 59% of train) and 65
  single-video patients (260 frames, 41%). Stratifying ensures each
  fraction has a stable mix of both types, preventing subsets dominated
  by heavy multi-video patients or sparse single-video ones.

  In UNM, per-video = per-patient (1 video per patient, 1:1 mapping).
  In INCA, per-patient is the correct unit because multiple videos from
  the same patient are not independent observations.

Outputs:
  {out_root}/label_fractions/patient_frac_10/stems.txt
  {out_root}/label_fractions/patient_frac_25/stems.txt
  {out_root}/label_fractions/patient_frac_50/stems.txt
  {out_root}/label_fractions/patient_fractions_summary.csv

Usage (local Windows):
  python prep_inca_patient_fractions.py

Usage (Colab):
  !python prep_inca_patient_fractions.py \
      --inca_root /content/drive/MyDrive/UNM_vertebras_seg_v3/data/inca_dataset \
      --metadata /content/drive/MyDrive/UNM_vertebras_seg_v3/data/video_frame_metadata.csv
"""

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path


SUBSET_SEED = 0
FRACS = [("10", 0.10), ("25", 0.25), ("50", 0.50)]

DEFAULT_INCA_ROOT = r"G:/My Drive/UNM_vertebras_seg_v3/data/inca_dataset"
DEFAULT_METADATA = r"G:/My Drive/UNM_vertebras_seg_v3/data/video_frame_metadata.csv"


def load_patient_mapping(metadata_path, train_patients, train_images_dir):
    """Build patient → [stems] mapping from metadata CSV + actual train files.

    Returns:
        patient_frames: {patient_id: [stem, ...]} sorted per patient
        patient_videos: {patient_id: set(video_ids)}
    """
    # video_id → patient_id from metadata
    video_to_patient = {}
    with open(metadata_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = int(row["paciente_id"])
            vid = int(row["video_id"])
            video_to_patient[vid] = pid

    # Actual stems from train/images/
    pattern = re.compile(r"^v(\d+)_f(\d+)\.png$")
    train_vid_frames = defaultdict(list)
    for p in sorted(train_images_dir.glob("*.png")):
        m = pattern.match(p.name)
        if m:
            vid = int(m.group(1))
            train_vid_frames[vid].append(p.stem)

    # Build patient → frames (train patients only)
    patient_frames = defaultdict(list)
    patient_videos = defaultdict(set)
    for vid, stems in train_vid_frames.items():
        pid = video_to_patient.get(vid)
        if pid is not None and pid in train_patients:
            patient_frames[pid].extend(stems)
            patient_videos[pid].add(vid)

    for pid in patient_frames:
        patient_frames[pid].sort()

    return dict(patient_frames), dict(patient_videos)


def main():
    parser = argparse.ArgumentParser(
        description="Generate nested per-patient label-fraction stem files for INCA."
    )
    parser.add_argument("--inca_root", type=str, default=DEFAULT_INCA_ROOT)
    parser.add_argument("--metadata", type=str, default=DEFAULT_METADATA)
    args = parser.parse_args()

    inca_root = Path(args.inca_root)
    metadata_path = Path(args.metadata)
    train_images_dir = inca_root / "train" / "images"
    label_fractions_dir = inca_root / "label_fractions"

    # Load split info
    split_info_path = inca_root / "split_info.json"
    with open(split_info_path, encoding="utf-8") as f:
        si = json.load(f)
    train_patients = set(si["train_patients"])
    val_patients = set(si["val_patients"])
    test_patients = set(si["test_patients"])

    print(f"Train patients: {len(train_patients)}")
    print(f"Val patients:   {len(val_patients)}")
    print(f"Test patients:  {len(test_patients)}")

    # Build mapping
    patient_frames, patient_videos = load_patient_mapping(
        metadata_path, train_patients, train_images_dir
    )
    total_frames = sum(len(f) for f in patient_frames.values())
    print(f"Train patients with frames: {len(patient_frames)}")
    print(f"Total train frames: {total_frames}")

    # Classify into multi-video and single-video
    multi_pids = sorted([p for p in patient_frames if len(patient_videos[p]) > 1])
    single_pids = sorted([p for p in patient_frames if len(patient_videos[p]) == 1])
    print(f"\nMulti-video patients:  {len(multi_pids)} -> {sum(len(patient_frames[p]) for p in multi_pids)} frames")
    print(f"Single-video patients: {len(single_pids)} -> {sum(len(patient_frames[p]) for p in single_pids)} frames")

    # Shuffle each group independently with the same seed
    rng_multi = random.Random(SUBSET_SEED)
    rng_single = random.Random(SUBSET_SEED)
    shuffled_multi = multi_pids[:]
    shuffled_single = single_pids[:]
    rng_multi.shuffle(shuffled_multi)
    rng_single.shuffle(shuffled_single)

    print(f"\nShuffled multi order:  {shuffled_multi}")
    print(f"Shuffled single order (first 10): {shuffled_single[:10]}")

    # Generate fractions
    selected_per_frac = {}
    all_stems_per_frac = {}

    for tag, target_pct in FRACS:
        n_multi = max(1, round(len(multi_pids) * target_pct))
        n_single = max(1, round(len(single_pids) * target_pct))

        sel_multi = shuffled_multi[:n_multi]
        sel_single = shuffled_single[:n_single]
        selected = sel_multi + sel_single

        stems = sorted([s for pid in selected for s in patient_frames[pid]])

        selected_per_frac[tag] = {
            "patients": selected,
            "multi": sel_multi,
            "single": sel_single,
            "n_multi": n_multi,
            "n_single": n_single,
        }
        all_stems_per_frac[tag] = stems

        # Write stems.txt
        out_dir = label_fractions_dir / f"patient_frac_{tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        stems_path = out_dir / "stems.txt"
        stems_path.write_text("\n".join(stems), encoding="utf-8")

        print(f"\n  patient_frac_{tag}: {n_multi} multi + {n_single} single = {len(selected)} patients, {len(stems)} frames ({len(stems)/total_frames*100:.1f}%)")
        print(f"    multi patients: {sel_multi}")
        print(f"    -> {stems_path}")

    # Verification 1: Nesting
    print("\n-- Nesting verification --")
    s10 = set(all_stems_per_frac["10"])
    s25 = set(all_stems_per_frac["25"])
    s50 = set(all_stems_per_frac["50"])
    full = set(s for stems in patient_frames.values() for s in stems)

    checks = [
        ("frac10 subset_of frac25", s10 <= s25),
        ("frac25 subset_of frac50", s25 <= s50),
        ("frac50 subset_of full",   s50 <= full),
    ]
    all_ok = True
    for label, ok in checks:
        print(f"  {label}: {'OK' if ok else 'FAIL'}")
        if not ok:
            all_ok = False

    # Verification 2: All stems exist as files
    print("\n-- File existence verification --")
    missing_files = []
    for tag, stems in all_stems_per_frac.items():
        for s in stems:
            if not (train_images_dir / f"{s}.png").is_file():
                missing_files.append((tag, s))
    if missing_files:
        print(f"  FAIL: {len(missing_files)} stems not found in train/images/")
        for t, s in missing_files[:5]:
            print(f"    frac_{t}: {s}")
        all_ok = False
    else:
        print(f"  OK: all {sum(len(s) for s in all_stems_per_frac.values())} stems exist in train/images/")

    # Verification 3: No val/test patients included
    print("\n-- Patient isolation verification --")
    for tag, info in selected_per_frac.items():
        for pid in info["patients"]:
            if pid in val_patients or pid in test_patients:
                print(f"  FAIL: patient {pid} in frac_{tag} is in val/test!")
                all_ok = False
    if all_ok:
        print("  OK: no val/test patients in any fraction")

    # Write summary CSV
    summary_path = label_fractions_dir / "patient_fractions_summary.csv"
    summary_rows = []
    for tag, _ in FRACS:
        info = selected_per_frac[tag]
        for pid in sorted(info["patients"]):
            ptype = "multi" if pid in multi_pids else "single"
            summary_rows.append({
                "fraction": f"patient_frac_{tag}",
                "patient_id": pid,
                "n_videos": len(patient_videos[pid]),
                "n_frames": len(patient_frames[pid]),
                "type": ptype,
            })
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fraction", "patient_id", "n_videos", "n_frames", "type"])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\n  Summary CSV -> {summary_path}")

    # Final report
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'fraction':>20s} {'patients':>9s} {'frames':>7s} {'% train':>8s}")
    print("-" * 48)
    for tag, _ in FRACS:
        info = selected_per_frac[tag]
        n = len(all_stems_per_frac[tag])
        print(f"  patient_frac_{tag:>2s}   {len(info['patients']):9d} {n:7d} {n/total_frames*100:7.1f}%")
    print(f"  {'full train':>16s}   {len(patient_frames):9d} {total_frames:7d} {100.0:7.1f}%")
    print(f"{'=' * 60}")

    if not all_ok:
        print("\n[ERROR] Some verification checks failed!")
        sys.exit(1)
    else:
        print("\nAll verifications passed.")


if __name__ == "__main__":
    main()
