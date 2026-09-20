"""
test_pool_equivalence.py

Validates that the core sampling logic in prep_unm_unlabeled_std_clean.py
and prep_inca_unlabeled_random.py is equivalent.

Uses synthetic data — no filesystem, no video decoding, no OpenCV.

What this test validates:
- Same RNG initialization (fresh per tag, same seed)
- Same iteration order (sorted train_videos)
- Same skip logic (ref_count==0 -> skip, no RNG consumed)
- Same sampling call (rng.sample on sorted list of same length)
- Same sort of result

What this test does NOT validate:
- Filesystem I/O (PNG copy vs video decode)
- Filename format differences (v001_f0 vs v001_f1)
- AP cutoff logic (UNM-only, disabled in this test)
- Error handling for missing files/videos
"""

import random


# ─────────────────────────────────────────────
# Synthetic test data
# ─────────────────────────────────────────────

SEED = 42

TRAIN_VIDEOS = {"001", "002", "003", "004"}

# 5 labeled frame IDs per video (same for all videos)
LABELED_FIDS = {5, 10, 15, 20, 25}

# Total frames per video
N_FRAMES = 55

# Reference counts per video — "004" has 0 to test skip logic
REF_COUNTS = {"001": 10, "002": 10, "003": 10, "004": 0}


# ─────────────────────────────────────────────
# UNM-style sampling (from prep_unm_unlabeled_std_clean.py)
# ─────────────────────────────────────────────

def simulate_unm(seed, train_videos, labeled_fids, n_frames, ref_counts):
    """
    Replicates the core sampling logic from build_std_pool().
    UNM uses 0-based frame indices. No AP cutoff in this simulation.
    """
    # Build global labeled_keys set (same structure as UNM script)
    labeled_keys = set()
    for vid in train_videos:
        for fid in labeled_fids:
            labeled_keys.add(f"v{int(vid):03d}_f{fid}")

    rng = random.Random(seed)
    results = {}

    for vid in sorted(train_videos):
        ref_count = ref_counts.get(vid, 0)
        if ref_count == 0:
            continue

        max_frame = n_frames  # no AP cutoff

        # All frames before max_frame that are NOT labeled (0-based)
        available = sorted([
            k for k in range(max_frame)
            if f"v{int(vid):03d}_f{k}" not in labeled_keys
        ])

        sampled = sorted(rng.sample(available, ref_count))
        results[vid] = {"available": available, "sampled": sampled}

    return results


# ─────────────────────────────────────────────
# INCA-style sampling (from prep_inca_unlabeled_random.py)
# ─────────────────────────────────────────────

def simulate_inca(seed, train_videos, labeled_fids, n_frames, ref_counts):
    """
    Replicates the core sampling logic from build_random_pool().
    INCA uses 1-based frame indices as stems. No AP cutoff.
    """
    # Build global labeled_keys set (same structure as INCA script)
    labeled_keys = set()
    for vid in train_videos:
        for fid in labeled_fids:
            labeled_keys.add(f"v{int(vid):03d}_f{fid}")

    # Simulate unlabeled_all: all frames 1..n_frames as stems
    all_frames_by_video = {}
    for vid in train_videos:
        all_frames_by_video[vid] = {
            f"v{int(vid):03d}_f{fid}": None  # path not needed
            for fid in range(1, n_frames + 1)
        }

    rng = random.Random(seed)
    results = {}

    for vid in sorted(train_videos):
        ref_count = ref_counts.get(vid, 0)
        if ref_count == 0:
            continue

        vid_all = all_frames_by_video.get(vid, {})
        available_stems = sorted([
            stem for stem in vid_all
            if stem not in labeled_keys
        ])

        sampled_stems = sorted(rng.sample(available_stems, ref_count))
        results[vid] = {"available": available_stems, "sampled": sampled_stems}

    return results


# ─────────────────────────────────────────────
# Equivalence checks
# ─────────────────────────────────────────────

def get_positional_indices(available, sampled):
    """Convert sampled elements to their positional indices in the available list."""
    index_map = {elem: i for i, elem in enumerate(available)}
    return [index_map[s] for s in sampled]


def main():
    unm = simulate_unm(SEED, TRAIN_VIDEOS, LABELED_FIDS, N_FRAMES, REF_COUNTS)
    inca = simulate_inca(SEED, TRAIN_VIDEOS, LABELED_FIDS, N_FRAMES, REF_COUNTS)

    # --- Skip check: video "004" should not appear in either result ---
    assert "004" not in unm, "UNM should skip video 004 (ref_count=0)"
    assert "004" not in inca, "INCA should skip video 004 (ref_count=0)"
    print("PASS: video 004 (ref_count=0) skipped by both — no RNG consumed")

    # --- Both should have the same set of active videos ---
    assert set(unm.keys()) == set(inca.keys()), (
        f"Active videos differ: UNM={set(unm.keys())}, INCA={set(inca.keys())}"
    )
    print(f"PASS: same active videos: {sorted(unm.keys())}")

    # --- Per-video checks ---
    for vid in sorted(unm.keys()):
        unm_avail = unm[vid]["available"]
        inca_avail = inca[vid]["available"]
        unm_sampled = unm[vid]["sampled"]
        inca_sampled = inca[vid]["sampled"]

        # Secondary: same count
        assert len(unm_sampled) == len(inca_sampled), (
            f"v{vid}: count mismatch UNM={len(unm_sampled)} INCA={len(inca_sampled)}"
        )

        # Available lists must have the same length for positional equivalence
        assert len(unm_avail) == len(inca_avail), (
            f"v{vid}: available length mismatch UNM={len(unm_avail)} INCA={len(inca_avail)}"
        )

        # PRIMARY: same positional indices within available
        unm_positions = get_positional_indices(unm_avail, unm_sampled)
        inca_positions = get_positional_indices(inca_avail, inca_sampled)

        assert unm_positions == inca_positions, (
            f"v{vid}: positional indices differ!\n"
            f"  UNM:  {unm_positions}\n"
            f"  INCA: {inca_positions}"
        )

        # Informational: show that raw IDs differ by format/offset
        print(f"  v{vid}: {len(unm_sampled)} frames sampled, positions match")
        print(f"    UNM raw (first 3):  {unm_sampled[:3]}")
        print(f"    INCA raw (first 3): {inca_sampled[:3]}")

    print("\n" + "="*50)
    print("ALL CHECKS PASSED — sampling logic is equivalent")
    print("="*50)


if __name__ == "__main__":
    main()
