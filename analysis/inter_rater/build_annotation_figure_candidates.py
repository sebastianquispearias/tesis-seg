"""
build_annotation_figure_candidates.py
=====================================
READ-ONLY over original data. Generates candidate figures for the annotation
appendix (INCA inter-annotator agreement). Does NOT modify or overwrite any
original mask or frame.

Pipeline (all in one run):
  STEP 2  per-frame metrics for the 898 doubly-annotated frames -> per_frame_metrics.csv
          (Dice, areas, relative area diff, connected components per mask,
           XOR area and XOR centroid relative to the union bounding box).
          Verifies mean 0.874 +/- 0.070 and median 0.899; stops if it disagrees.
  STEP 3  selects 8-12 candidates per figure group.
  STEP 4  contact-sheet PNGs per group + Dice histogram + manifest.csv.

Masks loaded at NATIVE resolution with the SAME processing as the canonical
inter-rater script (convert L, binarize >127, invert if foreground > 10%).
The 3 resolution-mismatch frames are excluded (they are not in the 898 CSV).

Output: resultados/inter_rater_inca/figure_candidates/   (new folder)

Reproducible: SEED = 42 fixed for any tie-break sampling.

Usage:  python build_annotation_figure_candidates.py
"""
import csv, os, sys, io, time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)

SEED = 42
np.random.seed(SEED)

PROJECT = Path(r"G:/My Drive/UNM_vertebras_seg_v3")
# Inputs are read from a LOCAL mirror (fast) populated by copy_inca_masks_local.py.
LOCAL = Path(r"C:/Users/User/temp_inter_rater/inca_fig")
METADATA_CSV = LOCAL / "video_frame_metadata.csv"
PAIR_CSV = PROJECT / "resultados" / "inter_rater_inca" / "inter_rater_per_frame.csv"
ROTULOS = LOCAL / "rotulos"                 # rotulos/batch-{b}/{lab}/{vf}/Mask.tif
FRAMES = LOCAL / "frames"                   # frames/{split}/{vf}.png
# Outputs still go to Drive (new folder), as agreed.
OUT_DIR = PROJECT / "resultados" / "inter_rater_inca" / "figure_candidates"

LABELERS = ["AM", "BB", "CS", "VC", "VR"]
N_PER_GROUP = 12
MAX_PER_VIDEO = 2          # variety: at most N candidates from the same video per group
PER_SHEET = 6              # candidates per contact sheet
COLOR_A = "#00E5FF"        # labeler_A contour (cyan)
COLOR_B = "#FF9800"        # labeler_B contour (orange)
CONN = np.ones((3, 3), int)  # 8-connectivity for connected components

# ----------------------------------------------------------------------------
# IO helpers (with retries: Google Drive streaming can hiccup)
# ----------------------------------------------------------------------------
def _read_image_L(path, retries=4):
    last = None
    for k in range(retries):
        try:
            with Image.open(str(path)) as im:
                return np.array(im.convert("L"))
        except Exception as e:  # noqa: BLE001
            last = e; time.sleep(0.5 * (k + 1))
    raise last

def process_mask(tif_path):
    """Canonical processing: grayscale, binarize >127, invert if fg>10%."""
    arr = _read_image_L(tif_path)
    arr = np.where(arr > 127, 255, 0).astype(np.uint8)
    if (arr == 255).sum() / arr.size > 0.10:
        arr = 255 - arr
    return arr

def load_frame(split, vf):
    p = FRAMES / split / f"{vf}.png"
    return _read_image_L(p)

def tif_path(batch, labeler, vf):
    return ROTULOS / f"batch-{batch}" / labeler / vf / "Mask.tif"

def dice(a, b):
    A = (a > 127); B = (b > 127)
    sa, sb = A.sum(), B.sum()
    if sa == 0 and sb == 0: return 1.0
    if sa == 0 or sb == 0: return 0.0
    return float(2.0 * (A & B).sum() / (sa + sb))

# ----------------------------------------------------------------------------
# STEP 2 — per-frame metrics
# ----------------------------------------------------------------------------
def step2_metrics():
    print("=" * 70); print("STEP 2: per-frame metrics (898 frames)"); print("=" * 70)
    batch_of = {r["video_frame"]: str(r["batch"])
                for r in csv.DictReader(open(METADATA_CSV, encoding="utf-8"))}
    pairs = list(csv.DictReader(open(PAIR_CSV, encoding="utf-8")))
    print(f"  pairs in CSV: {len(pairs)}")

    rows = []; errors = []; dice_mismatch = 0
    for i, r in enumerate(pairs):
        vf = r["video_frame"]; split = r["split"]
        la, lb = r["labeler_A"], r["labeler_B"]; b = batch_of.get(vf)
        try:
            ma = process_mask(tif_path(b, la, vf))
            mb = process_mask(tif_path(b, lb, vf))
        except Exception as e:  # noqa: BLE001
            errors.append((vf, str(e)[:60])); continue
        if ma.shape != mb.shape:
            errors.append((vf, "shape-mismatch")); continue

        A = ma > 127; B = mb > 127
        area_a = int(A.sum()); area_b = int(B.sum())
        d = dice(ma, mb)
        # cross-check against canonical Dice in the CSV
        if abs(d - float(r["dice"])) > 1e-3:
            dice_mismatch += 1

        rel_area = abs(area_a - area_b) / max(area_a, area_b, 1)
        _, ncomp_a = ndimage.label(A, structure=CONN)
        _, ncomp_b = ndimage.label(B, structure=CONN)

        xor = A ^ B
        xor_area = int(xor.sum())
        union = A | B
        ys, xs = np.where(union)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        uh = max(y1 - y0, 1); uw = max(x1 - x0, 1)
        if xor_area > 0:
            yx, xx = np.where(xor)
            cy, cx = yx.mean(), xx.mean()
            rel_y = (cy - y0) / uh; rel_x = (cx - x0) / uw
        else:
            rel_y = rel_x = float("nan")
        # frame contrast inside the vertebra ROI (union bbox): std of intensities
        try:
            frame = load_frame(split, vf)
            roi = frame[y0:y1 + 1, x0:x1 + 1]
            contrast = float(roi.std())
        except Exception:  # noqa: BLE001
            contrast = float("nan")

        rows.append({
            "video_frame": vf, "video": vf.split("_")[0], "split": split,
            "labeler_A": la, "labeler_B": lb,
            "resolution": f"{ma.shape[0]}x{ma.shape[1]}",
            "dice": round(d, 5), "area_A": area_a, "area_B": area_b,
            "rel_area_diff": round(rel_area, 5),
            "ncomp_A": int(ncomp_a), "ncomp_B": int(ncomp_b),
            "xor_area": xor_area,
            "xor_rel_x": round(rel_x, 5) if rel_x == rel_x else "",
            "xor_rel_y": round(rel_y, 5) if rel_y == rel_y else "",
            "contrast": round(contrast, 3) if contrast == contrast else "",
            "batch": b,
        })
        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{len(pairs)}  (errors={len(errors)})")

    print(f"  computed {len(rows)} frames; load errors {len(errors)}; "
          f"Dice mismatches vs CSV {dice_mismatch}")
    if errors:
        for vf, e in errors[:10]: print(f"    ERROR {vf}: {e}")

    d = np.array([r["dice"] for r in rows])
    mean, std, med = d.mean(), d.std(), np.median(d)
    print(f"  RECOMPUTED  mean={mean:.4f}  std={std:.4f}  median={med:.4f}")

    # Verification gate (as requested): stop if it disagrees -> bad pairing
    if len(rows) != 898:
        print(f"\n  *** STOP: expected 898 frames, got {len(rows)}. "
              f"Load errors prevent full verification. ***"); sys.exit(2)
    if abs(mean - 0.874) > 0.003 or abs(med - 0.899) > 0.003:
        print(f"\n  *** STOP: mean/median disagree with 0.874/0.899 "
              f"-> pairing likely wrong. ***"); sys.exit(3)
    if dice_mismatch > 0:
        print(f"\n  *** STOP: {dice_mismatch} frames disagree with canonical CSV Dice. ***")
        sys.exit(4)
    print("  Verification OK (mean 0.874 +/- 0.070, median 0.899, matches canonical CSV).")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["video_frame", "video", "split", "labeler_A", "labeler_B", "resolution",
              "dice", "area_A", "area_B", "rel_area_diff", "ncomp_A", "ncomp_B",
              "xor_area", "xor_rel_x", "xor_rel_y", "contrast", "batch"]
    with open(OUT_DIR / "per_frame_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"  saved {OUT_DIR / 'per_frame_metrics.csv'}")
    return rows, float(mean), float(med)

# ----------------------------------------------------------------------------
# STEP 3 — candidate selection
# ----------------------------------------------------------------------------
def _take(sorted_rows, n=N_PER_GROUP, per_video=MAX_PER_VIDEO):
    out = []; seen = defaultdict(int)
    for r in sorted_rows:
        if seen[r["video"]] >= per_video:
            continue
        out.append(r); seen[r["video"]] += 1
        if len(out) >= n:
            break
    return out

def step3_select(rows, median):
    print("\n" + "=" * 70); print("STEP 3: candidate selection"); print("=" * 70)
    has_xor = [r for r in rows if r["xor_rel_y"] != ""]
    contrasts = np.array([r["contrast"] for r in rows if r["contrast"] != ""])
    p90 = float(np.percentile([r["dice"] for r in rows], 90))
    med_contrast = float(np.median(contrasts)) if len(contrasts) else 0.0

    groups = {}
    # (a) typical agreement: Dice closest to the median
    groups["a_typical"] = _take(sorted(rows, key=lambda r: abs(r["dice"] - median)))
    # (b) strong disagreement: lowest Dice
    groups["b_strong_disagree"] = _take(sorted(rows, key=lambda r: r["dice"]))
    # (c) extra/missing vertebra: component count differs OR rel area diff > 15%
    c = [r for r in rows if r["ncomp_A"] != r["ncomp_B"] or r["rel_area_diff"] > 0.15]
    groups["c_extra_missing"] = _take(sorted(c, key=lambda r: -r["rel_area_diff"]))
    # (d) upper-boundary disagreement: XOR centroid in the upper third of the union bbox
    d = [r for r in has_xor if float(r["xor_rel_y"]) < 1.0 / 3.0]
    groups["d_upper_boundary"] = _take(sorted(d, key=lambda r: -r["xor_area"]))
    # (e) clean frame: high Dice AND good contrast
    e = [r for r in rows if r["dice"] >= p90 and r["contrast"] != "" and r["contrast"] >= med_contrast]
    groups["e_clean"] = _take(sorted(e, key=lambda r: -r["dice"]))

    for g, lst in groups.items():
        print(f"  {g:20s}: {len(lst)} candidates")
    return groups

# ----------------------------------------------------------------------------
# STEP 4 — contact sheets + histogram + manifest
# ----------------------------------------------------------------------------
def _bbox_pad(mask_bool, pad_frac=0.25, shape=None):
    ys, xs = np.where(mask_bool)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    ph = int((y1 - y0 + 1) * pad_frac) + 5; pw = int((x1 - x0 + 1) * pad_frac) + 5
    H, W = shape
    return max(0, y0 - ph), min(H, y1 + ph + 1), max(0, x0 - pw), min(W, x1 + pw + 1)

def _panels_for(rows_subset):
    """Yield (row, frame, mask_a, mask_b) loading native-res data."""
    batch_of = {r["video_frame"]: str(r["batch"])
                for r in csv.DictReader(open(METADATA_CSV, encoding="utf-8"))}
    for r in rows_subset:
        vf = r["video_frame"]; b = batch_of[vf]
        try:
            ma = process_mask(tif_path(b, r["labeler_A"], vf))
            mb = process_mask(tif_path(b, r["labeler_B"], vf))
            fr = load_frame(r["split"], vf)
        except Exception as e:  # noqa: BLE001
            print(f"    panel load failed {vf}: {e}"); continue
        yield r, fr, ma, mb

def _draw_sheet(cands, title, out_png):
    n = len(cands)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes.reshape(1, 3)
    for ax_row, (r, fr, ma, mb) in zip(axes, _panels_for(cands)):
        A = ma > 127; B = mb > 127
        # col0: original grayscale
        ax_row[0].imshow(fr, cmap="gray"); ax_row[0].set_axis_off()
        ax_row[0].set_title(f"{r['video_frame']}  [{r['split']}]", fontsize=8)
        # col1: overlay contours (A vs B), contours only
        ax_row[1].imshow(fr, cmap="gray")
        ax_row[1].contour(A, levels=[0.5], colors=COLOR_A, linewidths=1.0)
        ax_row[1].contour(B, levels=[0.5], colors=COLOR_B, linewidths=1.0)
        ax_row[1].set_axis_off()
        ax_row[1].set_title(f"Dice={r['dice']:.3f}  dA={r['rel_area_diff']:.2f}  "
                            f"comp {r['ncomp_A']}/{r['ncomp_B']}", fontsize=8)
        # col2: zoom to XOR region (or union if XOR empty)
        xor = A ^ B
        bb = _bbox_pad(xor if xor.any() else (A | B), shape=fr.shape)
        if bb:
            y0, y1, x0, x1 = bb
            ax_row[2].imshow(fr[y0:y1, x0:x1], cmap="gray")
            ax_row[2].contour(A[y0:y1, x0:x1], levels=[0.5], colors=COLOR_A, linewidths=1.2)
            ax_row[2].contour(B[y0:y1, x0:x1], levels=[0.5], colors=COLOR_B, linewidths=1.2)
        ax_row[2].set_axis_off(); ax_row[2].set_title("XOR zoom", fontsize=8)
    # legend
    fig.legend(handles=[plt.Line2D([0], [0], color=COLOR_A, lw=2, label="Annotator A"),
                        plt.Line2D([0], [0], color=COLOR_B, lw=2, label="Annotator B")],
               loc="upper right", fontsize=9)
    fig.suptitle(title, fontsize=11, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"    saved {out_png}")

def step4_figures(groups, rows, mean, median):
    print("\n" + "=" * 70); print("STEP 4: contact sheets + histogram + manifest"); print("=" * 70)
    GROUP_TITLES = {
        "a_typical": "(a) Typical agreement (Dice near median)",
        "b_strong_disagree": "(b) Strong disagreement (lowest Dice)",
        "c_extra_missing": "(c) Extra/missing vertebra (component or area diff)",
        "d_upper_boundary": "(d) Upper-boundary disagreement (XOR in upper third)",
        "e_clean": "(e) Clean frame (high Dice + good contrast)",
    }
    manifest = []
    for g, cands in groups.items():
        sheets = [cands[i:i + PER_SHEET] for i in range(0, len(cands), PER_SHEET)]
        for si, sheet in enumerate(sheets, 1):
            out_png = OUT_DIR / f"contact_{g}_{si}.png"
            _draw_sheet(sheet, f"{GROUP_TITLES[g]}  — sheet {si}", out_png)
            for rank, r in enumerate(sheet, 1):
                manifest.append({
                    "group": g, "rank": (si - 1) * PER_SHEET + rank,
                    "video": r["video"], "frame": r["video_frame"], "split": r["split"],
                    "dice": r["dice"], "rel_area_diff": r["rel_area_diff"],
                    "ncomp_A": r["ncomp_A"], "ncomp_B": r["ncomp_B"],
                    "xor_rel_y": r["xor_rel_y"], "contrast": r["contrast"],
                    "png_path": str(out_png.relative_to(PROJECT)),
                })

    # histogram of all 898 Dice
    d = np.array([r["dice"] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(d, bins=40, color="#4C72B0", edgecolor="white")
    ax.axvline(mean, color="crimson", ls="--", lw=1.5, label=f"mean {mean:.3f}")
    ax.axvline(median, color="green", ls="--", lw=1.5, label=f"median {median:.3f}")
    ax.set_xlabel("Dice (annotator A vs B)"); ax.set_ylabel("frames")
    ax.set_title(f"INCA inter-annotator Dice (n={len(d)})"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT_DIR / "dice_histogram.png", dpi=130); plt.close(fig)
    print(f"    saved {OUT_DIR / 'dice_histogram.png'}")

    mfields = ["group", "rank", "video", "frame", "split", "dice", "rel_area_diff",
               "ncomp_A", "ncomp_B", "xor_rel_y", "contrast", "png_path"]
    with open(OUT_DIR / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=mfields); w.writeheader(); w.writerows(manifest)
    print(f"    saved {OUT_DIR / 'manifest.csv'} ({len(manifest)} rows)")

def main():
    print(f"Output dir: {OUT_DIR}")
    print("READ-ONLY over original masks/frames. Seed =", SEED)
    rows, mean, median = step2_metrics()
    groups = step3_select(rows, median)
    step4_figures(groups, rows, mean, median)
    print("\nDONE.")

if __name__ == "__main__":
    main()
