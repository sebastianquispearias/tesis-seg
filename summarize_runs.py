"""
summarize_runs.py
=================
Scans all experiment folders under a runs root directory and produces a
consolidated boundary_metrics_summary.csv, then collects all run_report JSON
files into a central folder for easy review.

Usage (Colab):
    !python summarize_runs.py --runs_root /content/drive/MyDrive/UNM_vertebras_seg_v3/runs

Usage (local Windows):
    python summarize_runs.py --runs_root "G:/My Drive/UNM_vertebras_seg_v3/runs"

The script expects folders matching the pattern:
    {runs_root}/{experiment_name}/seed_{seed}/

For each run it looks for:
    test_boundary_metrics.csv  (per-image BF1/ASSD/HD95 for test split)
    val_boundary_metrics.csv   (per-image BF1/ASSD/HD95 for val split)
    *_run_report.json          (portable run summary — glob pattern)

Output:
    {runs_root}/boundary_metrics_summary.csv
    {runs_root}/reports/run_reports/*.json   (collected run_report files)
"""

import argparse
import csv
import json
import math
import os
import re
import shutil
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def _mean_std(values):
    """Return (mean, std) rounded to 4 dp, or (None, None) if list is empty."""
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    s = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
    return round(m, 4), round(s, 4)


def aggregate_boundary_csv(path: Path) -> dict | None:
    """
    Read a per-image boundary CSV (columns: idx, name, bf1, assd, hd95).
    Returns an aggregate dict or None if the file is missing or unreadable.
    """
    if not path.is_file():
        return None
    try:
        bf1s, assds, hd95s = [], [], []
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                bf1s.append(float(row["bf1"]))
                a, h = float(row["assd"]), float(row["hd95"])
                if math.isfinite(a):
                    assds.append(a)
                if math.isfinite(h):
                    hd95s.append(h)
        if not bf1s:
            return None
        bf1_m,  bf1_s  = _mean_std(bf1s)
        assd_m, assd_s = _mean_std(assds)
        hd95_m, hd95_s = _mean_std(hd95s)
        return {
            "n_images":    len(bf1s),
            "bf1_mean":    bf1_m,
            "bf1_std":     bf1_s,
            "assd_mean_px": assd_m,
            "assd_std_px":  assd_s,
            "hd95_mean_px": hd95_m,
            "hd95_std_px":  hd95_s,
        }
    except Exception as exc:
        print(f"  [WARN] could not read {path}: {exc}")
        return None


def scan_run(run_dir: Path, exp_name: str, seed: str) -> list[dict]:
    """
    Inspect one run directory and return one row per split (test / val).
    Always returns a row even if files are absent (status='missing').
    """
    rows = []
    has_report = bool(sorted(run_dir.glob("*_run_report.json")))

    for split in ("test", "val"):
        csv_path = run_dir / f"{split}_boundary_metrics.csv"
        agg = aggregate_boundary_csv(csv_path)

        if agg is None:
            row = {
                "experiment_name": exp_name,
                "seed":            seed,
                "split":           split,
                "n_images":        None,
                "bf1_mean":        None,
                "bf1_std":         None,
                "assd_mean_px":    None,
                "assd_std_px":     None,
                "hd95_mean_px":    None,
                "hd95_std_px":     None,
                "has_run_report":  has_report,
                "source_file":     str(csv_path) if csv_path.is_file() else "MISSING",
                "status":          "ok" if csv_path.is_file() else "missing",
            }
        else:
            row = {
                "experiment_name": exp_name,
                "seed":            seed,
                "split":           split,
                **agg,
                "has_run_report":  has_report,
                "source_file":     str(csv_path),
                "status":          "ok",
            }
        rows.append(row)

    return rows


def scan_all(runs_root: Path) -> list[dict]:
    """
    Walk runs_root looking for {exp_name}/seed_{seed}/ directories.
    Returns list of row dicts for the summary CSV.
    """
    rows = []
    seed_pattern = re.compile(r"^seed_(\d+)$")

    if not runs_root.is_dir():
        print(f"[ERROR] runs_root not found: {runs_root}")
        return rows

    for exp_dir in sorted(runs_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        exp_name = exp_dir.name

        for seed_dir in sorted(exp_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            m = seed_pattern.match(seed_dir.name)
            if not m:
                continue
            seed = m.group(1)
            rows.extend(scan_run(seed_dir, exp_name, seed))

    return rows


# ── main ─────────────────────────────────────────────────────────────────────

FIELDNAMES = [
    "experiment_name", "seed", "split",
    "n_images",
    "bf1_mean", "bf1_std",
    "assd_mean_px", "assd_std_px",
    "hd95_mean_px", "hd95_std_px",
    "has_run_report", "source_file", "status",
]


def main():
    parser = argparse.ArgumentParser(description="Consolidate boundary metrics across all experiment runs.")
    parser.add_argument(
        "--runs_root",
        type=str,
        default=None,
        help="Path to the runs/ directory (e.g. /content/drive/MyDrive/UNM_vertebras_seg_v3/runs)",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Output CSV path (default: {runs_root}/boundary_metrics_summary.csv)",
    )
    args = parser.parse_args()

    # Default: look for runs/ sibling to this script's directory
    if args.runs_root is None:
        script_dir = Path(__file__).resolve().parent
        args.runs_root = str(script_dir.parent / "runs")

    runs_root = Path(args.runs_root)
    out_csv = Path(args.out_csv) if args.out_csv else runs_root / "boundary_metrics_summary.csv"

    print(f"Scanning: {runs_root}")
    rows = scan_all(runs_root)

    if not rows:
        print("[WARN] No run directories found. Check --runs_root.")
        return

    # Print summary to stdout
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_missing = sum(1 for r in rows if r["status"] == "missing")
    print(f"\nFound {len(rows)} split-rows across {len(rows)//2} runs  "
          f"({n_ok} OK, {n_missing} missing boundary files)")

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)

    print(f"\nSummary saved: {out_csv}")

    # Print a quick readable table for test split
    print("\n--- TEST split summary ---")
    test_rows = [r for r in rows if r["split"] == "test" and r["status"] == "ok"]
    if test_rows:
        hdr = f"{'experiment':<28} {'seed':>4}  {'BF1':>6}  {'ASSD':>6}  {'HD95':>7}"
        print(hdr)
        print("-" * len(hdr))
        for r in test_rows:
            print(
                f"{r['experiment_name']:<28} {r['seed']:>4}  "
                f"{r['bf1_mean']:>6.4f}  {r['assd_mean_px']:>6.2f}  {r['hd95_mean_px']:>7.2f}"
            )
    else:
        print("  (no complete test runs found)")

    # ── Collect run_report files into one central folder ─────────────────────
    reports_dir = runs_root / "reports" / "run_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    seed_pattern_collect = re.compile(r"^seed_(\d+)$")
    n_copied = 0
    for exp_d in sorted(runs_root.iterdir()):
        if not exp_d.is_dir():
            continue
        for seed_d in sorted(exp_d.iterdir()):
            mc = seed_pattern_collect.match(seed_d.name)
            if not mc:
                continue
            for rpt in sorted(seed_d.glob("*_run_report.json")):
                shutil.copy2(rpt, reports_dir / rpt.name)
                n_copied += 1

    print(f"\nCollected {n_copied} run_report file(s) -> {reports_dir}")


if __name__ == "__main__":
    main()
