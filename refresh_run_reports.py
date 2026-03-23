"""
refresh_run_reports.py
======================
Regenerates (overwrites) *_run_report.json in place for all existing run
directories, using only the artifacts already present in each folder.
No retraining or re-evaluation is required.

Usage (Colab):
    !python refresh_run_reports.py \\
        --runs_root /content/drive/MyDrive/UNM_vertebras_seg_v3/runs

Usage (local Windows):
    python refresh_run_reports.py --runs_root "G:/My Drive/UNM_vertebras_seg_v3/runs"

Artifacts read (all optional — missing ones become null in the report):
    config.json                  required; run is skipped if absent
    test_metrics.csv             required; run is skipped if absent
    diagnostic_summary.json      optional
    diagnostics_epoch.csv        optional (best-epoch pl_stats + val_boundary)
    test_boundary_metrics.csv    optional
    val_boundary_metrics.csv     optional
    c2c4_comparison.csv          optional
    *_run_report.json (existing) optional (used to recover val_metrics if present)
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

# ── make tesis_seg/src importable when run from tesis_seg/ ──────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from src.evaluate import write_run_report  # noqa: E402  (after sys.path patch)


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_test_metrics_csv(path: Path) -> dict | None:
    """Read a metric,value CSV (as written by write_simple_metrics_csv)."""
    if not path.is_file():
        return None
    try:
        d = {}
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    d[row["metric"]] = float(row["value"])
                except (ValueError, KeyError):
                    d[row["metric"]] = row.get("value")
        return d or None
    except Exception:
        return None


def _recover_val_metrics(run_dir: Path) -> dict | None:
    """
    Try to recover val_metrics from an existing *_run_report.json.
    Returns None if no report exists or the field is absent/null.
    """
    reports = sorted(run_dir.glob("*_run_report.json"))
    for rpt in reports:
        data = _load_json(rpt)
        if data and data.get("val_metrics"):
            return data["val_metrics"]
    return None


# ── per-run logic ─────────────────────────────────────────────────────────────

class RunResult:
    OK      = "ok"
    SKIPPED = "skipped"

    def __init__(self, run_dir: Path, status: str, reason: str = ""):
        self.run_dir = run_dir
        self.status  = status
        self.reason  = reason

    def __str__(self):
        label = f"{self.run_dir.parent.name}/{self.run_dir.name}"
        if self.reason:
            return f"  [{self.status.upper():7s}] {label}  ({self.reason})"
        return f"  [{self.status.upper():7s}] {label}"


def refresh_one(run_dir: Path) -> RunResult:
    cfg_path         = run_dir / "config.json"
    test_metrics_path = run_dir / "test_metrics.csv"

    if not cfg_path.is_file():
        return RunResult(run_dir, RunResult.SKIPPED, "config.json missing")
    if not test_metrics_path.is_file():
        return RunResult(run_dir, RunResult.SKIPPED, "test_metrics.csv missing")

    cfg = _load_json(cfg_path)
    if cfg is None:
        return RunResult(run_dir, RunResult.SKIPPED, "config.json unreadable")

    test_metrics = _read_test_metrics_csv(test_metrics_path)
    if test_metrics is None:
        return RunResult(run_dir, RunResult.SKIPPED, "test_metrics.csv unreadable")

    # Recover val_metrics from existing report if available (may be None for old runs)
    val_metrics = _recover_val_metrics(run_dir)

    # Ensure exp_dir points to this run_dir so write_run_report reads artifacts correctly
    cfg["exp_dir"] = str(run_dir)

    # write_run_report() reads everything else from exp_dir internally:
    #   diagnostic_summary.json, diagnostics_epoch.csv (new best-epoch fields),
    #   test/val_boundary_metrics.csv, c2c4_comparison.csv
    try:
        report_path = write_run_report(
            cfg=cfg,
            test_metrics=test_metrics,
            exp_dir=str(run_dir),
            val_metrics=val_metrics,
        )
        return RunResult(run_dir, RunResult.OK, f"-> {Path(report_path).name}")
    except Exception as exc:
        return RunResult(run_dir, RunResult.SKIPPED, f"write_run_report failed: {exc}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Refresh *_run_report.json files in place from existing run artifacts."
    )
    parser.add_argument(
        "--runs_root", type=str, default=None,
        help="Path to the runs/ directory. Defaults to ../runs relative to this script.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be refreshed without writing anything.",
    )
    args = parser.parse_args()

    if args.runs_root is None:
        args.runs_root = str(_HERE.parent / "runs")

    runs_root = Path(args.runs_root)
    if not runs_root.is_dir():
        print(f"[ERROR] runs_root not found: {runs_root}")
        return

    seed_pattern = re.compile(r"^seed_(\d+)$")
    results: list[RunResult] = []

    for exp_dir in sorted(runs_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        for seed_dir in sorted(exp_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_pattern.match(seed_dir.name):
                continue
            if args.dry_run:
                print(f"  [DRY RUN] would refresh: {exp_dir.name}/{seed_dir.name}")
                continue
            result = refresh_one(seed_dir)
            results.append(result)
            print(result)

    if args.dry_run:
        return

    n_ok      = sum(1 for r in results if r.status == RunResult.OK)
    n_skipped = sum(1 for r in results if r.status == RunResult.SKIPPED)

    print(f"\n{'='*55}")
    print(f"Refresh complete")
    print(f"  Runs found   : {len(results)}")
    print(f"  Refreshed OK : {n_ok}")
    print(f"  Skipped      : {n_skipped}")
    if n_skipped:
        print("\nSkipped runs (reasons above):")
        for r in results:
            if r.status == RunResult.SKIPPED:
                print(f"  {r.run_dir.parent.name}/{r.run_dir.name}: {r.reason}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
