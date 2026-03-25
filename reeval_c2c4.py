"""
reeval_c2c4.py
==============
Re-runs the C2-C4 post-processing evaluation on all existing experiment folders
WITHOUT retraining or re-running model inference.

For each seed folder it:
  1. Reads config.json
  2. Verifies test_preds/ exists
  3. Calls compare_c2c4_manual_vs_auto() → overwrites c2c4_comparison.csv
  4. Prints a one-line summary per run

Usage (local Windows):
    python reeval_c2c4.py --runs_root "G:/My Drive/UNM_vertebras_seg_v3/runs" ^
        --rotulos_dir "G:/My Drive/UNM TalkBank Dysphagia/rotulos"

Usage (Colab):
    !python reeval_c2c4.py \\
        --runs_root /content/drive/MyDrive/UNM_vertebras_seg_v3/runs \\
        --rotulos_dir "/content/drive/MyDrive/UNM TalkBank Dysphagia/rotulos"

Note: --rotulos_dir is required because config.json stores Colab paths that are
not usable in a local environment.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

# Allow running from tesis_seg/ or from workspace root
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from src.ruler_eval import compare_c2c4_manual_vs_auto, c2c4_landmark_summary


def _read_config(run_dir: Path) -> dict | None:
    cfg_path = run_dir / "config.json"
    if not cfg_path.is_file():
        return None
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  [WARN] could not read {cfg_path}: {exc}")
        return None


def reeval_one(run_dir: Path, exp_name: str, seed: str, rotulos_dir: str, dry_run: bool) -> dict | None:
    """
    Re-run c2c4 evaluation for one seed directory.
    Returns a summary dict, or None if skipped.
    """
    cfg = _read_config(run_dir)
    if cfg is None:
        print(f"  [SKIP] {exp_name}/seed_{seed}: config.json missing or unreadable")
        return None

    pred_masks_dir = run_dir / "test_preds"
    if not pred_masks_dir.is_dir():
        print(f"  [SKIP] {exp_name}/seed_{seed}: test_preds/ not found")
        return None

    out_csv = run_dir / "c2c4_comparison.csv"
    target_size = tuple(cfg.get("target_size", [320, 320]))

    if dry_run:
        print(f"  [DRY RUN] would reeval {exp_name}/seed_{seed} -> {out_csv.name}")
        return None

    try:
        df = compare_c2c4_manual_vs_auto(
            rotulos_dir=rotulos_dir,
            pred_masks_dir=str(pred_masks_dir),
            out_csv=str(out_csv),
            target_size=target_size,
        )
    except Exception as exc:
        print(f"  [ERROR] {exp_name}/seed_{seed}: {exc}")
        return None

    summary = c2c4_landmark_summary(df) or {}
    n        = summary.get("n_valid", 0)
    mae      = summary.get("mean_abs_err_px", float("nan"))
    lm_mae   = summary.get("mean_err_landmark_mean_px", float("nan"))
    pct5     = summary.get("pct_both_lt5px", float("nan"))

    print(
        f"  {exp_name:<30} seed={seed}  "
        f"n={n}  mae={mae:.1f}px  lm_mae={lm_mae:.1f}px  pct_both<5px={pct5:.0f}%"
    )

    return {"exp_name": exp_name, "seed": seed, **summary}


def main():
    parser = argparse.ArgumentParser(
        description="Re-run C2-C4 post-processing on all existing experiment runs."
    )
    parser.add_argument(
        "--runs_root", type=str, default=None,
        help="Path to the runs/ directory. Default: ../runs relative to this script.",
    )
    parser.add_argument(
        "--rotulos_dir", type=str, required=True,
        help="Path to the manual annotation (rotulos) directory.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Scan and print what would be done without writing any files.",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root) if args.runs_root else _HERE.parent / "runs"
    if not runs_root.is_dir():
        print(f"[ERROR] runs_root not found: {runs_root}")
        sys.exit(1)

    seed_pattern = re.compile(r"^seed_(\d+)$")
    results = []

    print(f"Scanning: {runs_root}")
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
            result = reeval_one(
                run_dir=seed_dir,
                exp_name=exp_name,
                seed=seed,
                rotulos_dir=args.rotulos_dir,
                dry_run=args.dry_run,
            )
            if result is not None:
                results.append(result)

    if not results:
        print("\nNo runs processed.")
        return

    # Final summary table sorted by (exp_name, seed)
    results.sort(key=lambda r: (r["exp_name"], r["seed"]))
    print("\n" + "=" * 75)
    print(f"{'Experiment':<30} {'Seed':>4}  {'N':>4}  {'MAE':>7}  {'LM_MAE':>8}  {'Pct<5px':>8}")
    print("-" * 75)
    for r in results:
        n      = r.get("n_valid", "?")
        mae    = r.get("mean_abs_err_px", float("nan"))
        lm_mae = r.get("mean_err_landmark_mean_px", float("nan"))
        pct5   = r.get("pct_both_lt5px", float("nan"))
        print(
            f"{r['exp_name']:<30} {r['seed']:>4}  {n:>4}  "
            f"{mae:>6.1f}px  {lm_mae:>7.1f}px  {pct5:>6.0f}%"
        )
    print("=" * 75)
    print(f"\nDone. {len(results)} run(s) processed.")
    print("Run refresh_run_reports.py afterward to update run_report.json files.")


if __name__ == "__main__":
    main()