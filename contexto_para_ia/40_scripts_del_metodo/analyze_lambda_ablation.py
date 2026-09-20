"""
analyze_lambda_ablation.py

Reads run reports from the lambda ablation runs (42 total) and produces:
  - resultados/lambda_ablation_detailed.csv (one row per run)
  - resultados/lambda_ablation_summary.csv (one row per dataset/exp/lambda)
  - resultados/lambda_ablation_decision.md (comparison + recommendation)

For each run extracts:
  - test sample_mean_f1 (from run_report.json -> test_metrics)
  - best_epoch (from training_summary)
  - semi_start_epoch (from config, if SSL)
  - patience_es (from config)
  - lambda_u (from config, if SSL)
  - total_epochs_run (from train_log.csv last row)
  - best_epoch >= semi_start_epoch indicator

Decision rules (all must be considered, not just best mean):
  - Consistent performance across seeds
  - Stability (low std)
  - Fair comparison vs supervised (same patience)
  - Reduction of pre-SSL checkpoints (best_epoch >= semi_start_epoch in all 3 seeds)

Usage:
    python analyze_lambda_ablation.py
"""

import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

PROJECT_ROOT = Path(r"G:/My Drive/UNM_vertebras_seg_v3")
# Ablation output roots (separated from historical runs)
RUNS_UNM = PROJECT_ROOT / "runs_lambda_ablation"
RUNS_INCA = PROJECT_ROOT / "runs_inca_lambda_ablation"
OUT_DIR = PROJECT_ROOT / "resultados"
OUT_DETAILED = OUT_DIR / "lambda_ablation_detailed.csv"
OUT_SUMMARY = OUT_DIR / "lambda_ablation_summary.csv"
OUT_MD = OUT_DIR / "lambda_ablation_decision.md"


# ─────────────────────────────────────────────
# Run inventory
# ─────────────────────────────────────────────

# Each entry: (dataset, experiment_tag, lambda_or_None, exp_dir_name, is_supervised)
RUN_INVENTORY = []

# UNM supervised baseline
for seed in [0, 1, 2]:
    RUN_INVENTORY.append({
        "dataset": "UNM",
        "runs_root": RUNS_UNM,
        "experiment": "supervised",
        "lambda_u": None,
        "exp_dir_name": "supervised_p40",
        "seed": seed,
        "is_supervised": True,
        "schedule": "p40",
    })

# UNM SSL experiments
for exp in ["semi_all_lateral", "mean_teacher_all_lateral", "semi_r10", "mean_teacher_r10"]:
    for lam_val, lam_tag in [(0.05, "lam005"), (0.1, "lam010")]:
        for seed in [0, 1, 2]:
            RUN_INVENTORY.append({
                "dataset": "UNM",
                "runs_root": RUNS_UNM,
                "experiment": exp,
                "lambda_u": lam_val,
                "exp_dir_name": f"{exp}_{lam_tag}_s15p40",
                "seed": seed,
                "is_supervised": False,
                "schedule": "s15p40",
            })

# INCA supervised baseline
for seed in [0, 1, 2]:
    RUN_INVENTORY.append({
        "dataset": "INCA_v2",
        "runs_root": RUNS_INCA,
        "experiment": "supervised_inca",
        "lambda_u": None,
        "exp_dir_name": "supervised_inca_p40",
        "seed": seed,
        "is_supervised": True,
        "schedule": "p40",
    })

# INCA SSL experiments
for exp in ["semi_inca_all_lateral", "mean_teacher_inca_all_lateral"]:
    for lam_val, lam_tag in [(0.05, "lam005"), (0.1, "lam010")]:
        for seed in [0, 1, 2]:
            RUN_INVENTORY.append({
                "dataset": "INCA_v2",
                "runs_root": RUNS_INCA,
                "experiment": exp,
                "lambda_u": lam_val,
                "exp_dir_name": f"{exp}_{lam_tag}_s7p40",
                "seed": seed,
                "is_supervised": False,
                "schedule": "s7p40",
            })

assert len(RUN_INVENTORY) == 42


# ─────────────────────────────────────────────
# Data extraction
# ─────────────────────────────────────────────

def load_run_report(exp_dir: Path, exp_dir_name: str, seed: int):
    """Load {exp_dir_name}_seed_{seed}_run_report.json from exp_dir."""
    report_path = exp_dir / f"{exp_dir_name}_seed_{seed}_run_report.json"
    if not report_path.is_file():
        return None, report_path
    with open(report_path, encoding="utf-8") as f:
        return json.load(f), report_path


def get_total_epochs(exp_dir: Path):
    """Return total epochs run from train_log.csv (last epoch + 1)."""
    train_log = exp_dir / "train_log.csv"
    if not train_log.is_file():
        return None
    max_epoch = -1
    with open(train_log, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                e = int(row.get("epoch", -1))
                if e > max_epoch:
                    max_epoch = e
            except (ValueError, TypeError):
                continue
    return max_epoch + 1 if max_epoch >= 0 else None


def extract_run_data(entry):
    """Return dict with all fields for detailed CSV, or None-filled if MISSING."""
    exp_dir = entry["runs_root"] / entry["exp_dir_name"] / f"seed_{entry['seed']}"
    report, report_path = load_run_report(exp_dir, entry["exp_dir_name"], entry["seed"])

    row = {
        "dataset": entry["dataset"],
        "experiment": entry["experiment"],
        "exp_dir_name": entry["exp_dir_name"],
        "seed": entry["seed"],
        "schedule": entry["schedule"],
        "is_supervised": entry["is_supervised"],
        "lambda_u": entry["lambda_u"],
        "report_path": str(report_path),
        "status": "MISSING",
        "test_sample_mean_f1": None,
        "best_epoch": None,
        "semi_start_epoch": None,
        "patience_es": None,
        "cfg_lambda_u": None,
        "total_epochs_run": None,
        "best_epoch_geq_semi_start": None,
    }

    if report is None:
        return row

    row["status"] = "OK"

    tm = report.get("test_metrics", {}) or {}
    row["test_sample_mean_f1"] = tm.get("sample_mean_f1") or tm.get("f1_mean")

    ts = report.get("training_summary", {}) or {}
    row["best_epoch"] = ts.get("best_epoch")

    cfg = report.get("config", {}) or {}
    row["semi_start_epoch"] = cfg.get("semi_start_epoch")
    row["patience_es"] = cfg.get("patience_es")
    row["cfg_lambda_u"] = cfg.get("lambda_u")

    row["total_epochs_run"] = get_total_epochs(exp_dir)

    if not entry["is_supervised"] and row["best_epoch"] is not None and row["semi_start_epoch"] is not None:
        row["best_epoch_geq_semi_start"] = row["best_epoch"] >= row["semi_start_epoch"]

    return row


# ─────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────

def aggregate(detailed_rows):
    """Group by (dataset, experiment, lambda_u) and compute stats across seeds."""
    groups = defaultdict(list)
    for r in detailed_rows:
        key = (r["dataset"], r["experiment"], r["lambda_u"])
        groups[key].append(r)

    summary = []
    for (dataset, experiment, lam), rows in groups.items():
        f1_vals = [r["test_sample_mean_f1"] for r in rows if r["test_sample_mean_f1"] is not None]
        best_epochs = [r["best_epoch"] for r in rows if r["best_epoch"] is not None]
        total_epochs = [r["total_epochs_run"] for r in rows if r["total_epochs_run"] is not None]

        n_seeds_found = len(rows)
        n_seeds_ok = sum(1 for r in rows if r["status"] == "OK")
        n_missing = n_seeds_found - n_seeds_ok

        mean_f1 = statistics.mean(f1_vals) if f1_vals else None
        std_f1 = statistics.stdev(f1_vals) if len(f1_vals) > 1 else (0.0 if f1_vals else None)
        min_f1 = min(f1_vals) if f1_vals else None
        max_f1 = max(f1_vals) if f1_vals else None

        mean_best_epoch = statistics.mean(best_epochs) if best_epochs else None
        mean_total_epochs = statistics.mean(total_epochs) if total_epochs else None

        # Semi_start_epoch should be fixed per group (take from first valid row)
        semi_start = next((r["semi_start_epoch"] for r in rows if r["semi_start_epoch"] is not None), None)

        # Count seeds with best_epoch >= semi_start_epoch
        is_supervised = rows[0]["is_supervised"]
        if is_supervised:
            n_seeds_post_ssl = None
        else:
            n_seeds_post_ssl = sum(1 for r in rows if r["best_epoch_geq_semi_start"] is True)

        summary.append({
            "dataset": dataset,
            "experiment": experiment,
            "lambda_u": lam,
            "is_supervised": is_supervised,
            "n_seeds_found": n_seeds_found,
            "n_seeds_ok": n_seeds_ok,
            "n_missing": n_missing,
            "f1_mean": round(mean_f1, 4) if mean_f1 is not None else None,
            "f1_std": round(std_f1, 4) if std_f1 is not None else None,
            "f1_min": round(min_f1, 4) if min_f1 is not None else None,
            "f1_max": round(max_f1, 4) if max_f1 is not None else None,
            "f1_seed0": round(rows[0]["test_sample_mean_f1"], 4) if len(rows) > 0 and rows[0]["test_sample_mean_f1"] is not None else None,
            "f1_seed1": round(rows[1]["test_sample_mean_f1"], 4) if len(rows) > 1 and rows[1]["test_sample_mean_f1"] is not None else None,
            "f1_seed2": round(rows[2]["test_sample_mean_f1"], 4) if len(rows) > 2 and rows[2]["test_sample_mean_f1"] is not None else None,
            "semi_start_epoch": semi_start,
            "mean_best_epoch": round(mean_best_epoch, 1) if mean_best_epoch is not None else None,
            "mean_total_epochs_run": round(mean_total_epochs, 1) if mean_total_epochs is not None else None,
            "n_seeds_best_epoch_geq_semi_start": n_seeds_post_ssl,
        })

    # Sort: dataset, experiment, lambda
    summary.sort(key=lambda r: (r["dataset"], r["experiment"], r["lambda_u"] if r["lambda_u"] is not None else -1))
    return summary


# ─────────────────────────────────────────────
# Decision logic
# ─────────────────────────────────────────────

def decide_lambda(summary):
    """Compare lam=0.05 vs lam=0.1 across all (dataset, experiment) pairs.

    Returns a structured decision dict with per-condition scores and a final
    recommendation.
    """
    # For each (dataset, experiment) pair that is SSL, compare the two lambdas
    pairs = defaultdict(dict)  # {(dataset, experiment): {lambda: row}}
    for r in summary:
        if r["is_supervised"]:
            continue
        pairs[(r["dataset"], r["experiment"])][r["lambda_u"]] = r

    comparisons = []
    score_005 = 0
    score_010 = 0
    tied = 0

    for key, by_lam in sorted(pairs.items()):
        if 0.05 not in by_lam or 0.1 not in by_lam:
            continue
        r005 = by_lam[0.05]
        r010 = by_lam[0.1]

        # Compare on:
        # 1. mean F1 (higher better)
        # 2. std F1 (lower better)
        # 3. n_seeds_best_epoch_geq_semi_start (higher better, max 3)

        verdict = {
            "dataset": key[0],
            "experiment": key[1],
            "mean_005": r005["f1_mean"],
            "mean_010": r010["f1_mean"],
            "std_005": r005["f1_std"],
            "std_010": r010["f1_std"],
            "post_ssl_005": r005["n_seeds_best_epoch_geq_semi_start"],
            "post_ssl_010": r010["n_seeds_best_epoch_geq_semi_start"],
            "winner_mean": None,
            "winner_std": None,
            "winner_post_ssl": None,
            "overall_winner": None,
        }

        # Mean comparison (handle None)
        if r005["f1_mean"] is not None and r010["f1_mean"] is not None:
            if abs(r005["f1_mean"] - r010["f1_mean"]) < 0.002:
                verdict["winner_mean"] = "tie"
            elif r005["f1_mean"] > r010["f1_mean"]:
                verdict["winner_mean"] = "0.05"
            else:
                verdict["winner_mean"] = "0.1"

        # Std comparison (lower is better)
        if r005["f1_std"] is not None and r010["f1_std"] is not None:
            if abs(r005["f1_std"] - r010["f1_std"]) < 0.002:
                verdict["winner_std"] = "tie"
            elif r005["f1_std"] < r010["f1_std"]:
                verdict["winner_std"] = "0.05"
            else:
                verdict["winner_std"] = "0.1"

        # Post-SSL count comparison (higher is better)
        p5 = r005["n_seeds_best_epoch_geq_semi_start"]
        p1 = r010["n_seeds_best_epoch_geq_semi_start"]
        if p5 is not None and p1 is not None:
            if p5 == p1:
                verdict["winner_post_ssl"] = "tie"
            elif p5 > p1:
                verdict["winner_post_ssl"] = "0.05"
            else:
                verdict["winner_post_ssl"] = "0.1"

        # Overall winner: majority vote across the 3 criteria
        votes_005 = sum(1 for k in ["winner_mean", "winner_std", "winner_post_ssl"]
                        if verdict[k] == "0.05")
        votes_010 = sum(1 for k in ["winner_mean", "winner_std", "winner_post_ssl"]
                        if verdict[k] == "0.1")
        if votes_005 > votes_010:
            verdict["overall_winner"] = "0.05"
            score_005 += 1
        elif votes_010 > votes_005:
            verdict["overall_winner"] = "0.1"
            score_010 += 1
        else:
            verdict["overall_winner"] = "tie"
            tied += 1

        comparisons.append(verdict)

    return {
        "comparisons": comparisons,
        "score_005": score_005,
        "score_010": score_010,
        "tied": tied,
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("Analyzing lambda ablation results...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Extract per-run data
    detailed_rows = []
    n_missing = 0
    for entry in RUN_INVENTORY:
        row = extract_run_data(entry)
        detailed_rows.append(row)
        if row["status"] != "OK":
            n_missing += 1
            print(f"  MISSING: {row['exp_dir_name']}/seed_{row['seed']}")

    n_ok = len(detailed_rows) - n_missing
    print(f"\nRuns found: {n_ok}/{len(detailed_rows)} ({n_missing} missing)")

    # Save detailed CSV
    fieldnames = list(detailed_rows[0].keys())
    with open(OUT_DETAILED, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detailed_rows)
    print(f"Saved: {OUT_DETAILED}")

    # Aggregate
    summary_rows = aggregate(detailed_rows)
    with open(OUT_SUMMARY, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved: {OUT_SUMMARY}")

    # Decision
    decision = decide_lambda(summary_rows)

    # Build markdown report
    md = []
    md.append("# Lambda Ablation — Decision Report\n")
    md.append(f"**Runs analyzed**: {n_ok}/{len(detailed_rows)} ({n_missing} missing)\n")
    md.append("**Schedule**: UNM `semi_start_epoch=15, patience_es=40`, "
              "INCA v2 `semi_start_epoch=7, patience_es=40`\n")
    md.append("**Lambdas compared**: 0.05 vs 0.1\n\n")

    # Supervised baselines
    md.append("## Supervised baselines (new schedule, patience=40)\n\n")
    md.append("| Dataset | Experiment | f1_mean | f1_std | Seeds | Best epoch (mean) |\n")
    md.append("|---|---|---|---|---|---|\n")
    for r in summary_rows:
        if not r["is_supervised"]:
            continue
        md.append(
            f"| {r['dataset']} | {r['experiment']} | {r['f1_mean']} | {r['f1_std']} | "
            f"{r['n_seeds_ok']}/3 | {r['mean_best_epoch']} |\n"
        )
    md.append("\n")

    # SSL comparison tables
    md.append("## SSL results by (dataset, experiment, lambda)\n\n")
    md.append(
        "| Dataset | Experiment | λ | f1_mean±std | seed0/1/2 | Mean best_epoch | "
        "semi_start | n_seeds post-SSL |\n"
    )
    md.append("|---|---|---|---|---|---|---|---|\n")
    for r in summary_rows:
        if r["is_supervised"]:
            continue
        f1_str = f"{r['f1_mean']:.4f}±{r['f1_std']:.4f}" if r['f1_mean'] is not None else "n/a"
        seeds = f"{r['f1_seed0']}/{r['f1_seed1']}/{r['f1_seed2']}"
        post_ssl = f"{r['n_seeds_best_epoch_geq_semi_start']}/3" if r['n_seeds_best_epoch_geq_semi_start'] is not None else "n/a"
        md.append(
            f"| {r['dataset']} | {r['experiment']} | {r['lambda_u']} | {f1_str} | "
            f"{seeds} | {r['mean_best_epoch']} | {r['semi_start_epoch']} | {post_ssl} |\n"
        )
    md.append("\n")

    # Decision table
    md.append("## Head-to-head: λ=0.05 vs λ=0.1\n\n")
    md.append(
        "For each condition we evaluate three criteria:\n"
        "- **Mean F1** (higher is better)\n"
        "- **Std F1** across seeds (lower is better, more consistent)\n"
        "- **Post-SSL checkpoints** — count of seeds where `best_epoch >= semi_start_epoch`\n"
        "  (higher is better, means SSL actually shaped the final model)\n\n"
        "Majority vote across the 3 criteria picks the winner per condition.\n\n"
    )
    md.append(
        "| Dataset | Experiment | mean 0.05 | mean 0.1 | std 0.05 | std 0.1 | "
        "post-SSL 0.05 | post-SSL 0.1 | winner |\n"
    )
    md.append("|---|---|---|---|---|---|---|---|---|\n")
    for c in decision["comparisons"]:
        md.append(
            f"| {c['dataset']} | {c['experiment']} | {c['mean_005']} | {c['mean_010']} | "
            f"{c['std_005']} | {c['std_010']} | {c['post_ssl_005']}/3 | {c['post_ssl_010']}/3 | "
            f"**{c['overall_winner']}** |\n"
        )
    md.append("\n")

    # Final recommendation
    md.append("## Final recommendation\n\n")
    md.append(
        f"- Conditions where **λ=0.05** wins: {decision['score_005']}\n"
        f"- Conditions where **λ=0.1** wins: {decision['score_010']}\n"
        f"- Tied: {decision['tied']}\n\n"
    )

    if decision["score_005"] > decision["score_010"]:
        rec = "0.05"
        rationale = "More conditions favor λ=0.05 across the joint criteria (mean, stability, post-SSL checkpoint count)."
    elif decision["score_010"] > decision["score_005"]:
        rec = "0.1"
        rationale = "More conditions favor λ=0.1 across the joint criteria (mean, stability, post-SSL checkpoint count)."
    else:
        rec = "tie — see conditions individually"
        rationale = (
            "The two lambdas are roughly equivalent under the joint criteria. "
            "Prefer λ=0.05 for smaller SSL influence (safer default) unless a specific "
            "condition drives the paper's main claim."
        )

    md.append(f"**Recommended λ: {rec}**\n\n")
    md.append(f"{rationale}\n\n")
    md.append(
        "### Caveats\n"
        "- Do not pick based on a single condition's best mean alone.\n"
        "- If one lambda wins on mean but loses on stability, prefer stability for the paper's rerun.\n"
        "- If `post-SSL` count is low (e.g. 0/3 or 1/3), the supposed SSL effect may actually be"
        " coming from the pre-SSL phase — inspect those runs before trusting their F1.\n"
    )

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.writelines(md)
    print(f"Saved: {OUT_MD}")

    # Console summary
    print("\n" + "=" * 60)
    print("DECISION SUMMARY")
    print("=" * 60)
    print(f"λ=0.05 wins in {decision['score_005']} conditions")
    print(f"λ=0.10 wins in {decision['score_010']} conditions")
    print(f"Tied in {decision['tied']} conditions")
    if decision["score_005"] > decision["score_010"]:
        print("\nRecommendation: use λ=0.05")
    elif decision["score_010"] > decision["score_005"]:
        print("\nRecommendation: use λ=0.10")
    else:
        print("\nRecommendation: tie — see decision markdown for per-condition details")


if __name__ == "__main__":
    main()
