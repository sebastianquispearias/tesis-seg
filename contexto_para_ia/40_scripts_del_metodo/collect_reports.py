"""
collect_reports.py

Copies run_report.json files from experiment directories into a flat
output folder. Only copies — never moves, modifies, or deletes originals.

Usage:
    python collect_reports.py \
        --runs_unm "G:/My Drive/UNM_vertebras_seg_v3/runs" \
        --runs_inca_v1 "G:/My Drive/UNM_vertebras_seg_v3/runs_inca" \
        --runs_inca_v2 "G:/My Drive/UNM_vertebras_seg_v3/runs_inca_v2" \
        --output "resultados"
"""

import argparse
import glob
import os
import re
import shutil


def find_report(seed_dir):
    """Find the run report file inside a seed directory.
    Returns the path to the chosen report, or None if not found.
    Prints warnings for missing or ambiguous reports.
    """
    candidates = sorted(glob.glob(os.path.join(seed_dir, "*_run_report.json")))
    exact = os.path.join(seed_dir, "run_report.json")
    if os.path.isfile(exact) and exact not in candidates:
        candidates.append(exact)
        candidates.sort()

    if not candidates:
        print(f"  WARNING: No report found in {seed_dir}")
        return None

    if len(candidates) > 1:
        print(f"  WARNING: Multiple reports in {seed_dir}")
        # Prefer exact name
        if os.path.isfile(exact):
            return exact
        return candidates[0]

    return candidates[0]


def process_base(base_dir, group_name, output_root):
    """Process one base directory (e.g. runs/, runs_inca/, runs_inca_v2/).
    Returns (n_experiments, n_copied, missing_list).
    """
    out_dir = os.path.join(output_root, group_name)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(base_dir):
        print(f"ERROR: Directory does not exist: {base_dir}")
        return 0, 0, []

    print(f"\n{'='*60}")
    print(f"Processing: {group_name} -> {base_dir}")
    print(f"Output:     {out_dir}")

    # Discover experiments: immediate subdirectories of base_dir
    experiments = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ])

    n_copied = 0
    missing = []
    exp_set = set()

    for exp_name in experiments:
        exp_dir = os.path.join(base_dir, exp_name)

        # Find seed_* subdirectories
        seed_dirs = sorted([
            d for d in os.listdir(exp_dir)
            if os.path.isdir(os.path.join(exp_dir, d)) and re.match(r"^seed_\d+$", d)
        ])

        if not seed_dirs:
            continue

        exp_set.add(exp_name)

        # Check for missing seeds (expect 0, 1, 2)
        found_seeds = {int(d.replace("seed_", "")) for d in seed_dirs}
        expected_seeds = {0, 1, 2}
        for s in expected_seeds - found_seeds:
            missing.append(f"{exp_name}/seed_{s} (directory missing)")

        for seed_name in seed_dirs:
            seed_dir = os.path.join(exp_dir, seed_name)
            seed_num = seed_name.replace("seed_", "")

            report_path = find_report(seed_dir)
            if report_path is None:
                missing.append(f"{exp_name}/{seed_name} (no report)")
                continue

            dst_name = f"{exp_name}_seed_{seed_num}_run_report.json"
            dst_path = os.path.join(out_dir, dst_name)
            shutil.copy2(report_path, dst_path)
            n_copied += 1

    n_experiments = len(exp_set)
    print(f"\nExperiments detected: {n_experiments}")
    print(f"Reports copied:       {n_copied}")
    print(f"Missing:              {len(missing)}")
    if missing:
        for m in missing:
            print(f"  - {m}")

    return n_experiments, n_copied, missing


def main():
    parser = argparse.ArgumentParser(
        description="Collect run_report.json files from experiment directories."
    )
    parser.add_argument("--runs_unm", type=str, default=None,
                        help="Base directory for UNM experiments")
    parser.add_argument("--runs_inca_v1", type=str, default=None,
                        help="Base directory for INCA v1 experiments")
    parser.add_argument("--runs_inca_v2", type=str, default=None,
                        help="Base directory for INCA v2 experiments")
    parser.add_argument("--output", type=str, default="resultados",
                        help="Output directory (default: resultados)")
    args = parser.parse_args()

    groups = []
    if args.runs_unm:
        groups.append(("runs_unm", args.runs_unm))
    if args.runs_inca_v1:
        groups.append(("runs_inca_v1", args.runs_inca_v1))
    if args.runs_inca_v2:
        groups.append(("runs_inca_v2", args.runs_inca_v2))

    if not groups:
        print("No directories provided. Use --runs_unm, --runs_inca_v1, or --runs_inca_v2.")
        return

    total_exp = 0
    total_copied = 0
    total_missing = 0

    for group_name, base_dir in groups:
        n_exp, n_copied, missing = process_base(base_dir, group_name, args.output)
        total_exp += n_exp
        total_copied += n_copied
        total_missing += len(missing)

    print(f"\n{'='*60}")
    print(f"TOTAL")
    print(f"  Experiments: {total_exp}")
    print(f"  Copied:      {total_copied}")
    print(f"  Missing:     {total_missing}")


if __name__ == "__main__":
    main()
