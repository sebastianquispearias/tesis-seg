"""Summarise every run_report.json under the given roots.

For each experiment it prints one line per seed with the test-set metrics that
the thesis reports, and then the across-seed mean and sample standard
deviation. All numbers come from the JSON reports, never from the manuscript.
"""
import json
import math
import os
import sys

ROOTS = sys.argv[1:]


def collect(root):
    runs = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            if not fn.endswith("run_report.json"):
                continue
            with open(os.path.join(dirpath, fn), "r", encoding="utf-8") as fh:
                rep = json.load(fh)
            ident = rep.get("run_identity", {})
            name = ident.get("experiment_name", fn)
            seed = ident.get("seed", -1)
            tm = rep.get("test_metrics", {}) or {}
            bm = rep.get("test_boundary_metrics", {}) or {}
            ts = rep.get("training_summary", {}) or {}
            cfg = rep.get("config", {}) or {}
            runs.setdefault(name, []).append({
                "seed": seed,
                "f1": tm.get("sample_mean_f1"),
                "iou": tm.get("sample_mean_iou"),
                "assd": bm.get("assd_mean_px"),
                "hd95": bm.get("hd95_mean_px"),
                "bf1": bm.get("bf1_mean"),
                "n": tm.get("n_images"),
                "best_epoch": ts.get("best_epoch"),
                "epochs_run": ts.get("epochs_run") or ts.get("last_epoch"),
                "arch": cfg.get("arch"),
                "backbone": cfg.get("backbone"),
                "unlab": cfg.get("unlabeled_subdir"),
                "dataset": cfg.get("dataset"),
            })
    return runs


def stat(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0, len(vals)
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, math.sqrt(var), len(vals)


def fmt(x, nd=4):
    return "  n/a " if x is None else "{:.{}f}".format(x, nd)


for root in ROOTS:
    runs = collect(root)
    print("=" * 78)
    print(root)
    for name in sorted(runs):
        rows = sorted(runs[name], key=lambda r: r["seed"])
        head = rows[0]
        print("-" * 78)
        print("{}   [{} / {} / {} / pool={}]".format(
            name, head["dataset"], head["arch"], head["backbone"], head["unlab"]))
        print("  seed   F1      IoU     ASSD px  HD95 px   BF1     n_img  best_ep")
        for r in rows:
            print("   {}    {}  {}  {}   {}  {}   {}    {}".format(
                r["seed"], fmt(r["f1"]), fmt(r["iou"]), fmt(r["assd"], 3),
                fmt(r["hd95"], 3), fmt(r["bf1"]), r["n"], r["best_epoch"]))
        for key, nd in (("f1", 4), ("assd", 3), ("hd95", 3)):
            m, s, k = stat([r[key] for r in rows])
            if m is not None:
                print("  {:>5} media {} +/- {}   n_semillas={}".format(
                    key.upper(), fmt(m, nd), fmt(s, nd), k))
