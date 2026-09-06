"""
diag_pervertebra_B.py — DEFINITIVE per-vertebra diagnostic (METHOD B only).
READ-ONLY. No training/inference. Uses existing binary predictions.

Method B (GT-guided partition): GT -> pad_to_square -> 320 (NEAREST); connected
components of GT ordered top-down = C2,C3,C4. Each predicted pixel is assigned to
the NEAREST GT vertebra within a dilated GT territory, so a merged prediction
spanning two vertebrae is split. Per-vertebra Dice = Dice(GT_i, pred assigned to i);
0 if a GT vertebra has no predicted pixels (missing-as-zero, the honest variant).

LIMITATION (disclose in thesis): vertebra identity is inferred from the top-down
order of connected components in the BINARY reference mask; in a minority of frames
where reference vertebrae touch, this labeling can be imprecise. Per-vertebra values
are indicative; the global Dice does not depend on this.

Outputs: resultados/diagnostico_dice/pervertebra_B_per_image.csv
         resultados/diagnostico_dice/pervertebra_B_summary.csv
"""
import os, csv, io, sys
import numpy as np
from PIL import Image
from scipy import ndimage
import cv2
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)

BASE = r"C:/Users/User/temp_inter_rater/diag"
PREDS = os.path.join(BASE, "preds")
OUT = r"G:/My Drive/UNM_vertebras_seg_v3/resultados/diagnostico_dice"
CONN = np.ones((3, 3), int)
DILATE_PX = 8

MODELS = [
    ("UNM__PL-r3", "UNM", "PL-r3", "unm/gt", [0, 1, 2]),
    ("UNM__MT-alllateral", "UNM", "MT-all-lateral", "unm/gt", [0, 1, 2]),
    ("INCA__MT-r15", "INCA", "MT-r15", "inca/gt", [0, 1, 2]),
]

def pad_sq(a, fill=0):
    im = Image.fromarray(a); w, h = im.size; s = max(w, h)
    c = Image.new(im.mode, (s, s), fill); c.paste(im, ((s-w)//2, (s-h)//2)); return np.array(c)
def gt320(p):
    m = np.array(Image.open(p).convert("L")); m = np.where(m > 127, 255, 0).astype(np.uint8)
    return cv2.resize(pad_sq(m, 0), (320, 320), interpolation=cv2.INTER_NEAREST) > 127
def predb(p): return np.array(Image.open(p).convert("L")) > 127
def dice(a, b):
    sa, sb = a.sum(), b.sum()
    return 1.0 if sa == 0 and sb == 0 else (0.0 if sa == 0 or sb == 0 else 2.0*(a & b).sum()/(sa+sb))

def analyze_B(gt, pr):
    lbl, n = ndimage.label(gt, structure=CONN)
    order = sorted(range(1, n + 1), key=lambda k: np.where(lbl == k)[0].mean())
    gl = np.zeros_like(lbl)
    for new, old in enumerate(order, 1):
        gl[lbl == old] = new
    idx = ndimage.distance_transform_edt(gl == 0, return_distances=False, return_indices=True)
    nearest = gl[tuple(idx)]
    territory = ndimage.binary_dilation(gt, iterations=DILATE_PX)
    res = {"dice_global": dice(gt, pr), "fp": int((pr & ~gt).sum()), "fn": int((~pr & gt).sum()),
           "n_gt": n}
    for i, nm in enumerate(["C2", "C3", "C4"], 1):
        if i <= n:
            d = dice(gl == i, pr & (nearest == i) & territory)
            res[f"dice_{nm}"] = d; res[f"missing_{nm}"] = int(d == 0.0)
        else:
            res[f"dice_{nm}"] = None; res[f"missing_{nm}"] = 0
    res["n_extra_pred"] = ndimage.label(pr & ~territory, structure=CONN)[1]
    return res

rows = []
for key, ds, model, gtrel, seeds in MODELS:
    gtd = os.path.join(BASE, gtrel)
    for s in seeds:
        prd = os.path.join(PREDS, key, f"seed_{s}")
        for f in sorted(x for x in os.listdir(prd) if x.lower().endswith(".png")):
            gp = os.path.join(gtd, f)
            if not os.path.isfile(gp): continue
            r = analyze_B(gt320(gp), predb(os.path.join(prd, f)))
            r.update(dataset=ds, model=model, seed=s, filename=f)
            rows.append(r)
        print(f"{key}/seed_{s}: done", flush=True)

cols = ["dataset", "model", "seed", "filename", "dice_global", "dice_C2", "dice_C3", "dice_C4",
        "missing_C2", "missing_C3", "missing_C4", "n_extra_pred", "fp", "fn", "n_gt"]
with open(os.path.join(OUT, "pervertebra_B_per_image.csv"), "w", newline="", encoding="utf-8") as fo:
    w = csv.DictWriter(fo, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

def seed_mean(rs, fn):
    per = {}
    for r in rs:
        per.setdefault(r["seed"], []).append(fn(r))
    return [np.mean([x for x in v if x is not None]) for v in per.values()]

out = []
print("\n===== METHOD B — per-vertebra (missing-as-zero), mean +/- std BETWEEN seeds =====")
for key, ds, model, gtrel, seeds in MODELS:
    rs = [r for r in rows if r["dataset"] == ds and r["model"] == model]
    ns = len(set(r["seed"] for r in rs))
    g = seed_mean(rs, lambda r: r["dice_global"])
    row = {"dataset": ds, "model": model, "n_seeds": ns, "n_img_per_seed": len(rs)//ns,
           "dice_global": f"{np.mean(g):.4f}", "dice_global_std": f"{np.std(g):.4f}"}
    print(f"\n--- {ds}/{model} ({ns} seeds) ---")
    print(f"  Dice global: {np.mean(g):.4f} +/- {np.std(g):.4f}")
    for c in ["C2", "C3", "C4"]:
        v = seed_mean(rs, lambda r: (r[f"dice_{c}"] if r[f"dice_{c}"] is not None else 0.0))
        row[c] = f"{np.mean(v):.4f}"; row[f"{c}_std"] = f"{np.std(v):.4f}"
        a = np.array([r["dice_global"] for r in rs if r[f"dice_{c}"] is not None])
        b = np.array([r[f"dice_{c}"] for r in rs if r[f"dice_{c}"] is not None])
        cc = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")
        row[f"corr_{c}"] = f"{cc:.3f}"
        print(f"  {c}: {np.mean(v):.4f} +/- {np.std(v):.4f}   corr(global,{c})={cc:.3f}")
    pex = seed_mean(rs, lambda r: float(r["n_extra_pred"] > 0))
    pmiss = seed_mean(rs, lambda r: float(r["missing_C2"] or r["missing_C3"] or r["missing_C4"]))
    fp = seed_mean(rs, lambda r: float(r["fp"])); fn = seed_mean(rs, lambda r: float(r["fn"]))
    row.update({"pct_extra": f"{100*np.mean(pex):.1f}", "pct_missing": f"{100*np.mean(pmiss):.1f}",
                "fp_px": f"{np.mean(fp):.0f}", "fn_px": f"{np.mean(fn):.0f}"})
    print(f"  %% extra: {100*np.mean(pex):.1f}%   %% missing: {100*np.mean(pmiss):.1f}%   FP:{np.mean(fp):.0f}  FN:{np.mean(fn):.0f}")
    out.append(row)

with open(os.path.join(OUT, "pervertebra_B_summary.csv"), "w", newline="", encoding="utf-8") as fo:
    w = csv.DictWriter(fo, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
print("\nsaved pervertebra_B_per_image.csv + pervertebra_B_summary.csv")
print("DONE")
