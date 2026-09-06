"""
entropy_extra.py — (1) Spearman entropy vs Dice + scatter; (2) uncertainty maps.
READ-ONLY. Uses saved test_probs/*.npy (seed_0). Binary entropy H=-p*log p-(1-p)*log(1-p).
Dice from probs binarized at 0.5 vs GT (pad_to_square->320 NEAREST).
Outputs in resultados/diagnostico_dice/entropy_figs/
"""
import os, io, sys, glob, csv
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.stats import spearmanr
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)

ROOT = r"G:/My Drive/UNM_vertebras_seg_v3"
GTLOC = r"C:/Users/User/temp_inter_rater/diag"
OUT = os.path.join(ROOT, "resultados", "diagnostico_dice", "entropy_figs")
os.makedirs(OUT, exist_ok=True)
EPS = 1e-7; LN2 = float(np.log(2))

# (label, dataset, rel_dir, gt_local, frames_local)
PAIRS = {
    "UNM": {"sup": "runs_final_v1/supervised", "mt": "runs_final_v1/mean_teacher_all_lateral",
            "gt": "unm/gt", "fr": "unm/frames"},
    "INCA-p10": {"sup": "runs_inca_final_v1/supervised_inca_patient10",
                 "mt": "runs_inca_final_v1/mean_teacher_inca_r10_patient10",
                 "gt": "inca/gt", "fr": "inca/frames"},
}

def pad_sq(a, fill=0):
    im = Image.fromarray(a); w, h = im.size; s = max(w, h)
    c = Image.new(im.mode, (s, s), fill); c.paste(im, ((s-w)//2, (s-h)//2)); return np.array(c)
def gt320(p):
    m = np.array(Image.open(p).convert("L")); m = np.where(m > 127, 255, 0).astype(np.uint8)
    return cv2.resize(pad_sq(m, 0), (320, 320), interpolation=cv2.INTER_NEAREST) > 127
def fr320(p):
    a = np.array(Image.open(p).convert("L"))
    return cv2.resize(pad_sq(a, 0), (320, 320), interpolation=cv2.INTER_LINEAR)
def H_map(p):
    p = np.clip(p.astype(np.float32), EPS, 1 - EPS)
    return -(p*np.log(p) + (1-p)*np.log(1-p))
def dice(a, b):
    sa, sb = a.sum(), b.sum()
    return 1.0 if sa == 0 and sb == 0 else (0.0 if sa == 0 or sb == 0 else 2.0*(a & b).sum()/(sa+sb))

# ---------- Task 1: per-image entropy vs Dice (seed_0) + Spearman + scatter ----------
percfg = {}
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
positions = {("UNM", "sup"): (0, 0), ("UNM", "mt"): (0, 1),
             ("INCA-p10", "sup"): (1, 0), ("INCA-p10", "mt"): (1, 1)}
rows_csv = []
for dsname, cfg in PAIRS.items():
    gtd = os.path.join(GTLOC, cfg["gt"])
    for role in ("sup", "mt"):
        pdir = os.path.join(ROOT, cfg[role], "seed_0", "test_probs")
        ent, dic, stems = [], [], []
        for f in sorted(glob.glob(os.path.join(pdir, "*.npy"))):
            stem = os.path.splitext(os.path.basename(f))[0]
            gp = os.path.join(gtd, stem + ".png")
            if not os.path.isfile(gp): continue
            p = np.load(f); gt = gt320(gp)
            ent.append(float(H_map(p).mean())); dic.append(dice(p >= 0.5, gt)); stems.append(stem)
        rho, pval = spearmanr(ent, dic)
        percfg[(dsname, role)] = (np.array(ent), np.array(dic), rho, pval)
        for st, e, d in zip(stems, ent, dic):
            rows_csv.append({"dataset": dsname, "role": role, "frame": st,
                             "mean_entropy": round(e, 5), "dice": round(d, 4)})
        r, c = positions[(dsname, role)]
        axes[r, c].scatter(ent, dic, s=14, alpha=0.6)
        axes[r, c].set_title(f"{dsname} / {'Supervisado' if role=='sup' else 'MT'}   "
                             f"Spearman rho={rho:.2f} (p={pval:.1e})", fontsize=10)
        axes[r, c].set_xlabel("entropia media por imagen"); axes[r, c].set_ylabel("Dice")
        print(f"{dsname}/{role}: n={len(ent)}  Spearman rho={rho:.3f} p={pval:.2e}", flush=True)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"scatter_entropy_dice.{ext}"), dpi=300, bbox_inches="tight")
plt.close(fig)
with open(os.path.join(OUT, "entropy_dice_per_image.csv"), "w", newline="", encoding="utf-8") as fo:
    w = csv.DictWriter(fo, fieldnames=["dataset", "role", "frame", "mean_entropy", "dice"]); w.writeheader(); w.writerows(rows_csv)
print("saved scatter + entropy_dice_per_image.csv")

# ---------- Task 2: uncertainty maps for 3 representative frames per dataset ----------
def pick_representative(dsname):
    ent, dic, *_ = percfg[(dsname, "mt")]
    stems = [r["frame"] for r in rows_csv if r["dataset"] == dsname and r["role"] == "mt"]
    order = np.argsort(dic)                 # sort by Dice
    med = len(order)//2
    idxs = [order[med-1], order[med], order[med+1]]   # 3 around the median (representative)
    return [stems[i] for i in idxs]

for dsname, cfg in PAIRS.items():
    frames = pick_representative(dsname)
    gtd = os.path.join(GTLOC, cfg["gt"]); frd = os.path.join(GTLOC, cfg["fr"])
    n = len(frames)
    fig, ax = plt.subplots(n, 3, figsize=(11, 3.4*n))
    for i, stem in enumerate(frames):
        fr = fr320(os.path.join(frd, stem + ".png"))
        gt = gt320(os.path.join(gtd, stem + ".png"))
        ps = np.load(os.path.join(ROOT, cfg["sup"], "seed_0", "test_probs", stem + ".npy"))
        pm = np.load(os.path.join(ROOT, cfg["mt"], "seed_0", "test_probs", stem + ".npy"))
        Hs, Hm = H_map(ps), H_map(pm)
        ds_ = dice(ps >= 0.5, gt); dm_ = dice(pm >= 0.5, gt)
        ax[i, 0].imshow(fr, cmap="gray"); ax[i, 0].set_title(f"{stem}", fontsize=9)
        im1 = ax[i, 1].imshow(Hs, cmap="magma", vmin=0, vmax=LN2)
        ax[i, 1].set_title(f"Supervisado  Dice={ds_:.2f}  H={Hs.mean():.4f}", fontsize=9)
        im2 = ax[i, 2].imshow(Hm, cmap="magma", vmin=0, vmax=LN2)
        ax[i, 2].set_title(f"MT  Dice={dm_:.2f}  H={Hm.mean():.4f}", fontsize=9)
        for a in ax[i]: a.set_axis_off()
        print(f"  {dsname} {stem}: sup Dice={ds_:.3f} H={Hs.mean():.4f} | MT Dice={dm_:.3f} H={Hm.mean():.4f}", flush=True)
    cbar = fig.colorbar(im2, ax=ax.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("entropia binaria (0..ln2)")
    fig.suptitle(f"Mapas de incertidumbre — {dsname} (misma escala 0..{LN2:.2f})", fontsize=12)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"uncertainty_maps_{dsname}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved uncertainty_maps_{dsname}")
print("DONE")
