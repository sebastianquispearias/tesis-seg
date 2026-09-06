"""
ent_rework.py — Entropy analysis, reworked. READ-ONLY (local .npy + GT).
Corrections:
 1) ALL seeds: per-seed Spearman + entropy tables -> mean +/- std BETWEEN seeds.
 2) BAND entropy (+/-5px around GT contour) as PRIMARY metric; global secondary.
 3) Cluster-respecting Spearman: aggregate per cluster (video in UNM, patient in
    INCA), then Spearman; report n_clusters.
 4) Confounder: Spearman(GT area, Dice) and Spearman(GT area, band entropy).
 5) Active learning sim (supervised): top-20% most uncertain (band entropy) mean
    Dice vs random-20% x1000.
 6) Bands +/-3, +/-5, +/-10.
 7) Calibration: pixel-level ECE (is the network overconfident?).
Outputs: resultados/diagnostico_dice/entropy_rework/  (CSVs + report PDF pieces)
"""
import os, io, sys, csv, glob
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.stats import spearmanr
import cv2
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)

LOCAL = r"C:/Users/User/temp_inter_rater/ent"
GTLOC = r"C:/Users/User/temp_inter_rater/diag"
OUT = r"G:/My Drive/UNM_vertebras_seg_v3/resultados/diagnostico_dice/entropy_rework"
os.makedirs(OUT, exist_ok=True)
EPS = 1e-7
BANDS = [3, 5, 10]

CONFIGS = [
    ("UNM", "Supervised", "UNM__sup", "unm/gt", [0, 1, 2, 3, 4]),
    ("UNM", "MT all-lateral", "UNM__mt", "unm/gt", [0, 1, 2]),
    ("INCA-p10", "Supervised", "INCApat10__sup", "inca/gt", [0, 1, 2]),
    ("INCA-p10", "MT r10", "INCApat10__mt", "inca/gt", [0, 1, 2]),
]

# INCA patient clusters from metadata
patient_of = {}
with open(os.path.join(LOCAL, "video_frame_metadata.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        patient_of[r["video_frame"]] = r.get("paciente_id", "")

def cluster_id(dataset, stem):
    if dataset == "UNM":
        return stem.split("_")[0]              # video
    return str(patient_of.get(stem, stem.split("_")[0]))  # patient

def pad_sq(a, fill=0):
    im = Image.fromarray(a); w, h = im.size; s = max(w, h)
    c = Image.new(im.mode, (s, s), fill); c.paste(im, ((s-w)//2, (s-h)//2)); return np.array(c)
def gt320(p):
    m = np.array(Image.open(p).convert("L")); m = np.where(m > 127, 255, 0).astype(np.uint8)
    return cv2.resize(pad_sq(m, 0), (320, 320), interpolation=cv2.INTER_NEAREST) > 127
def H_map(p):
    p = np.clip(p.astype(np.float32), EPS, 1 - EPS)
    return -(p*np.log(p) + (1-p)*np.log(1-p))
def dice(a, b):
    sa, sb = a.sum(), b.sum()
    return 1.0 if sa == 0 and sb == 0 else (0.0 if sa == 0 or sb == 0 else 2.0*(a & b).sum()/(sa+sb))
def band_mask(gt, px):
    return ndimage.binary_dilation(gt, iterations=px) & ~ndimage.binary_erosion(gt, iterations=px)

def ms(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), float("nan"))

# ---- gather per (config,seed,image) ----
# store: entropy per band, global, dice, gt area, cluster
data = {}   # key (ds,name) -> list of dict per (seed,image)
calib = {}  # key (ds,name) -> accumulators for ECE (20 bins)
for ds, name, key, gtrel, seeds in CONFIGS:
    gtd = os.path.join(GTLOC, gtrel)
    recs = []
    bins = np.linspace(0, 1, 21); conf_sum = np.zeros(20); acc_sum = np.zeros(20); cnt = np.zeros(20)
    for s in seeds:
        pdir = os.path.join(LOCAL, key, f"seed_{s}")
        for f in sorted(glob.glob(os.path.join(pdir, "*.npy"))):
            stem = os.path.splitext(os.path.basename(f))[0]
            gp = os.path.join(gtd, stem + ".png")
            if not os.path.isfile(gp): continue
            p = np.load(f); gt = gt320(gp); H = H_map(p)
            rec = {"seed": s, "stem": stem, "cluster": cluster_id(ds, stem),
                   "dice": dice(p >= 0.5, gt), "area": int(gt.sum()),
                   "H_global": float(H.mean())}
            for px in BANDS:
                bm = band_mask(gt, px)
                rec[f"H_b{px}"] = float(H[bm].mean()) if bm.any() else np.nan
            recs.append(rec)
            # calibration accumulation (all pixels)
            pf = p.astype(np.float32).ravel(); gf = gt.ravel().astype(np.float32)
            idx = np.clip((pf * 20).astype(int), 0, 19)
            np.add.at(conf_sum, idx, pf); np.add.at(acc_sum, idx, gf); np.add.at(cnt, idx, 1.0)
        print(f"{ds}/{name}/seed_{s}: done", flush=True)
    data[(ds, name)] = recs
    calib[(ds, name)] = (conf_sum, acc_sum, cnt)

# ---- (1)+(2) entropy tables: band (primary) + global (secondary), mean+/-std BETWEEN seeds ----
print("\n===== (1/2) ENTROPÍA: banda (principal) y global (secundaria), media±std ENTRE seeds =====")
ent_rows = []
for (ds, name), recs in data.items():
    seeds = sorted(set(r["seed"] for r in recs))
    row = {"dataset": ds, "config": name, "n_seeds": len(seeds)}
    line = f"{ds}/{name:16s}"
    for px in BANDS:
        per_seed = [np.nanmean([r[f"H_b{px}"] for r in recs if r["seed"] == s]) for s in seeds]
        m, sd = ms(per_seed); row[f"Hband{px}"] = f"{m:.4f}"; row[f"Hband{px}_std"] = f"{sd:.4f}"
        line += f"  b{px}={m:.4f}±{sd:.4f}"
    g = [np.mean([r["H_global"] for r in recs if r["seed"] == s]) for s in seeds]
    gm, gs = ms(g); row["Hglobal"] = f"{gm:.4f}"; row["Hglobal_std"] = f"{gs:.4f}"
    line += f"  | global={gm:.4f}±{gs:.4f}"
    ent_rows.append(row); print(line)

# ---- (3) Spearman image-level (per seed) AND cluster-level ----
print("\n===== (3) SPEARMAN entropía_banda±5 ↔ Dice: por imagen y por CLÚSTER =====")
sp_rows = []
for (ds, name), recs in data.items():
    seeds = sorted(set(r["seed"] for r in recs))
    img_rho, clu_rho, nclus = [], [], None
    for s in seeds:
        rs = [r for r in recs if r["seed"] == s]
        e = [r["H_b5"] for r in rs]; d = [r["dice"] for r in rs]
        img_rho.append(spearmanr(e, d)[0])
        # cluster means
        cl = {}
        for r in rs:
            cl.setdefault(r["cluster"], []).append((r["H_b5"], r["dice"]))
        ce = [np.nanmean([x[0] for x in v]) for v in cl.values()]
        cd = [np.mean([x[1] for x in v]) for v in cl.values()]
        clu_rho.append(spearmanr(ce, cd)[0]); nclus = len(cl)
    im, isd = ms(img_rho); cm, csd = ms(clu_rho)
    sp_rows.append({"dataset": ds, "config": name, "n_clusters": nclus,
                    "rho_img": f"{im:.3f}", "rho_img_std": f"{isd:.3f}",
                    "rho_cluster": f"{cm:.3f}", "rho_cluster_std": f"{csd:.3f}"})
    print(f"{ds}/{name:16s} n_clusters={nclus:3d}  rho_img={im:.3f}±{isd:.3f}  rho_cluster={cm:.3f}±{csd:.3f}")

# ---- (4) confounder: area vs dice, area vs band entropy (per seed) ----
print("\n===== (4) CONFUSOR: área GT ↔ Dice y área GT ↔ entropía_banda =====")
conf_rows = []
for (ds, name), recs in data.items():
    seeds = sorted(set(r["seed"] for r in recs))
    ad, ae = [], []
    for s in seeds:
        rs = [r for r in recs if r["seed"] == s]
        ad.append(spearmanr([r["area"] for r in rs], [r["dice"] for r in rs])[0])
        ae.append(spearmanr([r["area"] for r in rs], [r["H_b5"] for r in rs])[0])
    adm, ads = ms(ad); aem, aes = ms(ae)
    conf_rows.append({"dataset": ds, "config": name,
                      "rho_area_dice": f"{adm:.3f}±{ads:.3f}", "rho_area_entropy": f"{aem:.3f}±{aes:.3f}"})
    print(f"{ds}/{name:16s}  area↔Dice={adm:+.3f}±{ads:.3f}   area↔Hband={aem:+.3f}±{aes:.3f}")

# ---- (5) active learning sim (supervised only) ----
print("\n===== (5) ACTIVE LEARNING (supervisado): 20% más incierto vs 20% aleatorio (1000 reps) =====")
al_rows = []
rng = np.random.default_rng(42)
for (ds, name), recs in data.items():
    if name != "Supervised": continue
    seeds = sorted(set(r["seed"] for r in recs))
    top_means, rand_means, rand_los, rand_his = [], [], [], []
    for s in seeds:
        rs = [r for r in recs if r["seed"] == s]
        e = np.array([r["H_b5"] for r in rs]); d = np.array([r["dice"] for r in rs])
        n = len(rs); k = max(1, int(round(0.20 * n)))
        top_idx = np.argsort(-e)[:k]                 # most uncertain
        top_means.append(d[top_idx].mean())
        rand = [d[rng.choice(n, k, replace=False)].mean() for _ in range(1000)]
        rand_means.append(np.mean(rand)); rand_los.append(np.percentile(rand, 2.5)); rand_his.append(np.percentile(rand, 97.5))
    tm, ts = ms(top_means); rm, rs_ = ms(rand_means)
    al_rows.append({"dataset": ds, "config": name,
                    "dice_top20_uncertain": f"{tm:.3f}±{ts:.3f}",
                    "dice_random20": f"{rm:.3f}", "random20_95CI": f"[{np.mean(rand_los):.3f},{np.mean(rand_his):.3f}]"})
    print(f"{ds}/{name:16s}  Dice top20% incierto={tm:.3f}±{ts:.3f}   Dice azar20%={rm:.3f} (95%CI {np.mean(rand_los):.3f}-{np.mean(rand_his):.3f})")

# ---- (7) calibration (ECE) ----
print("\n===== (7) CALIBRACIÓN (ECE por píxel; ¿sobreconfiado?) =====")
cal_rows = []
for (ds, name), (conf_sum, acc_sum, cnt) in calib.items():
    tot = cnt.sum(); w = cnt / tot
    conf = np.where(cnt > 0, conf_sum / np.maximum(cnt, 1), 0)
    acc = np.where(cnt > 0, acc_sum / np.maximum(cnt, 1), 0)
    ece = float(np.sum(w * np.abs(acc - conf)))
    # overconfidence: mean(conf - acc) weighted (positive => overconfident)
    over = float(np.sum(w * (conf - acc)))
    cal_rows.append({"dataset": ds, "config": name, "ECE": f"{ece:.4f}", "over_minus_under": f"{over:+.4f}"})
    print(f"{ds}/{name:16s}  ECE={ece:.4f}   (conf-acc)={over:+.4f}  {'sobreconfiado' if over>0 else 'subconfiado'}")

# ---- save CSVs ----
def save(fn, rows):
    with open(os.path.join(OUT, fn), "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
save("entropy_bands.csv", ent_rows)
save("spearman.csv", sp_rows)
save("confounder.csv", conf_rows)
if al_rows: save("active_learning.csv", al_rows)
save("calibration.csv", cal_rows)
# per-image dump
allrecs = []
for (ds, name), recs in data.items():
    for r in recs:
        allrecs.append({"dataset": ds, "config": name, **r})
with open(os.path.join(OUT, "per_image_allseeds.csv"), "w", newline="", encoding="utf-8") as fo:
    w = csv.DictWriter(fo, fieldnames=list(allrecs[0].keys())); w.writeheader(); w.writerows(allrecs)
print("\nsaved CSVs to", OUT)
print("DONE")
