"""
entropy_analysis.py — Per-pixel binary entropy from saved probability maps.
READ-ONLY. No retraining/inference. H = -p*log(p) - (1-p)*log(1-p) (natural log).
Splits entropy into: boundary band (+/-5 px around GT contour) vs rest of image.
Probs read from Drive test_probs/*.npy (320x320 float32); GT from local mirror,
processed pad_to_square -> 320 NEAREST (same as the pipeline).
Outputs: resultados/diagnostico_dice/entropy_summary.csv
"""
import os, io, sys, csv, glob
import numpy as np
from PIL import Image
from scipy import ndimage
import cv2
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)

ROOT = r"G:/My Drive/UNM_vertebras_seg_v3"
GTLOC = r"C:/Users/User/temp_inter_rater/diag"
OUT = os.path.join(ROOT, "resultados", "diagnostico_dice")
EPS = 1e-7
BAND_PX = 5

CONFIGS = [
    ("UNM", "Supervised", "runs_final_v1/supervised", "unm/gt", [0, 1, 2, 3, 4]),
    ("UNM", "MT all-lateral", "runs_final_v1/mean_teacher_all_lateral", "unm/gt", [0, 1, 2]),
    ("INCA", "Supervised", "runs_inca_final_v1/supervised_inca", "inca/gt", [0, 1, 2]),
    ("INCA", "MT r3 (best)", "runs_inca_final_v1/mean_teacher_inca_r3", "inca/gt", [0, 1, 2]),
]

def pad_sq(a, fill=0):
    im = Image.fromarray(a); w, h = im.size; s = max(w, h)
    c = Image.new(im.mode, (s, s), fill); c.paste(im, ((s-w)//2, (s-h)//2)); return np.array(c)
def gt320(p):
    m = np.array(Image.open(p).convert("L")); m = np.where(m > 127, 255, 0).astype(np.uint8)
    return cv2.resize(pad_sq(m, 0), (320, 320), interpolation=cv2.INTER_NEAREST) > 127
def bin_entropy(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))

rows = []
for ds, name, rel, gtrel, seeds in CONFIGS:
    gtd = os.path.join(GTLOC, gtrel)
    seed_all, seed_band, seed_rest = [], [], []
    used_seeds = []
    for s in seeds:
        pdir = os.path.join(ROOT, rel, f"seed_{s}", "test_probs")
        files = glob.glob(os.path.join(pdir, "*.npy"))
        if not files:
            continue
        Hs, Hb, Hr = [], [], []
        for f in files:
            stem = os.path.splitext(os.path.basename(f))[0]
            gp = os.path.join(gtd, stem + ".png")
            if not os.path.isfile(gp):
                continue
            p = np.load(f).astype(np.float32)
            H = bin_entropy(p)
            gt = gt320(gp)
            band = ndimage.binary_dilation(gt, iterations=BAND_PX) & ~ndimage.binary_erosion(gt, iterations=BAND_PX)
            rest = ~band
            Hs.append(float(H.mean()))
            Hb.append(float(H[band].mean()) if band.any() else np.nan)
            Hr.append(float(H[rest].mean()) if rest.any() else np.nan)
        if Hs:
            seed_all.append(np.nanmean(Hs)); seed_band.append(np.nanmean(Hb)); seed_rest.append(np.nanmean(Hr))
            used_seeds.append(s)
        print(f"{ds}/{name}/seed_{s}: {len(Hs)} imgs", flush=True)
    def ms(v): return (float(np.mean(v)), float(np.std(v)))
    ha, hasd = ms(seed_all); hb, hbsd = ms(seed_band); hr, hrsd = ms(seed_rest)
    rows.append({"dataset": ds, "config": name, "n_seeds": len(used_seeds),
                 "H_mean": f"{ha:.4f}", "H_mean_std": f"{hasd:.4f}",
                 "H_band": f"{hb:.4f}", "H_band_std": f"{hbsd:.4f}",
                 "H_rest": f"{hr:.4f}", "H_rest_std": f"{hrsd:.4f}",
                 "band_over_rest": f"{hb/hr:.1f}" if hr else "NA"})
    print(f"  => H_mean={ha:.4f}  H_band={hb:.4f}  H_rest={hr:.4f}  (band/rest={hb/hr:.1f}x)")

with open(os.path.join(OUT, "entropy_summary.csv"), "w", newline="", encoding="utf-8") as fo:
    w = csv.DictWriter(fo, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("saved entropy_summary.csv")
print("DONE")
