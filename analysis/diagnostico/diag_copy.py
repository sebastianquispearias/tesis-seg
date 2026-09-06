"""Copy GT masks, predictions and frames to local disk for the diagnostic (fast reads). READ-ONLY on Drive."""
import os, sys, io, shutil, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)
P = r"G:/My Drive/UNM_vertebras_seg_v3"
DEST = r"C:/Users/User/temp_inter_rater/diag"

SRC = {
    "unm/gt":       f"{P}/test/masks",
    "unm/frames":   f"{P}/test/images",
    "unm/pl_r10":   f"{P}/runs_final_v1/semi_r10/seed_0/test_preds",
    "unm/mt_r15":   f"{P}/runs_final_v1/mean_teacher_std_matched_r15/seed_0/test_preds",
    "inca/gt":      f"{P}/data/inca_dataset/test/masks",
    "inca/frames":  f"{P}/data/inca_dataset/test/images",
    "inca/mt_r15":  f"{P}/runs_inca_final_v1/mean_teacher_inca_std_matched_r15/seed_0/test_preds",
}
for rel, src in SRC.items():
    dst = os.path.join(DEST, rel); os.makedirs(dst, exist_ok=True)
    if not os.path.isdir(src):
        print("MISSING SRC", src); continue
    n = 0
    for f in os.listdir(src):
        if not f.lower().endswith(".png"):
            continue
        d = os.path.join(dst, f)
        if os.path.isfile(d) and os.path.getsize(d) > 0:
            n += 1; continue
        for k in range(4):
            try:
                shutil.copy2(os.path.join(src, f), d); n += 1; break
            except Exception:
                time.sleep(0.4 * (k + 1))
    print(f"{rel}: {n} pngs")
print("DONE copy")
