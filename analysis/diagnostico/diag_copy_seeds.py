"""Copy all-seed test_preds to local for the multi-seed diagnostic. READ-ONLY on Drive."""
import os, sys, io, shutil, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)
P = r"G:/My Drive/UNM_vertebras_seg_v3"
DEST = r"C:/Users/User/temp_inter_rater/diag/preds"

MODELS = {
    "UNM__MT-r15": (f"{P}/runs_final_v1/mean_teacher_std_matched_r15", [0, 1, 2]),
    "UNM__PL-r10": (f"{P}/runs_final_v1/semi_r10", [0, 1, 2, 3, 4]),
    "INCA__MT-r15": (f"{P}/runs_inca_final_v1/mean_teacher_inca_std_matched_r15", [0, 1, 2]),
}
for key, (base, seeds) in MODELS.items():
    for s in seeds:
        src = f"{base}/seed_{s}/test_preds"
        dst = os.path.join(DEST, key, f"seed_{s}")
        os.makedirs(dst, exist_ok=True)
        if not os.path.isdir(src):
            print("MISSING", src); continue
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
        print(f"{key}/seed_{s}: {n} preds")
print("DONE copy seeds")
