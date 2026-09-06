"""Copy all test_probs .npy needed for the entropy rework to local disk. READ-ONLY on Drive."""
import os, sys, io, shutil, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)
P = r"G:/My Drive/UNM_vertebras_seg_v3"
DEST = r"C:/Users/User/temp_inter_rater/ent"
CONFIGS = {
    "UNM__sup": ("runs_final_v1/supervised", [0, 1, 2, 3, 4]),
    "UNM__mt": ("runs_final_v1/mean_teacher_all_lateral", [0, 1, 2]),
    "INCApat10__sup": ("runs_inca_final_v1/supervised_inca_patient10", [0, 1, 2]),
    "INCApat10__mt": ("runs_inca_final_v1/mean_teacher_inca_r10_patient10", [0, 1, 2]),
}
for key, (base, seeds) in CONFIGS.items():
    for s in seeds:
        src = f"{P}/{base}/seed_{s}/test_probs"
        dst = os.path.join(DEST, key, f"seed_{s}")
        os.makedirs(dst, exist_ok=True)
        if not os.path.isdir(src):
            print("MISSING", src); continue
        n = 0
        for f in os.listdir(src):
            if not f.endswith(".npy"): continue
            d = os.path.join(dst, f)
            if os.path.isfile(d) and os.path.getsize(d) > 0:
                n += 1; continue
            for k in range(4):
                try:
                    shutil.copy2(os.path.join(src, f), d); n += 1; break
                except Exception:
                    time.sleep(0.4 * (k + 1))
        print(f"{key}/seed_{s}: {n}", flush=True)
# metadata for INCA patient clusters
shutil.copy2(f"{P}/data/video_frame_metadata.csv", os.path.join(DEST, "video_frame_metadata.csv"))
print("DONE copy")
