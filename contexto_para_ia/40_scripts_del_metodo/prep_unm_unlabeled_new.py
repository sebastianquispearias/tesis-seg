import os
import cv2
from pathlib import Path
from collections import defaultdict

ROOT = r"G:\My Drive\UNM TalkBank Dysphagia"
OUT_ROOT = r"G:\My Drive\UNM_vertebras_seg_v3"

VIDEOS_DIR     = os.path.join(ROOT, "videos")
VIDEOS_SEL_DIR = os.path.join(ROOT, "videos-selecionados")

NEIGHBOR_RADIUS     = 20
MAX_UNLAB_PER_VIDEO = 0
RUN_TAG = f"r{NEIGHBOR_RADIUS}_max{MAX_UNLAB_PER_VIDEO}"

UNLAB_IMG_DIR = os.path.join(OUT_ROOT, f"unlabeling_{RUN_TAG}", "images")


def dual_print(*args, **kwargs):
    print(*args, **kwargs)


def ensure_dir(path: str):
    if path is None or path == "":
        raise ValueError(f"Ruta inválida: {path}")
    os.makedirs(path, exist_ok=True)
# ===== Cargar labeled desde las CARPETAS (no CSV) =====

def load_labeled_from_folders(out_root: str):
    """
    Lee los nombres de archivo en:
      - train/masks
      - val/masks
      - test/masks
    Asume nombres tipo vXXX_fYYY.png.
    Reconstruye:
      - labeled_keys = set("vXXX_fYYY")
      - labeled_by_video[vid] = {frame_ids}
      - train_videos, val_videos, test_videos (sets de vid)
    """
    split_map = {
        "train": os.path.join(out_root, "train", "masks"),
        "val":   os.path.join(out_root, "val", "masks"),
        "test":  os.path.join(out_root, "test", "masks"),
    }

    labeled_by_video = defaultdict(set)
    labeled_keys = set()
    train_videos = set()
    val_videos = set()
    test_videos = set()

    for split, mdir in split_map.items():
        if not os.path.isdir(mdir):
            dual_print(f"[WARN] No existe carpeta de máscaras para '{split}': {mdir}")
            continue

        for fname in os.listdir(mdir):
            if not fname.lower().endswith(".png"):
                continue
            stem = Path(fname).stem  # vXXX_fYYY
            if "_f" not in stem:
                continue

            try:
                vpart, fpart = stem.split("_f")
                vid = vpart.replace("v", "")
                fid = int(fpart)
            except Exception:
                dual_print(f"[WARN] Nombre raro (se ignora): {fname}")
                continue

            labeled_keys.add(stem)
            labeled_by_video[vid].add(fid)

            if split == "train":
                train_videos.add(vid)
            elif split == "val":
                val_videos.add(vid)
            elif split == "test":
                test_videos.add(vid)

    dual_print("=== SPLITS detectados desde carpetas ===")
    dual_print(f"Videos TRAIN: {sorted(train_videos)}")
    dual_print(f"Videos VAL:   {sorted(val_videos)}")
    dual_print(f"Videos TEST:  {sorted(test_videos)}")
    dual_print(f"Total frames labeled (train+val+test): {len(labeled_keys)}")

    return labeled_keys, labeled_by_video, train_videos, val_videos, test_videos


# ===== Utilidades de video =====

def find_video_file(vid: str):
    """
    Busca el archivo de video correspondiente a un ID,
    probando nombres típicos en VIDEOS_SEL_DIR y VIDEOS_DIR.
    """
    vid_int = int(vid)
    candidates = [
        os.path.join(VIDEOS_SEL_DIR, f"v{vid_int:03d}.avi"),
        os.path.join(VIDEOS_SEL_DIR, f"{vid_int:03d}.avi"),
        os.path.join(VIDEOS_DIR,     f"v{vid_int:03d}.avi"),
        os.path.join(VIDEOS_DIR,     f"{vid_int}.avi"),
        os.path.join(VIDEOS_SEL_DIR, f"v{vid_int:03d}.mp4"),
        os.path.join(VIDEOS_DIR,     f"v{vid_int:03d}.mp4"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def count_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def read_frame(video_path: str, idx: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame  # BGR


def save_gray(frame, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(out_path, frame)


# ===== Construcción de unlabeled =====

def build_unlabeled():

    if not os.path.exists(OUT_ROOT):
        raise FileNotFoundError(f"{OUT_ROOT} no existe. Corre antes prep_unm_vertebras.py con este mismo OUT_ROOT.")

    ensure_dir(UNLAB_IMG_DIR)

    existing = [f for f in os.listdir(UNLAB_IMG_DIR) if f.lower().endswith(".png")]
    if existing:
        dual_print(f"[INFO] Carpeta destino ya contiene {len(existing)} PNG: {UNLAB_IMG_DIR}")

    labeled_keys, labeled_by_video, train_videos, val_videos, test_videos = \
        load_labeled_from_folders(OUT_ROOT)

    dual_print("\n=== Construyendo UNLABELED solo desde videos TRAIN ===")
    dual_print(f"Videos TRAIN candidatos: {sorted(train_videos)}")
    dual_print(f"Videos TEST (excluidos): {sorted(test_videos)}")

    total_unlab = 0
    total_skipped_overlap = 0
    missing_video = 0
    processed_videos = 0

    for vid in sorted(train_videos):
        video_path = find_video_file(vid)
        if video_path is None:
            dual_print(f"[WARN] No se encontró archivo de video para v{vid}")
            missing_video += 1
            continue

        n_frames = count_frames(video_path)
        if n_frames <= 0:
            dual_print(f"[WARN] Video v{vid} sin frames legibles")
            continue

        labeled_frames = sorted(labeled_by_video.get(vid, set()))
        if not labeled_frames:
            continue

        candidates = set()

        # vecinos temporales de cada frame etiquetado
        for f in labeled_frames:
            for k in range(f - NEIGHBOR_RADIUS, f + NEIGHBOR_RADIUS + 1):
                if k < 0 or k >= n_frames:
                    continue
                abv = f"v{int(vid):03d}_f{k}"
                if abv in labeled_keys:
                    total_skipped_overlap += 1
                    continue
                candidates.add(k)

        if not candidates:
            continue

        candidates = sorted(candidates)

        saved_for_video = 0
        for k in candidates:
            frame = read_frame(video_path, k)
            if frame is None:
                continue

            out_name = f"v{int(vid):03d}_f{k}.png"
            out_path = os.path.join(UNLAB_IMG_DIR, out_name)
            if os.path.exists(out_path):
                continue

            save_gray(frame, out_path)
            saved_for_video += 1
            total_unlab += 1

        dual_print(f"v{int(vid):03d}: GT={len(labeled_frames)}, "
                   f"cand_unlab={len(candidates)}, guardados={saved_for_video}, "
                   f"frames_video={n_frames}")

        processed_videos += 1

    dual_print("\n=== RESUMEN FINAL UNLABELED ===")
    dual_print(f"Videos TRAIN procesados: {processed_videos}")
    dual_print(f"Videos TRAIN sin archivo de video: {missing_video}")
    dual_print(f"Total frames UNLABELED guardados: {total_unlab}")
    dual_print(f"Frames descartados por solapar con labeled: {total_skipped_overlap}")
    dual_print(f"Salida: {UNLAB_IMG_DIR}")


if __name__ == "__main__":
    build_unlabeled()
