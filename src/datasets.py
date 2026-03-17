import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.preprocessing import preprocess_image_and_mask, pad_to_square


def seed_everything_for_worker(seed: int, worker_id: int) -> None:
    worker_seed = seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_torch_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _seed_worker_factory(seed: int):
    def _seed_worker(worker_id: int):
        seed_everything_for_worker(seed, worker_id)
    return _seed_worker


def list_png_files(folder: str | Path) -> List[str]:
    folder = str(folder)
    if not os.path.isdir(folder):
        return []
    return sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(".png")
    )


def build_split_paths(cfg: dict, split: str) -> Tuple[str, str]:
    img_root = cfg["img_root"]
    msk_root = cfg["msk_root"]

    images_dir = os.path.join(img_root, split, "images")
    masks_dir = os.path.join(msk_root, split, "masks")
    return images_dir, masks_dir


def build_unlabeled_path(cfg: dict) -> str:
    return os.path.join(cfg["img_root"], "unlabeling", "images")


def flatten_collate(batch):
    """
    Útil si en algún momento devuelves listas de muestras por imagen.
    Por ahora también funciona como collate normal.
    """
    if len(batch) == 0:
        return batch

    first = batch[0]

    if isinstance(first, list):
        flat = []
        for item in batch:
            flat.extend(item)
        return torch.utils.data.default_collate(flat)

    return torch.utils.data.default_collate(batch)


class SegmentationDataset(Dataset):
    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        cfg: dict,
        transform=None,
        split: str = "train",
    ):
        self.images_dir = str(images_dir)
        self.masks_dir = str(masks_dir)
        self.cfg = cfg
        self.transform = transform
        self.split = split

        img_files = set(list_png_files(self.images_dir))
        msk_files = set(list_png_files(self.masks_dir))
        self.files = sorted(list(img_files & msk_files))

        if len(self.files) == 0:
            raise RuntimeError(
                f"No se encontraron pares imagen-máscara en:\n"
                f"images_dir={self.images_dir}\n"
                f"masks_dir={self.masks_dir}"
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fname = self.files[idx]

        img_path = os.path.join(self.images_dir, fname)
        msk_path = os.path.join(self.masks_dir, fname)

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"No pude leer imagen: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"No pude leer máscara: {msk_path}")

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        prep = preprocess_image_and_mask(
            image_uint8=image,
            mask_uint8=mask,
            target_size=self.cfg["target_size"],
            use_pad=self.cfg["use_pad"],
            imagenet_norm=self.cfg["imagenet_norm"],
            image_preproc=self.cfg["image_preproc"],
            mask_smoothing=self.cfg["mask_smoothing"],
            debug=self.cfg.get("debug", False),
        )

        return {
            "image": torch.from_numpy(prep["image"]).float(),
            "mask": torch.from_numpy(prep["mask"]).float(),
            "name": fname,
        }


class UnlabeledFramesDataset(Dataset):
    """
    Devuelve dos vistas de la misma imagen unlabeled:
    - weak
    - strong
    Luego aplica el mismo preprocesamiento base que en supervisado.
    """
    def __init__(
        self,
        images_dir: str | Path,
        cfg: dict,
        transform_weak=None,
        transform_strong=None,
    ):
        self.images_dir = str(images_dir)
        self.cfg = cfg
        self.weak_tf = transform_weak
        self.strong_tf = transform_strong
        self.files = list_png_files(self.images_dir)

        if len(self.files) == 0:
            raise RuntimeError(f"No se encontraron PNG en {self.images_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def _preprocess_image_only(self, image_uint8: np.ndarray) -> np.ndarray:
        dummy_mask = np.zeros(image_uint8.shape[:2], dtype=np.uint8)

        prep = preprocess_image_and_mask(
            image_uint8=image_uint8,
            mask_uint8=dummy_mask,
            target_size=self.cfg["target_size"],
            use_pad=self.cfg["use_pad"],
            imagenet_norm=self.cfg["imagenet_norm"],
            image_preproc=self.cfg["image_preproc"],
            mask_smoothing="none",
            debug=self.cfg.get("debug", False),
        )
        return prep["image"]

    def __getitem__(self, idx: int):
        fname = self.files[idx]
        path = os.path.join(self.images_dir, fname)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"No pude leer {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        weak_img = img.copy()
        strong_img = img.copy()

        if self.weak_tf is not None:
            weak_img = self.weak_tf(image=weak_img)["image"]

        if self.strong_tf is not None:
            strong_img = self.strong_tf(image=strong_img)["image"]

        weak_img = self._preprocess_image_only(weak_img)
        strong_img = self._preprocess_image_only(strong_img)

        return {
            "weak_image": torch.from_numpy(weak_img).float(),
            "strong_image": torch.from_numpy(strong_img).float(),
            "name": fname,
        }


class TemporalUnlabeledPairsDataset(Dataset):
    """
    Construye pares (frame_t, frame_t+delta) del mismo video
    a partir de archivos tipo: v025_f239.png
    """
    def __init__(
        self,
        images_dir: str | Path,
        cfg: dict,
        transform=None,
        max_delta: Optional[int] = None,
    ):
        self.images_dir = str(images_dir)
        self.cfg = cfg
        self.transform = transform
        self.max_delta = int(max_delta if max_delta is not None else cfg["max_temp_delta"])

        all_files = list_png_files(self.images_dir)
        if len(all_files) == 0:
            raise RuntimeError(f"No se encontraron PNG en {self.images_dir}")

        self.pairs = self._build_pairs(all_files)

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No se pudieron construir pares temporales en {self.images_dir} "
                f"con max_delta={self.max_delta}"
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def _parse_name(self, fname: str) -> Tuple[int, int]:
        """
        Espera nombres tipo v025_f239.png
        """
        base = os.path.splitext(fname)[0]
        vpart, fpart = base.split("_")
        vid = int(vpart[1:])
        frame = int(fpart[1:])
        return vid, frame

    def _build_pairs(self, all_files: List[str]) -> List[Tuple[str, str]]:
        parsed = [self._parse_name(f) for f in all_files]
        pairs = []

        for i in range(len(all_files) - 1):
            vid0, frame0 = parsed[i]
            j = i + 1

            while j < len(all_files):
                vid1, frame1 = parsed[j]

                if vid1 != vid0:
                    break

                delta = frame1 - frame0
                if 1 <= delta <= self.max_delta:
                    pairs.append((all_files[i], all_files[j]))
                elif delta > self.max_delta:
                    break

                j += 1

        return pairs

    def _preprocess_image_only(self, image_uint8: np.ndarray) -> np.ndarray:
        dummy_mask = np.zeros(image_uint8.shape[:2], dtype=np.uint8)

        prep = preprocess_image_and_mask(
            image_uint8=image_uint8,
            mask_uint8=dummy_mask,
            target_size=self.cfg["target_size"],
            use_pad=self.cfg["use_pad"],
            imagenet_norm=self.cfg["imagenet_norm"],
            image_preproc=self.cfg["image_preproc"],
            mask_smoothing="none",
            debug=self.cfg.get("debug", False),
        )
        return prep["image"]

    def __getitem__(self, idx: int):
        fname_a, fname_b = self.pairs[idx]

        path_a = os.path.join(self.images_dir, fname_a)
        path_b = os.path.join(self.images_dir, fname_b)

        img_a = cv2.imread(path_a, cv2.IMREAD_COLOR)
        img_b = cv2.imread(path_b, cv2.IMREAD_COLOR)

        if img_a is None:
            raise RuntimeError(f"No pude leer {path_a}")
        if img_b is None:
            raise RuntimeError(f"No pude leer {path_b}")

        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img_a = self.transform(image=img_a)["image"]
            img_b = self.transform(image=img_b)["image"]

        img_a = self._preprocess_image_only(img_a)
        img_b = self._preprocess_image_only(img_b)

        return {
            "image_t": torch.from_numpy(img_a).float(),
            "image_tp": torch.from_numpy(img_b).float(),
            "name_t": fname_a,
            "name_tp": fname_b,
        }


def build_supervised_datasets(cfg: dict, train_tf=None):
    train_images, train_masks = build_split_paths(cfg, "train")
    val_images, val_masks = build_split_paths(cfg, "val")
    test_images, test_masks = build_split_paths(cfg, "test")

    train_ds = SegmentationDataset(
        images_dir=train_images,
        masks_dir=train_masks,
        cfg=cfg,
        transform=train_tf,
        split="train",
    )

    val_ds = SegmentationDataset(
        images_dir=val_images,
        masks_dir=val_masks,
        cfg=cfg,
        transform=None,
        split="val",
    )

    test_ds = SegmentationDataset(
        images_dir=test_images,
        masks_dir=test_masks,
        cfg=cfg,
        transform=None,
        split="test",
    )

    return train_ds, val_ds, test_ds


def build_unlabeled_datasets(cfg: dict, weak_tf=None, strong_tf=None):
    unlabeled_images_dir = build_unlabeled_path(cfg)

    if not os.path.isdir(unlabeled_images_dir):
        return None, None

    unlabeled_ds = UnlabeledFramesDataset(
        images_dir=unlabeled_images_dir,
        cfg=cfg,
        transform_weak=weak_tf,
        transform_strong=strong_tf,
    )

    temporal_unlab_ds = TemporalUnlabeledPairsDataset(
        images_dir=unlabeled_images_dir,
        cfg=cfg,
        transform=weak_tf,
        max_delta=cfg["max_temp_delta"],
    )

    return unlabeled_ds, temporal_unlab_ds


def build_dataloaders(
    cfg: dict,
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    unlabeled_ds: Optional[Dataset] = None,
    temporal_unlab_ds: Optional[Dataset] = None,
):
    seed = cfg["seed"]
    g = make_torch_generator(seed)
    seed_worker = _seed_worker_factory(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
    )

    unlabeled_loader = None
    temporal_unlab_loader = None

    if cfg.get("use_semi", False) and unlabeled_ds is not None:
        batch_size_unlab = max(1, cfg["batch_size"] // 4)

        unlabeled_loader = DataLoader(
            unlabeled_ds,
            batch_size=batch_size_unlab,
            shuffle=True,
            num_workers=cfg["num_workers"],
            pin_memory=True,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

    if cfg.get("use_semi", False) and temporal_unlab_ds is not None:
        temp_batch_size = max(1, cfg["batch_size"] // 4)

        temporal_unlab_loader = DataLoader(
            temporal_unlab_ds,
            batch_size=temp_batch_size,
            shuffle=True,
            num_workers=cfg["num_workers"],
            pin_memory=True,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "unlabeled_loader": unlabeled_loader,
        "temporal_unlab_loader": temporal_unlab_loader,
    }