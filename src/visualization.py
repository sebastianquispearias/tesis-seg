import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.metrics import boundary_f1_score, assd_hd95


def show_config_summary(cfg: dict):
    print("===== CONFIG =====")
    for k, v in cfg.items():
        print(f"{k}: {v}")


def chw_to_hwc(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3 and x.shape[0] in [1, 3]:
        return np.transpose(x, (1, 2, 0))
    return x


def _unwrap_sample(sample):
    if isinstance(sample, list):
        if len(sample) == 0:
            raise ValueError("La muestra es una lista vacía.")
        return sample[0]
    return sample


def show_dataset_examples(dataset, n: int = 4):
    n = min(n, len(dataset))
    fig, axes = plt.subplots(n, 3, figsize=(10, 4 * n))

    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n):
        sample = _unwrap_sample(dataset[i])

        img = sample["image"].numpy()
        msk = sample["mask"].numpy()[0]

        img = chw_to_hwc(img)
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]

        axes[i, 0].imshow(img, cmap=None if img.ndim == 3 else "gray")
        axes[i, 0].set_title(f"Image - {sample.get('name', i)}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(msk, cmap="gray")
        axes[i, 1].set_title("Mask")
        axes[i, 1].axis("off")

        if img.ndim == 2:
            overlay = np.stack([img, img, img], axis=-1)
        else:
            overlay = img.copy()

        overlay = overlay.astype(np.float32)
        overlay[..., 0] = np.maximum(overlay[..., 0], msk * 0.8)

        axes[i, 2].imshow(np.clip(overlay, 0, 1) if overlay.ndim == 3 else overlay)
        axes[i, 2].set_title("Overlay")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()


def _to_uint8_rgb(img_float: np.ndarray) -> np.ndarray:
    img = np.clip(img_float, 0.0, 1.0)
    img_u8 = (img * 255).astype(np.uint8)
    if img_u8.ndim == 2:
        img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
    elif img_u8.shape[-1] == 1:
        img_u8 = cv2.cvtColor(img_u8[..., 0], cv2.COLOR_GRAY2RGB)
    return img_u8


@torch.no_grad()
def show_predictions(model, loader, device="cuda", thr=0.5, max_show=6, out_dir=None, tol_px=5):
    model.eval()
    shown = 0

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    for batch in loader:
        xb = batch["image"].to(device).float()
        yb = batch["mask"].to(device).float()
        names = batch.get("name", [f"sample_{i}" for i in range(xb.shape[0])])

        logits = model(xb)
        probs = torch.sigmoid(logits)
        preds = (probs >= thr).float()

        for i in range(xb.shape[0]):
            if shown >= max_show:
                return

            img = xb[i].detach().cpu().numpy()
            gt = yb[i, 0].detach().cpu().numpy()
            pr = preds[i, 0].detach().cpu().numpy()
            prob_map = probs[i, 0].detach().cpu().numpy()

            img = np.transpose(img, (1, 2, 0))
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img[..., 0]

            # Per-image metrics
            gt_bool = gt.astype(bool)
            pr_bool = pr.astype(bool)

            tp = int((pr_bool & gt_bool).sum())
            fp = int((pr_bool & (~gt_bool)).sum())
            fn = int(((~pr_bool) & gt_bool).sum())
            iou_i = tp / (tp + fp + fn + 1e-7)
            f1_i = (2 * tp) / (2 * tp + fp + fn + 1e-7)

            bf1_i = boundary_f1_score(pr_bool, gt_bool, r_tol_px=tol_px)
            assd_i, hd95_i = assd_hd95(pr_bool, gt_bool)

            assd_txt = f"{assd_i:.2f}px" if not np.isnan(assd_i) else "—"
            hd95_txt = f"{hd95_i:.2f}px" if not np.isnan(hd95_i) else "—"

            # Overlay: GT=red, Pred=green, 50-50 blend
            img_u8 = _to_uint8_rgb(img)
            ov_gt = img_u8.copy()
            ov_gt[gt_bool] = [255, 0, 0]
            ov_pr = img_u8.copy()
            ov_pr[pr_bool] = [0, 255, 0]
            overlay = cv2.addWeighted(ov_gt, 0.5, ov_pr, 0.5, 0)

            img_vis = np.clip(img, 0, 1) if img.ndim == 3 else img

            fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
            axs = axs.ravel()

            axs[0].imshow(img_vis, cmap=None if img_vis.ndim == 3 else "gray")
            axs[0].set_title("Imagem")
            axs[0].axis("off")

            axs[1].imshow(gt, cmap="gray")
            axs[1].set_title("GT")
            axs[1].axis("off")

            im2 = axs[2].imshow(prob_map, vmin=0, vmax=1, cmap="viridis")
            axs[2].set_title(f"Prob (thr={thr})")
            axs[2].axis("off")
            fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

            axs[3].imshow(pr, cmap="gray")
            axs[3].set_title("Pred binaria")
            axs[3].axis("off")

            axs[4].imshow(overlay)
            axs[4].set_title("Overlay GT(rojo)+Pred(verde)")
            axs[4].axis("off")

            axs[5].axis("off")
            axs[5].text(
                0.0, 0.9,
                (
                    f"F1: {f1_i:.3f}\n"
                    f"IoU: {iou_i:.3f}\n"
                    f"BF1@{tol_px}px: {bf1_i:.3f}\n"
                    f"ASSD: {assd_txt}   HD95: {hd95_txt}"
                ),
                fontsize=12, family="monospace", va="top",
                transform=axs[5].transAxes,
            )

            plt.suptitle(
                f"F1={f1_i:.3f} | IoU={iou_i:.3f} | BF1@{tol_px}px={bf1_i:.3f} | ASSD={assd_txt} | HD95={hd95_txt}",
                y=1.02,
            )

            if out_dir is not None:
                base_name = os.path.splitext(str(names[i]))[0]
                save_path = os.path.join(out_dir, f"{base_name}.png")
                plt.savefig(save_path, dpi=150, bbox_inches="tight")

            plt.show()
            plt.close(fig)
            shown += 1


def plot_history(history: list[dict]):
    if not history:
        print("No hay history para plotear.")
        return

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    val_f1 = [row.get("val_f1", float("nan")) for row in history]
    val_iou = [row.get("val_iou", float("nan")) for row in history]

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss history")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, val_f1, label="val_f1")
    plt.plot(epochs, val_iou, label="val_iou")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Validation metrics")
    plt.legend()
    plt.tight_layout()
    plt.show()
