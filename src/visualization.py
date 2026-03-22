import csv
import os
import re

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
def show_predictions(model, loader, device="cuda", thr=0.5, max_show=6, out_dir=None, tol_px=5, c2c4_csv=None):
    model.eval()
    shown = 0

    # Load C2-C4 lookup dict keyed by stem (optional)
    c2c4_lookup = {}
    if c2c4_csv is not None and os.path.isfile(c2c4_csv):
        try:
            with open(c2c4_csv, encoding="utf-8") as _f:
                for _row in csv.DictReader(_f):
                    c2c4_lookup[_row["stem"]] = _row
        except Exception:
            c2c4_lookup = {}

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

            # Case identity
            stem = os.path.splitext(str(names[i]))[0]
            m = re.match(r"v(\d+)_f(\d+)", stem)
            case_label = f"v{m.group(1)} · f{m.group(2)}" if m else stem

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
            axs[0].set_title("Image")
            axs[0].axis("off")

            axs[1].imshow(gt, cmap="gray")
            axs[1].set_title("GT mask")
            axs[1].axis("off")

            im2 = axs[2].imshow(prob_map, vmin=0, vmax=1, cmap="viridis")
            axs[2].set_title(f"Prob map (thr={thr})")
            axs[2].axis("off")
            fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

            axs[3].imshow(pr, cmap="gray")
            axs[3].set_title("Pred binary")
            axs[3].axis("off")

            axs[4].imshow(overlay)
            axs[4].set_title("Overlay  GT(red) · Pred(green)")
            axs[4].axis("off")

            # Panel 5: C2-C4 overlay (or placeholder if data unavailable)
            axs[5].axis("off")
            c2c4_row = c2c4_lookup.get(stem)
            if c2c4_row is not None:
                try:
                    p_c2 = (int(float(c2c4_row["p_c2_x"])), int(float(c2c4_row["p_c2_y"])))
                    p_c4 = (int(float(c2c4_row["p_c4_x"])), int(float(c2c4_row["p_c4_y"])))
                    c24_overlay = img_u8.copy()
                    # Tint predicted mask region blue
                    pr_bool_u8 = pr.astype(bool)
                    c24_overlay[pr_bool_u8, 2] = np.clip(
                        c24_overlay[pr_bool_u8, 2].astype(int) // 2 + 100, 0, 255
                    )
                    # Auto C2-C4: yellow line, cyan C2, magenta C4
                    cv2.line(c24_overlay, p_c2, p_c4, (255, 255, 0), 2)
                    cv2.circle(c24_overlay, p_c2, 5, (0, 255, 255), -1)
                    cv2.circle(c24_overlay, p_c4, 5, (255, 0, 255), -1)
                    title5 = "C2●(cyan) C4●(mag)  auto(yellow)"
                    # Manual reference line (red) if manual points are available
                    if c2c4_row.get("p1_man_x"):
                        p1m = (int(float(c2c4_row["p1_man_x"])), int(float(c2c4_row["p1_man_y"])))
                        p2m = (int(float(c2c4_row["p2_man_x"])), int(float(c2c4_row["p2_man_y"])))
                        cv2.line(c24_overlay, p1m, p2m, (255, 0, 0), 2)
                        title5 += "  manual(red)"
                        if c2c4_row.get("err_landmark_mean_px"):
                            title5 += f"  lm={float(c2c4_row['err_landmark_mean_px']):.1f}px"
                    axs[5].imshow(c24_overlay)
                    axs[5].set_title(title5, fontsize=9)
                except Exception:
                    axs[5].text(0.5, 0.5, "C2-C4\n(draw error)", ha="center", va="center",
                                transform=axs[5].transAxes, color="red", fontsize=10)
            else:
                axs[5].text(0.5, 0.5, "C2-C4\nnot available", ha="center", va="center",
                            transform=axs[5].transAxes, color="gray", fontsize=11)

            plt.suptitle(
                f"{case_label}  |  F1={f1_i:.3f} | IoU={iou_i:.3f} | "
                f"BF1@{tol_px}px={bf1_i:.3f} | ASSD={assd_txt} | HD95={hd95_txt}",
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
