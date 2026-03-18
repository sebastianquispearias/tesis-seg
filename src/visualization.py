import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def show_config_summary(cfg: dict):
    print("===== CONFIG =====")
    for k, v in cfg.items():
        print(f"{k}: {v}")


def chw_to_hwc(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3 and x.shape[0] in [1, 3]:
        return np.transpose(x, (1, 2, 0))
    return x


def show_dataset_examples(dataset, n: int = 4):
    n = min(n, len(dataset))
    fig, axes = plt.subplots(n, 3, figsize=(10, 4 * n))

    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n):
        sample = dataset[i]
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


@torch.no_grad()
def show_predictions(model, loader, device="cuda", thr=0.5, max_show=6, out_dir=None):
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

            img = np.transpose(img, (1, 2, 0))
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img[..., 0]

            img_vis = img.copy()
            if img_vis.ndim == 3 and img_vis.shape[-1] == 3:
                img_vis = np.clip(img_vis, 0, 1)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(img_vis, cmap=None if img_vis.ndim == 3 else "gray")
            axes[0].set_title("Image")
            axes[0].axis("off")

            axes[1].imshow(gt, cmap="gray")
            axes[1].set_title("GT")
            axes[1].axis("off")

            axes[2].imshow(pr, cmap="gray")
            axes[2].set_title("Prediction")
            axes[2].axis("off")

            plt.tight_layout()

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
    val_f1 = [row["val_f1"] for row in history]
    val_iou = [row["val_iou"] for row in history]

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