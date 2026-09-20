import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        num = 2.0 * (probs * targets).sum() + self.eps
        den = probs.sum() + targets.sum() + self.eps
        return 1.0 - (num / den)


class BCEPlusDice(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, epoch: int | None = None):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        total = self.bce_weight * bce_loss + self.dice_weight * dice_loss

        components = {
            "bce": float(bce_loss.detach()),
            "dice_loss": float(dice_loss.detach()),
            "sup_total": float(total.detach()),
        }
        return total, components


def distance_maps(targets: torch.Tensor):
    """Euclidean distance maps of a batch of binary reference masks.

    Returns two arrays with the same shape as the input. The first is the signed
    distance to the reference contour, negative inside the object and positive
    outside it. The second is the unsigned distance to that same contour. Samples
    whose mask is entirely background or entirely foreground have no contour, so
    both of their maps are left at zero and the boundary term ignores them.

    The transform runs on the CPU because it is a sequential scan that no GPU
    kernel here would accelerate; at 320x320 and a batch of five it costs a few
    milliseconds against a training step of a quarter of a second.
    """
    binary = (targets.detach().cpu().numpy() > 0.5)
    signed = np.zeros(binary.shape, dtype=np.float32)
    to_contour = np.zeros(binary.shape, dtype=np.float32)

    for b in range(binary.shape[0]):
        for c in range(binary.shape[1]):
            mask = binary[b, c]
            if not mask.any() or mask.all():
                continue
            outside = distance_transform_edt(~mask)
            inside = distance_transform_edt(mask)
            signed[b, c] = outside - inside
            to_contour[b, c] = outside + inside

    return signed, to_contour


class BCEPlusDicePlusBoundary(nn.Module):
    """Region loss with an added contour term, for the boundary-error question.

    The supervised loss used throughout this work is binary cross-entropy plus
    Dice, both of which are computed over the region and treat a pixel at the
    centre of a vertebra exactly like one on its edge. This class keeps that loss
    untouched and adds a third term that does look at the edge, so that the two
    can be compared with a single scalar changing between runs.

    Two formulations are available, chosen by ``mode``.

    ``dw`` weights the per-pixel cross-entropy by a Gaussian of the distance to
    the reference contour, normalised so that its magnitude stays comparable to
    the plain cross-entropy. It concentrates the existing supervision near the
    edge without introducing a differently scaled quantity.

    ``kervadec`` is the boundary loss of Kervadec et al. (2019): the integral of
    the predicted probability against the signed distance map of the reference.
    It is negative where the prediction correctly falls inside the object and
    positive where it spills outside, so minimising it moves the predicted
    contour towards the reference one along the shortest path.

    The two terms are not naturally on the same scale: distances are measured in
    pixels and reach a hundred or more, whereas a cross-entropy sits near one. The
    signed map is therefore divided by its largest magnitude in the batch, which
    leaves the sign and the relative geometry untouched and lets a single range of
    ``boundary_weight`` be swept across both modes. The undivided value is still
    reported, so the raw magnitude is never hidden.

    The term can be ramped in over a number of epochs, in the same way the
    unsupervised weight is ramped during semi-supervised training, which keeps
    the early epochs governed by the region loss alone.
    """

    def __init__(
        self,
        mode: str = "dw",
        boundary_weight: float = 1.0,
        dw_sigma: float = 5.0,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        start_epoch: int = 0,
        warmup_epochs: int = 0,
    ):
        super().__init__()
        if mode not in ("dw", "kervadec"):
            raise ValueError(f"boundary_mode no soportado: {mode}")
        self.mode = mode
        self.boundary_weight = float(boundary_weight)
        self.dw_sigma = float(dw_sigma)
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.start_epoch = int(start_epoch)
        self.warmup_epochs = int(warmup_epochs)

        self.bce_pixelwise = nn.BCEWithLogitsLoss(reduction="none")
        self.dice = DiceLoss()

    def current_weight(self, epoch: int | None) -> float:
        """Weight of the contour term at this epoch, after the optional ramp."""
        if epoch is None or self.warmup_epochs <= 0:
            return self.boundary_weight
        if epoch < self.start_epoch:
            return 0.0
        progress = (epoch - self.start_epoch) / float(self.warmup_epochs)
        return self.boundary_weight * min(1.0, progress)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, epoch: int | None = None):
        bce_map = self.bce_pixelwise(logits, targets)
        bce_loss = bce_map.mean()
        dice_loss = self.dice(logits, targets)

        signed_np, to_contour_np = distance_maps(targets)

        if self.mode == "dw":
            gauss = np.exp(-(to_contour_np ** 2) / (2.0 * self.dw_sigma ** 2))
            weights = torch.from_numpy(gauss).to(logits.device, dtype=bce_map.dtype)
            # Normalising by the mean weight keeps this term on the same scale as
            # the plain cross-entropy, so that boundary_weight means the same
            # thing here as it does in the other mode.
            denom = weights.mean().clamp_min(1e-6)
            boundary_loss = (weights * bce_map).mean() / denom
            raw_scale = 1.0
        else:
            signed = torch.from_numpy(signed_np).to(logits.device, dtype=bce_map.dtype)
            raw_scale = float(np.abs(signed_np).max())
            if raw_scale > 0.0:
                signed = signed / raw_scale
            boundary_loss = (signed * torch.sigmoid(logits)).mean()

        weight_now = self.current_weight(epoch)
        total = (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
            + weight_now * boundary_loss
        )

        components = {
            "bce": float(bce_loss.detach()),
            "dice_loss": float(dice_loss.detach()),
            "boundary_loss": float(boundary_loss.detach()),
            "boundary_raw_scale": raw_scale,
            "boundary_weight_now": float(weight_now),
            "sup_total": float(total.detach()),
        }
        return total, components


def build_criterion(cfg: dict):
    loss_name = cfg.get("loss_name", "bce_dice").lower()

    if loss_name in ["bce_dice", "bce_plus_dice"]:
        return BCEPlusDice()

    if loss_name in ["bce_dice_boundary", "bce_plus_dice_plus_boundary"]:
        return BCEPlusDicePlusBoundary(
            mode=cfg.get("boundary_mode", "dw"),
            boundary_weight=cfg.get("boundary_weight", 1.0),
            dw_sigma=cfg.get("dw_sigma", 5.0),
            start_epoch=cfg.get("boundary_start_epoch", 0),
            warmup_epochs=cfg.get("boundary_warmup_epochs", 0),
        )

    raise ValueError(f"Loss no soportada: {loss_name}")
