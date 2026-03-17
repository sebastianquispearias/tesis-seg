import torch
import torch.nn as nn
import torch.nn.functional as F


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


def build_criterion(cfg: dict):
    loss_name = cfg.get("loss_name", "bce_dice").lower()

    if loss_name in ["bce_dice", "bce_plus_dice"]:
        return BCEPlusDice()

    raise ValueError(f"Loss no soportada: {loss_name}")