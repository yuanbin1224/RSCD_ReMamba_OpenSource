import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_binary_target(target):

    if target.dim() == 4 and target.shape[1] > 1:
        target = torch.argmax(target, dim=1, keepdim=True)
    elif target.dim() == 3:
        target = target.unsqueeze(1)
    return (target.float() > 0).float()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, eps=1e-6):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits, target):
        target = normalize_binary_target(target)
        prob = torch.sigmoid(logits)
        prob = prob.contiguous().view(prob.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)

        intersection = (prob * target).sum(dim=1)
        union = prob.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth + self.eps)
        return 1.0 - dice.mean()


class WeightedBCEDiceLoss(nn.Module):

    def __init__(self, pos_weight=2.0, dice_weight=0.5, dynamic_pos_weight=False):
        super().__init__()
        self.register_buffer("static_pos_weight", torch.tensor([float(pos_weight)]))
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
        self.dynamic_pos_weight = dynamic_pos_weight

    def _pos_weight(self, target):
        if not self.dynamic_pos_weight:
            return self.static_pos_weight.to(device=target.device, dtype=target.dtype)

        pos = target.sum().clamp_min(1.0)
        neg = (target.numel() - target.sum()).clamp_min(1.0)
        return (neg / pos).clamp(1.0, 20.0).detach().view(1).to(device=target.device, dtype=target.dtype)

    def forward(self, output, target):
        logits = output["change_pred"] if isinstance(output, dict) else output
        target = normalize_binary_target(target)
        if target.shape[-2:] != logits.shape[-2:]:
            target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")

        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self._pos_weight(target))
        dice = self.dice(logits, target)
        return bce + self.dice_weight * dice
