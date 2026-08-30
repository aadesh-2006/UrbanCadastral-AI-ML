"""
Loss functions and evaluation metrics for building footprint segmentation.
Combines Binary Cross Entropy with Soft Dice Loss for robust boundary learning.
"""

import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.bce_weight * loss_bce + self.dice_weight * loss_dice

def compute_iou(preds_binary, targets):
    preds_flat = (preds_binary > 0.5).view(-1)
    targets_flat = (targets > 0.5).view(-1)
    intersection = (preds_flat & targets_flat).sum().float()
    union = (preds_flat | targets_flat).sum().float()
    if union == 0:
        return 1.0
    return (intersection / union).item()

def compute_dice(preds_binary, targets, smooth=1e-5):
    preds_flat = (preds_binary > 0.5).view(-1).float()
    targets_flat = (targets > 0.5).view(-1).float()
    intersection = (preds_flat * targets_flat).sum()
    return ((2.0 * intersection + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)).item()
