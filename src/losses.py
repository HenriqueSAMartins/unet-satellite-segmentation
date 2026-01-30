import torch
import torch.nn as nn

def make_ce_loss(ignore_index: int = 255, class_weights=None):
    """
    logits: (B, C, H, W)
    target: (B, H, W) with labels in [0..C-1] or ignore_index
    """
    weight = None
    if class_weights is not None:
        weight = torch.tensor(class_weights, dtype=torch.float32)
    return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
