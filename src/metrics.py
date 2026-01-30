import torch

@torch.no_grad()
def confusion_matrix(pred, target, num_classes: int, ignore_index: int = 255):
    """
    pred: (B,H,W) int64
    target: (B,H,W) int64
    """
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]

    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(target.view(-1), pred.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm

@torch.no_grad()
def iou_from_cm(cm: torch.Tensor):
    """
    cm: (C,C)
    returns iou_per_class: (C,), miou: scalar
    """
    tp = torch.diag(cm).float()
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp
    denom = tp + fp + fn
    iou = torch.where(denom > 0, tp / denom, torch.zeros_like(denom))
    miou = iou[denom > 0].mean() if (denom > 0).any() else torch.tensor(0.0)
    return iou, miou
