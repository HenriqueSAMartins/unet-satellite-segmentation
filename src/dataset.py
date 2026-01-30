from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from tifffile import imread

def _ensure_hwc(x: np.ndarray) -> np.ndarray:
    # (C,H,W) -> (H,W,C)
    if x.ndim == 3 and x.shape[0] in (3,4,5,6) and x.shape[1] > 32 and x.shape[2] > 32:
        return np.transpose(x, (1,2,0))
    return x

def _stretch01(band: np.ndarray, p_low=2, p_high=98) -> np.ndarray:
    lo, hi = np.percentile(band, (p_low, p_high))
    return np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)

class SatelliteSegDataset(Dataset):
    """
    Returns:
      x: float32 tensor (4, H, W)
      y: int64 tensor (H, W) with labels, optionally remapped to ignore_index
    """
    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path | None = None,
        ignore_labels=(0, 1),
        ignore_index=255,
        stretch_pct=(2, 98),
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir is not None else None
        self.ignore_labels = set(ignore_labels) if ignore_labels is not None else set()
        self.ignore_index = int(ignore_index)
        self.stretch_pct = stretch_pct

        self.img_paths = sorted(self.images_dir.glob("*.tif")) + sorted(self.images_dir.glob("*.tiff"))
        if len(self.img_paths) == 0:
            raise FileNotFoundError(f"No TIFF files found in {self.images_dir}")

        if self.masks_dir is not None and not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks dir not found: {self.masks_dir}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        img = _ensure_hwc(imread(str(img_path))).astype(np.float32)  # H,W,4 (B,G,R,NIR)

        # normalize for training stability (simple & robust)
        p1, p2 = self.stretch_pct
        for c in range(img.shape[-1]):
            img[..., c] = _stretch01(img[..., c], p1, p2)

        # HWC -> CHW
        x = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()  # (4,H,W)

        if self.masks_dir is None:
            return x

        msk_path = self.masks_dir / img_path.name
        y = imread(str(msk_path)).astype(np.int64)  # (H,W)

        if len(self.ignore_labels) > 0:
            for lab in self.ignore_labels:
                y[y == lab] = self.ignore_index

        y = torch.from_numpy(y).long()
        return x, y
