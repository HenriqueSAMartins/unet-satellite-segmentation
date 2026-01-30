from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from tifffile import imread

from unet import UNet


# =========================
# CONFIG (modify here)
# =========================
ROOT = Path(".")
CKPT_PATH = ROOT / "outputs" / "best_unet.pth"

SPLIT = "test"     # "train" or "test"
SHOW_NIR = True
SHOW_OVERLAY = True
OVERLAY_ALPHA = 0.35

NUM_CLASSES = 10
IGNORE_INDEX = 255  # used in training; here for consistency only

# Class palette (adjust if desired)
# (discrete colormap via matplotlib; you can also keep the default)
CLASS_NAMES = {
    0: "no_data",
    1: "clouds",
    2: "artificial",
    3: "cultivated",
    4: "broadleaf",
    5: "coniferous",
    6: "herbaceous",
    7: "natural_soil",
    8: "permanent_snow",
    9: "water",
}


# =========================
# HELPERS
# =========================
def ensure_hwc(x: np.ndarray) -> np.ndarray:
    # if comes as (C,H,W), convert to (H,W,C)
    if x.ndim == 3 and x.shape[0] in (3,4,5,6) and x.shape[1] > 32 and x.shape[2] > 32:
        return np.transpose(x, (1,2,0))
    return x

def stretch01(band: np.ndarray, p_low=2, p_high=98) -> np.ndarray:
    lo, hi = np.percentile(band, (p_low, p_high))
    return np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)

def preprocess_image(img_hwc: np.ndarray) -> torch.Tensor:
    """
    img_hwc: (H,W,4) float/uint
    returns x: (1,4,H,W) float32 in [0,1]
    """
    img = img_hwc.astype(np.float32)
    for c in range(img.shape[-1]):
        img[..., c] = stretch01(img[..., c], 2, 98)

    x = torch.from_numpy(np.transpose(img, (2,0,1))).float().unsqueeze(0)
    return x

def read_pair(split: str, index: int = 0, filename: str | None = None):
    images_dir = ROOT / "dataset" / split / "images"
    masks_dir  = ROOT / "dataset" / split / "masks"

    img_files = sorted(list(images_dir.glob("*.tif"))) + sorted(list(images_dir.glob("*.tiff")))
    if len(img_files) == 0:
        raise FileNotFoundError(f"No TIFF images found in: {images_dir}")

    if filename is None:
        img_path = img_files[index]
    else:
        img_path = images_dir / filename
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

    msk_path = masks_dir / img_path.name
    if not msk_path.exists():
        raise FileNotFoundError(f"Mask not found for image: {img_path.name} in {masks_dir}")

    img = ensure_hwc(imread(str(img_path)))
    msk = imread(str(msk_path)).astype(np.int64)  # (H,W)

    return img_path.name, img, msk


def load_model():
    device = torch.device("cpu")
    model = UNet(in_channels=4, num_classes=NUM_CLASSES, base_channels=32).to(device)
    state = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device


def predict_mask(model, device, img_hwc: np.ndarray) -> np.ndarray:
    x = preprocess_image(img_hwc).to(device)  # (1,4,H,W)
    with torch.no_grad():
        logits = model(x)                 # (1,C,H,W)
        pred = torch.argmax(logits, dim=1)[0]  # (H,W)
    return pred.cpu().numpy().astype(np.int64)


def plot_triplet(title_prefix: str, rgb: np.ndarray, nir: np.ndarray | None,
                 gt: np.ndarray, pred: np.ndarray):
    cols = 3 + (1 if SHOW_NIR else 0) + (1 if SHOW_OVERLAY else 0)
    plt.figure(figsize=(4.5 * cols, 5))

    k = 1
    plt.subplot(1, cols, k)
    plt.title(f"{title_prefix} RGB")
    plt.imshow(rgb)
    plt.axis("off")
    k += 1

    if SHOW_NIR and nir is not None:
        plt.subplot(1, cols, k)
        plt.title(f"{title_prefix} NIR")
        plt.imshow(nir, cmap="gray")
        plt.axis("off")
        k += 1

    plt.subplot(1, cols, k)
    plt.title(f"{title_prefix} GT Mask")
    plt.imshow(gt)
    plt.axis("off")
    k += 1

    plt.subplot(1, cols, k)
    plt.title(f"{title_prefix} Pred Mask")
    plt.imshow(pred)
    plt.axis("off")
    k += 1

    if SHOW_OVERLAY:
        plt.subplot(1, cols, k)
        plt.title(f"{title_prefix} Overlay (Pred)")
        plt.imshow(rgb)
        plt.imshow(pred, alpha=OVERLAY_ALPHA)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# =========================
# MAIN ENTRY
# =========================
def visualize_one(split: str = "test", index: int = 0, filename: str | None = None):
    assert split in ("train", "test")

    name, img, gt = read_pair(split=split, index=index, filename=filename)

    # img: (H,W,4) with bands [B,G,R,NIR] (assumed)
    if img.ndim != 3 or img.shape[-1] < 4:
        raise ValueError(f"Expected image (H,W,4). Got: {img.shape}")

    b = img[..., 0].astype(np.float32)
    g = img[..., 1].astype(np.float32)
    r = img[..., 2].astype(np.float32)
    nir = img[..., 3].astype(np.float32)

    # For display only: normalize RGB to [0,1]
    rgb = np.stack([stretch01(r), stretch01(g), stretch01(b)], axis=-1)
    nir_disp = stretch01(nir)

    model, device = load_model()
    pred = predict_mask(model, device, img)

    print(f"[{split}] {name}")
    print("  GT labels:", np.unique(gt))
    print("  Pred labels:", np.unique(pred))

    plot_triplet(title_prefix=f"{split.upper()} {name}", rgb=rgb, nir=nir_disp, gt=gt, pred=pred)


if __name__ == "__main__":
    # Choose ONE:
    visualize_one(split=SPLIT, index=0)
    # visualize_one(split=SPLIT, filename="10107.tif")
