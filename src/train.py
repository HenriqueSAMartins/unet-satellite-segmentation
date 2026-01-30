# src/train.py
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

from dataset import SatelliteSegDataset
from unet import UNet
from losses import make_ce_loss
from metrics import confusion_matrix, iou_from_cm


# ======================
# GENERAL CONFIGURATION
# ======================
NUM_CLASSES = 10
IGNORE_INDEX = 255
IGNORE_LABELS = (0, 1)

BATCH_SIZE = 32  # CPU-friendly
EPOCHS = 2 #20
LR = 1e-3
BASE_CHANNELS = 32

VAL_SPLIT = 0.2
SEED = 42

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

BEST_CKPT_PATH = OUTPUT_DIR / "best_unet.pth"
LAST_CKPT_PATH = OUTPUT_DIR / "last_unet.pth"
HIST_PATH = OUTPUT_DIR / "training_history.pt"


def save_history_atomic(path: Path, obj: dict):
    """Write history safely (avoid corruption if interrupted mid-write)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def main():
    device = torch.device("cpu")
    torch.manual_seed(SEED)

    # ======================
    # DATASET + SPLIT
    # ======================
    full_ds = SatelliteSegDataset(
        images_dir=Path("dataset/train/images"),
        masks_dir=Path("dataset/train/masks"),
        ignore_labels=IGNORE_LABELS,
        ignore_index=IGNORE_INDEX
    )

    n_total = len(full_ds)
    n_val = int(VAL_SPLIT * n_total)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )

    # train_loader = DataLoader(
    #     train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    # )
    # val_loader = DataLoader(
    #     val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    # )
    train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=6, # number of CPU cores / 2
    persistent_workers=True,
    prefetch_factor=4,
)

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=6,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # ======================
    # MODEL, LOSS, OPTIMIZER
    # ======================
    model = UNet(
        in_channels=4,
        num_classes=NUM_CLASSES,
        base_channels=BASE_CHANNELS
    ).to(device)

    criterion = make_ce_loss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # ======================
    # LOAD HISTORY / RESUME (optional)
    # ======================
    history = {
        "epoch": [],
        "train_loss": [],
        "val_miou": [],
        "val_iou_per_class": [],  # list of lists
        "best_miou": 0.0,
    }
    best_miou = 0.0
    start_epoch = 1

    # If you want auto-resume when last checkpoint exists:
    if LAST_CKPT_PATH.exists():
        ckpt = torch.load(LAST_CKPT_PATH, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        best_miou = float(ckpt.get("best_miou", 0.0))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"🔁 Resuming from {LAST_CKPT_PATH} at epoch {start_epoch} (best_mIoU={best_miou:.4f})")

    if HIST_PATH.exists():
        history = torch.load(HIST_PATH, map_location="cpu")
        # keep best_miou consistent with history if present
        best_miou = max(best_miou, float(history.get("best_miou", 0.0)))

    # ======================
    # TRAINING LOOP
    # ======================
    for epoch in range(start_epoch, EPOCHS + 1):
        # -------- TRAIN --------
        model.train()
        train_loss = 0.0

        for i, (x, y) in enumerate(train_loader):
            print(f"Batch {i+1}/{len(train_loader)}")
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

        # -------- VALIDATION --------
        model.eval()
        cm = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = torch.argmax(logits, dim=1)

                cm += confusion_matrix(
                    pred, y,
                    num_classes=NUM_CLASSES,
                    ignore_index=IGNORE_INDEX
                )

        iou, miou = iou_from_cm(cm)

        # -------- LOG --------
        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] | "
            f"Train loss: {train_loss:.4f} | "
            f"Val mIoU: {miou:.4f}"
        )

        # -------- SAVE PER-EPOCH ARTIFACTS --------
        # (optional) save confusion matrix per epoch
        torch.save(cm.cpu(), OUTPUT_DIR / f"val_cm_epoch_{epoch:02d}.pt")

        # update history
        history["epoch"].append(int(epoch))
        history["train_loss"].append(float(train_loss))
        history["val_miou"].append(float(miou))
        history["val_iou_per_class"].append([float(v) for v in iou])

        # -------- CHECKPOINT (last) --------
        torch.save(
            {
                "epoch": int(epoch),
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "best_miou": float(best_miou),
                "config": {
                    "NUM_CLASSES": NUM_CLASSES,
                    "IGNORE_INDEX": IGNORE_INDEX,
                    "IGNORE_LABELS": IGNORE_LABELS,
                    "BATCH_SIZE": BATCH_SIZE,
                    "EPOCHS": EPOCHS,
                    "LR": LR,
                    "BASE_CHANNELS": BASE_CHANNELS,
                    "VAL_SPLIT": VAL_SPLIT,
                    "SEED": SEED,
                },
            },
            LAST_CKPT_PATH
        )

        # -------- CHECKPOINT (best) --------
        if miou > best_miou:
            best_miou = float(miou)
            history["best_miou"] = float(best_miou)
            torch.save(model.state_dict(), BEST_CKPT_PATH)
            print(f"  ✔ New best model saved (mIoU={best_miou:.4f})")

        # -------- SAVE HISTORY (every epoch, safe) --------
        history["best_miou"] = float(best_miou)
        save_history_atomic(HIST_PATH, history)
        print(f"  💾 History updated: {HIST_PATH}")

    print("Training finished.")
    print("Best validation mIoU:", best_miou)
    print("Best checkpoint saved at:", BEST_CKPT_PATH)
    print("Last checkpoint saved at:", LAST_CKPT_PATH)
    print("History saved at:", HIST_PATH)


if __name__ == "__main__":
    main()
