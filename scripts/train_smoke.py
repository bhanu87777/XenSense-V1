"""Train the lightweight smoke/fog CNN (xensense/models/smoke_cnn.py).

Dataset layout (any smoke-vs-clear image set, e.g. frames from RESIDE/RTTS):
    data/smoke_dataset/
        smoke/      *.jpg|*.png   (smoke or fog present)
        clear/      *.jpg|*.png   (normal conditions)

Usage:
    python scripts/train_smoke.py --data data/smoke_dataset --epochs 15
Saves best weights to weights/smoke_cnn.pt — the pipeline picks them up
automatically on the next run.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xensense.models.smoke_cnn import SmokeCNN  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class SmokeDataset(Dataset):
    def __init__(self, root: str, augment: bool = False):
        self.samples = []
        for label, sub in [(1, "smoke"), (0, "clear")]:
            folder = Path(root) / sub
            if not folder.is_dir():
                raise FileNotFoundError(f"Expected folder: {folder}")
            for f in folder.iterdir():
                if f.suffix.lower() in IMG_EXTS:
                    self.samples.append((str(f), label))
        if not self.samples:
            raise RuntimeError(f"No images found under {root}")
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = cv2.imread(path)
        img = cv2.resize(img, (128, 128))
        if self.augment:
            if np.random.rand() < 0.5:
                img = cv2.flip(img, 1)
            if np.random.rand() < 0.3:
                img = cv2.convertScaleAbs(img, alpha=np.random.uniform(0.8, 1.2),
                                          beta=np.random.uniform(-15, 15))
        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        return tensor, torch.tensor(float(label))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root (smoke/ + clear/)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="weights/smoke_cnn.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    full = SmokeDataset(args.data, augment=True)
    n_val = max(1, int(len(full) * 0.15))
    train_set, val_set = random_split(full, [len(full) - n_val, n_val])
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch)

    model = SmokeCNN().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optim.step()
            total_loss += loss.item() * len(x)

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = (torch.sigmoid(model(x)) > 0.5).float()
                correct += (pred == y).sum().item()
                total += len(y)
        acc = correct / max(total, 1)
        print(f"epoch {epoch:02d}  loss {total_loss / len(train_set):.4f}  val_acc {acc:.3f}")
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.out)
            print(f"  saved -> {args.out}")

    print(f"Best validation accuracy: {best_acc:.3f} "
          f"(target from report 3.3: > 0.85)")


if __name__ == "__main__":
    main()
