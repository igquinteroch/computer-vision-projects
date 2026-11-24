"""Train a keypoint model with a person-level validation split.

Usage (from the project root):
    python train.py v1
    python train.py v2

Each run writes the best checkpoint (by validation NME) to ``models/`` and a
per-epoch CSV log to ``logs/``. The holdout split is never loaded here.
"""
import argparse
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import config
from data_load import (CenterCrop, FacialKeypointsDataset, Normalize, RandomCrop,
                       RandomHorizontalFlip, RandomRotateScale, Rescale, ToTensor,
                       split_by_identity)
from models import ARCHITECTURES


class WingLoss(nn.Module):
    """Wing loss (Feng et al., CVPR 2018), computed in 224×224 input pixels.

    Logarithmic for errors below ``w`` (so small and medium errors still get a
    useful gradient), L1 above it. ``w = 10``, ``epsilon = 2`` are the paper's
    defaults.
    """

    def __init__(self, w: float = 10.0, epsilon: float = 2.0, scale: float = config.KEYPOINT_STD):
        super().__init__()
        self.w, self.epsilon, self.scale = w, epsilon, scale
        self.c = w - w * np.log(1 + w / epsilon)

    def forward(self, pred, target):
        x = (pred - target).abs() * self.scale  # normalized units -> pixels
        loss = torch.where(x < self.w, self.w * torch.log1p(x / self.epsilon), x - self.c)
        return loss.mean()


@dataclass
class TrainConfig:
    name: str
    arch: str = "net"
    loss: str = "mse"
    augment: bool = False
    batch_size: int = 32
    lr: float = 5e-4
    max_epochs: int = 100
    patience: int = 12               # early stopping, on validation NME
    lr_patience: int = 5             # ReduceLROnPlateau, on validation loss
    lr_factor: float = 0.5
    seed: int = 42
    weights: Path = field(default=None)

    def __post_init__(self):
        self.weights = self.weights or config.MODELS_DIR / f"keypoints_cnn_{self.name}.pt"


RUNS = {
    "v1": TrainConfig(name="v1"),
    "v2": TrainConfig(name="v2", arch="net_v2", loss="wing", augment=True),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train and validation frames. The third (holdout) split is discarded here."""
    train, val, _holdout = split_by_identity(pd.read_csv(config.TRAIN_CSV),
                                             config.SPLIT_FRACTIONS, config.SPLIT_SEED)
    return train, val


def train_transform(augment: bool):
    steps = [Rescale(config.RESCALE_SIZE)]
    if augment:
        steps += [RandomHorizontalFlip(), RandomRotateScale(15, (0.9, 1.1))]
    steps += [RandomCrop(config.IMAGE_SIZE), Normalize(), ToTensor()]
    return transforms.Compose(steps)


EVAL_TRANSFORM = transforms.Compose([Rescale(config.RESCALE_SIZE), CenterCrop(config.IMAGE_SIZE),
                                     Normalize(), ToTensor()])


def make_loss(name: str) -> nn.Module:
    return {"mse": nn.MSELoss(), "wing": WingLoss()}[name]


@torch.no_grad()
def evaluate(net, images, targets, criterion, batch_size: int = 128):
    """Loss, NME and mean point error (224-frame pixels) on preloaded tensors."""
    net.eval()
    device = next(net.parameters()).device
    preds = torch.cat([net(images[i: i + batch_size].to(device)).cpu()
                       for i in range(0, len(images), batch_size)])
    loss = criterion(preds, targets.flatten(1)).item()
    p = preds.view(-1, config.NUM_KEYPOINTS, 2) * config.KEYPOINT_STD
    t = targets * config.KEYPOINT_STD
    err = (p - t).norm(dim=-1)                                       # (N, 68) pixels
    iod = (t[:, config.LEFT_EYE_OUTER] - t[:, config.RIGHT_EYE_OUTER]).norm(dim=-1)
    nme = (err.mean(1) / iod).mean().item()
    return loss, nme, err.mean().item()


def train(cfg: TrainConfig, device: torch.device = config.DEVICE, verbose: bool = True) -> pd.DataFrame:
    seed_everything(cfg.seed)
    train_df, val_df = load_splits()
    train_ds = FacialKeypointsDataset(train_df, config.TRAIN_DIR, train_transform(cfg.augment), cache=True)
    val_ds = FacialKeypointsDataset(val_df, config.TRAIN_DIR, EVAL_TRANSFORM)
    val_x = torch.stack([val_ds[i]["image"] for i in range(len(val_ds))])
    val_y = torch.stack([val_ds[i]["keypoints"] for i in range(len(val_ds))])
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(cfg.seed))

    net = ARCHITECTURES[cfg.arch]().to(device)
    criterion = make_loss(cfg.loss)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience)

    if verbose:
        n_params = sum(p.numel() for p in net.parameters())
        print(f"[{cfg.name}] {cfg.arch}, {n_params / 1e6:.1f} M params, loss={cfg.loss}, "
              f"augment={cfg.augment} | train {len(train_ds)} imgs, val {len(val_ds)} imgs | {device}")

    rows, best_nme, best_epoch = [], float("inf"), 0
    for epoch in range(1, cfg.max_epochs + 1):
        start = time.time()
        net.train()
        running, seen = 0.0, 0
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["keypoints"].flatten(1).to(device)
            loss = criterion(net(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * len(x)
            seen += len(x)
        val_loss, val_nme, val_px = evaluate(net, val_x, val_y, criterion)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        improved = val_nme < best_nme
        if improved:
            best_nme, best_epoch = val_nme, epoch
            cfg.weights.parent.mkdir(exist_ok=True)
            torch.save(net.state_dict(), cfg.weights)
        rows.append({"epoch": epoch, "train_loss": running / seen, "val_loss": val_loss,
                     "val_nme": val_nme, "val_px_224": val_px, "lr": lr,
                     "seconds": time.time() - start, "best": improved})
        if verbose:
            print(f"epoch {epoch:3d} | train {running / seen:.4f} | val {val_loss:.4f} | "
                  f"NME {100 * val_nme:5.2f}% | {val_px:5.2f}px | lr {lr:.1e} | "
                  f"{time.time() - start:4.1f}s{' *' if improved else ''}", flush=True)
        if epoch - best_epoch >= cfg.patience:
            if verbose:
                print(f"early stop: no NME improvement for {cfg.patience} epochs")
            break

    log = pd.DataFrame(rows)
    config.LOGS_DIR.mkdir(exist_ok=True)
    log.to_csv(config.LOGS_DIR / f"training_log_{cfg.name}.csv", index=False)
    if verbose:
        print(f"best val NME {100 * best_nme:.2f}% at epoch {best_epoch} -> {cfg.weights.name}")
    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", choices=sorted(RUNS))
    args = parser.parse_args()
    print({k: str(v) for k, v in asdict(RUNS[args.run]).items()})
    train(RUNS[args.run])
