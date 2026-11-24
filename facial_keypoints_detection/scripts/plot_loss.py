"""Plot training curves from logs/ and save them to assets/.

- assets/loss_curve.png: validation NME and train/val loss per epoch for v1 and v2.
- assets/v0_loss_plateau.png: the submitted model's logged loss against the MSE
  of always predicting the mean shape.

Usage (from the project root):
    python scripts/plot_loss.py
"""
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
COLORS = {"v1": "#2a78d6", "v2": "#e07b39"}
LABELS = {"v1": "v1 · v0 architecture, input fixed", "v2": "v2 · compact head + aug + Wing"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def mean_shape_mse(augment_passes: int = 3) -> float:
    """MSE (normalized units) of the best constant predictor under the v0 training transform."""
    from torchvision import transforms
    from data_load import FacialKeypointsDataset, Normalize, RandomCrop, Rescale, ToTensor
    np.random.seed(0)
    ds = FacialKeypointsDataset(config.TRAIN_CSV, config.TRAIN_DIR, cache=True, transform=transforms.Compose(
        [Rescale(config.RESCALE_SIZE), RandomCrop(config.IMAGE_SIZE), Normalize(), ToTensor()]))
    y = np.stack([ds[i]["keypoints"].numpy().ravel()
                  for _ in range(augment_passes) for i in range(len(ds))])
    return float(((y - y.mean(0)) ** 2).mean())


def plot_curves(baseline_nme: float = None) -> Path:
    logs = {k: pd.read_csv(config.LOGS_DIR / f"training_log_{k}.csv") for k in ("v1", "v2")}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=150, facecolor=SURFACE)

    ax = axes[0]
    for k, log in logs.items():
        ax.plot(log.epoch, 100 * log.val_nme, color=COLORS[k], linewidth=2, label=LABELS[k])
        best = log.loc[log.val_nme.idxmin()]
        ax.scatter(best.epoch, 100 * best.val_nme, color=COLORS[k], zorder=3, s=25)
        ax.annotate(f"{100 * best.val_nme:.1f}%", (best.epoch, 100 * best.val_nme),
                    xytext=(0, -14), textcoords="offset points", ha="center", color=INK, fontsize=9)
    if baseline_nme is not None:
        ax.axhline(100 * baseline_nme, color=MUTED, linestyle="--", linewidth=1)
        ax.text(1, 100 * baseline_nme + 0.6, "mean-shape baseline", color=MUTED, fontsize=9)
    ax.set_title("Validation NME (unseen people)", color=INK, loc="left")
    ax.set_xlabel("epoch", color=MUTED)
    ax.set_ylabel("NME, % of inter-ocular distance", color=MUTED)
    ax.legend(frameon=False, labelcolor=INK, fontsize=8)

    for ax, k, unit in [(axes[1], "v1", "MSE, normalized units"), (axes[2], "v2", "Wing loss, px")]:
        log = logs[k]
        ax.plot(log.epoch, log.train_loss, color=COLORS[k], alpha=0.45, linewidth=1.5, label="train")
        ax.plot(log.epoch, log.val_loss, color=COLORS[k], linewidth=2, label="validation")
        ax.set_title(f"{k} loss", color=INK, loc="left")
        ax.set_xlabel("epoch", color=MUTED)
        ax.set_ylabel(unit, color=MUTED)
        ax.legend(frameon=False, labelcolor=INK, fontsize=8)
    for ax in axes:
        _style(ax)

    out = config.ASSETS_DIR / "loss_curve.png"
    config.ASSETS_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_v0_plateau(constant_mse: float) -> Path:
    text = (config.LOGS_DIR / "training_log_v0.txt").read_text()
    loss = np.array([float(x) for x in re.findall(r"Avg\. Loss: ([\d.]+)", text)])
    per_epoch = len(loss) // 10
    x = np.arange(1, len(loss) + 1) / per_epoch

    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=150, facecolor=SURFACE)
    ax.plot(x, loss, color="#a9c8ef", linewidth=0.8, label="v0 logged loss (mean of 10 batches)")
    smooth = np.convolve(loss, np.ones(per_epoch) / per_epoch, mode="valid")
    ax.plot(x[per_epoch - 1:], smooth, color="#2a78d6", linewidth=2, label="1-epoch moving average")
    ax.axhline(constant_mse, color="#c9408a", linestyle="--", linewidth=1.5,
               label=f"always predict the mean shape ({constant_mse:.3f})")
    ax.set_ylim(0, 0.45)
    ax.set_xlabel("epoch", color=MUTED)
    ax.set_ylabel("MSE, normalized units", color=MUTED)
    ax.set_title("Submitted model (v0): the training loss never beats a constant prediction",
                 color=INK, loc="left")
    ax.legend(frameon=False, labelcolor=INK, fontsize=8, loc="upper right")
    _style(ax)
    out = config.ASSETS_DIR / "v0_loss_plateau.png"
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def main(baseline_nme: float = None) -> None:
    print("Saved", plot_curves(baseline_nme))
    print("Saved", plot_v0_plateau(mean_shape_mse()))


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else None)
