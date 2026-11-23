"""Plot the training loss from training_log.txt and save it to assets/loss_curve.png.

Usage (from the project root):
    python scripts/plot_loss.py
"""
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

LINE_RE = re.compile(r"Epoch \[(\d+)/\d+\], Step \[(\d+)/(\d+)\], Loss: ([\d.]+)")
WINDOW = 200  # steps in the moving average

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
RAW, SMOOTH = "#a9c8ef", "#2a78d6"  # one hue: light tint for raw, full for trend


def parse_log(path: Path):
    epochs, losses, steps_per_epoch = [], [], None
    for line in path.read_text().splitlines():
        m = LINE_RE.search(line)
        if m:
            epochs.append(int(m.group(1)))
            steps_per_epoch = int(m.group(3))
            losses.append(float(m.group(4)))
    return np.array(epochs), np.array(losses), steps_per_epoch


def main() -> None:
    epochs, losses, steps_per_epoch = parse_log(config.TRAINING_LOG)
    steps = np.arange(1, len(losses) + 1)
    smooth = np.convolve(losses, np.ones(WINDOW) / WINDOW, mode="valid")

    fig, ax = plt.subplots(figsize=(9, 4), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(steps, losses, color=RAW, linewidth=0.6, label="per step")
    ax.plot(steps[WINDOW - 1:], smooth, color=SMOOTH, linewidth=2,
            label=f"{WINDOW}-step moving average")

    for e in range(1, epochs.max()):
        ax.axvline(e * steps_per_epoch, color=MUTED, linewidth=1, linestyle="--")
    for e in range(1, epochs.max() + 1):
        ax.text((e - 0.5) * steps_per_epoch, 5.6, f"epoch {e}",
                ha="center", color=MUTED, fontsize=9)

    final = smooth[-1]
    ax.annotate(f"{final:.2f}", xy=(steps[-1], final), xytext=(6, 0),
                textcoords="offset points", va="center", color=INK, fontsize=9)

    ax.set_ylim(1.5, 6)
    ax.set_xlim(0, len(losses) + 900)
    ax.set_xlabel("training step (batch of 64)", color=MUTED)
    ax.set_ylabel("cross-entropy loss", color=MUTED)
    ax.set_title("Training loss — COCO 2014, 3 epochs", color=INK, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, loc="upper right", labelcolor=INK, bbox_to_anchor=(1, 0.92))

    config.ASSETS_DIR.mkdir(exist_ok=True)
    out = config.ASSETS_DIR / "loss_curve.png"
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    print(f"Saved {out}  (final moving-average loss {final:.3f}, "
          f"perplexity {np.exp(final):.2f})")
    print(f"First-epoch-end avg {losses[:steps_per_epoch][-WINDOW:].mean():.3f}, "
          f"second {losses[steps_per_epoch:2*steps_per_epoch][-WINDOW:].mean():.3f}")


if __name__ == "__main__":
    main()
