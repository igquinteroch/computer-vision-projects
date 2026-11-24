"""Regenerate the noise / horizon sweeps: assets/sweep_results.csv and two figures.

Each point is the mean over 20 seeded runs (seeds 0-19) of the random world
(5 landmarks, 100 x 100, sensing range 50, step 20).

Usage (from graph_slam/): python scripts/run_sweeps.py
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import metrics  # noqa: E402
import viz  # noqa: E402

NOISE_VALUES = [0.5, 1, 2, 4, 8]
N_VALUES = [10, 20, 50, 100, 200]
KEYS = ("final_pose_err", "pose_rmse", "landmark_rmse", "dr_final_pose_err", "dr_pose_rmse")


def run():
    """Return ``(rows_noise, rows_N)``: per-seed results of both sweeps."""
    rows_noise = metrics.sweep("noise", NOISE_VALUES, N=20)
    rows_N = metrics.sweep("N", N_VALUES, motion_noise=2.0, measurement_noise=2.0)
    return rows_noise, rows_N


def main():
    assets = ROOT / "assets"
    rows_noise, rows_N = run()
    with open(assets / "sweep_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["param", "value", "seed", *KEYS])
        w.writeheader()
        for r in rows_noise + rows_N:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})

    fig = viz.sweep_figure(metrics.summarize(rows_noise, KEYS),
                           "noise amplitude (motion = measurement)", logx=True)
    fig.savefig(assets / "error_vs_noise.png", dpi=120)
    fig = viz.sweep_figure(metrics.summarize(rows_N, KEYS), "number of poses N", logx=True)
    fig.savefig(assets / "error_vs_N.png", dpi=120)
    print("wrote sweep_results.csv, error_vs_noise.png, error_vs_N.png to", assets)


if __name__ == "__main__":
    main()
