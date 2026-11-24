"""Plots and animations for Graph SLAM runs."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

import slam as slam_lib

TRUE = "#8a8f98"
EST = "#1f6feb"
DR = "#d29922"
LM_TRUE = "#6e40c9"
ERR = "#cf222e"


def setup_world_ax(ax, world_size, title=None):
    ax.set_xlim(0, world_size)
    ax.set_ylim(0, world_size)
    ax.set_aspect("equal")
    ax.grid(True, color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if title:
        ax.set_title(title, fontsize=11)


def plot_world(ax, world_size, true_poses=None, est_poses=None, true_landmarks=None,
               est_landmarks=None, dr_poses=None, robot=None, measurement_range=None,
               measurements=None, title=None, legend=True):
    """Draw any subset of: true/estimated/dead-reckoning paths, true/estimated landmarks,
    the robot with its square sensing window, and the current measurements as rays."""
    setup_world_ax(ax, world_size, title)
    if true_poses is not None:
        p = np.asarray(true_poses)
        ax.plot(p[:, 0], p[:, 1], "-", color=TRUE, lw=1.5, label="true path")
    if dr_poses is not None:
        p = np.asarray(dr_poses)
        ax.plot(p[:, 0], p[:, 1], "--", color=DR, lw=1.2, label="dead reckoning")
    if est_poses is not None:
        p = np.asarray(est_poses)
        ax.plot(p[:, 0], p[:, 1], "o-", color=EST, lw=1.2, ms=2.5, label="SLAM estimate")
    if true_landmarks is not None:
        lm = np.asarray(true_landmarks)
        ax.plot(lm[:, 0], lm[:, 1], "x", color=LM_TRUE, ms=8, mew=2, label="true landmarks")
    if est_landmarks is not None:
        lm = np.asarray(est_landmarks)
        ok = ~np.isnan(lm).any(axis=1)
        if true_landmarks is not None:
            for (ex, ey), (tx, ty) in zip(lm[ok], np.asarray(true_landmarks)[ok]):
                ax.plot([ex, tx], [ey, ty], "-", color=ERR, lw=1)
        ax.plot(lm[ok, 0], lm[ok, 1], "o", mfc="none", mec=EST, ms=8, mew=1.5,
                label="estimated landmarks")
    if robot is not None:
        rx, ry = robot
        if measurement_range is not None and measurement_range > 0:
            r = measurement_range
            ax.add_patch(Rectangle((rx - r, ry - r), 2 * r, 2 * r, fill=True, fc=EST,
                                   alpha=0.07, ec=EST, lw=0.8))
        for _, dx, dy in measurements or []:
            ax.plot([rx, rx + dx], [ry, ry + dy], "-", color=EST, lw=0.8, alpha=0.6)
        ax.plot(rx, ry, "o", color=ERR, ms=9, zorder=5, label="robot (true)")
    if legend:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    return ax


def plot_omega(ax, omega, N, num_landmarks, title=None):
    """Sparsity pattern of ``omega`` with pose / landmark blocks marked."""
    nz = (np.abs(omega) > 0).astype(float)
    ax.imshow(nz, cmap="Blues", vmin=0, vmax=1.3, interpolation="nearest")
    edge = 2 * N - 0.5
    ax.axhline(edge, color=ERR, lw=1)
    ax.axvline(edge, color=ERR, lw=1)
    size = 2 * (N + num_landmarks)
    ax.set_xticks([N - 0.5, 2 * N + num_landmarks - 0.5])
    ax.set_xticklabels(["poses", "landmarks"])
    ax.set_yticks([N - 0.5, 2 * N + num_landmarks - 0.5])
    ax.set_yticklabels(["poses", "landmarks"], rotation=90, va="center")
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    if title:
        ax.set_title(title, fontsize=11)
    return ax


def incremental_estimates(sim):
    """Full SLAM re-solved on every prefix ``data[:t]``: list of (poses, landmarks), t = 1..N-1."""
    out = []
    for t in range(1, sim.N):
        out.append(slam_lib.slam_observed(sim.data[:t], sim.num_landmarks, **sim.slam_args()))
    return out


def current_pose_errors(sim, estimates=None):
    """Error of the newest pose estimate at each time step (what an online system would report)."""
    estimates = estimates if estimates is not None else incremental_estimates(sim)
    return np.array([np.linalg.norm(P[-1] - sim.true_poses[t])
                     for t, (P, _) in enumerate(estimates, start=1)])


def animate_slam(sim, path, fps=6, hold_last=12, dpi=80, closure_step=None):
    """Write a GIF: left, the world with the map forming; right, current-pose error over time."""
    estimates = incremental_estimates(sim)
    errors = current_pose_errors(sim, estimates)
    steps = np.arange(1, sim.N)

    fig, (ax, ax_err) = plt.subplots(1, 2, figsize=(10, 5),
                                     gridspec_kw=dict(width_ratios=[1, 0.85]))
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.2, top=0.9, wspace=0.22)

    def draw(frame):
        t = min(frame, len(estimates) - 1)
        poses, landmarks = estimates[t]
        ax.clear()
        plot_world(ax, sim.world_size, true_poses=sim.true_poses[:t + 2], est_poses=poses,
                   true_landmarks=sim.landmarks, est_landmarks=landmarks,
                   robot=sim.true_poses[t + 1], measurement_range=sim.measurement_range,
                   measurements=sim.data[t + 1][0] if t + 1 < len(sim.data) else None,
                   legend=False, title="step %d / %d" % (t + 1, len(estimates)))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3, fontsize=8,
                  frameon=False)
        ax_err.clear()
        ax_err.plot(steps[:t + 1], errors[:t + 1], "-", color=EST, lw=1.8)
        ax_err.set_xlim(0, len(steps) + 1)
        ax_err.set_ylim(0, errors.max() * 1.1)
        ax_err.set_xlabel("time step")
        ax_err.set_ylabel("error of the current pose estimate")
        ax_err.grid(True, color="#e6e6e6")
        for s in ("top", "right"):
            ax_err.spines[s].set_visible(False)
        if closure_step is not None:
            ax_err.axvline(closure_step, color=ERR, ls=":", lw=1.2)
            ax_err.text(closure_step + 0.5, errors.max() * 1.02, "back at start", color=ERR,
                        fontsize=9, va="top")
        ax_err.set_title("current-pose error", fontsize=11)

    anim = FuncAnimation(fig, draw, frames=len(estimates) + hold_last)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return errors


def plot_sweep(ax, summary, key, label, color, xlabel=None, logx=False):
    """Mean +/- std of ``summary[value][key]`` (output of ``metrics.summarize``)."""
    xs = np.array(sorted(summary))
    mean = np.array([summary[x][key][0] for x in xs])
    std = np.array([summary[x][key][1] for x in xs])
    ax.plot(xs, mean, "o-", color=color, lw=1.8, ms=4, label=label)
    ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.15)
    if logx:
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([("%g" % x) for x in xs])
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(True, color="#e6e6e6")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


def sweep_figure(summary, xlabel, logx=False, title=None):
    """Two panels for a ``metrics.summarize`` result: pose RMSE (SLAM vs dead reckoning)
    and landmark RMSE, as mean +/- std over seeds."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    plot_sweep(a, summary, "dr_pose_rmse", "dead reckoning", DR, xlabel, logx)
    plot_sweep(a, summary, "pose_rmse", "Graph SLAM", EST, xlabel, logx)
    a.set_ylabel("pose RMSE")
    a.legend(fontsize=9, frameon=False)
    plot_sweep(b, summary, "landmark_rmse", "Graph SLAM", EST, xlabel, logx)
    b.set_ylabel("landmark RMSE")
    b.set_ylim(bottom=0)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig
