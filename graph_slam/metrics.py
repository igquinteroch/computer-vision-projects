"""Error metrics against ground truth, a dead-reckoning baseline, and seeded sweeps.

No trajectory alignment is needed: SLAM anchors the first pose at the true
starting point, so estimates and ground truth share the same frame.
"""

import numpy as np

import slam as slam_lib
import world


def final_pose_error(est_poses, true_poses):
    return float(np.linalg.norm(np.asarray(est_poses)[-1] - np.asarray(true_poses)[-1]))


def rmse(est, true):
    """Root mean squared Euclidean error over rows of (n, 2) arrays.

    Rows where ``est`` is NaN (landmarks that were never observed) are skipped.
    """
    err = np.asarray(est) - np.asarray(true)
    err = err[~np.isnan(err).any(axis=1)]
    return float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))


def dead_reckoning(data, start):
    """Poses obtained by integrating the commanded motions only (no measurements)."""
    moves = np.array([motion for _, motion in data], dtype=float)
    return np.vstack([start, np.asarray(start, dtype=float) + np.cumsum(moves, axis=0)])


def evaluate(sim):
    """Run full SLAM and dead reckoning on a ``Simulation`` and return the error metrics.

    Landmarks that were never observed are left out of the landmark RMSE.
    """
    poses, landmarks = slam_lib.slam_observed(sim.data, sim.num_landmarks, **sim.slam_args())
    dr = dead_reckoning(sim.data, sim.start)
    return {
        "final_pose_err": final_pose_error(poses, sim.true_poses),
        "pose_rmse": rmse(poses, sim.true_poses),
        "landmark_rmse": rmse(landmarks, sim.landmarks),
        "dr_final_pose_err": final_pose_error(dr, sim.true_poses),
        "dr_pose_rmse": rmse(dr, sim.true_poses),
    }


def sweep(param, values, seeds=range(20), **base):
    """Evaluate ``simulate_random`` for each value of ``param`` and each seed.

    ``base`` holds the remaining ``simulate_random`` arguments. Setting
    ``param="noise"`` changes motion and measurement noise together.
    Returns a list of dicts, one per (value, seed).
    """
    rows = []
    for v in values:
        kwargs = dict(base)
        if param == "noise":
            kwargs["motion_noise"] = kwargs["measurement_noise"] = v
        else:
            kwargs[param] = v
        for s in seeds:
            sim = world.simulate_random(seed=s, **kwargs)
            rows.append(dict(param=param, value=v, seed=s, **evaluate(sim)))
    return rows


def summarize(rows, keys=("final_pose_err", "pose_rmse", "landmark_rmse", "dr_pose_rmse")):
    """Mean and std per swept value: ``{value: {key: (mean, std)}}``."""
    out = {}
    for v in sorted({r["value"] for r in rows}):
        sub = [r for r in rows if r["value"] == v]
        out[v] = {k: (float(np.mean([r[k] for r in sub])), float(np.std([r[k] for r in sub])))
                  for k in keys}
    return out
