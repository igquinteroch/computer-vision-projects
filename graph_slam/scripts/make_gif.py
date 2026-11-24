"""Regenerate assets/slam_loop.gif: Graph SLAM on the square circuit, with loop closure.

Usage (from graph_slam/): python scripts/make_gif.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import viz  # noqa: E402
import world  # noqa: E402

# Same configuration as notebook 04 (loop world). Seed 6 is shown because the
# loop-closure correction is easy to see; the 8-seed average is in the README.
LOOP = dict(spacing=20.0, offset=6.0, laps=1.25)
RUN = dict(step=5.0, measurement_range=12.0, motion_noise=3.0, measurement_noise=1.0, seed=6)


def main():
    waypoints, landmarks = world.loop_world(**LOOP)
    sim = world.simulate_waypoints(waypoints, landmarks, name="loop", **RUN)
    half = sim.N // 2
    closure = half + int(np.argmin(np.linalg.norm(sim.true_poses[half:] - waypoints[0], axis=1)))
    out = ROOT / "assets" / "slam_loop.gif"
    errors = viz.animate_slam(sim, out, closure_step=closure)
    print("wrote %s (%.1f KB)" % (out, out.stat().st_size / 1024))
    print("current-pose error: max before closure %.2f, just before %.2f, mean after %.2f"
          % (errors[:closure - 1].max(), errors[closure - 2], errors[closure + 1:].mean()))


if __name__ == "__main__":
    main()
