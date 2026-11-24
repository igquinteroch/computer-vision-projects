"""Worlds and simulated runs with ground truth.

``simulate_random`` reproduces the original data generator: random integer
landmarks, and a robot that drives in a straight line and picks a new random
heading whenever the next move would leave the world. A run is retried until
every landmark has been seen at least once (otherwise SLAM is unsolvable).

``simulate_waypoints`` drives the robot along a pre-planned path instead, which
is used for the structured worlds (a corridor and a closed loop).
"""

from dataclasses import dataclass, field

import numpy as np

from robot import Robot


@dataclass
class Simulation:
    data: list                    # [[measurements, [dx, dy]], ...], N - 1 steps
    true_poses: np.ndarray        # (N, 2)
    landmarks: np.ndarray         # (L, 2)
    world_size: float
    measurement_range: float
    motion_noise: float
    measurement_noise: float
    start: tuple
    name: str = "random"
    extra: dict = field(default_factory=dict)

    @property
    def N(self):
        return len(self.true_poses)

    @property
    def num_landmarks(self):
        return len(self.landmarks)

    def slam_args(self):
        """Positional + keyword arguments shared by every SLAM function."""
        return dict(world_size=self.world_size, motion_noise=self.motion_noise,
                    measurement_noise=self.measurement_noise, start=self.start)


# --------------------------------------------------------------------------- random world

def random_landmarks(num_landmarks, world_size, rng):
    """Integer landmark positions, uniform over the world (as in the original generator)."""
    return [[float(round(rng.random() * world_size)), float(round(rng.random() * world_size))]
            for _ in range(num_landmarks)]


def simulate_random(N=20, num_landmarks=5, world_size=100.0, measurement_range=50.0,
                    motion_noise=2.0, measurement_noise=2.0, distance=20.0, seed=0):
    """Random landmarks + straight-line motion with random re-heading at the walls."""
    rng = np.random.default_rng(seed)
    start = (world_size / 2.0, world_size / 2.0)
    landmarks = random_landmarks(num_landmarks, world_size, rng)

    while True:
        r = Robot(world_size, measurement_range, motion_noise, measurement_noise,
                  start=start, rng=rng)
        r.landmarks = landmarks
        data, poses = [], [r.pose]
        seen = set()

        heading = rng.random() * 2.0 * np.pi
        dx, dy = np.cos(heading) * distance, np.sin(heading) * distance
        for _ in range(N - 1):
            Z = r.sense()
            seen.update(m[0] for m in Z)
            while not r.move(dx, dy):
                heading = rng.random() * 2.0 * np.pi
                dx, dy = np.cos(heading) * distance, np.sin(heading) * distance
            data.append([Z, [dx, dy]])
            poses.append(r.pose)

        if len(seen) == num_landmarks:
            break

    return Simulation(data, np.array(poses), np.array(landmarks), world_size,
                      measurement_range, motion_noise, measurement_noise, start, "random")


# --------------------------------------------------------------------------- waypoint worlds

def simulate_waypoints(waypoints, landmarks, step=5.0, world_size=100.0, measurement_range=15.0,
                       motion_noise=1.0, measurement_noise=1.0, seed=0, name="waypoints"):
    """Follow ``waypoints`` with a simple path follower.

    At each step the robot commands a move of length ``step`` (or less, to land
    on the waypoint) toward the next waypoint, from its *true* position, like a
    robot tracking a line on the floor. The commanded move is what it reports as
    odometry; the executed move adds noise. The robot therefore stays on the
    track, while an estimator that integrates odometry drifts.
    """
    rng = np.random.default_rng(seed)
    start = tuple(float(v) for v in waypoints[0])
    r = Robot(world_size, measurement_range, motion_noise, measurement_noise, start=start, rng=rng)
    r.landmarks = [list(map(float, l)) for l in landmarks]
    data, poses = [], [r.pose]
    targets = [np.array(w, dtype=float) for w in waypoints[1:]]
    k = 0
    while k < len(targets):
        to_go = targets[k] - r.pose
        dist = np.linalg.norm(to_go)
        if dist < step / 2 and k < len(targets) - 1:
            k += 1          # close enough: aim at the next waypoint
            continue
        if dist < step / 2:
            break           # reached the final waypoint
        move = to_go * min(1.0, step / dist)
        Z = r.sense()
        if not r.move(*move):
            raise RuntimeError("path leaves the world at %s" % (r,))
        data.append([Z, [float(move[0]), float(move[1])]])
        poses.append(r.pose)
    return Simulation(data, np.array(poses), np.array(r.landmarks), world_size, measurement_range,
                      motion_noise, measurement_noise, start, name,
                      extra={"waypoints": np.array(waypoints, dtype=float)})


def corridor_world(length=80.0, width=10.0, spacing=10.0, x0=10.0, y=50.0):
    """Two rows of landmarks along the walls of a straight corridor, and an out-and-back path."""
    xs = np.arange(x0, x0 + length + 1e-9, spacing)
    landmarks = [[x, y - width / 2] for x in xs] + [[x, y + width / 2] for x in xs]
    waypoints = [(x0, y), (x0 + length, y), (x0, y)]
    return waypoints, landmarks


def loop_world(lo=20.0, hi=80.0, spacing=15.0, offset=6.0, laps=1.0):
    """A square circuit with landmarks on its outer side, driven ``laps`` times from ``(lo, lo)``.

    A fractional ``laps`` continues along the circuit past the start, so the
    robot re-observes the first landmarks (loop closure).
    """
    corners = [(lo, lo), (hi, lo), (hi, hi), (lo, hi)]
    landmarks = []
    side = hi - lo
    n = int(round(side / spacing))
    for k in range(4):
        (ax, ay), (bx, by) = corners[k], corners[(k + 1) % 4]
        # outward normal of a counter-clockwise square
        nx, ny = (by - ay) / side, -(bx - ax) / side
        for t in range(n):
            f = t / n
            landmarks.append([ax + f * (bx - ax) + offset * nx, ay + f * (by - ay) + offset * ny])

    full = int(np.floor(laps))
    waypoints = [corners[0]]
    for _ in range(full):
        waypoints += corners[1:] + [corners[0]]
    rest = (laps - full) * 4 * side
    k = 0
    while rest > 1e-9:
        (ax, ay), (bx, by) = corners[k % 4], corners[(k + 1) % 4]
        f = min(1.0, rest / side)
        waypoints.append((ax + f * (bx - ax), ay + f * (by - ay)))
        rest -= f * side
        k += 1
    return waypoints, landmarks
