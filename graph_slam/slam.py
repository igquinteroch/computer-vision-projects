"""Graph SLAM in 2D with known data association.

State layout (interleaved): ``[x0, y0, x1, y1, ..., x_{N-1}, y_{N-1}, Lx0, Ly0, ...]``.
Pose ``i`` lives at indices ``2i, 2i+1`` and landmark ``j`` at ``2(N+j), 2(N+j)+1``.

Every motion or measurement says "b - a = d" for two state entries. Each one adds
a small block to the information matrix ``omega`` and vector ``xi``, and the
best estimate is the solution of ``omega @ mu = xi`` (that is, ``mu = omega^-1 xi``).

``data`` has one entry per time step, ``[measurements, [dx, dy]]``: the robot
senses ``[[j, dx, dy], ...]`` at pose ``i`` and then moves by ``[dx, dy]`` to
pose ``i + 1``. A run with ``N`` poses therefore has ``N - 1`` steps.
"""

import numpy as np


def pose_index(i):
    return 2 * i


def landmark_index(j, N):
    return 2 * (N + j)


def _start(world_size, start):
    if start is None:
        return (world_size / 2.0, world_size / 2.0)
    return start


def initialize_constraints(N, num_landmarks, world_size, start=None):
    """Zero ``omega``/``xi`` with the first pose anchored at ``start`` (default: world center)."""
    size = 2 * (N + num_landmarks)
    omega = np.zeros((size, size))
    xi = np.zeros((size, 1))
    x0, y0 = _start(world_size, start)
    omega[0, 0] = omega[1, 1] = 1.0
    xi[0], xi[1] = x0, y0
    return omega, xi


def add_constraint(omega, xi, a, b, d, w):
    """Add "b - a = d" with weight ``w`` for x (indices a, b) and y (a+1, b+1)."""
    for k in range(2):
        ak, bk = a + k, b + k
        omega[ak, ak] += w
        omega[bk, bk] += w
        omega[ak, bk] -= w
        omega[bk, ak] -= w
        xi[ak] -= d[k] * w
        xi[bk] += d[k] * w


def build_system(data, N, num_landmarks, world_size, motion_noise, measurement_noise,
                 start=None):
    """Fill ``omega`` and ``xi`` with every motion and measurement constraint in ``data``."""
    omega, xi = initialize_constraints(N, num_landmarks, world_size, start)
    w_meas = 1.0 / measurement_noise
    w_motion = 1.0 / motion_noise
    for i, (measurements, motion) in enumerate(data):
        for j, dx, dy in measurements:
            add_constraint(omega, xi, pose_index(i), landmark_index(j, N), (dx, dy), w_meas)
        add_constraint(omega, xi, pose_index(i), pose_index(i + 1), motion, w_motion)
    return omega, xi


def slam(data, N, num_landmarks, world_size, motion_noise, measurement_noise, start=None):
    """Full Graph SLAM: return ``mu`` (column vector) with all poses and all landmarks.

    Solves ``omega @ mu = xi`` with ``np.linalg.solve``, which gives the same
    result as ``inv(omega) @ xi`` without forming the inverse. Every landmark
    must be observed at least once, otherwise ``omega`` is singular.
    """
    omega, xi = build_system(data, N, num_landmarks, world_size, motion_noise,
                             measurement_noise, start)
    return np.linalg.solve(omega, xi)


def unpack(mu, N, num_landmarks):
    """Split ``mu`` into ``poses`` (N, 2) and ``landmarks`` (num_landmarks, 2)."""
    mu = np.asarray(mu).ravel()
    poses = mu[:2 * N].reshape(N, 2)
    landmarks = mu[2 * N:2 * (N + num_landmarks)].reshape(num_landmarks, 2)
    return poses, landmarks


def slam_observed(data, num_landmarks, world_size, motion_noise, measurement_noise,
                  start=None):
    """Full SLAM on a prefix of a run, where some landmarks may not be seen yet.

    Unobserved landmarks are dropped from the system before solving and come
    back as ``NaN``. Returns ``(poses, landmarks)`` with ``len(data) + 1`` poses.
    """
    N = len(data) + 1
    omega, xi = build_system(data, N, num_landmarks, world_size, motion_noise,
                             measurement_noise, start)
    seen = sorted({m[0] for measurements, _ in data for m in measurements})
    keep = list(range(2 * N))
    for j in seen:
        keep += [landmark_index(j, N), landmark_index(j, N) + 1]
    mu = np.full(2 * (N + num_landmarks), np.nan)
    mu[keep] = np.linalg.solve(omega[np.ix_(keep, keep)], xi[keep]).ravel()
    return unpack(mu, N, num_landmarks)


def online_slam(data, num_landmarks, world_size, motion_noise, measurement_noise,
                start=None, return_system=False):
    """Online Graph SLAM: keep only the *current* pose and the landmarks in ``omega``.

    State layout: ``[x_t, y_t, Lx0, Ly0, ...]`` (size ``2 + 2 * num_landmarks``).
    At each step the robot's measurements are added to the current pose, the
    motion adds a new pose, and the old pose is marginalized out with the Schur
    complement::

        omega' = B - A^T omega_old^-1 A
        xi'    = xi_B - A^T omega_old^-1 xi_old

    where ``omega_old`` is the old-pose block, ``A`` its coupling to the rest and
    ``B`` the rest. For this linear problem the result is *exactly* the final
    pose and landmarks of full SLAM, at constant memory per step. Landmarks that
    were never observed come back as ``NaN``.

    Returns ``(final_pose, landmarks)``; with ``return_system=True`` also
    returns the final ``(omega, xi)``.
    """
    L2 = 2 * num_landmarks
    omega = np.zeros((2 + L2, 2 + L2))
    xi = np.zeros((2 + L2, 1))
    x0, y0 = _start(world_size, start)
    omega[0, 0] = omega[1, 1] = 1.0
    xi[0], xi[1] = x0, y0
    w_meas = 1.0 / measurement_noise
    w_motion = 1.0 / motion_noise

    for measurements, motion in data:
        for j, dx, dy in measurements:
            add_constraint(omega, xi, 0, 2 + 2 * j, (dx, dy), w_meas)

        # Expand to [old pose, new pose, landmarks] and link old -> new.
        big = np.zeros((4 + L2, 4 + L2))
        big_xi = np.zeros((4 + L2, 1))
        old = [0, 1] + list(range(4, 4 + L2))
        big[np.ix_(old, old)] = omega
        big_xi[old] = xi
        add_constraint(big, big_xi, 0, 2, motion, w_motion)

        # Marginalize the old pose (first two rows/cols).
        A = big[:2, 2:]
        B = big[2:, 2:]
        omega_old_inv = np.linalg.inv(big[:2, :2])
        omega = B - A.T @ omega_old_inv @ A
        xi = big_xi[2:] - A.T @ omega_old_inv @ big_xi[:2]

    # Landmarks never observed have empty rows; leave them out of the solve (NaN).
    keep = np.flatnonzero(np.abs(omega).sum(axis=1) > 0)
    mu = np.full(2 + L2, np.nan)
    mu[keep] = np.linalg.solve(omega[np.ix_(keep, keep)], xi[keep]).ravel()
    pose, landmarks = mu[:2], mu[2:].reshape(num_landmarks, 2)
    if return_system:
        return pose, landmarks, (omega, xi)
    return pose, landmarks
