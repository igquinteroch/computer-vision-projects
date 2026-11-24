"""A point robot in a 2D square world that moves with noise and senses landmarks.

Motion and measurements both carry *uniform* noise: each component gets an
independent sample from ``[-1, 1) * noise``. The sensor reports the x- and
y-offset from the robot to every landmark inside a square window of half-width
``measurement_range`` (both ``|dx|`` and ``|dy|`` must be within range).
"""

import numpy as np


class Robot:
    """Robot with noisy odometry and a noisy landmark sensor.

    Args:
        world_size: side of the square world; positions live in ``[0, world_size]``.
        measurement_range: half-width of the square sensing window. ``-1`` means
            every landmark is always visible.
        motion_noise: amplitude of the uniform noise added to each move component.
        measurement_noise: amplitude of the uniform noise added to each measurement.
        start: initial ``(x, y)``; defaults to the center of the world.
        rng: ``np.random.Generator`` used for all noise, so runs are reproducible.
    """

    def __init__(self, world_size=100.0, measurement_range=30.0, motion_noise=1.0,
                 measurement_noise=1.0, start=None, rng=None):
        self.world_size = world_size
        self.measurement_range = measurement_range
        self.motion_noise = motion_noise
        self.measurement_noise = measurement_noise
        self.rng = rng if rng is not None else np.random.default_rng()
        if start is None:
            start = (world_size / 2.0, world_size / 2.0)
        self.x, self.y = float(start[0]), float(start[1])
        self.landmarks = []

    @property
    def num_landmarks(self):
        return len(self.landmarks)

    def rand(self):
        """Uniform sample in ``[-1, 1)``."""
        return self.rng.uniform(-1.0, 1.0)

    def move(self, dx, dy):
        """Try to move by ``(dx, dy)`` plus noise.

        Returns ``False`` (and leaves the robot in place) if the noisy target
        falls outside the world.
        """
        x = self.x + dx + self.rand() * self.motion_noise
        y = self.y + dy + self.rand() * self.motion_noise
        if x < 0.0 or x > self.world_size or y < 0.0 or y > self.world_size:
            return False
        self.x, self.y = x, y
        return True

    def sense(self):
        """Return ``[[j, dx, dy], ...]`` for every landmark ``j`` inside the sensing window.

        ``dx, dy`` are *landmark minus robot*, with noise added before the range
        check. Returns ``[]`` if nothing is visible.
        """
        measurements = []
        for j, (lx, ly) in enumerate(self.landmarks):
            dx = lx - self.x + self.rand() * self.measurement_noise
            dy = ly - self.y + self.rand() * self.measurement_noise
            if self.measurement_range == -1 or (
                    abs(dx) <= self.measurement_range and abs(dy) <= self.measurement_range):
                measurements.append([j, dx, dy])
        return measurements

    @property
    def pose(self):
        return np.array([self.x, self.y])

    def __repr__(self):
        return "Robot(x=%.3f, y=%.3f)" % (self.x, self.y)
