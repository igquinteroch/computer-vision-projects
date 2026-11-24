"""Regression and consistency tests for Graph SLAM."""

import numpy as np
import pytest

import slam
import world
from data import udacity_test_cases as U

# Outputs of this implementation on the Udacity reference runs (6 decimals).
# The values published with the course differ by up to ~0.4 / ~1.3; see notebook 02.
REGRESSION = {
    1: dict(
        poses=[[50.0, 50.0], [37.972975, 33.651912], [26.185116, 18.155435],
               [13.745157, 2.116262], [28.096774, 16.783059], [42.384321, 30.901793],
               [55.8308, 44.496506], [70.857103, 59.699396], [85.697008, 75.542508],
               [74.011489, 92.434232], [53.544312, 96.453561], [34.525055, 100.079933],
               [48.622994, 83.9533], [60.196908, 68.107405], [73.777601, 52.934713],
               [87.131727, 38.538071], [80.302521, 20.508134], [72.7983, 2.944932],
               [55.245333, 13.255252], [37.416139, 22.317253]],
        landmarks=[[82.955986, 13.538801], [70.494665, 74.141416], [36.739703, 61.28114],
                   [18.698231, 66.059895], [20.634705, 16.874728]],
    ),
    2: dict(
        poses=[[50.0, 50.0], [69.181142, 45.665388], [87.742709, 39.70348],
               [76.269835, 56.310672], [64.316936, 72.176188], [52.257141, 88.153615],
               [44.058877, 69.400919], [37.001521, 49.917658], [30.924074, 30.955025],
               [23.507918, 11.419308], [34.179947, 27.132997], [44.154893, 43.845854],
               [54.805819, 60.920473], [65.697876, 78.545635], [77.468115, 95.626449],
               [96.801604, 98.821076], [75.956632, 99.971263], [70.199508, 81.180875],
               [64.053677, 61.722989], [58.107034, 42.627507]],
        landmarks=[[76.778608, 42.887249], [85.064592, 77.438265], [13.548356, 95.652341],
                   [59.448575, 39.595329], [69.263241, 94.239943]],
    ),
}

CASES = {1: U.TEST_DATA_1, 2: U.TEST_DATA_2}
P = U.PARAMS


@pytest.mark.parametrize("case", [1, 2])
def test_udacity_cases_regression(case):
    mu = slam.slam(CASES[case], P["N"], P["num_landmarks"], P["world_size"],
                   P["motion_noise"], P["measurement_noise"])
    poses, landmarks = slam.unpack(mu, P["N"], P["num_landmarks"])
    np.testing.assert_allclose(poses, REGRESSION[case]["poses"], atol=1e-5)
    np.testing.assert_allclose(landmarks, REGRESSION[case]["landmarks"], atol=1e-5)


def _udacity_runs():
    for case in (1, 2):
        yield "udacity-%d" % case, CASES[case], P["num_landmarks"], dict(
            world_size=P["world_size"], motion_noise=P["motion_noise"],
            measurement_noise=P["measurement_noise"])


def _simulated_runs():
    sim = world.simulate_random(N=60, num_landmarks=8, seed=3)
    yield "random", sim.data, sim.num_landmarks, sim.slam_args()
    sim = world.simulate_waypoints(*world.loop_world(laps=1.25), seed=1, name="loop")
    yield "loop", sim.data, sim.num_landmarks, sim.slam_args()


RUNS = list(_udacity_runs()) + list(_simulated_runs())


@pytest.mark.parametrize("name,data,num_landmarks,kwargs", RUNS, ids=[r[0] for r in RUNS])
def test_online_matches_full(name, data, num_landmarks, kwargs):
    N = len(data) + 1
    mu = slam.slam(data, N, num_landmarks, **kwargs)
    poses, landmarks = slam.unpack(mu, N, num_landmarks)
    pose, online_landmarks = slam.online_slam(data, num_landmarks, **kwargs)
    np.testing.assert_allclose(pose, poses[-1], atol=1e-8)
    np.testing.assert_allclose(online_landmarks, landmarks, atol=1e-8)
