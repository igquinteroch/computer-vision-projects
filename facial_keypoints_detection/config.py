"""Project-wide paths, device selection and data constants.

Override the dataset location with the ``FKD_DATA_ROOT`` environment variable.
It must point to the directory that contains ``training/``, ``test/`` and the
two ``*_frames_keypoints.csv`` files.
"""
import os
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_ROOT = Path(os.environ.get("FKD_DATA_ROOT", PROJECT_ROOT / "data"))
TRAIN_CSV = DATA_ROOT / "training_frames_keypoints.csv"
TRAIN_DIR = DATA_ROOT / "training"
TEST_CSV = DATA_ROOT / "test_frames_keypoints.csv"
TEST_DIR = DATA_ROOT / "test"
DATA_URL = ("https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/"
            "5aea1b91_train-test-data/train-test-data.zip")

MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"
IMAGES_DIR = PROJECT_ROOT / "images"
CASCADE_DIR = PROJECT_ROOT / "detector_architectures"
FACE_CASCADE = CASCADE_DIR / "haarcascade_frontalface_default.xml"

# Weights submitted to the Nanodegree (10 epochs on the full training CSV).
# Kept as a reference point for the retrained model.
V0_WEIGHTS = MODELS_DIR / "keypoints_cnn_v0.pt"
V0_INPUT_SCALE = 1 / 255  # v0 was trained on [0, 1/255] inputs (see data_load.read_rgb)
V1_WEIGHTS = MODELS_DIR / "keypoints_cnn_v1.pt"   # v0 architecture, fixed input scale
V2_WEIGHTS = MODELS_DIR / "keypoints_cnn_v2.pt"   # compact head + BatchNorm, augmentation, Wing loss
# Model published in the release, chosen on the validation split (notebook 02).
FINAL_WEIGHTS = V2_WEIGHTS
LOGS_DIR = PROJECT_ROOT / "logs"

# Input and keypoint normalization (must match the trained weights).
IMAGE_SIZE = 224
RESCALE_SIZE = 250        # shorter side before the 224×224 crop
KEYPOINT_MEAN = 100.0     # pixels; keypoints are fed to the net as (pts - mean) / std
KEYPOINT_STD = 50.0
NUM_KEYPOINTS = 68

# 0-indexed landmark ranges of the 68-point (iBUG / Multi-PIE) layout.
REGIONS = {
    "jaw": range(0, 17),
    "eyebrows": range(17, 27),
    "nose": range(27, 36),
    "eyes": range(36, 48),
    "mouth": range(48, 68),
}
# Outer eye corners, used to normalize errors (inter-ocular distance).
LEFT_EYE_OUTER, RIGHT_EYE_OUTER = 36, 45


def _flip_permutation():
    """Index map that relabels the 68 points after a horizontal flip."""
    pairs = ([(i, 16 - i) for i in range(8)]                        # jaw
             + [(17 + i, 26 - i) for i in range(5)]                 # eyebrows
             + [(31, 35), (32, 34)]                                 # nostrils
             + [(36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46)]  # eyes
             + [(48, 54), (49, 53), (50, 52), (55, 59), (56, 58)]   # outer lip
             + [(60, 64), (61, 63), (65, 67)])                      # inner lip
    perm = list(range(NUM_KEYPOINTS))
    for a, b in pairs:
        perm[a], perm[b] = b, a
    return perm


FLIP_PERMUTATION = _flip_permutation()

# Train / validation / holdout split of the training CSV, by person.
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
SPLIT_SEED = 42


def get_device() -> torch.device:
    """Pick the best available accelerator: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
