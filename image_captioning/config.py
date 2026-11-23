"""Project-wide paths, device selection and final hyperparameters.

Override the COCO location with the ``COCO_ROOT`` environment variable. It must
point to the directory that *contains* ``cocoapi/`` (with ``annotations/`` and
``images/`` inside it).
"""
import os
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent

# Directory containing cocoapi/annotations and cocoapi/images.
COCO_ROOT = Path(os.environ.get("COCO_ROOT", PROJECT_ROOT / "data"))

MODELS_DIR = PROJECT_ROOT / "models"
VOCAB_FILE = PROJECT_ROOT / "vocab.pkl"
ASSETS_DIR = PROJECT_ROOT / "assets"
TRAINING_LOG = PROJECT_ROOT / "training_log.txt"

# Final model configuration (must match the released weights).
EMBED_SIZE = 512
HIDDEN_SIZE = 512
VOCAB_THRESHOLD = 4
ENCODER_WEIGHTS = MODELS_DIR / "encoder-3.pkl"
DECODER_WEIGHTS = MODELS_DIR / "decoder-3.pkl"


def get_device() -> torch.device:
    """Pick the best available accelerator: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
