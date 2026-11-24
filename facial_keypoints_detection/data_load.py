"""Facial keypoints dataset, transforms and data helpers.

``FacialKeypointsDataset`` and the ``Normalize`` / ``Rescale`` / ``RandomCrop`` /
``ToTensor`` transforms come from the Udacity starter code (itself based on the
PyTorch data loading tutorial). ``CenterCrop``, the in-memory cache, the download
helper, ``read_rgb``, the identity-grouped split and ``denormalize_keypoints`` were added for
this project.
"""
import io
import re
import zipfile
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from torch.utils.data import Dataset

import config


def read_rgb(path) -> np.ndarray:
    """Read an image as RGB ``uint8``.

    The dataset files are RGBA PNGs with a ``.jpg`` extension. The starter code
    read them with ``matplotlib.image.imread``, which returns float32 in [0, 1]
    for PNGs; ``Normalize`` then divided by 255 a second time, so the submitted
    model (v0) was trained on inputs in [0, 1/255]. Reading as ``uint8`` makes
    ``Normalize`` produce the intended [0, 1] range.
    """
    return np.asarray(Image.open(path).convert("RGB"))


# --------------------------------------------------------------------------- data


def download_data(root: Path = config.DATA_ROOT) -> Path:
    """Download and unzip the dataset into ``root`` (no-op if already there)."""
    if (root / "training_frames_keypoints.csv").exists():
        return root
    root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset (~340 MB) to {root} ...")
    response = requests.get(config.DATA_URL, timeout=600)
    response.raise_for_status()
    zipfile.ZipFile(io.BytesIO(response.content)).extractall(root)
    return root


def identity_of(filename: str) -> str:
    """``'Luis_Fonsi_21.jpg'`` -> ``'Luis_Fonsi'``.

    File names are ``<person>_<video><frame>.jpg``; the two trailing digits are
    the clip index and the frame index within the clip.
    """
    return re.sub(r"_\d\d\.jpg$", "", filename)


def split_by_identity(frame: pd.DataFrame, fractions: Sequence[float],
                      seed: int = 42) -> Tuple[pd.DataFrame, ...]:
    """Split a keypoints frame into disjoint subsets of *people*.

    Every frame of a person ends up in the same subset, so a person seen in
    training never appears in validation. ``fractions`` are the shares of
    identities for each subset and must sum to 1.
    """
    assert abs(sum(fractions) - 1) < 1e-9
    ids = np.array(sorted(frame.iloc[:, 0].map(identity_of).unique()))
    np.random.default_rng(seed).shuffle(ids)
    bounds = np.cumsum([0] + [round(f * len(ids)) for f in fractions[:-1]] + [len(ids)])
    person = frame.iloc[:, 0].map(identity_of)
    return tuple(frame[person.isin(set(ids[a:b]))].reset_index(drop=True)
                 for a, b in zip(bounds[:-1], bounds[1:]))


class FacialKeypointsDataset(Dataset):
    """Face images with their 68 (x, y) keypoints.

    Args:
        csv_file: Path to a keypoints CSV, or an already loaded ``DataFrame``
            (first column = file name, then 136 coordinates).
        root_dir: Directory with the images.
        transform: Optional callable applied to each ``{'image', 'keypoints'}`` sample.
        cache: Keep decoded images in memory after the first read.
    """

    def __init__(self, csv_file, root_dir, transform=None, cache: bool = False):
        self.key_pts_frame = (csv_file if isinstance(csv_file, pd.DataFrame)
                              else pd.read_csv(csv_file))
        self.root_dir = Path(root_dir)
        self.transform = transform
        self._cache = {} if cache else None

    def __len__(self):
        return len(self.key_pts_frame)

    def _read(self, idx):
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]
        image = read_rgb(self.root_dir / self.key_pts_frame.iloc[idx, 0])
        if self._cache is not None:
            self._cache[idx] = image
        return image

    def __getitem__(self, idx):
        image = self._read(idx)
        key_pts = self.key_pts_frame.iloc[idx, 1:].to_numpy().astype("float").reshape(-1, 2)
        sample = {"image": image, "keypoints": key_pts}
        if self.transform:
            sample = self.transform(sample)
        return sample


# ---------------------------------------------------------------------- transforms


class Normalize:
    """Convert a color image to grayscale in [0, 1] and normalize the keypoints."""

    def __call__(self, sample):
        image, key_pts = sample["image"], sample["keypoints"]
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) / 255.0
        key_pts_copy = (key_pts - config.KEYPOINT_MEAN) / config.KEYPOINT_STD
        return {"image": image_copy, "keypoints": key_pts_copy}


class Rescale:
    """Rescale the image (and keypoints) to a given size.

    Args:
        output_size: If a tuple, the exact ``(h, w)``. If an int, the shorter
            image side is matched to it and the aspect ratio is kept.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample["image"], sample["keypoints"]
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(image, (new_w, new_h))
        key_pts = key_pts * [new_w / w, new_h / h]
        return {"image": img, "keypoints": key_pts}


class RandomCrop:
    """Crop the image (and shift the keypoints) at a random position."""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample["image"], sample["keypoints"]
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        image = image[top: top + new_h, left: left + new_w]
        return {"image": image, "keypoints": key_pts - [left, top]}


class CenterCrop(RandomCrop):
    """Deterministic center crop, used for validation and test."""

    def __call__(self, sample):
        image, key_pts = sample["image"], sample["keypoints"]
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top, left = (h - new_h) // 2, (w - new_w) // 2
        image = image[top: top + new_h, left: left + new_w]
        return {"image": image, "keypoints": key_pts - [left, top]}


class RandomHorizontalFlip:
    """Mirror the image with probability ``p`` and relabel the keypoints.

    After a flip, the point that was the left eye corner sits where the right
    one should be, so the 68 points are reordered with ``config.FLIP_PERMUTATION``.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample):
        image, key_pts = sample["image"], sample["keypoints"]
        if np.random.rand() >= self.p:
            return sample
        w = image.shape[1]
        image = np.ascontiguousarray(image[:, ::-1])
        key_pts = key_pts.copy()
        key_pts[:, 0] = (w - 1) - key_pts[:, 0]
        return {"image": image, "keypoints": key_pts[config.FLIP_PERMUTATION]}


class RandomRotateScale:
    """Small random rotation and zoom about the image center.

    Args:
        max_degrees: Rotation angle drawn uniformly from ``[-max_degrees, max_degrees]``.
        scale_range: Zoom factor drawn uniformly from this interval.
    """

    def __init__(self, max_degrees: float = 15.0, scale_range=(0.9, 1.1)):
        self.max_degrees = max_degrees
        self.scale_range = scale_range

    def __call__(self, sample):
        image, key_pts = sample["image"], sample["keypoints"]
        h, w = image.shape[:2]
        angle = np.random.uniform(-self.max_degrees, self.max_degrees)
        scale = np.random.uniform(*self.scale_range)
        m = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, scale)
        image = cv2.warpAffine(image, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        key_pts = key_pts @ m[:, :2].T + m[:, 2]
        return {"image": image, "keypoints": key_pts}


class ToTensor:
    """Convert the numpy image (H x W [x C]) and keypoints to tensors (C x H x W)."""

    def __call__(self, sample):
        image, key_pts = sample["image"], sample["keypoints"]
        if image.ndim == 2:
            image = image[:, :, None]
        image = image.transpose((2, 0, 1))
        return {"image": torch.from_numpy(image).float(),
                "keypoints": torch.from_numpy(key_pts).float()}


def denormalize_keypoints(pts):
    """Network-space keypoints -> pixel coordinates in the 224×224 input frame."""
    return pts * config.KEYPOINT_STD + config.KEYPOINT_MEAN
