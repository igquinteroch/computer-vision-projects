"""Evaluation: predictions in original-image pixels, point errors and NME."""
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pandas as pd
import torch

import config
from data_load import denormalize_keypoints, read_rgb


def _eval_input(image: np.ndarray):
    """Rescale(250) + CenterCrop(224) + grayscale, as at training time.

    Returns the network input and the affine map back to the original image:
    ``original = (net_frame + offset) / scale``.
    """
    h, w = image.shape[:2]
    scale = config.RESCALE_SIZE / min(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    top, left = (new_h - config.IMAGE_SIZE) // 2, (new_w - config.IMAGE_SIZE) // 2
    crop = resized[top: top + config.IMAGE_SIZE, left: left + config.IMAGE_SIZE]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) / 255.0
    sx, sy = new_w / w, new_h / h
    return gray.astype(np.float32), np.array([left, top]), np.array([sx, sy])


@torch.no_grad()
def predict_frame(net, frame: pd.DataFrame, root_dir: Path, batch_size: int = 64) -> np.ndarray:
    """Predict keypoints for every row of a keypoints frame.

    Returns an array ``(N, 68, 2)`` in the pixel coordinates of the original images.
    """
    device = next(net.parameters()).device
    net.eval()
    preds = []
    for start in range(0, len(frame), batch_size):
        names = frame.iloc[start: start + batch_size, 0]
        inputs, offsets, scales = zip(*(_eval_input(read_rgb(Path(root_dir) / n)) for n in names))
        x = torch.from_numpy(np.stack(inputs)[:, None]).to(device)
        out = net(x).cpu().numpy().reshape(len(names), -1, 2)
        pts = denormalize_keypoints(out)
        preds.append((pts + np.array(offsets)[:, None]) / np.array(scales)[:, None])
    return np.concatenate(preds)


def _frame_maps(frame: pd.DataFrame, root_dir: Path):
    """Per-image (offset, scale) of the eval preprocessing, without running the net."""
    maps = []
    for n in frame.iloc[:, 0]:
        _, offset, scale = _eval_input(read_rgb(Path(root_dir) / n))
        maps.append((offset, scale))
    return maps


def fit_mean_shape(frame: pd.DataFrame, root_dir: Path) -> np.ndarray:
    """Mean keypoints ``(68, 2)`` in the 224×224 network frame."""
    gt = ground_truth(frame)
    return np.mean([g * s - o for g, (o, s) in zip(gt, _frame_maps(frame, root_dir))], axis=0)


def predict_mean_shape(mean_shape: np.ndarray, frame: pd.DataFrame, root_dir: Path) -> np.ndarray:
    """Baseline: place the same mean shape in every image's 224 crop, ``(N, 68, 2)``."""
    return np.stack([(mean_shape + o) / s for o, s in _frame_maps(frame, root_dir)])


def ground_truth(frame: pd.DataFrame) -> np.ndarray:
    """Keypoints of a frame as ``(N, 68, 2)``."""
    return frame.iloc[:, 1:].to_numpy(dtype=float).reshape(len(frame), -1, 2)


def point_errors(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Euclidean error per keypoint, ``(N, 68)``, in pixels."""
    return np.linalg.norm(pred - gt, axis=-1)


def interocular(gt: np.ndarray) -> np.ndarray:
    """Distance between the outer eye corners, ``(N,)``."""
    return np.linalg.norm(gt[:, config.LEFT_EYE_OUTER] - gt[:, config.RIGHT_EYE_OUTER], axis=-1)


def nme(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Normalized mean error per image: mean point error / inter-ocular distance."""
    return point_errors(pred, gt).mean(axis=1) / interocular(gt)


def summary(pred: np.ndarray, gt: np.ndarray, failure_threshold: float = 0.10) -> Dict[str, float]:
    """Headline metrics for a set of predictions."""
    err, n = point_errors(pred, gt), nme(pred, gt)
    return {
        "mean_px": err.mean(),
        "median_px": np.median(err.mean(axis=1)),
        "nme_%": 100 * n.mean(),
        "median_nme_%": 100 * np.median(n),
        f"failure_rate_%(nme>{failure_threshold:.2f})": 100 * (n > failure_threshold).mean(),
    }


def region_nme(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Mean NME (in %) restricted to each facial region."""
    err = point_errors(pred, gt) / interocular(gt)[:, None]
    return {name: 100 * err[:, list(idx)].mean() for name, idx in config.REGIONS.items()}


def ced(nmes: np.ndarray, max_nme: float = 0.3, steps: int = 301):
    """Cumulative error distribution: share of images with NME below each threshold."""
    thresholds = np.linspace(0, max_nme, steps)
    return thresholds, (nmes[:, None] <= thresholds[None]).mean(axis=0)
