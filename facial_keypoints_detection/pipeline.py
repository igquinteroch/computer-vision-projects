"""End-to-end inference on arbitrary photos: Haar face detection -> CNN keypoints.

The dataset faces are loosely cropped (forehead, ears and chin are all visible),
while a Haar box is tight around the eyes and mouth. Feeding the tight box to the
network shows it a much bigger face than it was trained on, so ``crop_face`` adds
a margin (``pad``) around each detection before resizing to 224×224.
"""
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

import config
from data_load import denormalize_keypoints

DEFAULT_PAD = 0.39  # margin per side, as a fraction of the Haar box (calibrated on train, notebook 04)


def load_rgb(path) -> np.ndarray:
    """Read an image file as an RGB ``uint8`` array."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def detect_faces(image: np.ndarray, scale_factor: float = 1.2, min_neighbors: int = 2,
                 min_size: float = 0.0, cascade_path: Path = config.FACE_CASCADE) -> np.ndarray:
    """Haar-cascade face boxes ``(x, y, w, h)`` for an RGB image.

    Args:
        min_size: Smallest box side to keep, as a fraction of the image's
            shorter side. Useful to drop tiny false positives in large photos.
    """
    cascade = cv2.CascadeClassifier(str(cascade_path))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    side = int(min_size * min(gray.shape))
    faces = cascade.detectMultiScale(gray, scale_factor, min_neighbors, minSize=(side, side))
    return np.asarray(faces).reshape(-1, 4)


def crop_face(image: np.ndarray, box, pad: float = DEFAULT_PAD) -> Tuple[np.ndarray, np.ndarray, float]:
    """Square crop around a box, enlarged by ``pad`` on each side, resized to 224.

    Areas outside the image are filled by replicating the border.

    Returns:
        The 224×224 RGB crop, its top-left corner in the image and the
        image-pixels-per-crop-pixel scale, so that
        ``image_xy = crop_xy * scale + origin``.
    """
    x, y, w, h = box
    side = max(w, h) * (1 + 2 * pad)
    cx, cy = x + w / 2, y + h / 2
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    side = int(round(side))
    H, W = image.shape[:2]
    padded = cv2.copyMakeBorder(image, max(0, -y0), max(0, y0 + side - H),
                                max(0, -x0), max(0, x0 + side - W), cv2.BORDER_REPLICATE)
    px, py = x0 + max(0, -x0), y0 + max(0, -y0)
    crop = padded[py: py + side, px: px + side]
    crop = cv2.resize(crop, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    return crop, np.array([x0, y0], dtype=float), side / config.IMAGE_SIZE


@torch.no_grad()
def predict_keypoints(net, image: np.ndarray, boxes, pad: float = DEFAULT_PAD) -> List[np.ndarray]:
    """Keypoints ``(68, 2)`` in image coordinates for each face box."""
    device = next(net.parameters()).device
    results = []
    for box in boxes:
        crop, origin, scale = crop_face(image, box, pad)
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        x = torch.from_numpy(gray)[None, None].to(device)
        pts = denormalize_keypoints(net(x).cpu().numpy().reshape(-1, 2))
        results.append(pts * scale + origin)
    return results


def overlay_sunglasses(image: np.ndarray, pts: np.ndarray, sunglasses_rgba: np.ndarray) -> np.ndarray:
    """Alpha-blend a sunglasses PNG over the eyes, placed from the keypoints.

    Width spans the outer eyebrow corners (17 → 26); height is the nose bridge
    length (27 → 34); the top-left corner sits on the left eyebrow (17).
    The overlay is axis-aligned: head roll is not compensated.
    """
    out = image.copy()
    x, y = int(pts[17, 0]), int(pts[17, 1])
    w = int(abs(pts[26, 0] - pts[17, 0]))
    h = int(abs(pts[34, 1] - pts[27, 1]))
    if w < 2 or h < 2:
        return out
    glasses = cv2.resize(sunglasses_rgba, (w, h), interpolation=cv2.INTER_CUBIC)
    # clip the overlay to the image bounds
    H, W = out.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return out
    g = glasses[y0 - y: y1 - y, x0 - x: x1 - x]
    alpha = g[:, :, 3:4].astype(float) / 255.0
    rgb = g[:, :, 2::-1].astype(float)            # BGRA -> RGB
    region = out[y0:y1, x0:x1].astype(float)
    out[y0:y1, x0:x1] = (alpha * rgb + (1 - alpha) * region).astype(out.dtype)
    return out
