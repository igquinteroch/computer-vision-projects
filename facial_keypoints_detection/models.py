"""CNNs that regress the 68 facial keypoints from a 224×224 grayscale face crop.

``Net`` is the architecture submitted to the Nanodegree (v0, retrained as v1).
``NetV2`` adds a fourth conv block, BatchNorm and adaptive pooling, which shrinks
the first fully connected layer from 44.3 M to 4.7 M weights.
"""
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class Net(nn.Module):
    """Three conv/ReLU/max-pool blocks followed by two fully connected layers.

    Input: ``(batch, 1, 224, 224)`` grayscale images in ``[0, 1]``.
    Output: ``(batch, 136)`` normalized keypoints, ``(x0, y0, x1, y1, ...)``;
    pixel coordinates are ``output * KEYPOINT_STD + KEYPOINT_MEAN``.

    Spatial size: 224 → conv5 220 → pool 110 → conv3 108 → pool 54 → conv3 52
    → pool 26, so ``fc1`` sees ``128 × 26 × 26 = 86,528`` features.

    Args:
        dropout: Dropout probability before the output layer.
        input_scale: Multiplier applied to the input. 1 for models trained on
            [0, 1] images; ``1/255`` for v0, which was trained on [0, 1/255]
            because of a double normalization (see ``data_load.read_rgb``).
            It is not stored in the ``state_dict``.
    """

    def __init__(self, dropout: float = 0.3, input_scale: float = 1.0):
        super().__init__()
        self.input_scale = input_scale
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128 * 26 * 26, 512)
        self.fc2 = nn.Linear(512, config.NUM_KEYPOINTS * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.input_scale
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.flatten(1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


class NetV2(nn.Module):
    """Four conv/BatchNorm/ReLU/max-pool blocks, adaptive pooling, two FC layers.

    Spatial size: 224 → 220 → 110 → 108 → 54 → 52 → 26 → 24 → 12, then
    adaptive average pooling to 6×6, so ``fc1`` sees ``256 × 6 × 6 = 9,216``
    features instead of 86,528.
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        chans = [(1, 32, 5), (32, 64, 3), (64, 128, 3), (128, 256, 3)]
        self.features = nn.Sequential(*[
            layer for c_in, c_out, k in chans for layer in (
                nn.Conv2d(c_in, c_out, k, bias=False), nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        ])
        self.pool = nn.AdaptiveAvgPool2d(6)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, config.NUM_KEYPOINTS * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.features(x)).flatten(1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


ARCHITECTURES = {"net": Net, "net_v2": NetV2}


def load_trained(weights: Union[str, Path] = config.V0_WEIGHTS,
                 device: Optional[torch.device] = None) -> nn.Module:
    """Load a saved ``state_dict`` into the matching architecture, in eval mode.

    The architecture is inferred from the weights (``NetV2`` has a ``features``
    block). The v0 weights automatically get ``input_scale = 1/255``.
    """
    device = device or config.DEVICE
    state = torch.load(weights, map_location="cpu", weights_only=True)
    if any(k.startswith("features.") for k in state):
        net = NetV2()
    else:
        v0 = Path(weights).resolve() == config.V0_WEIGHTS.resolve()
        net = Net(input_scale=config.V0_INPUT_SCALE if v0 else 1.0)
    net.load_state_dict(state)
    return net.to(device).eval()
