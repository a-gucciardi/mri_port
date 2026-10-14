import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


def normalize_pair(pair: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(pair, dtype=np.float32)
    for index, channel in enumerate(pair):
        foreground = channel > 0
        low, high = np.percentile(channel[foreground], [1, 99])
        if not np.isfinite([low, high]).all() or high <= low:
            raise ValueError("MRI channel has no usable intensity range")
        normalized[index, foreground] = np.clip((channel[foreground] - low) / (high - low), 0, 1)
    return normalized


def resize_slice(pair: np.ndarray, z: int, image_size: int) -> Tensor:
    image = torch.from_numpy(np.ascontiguousarray(pair[:, :, :, z]))
    return F.interpolate(
        image.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False
    ).squeeze(0)


def slice_indices(depth: int, step: int) -> list[int]:
    if depth < 4 or step < 1:
        raise ValueError("Depth must be >=4 and slice step >=1")
    return list(range(int(depth * 0.2), int(depth * 0.8), step))
