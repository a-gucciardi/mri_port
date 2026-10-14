from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from cvproj.data.prepare import Scan, load_pair, resolve_scan
from cvproj.data.transforms import normalize_pair, resize_slice, slice_indices


class MRISlices(Dataset[tuple[Tensor, str]]):
    def __init__(self, root: Path, scans: list[Scan], image_size: int, step: int) -> None:
        self.root = root
        self.scans = scans
        self.image_size = image_size
        self.indices = [
            (i, z) for i, scan in enumerate(scans) for z in slice_indices(scan.shape[2], step)
        ]
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
        if not self.indices:
            raise ValueError("No MRI slices selected")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Tensor, str]:
        scan_index, z = self.indices[index]
        scan = self.scans[scan_index]
        if scan_index not in self.cache:
            pair = load_pair(
                resolve_scan(self.root, scan.t1),
                resolve_scan(self.root, scan.t2) if scan.t2 is not None else None,
            )
            if tuple(pair.shape[1:]) != scan.shape:
                raise ValueError(f"Geometry changed: {scan.t1}")
            self.cache[scan_index] = normalize_pair(pair)
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(scan_index)
        return resize_slice(self.cache[scan_index], z, self.image_size), scan.subject_id


class SyntheticSlices(Dataset[tuple[Tensor, str]]):
    def __init__(self, image_size: int, split: str, channels: int = 2) -> None:
        generator = torch.Generator().manual_seed({"train": 10, "val": 20, "test": 30}[split])
        self.images = torch.rand(8, channels, image_size, image_size, generator=generator)
        self.split = split

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Tensor, str]:
        return self.images[index], f"synthetic-{self.split}-{index // 2}"
