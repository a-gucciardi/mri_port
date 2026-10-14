import hashlib
import json
import os
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from cvproj.config import RunConfig
from cvproj.data.dataset import MRISlices, SyntheticSlices
from cvproj.data.prepare import load_manifest, sha256
from cvproj.model import Autoencoder


def get_config(raw: DictConfig) -> RunConfig:
    config = OmegaConf.to_object(raw)
    if not isinstance(config, RunConfig):
        raise TypeError("Expected structured RunConfig")
    for name in ("batch_size", "max_epochs", "width", "threads", "slice_step"):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive")
    if config.image_size < 16 or config.image_size % 4:
        raise ValueError("image_size must be >=16 and divisible by 4")
    if not np.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if config.subset_subjects < 0:
        raise ValueError("subset_subjects must be nonnegative")
    if config.device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if config.split not in ("val", "test"):
        raise ValueError("Evaluation split must be val or test")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; install with uv sync --extra cuda")
    return config


def initialize(config: RunConfig) -> torch.device:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device(config.device)


def datasets(
    config: RunConfig, splits: tuple[str, ...]
) -> tuple[dict[str, Dataset[tuple[Tensor, str]]], str]:
    if config.synthetic:
        return {
            split: SyntheticSlices(config.image_size, split) for split in splits
        }, "synthetic-v1"
    scans = load_manifest(Path(config.manifest), Path(config.data_root))
    result: dict[str, Dataset[tuple[Tensor, str]]] = {}
    for split in splits:
        selected = [scan for scan in scans if scan.split == split]
        if config.subset_subjects:
            subjects = sorted({scan.subject_id for scan in selected})[: config.subset_subjects]
            selected = [scan for scan in selected if scan.subject_id in subjects]
        result[split] = MRISlices(
            Path(config.data_root), selected, config.image_size, config.slice_step
        )
    return result, sha256(Path(config.manifest))


def loader(dataset: Dataset[tuple[Tensor, str]], config: RunConfig) -> DataLoader:
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)


def reconstruction_error(predicted: Tensor, images: Tensor) -> Tensor:
    mask = images > 0
    denominator = mask.flatten(1).sum(1)
    if (denominator == 0).any():
        raise ValueError("Slice has no positive foreground")
    return ((predicted - images).square() * mask).flatten(1).sum(1) / denominator


def score(
    model: Autoencoder, data: DataLoader, device: torch.device
) -> tuple[dict[str, float], dict[str, float]]:
    model.eval()
    per_subject: dict[str, list[float]] = defaultdict(list)
    with torch.inference_mode():
        for images, subjects in data:
            images = images.to(device)
            errors = reconstruction_error(model(images), images)
            if not torch.isfinite(errors).all():
                raise ValueError("Non-finite reconstruction errors")
            for subject, error in zip(subjects, errors.cpu().tolist(), strict=True):
                per_subject[subject].append(error)
    if not per_subject:
        raise ValueError("No evaluation subjects")
    subject_scores = {
        subject: float(np.mean(values)) for subject, values in sorted(per_subject.items())
    }
    return {"reconstruction_mse": float(np.mean(list(subject_scores.values())))}, subject_scores


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def provenance(config: RunConfig, manifest_hash: str) -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    source = hashlib.sha256()
    for path in sorted(package_root.rglob("*")):
        if path.suffix in (".py", ".yaml"):
            source.update(path.relative_to(package_root).as_posix().encode())
            source.update(path.read_bytes())
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    return {
        "manifest_sha256": manifest_hash,
        "source_sha256": source.hexdigest(),
        "lock_sha256": sha256(Path("uv.lock")),
        "git_commit": git.stdout.strip() if git.returncode == 0 else None,
        "git_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": config.device,
        "device_name": torch.cuda.get_device_name() if config.device == "cuda" else "cpu",
        "seed": config.seed,
        "synthetic": config.synthetic,
        "cohort_status": "synthetic" if config.synthetic else "health_unverified",
    }


def evaluation_protocol(config: RunConfig, manifest_hash: str, split: str) -> dict[str, Any]:
    return {
        "version": 1,
        "manifest_sha256": manifest_hash,
        "split": split,
        "synthetic": config.synthetic,
        "image_size": config.image_size,
        "slice_step": config.slice_step,
        "subset_subjects": config.subset_subjects,
        "normalization": "positive-voxel-per-volume-p1-p99-v1",
        "aggregation": "mean-slice-mse-then-mean-subject-v1",
        "modalities": ["T1w", "T2w"],
    }
