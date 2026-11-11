import json
import logging
import time
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from cvproj.data.dataset import MRISlices
from cvproj.data.prepare import DATASET_MODALITIES
from cvproj.model import Autoencoder
from cvproj.runtime import (
    datasets,
    evaluation_protocol,
    get_config,
    initialize,
    loader,
    provenance,
    reconstruction_error,
    score,
    write_json,
)

LOG = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(raw: DictConfig) -> None:
    config = get_config(raw)
    device = initialize(config)
    output = Path(HydraConfig.get().runtime.output_dir)
    if any((output / name).exists() for name in ("run.json", "history.jsonl", "best.pt")):
        raise FileExistsError(f"Run directory already contains training results: {output}")
    data, manifest_hash = datasets(config, ("train", "val"))
    model = Autoencoder(config.width, channels=len(DATASET_MODALITIES[config.dataset_name])).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    write_json(output / "run.json", provenance(config, manifest_hash))
    write_json(output / "protocol.json", evaluation_protocol(config, manifest_hash, "val"))
    best = float("inf")
    for epoch in range(config.max_epochs):
        started = time.perf_counter()
        train_error = 0.0
        train_slices = 0
        model.train()
        dataset = data["train"]
        if isinstance(dataset, MRISlices):
            # Keep slices contiguous to avoid repeatedly decompressing full volumes.
            generator = torch.Generator().manual_seed(config.seed + epoch)
            order = torch.randperm(len(dataset.scans), generator=generator).tolist()
            ranks = {scan_index: i for i, scan_index in enumerate(order)}
            dataset.indices.sort(key=lambda entry: (ranks[entry[0]], entry[1]))
        for images, _ in loader(dataset, config):
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = reconstruction_error(model(images), images).mean()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            train_error += loss.item() * len(images)
            train_slices += len(images)
        metrics, subjects = score(model, loader(data["val"], config), device)
        LOG.info("epoch=%d val_reconstruction_mse=%.6f", epoch, metrics["reconstruction_mse"])
        with (output / "history.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_slice_mse": train_error / train_slices,
                        "elapsed_seconds": time.perf_counter() - started,
                        **metrics,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
        if metrics["reconstruction_mse"] < best:
            best = metrics["reconstruction_mse"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": OmegaConf.to_container(raw, resolve=True),
                    "manifest_sha256": manifest_hash,
                    "epoch": epoch,
                },
                output / "best.pt",
            )
            write_json(output / "metrics.json", metrics)
            write_json(output / "subject_scores.json", subjects)
    LOG.info("Run saved to %s (best validation MSE %.6f)", output, best)


if __name__ == "__main__":
    main()
