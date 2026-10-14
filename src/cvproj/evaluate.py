import logging
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from cvproj.data.prepare import sha256
from cvproj.model import Autoencoder
from cvproj.runtime import (
    datasets,
    evaluation_protocol,
    get_config,
    initialize,
    loader,
    provenance,
    score,
    write_json,
)

LOG = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(raw: DictConfig) -> None:
    config = get_config(raw)
    if not config.checkpoint:
        raise ValueError("checkpoint=<path/to/best.pt> is required")
    output = Path(HydraConfig.get().runtime.output_dir)
    if any((output / name).exists() for name in ("run.json", "metrics.json", "best.pt")):
        raise FileExistsError(f"Run directory already contains results: {output}")
    device = initialize(config)
    data, manifest_hash = datasets(config, (config.split,))
    checkpoint = torch.load(config.checkpoint, map_location=device, weights_only=True)
    if checkpoint["manifest_sha256"] != manifest_hash:
        raise ValueError("Checkpoint was trained with a different dataset manifest")
    for field in ("synthetic", "image_size", "slice_step"):
        if checkpoint["config"][field] != getattr(config, field):
            raise ValueError(f"Checkpoint preprocessing mismatch: {field}")
    model = Autoencoder(checkpoint["config"]["width"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    metrics, subjects = score(model, loader(data[config.split], config), device)
    write_json(output / "metrics.json", metrics)
    write_json(output / "subject_scores.json", subjects)
    write_json(output / "protocol.json", evaluation_protocol(config, manifest_hash, config.split))
    record = provenance(config, manifest_hash)
    record["checkpoint_sha256"] = sha256(Path(config.checkpoint))
    write_json(output / "run.json", record)
    LOG.info("%s metrics=%s saved to %s", config.split, metrics, output)

    model.eval()

    with torch.inference_mode():
        images, subject_ids = next(iter(loader(data[config.split], config)))
        images = images.to(device)
        reconstructed = model(images)

    print("Subjects:", subject_ids)
    print("Input shape:", images.shape)
    print("Reconstruction shape:", reconstructed.shape)


if __name__ == "__main__":
    main()
