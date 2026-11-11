from dataclasses import dataclass

from hydra.core.config_store import ConfigStore


@dataclass
class RunConfig:
    seed: int = 0
    dataset_name: str = "BOBs"
    data_root: str = "/data/bobs"
    manifest: str = "eval/frozen_manifest.json"
    synthetic: bool = False
    image_size: int = 128
    slice_step: int = 4
    subset_subjects: int = 0
    batch_size: int = 16
    max_epochs: int = 10
    learning_rate: float = 0.001
    width: int = 16
    device: str = "cpu"
    threads: int = 4
    checkpoint: str = ""
    split: str = "val"


ConfigStore.instance().store(name="schema", node=RunConfig)
