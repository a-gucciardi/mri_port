import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from cvproj.config import RunConfig
from cvproj.data.prepare import freeze
from cvproj.model import Autoencoder


@pytest.mark.parametrize("channels", [1, 2])
def test_model_preserves_shape(channels: int) -> None:
    inputs = torch.rand(2, channels, 32, 32)
    result = Autoencoder(4, channels=channels)(inputs)
    assert result.shape == inputs.shape
    assert result.min() >= 0 and result.max() <= 1


def run(module: str, overrides: list[str]) -> None:
    subprocess.run(
        [sys.executable, "-m", module, *overrides],
        check=True,
        timeout=90,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("dataset", ["bobs", "ixi"])
def test_synthetic_training_and_evaluation_are_reproducible(tmp_path: Path, dataset: str) -> None:
    first, repeat, evaluation = (tmp_path / name for name in ("first", "repeat", "evaluation"))
    options = [
        f"dataset={dataset}",
        "synthetic=true",
        "max_epochs=1",
        "image_size=32",
        "width=4",
        "threads=1",
    ]
    for output in (first, repeat):
        run("cvproj.train", [*options, f"hydra.run.dir={output}"])
        assert (output / "best.pt").is_file()
        assert (output / ".hydra/config.yaml").is_file()
        assert json.loads((output / "run.json").read_text())["synthetic"] is True
    assert (first / "metrics.json").read_bytes() == (repeat / "metrics.json").read_bytes()
    run(
        "cvproj.evaluate",
        [*options, f"checkpoint={first / 'best.pt'}", f"hydra.run.dir={evaluation}"],
    )
    assert (first / "metrics.json").read_bytes() == (evaluation / "metrics.json").read_bytes()
    assert (first / "protocol.json").read_bytes() == (evaluation / "protocol.json").read_bytes()
    if dataset == "bobs":
        checkpoint = torch.load(first / "best.pt", weights_only=True)
        checkpoint["config"].pop("dataset_name")
        legacy = tmp_path / "legacy.pt"
        torch.save(checkpoint, legacy)
        run(
            "cvproj.evaluate",
            [*options, f"checkpoint={legacy}", f"hydra.run.dir={tmp_path / 'legacy_eval'}"],
        )
        assert (first / "metrics.json").read_bytes() == (
            tmp_path / "legacy_eval/metrics.json"
        ).read_bytes()


@pytest.mark.parametrize(
    ("dataset", "name", "modalities", "manifest"),
    [
        ("bobs", "BOBs", ["T1w", "T2w"], "eval/frozen_manifest.json"),
        ("ixi", "IXI", ["T1w"], "eval/ixi_manifest.json"),
    ],
)
def test_dataset_configs_select_matching_manifest(
    dataset: str, name: str, modalities: list[str], manifest: str
) -> None:
    from cvproj.runtime import evaluation_protocol, get_config

    config_dir = Path(__file__).parents[1] / "src/cvproj/conf"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = get_config(compose(config_name="config", overrides=[f"dataset={dataset}"]))
    assert isinstance(config, RunConfig)
    assert config.dataset_name == name
    assert config.manifest == manifest
    assert config.checkpoint == ""
    assert evaluation_protocol(config, "example", "val")["modalities"] == modalities


def test_real_ixi_training_and_checkpoint_mismatch(ixi_root: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "ixi_manifest.json"
    freeze(ixi_root, manifest, dataset="IXI")
    output = tmp_path / "trained"
    options = [
        "dataset=ixi",
        f"data_root={ixi_root}",
        f"manifest={manifest}",
        "max_epochs=1",
        "image_size=32",
        "width=4",
        "threads=1",
    ]
    run("cvproj.train", [*options, f"hydra.run.dir={output}"])
    evaluation = tmp_path / "evaluated"
    run(
        "cvproj.evaluate",
        [*options, f"checkpoint={output / 'best.pt'}", f"hydra.run.dir={evaluation}"],
    )
    assert (output / "metrics.json").read_bytes() == (evaluation / "metrics.json").read_bytes()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cvproj.evaluate",
            "dataset=bobs",
            f"checkpoint={output / 'best.pt'}",
            f"hydra.run.dir={tmp_path / 'wrong_dataset'}",
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode != 0
    assert "Checkpoint was trained on IXI" in result.stderr
