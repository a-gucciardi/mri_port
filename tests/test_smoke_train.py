import json
import subprocess
import sys
from pathlib import Path

import torch

from cvproj.model import Autoencoder


def test_model_preserves_shape() -> None:
    inputs = torch.rand(2, 2, 32, 32)
    result = Autoencoder(4)(inputs)
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


def test_synthetic_training_and_evaluation_are_reproducible(tmp_path: Path) -> None:
    first, repeat, evaluation = (tmp_path / name for name in ("first", "repeat", "evaluation"))
    options = ["synthetic=true", "max_epochs=1", "image_size=32", "width=4", "threads=1"]
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
