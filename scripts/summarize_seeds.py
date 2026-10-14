"""Summarize comparable, distinct-seed runs using sample standard deviation."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    if len(args.runs) < 2:
        raise ValueError("At least two seed runs required")
    protocols = [json.loads((run / "protocol.json").read_text()) for run in args.runs]
    records = [json.loads((run / "run.json").read_text()) for run in args.runs]
    seeds = [record["seed"] for record in records]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be distinct")
    if any(protocol != protocols[0] for protocol in protocols):
        raise ValueError("Evaluation protocols differ")
    configs = [
        OmegaConf.to_container(OmegaConf.load(run / ".hydra/config.yaml"), resolve=True)
        for run in args.runs
    ]
    for config in configs:
        if not isinstance(config, dict):
            raise ValueError("Expected a saved run config")
        config.pop("seed")
    if any(config != configs[0] for config in configs):
        raise ValueError("Run configs differ beyond seed")
    for field in ("source_sha256", "lock_sha256", "device_name", "torch_version", "cuda_version"):
        if any(record[field] != records[0][field] for record in records):
            raise ValueError(f"Run provenance differs: {field}")
    metrics = [json.loads((run / "metrics.json").read_text()) for run in args.runs]
    if any(set(row) != set(metrics[0]) for row in metrics):
        raise ValueError("Metric keys differ")
    for metric in metrics[0]:
        values = [row[metric] for row in metrics]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError(f"Non-finite or nonnumeric metric: {metric}")
        print(
            f"{metric}: {mean(values):.6f} +/- {stdev(values):.6f} (n={len(values)}, seeds={seeds})"
        )


if __name__ == "__main__":
    main()
