import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/eval_gate.py"
spec = importlib.util.spec_from_file_location("eval_gate", SCRIPT)
assert spec is not None and spec.loader is not None
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_thresholds_and_individual_subgroups() -> None:
    rules = json.loads((SCRIPT.parent.parent / "eval/gate_rules.json").read_text())
    baseline = {
        "auroc": 0.9,
        "auprc": 0.8,
        "auroc/lesion_small": 0.8,
        "ece": 0.02,
        "latency_ms_p95": 10.0,
    }
    candidate = {
        "auroc": 0.89,
        "auprc": 0.79,
        "auroc/lesion_small": 0.78,
        "ece": 0.025,
        "latency_ms_p95": 12.0,
    }
    assert gate.compare(candidate, baseline, rules) == []
    for metric, rule in rules.items():
        regressed = candidate.copy()
        regressed[metric] += -0.0001 if rule["direction"] == "higher_is_better" else 0.0001
        assert gate.compare(regressed, baseline, rules) == [metric]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "0.9", None])
def test_invalid_metrics_fail_closed(value: Any) -> None:
    with pytest.raises(ValueError, match="finite number"):
        gate.compare(
            {"m": value}, {"m": 0.9}, {"m": {"direction": "higher_is_better", "tolerance": 0.01}}
        )


@pytest.mark.parametrize(
    "rule",
    [
        {"direction": "unknown", "tolerance": 0.01},
        {"direction": "higher_is_better", "tolerance": -0.01},
        {"direction": "lower_is_better", "tolerance": float("nan")},
        {"direction": "higher_is_better"},
    ],
)
def test_invalid_rules_fail_closed(rule: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        gate.compare({"m": 1.0}, {"m": 1.0}, {"m": rule})


def test_missing_subgroup_is_a_failure() -> None:
    assert gate.compare(
        {}, {"m": 1.0}, {"m": {"direction": "higher_is_better", "tolerance": 0.01}}
    ) == ["m"]


def test_incompatible_protocols_fail(tmp_path: Path) -> None:
    files = {
        "candidate": {"m": 1.0},
        "baseline": {"m": 1.0},
        "rules": {"m": {"direction": "higher_is_better", "tolerance": 0.01}},
        "candidate-protocol": {"manifest": "new"},
        "baseline-protocol": {"manifest": "old"},
    }
    args = []
    for name, payload in files.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload))
        args.extend([f"--{name}", str(path)])
    result = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)
    assert result.returncode == 2
    assert "protocols differ" in result.stderr
