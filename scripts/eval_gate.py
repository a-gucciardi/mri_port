#!/usr/bin/env python
"""Fail closed on invalid inputs, incompatible protocols, or metric regressions."""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def load(path: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Expected nonempty JSON object: {path}")
    return value


def finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def compare(
    candidate: dict[str, Any], baseline: dict[str, Any], rules: dict[str, Any]
) -> list[str]:
    if not rules:
        raise ValueError("Gate rules cannot be empty")
    failures = []
    for metric, rule in rules.items():
        if not isinstance(rule, dict) or set(rule) != {"direction", "tolerance"}:
            raise ValueError(f"Invalid rule for {metric}")
        direction = rule["direction"]
        if direction not in ("higher_is_better", "lower_is_better"):
            raise ValueError(f"Unknown direction for {metric}: {direction}")
        tolerance = finite_number(rule["tolerance"], f"{metric} tolerance")
        if tolerance < 0:
            raise ValueError(f"{metric} tolerance must be nonnegative")
        if metric not in candidate or metric not in baseline:
            failures.append(metric)
            print(f"[FAIL] {metric}: missing from candidate or baseline")
            continue
        c = finite_number(candidate[metric], f"candidate {metric}")
        b = finite_number(baseline[metric], f"baseline {metric}")
        boundary = b - tolerance if direction == "higher_is_better" else b + tolerance
        ok = c >= boundary if direction == "higher_is_better" else c <= boundary
        print(
            f"[{'OK  ' if ok else 'FAIL'}] {metric}: candidate={c:.6f} baseline={b:.6f} "
            f"tol={tolerance}"
        )
        if not ok:
            failures.append(metric)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--rules", required=True)
    parser.add_argument("--candidate-protocol")
    parser.add_argument("--baseline-protocol")
    args = parser.parse_args()
    try:
        if bool(args.candidate_protocol) != bool(args.baseline_protocol):
            raise ValueError("Provide both protocol files, or neither")
        if args.candidate_protocol:
            if load(args.candidate_protocol) != load(args.baseline_protocol):
                raise ValueError("Evaluation protocols differ; results are not comparable")
        failures = compare(load(args.candidate), load(args.baseline), load(args.rules))
    except (OSError, ValueError, TypeError, OverflowError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2
    if failures:
        print(f"Gate failed on: {', '.join(failures)}")
        return 1
    print("Gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
