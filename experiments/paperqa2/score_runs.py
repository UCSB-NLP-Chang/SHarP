#!/usr/bin/env python3
"""Score explicit final-letter results against a protected target file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from protocol import aggregate_scores, score_answer


def load_jsonl(path: Path, id_field: str) -> dict[str, dict[str, Any]]:
    with path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    by_id = {str(row[id_field]): row for row in rows}
    if len(by_id) != len(rows):
        raise SystemExit(f"Duplicate {id_field} values in {path}")
    return by_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--protected-targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_jsonl(args.results, "question_id")
    targets = load_jsonl(args.protected_targets, "question_id")
    unknown = set(results) - set(targets)
    if unknown:
        raise SystemExit(f"Results contain unknown question IDs: {sorted(unknown)}")

    missing = set(targets) - set(results)
    if missing:
        raise SystemExit(
            f"Partial run: {len(results)}/{len(targets)} questions have results; "
            f"{len(missing)} questions were not run. Complete the run before scoring."
        )

    per_task = []
    for question_id, target in targets.items():
        result = results.get(question_id)
        per_task.append(score_answer(None if result is None else result.get("answer"), target))
    output = {
        "metrics": aggregate_scores(per_task),
        "per_task": per_task,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
