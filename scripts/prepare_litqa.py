#!/usr/bin/env python3
"""Materialize the frozen LitQA2 task IDs using the recorded choice permutation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/paperqa2"))
from protocol import materialize_question


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", type=Path, required=True, help="LAB-Bench LitQA2/litqa-v2-public.jsonl")
    p.add_argument("--split", choices=["train", "validation"], required=True)
    p.add_argument("--output", type=Path, required=True, help="A new runtime directory, outside version control")
    a = p.parse_args()
    ids = json.loads((ROOT / f"configs/splits/litqa2_{a.split}.json").read_text())["question_ids"]
    source = [json.loads(line) for line in a.records.read_text().splitlines() if line.strip()]
    records = {str(r["id"]): r for r in source}
    if len(records) != len(source):
        p.error("Duplicate IDs in source records")
    missing = set(ids) - set(records)
    if missing:
        p.error(f"Source is missing {len(missing)} frozen task IDs")
    questions = [materialize_question(records[qid]) for qid in ids]
    a.output.mkdir(parents=True, exist_ok=False)
    for name, values in [("agent_inputs.jsonl", [q.agent_input() for q in questions]),
                         ("protected_targets.jsonl", [q.protected_target() for q in questions])]:
        path = a.output / name
        path.write_text("".join(json.dumps(v, sort_keys=True) + "\n" for v in values))
        path.chmod(0o600)
    print(f"Materialized {len(questions)} tasks. Supply only agent_inputs.jsonl to the runner.")


if __name__ == "__main__":
    main()
