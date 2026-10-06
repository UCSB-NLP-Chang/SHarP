"""Build deterministic DeepPlanning-Shopping cohorts.

Use the stored train/validation split, mode-specific salts and seed bases.
Order tasks by SHA-256 of salt plus question ID; rotate arm order by task index.
Validate disjoint splits and, when supplied, the reference validation cohort.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path

CODE_FINE = Path(__file__).resolve().parent
PARENT_CODE = Path(os.environ["JIT_PARENT_CODE"])
for p in (str(PARENT_CODE), str(CODE_FINE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import common  # noqa: E402
import units as U  # noqa: E402

MODES = {"screen": {"salt": "jit_shopping_fine_v1:", "seed_base": 20260923, "split": "train", "arms": U.SCREEN_ARMS, "k": 1},
         "premise": {"salt": "jit_shopping_premise_v1:", "seed_base": 20260924, "split": "validation", "arms": U.PREMISE_ARMS, "k": 3}}
PARENT_COHORT = common.ROOT / "parent_cohort.json"


def build(mode: str, items=None, ids=None) -> dict:
    m = MODES[mode]
    items = items if items is not None else common.load_items()
    by_id = {i["question_id"]: i for i in items}
    sp = json.loads((CODE_FINE / "splits_v1.json").read_text())["benchmarks"]["deepplanning_shopping"]
    if ids is None:
        ids = list(sp[m["split"]]["ids"])
        train, val = set(sp["train"]["ids"]), set(sp["validation"]["ids"])
        assert not (train & val) and len(train) == 50 and len(val) == 30
        if mode == "premise" and PARENT_COHORT.is_file():
            assert set(ids) == {t["question_id"] for t in json.loads(PARENT_COHORT.read_text())["tasks"]}, "validation IDs differ from the reference cohort"
    assert all(q in by_id for q in ids), [q for q in ids if q not in by_id]
    key = lambda q: hashlib.sha256((m["salt"] + q).encode()).hexdigest()
    ordered = sorted(ids, key=key); arms = m["arms"]; n = len(arms); tasks = []
    for i, q in enumerate(ordered):
        item = by_id[q]; db = Path(item["_db_dir"]); rot = arms[i % n:] + arms[:i % n]
        tasks.append({"index": i, "question_id": q, "level": str(item["_level"]), "cohort_key": key(q),
                      "arms": [U.trial_label(a, k) for k in range(m["k"]) for a in rot], "seed": m["seed_base"] + i,
                      "data_hashes": {f: common.digest(db / f) for f in ("products.jsonl", "user_info.json", "validation_cases.json")}})
    return {"name": f"jit_shopping_{'fine' if mode == 'screen' else 'premise'}_v1", "mode": mode, "salt": m["salt"],
            "seed_base": m["seed_base"], "k_trials": m["k"], "arms": arms, "units": [list(u) for u in U.UNITS],
            "split": m["split"], "levels": {lv: sum(t["level"] == lv for t in tasks) for lv in ("1", "2", "3")},
            "n_tasks": len(tasks), "n_cells": len(tasks) * n * m["k"], "tasks": tasks,
            "gold_unattainable": {q: v for q, v in common.GOLD_UNATTAINABLE.items() if q in ids}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=list(MODES)); ap.add_argument("--out"); ap.add_argument("--verify")
    a = ap.parse_args(); c = build(a.mode)
    if a.verify:
        same = json.loads(Path(a.verify).read_text()) == c
        print(json.dumps({"verify": a.verify, "identical": same, "n_cells": c["n_cells"]})); sys.exit(0 if same else 1)
    if a.out:
        Path(a.out).write_text(json.dumps(c, indent=1) + "\n")
    print(json.dumps({k: v for k, v in c.items() if k not in ("tasks", "units")}, indent=1))
