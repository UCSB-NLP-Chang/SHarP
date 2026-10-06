"""Deterministic cohort for the retail single-off screen.

Tasks: the 50-task retail training set in `splits_v1.json`, sampled from the official
`train` split and stratified by action-count buckets to match the validation profile.
The stored IDs are hashed into the contract.  The official `test` split (40) is the validation set and stays
untouched; recorded only for auditability.  H5 skill provenance is flagged, never filtered.
Order: ascending SHA256(SALT + task_id).  Per-task seed SEED_BASE + index, identical across all 49 arms.
Arms: `full` + one `off_<uid>` per unit in units.py (49).  k = 1 trial per cell.

  python cohort_fine.py --data <TAU2_DATA_DIR> --skills <skills.py> --out cohort.json
  python cohort_fine.py --data ... --skills ... --verify cohort.json
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

from cohort import skill_sources, sha256_file, OFFICIAL_SHA256, DATA_FILES   # shared cohort utilities
import units as U

SALT = "life_tau2_fine_retail_v1:"
DOMAIN, SPLIT = "retail", "train"
SEED_BASE = 20260921
K_TRIALS = 1
SPLITS_FILE = Path(__file__).resolve().parent / "splits_v1.json"


def key(task_id: str) -> str:
    return hashlib.sha256((SALT + task_id).encode()).hexdigest()


def build(data_dir: Path, skills_py: Path) -> dict:
    sources = skill_sources(Path(skills_py))[DOMAIN]
    ddir = Path(data_dir) / "tau2" / "domains" / DOMAIN
    data_sha = {f"{DOMAIN}/{f}": sha256_file(ddir / f) for f in DATA_FILES}
    split = json.loads((ddir / "split_tasks.json").read_text())
    train, test = list(split[SPLIT]), list(split["test"])
    sp = json.loads(SPLITS_FILE.read_text())["benchmarks"]["tau2_retail"]
    chosen_ids = [str(t) for t in sp["train"]["ids"]]
    assert len(chosen_ids) == 50 and len(set(chosen_ids)) == 50 and set(chosen_ids) <= set(train), "splits_v1 train ids must be 50 official train ids"
    assert set(str(t) for t in sp["validation"]["ids"]) == set(test), "splits_v1 validation must equal the official test split"
    chosen = sorted(chosen_ids, key=key)
    n = len(U.ARMS)
    tasks = [{"index": i, "domain": DOMAIN, "task_id": t, "key": key(t), "seed": SEED_BASE + i,
              "h5_skill_source": t in sources, "arm_order": U.ARMS[i % n:] + U.ARMS[:i % n]}
             for i, t in enumerate(chosen)]
    official = {k: v for k, v in OFFICIAL_SHA256.items() if k.startswith(DOMAIN + "/")}
    return {
        "name": "life_tau2_fine_retail_v1", "salt": SALT, "seed_base": SEED_BASE, "k_trials": K_TRIALS,
        "arms": U.ARMS, "units": [u.__dict__ | {"tools": list(u.tools)} for u in U.UNITS],
        "arm_order_rule": f"task-major; within a task rotate the {n} arms by index mod {n}",
        "splits_v1_sha256": sha256_file(SPLITS_FILE), "splits_v1_train_rule": sp["train_rule"],
        "domains": {DOMAIN: {"split": SPLIT, "train_size": len(train), "selected": len(chosen),
                             "test_size_heldout": len(test),
                             "h5_skill_source_ids_in_train_selected": sorted(set(chosen) & sources, key=int),
                             "h5_skill_source_ids_in_test": sorted(set(test) & sources, key=int)}},
        "n_tasks": len(tasks), "n_cells": len(tasks) * n * K_TRIALS, "tasks": tasks,
        "data_sha256": data_sha, "official_sha256": official,
        "data_matches_official": all(data_sha[k] == v for k, v in official.items()),
        "heldout_validation": {"domain": DOMAIN, "split": "test", "ids": test},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--skills", required=True)
    ap.add_argument("--out"); ap.add_argument("--verify")
    a = ap.parse_args()
    c = build(Path(a.data), Path(a.skills))
    if a.verify:
        same = json.loads(Path(a.verify).read_text()) == c
        print(json.dumps({"verify": a.verify, "identical": same, "n_cells": c["n_cells"]}))
        sys.exit(0 if same else 1)
    if a.out:
        Path(a.out).write_text(json.dumps(c, indent=2) + "\n")
    print(json.dumps({k: v for k, v in c.items() if k not in {"tasks", "units", "arms", "heldout_validation"}}, indent=2))
