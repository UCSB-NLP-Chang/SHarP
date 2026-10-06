"""Deterministic cohort for the airline single-off screen.

Tasks: the whole official tau2 airline `train` split (30 ids). The `test` split (20 ids) is reserved for validation
and recorded for split verification.  H5 skill provenance is flagged, never filtered.
Order: ascending SHA256(SALT + task_id).  Per-task seed SEED_BASE + index, identical across all 45 arms.
Arms: `full` + one `off_<uid>` per unit in units.py (45).  k = 1 trial per cell.

Future trials (if added) are separate cells with the arm label suffixed `@t<n>` (n >= 1); trial 0 is
the unsuffixed arm, so adding trials never re-runs or renames an existing cell.

  python cohort_fine.py --data <TAU2_DATA_DIR> --skills <skills.py> --out cohort.json
  python cohort_fine.py --data ... --skills ... --verify cohort.json
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

from cohort import skill_sources, sha256_file, OFFICIAL_SHA256, DATA_FILES   # shared cohort utilities
import units as U

SALT = "life_tau2_fine_airline_v1:"
DOMAIN, SPLIT, TARGET = "airline", "train", 30
SEED_BASE = 20260919
K_TRIALS = 1


def key(task_id: str) -> str:
    return hashlib.sha256((SALT + task_id).encode()).hexdigest()


def build(data_dir: Path, skills_py: Path) -> dict:
    sources = skill_sources(Path(skills_py))[DOMAIN]
    ddir = Path(data_dir) / "tau2" / "domains" / DOMAIN
    data_sha = {f"{DOMAIN}/{f}": sha256_file(ddir / f) for f in DATA_FILES}
    split = json.loads((ddir / "split_tasks.json").read_text())
    train, test = list(split[SPLIT]), list(split["test"])
    chosen = sorted(train, key=key)[:TARGET]
    n = len(U.ARMS)
    tasks = [{"index": i, "domain": DOMAIN, "task_id": t, "key": key(t), "seed": SEED_BASE + i,
              "h5_skill_source": t in sources, "arm_order": U.ARMS[i % n:] + U.ARMS[:i % n]}
             for i, t in enumerate(chosen)]
    official = {k: v for k, v in OFFICIAL_SHA256.items() if k.startswith(DOMAIN + "/")}
    return {
        "name": "life_tau2_fine_airline_v1", "salt": SALT, "seed_base": SEED_BASE, "k_trials": K_TRIALS,
        "arms": U.ARMS, "units": [u.__dict__ for u in U.UNITS],
        "arm_order_rule": f"task-major; within a task rotate the {n} arms by index mod {n}",
        "domains": {DOMAIN: {"split": SPLIT, "train_size": len(train), "selected": len(chosen),
                             "test_size_heldout": len(test),
                             "h5_skill_source_ids_in_train": sorted(set(train) & sources, key=int),
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
