"""Deterministic airline validation cohort for the performance-oriented and efficiency-oriented ladders.

Use all 20 tasks in the official tau2 airline test split, separate from screening.
Order tasks by ascending SHA256(SALT + task_id). Evaluate 29 configurations
from rungs.ARMS on each task, with matched seeds across configurations.
Three trials per task/configuration; seed SEED_BASE + 1000*trial + task index.
Trial 0 uses the bare arm label; trials 1 and 2 use <arm>@t<n>.

  python cohort_ladder.py --data <TAU2_DATA_DIR> --skills <skills.py> --out cohort.json
  python cohort_ladder.py --data ... --skills ... --verify cohort.json
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

from cohort import skill_sources, sha256_file, OFFICIAL_SHA256, DATA_FILES   # shared cohort utilities
import units as U
import rungs as RG

SALT = "life_tau2_ladder_airline_v2:"
DOMAIN, SPLIT = "airline", "test"
SEED_BASE = 20260920
K_TRIALS = 3
TRAIN_SCREEN = "life_tau2_fine_airline_v1"   # Screening cohort (30 tasks) used for the component tables.


def key(task_id: str) -> str:
    return hashlib.sha256((SALT + task_id).encode()).hexdigest()


def trial_label(arm: str, trial: int) -> str:
    return arm if trial == 0 else f"{arm}@t{trial}"


def split_label(label: str) -> tuple[str, int]:
    arm, _, t = label.partition("@t")
    return arm, (int(t) if t else 0)


def build(data_dir: Path, skills_py: Path) -> dict:
    sources = skill_sources(Path(skills_py))[DOMAIN]
    ddir = Path(data_dir) / "tau2" / "domains" / DOMAIN
    data_sha = {f"{DOMAIN}/{f}": sha256_file(ddir / f) for f in DATA_FILES}
    split = json.loads((ddir / "split_tasks.json").read_text())
    test, train = list(split[SPLIT]), list(split["train"])
    chosen = sorted(test, key=key)
    n = len(RG.ARMS)
    tasks = []
    for i, t in enumerate(chosen):
        rot = RG.ARMS[i % n:] + RG.ARMS[:i % n]
        tasks.append({"index": i, "domain": DOMAIN, "task_id": t, "key": key(t),
                      "seed": SEED_BASE + i, "seeds": [SEED_BASE + 1000 * k + i for k in range(K_TRIALS)],
                      "h5_skill_source": t in sources,
                      "arm_order": [trial_label(a, k) for k in range(K_TRIALS) for a in rot]})
    official = {k: v for k, v in OFFICIAL_SHA256.items() if k.startswith(DOMAIN + "/")}
    return {
        "name": "life_tau2_ladder_airline_v2", "salt": SALT, "seed_base": SEED_BASE, "k_trials": K_TRIALS,
        "arms": RG.ARMS, "rungs": RG.RUNGS, "rung_tables": {k: v.name for k, v in RG.TABLES.items()},
        "on_units": {a: RG.on_units(a) for a in RG.ARMS}, "units": [u.__dict__ for u in U.UNITS],
        "arm_order_rule": f"task-major; within a task, trial-major, the {n} arms rotated by index mod {n}",
        "trial_label_rule": "trial 0 = bare arm label; trial k>=1 = '<arm>@t<k>'; seed = seeds[k]",
        "domains": {DOMAIN: {"split": SPLIT, "test_size": len(test), "selected": len(chosen),
                             "train_size_not_used": len(train), "train_screen": TRAIN_SCREEN,
                             "h5_skill_source_ids_in_test": sorted(set(test) & sources, key=int)}},
        "n_tasks": len(tasks), "n_cells": len(tasks) * n * K_TRIALS, "tasks": tasks,
        "data_sha256": data_sha, "official_sha256": official,
        "data_matches_official": all(data_sha[k] == v for k, v in official.items()),
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
    print(json.dumps({k: v for k, v in c.items() if k not in {"tasks", "units", "on_units", "arms", "rungs"}}, indent=2))
