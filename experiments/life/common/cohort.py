"""Deterministic LIFE/tau2 Airline and Retail cohort.

Selection rule (frozen): for each domain, take the upstream `test` split ids
compute SHA256(SALT + id),
sort ascending by that hex digest, take the first TARGET_PER_DOMAIN ids.
No outcome-based filtering.  Documented H5 skill sources (task_ids listed in
tau2/harness/skills.py) are checked against the test split BEFORE sampling and
would be excluded; for airline and retail every listed source id is in the
train split, so nothing is excluded.

Usage:
  python cohort.py --data <TAU2_DATA_DIR> --skills <path to skills.py> --out cohort.json
  python cohort.py --data ... --skills ... --verify cohort.json
"""
from __future__ import annotations
import argparse, ast, hashlib, json, sys
from pathlib import Path

SALT = "reexplore_life_tau2_v1:"
TARGET_PER_DOMAIN = 30
DOMAINS = ["airline", "retail"]
ARMS = ["full", "baseline", "reference"]
SEED_BASE = 20260908
DATA_FILES = ["db.json", "policy.md", "tasks.json", "split_tasks.json"]
# Official tau2-bench data at sierra-research/tau2-bench@672227c6b6676edc20d57ea53b7000262aae77b9
# (verified against the pinned upstream data)
OFFICIAL_SHA256 = {
    "airline/db.json": "1af9fea6e03ca7ca15a22bb3fcaf3e351393e3fc9070b6777947da8996f7531b",
    "airline/policy.md": "10dc0525421521208be39cee235bba84a16e2bcba9899eb93d92cd81d2f62fc4",
    "airline/tasks.json": "ccd8ba737b4cc371415af70151187788f728d6108d0916e73bb4317b40542052",
    "airline/split_tasks.json": "b22ced4d9a9850ac9aea31c53bdcb6d6009058140bd9acc7db37c1d36222ba8b",
    "retail/db.json": "413a65160adbdb5fde0ffc0015c49b6d70250b10c18128de169b597af7766765",
    "retail/policy.md": "2c9652afbce57d6e087768d37cda64d31c53d50b3e3225cfdb791bac66466467",
    "retail/tasks.json": "8e03ebce7901bd6218e7a7dc3105faa9324091a68058f7fe61c65262868812e8",
    "retail/split_tasks.json": "ed0580ec52575b63fbf76568af42490da6ee7783ecb4aa81af46961291358f20",
}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def key(task_id: str) -> str:
    return hashlib.sha256((SALT + task_id).encode()).hexdigest()


def skill_sources(skills_py: Path) -> dict[str, set[str]]:
    """Task ids listed as H5 skill provenance, per domain, parsed from skills.py without importing tau2."""
    tree = ast.parse(skills_py.read_text())
    out = {}
    names = {"AIRLINE_SKILLS": "airline", "RETAIL_SKILLS": "retail"}
    for node in tree.body:
        tgt = None
        if isinstance(node, ast.Assign):
            tgt = getattr(node.targets[0], "id", None)
        elif isinstance(node, ast.AnnAssign):
            tgt = getattr(node.target, "id", None)
        if tgt in names:
            ids = set()
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "Skill":
                    for kw in call.keywords:
                        if kw.arg == "task_ids":
                            ids |= {str(c.value) for c in kw.value.elts}
            out[names[tgt]] = ids
    assert set(out) == set(DOMAINS), out.keys()
    return out


def build(data_dir: Path, skills_py: Path) -> dict:
    sources = skill_sources(skills_py)
    domains, entries, data_sha = {}, [], {}
    for d in DOMAINS:
        ddir = data_dir / "tau2" / "domains" / d
        for f in DATA_FILES:
            data_sha[f"{d}/{f}"] = sha256_file(ddir / f)
        split = json.loads((ddir / "split_tasks.json").read_text())
        test_ids = list(split["test"])
        assert len(set(test_ids)) == len(test_ids)
        excluded = sorted(set(test_ids) & sources[d], key=int)
        eligible = [t for t in test_ids if t not in sources[d]]
        ranked = sorted(eligible, key=key)
        chosen = ranked[:TARGET_PER_DOMAIN]
        domains[d] = {
            "split": "test",
            "test_size": len(test_ids),
            "skill_source_ids_in_test_excluded": excluded,
            "skill_source_ids_total": len(sources[d]),
            "eligible": len(eligible),
            "target": TARGET_PER_DOMAIN,
            "selected": len(chosen),
            "note": (
                "test split smaller than target; all eligible test tasks taken"
                if len(chosen) < TARGET_PER_DOMAIN else "hash sample"
            ),
        }
        for tid in chosen:
            i = len(entries)
            entries.append({
                "index": i,
                "domain": d,
                "task_id": tid,
                "key": key(tid),
                "seed": SEED_BASE + i,
                "arm_order": ARMS[i % 3:] + ARMS[:i % 3],
            })
    return {
        "name": "reexplore_life_tau2_v1",
        "salt": SALT,
        "seed_base": SEED_BASE,
        "arms": ARMS,
        "arm_order_rule": "task-major cyclic Latin: rotate [full, baseline, reference] by index mod 3",
        "domains": domains,
        "n_tasks": len(entries),
        "n_cells": len(entries) * len(ARMS),
        "tasks": entries,
        "data_sha256": data_sha,
        "official_sha256": OFFICIAL_SHA256,
        "data_matches_official": all(data_sha[k] == v for k, v in OFFICIAL_SHA256.items()),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--skills", required=True)
    ap.add_argument("--out")
    ap.add_argument("--verify")
    a = ap.parse_args()
    c = build(Path(a.data), Path(a.skills))
    if a.verify:
        ref = json.loads(Path(a.verify).read_text())
        same = ref == c
        print(json.dumps({"verify": a.verify, "identical": same, "n_tasks": c["n_tasks"],
                          "data_matches_official": c["data_matches_official"]}))
        sys.exit(0 if same else 1)
    if a.out:
        Path(a.out).write_text(json.dumps(c, indent=2) + "\n")
    print(json.dumps({k: v for k, v in c.items() if k != "tasks"}, indent=2))
