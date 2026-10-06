"""Run one DeepPlanning-Shopping task/configuration cell.

Use run_cell.py for isolated attempts, telemetry, failure classification, and
scoring. Launch run_seed_variant.py with JITFINE_OFF selecting disabled modules.
Trial k uses the task seed plus 1000*k.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

CODE_FINE = Path(__file__).resolve().parent
PARENT_CODE = Path(os.environ["JIT_PARENT_CODE"])
for p in (str(PARENT_CODE), str(CODE_FINE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import common  # noqa: E402
import run_cell as RC  # noqa: E402
import units as U  # noqa: E402

OUTSIDE_CONTRACT = {"batch_fine.py", "checks_fine.py", "aggregate_fine.py"}


def contract() -> dict:
    up = common.UPSTREAM
    files = {"code_fine": {p.name: common.digest(p) for p in sorted(CODE_FINE.glob("*.py")) if p.name not in OUTSIDE_CONTRACT},
             "parent_code": {p.name: common.digest(p) for p in sorted(PARENT_CODE.glob("*.py"))},
             "agentfold": {p.name: common.digest(p) for p in sorted((up / "harness_factory" / "harnesses" / "agentfold").glob("*")) if p.is_file()},
             "adapter": common.digest(up / "benchmark" / "adapter" / "deepplanning.py"),
             "bench_yaml": common.digest(up / "benchmark" / "config" / "deepplanning_shopping.yaml"),
             "cohort": common.digest(common.ROOT / "cohort.json"), "protocol": common.digest(common.ROOT / "PROTOCOL.md"),
             "splits_v1": common.digest(CODE_FINE / "splits_v1.json"),
             "units": [list(u) for u in U.UNITS], "upstream_commit": common.UPSTREAM_COMMIT,
             "settings": {"model": common.MODEL_NAME, "thinking": False, "temperature": 0.0, "max_steps": common.MAX_STEPS,
                          "max_tokens_exec": common.EXEC_MAX_TOKENS, "context": common.MAX_MODEL_LEN}}
    files["sha256"] = common.digest_text(json.dumps(files, sort_keys=True))
    return files


def freeze_contract() -> dict:
    """The dispatcher writes contract.json once before any cell; a cell only compares (no concurrent writes)."""
    c = contract(); path = common.ROOT / "contract.json"
    if path.exists():
        if common.load_json(path) != c:
            raise SystemExit("Frozen contract changed; manual audit required")
    else:
        common.save(path, c)
    return c


def kernel_command(label: str, qid: str, base: str, run_dir: Path) -> list:
    return [common.PYTHON, str(CODE_FINE / "run_seed_variant.py"), "--bench", "shopping", "--harness", "agentfold",
            "--exec-model", common.MODEL_NAME, "--exec-base", base, "--exec-key", os.environ.get("OPENAI_API_KEY", "EMPTY"),
            "--max-steps", str(common.MAX_STEPS), "--workers", "1", "--attempts", "1",
            "--cases", qid, "--dataset-path", str(common.DATASET), "--output", str(run_dir)]


RC.kernel_command = kernel_command            # run_kernel_cell looks the name up at call time


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=int, required=True); p.add_argument("--arm", required=True)
    p.add_argument("--base", required=True)
    a = p.parse_args()
    cohort = common.load_cohort(); task = RC.load_task(cohort, a.index)
    assert a.arm in task["arms"], (a.arm, task["arms"][:3])
    arm, trial = U.split_label(a.arm)
    off = U.off_set(arm)
    ctr = freeze_contract()
    qid = task["question_id"]
    cell = common.ROOT / "cells" / qid / a.arm
    canonical = cell / "canonical.json"
    if canonical.exists():
        existing = common.load_json(canonical)
        assert existing.get("contract_sha256") == ctr["sha256"], "canonical cell from another contract"
        print(json.dumps({"cell": qid, "arm": a.arm, "status": "preserved"}), flush=True)
        return 0
    attempt = cell / "attempts" / str(time.time_ns()); attempt.mkdir(parents=True)
    item = RC.find_item(qid); RC.verify_hashes(task, item)
    common.save(attempt / "cell.json", {"question_id": qid, "arm": a.arm, "off": sorted(off), "trial": trial,
                                        "index": a.index, "base": a.base, "started": time.time(), "fake_model": common.FAKE_MODEL})
    os.environ["JITFINE_OFF"] = ",".join(sorted(off))
    os.environ["JIT_PARENT_CODE"] = str(PARENT_CODE)
    task_t = dict(task, seed=int(task["seed"]) + 1000 * trial)
    _finish = RC.finish

    def finish(cell_, attempt_, output):             # stamp the contract + the realised variant into the canonical cell
        rz = attempt_ / "realization_fine.json"
        output.update(contract_sha256=ctr["sha256"], trial=trial, base_arm=arm, off_units=sorted(off),
                      n_on=len(U.UIDS) - len(off), base_url=a.base,
                      finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      realization={k: v for k, v in (common.load_json(rz) if rz.is_file() else {}).items()
                                   if k in ("off", "code_patched", "system_prompt_sha256", "step_prompt_sha256", "execute_code_tool")})
        return _finish(cell_, attempt_, output)

    RC.finish = finish
    return RC.run_kernel_cell(task_t, a.arm, a.base, cell, attempt, item)


if __name__ == "__main__":
    raise SystemExit(main())
