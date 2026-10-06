"""Canonical cell scoring, token accounting, and infrastructure-failure handling."""
from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import common  # noqa: E402

common.ensure_upstream_on_path()


def log(message: str) -> None:
    print(json.dumps(message) if not isinstance(message, str) else message, flush=True)


def load_task(cohort: dict, index: int) -> dict:
    task = cohort["tasks"][index]
    assert task["index"] == index, (task["index"], index)
    return task


def find_item(qid: str):
    for item in common.load_items():
        if item["question_id"] == qid:
            return item
    raise SystemExit(f"question_id {qid} not in dataset {common.DATASET}")


def verify_hashes(task: dict, item: dict) -> None:
    db = Path(item["_db_dir"])
    for name, value in task["data_hashes"].items():
        assert common.digest(db / name) == value, f"dataset hash changed: {task['question_id']}/{name}"


def invalid(cell: Path, attempt: Path, category: str, reason: str, code: int) -> int:
    marker = {"category": category, "reason": reason[:4000], "attempt": str(attempt), "time": time.time()}
    common.save(attempt / "invalid.json", marker)
    common.save(cell / f"invalid_{time.time_ns()}.json", marker)
    log(json.dumps({"cell": cell.parent.name, "arm": cell.name, "status": category, "reason": reason[:300]}))
    return code


def telemetry_summary(path: Path) -> dict:
    events = common.read_jsonl(path)
    calls = [e for e in events if e.get("event") == "model_call"]
    by_role = collections.defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0})
    for c in calls:
        r = by_role[c.get("role", "unknown")]
        r["calls"] += 1
        r["input_tokens"] += int(c.get("input_tokens") or 0)
        r["output_tokens"] += int(c.get("output_tokens") or 0)
    # Split events by validation round (exec runs); the last round is the scored one.
    rounds = []
    current = None
    for e in events:
        if e.get("event") == "validation_start":
            current = {"events": []}
            rounds.append(current)
        elif current is not None:
            current["events"].append(e)
    last = rounds[-1]["events"] if rounds else []
    last_calls = [e for e in last if e.get("event") == "model_call"]
    return {
        "n_events": len(events),
        "by_role": dict(by_role),
        "all_input_tokens": sum(int(c.get("input_tokens") or 0) for c in calls),
        "all_output_tokens": sum(int(c.get("output_tokens") or 0) for c in calls),
        "all_llm_calls": len(calls),
        "thinking_off_all_calls": all(bool(c.get("thinking_off")) for c in calls) if calls else True,
        "reasoning_present_calls": sum(1 for e in events if e.get("event") == "reasoning_present"),
        "output_truncation_calls": sum(1 for c in calls if c.get("finish_reason") == "length"),
        "validation_rounds": len(rounds),
        "last_round": {
            "llm_calls": len(last_calls),
            "input_tokens": sum(int(c.get("input_tokens") or 0) for c in last_calls),
            "output_tokens": sum(int(c.get("output_tokens") or 0) for c in last_calls),
            "activations": dict(collections.Counter(e["key"] for e in last if e.get("event") == "activation")),
            "termination_events": [e.get("reason") for e in last if e.get("event") == "termination"],
            "sandbox_errors": sum(1 for e in last if e.get("event") == "sandbox_error"),
        },
        "activations_total": dict(collections.Counter(e["key"] for e in events if e.get("event") == "activation")),
        "context_exhausted_events": sum(1 for e in events if e.get("event") == "context_exhausted"),
        "model_errors": [e.get("error", "")[:200] for e in events if e.get("event") == "model_error"][-5:],
    }


def kernel_termination(record: dict, report: dict, tsum: dict) -> str:
    last = tsum["last_round"]
    if last["termination_events"]:
        return last["termination_events"][-1]
    if record.get("run_error") or record.get("validation_error"):
        return "error"
    records = report.get("validation_records") or []
    steps = (records[-1].get("trajectories") or []) if records else []
    if steps and any(tc.get("name") == "final_answer" for tc in (steps[-1].get("tool_calls") or [])):
        return "final_answer"
    if last["activations"].get("forced_finalization"):
        return "max_steps_forced_finalization"
    if steps and steps[-1].get("action_reasoning", "").startswith("Forced final answer"):
        return "max_steps_forced_finalization"
    return "max_steps" if steps else "no_steps"


# --------------------------------------------------------------------------- #
# Kernel arms
# --------------------------------------------------------------------------- #


def run_kernel_cell(task: dict, arm: str, base: str, cell: Path, attempt: Path, item: dict) -> int:
    qid = task["question_id"]
    run_dir = cell / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    env = common.child_env(attempt, task["seed"])
    cmd = kernel_command(arm, qid, base, run_dir)
    common.save(attempt / "command.json", {"cmd": ["<redacted>" if i and cmd[i-1].endswith("-key") else v for i, v in enumerate(cmd)], "cwd": str(common.UPSTREAM), "env_keys": sorted(k for k in env if k.startswith("JIT_"))})
    started = time.time()
    with (attempt / "run.log").open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=str(common.UPSTREAM), env=env, stdout=fh, stderr=subprocess.STDOUT)
    seconds = time.time() - started
    log_text = (attempt / "run.log").read_text(encoding="utf-8", errors="replace")
    tail = log_text[-3000:]
    scores_path = run_dir / "scores.jsonl"
    record = next((r for r in common.read_jsonl(scores_path) if r.get("question_id") == qid), None)
    tsum = telemetry_summary(attempt / "telemetry.jsonl")

    if proc.returncode != 0 or record is None:
        reason = f"exit={proc.returncode} record={'present' if record else 'missing'}: " + tail
        if not common.endpoint_healthy(base):
            reason = "endpoint_unhealthy " + reason
        category = "infrastructure" if common.classify_error(reason) != "behavioural" else "runtime_unclassified"
        return invalid(cell, attempt, category, reason, 75 if category == "infrastructure" else 76)

    error_text = f"{record.get('run_error', '')} {record.get('validation_error', '')}".strip()
    kind = common.classify_error(error_text)
    if kind == "infra":
        return invalid(cell, attempt, "infrastructure", error_text, 75)
    if not tsum["thinking_off_all_calls"]:
        return invalid(cell, attempt, "contract_violation", "a request was sent without enable_thinking=false", 76)

    report = {}
    try:
        report = common.load_json(record["report_path"])
    except Exception:  # noqa: BLE001
        pass
    evaluation = record.get("evaluation_result") or {}
    score = {k: evaluation.get(k) for k in ("score", "case_score", "match_rate", "matched_products", "expected_products",
                                            "matched_coupons", "expected_coupons", "extra_in_cart", "cart_path", "error")
             if k in evaluation}
    score.setdefault("score", 0.0)
    score.setdefault("case_score", 0.0)
    score.setdefault("match_rate", 0.0)
    if score.get("error"):
        return invalid(cell, attempt, "scoring_infrastructure", str(score["error"]), 75)
    termination = kernel_termination(record, report, tsum)
    tool_rows = common.read_jsonl(attempt / "tools.jsonl")
    output = base_output(task, arm, attempt, score, termination, tsum, seconds)
    output.update({
        "steps_used": int(record.get("steps_used") or 0),
        "exec_tokens_final_round_upstream": {"input": int(record.get("input_token_count") or 0),
                                            "output": int(record.get("output_token_count") or 0)},
        "tool_calls": len(tool_rows),
        "tool_counts": dict(collections.Counter(t["name"] for t in tool_rows)),
        "tool_errors": sum(1 for t in tool_rows if t.get("error")),
        "errors": {"run_error": str(record.get("run_error", ""))[:1000], "validation_error": str(record.get("validation_error", ""))[:1000]},
        "rounds": int(record.get("rounds") or 0),
        "report_path": record.get("report_path"),
    })
    return finish(cell, attempt, output)


def base_output(task: dict, arm: str, attempt: Path, score: dict, termination: str, tsum: dict, seconds: float) -> dict:
    return {
        "question_id": task["question_id"], "level": task["level"], "index": task["index"], "arm": arm,
        "valid": True, "attempt": str(attempt.relative_to(common.ROOT)), "cohort_sha256": common.cohort_sha256(),
        "seed": task["seed"], "score": score, "termination": termination, "seconds": seconds,
        "llm_calls": tsum["all_llm_calls"], "llm_calls_by_role": tsum["by_role"],
        "input_tokens": tsum["all_input_tokens"], "output_tokens": tsum["all_output_tokens"],
        "total_tokens": tsum["all_input_tokens"] + tsum["all_output_tokens"],
        "exec_tokens": {"input": tsum["last_round"]["input_tokens"], "output": tsum["last_round"]["output_tokens"],
                        "llm_calls": tsum["last_round"]["llm_calls"]},
        "activations": tsum["last_round"]["activations"], "activations_total": tsum["activations_total"],
        "validation_rounds": tsum["validation_rounds"],
        "output_truncation_calls": tsum["output_truncation_calls"],
        "reasoning_present_calls": tsum["reasoning_present_calls"],
        "thinking_off_all_calls": tsum["thinking_off_all_calls"],
        "context_exhausted_events": tsum["context_exhausted_events"],
        "sandbox_errors": tsum["last_round"]["sandbox_errors"],
    }


def finish(cell: Path, attempt: Path, output: dict) -> int:
    common.save(attempt / "result.json", output)
    common.save(cell / "canonical.json", output)
    log(json.dumps({"cell": output["question_id"], "arm": output["arm"], "status": "valid",
                    "match_rate": output["score"]["match_rate"], "case_score": output["score"]["case_score"],
                    "termination": output["termination"], "tokens": output["total_tokens"], "llm_calls": output["llm_calls"]}))
    return 0
