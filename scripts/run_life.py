#!/usr/bin/env python3
"""Run the frozen LIFE screen or validation cells through a local entry point.

--stub executes gold-action replay through the real tools and evaluator. It tests
software integration only and never estimates model performance.
"""
from __future__ import annotations
import argparse
import csv
import fcntl
import importlib
import json
import os
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=["airline", "retail"], required=True)
    p.add_argument("--phase", choices=["screen", "validation", "all-pruned", "performance"], default="screen")
    p.add_argument("--upstream", type=Path, default=REPO / "external/life/TauBench")
    p.add_argument("--output", type=Path)
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "Qwen3.5-122B-A10B"))
    p.add_argument("--arm", help="One arm label; default: all frozen arms")
    p.add_argument("--limit-tasks", type=int, help="Positive task limit for an integration run")
    p.add_argument("--stub", action="store_true")
    a = p.parse_args()
    if a.phase == "performance" and a.domain == "airline":
        p.error("Airline performance arms are included in --phase validation (A-prefixed labels)")
    if a.limit_tasks is not None and a.limit_tasks <= 0:
        p.error("--limit-tasks must be positive")
    if not a.stub and not a.base_url:
        p.error("Set OPENAI_BASE_URL or --base-url for model inference")
    upstream = a.upstream.resolve()
    if not (upstream / "src/tau2/harness/skills.py").is_file():
        p.error("Expected the pinned LIFE-harness TauBench checkout at --upstream")
    source = REPO / "experiments/life" / a.domain / a.phase
    common = REPO / "experiments/life/common"
    out = (a.output or REPO / "outputs/life" / a.domain / a.phase / ("stub" if a.stub else "model")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = {"domain": a.domain, "phase": a.phase, "stub": a.stub,
                "model": "stub" if a.stub else a.model}
    marker = out / "run_identity.json"
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous != identity:
            differences = [f"{key}: {previous.get(key)!r} -> {identity.get(key)!r}"
                           for key in sorted(previous.keys() | identity.keys())
                           if previous.get(key) != identity.get(key)]
            p.error("Output directory identity differs (" + "; ".join(differences)
                    + "); choose a new directory")
    marker.write_text(json.dumps(identity, indent=2) + "\n")
    os.environ.update(LIFE_ROOT=str(out), LIFE_FINE_ROOT=str(out), LIFE_LADDER_ROOT=str(out),
                      LIFE_UPSTREAM=str(upstream), TAU2_DATA_DIR=str(upstream / "data"),
                      LITELLM_LOCAL_MODEL_COST_MAP="True")
    shutil.copyfile(source / "PROTOCOL.md", out / "PROTOCOL.md")
    sys.path[:0] = [str(source), str(common), str(upstream / "src")]
    runner = importlib.import_module("fine" if a.phase == "screen" else "ladder")
    cohort = runner.cohort()
    tasks = cohort["tasks"][:a.limit_tasks] if a.limit_tasks else cohort["tasks"]
    if a.arm and not any(a.arm in task["arm_order"] for task in tasks):
        p.error("Unknown arm label for this frozen cohort")
    rows, failures = [], []
    for entry in tasks:
        for arm in entry["arm_order"]:
            if a.arm and arm != a.arm:
                continue
            factory = None
            if a.stub:
                from smoke import StubCompletion
                task = runner.C.get_task(a.domain, entry["task_id"])
                factory = lambda directory, seed, task=task: StubCompletion(directory, seed, task)
            result = runner.run(a.base_url or "stub://none", "stub" if a.stub else a.model,
                                entry["task_id"], arm, transport_factory=factory, out_root=out)
            print(f"{a.domain}/{entry['task_id']}/{arm}: {result['status']}", flush=True)
            if result["status"] != "valid":
                failures.append({"task": entry["task_id"], "arm": arm, "status": result["status"]})
            elif a.phase == "screen" and arm != "full":
                unit = runner.U.unit_of(arm)
                rows.append({"task_id": entry["task_id"], "component": unit.target,
                             "score": float(result["reward"] == 1.0), "tokens": result["tokens"]})
    if a.phase == "screen" and not a.stub:
        with (out / "single_off.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["task_id", "component", "score", "tokens"])
            writer.writeheader(); writer.writerows(rows)
        (out / "components.json").write_text(json.dumps([u.target for u in runner.U.UNITS], indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps({**identity, "tasks": len(tasks), "failures": failures}, indent=2) + "\n")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
