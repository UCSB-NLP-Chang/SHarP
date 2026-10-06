"""Run upstream's seed-harness entry point on a harness VARIANT (units off), upstream tree untouched.

    JITFINE_OFF=uid,uid,...  run_seed_variant.py <args for scripts.run_seed_harness>

Install jit_patches first (thinking off, telemetry,
official tools, bwrap sandbox, activation counters), then
  * the variant harness is built under $JIT_ATTEMPT_DIR/harness_root/agentfold and the seed runner's
    HARNESS_ROOT is pointed at it (it installs that directory verbatim as a preset harness);
  * DeepPlanningShoppingAdapter.format_task is wrapped so the task text is cut at its own boundaries;
  * with t_execute_code off, a copy of the benchmark YAML without `execute_code` is passed via --config.
Every realised prompt / task text is written to $JIT_ATTEMPT_DIR/realization_fine.json(l) for audit.
"""
from __future__ import annotations
import hashlib, json, os, sys
from pathlib import Path

CODE_FINE = Path(__file__).resolve().parent
PARENT_CODE = Path(os.environ["JIT_PARENT_CODE"])
for p in (str(PARENT_CODE), str(CODE_FINE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import common  # noqa: E402

common.ensure_upstream_on_path()
import jit_patches  # noqa: E402,F401
import units as U  # noqa: E402
import variant as V  # noqa: E402


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    off = frozenset(u for u in os.environ.get("JITFINE_OFF", "").split(",") if u)
    assert off <= set(U.UIDS), sorted(off - set(U.UIDS))
    attempt = Path(os.environ["JIT_ATTEMPT_DIR"])
    pristine = common.UPSTREAM / "harness_factory" / "harnesses" / V.HARNESS
    audit = V.build_harness(off, pristine, attempt / "harness_root")

    from benchmark.adapter import deepplanning as D
    orig_format = D.DeepPlanningShoppingAdapter.format_task

    def format_task(self, item):
        task = orig_format(self, item)
        s1 = D.SHOPPING_PROMPTS.get(item["_level"], D.SHOPPING_SYSTEM_PROMPT_L1)
        cut = V.cut_task(task, s1, off)
        with (attempt / "realization_fine.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"question_id": item.get("question_id"), "level": item["_level"],
                                 "task_sha256_upstream": sha(task), "task_sha256": sha(cut), "task_text": cut}, ensure_ascii=False) + "\n")
        return cut

    D.DeepPlanningShoppingAdapter.format_task = format_task

    args = list(sys.argv[1:])
    if "t_execute_code" in off:
        cfg = V.config_without_execute_code(common.UPSTREAM / "benchmark" / "config" / "deepplanning_shopping.yaml",
                                            attempt / "deepplanning_shopping.no_execute_code.yaml")
        args += ["--config", str(cfg)]
        if common.FAKE_MODEL:       # the scripted dry-run model must not ask for a tool this arm does not have
            import fake_model
            _orig = fake_model._executed_tools
            fake_model._executed_tools = lambda: _orig() | {"execute_code"}

    (attempt / "realization_fine.json").write_text(json.dumps({
        "off": sorted(off), "n_on": len(U.UIDS) - len(off), "harness_dir": audit["harness_dir"],
        "code_patched": audit["code_patched"], "prompt_yaml_rewritten": audit["prompt_yaml_rewritten"],
        "system_prompt_sha256": sha(audit["system_prompt"]), "step_prompt_sha256": sha(audit["step_prompt"]),
        "system_prompt": audit["system_prompt"], "step_prompt": audit["step_prompt"],
        "harness_file_sha256": {f: hashlib.sha256((Path(audit["harness_dir"]) / f).read_bytes()).hexdigest()
                                for f in V.HARNESS_FILES if (Path(audit["harness_dir"]) / f).is_file()},
        "execute_code_tool": "t_execute_code" not in off, "args": ["<redacted>" if i and args[i-1] == "--exec-key" else v for i, v in enumerate(args)]}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    import scripts.run_seed_harness as seed
    seed.HARNESS_ROOT = attempt / "harness_root"
    sys.argv = ["scripts.run_seed_harness"] + args
    os.chdir(common.UPSTREAM)
    print(f"[run_seed_variant] off={sorted(off)} patches={jit_patches.PATCHES_APPLIED} fake_model={common.FAKE_MODEL}", file=sys.stderr, flush=True)
    return int(seed.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
