"""Run one (domain, task, arm) cell through the unmodified upstream Life/tau2 runtime.

Arms:
  full      = upstream LLMAgent + Life harness exactly as run_model.sh
              (airline: --enabled --h2 --h3 --h4 --h5 --h5-top-k 1; retail: --nl --enabled --h2 --h3 --h4 --h5)
  baseline  = upstream LLMAgent, harness disabled (run_model.sh *-baseline)
  reference = constitutive native-tool-call loop (reference.py), policy text as task rules, no H2-H5
Scoring: upstream evaluate_simulation; airline EvaluationType.ALL, retail ALL_WITH_NL_ASSERTIONS (run_model.sh --nl).
The retail natural-language judge uses the configured endpoint; its tokens are recorded separately.
Exit codes: 0 valid (scored), 75 infra-invalid (never scored, cell stays missing), 3 unscored (needs audit).
"""
from __future__ import annotations
import argparse, copy, importlib, json, os, sys, time, traceback
from pathlib import Path
from loguru import logger
logger.remove()  # omit benchmark text from console logs
from pydantic import ValidationError
from runtime import (ROOT, CODE, RecordedCompletion, InfraInvalid, BehavioralStop, save, load, digest, filehash,
                     MAX_TOKENS, MAX_MODEL_LEN)
from reference import factory
import tau2.config as tau2_config
import tau2.evaluator.evaluator_nl_assertions as nl_mod
from tau2.utils import llm_utils
from tau2.registry import registry
from tau2.runner import build
from tau2.data_model.simulation import TextRunConfig, TerminationReason
from tau2.environment.toolkit import ToolType
from tau2.evaluator.evaluator import evaluate_simulation, EvaluationType

ARMS = ["full", "baseline", "reference"]
DOMAINS = ["airline", "retail"]
JUDGE_MODEL = "openai/judge122b"
UPSTREAM = Path(os.environ.get("LIFE_UPSTREAM", str(ROOT / "upstream" / "TauBench")))

registry.register_agent_factory(factory, "constitutive_reference", metadata={"solo_mode": False})
ORIGINAL_BUILD_AGENT = build.build_agent
CURRENT_AUDIT: dict = {}


def domain_api(domain):
    m = importlib.import_module(f"tau2.domains.{domain}.environment")
    return m.get_environment, m.get_tasks


def get_task(domain, task_id):
    _, get_tasks = domain_api(domain)
    return next(t for t in get_tasks("test") if t.id == task_id)


def audited_build_agent(agent_name, environment, **kwargs):
    if kwargs.get("harness_h5"):
        assert isinstance(kwargs.get("harness_h5_query"), str), "oracle H5 query forbidden"
        CURRENT_AUDIT["h5_query_source"] = "actual_first_user_utterance"
        CURRENT_AUDIT["h5_query_sha256"] = digest(kwargs["harness_h5_query"])
    a = ORIGINAL_BUILD_AGENT(agent_name, environment, **kwargs)
    original = [t.name for t in a.tools]
    if agent_name == "constitutive_reference":
        a.tools = [t for t in a.tools if environment.tools.tool_type(t.name) != ToolType.THINK]
    CURRENT_AUDIT.update(agent_class=type(a).__name__, environment_toolkit=type(environment.tools).__name__,
                         tools=[t.name for t in a.tools],
                         removed_tools=[n for n in original if n not in [t.name for t in a.tools]],
                         tool_schema_sha256=digest([t.openai_schema for t in a.tools]),
                         system_prompt_sha256=digest(a.system_prompt), policy_sha256=digest(environment.get_policy()),
                         h5_skill_block_present="## Task-Specific Guidance" in a.system_prompt)
    return a


build.build_agent = audited_build_agent


def harness_env_kwargs(arm):
    full = arm == "full"
    return {"harness_enabled": full, "harness_h3": full, "harness_h4": full}


def config_for(domain, arm, seed):
    full = arm == "full"
    llm_args = {"temperature": 0, "max_tokens": MAX_TOKENS, "num_retries": 0, "seed": seed}
    return TextRunConfig(
        domain=domain, agent="llm_agent" if arm != "reference" else "constitutive_reference",
        llm_agent="openai/agent122b", llm_user="openai/user122b",
        llm_args_agent=copy.deepcopy(llm_args), llm_args_user=copy.deepcopy(llm_args), seed=seed,
        max_steps=200, max_errors=10, timeout=None,
        harness_enabled=full, harness_h3=full, harness_h4=full, harness_h5=full,
        harness_h5_rag=False, harness_h5_rag_top_k=(1 if domain == "airline" else 3), harness_h5_rag_tool=False,
        enforce_communication_protocol=False)


def eval_type(domain):
    return EvaluationType.ALL_WITH_NL_ASSERTIONS if domain == "retail" else EvaluationType.ALL


def set_judge(seed):
    tau2_config.DEFAULT_LLM_NL_ASSERTIONS = JUDGE_MODEL
    nl_mod.DEFAULT_LLM_NL_ASSERTIONS = JUDGE_MODEL
    args = tau2_config.DEFAULT_LLM_NL_ASSERTIONS_ARGS  # same dict object imported by nl_mod
    args.update({"temperature": 0, "seed": seed, "max_tokens": MAX_TOKENS, "num_retries": 0})
    assert nl_mod.DEFAULT_LLM_NL_ASSERTIONS_ARGS is args
    return dict(args)


def cell_key(domain, task_id, arm):
    return f"{domain}_{task_id}_{arm}"


def contract():
    cohort = load(CODE / "cohort.json")
    c = {"cohort": filehash(CODE / "cohort.json"), "protocol": filehash(CODE.parent / "PROTOCOL.md"),
         "code": {p.name: filehash(p) for p in sorted(CODE.glob("*.py")) if p.name not in {"aggregate.py", "checks.py"}},
         "upstream_src": {str(p.relative_to(UPSTREAM)): filehash(p)
                          for p in sorted((UPSTREAM / "src").rglob("*.py")) if "__pycache__" not in p.parts},
         "data": {k: filehash(UPSTREAM / "data" / "tau2" / "domains" / k) for k in cohort["data_sha256"]},
         "settings": {"served_model": "Qwen3.5-122B-A10B", "thinking": False, "temperature": 0, "max_tokens": MAX_TOKENS,
                      "max_model_len": MAX_MODEL_LEN, "max_steps": 200, "max_errors": 10, "timeout": None,
                      "h5_top_k": {"airline": 1, "retail": 3}, "retail_nl_judge": JUDGE_MODEL}}
    assert c["data"] == cohort["data_sha256"], "upstream data changed since cohort freeze"
    c["sha256"] = digest(c)
    return c


def freeze(out_root):
    c = contract(); out = Path(out_root) / "contract.json"
    if out.exists():
        old = load(out)
        if old != c:
            if list((Path(out_root) / "cells").glob("*.json")):
                raise SystemExit("Frozen contract changed after cells exist; manual audit required")
            save(out, c)
    else:
        save(out, c)
    return c


def finalize_after_stop(orch, rec, reason):
    if orch is None or orch._run_start_perf is None:
        return None
    orch.done = True
    if reason == "context_exhaustion":
        orch.termination_reason = TerminationReason.CONTEXT_WINDOW_EXCEEDED
    else:
        orch.termination_reason = TerminationReason.USER_ERROR if rec.last_role == "user" else TerminationReason.AGENT_ERROR
    return orch._finalize()


def score(simulation, task, domain, arm, folder, orch=None):
    """Upstream evaluator (+ retail NL judge). Returns (reward, replay_ok)."""
    reward = evaluate_simulation(simulation, task, eval_type(domain), False, domain, env_kwargs=harness_env_kwargs(arm))
    simulation.reward_info = reward
    save(folder / "simulation.json", simulation.model_dump(mode="json"))
    replay_ok = None
    if orch is not None and simulation.termination_reason in {TerminationReason.AGENT_STOP, TerminationReason.USER_STOP}:
        try:
            get_environment, _ = domain_api(domain)
            replay = get_environment(**harness_env_kwargs(arm)); s = task.initial_state
            replay.set_state(s.initialization_data if s else None, s.initialization_actions if s else None, simulation.messages)
            replay_ok = (replay.get_db_hash() == orch.environment.get_db_hash()
                         and replay.get_user_db_hash() == orch.environment.get_user_db_hash())
        except Exception as e:  # flag only; the official verdict stands
            replay_ok = False; save(folder / "replay_error.json", {"type": type(e).__name__, "message": str(e)[:1000]})
    return reward.reward, replay_ok


def run_cell(base, served, domain, task_id, arm, transport_factory=None, out_root=None, enforce_contract=True):
    out_root = Path(out_root or ROOT); key = cell_key(domain, task_id, arm)
    ctr = freeze(out_root) if enforce_contract else contract()
    cohort = load(CODE / "cohort.json"); spec = next(t for t in cohort["tasks"] if t["domain"] == domain and t["task_id"] == task_id)
    marker = out_root / "cells" / f"{key}.json"
    if marker.exists():
        existing = load(marker); assert existing["contract_sha256"] == ctr["sha256"]; return existing
    task = get_task(domain, task_id)
    # Re-score path: a previous attempt finished inference but the retail judge failed on infra -> no new inference.
    prior = sorted((out_root / "attempts" / key).glob("*/result.json"))
    if prior:
        last = load(prior[-1])
        if last.get("status") == "judge_infra_invalid" and (prior[-1].parent / "simulation.json").exists():
            return rescore_cell(base, served, domain, task, arm, spec, prior[-1].parent, ctr, transport_factory, out_root)
    folder = out_root / "attempts" / key / str(time.time_ns()); folder.mkdir(parents=True)
    CURRENT_AUDIT.clear()
    rec = transport_factory(folder / "api", spec["seed"]) if transport_factory else RecordedCompletion(base, served, folder / "api", spec["seed"])
    llm_utils.completion = rec
    judge_args = set_judge(spec["seed"])
    cfg = config_for(domain, arm, spec["seed"])
    save(folder / "config.json", {"run_config": cfg.model_dump(mode="json"), "eval_type": eval_type(domain).value,
                                  "env_kwargs": harness_env_kwargs(arm), "judge": {"model": JUDGE_MODEL, "args": judge_args},
                                  "base_url": base, "served_model": served})
    status, reason, simulation, orch, reward, replay_ok = "valid", None, None, None, None, None
    try:
        orch = build.build_text_orchestrator(cfg, copy.deepcopy(task), seed=spec["seed"])
        save(folder / "realized.json", CURRENT_AUDIT)
        save(folder / "resolved_prompt.json", {"system": orch.agent.system_prompt, "tools": [t.openai_schema for t in orch.agent.tools],
                                               "user_system": orch.user.system_prompt})
        simulation = orch.run()
    except BehavioralStop as e:
        reason = str(e); simulation = finalize_after_stop(orch, rec, reason); reward = 0.0
    except (json.JSONDecodeError, ValidationError) as e:
        reason = "model_output_format_failure"; save(folder / "behavior_error.json", {"type": type(e).__name__, "trace": traceback.format_exc()})
        simulation = finalize_after_stop(orch, rec, reason); reward = 0.0
    except ValueError as e:
        if "must have either content or tool_calls" in str(e):
            reason = "empty_message"; save(folder / "behavior_error.json", {"type": type(e).__name__, "message": str(e)[:500]})
            simulation = finalize_after_stop(orch, rec, reason); reward = 0.0
        else:
            status, reason = "runtime_unscored", type(e).__name__
            save(folder / "runtime_error.json", {"type": type(e).__name__, "trace": traceback.format_exc()})
    except InfraInvalid as e:
        status, reason = "infra_invalid", str(e)
    except BaseException as e:
        status, reason = "runtime_unscored", type(e).__name__
        save(folder / "runtime_error.json", {"type": type(e).__name__, "trace": traceback.format_exc()})
    if orch is not None:
        save(folder / "partial_trajectory.json", [x.model_dump(mode="json") for x in getattr(orch, "trajectory", [])])
        save(folder / "final_state.json", {"agent_db": orch.environment.get_db_hash(), "user_db": orch.environment.get_user_db_hash()})
    if simulation is not None:
        save(folder / "simulation.json", simulation.model_dump(mode="json"))
        if reward is None and status == "valid":
            try:
                reward, replay_ok = score(simulation, task, domain, arm, folder, orch)
            except InfraInvalid as e:
                status, reason = "judge_infra_invalid", str(e)
            except BaseException as e:
                status, reason = "scoring_unscored", type(e).__name__
                save(folder / "scoring_error.json", {"type": type(e).__name__, "trace": traceback.format_exc()})
    return write_result(folder, out_root, key, domain, task_id, arm, spec, ctr, status, reason, reward, simulation, orch, rec,
                        replay_ok, base)


def rescore_cell(base, served, domain, task, arm, spec, folder, ctr, transport_factory, out_root):
    from tau2.data_model.simulation import SimulationRun
    key = cell_key(domain, task.id, arm)
    simulation = SimulationRun.model_validate(load(folder / "simulation.json"))
    rec = transport_factory(folder / "api_rescore", spec["seed"]) if transport_factory else RecordedCompletion(base, served, folder / "api_rescore", spec["seed"])
    llm_utils.completion = rec; set_judge(spec["seed"])
    status, reason, reward, replay_ok = "valid", "rescored_after_judge_infra", None, None
    try:
        reward, replay_ok = score(simulation, task, domain, arm, folder)
    except InfraInvalid as e:
        status, reason = "judge_infra_invalid", str(e)
    except BaseException as e:
        status, reason = "scoring_unscored", type(e).__name__
        save(folder / "scoring_error.json", {"type": type(e).__name__, "trace": traceback.format_exc()})
    prev = load(folder / "result.json")
    prev_usage = {k: prev.get(k) for k in ["llm_calls", "tokens", "input_tokens", "output_tokens", "agent_tokens", "user_tokens",
                                             "agent_input_tokens", "agent_output_tokens", "user_input_tokens", "user_output_tokens",
                                             "agent_llm_calls", "user_llm_calls", "output_truncated_calls", "reasoning_calls",
                                             "token_count_mismatches", "max_input_tokens"]}
    return write_result(folder, out_root, key, domain, task.id, arm, spec, ctr, status, reason, reward, simulation, None, rec,
                        replay_ok, base, usage_override={**prev_usage, "judge_tokens": rec.summary()["judge_tokens"],
                                                         "judge_llm_calls": rec.summary()["judge_llm_calls"]})


def write_result(folder, out_root, key, domain, task_id, arm, spec, ctr, status, reason, reward, simulation, orch, rec, replay_ok,
                 base, usage_override=None):
    trajectory = simulation.messages if simulation is not None else (getattr(orch, "trajectory", []) if orch else [])
    calls = [tc for msg in trajectory for tc in (getattr(msg, "tool_calls", None) or [])]
    tool_errors = sum(1 for m in trajectory if getattr(m, "role", None) == "tool" and getattr(m, "error", False))
    termination = simulation.termination_reason.value if simulation is not None else reason
    realized = load(folder / "realized.json") if (folder / "realized.json").exists() else {}
    result = {"cell": key, "domain": domain, "task_id": task_id, "index": spec["index"], "arm": arm, "task_key": spec["key"],
              "seed": spec["seed"], "contract_sha256": ctr["sha256"], "attempt": str(folder.relative_to(out_root)),
              "status": status, "behavior_or_error": reason, "reward": reward, "termination": termination,
              "reward_breakdown": ({str(getattr(k, "value", k)): v for k, v in (simulation.reward_info.reward_breakdown or {}).items()}
                                   if simulation is not None and simulation.reward_info else None),
              "n_messages": len(trajectory), "tool_calls": len(calls),
              "agent_tool_calls": sum(tc.requestor == "assistant" for tc in calls),
              "user_tool_calls": sum(tc.requestor == "user" for tc in calls), "tool_errors": tool_errors,
              "replay_matches_runtime": replay_ok, "base_url": base, "realized": realized,
              **(usage_override or rec.summary()),
              "artifact_hashes": {str(p.relative_to(folder)): filehash(p) for p in sorted(folder.rglob("*.json")) if p.name != "result.json"},
              "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    save(folder / "result.json", result)
    if status == "valid":
        assert reward is not None; save(out_root / "cells" / f"{key}.json", result)
    print(json.dumps({k: result[k] for k in ["cell", "status", "behavior_or_error", "reward", "termination", "tokens", "llm_calls"]}), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True); p.add_argument("--served", default="Qwen3.5-122B-A10B")
    p.add_argument("--domain", choices=DOMAINS, required=True); p.add_argument("--task", required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    a = p.parse_args()
    r = run_cell(a.base, a.served, a.domain, a.task, a.arm)
    raise SystemExit(0 if r["status"] == "valid" else 75 if r["status"] in {"infra_invalid", "judge_infra_invalid"} else 3)
