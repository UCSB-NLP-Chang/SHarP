"""Preflight for LIFE/tau2.

Offline (CPU, no endpoint):  cohort re-derivation, data hashes vs official tau2, H5 provenance,
  gold-action replay on every cohort task in BOTH harness modes (native vs H2+H3+H4 toolkit) with
  identical final DB hashes, and a synthetic gold trajectory scored 1.0 by the upstream evaluator
  (airline: ALL; retail: ENV x COMMUNICATE offline, NL judge needs the endpoint), agent-build audit.
Live (--base): arithmetic canary (no reasoning_content), native tool-call canary through the
  LLMAgent path (full/baseline) and the reference path, user-simulator init for both domains
  (real cohort task, first utterance only, unscored), retail NL-judge canary (json_object), and
  tokenizer-vs-server prompt-token agreement on the canary requests.
"""
from __future__ import annotations
import argparse, copy, json, sys, time, traceback
from pathlib import Path
from loguru import logger
logger.remove()
from runtime import ROOT, CODE, RecordedCompletion, InfraInvalid, BehavioralStop, save, load, digest, filehash
import cell as C
import cohort as CO
from reference import ReferenceAgent
from tau2.data_model.message import AssistantMessage, SystemMessage, ToolCall, ToolMessage, UserMessage
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.environment.tool import Tool
from tau2.evaluator.evaluator import evaluate_simulation, EvaluationType
from tau2.evaluator.evaluator_env import EnvironmentEvaluator
from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
from tau2.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator
from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE
from tau2.runner import build
from tau2.utils import llm_utils

UPSTREAM = C.UPSTREAM


def gold_simulation(domain, task, env_kwargs):
    """Synthetic trajectory that performs exactly the gold actions through env.get_response (the orchestrator path)."""
    get_environment, _ = C.domain_api(domain)
    env = get_environment(**env_kwargs); s = task.initial_state
    env.set_state(s.initialization_data if s else None, s.initialization_actions if s else None, [])
    msgs = [copy.deepcopy(DEFAULT_FIRST_AGENT_MESSAGE), UserMessage(role="user", content="Hello, I need some help with my account today.")]
    errors = []
    for i, a in enumerate(task.evaluation_criteria.actions or []):
        tc = ToolCall(id=f"gold_{i}", name=a.name, arguments=a.arguments, requestor=a.requestor)
        tm = env.get_response(tc)
        if tm.error:
            errors.append({"action": a.name, "content_sha256": digest(tm.content)})
        msgs.append(AssistantMessage(role="assistant", tool_calls=[tc]) if a.requestor == "assistant" else UserMessage(role="user", tool_calls=[tc]))
        msgs.append(tm)
    info = list(task.evaluation_criteria.communicate_info or [])
    msgs.append(AssistantMessage(role="assistant", content="Everything is completed. " + " ".join(info + [x.replace(",", "") for x in info])))
    msgs.append(UserMessage(role="user", content="###STOP###"))
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    sim = SimulationRun(id="gold", task_id=task.id, start_time=now, end_time=now, duration=0.0,
                        termination_reason=TerminationReason.USER_STOP, messages=msgs, seed=0)
    return sim, env, errors


def offline(out_dir):
    cohort = load(CODE / "cohort.json")
    rebuilt = CO.build(UPSTREAM / "data", UPSTREAM / "src" / "tau2" / "harness" / "skills.py")
    assert rebuilt == cohort, "cohort.json does not re-derive from the pinned data"
    assert cohort["data_matches_official"], "vendored data differs from official tau2-bench"
    checks = {"cohort_rederives": True, "data_matches_official": True, "n_tasks": cohort["n_tasks"], "n_cells": cohort["n_cells"],
              "domains": cohort["domains"], "gold": [], "agent_audit": {}}
    for spec in cohort["tasks"]:
        d, tid = spec["domain"], spec["task_id"]; task = C.get_task(d, tid)
        assert not task.initial_state or not task.initial_state.message_history
        basis = [x.value for x in task.evaluation_criteria.reward_basis]
        row = {"index": spec["index"], "domain": d, "task_id": tid, "reward_basis": basis,
               "n_actions": len(task.evaluation_criteria.actions or []), "n_communicate": len(task.evaluation_criteria.communicate_info or []),
               "n_nl_assertions": len(task.evaluation_criteria.nl_assertions or []), "modes": {}}
        hashes = {}
        for mode, kw in [("native", C.harness_env_kwargs("baseline")), ("harness", C.harness_env_kwargs("full"))]:
            sim, env, errors = gold_simulation(d, task, kw)
            hashes[mode] = (env.get_db_hash(), env.get_user_db_hash())
            env_r = EnvironmentEvaluator.calculate_reward(domain=d, task=task, full_trajectory=sim.messages, solo_mode=False, env_kwargs=dict(kw)).reward
            comm_r = CommunicateEvaluator.calculate_reward(task=task, full_trajectory=sim.messages).reward
            m = {"gold_action_errors": errors, "env_reward": env_r, "communicate_reward": comm_r}
            if d == "airline":
                m["all_reward"] = evaluate_simulation(sim, task, EvaluationType.ALL, False, d, env_kwargs=dict(kw)).reward
                m["offline_reward"] = m["all_reward"]
            else:
                m["offline_reward"] = env_r * comm_r  # NL_ASSERTION component needs the live judge
            row["modes"][mode] = m
        row["harness_gold_hash_equal"] = hashes["native"] == hashes["harness"]
        row["offline_reward_ok"] = all(m["offline_reward"] == 1.0 for m in row["modes"].values())
        checks["gold"].append(row)
    bad = [r for r in checks["gold"] if not (r["harness_gold_hash_equal"] and r["offline_reward_ok"])]
    checks["gold_all_ok"] = not bad; checks["gold_failures"] = [(r["domain"], r["task_id"]) for r in bad]
    # Agent-build audit on the first cohort task per domain (synthetic H5 query; never oracle metadata).
    for d in C.DOMAINS:
        task = C.get_task(d, next(t["task_id"] for t in cohort["tasks"] if t["domain"] == d))
        get_environment, _ = C.domain_api(d)
        native = get_environment(); full_env = get_environment(**C.harness_env_kwargs("full"))
        assert native.get_policy() == full_env.get_policy()
        args = C.config_for(d, "baseline", 1).llm_args_agent
        C.CURRENT_AUDIT.clear()
        b = C.audited_build_agent("llm_agent", native, llm="openai/agent122b", llm_args=copy.deepcopy(args), task=task); base_a = dict(C.CURRENT_AUDIT)
        C.CURRENT_AUDIT.clear()
        r = C.audited_build_agent("constitutive_reference", native, llm="openai/agent122b", llm_args=copy.deepcopy(args), task=task); ref_a = dict(C.CURRENT_AUDIT)
        C.CURRENT_AUDIT.clear()
        f = C.audited_build_agent("llm_agent", full_env, llm="openai/agent122b", llm_args=copy.deepcopy(args), task=task, harness_h5=True,
                                  harness_h5_query="Hello, I need some help with my account today.", harness_h5_rag_top_k=1 if d == "airline" else 3)
        full_a = dict(C.CURRENT_AUDIT)
        assert "think" not in [t.name for t in r.tools] and not ref_a["removed_tools"]
        assert full_a["tool_schema_sha256"] != base_a["tool_schema_sha256"], "H3 did not change tool schemas"
        assert base_a["tool_schema_sha256"] == ref_a["tool_schema_sha256"], "reference must expose the native tool interface"
        assert ref_a["system_prompt_sha256"] != base_a["system_prompt_sha256"]
        assert full_a["policy_sha256"] == base_a["policy_sha256"] == ref_a["policy_sha256"]
        checks["agent_audit"][d] = {"baseline": base_a, "reference": ref_a, "full": full_a,
                                    "same_policy_text": True, "policy_sha256": base_a["policy_sha256"]}
    checks["contract_sha256"] = C.contract()["sha256"]
    save(Path(out_dir) / "offline.json", checks)
    return checks


def read_value() -> str:
    """Read the value stored in this synthetic test environment."""
    return "17"


def live(base, served, out_dir):
    out = Path(out_dir); checks = {"base_url": base, "served": served, "canaries": {}}
    cohort = load(CODE / "cohort.json"); seed = cohort["seed_base"]

    def new_rec(name):
        rec = RecordedCompletion(base, served, out / name / str(time.time_ns()), seed); llm_utils.completion = rec; return rec

    # 1. arithmetic canary: thinking really off, tokenizer agreement.
    rec = new_rec("canary_arithmetic")
    r = rec("openai/agent122b", [{"role": "user", "content": "What is 2 plus 3? Respond with the single digit."}], temperature=0, max_tokens=16384, seed=seed)
    assert (r.choices[0].message.content or "").strip() == "5", "arithmetic canary failed"
    assert rec.calls[0]["reasoning_chars"] == 0 and rec.calls[0]["input_count_match"], rec.calls[0]
    checks["canaries"]["arithmetic"] = rec.summary()
    # 2. native tool-call canaries: LLMAgent path (full & baseline share it) and reference path.
    tool = Tool(read_value)
    for arm in ["llm_agent", "reference"]:
        rec = new_rec(f"canary_tool_{arm}")
        msg = [SystemMessage(role="system", content="Use the provided tool to read the value, then answer with that value."),
               UserMessage(role="user", content="Read the stored value using read_value.")]
        args = {"temperature": 0, "max_tokens": 16384, "seed": seed, "num_retries": 0}
        if arm == "llm_agent":
            response = llm_utils.generate("openai/agent122b", msg, [tool], **args)
        else:
            agent = ReferenceAgent([tool], "Use tools to read the value.", "openai/agent122b", args)
            state = agent.get_init_state(); response, state = agent.generate_next_message(msg[-1], state)
        assert response.tool_calls and response.tool_calls[0].name == "read_value" and response.tool_calls[0].arguments == {}, "tool canary"
        tc = response.tool_calls[0]
        obs = ToolMessage(role="tool", id=tc.id, content=read_value(), requestor="assistant", error=False)
        if arm == "llm_agent":
            msg.extend([response, obs]); response = llm_utils.generate("openai/agent122b", msg, [tool], **args)
        else:
            response, state = agent.generate_next_message(obs, state)
        assert "17" in (response.content or "") and not response.tool_calls, "observation/final canary"
        s = rec.summary(); assert s["reasoning_calls"] == 0 and s["token_count_mismatches"] == 0, s
        checks["canaries"][f"tool_{arm}"] = s
    # 3. user-simulator init check on the first cohort task of each domain (first utterance only; unscored).
    for d in C.DOMAINS:
        tid = next(t for t in cohort["tasks"] if t["domain"] == d)["task_id"]; task = C.get_task(d, tid)
        get_environment, _ = C.domain_api(d); env = get_environment()
        rec = new_rec(f"canary_user_{d}")
        cfg = C.config_for(d, "baseline", seed)
        user = build.build_user("user_simulator", env, task, llm=cfg.llm_user, llm_args=cfg.llm_args_user)
        st = user.get_init_state(message_history=[copy.deepcopy(DEFAULT_FIRST_AGENT_MESSAGE)])
        um, _ = user.generate_next_message(copy.deepcopy(DEFAULT_FIRST_AGENT_MESSAGE), st)
        assert um.content and "###STOP###" not in um.content and not um.tool_calls, "user simulator init"
        s = rec.summary(); assert s["reasoning_calls"] == 0, s
        checks["canaries"][f"user_{d}"] = {**s, "first_utterance_chars": len(um.content), "first_utterance_sha256": digest(um.content)}
    # 4. retail NL-judge canary (json_object response format through the same transport).
    rec = new_rec("canary_judge"); C.set_judge(seed)
    traj = [AssistantMessage(role="assistant", content="Hi! How can I help you today?"),
            UserMessage(role="user", content="I want to know the weather policy."),
            AssistantMessage(role="assistant", content="Thank you for asking. Our policy is that weather delays are refundable.")]
    res = NLAssertionsEvaluator.evaluate_nl_assertions(traj, ["The agent thanks the customer.", "The agent insults the customer."])
    assert len(res) == 2 and all(isinstance(x.met, bool) for x in res), res
    assert res[0].met is True, [(x.nl_assertion, x.met) for x in res]  # discriminative sanity; the negative case is informative only
    s = rec.summary(); assert s["judge_llm_calls"] == 1 and s["reasoning_calls"] == 0, s
    checks["canaries"]["judge"] = {**s, "verdicts": [x.met for x in res]}
    checks["contract_sha256"] = C.contract()["sha256"]; checks["ready_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(out / "complete.json", checks)
    return checks


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--base"); p.add_argument("--served", default="Qwen3.5-122B-A10B")
    p.add_argument("--offline-only", action="store_true"); p.add_argument("--out", default=str(ROOT / "preflight"))
    a = p.parse_args()
    try:
        off = offline(a.out)
        print(json.dumps({"offline": True, "gold_all_ok": off["gold_all_ok"], "gold_failures": off["gold_failures"]}), flush=True)
        if not off["gold_all_ok"]:
            sys.exit(3)
        if not a.offline_only:
            assert a.base, "--base required"
            lv = live(a.base, a.served, a.out)
            print(json.dumps({"live": True, "canaries": list(lv["canaries"])}), flush=True)
    except InfraInvalid as e:
        save(Path(a.out) / "error.json", {"type": "InfraInvalid", "reason": str(e), "trace": traceback.format_exc()})
        print("PREFLIGHT_INFRA", str(e), flush=True); sys.exit(75)
    except BaseException as e:
        save(Path(a.out) / "error.json", {"type": type(e).__name__, "trace": traceback.format_exc()})
        print("PREFLIGHT_ERROR", type(e).__name__, str(e)[:300], flush=True); sys.exit(3)
