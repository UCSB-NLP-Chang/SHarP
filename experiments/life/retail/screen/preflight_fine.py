"""Preflight for the retail single-off screen.

Run the shared preflight checks for this screening cohort:
  offline() -- cohort re-derivation, official data hashes, gold replay (native + full; retail offline
               reward = ENV x COMMUNICATE, the NL component needs the live judge), agent audit
  live()    -- arithmetic / tool-call / user-simulator / judge canaries, tokenizer-server agreement
and adds, offline:
  gold_all_arms()  gold-action replay in EVERY arm's toolkit (49): identical final DB hashes and
                   ENV x COMMUNICATE reward 1.0 everywhere -- removing one H2 rule / H4 annotator /
                   H3 hint / the stuck-loop must never change what a valid action does.
  arm_audit()      builds all 49 agents on one task and checks each arm differs from `full` in
                   exactly the place its unit lives: tool schema only for H3 arms (and only that
                   tool), system prompt only for the AI arm and for an H5 arm whose skill `full`
                   retrieved, nothing model-visible for H2 / H4 arms (their effect is at execution).
  skill_gate()     with no arm active the patched retrieval equals upstream on every cohort query;
                   under each H5 arm its skill is never returned.
Writes <out>/complete_fine.json.  Exit codes: 0 ok, 75 infra (endpoint), 3 error.
"""
from __future__ import annotations
import argparse, copy, json, sys, time, traceback
from pathlib import Path
from loguru import logger
logger.remove()
from runtime import InfraInvalid, save, digest
import cell as C
import fine as F
import units as U
import cohort_fine as CO
import preflight as P
from tau2.evaluator.evaluator_env import EnvironmentEvaluator
from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
from tau2.harness import skills as SK

P.CODE = F.CODE_FINE
P.CO = CO


def proxy_query(task):
    ins = getattr(getattr(task, "user_scenario", None), "instructions", None)
    return str((ins.get("reason_for_call") if isinstance(ins, dict) else getattr(ins, "reason_for_call", "")) or "")


def gold_all_arms(out):
    bad, rows = [], []
    for spec in F.cohort()["tasks"]:
        task = C.get_task(F.DOMAIN, spec["task_id"]); hashes = set(); rewards = {}
        for arm in F.ARMS:
            F.config_for(F.DOMAIN, arm, 1)
            kw = F.harness_env_kwargs(arm)
            sim, env, errors = P.gold_simulation(F.DOMAIN, task, kw)
            env_r = EnvironmentEvaluator.calculate_reward(domain=F.DOMAIN, task=task, full_trajectory=sim.messages,
                                                          solo_mode=False, env_kwargs=dict(kw)).reward
            comm_r = CommunicateEvaluator.calculate_reward(task=task, full_trajectory=sim.messages).reward
            rewards[arm] = env_r * comm_r
            hashes.add((env.get_db_hash(), env.get_user_db_hash()))
        ok = len(hashes) == 1 and all(v == 1.0 for v in rewards.values())
        rows.append({"task_id": spec["task_id"], "db_hashes_agree": len(hashes) == 1,
                     "arms_below_1": [a for a, v in rewards.items() if v != 1.0]})
        if not ok:
            bad.append(spec["task_id"])
    F.config_for(F.DOMAIN, "baseline", 1)             # clears CURRENT_ARM
    save(Path(out) / "gold_all_arms.json", {"failures": bad, "rows": rows})
    return {"n_tasks": len(rows), "n_arms": len(F.ARMS), "all_ok": not bad, "failures": bad}


def skill_gate():
    qs = [proxy_query(C.get_task(F.DOMAIN, t["task_id"])) for t in F.cohort()["tasks"]]
    F.config_for(F.DOMAIN, "baseline", 1)             # no single-off configuration active
    ours = [[s.id for s in SK.retrieve_skills(F.DOMAIN, q, top_k=F.H5_TOP_K)] for q in qs]
    SK._retail_skill_matches_context = F._UP_GATE
    try:
        up = [[s.id for s in SK.retrieve_skills(F.DOMAIN, q, top_k=F.H5_TOP_K)] for q in qs]
    finally:
        SK._retail_skill_matches_context = F._skill_gate
    assert ours == up, "patched gate changed retrieval with no arm active"
    for u in U.UNITS:
        if u.layer != "H5":
            continue
        F.config_for(F.DOMAIN, f"off_{u.uid}", 1)
        got = {sid for q in qs for sid in (s.id for s in SK.retrieve_skills(F.DOMAIN, q, top_k=F.H5_TOP_K))}
        assert u.target not in got, (u.uid, "gated skill was still retrieved")
    F.config_for(F.DOMAIN, "baseline", 1)
    return {"queries": len(qs), "full_equals_upstream": True,
            "queries_with_a_skill": sum(bool(x) for x in up),
            "skills_retrieved_by_full": sorted({sid for x in up for sid in x})}


def arm_audit(out):
    coh = F.cohort()
    qs = {t["task_id"]: proxy_query(C.get_task(F.DOMAIN, t["task_id"])) for t in coh["tasks"]}
    F.config_for(F.DOMAIN, "baseline", 1)
    tid = next(t for t, q in qs.items() if SK.retrieve_skills(F.DOMAIN, q, top_k=F.H5_TOP_K))  # H5 fires here
    task, query = C.get_task(F.DOMAIN, tid), qs[tid]
    got = {}
    for arm in F.ARMS:
        cfg = F.config_for(F.DOMAIN, arm, 1)
        ge, _ = C.domain_api(F.DOMAIN); env = ge(**F.harness_env_kwargs(arm))
        C.CURRENT_AUDIT.clear()
        a = F.audited_build_agent(cfg.agent, env, llm="openai/agent122b", llm_args=copy.deepcopy(cfg.llm_args_agent),
                                  task=task, harness_h5=True, harness_h5_query=query, harness_h5_rag_top_k=F.H5_TOP_K)
        got[arm] = {"prompt": digest(a.system_prompt),
                    "schema": {t.name: digest(t.openai_schema) for t in a.tools},
                    "audit": dict(C.CURRENT_AUDIT)}
    F.config_for(F.DOMAIN, "baseline", 1)
    full = got["full"]
    injected = full["audit"]["h5_skill_ids"]
    assert injected, "audit task must exercise H5"
    for arm in F.ARMS[1:]:
        u, g = U.unit_of(arm), got[arm]
        diff_tools = {t for t in full["schema"] if g["schema"].get(t) != full["schema"][t]}
        assert set(g["schema"]) == set(full["schema"]), arm
        assert diff_tools == ({u.target} if u.layer == "H3" else set()), (arm, diff_tools)
        prompt_moves = u.layer == "AI" or (u.layer == "H5" and u.target in injected)
        assert (g["prompt"] != full["prompt"]) == prompt_moves, (arm, "system prompt moved unexpectedly")
    save(Path(out) / "arm_audit.json", {"audit_task_id": tid, "full_injected_skills": injected,
                                         "arms": {a: g["audit"] for a, g in got.items()}})
    return {"audit_task_id": tid, "full_injected_skills": injected, "arms_checked": len(got)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base"); p.add_argument("--served", default="Qwen3.5-122B-A10B")
    p.add_argument("--offline-only", action="store_true"); p.add_argument("--out", default=str(F.OUT_ROOT / "preflight"))
    a = p.parse_args(); out = Path(a.out)
    checks = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        off = P.offline(out)
        checks["offline"] = {k: off[k] for k in ["cohort_rederives", "data_matches_official", "n_tasks", "n_cells",
                                                 "gold_all_ok", "gold_failures"]}
        if not off["gold_all_ok"]:
            save(out / "error.json", {"type": "GoldReplayFailed", "failures": off["gold_failures"]}); return 3
        checks["gold_all_arms"] = gold_all_arms(out)
        if not checks["gold_all_arms"]["all_ok"]:
            save(out / "error.json", {"type": "GoldArmMismatch", **checks["gold_all_arms"]}); return 3
        checks["skill_gate"] = skill_gate()
        checks["arm_audit"] = arm_audit(out)
        checks["cells_planned"] = len(F.cells_in_order())
        checks["contract_sha256"] = F.contract()["sha256"]
        if not a.offline_only:
            assert a.base, "--base required"
            checks["live"] = {k: v for k, v in P.live(a.base, a.served, out)["canaries"].items()}
        checks["all_ok"] = True
        checks["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save(out / ("complete_fine.json" if not a.offline_only else "offline_fine.json"), checks)
        print(json.dumps({"preflight_fine": "ok", "cells": checks["cells_planned"], "live": sorted(checks.get("live", {}))}), flush=True)
        return 0
    except InfraInvalid as e:
        save(out / "error.json", {"type": "InfraInvalid", "reason": str(e), "trace": traceback.format_exc()})
        print("PREFLIGHT_INFRA", str(e), flush=True); return 75
    except BaseException as e:
        save(out / "error.json", {"type": type(e).__name__, "trace": traceback.format_exc()})
        print("PREFLIGHT_ERROR", type(e).__name__, str(e)[:300], flush=True); return 3


if __name__ == "__main__":
    sys.exit(main())
