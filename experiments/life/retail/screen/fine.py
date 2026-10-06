"""Run retail single-module ablations on the frozen cohort.

Install component switches at import time in each cell subprocess. Shared runtime
code handles model calls, task execution, scoring, and infrastructure failures.
Trial labels select the corresponding per-task seed from the stored cohort.
"""
from __future__ import annotations
import argparse, dataclasses, os
from pathlib import Path
from loguru import logger
logger.remove()   # omit benchmark text from console logs
from runtime import ROOT, load, digest, filehash
import cell as C
from tau2.agent.llm_agent import AGENT_INSTRUCTION, LLMAgent
from tau2.domains.retail import environment as retail_env
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.utils import RETAIL_DB_PATH, RETAIL_POLICY_PATH
from tau2.environment.environment import Environment
from tau2.harness import retail as HR
from tau2.harness import skills as SK
from tau2.registry import registry
from tau2.runner import build
import units as U

CODE_FINE = Path(__file__).resolve().parent
OUT_ROOT = Path(os.environ.get("LIFE_FINE_ROOT", str(ROOT)))
DOMAIN, SPLIT, H5_TOP_K = "retail", "train", 3      # upstream run_model.sh retail: --h5 (default top-k 3; retrieval caps to 2)
ARMS = U.ARMS
H5_BLOCK = "## Task-Specific Guidance"
NEVER = 10 ** 9
OUTSIDE_CONTRACT = {"batch_fine.py", "preflight_fine.py", "checks_fine.py", "aggregate_fine.py"}
CURRENT_ARM: str | None = None

C.CODE = CODE_FINE          # cell.run_cell resolves cohort.json through cell.CODE at call time
C.DOMAINS = [DOMAIN]


def cohort():
    return load(CODE_FINE / "cohort.json")


def cells_in_order(coh=None):
    """Task-major: all 49 arms of a task, in that task's rotated order, before the next task."""
    coh = coh or cohort()
    return [(DOMAIN, t["task_id"], arm) for t in coh["tasks"] for arm in t["arm_order"]]


# --------------------------------------------------------------------------- AI unit: the agent
AGENT_NOAI = "llm_agent_noai"


class NoAgentInstructionAgent(LLMAgent):
    """LLMAgent's loop verbatim; system prompt = the domain policy it was handed (which carries the
    H5 block when one was retrieved).  AGENT_INSTRUCTION and the XML wrapper are gone, nothing added."""

    @property
    def system_prompt(self) -> str:
        return self.domain_policy


if registry.get_agent_factory(AGENT_NOAI) is None:
    registry.register_agent_factory(
        lambda tools, domain_policy, **kw: NoAgentInstructionAgent(
            tools=tools, domain_policy=domain_policy, llm=kw["llm"], llm_args=kw["llm_args"]),
        AGENT_NOAI, metadata={"solo_mode": False})


# --------------------------------------------------------------------------- H2 / H3 / H4 units: the toolkit
BASE_TK = HR.H3H4HarnessedRetailTools
_TK: dict[str, type] = {}


def toolkit_cls(arm: str) -> type:
    """BASE_TK itself for `full` / H5 / AI arms; otherwise a subclass with exactly one unit removed."""
    if arm in _TK:
        return _TK[arm]
    u = U.unit_of(arm)
    attrs: dict = {}
    if u is not None and u.layer == "H2":        # every instance of the class, from every tool's list
        attrs["harness_rules"] = {t: [r for r in rs if type(r).__name__ != u.target]
                                  for t, rs in BASE_TK.harness_rules.items()}
    elif u is not None and u.layer == "H3":
        attrs["_h3_hints"] = {t: h for t, h in BASE_TK._h3_hints.items() if t != u.target}
    elif u is not None and u.layer == "H4" and u.target == "stuck_loop_alert":
        attrs["_STUCK_LOOP_THRESHOLD"] = NEVER
    elif u is not None and u.layer == "H4":
        attrs["harness_annotators"] = {t: [a for a in as_ if type(a).__name__ != u.target]
                                       for t, as_ in BASE_TK.harness_annotators.items()}
    _TK[arm] = type(f"Fine_{arm}", (BASE_TK,), attrs) if attrs else BASE_TK
    return _TK[arm]


_UP_GET_ENV = retail_env.get_environment


def get_environment(db=None, solo_mode=False, harness_enabled=False, harness_h3=False, harness_h4=False):
    """Every single-off configuration keeps H2/H3/H4 on; the arm set by config_for picks the toolkit subclass.
    Any other flag combination, or no current single-off configuration, is delegated to upstream unchanged."""
    combo = (bool(harness_enabled), bool(harness_h3), bool(harness_h4))
    if solo_mode or combo != (True, True, True) or CURRENT_ARM not in ARMS:
        return _UP_GET_ENV(db=db, solo_mode=solo_mode, harness_enabled=harness_enabled,
                           harness_h3=harness_h3, harness_h4=harness_h4)
    if db is None:
        db = RetailDB.load(RETAIL_DB_PATH)
    with open(RETAIL_POLICY_PATH, "r") as fp:
        policy = fp.read()
    return Environment(domain_name="retail", policy=policy, tools=toolkit_cls(CURRENT_ARM)(db))


def _install_env():
    """Bind BOTH resolution paths: cell.domain_api() reads the module attribute, while
    build_environment() and the evaluator read registry._domains, which captured the original
    function object at import (register_domain refuses to re-register, hence the direct rebind)."""
    if getattr(retail_env.get_environment, "_fine", False):
        return
    assert retail_env.get_environment is _UP_GET_ENV
    assert registry.get_env_constructor("retail") is _UP_GET_ENV, "registry retail slot is not upstream"
    get_environment._fine = True
    retail_env.get_environment = get_environment
    registry._domains["retail"] = get_environment
    assert registry.get_env_constructor("retail") is get_environment


_install_env()


# --------------------------------------------------------------------------- H5 units: the skill gate
_UP_GATE = SK._retail_skill_matches_context


def _skill_gate(skill, query):
    u = U.unit_of(CURRENT_ARM) if CURRENT_ARM in ARMS else None
    if u is not None and u.layer == "H5" and skill.id == u.target:
        return False
    return _UP_GATE(skill, query)


if SK._retail_skill_matches_context is _UP_GATE:
    SK._retail_skill_matches_context = _skill_gate   # retrieve_skills looks the gate up at call time


# --------------------------------------------------------------------------- task split / config
def get_task(domain, task_id):
    _, get_tasks = C.domain_api(domain)
    return next(t for t in get_tasks(SPLIT) if t.id == task_id)


C.get_task = get_task

_ORIG_CONFIG_FOR = C.config_for
_ORIG_ENV_KW = C.harness_env_kwargs


def harness_env_kwargs(arm):
    if arm in ARMS:
        return {"harness_enabled": True, "harness_h3": True, "harness_h4": True}
    return _ORIG_ENV_KW(arm)


C.harness_env_kwargs = harness_env_kwargs


def config_for(domain, arm, seed):
    global CURRENT_ARM
    if arm not in ARMS:                              # preflight canaries still use upstream arm names
        CURRENT_ARM = None
        return _ORIG_CONFIG_FOR(domain, arm, seed)
    CURRENT_ARM = arm
    u = U.unit_of(arm)
    base = _ORIG_CONFIG_FOR(domain, "full", seed)   # upstream `full` run config, inherited verbatim
    ai_off = u is not None and u.layer == "AI"
    cfg = base.model_copy(update={"agent": AGENT_NOAI}) if ai_off else base
    d0, d1 = base.model_dump(mode="json"), cfg.model_dump(mode="json")
    assert {k for k in set(d0) | set(d1) if d0.get(k) != d1.get(k)} == ({"agent"} if ai_off else set()), arm
    assert cfg.harness_enabled and cfg.harness_h3 and cfg.harness_h4 and cfg.harness_h5, arm
    assert not cfg.harness_h5_rag and not cfg.harness_h5_rag_tool and cfg.harness_h5_rag_top_k == H5_TOP_K
    assert cfg.max_steps == 200 and cfg.max_errors == 10 and cfg.seed == seed
    return cfg


C.config_for = config_for


# --------------------------------------------------------------------------- contract
def contract():
    coh = cohort()
    c = {"cohort": filehash(CODE_FINE / "cohort.json"), "protocol": filehash(OUT_ROOT / "PROTOCOL.md"),
         "splits_v1": filehash(CODE_FINE / "splits_v1.json"),
         "code": {p.name: filehash(p) for p in sorted(Path(C.__file__).resolve().parent.glob("*.py"))},
         "code_fine": {p.name: filehash(p) for p in sorted(CODE_FINE.glob("*.py")) if p.name not in OUTSIDE_CONTRACT},
         "upstream_src": {str(p.relative_to(C.UPSTREAM)): filehash(p)
                          for p in sorted((C.UPSTREAM / "src").rglob("*.py")) if "__pycache__" not in p.parts},
         "data": {k: filehash(C.UPSTREAM / "data" / "tau2" / "domains" / k) for k in coh["data_sha256"]},
         "settings": {"served_model": "Qwen3.5-122B-A10B", "thinking": False, "temperature": 0,
                      "max_tokens": 16384, "max_model_len": 262144, "max_steps": 200, "max_errors": 10,
                      "timeout": None, "domain": DOMAIN, "split": SPLIT, "h5_top_k": H5_TOP_K,
                      "eval": "ALL_WITH_NL_ASSERTIONS", "nl_judge": C.JUDGE_MODEL,
                      "k_trials": coh["k_trials"], "arms": ARMS,
                      "units": [dataclasses.asdict(u) for u in U.UNITS],
                      "loop": "upstream LLMAgent for every arm", "h5_prefill": "on in every arm",
                      "outside_contract": sorted(OUTSIDE_CONTRACT)}}
    assert c["data"] == coh["data_sha256"], "upstream data changed since cohort freeze"
    c["sha256"] = digest(c)
    return c


C.contract = contract


# --------------------------------------------------------------------------- build audit (derived, not whitelisted)
_PREV_BUILD = build.build_agent        # = cell.audited_build_agent, installed when `cell` was imported
ALL = {layer: {u.target for u in U.UNITS if u.layer == layer} for layer in ("H2", "H3", "H4", "H5")}


def audited_build_agent(agent_name, environment, **kwargs):
    a = _PREV_BUILD(agent_name, environment, **kwargs)
    arm = CURRENT_ARM
    if arm in ARMS and agent_name in {"llm_agent", AGENT_NOAI}:
        u = U.unit_of(arm)
        gone = lambda layer: {u.target} if (u is not None and u.layer == layer) else set()
        tk, sp, policy = environment.tools, a.system_prompt, environment.get_policy()
        rules = {type(r).__name__ for rs in tk.harness_rules.values() for r in rs}
        anns = {type(x).__name__ for xs in tk.harness_annotators.values() for x in xs}
        assert rules == ALL["H2"] - gone("H2"), (arm, sorted(rules ^ ALL["H2"]))
        if u is not None and u.layer == "H2":     # the class must be gone from EVERY tool's list
            assert all(type(r).__name__ != u.target for rs in tk.harness_rules.values() for r in rs), arm
        assert set(tk._h3_hints) == ALL["H3"] - gone("H3"), (arm, sorted(set(tk._h3_hints) ^ ALL["H3"]))
        assert anns == ALL["H4"] - gone("H4") - {"stuck_loop_alert"}, (arm, sorted(anns))
        stuck_off = u is not None and u.target == "stuck_loop_alert"
        assert (tk._STUCK_LOOP_THRESHOLD == NEVER) == stuck_off, (arm, tk._STUCK_LOOP_THRESHOLD)
        ai_off = u is not None and u.layer == "AI"
        assert isinstance(a, LLMAgent) and (type(a) is NoAgentInstructionAgent) == ai_off, (arm, type(a))
        assert (AGENT_INSTRUCTION in sp) == (not ai_off), arm
        assert (("<instructions>" in sp) or ("<policy>" in sp)) == (not ai_off), arm
        assert kwargs.get("harness_h5") is True, (arm, "H5 prefill + retrieval must run in every arm")
        injected = [s.id for s in SK.RETAIL_SKILLS if s.title in sp]
        assert not (gone("H5") & set(injected)), (arm, "gated-out skill reached the prompt", injected)
        if ai_off:   # prompt IS domain_policy: equals the raw policy exactly when H5 injected nothing
            assert (sp == policy) == (H5_BLOCK not in sp), arm
        else:        # the <instructions>/<policy> wrapper is present, so never equal to the raw policy
            assert sp != policy, arm
        C.CURRENT_AUDIT.update(arm=arm, unit=(u.uid if u else None), toolkit_class=type(tk).__name__,
                               n_h2_rules=len(rules), n_h2_instances=sum(len(rs) for rs in tk.harness_rules.values()),
                               n_h3_hints=len(tk._h3_hints), n_h4_annotators=len(anns),
                               stuck_loop_armed=not stuck_off, agent_class=type(a).__name__,
                               h5_skill_ids=injected, system_prompt_equals_policy=(sp == policy))
    return a


build.build_agent = audited_build_agent


# --------------------------------------------------------------------------- entry point
def run(base, served, task_id, arm, transport_factory=None, out_root=None, enforce_contract=True):
    assert arm in ARMS, arm
    return C.run_cell(base, served, DOMAIN, task_id, arm, transport_factory=transport_factory,
                      out_root=Path(out_root or OUT_ROOT), enforce_contract=enforce_contract)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True); p.add_argument("--served", default="Qwen3.5-122B-A10B")
    p.add_argument("--task", required=True); p.add_argument("--arm", choices=ARMS, required=True)
    a = p.parse_args()
    r = run(a.base, a.served, a.task, a.arm)
    raise SystemExit(0 if r["status"] == "valid" else 75 if r["status"] in {"infra_invalid", "judge_infra_invalid"} else 3)
