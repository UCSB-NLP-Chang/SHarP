"""Run airline pruning configurations on the frozen cohort.

Install component switches at import time in each cell subprocess. Shared runtime
code handles model calls, task execution, scoring, and infrastructure failures.
Trial labels select the corresponding per-task seed from the stored cohort.
"""
from __future__ import annotations
import argparse, copy, dataclasses, os
from pathlib import Path
from loguru import logger
logger.remove()   # omit benchmark text from console logs
from runtime import ROOT, load, digest, filehash
import cell as C
from tau2.agent.llm_agent import AGENT_INSTRUCTION, LLMAgent
from tau2.domains.airline import environment as airline_env
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.utils import AIRLINE_DB_PATH, AIRLINE_POLICY_PATH
from tau2.environment.environment import Environment
from tau2.harness import airline as HA
from tau2.harness import skills as SK
from tau2.registry import registry
from tau2.runner import build
import units as U
import rungs as RG
import cohort_ladder as CO

CODE_LADDER = Path(__file__).resolve().parent
OUT_ROOT = Path(os.environ.get("LIFE_LADDER_ROOT", str(ROOT)))
DOMAIN, H5_TOP_K = "airline", 1
ARMS = RG.ARMS
H5_BLOCK = "## Task-Specific Guidance"
NEVER = 10 ** 9
OUTSIDE_CONTRACT = {"batch_ladder.py", "preflight_ladder.py", "checks_ladder.py", "aggregate_ladder.py"}
CURRENT_ARM: str | None = None       # bare arm name (no trial suffix)
CURRENT_TRIAL: int = 0

C.CODE = CODE_LADDER          # cell.run_cell resolves cohort.json through cell.CODE at call time
C.DOMAINS = [DOMAIN]


def cohort():
    return load(CODE_LADDER / "cohort.json")


def cell_labels(coh=None):
    coh = coh or cohort()
    return sorted({lab for t in coh["tasks"] for lab in t["arm_order"]})


def cells_in_order(coh=None):
    """Task-major: 29 configurations and 3 trials per task, in the stored order."""
    coh = coh or cohort()
    return [(DOMAIN, t["task_id"], lab) for t in coh["tasks"] for lab in t["arm_order"]]


# --------------------------------------------------------------------------- trials: per-trial seed
_UP_LOAD = C.load


def _load_with_trial(path):
    """cell.run_cell reads spec['seed'] from cohort.json; for trial k >= 1 hand it the task's k-th seed."""
    obj = _UP_LOAD(path)
    if CURRENT_TRIAL and Path(path).name == "cohort.json" and Path(path).parent == CODE_LADDER:
        obj = copy.deepcopy(obj)
        for t in obj["tasks"]:
            t["seed"] = t["seeds"][CURRENT_TRIAL]
    return obj


C.load = _load_with_trial


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
BASE_TK = HA.H3H4HarnessedAirlineTools
_TK: dict[str, type] = {}


def off_by_layer(arm: str) -> dict[str, set[str]]:
    off = RG.off_units(arm)
    out = {layer: set() for layer in ("H2", "H3", "H4", "H5", "AI")}
    for uid in off:
        u = U.BY_UID[uid]; out[u.layer].add(u.target)
    return out


def toolkit_cls(arm: str) -> type:
    """BASE_TK itself when no H2/H3/H4 unit is off; otherwise a subclass with exactly those removed."""
    if arm in _TK:
        return _TK[arm]
    off = off_by_layer(arm)
    attrs: dict = {}
    if off["H2"]:
        attrs["harness_rules"] = {t: [r for r in rs if type(r).__name__ not in off["H2"]]
                                  for t, rs in BASE_TK.harness_rules.items()}
    if off["H3"]:
        attrs["_h3_hints"] = {t: h for t, h in BASE_TK._h3_hints.items() if t not in off["H3"]}
    if "stuck_loop_alert" in off["H4"]:
        attrs["_STUCK_LOOP_THRESHOLD"] = NEVER
    if off["H4"] - {"stuck_loop_alert"}:
        attrs["harness_annotators"] = {t: [a for a in as_ if type(a).__name__ not in off["H4"]]
                                       for t, as_ in BASE_TK.harness_annotators.items()}
    _TK[arm] = type(f"Ladder_{arm}", (BASE_TK,), attrs) if attrs else BASE_TK
    return _TK[arm]


_UP_GET_ENV = airline_env.get_environment


def get_environment(db=None, solo_mode=False, harness_enabled=False, harness_h3=False, harness_h4=False):
    """Every ladder arm keeps H2/H3/H4 flags on; the arm set by config_for picks the toolkit subclass.
    Any other flag combination, or no current ladder arm, is delegated to upstream unchanged."""
    combo = (bool(harness_enabled), bool(harness_h3), bool(harness_h4))
    if solo_mode or combo != (True, True, True) or CURRENT_ARM not in ARMS:
        return _UP_GET_ENV(db=db, solo_mode=solo_mode, harness_enabled=harness_enabled,
                           harness_h3=harness_h3, harness_h4=harness_h4)
    if db is None:
        db = FlightDB.load(AIRLINE_DB_PATH)
    with open(AIRLINE_POLICY_PATH) as fp:
        policy = fp.read()
    return Environment(domain_name="airline", policy=policy, tools=toolkit_cls(CURRENT_ARM)(db))


def _install_env():
    if getattr(airline_env.get_environment, "_ladder", False):
        return
    assert airline_env.get_environment is _UP_GET_ENV
    assert registry.get_env_constructor("airline") is _UP_GET_ENV, "registry airline slot is not upstream"
    get_environment._ladder = True
    airline_env.get_environment = get_environment
    registry._domains["airline"] = get_environment
    assert registry.get_env_constructor("airline") is get_environment


_install_env()


# --------------------------------------------------------------------------- H5 units: the skill gate
_UP_GATE = SK._airline_skill_matches_context


def _skill_gate(skill, query):
    if CURRENT_ARM in ARMS and skill.id in off_by_layer(CURRENT_ARM)["H5"]:
        return False
    return _UP_GATE(skill, query)


if SK._airline_skill_matches_context is _UP_GATE:
    SK._airline_skill_matches_context = _skill_gate   # retrieve_skills looks the gate up at call time


# --------------------------------------------------------------------------- config
_ORIG_CONFIG_FOR = C.config_for
_ORIG_ENV_KW = C.harness_env_kwargs


def harness_env_kwargs(arm):
    bare, _ = CO.split_label(arm)
    if bare in ARMS:
        return {"harness_enabled": True, "harness_h3": True, "harness_h4": True}
    return _ORIG_ENV_KW(arm)


C.harness_env_kwargs = harness_env_kwargs


def config_for(domain, arm, seed):
    global CURRENT_ARM, CURRENT_TRIAL
    bare, trial = CO.split_label(arm)
    if bare not in ARMS:                             # preflight canaries still use upstream arm names
        CURRENT_ARM, CURRENT_TRIAL = None, 0
        return _ORIG_CONFIG_FOR(domain, arm, seed)
    CURRENT_ARM, CURRENT_TRIAL = bare, trial
    base = _ORIG_CONFIG_FOR(domain, "full", seed)   # upstream `full` run config, inherited verbatim
    ai_off = bool(off_by_layer(bare)["AI"])
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
    c = {"cohort": filehash(CODE_LADDER / "cohort.json"), "protocol": filehash(OUT_ROOT / "PROTOCOL.md"),
         "rung_tables": {k: filehash(v) for k, v in RG.TABLES.items()},
         "code": {p.name: filehash(p) for p in sorted(Path(C.__file__).resolve().parent.glob("*.py"))},
         "code_ladder": {p.name: filehash(p) for p in sorted(CODE_LADDER.glob("*.py")) if p.name not in OUTSIDE_CONTRACT},
         "upstream_src": {str(p.relative_to(C.UPSTREAM)): filehash(p)
                          for p in sorted((C.UPSTREAM / "src").rglob("*.py")) if "__pycache__" not in p.parts},
         "data": {k: filehash(C.UPSTREAM / "data" / "tau2" / "domains" / k) for k in coh["data_sha256"]},
         "settings": {"served_model": "Qwen3.5-122B-A10B", "thinking": False, "temperature": 0,
                      "max_tokens": 16384, "max_model_len": 262144, "max_steps": 200, "max_errors": 10,
                      "timeout": None, "domain": DOMAIN, "split": CO.SPLIT, "h5_top_k": H5_TOP_K,
                      "k_trials": coh["k_trials"], "arms": ARMS, "rungs": RG.RUNGS,
                      "on_units": {a: RG.on_units(a) for a in ARMS},
                      "units": [dataclasses.asdict(u) for u in U.UNITS],
                      "loop": "upstream LLMAgent for every arm", "h5_prefill": "on in every arm",
                      "outside_contract": sorted(OUTSIDE_CONTRACT)}}
    assert c["data"] == coh["data_sha256"], "upstream data changed since cohort freeze"
    c["sha256"] = digest(c)
    return c


C.contract = contract


# --------------------------------------------------------------------------- build audit (derived from the arm's off set)
_PREV_BUILD = build.build_agent        # = cell.audited_build_agent, installed when `cell` was imported
ALL = {layer: {u.target for u in U.UNITS if u.layer == layer} for layer in ("H2", "H3", "H4", "H5")}


def audited_build_agent(agent_name, environment, **kwargs):
    a = _PREV_BUILD(agent_name, environment, **kwargs)
    arm = CURRENT_ARM
    if arm in ARMS and agent_name in {"llm_agent", AGENT_NOAI}:
        off = off_by_layer(arm)
        tk, sp, policy = environment.tools, a.system_prompt, environment.get_policy()
        rules = {type(r).__name__ for rs in tk.harness_rules.values() for r in rs}
        anns = {type(x).__name__ for xs in tk.harness_annotators.values() for x in xs}
        assert rules == ALL["H2"] - off["H2"], (arm, sorted(rules ^ (ALL["H2"] - off["H2"])))
        assert set(tk._h3_hints) == ALL["H3"] - off["H3"], (arm, sorted(set(tk._h3_hints) ^ (ALL["H3"] - off["H3"])))
        assert anns == ALL["H4"] - off["H4"] - {"stuck_loop_alert"}, (arm, sorted(anns))
        stuck_off = "stuck_loop_alert" in off["H4"]
        assert (tk._STUCK_LOOP_THRESHOLD == NEVER) == stuck_off, (arm, tk._STUCK_LOOP_THRESHOLD)
        ai_off = bool(off["AI"])
        assert isinstance(a, LLMAgent) and (type(a) is NoAgentInstructionAgent) == ai_off, (arm, type(a))
        assert (AGENT_INSTRUCTION in sp) == (not ai_off), arm
        assert (("<instructions>" in sp) or ("<policy>" in sp)) == (not ai_off), arm
        assert kwargs.get("harness_h5") is True, (arm, "H5 prefill + retrieval must run in every arm")
        injected = [s.id for s in SK.AIRLINE_SKILLS if s.title in sp]
        assert not (off["H5"] & set(injected)), (arm, "gated-out skill reached the prompt", injected)
        if ai_off:   # prompt IS domain_policy: equals the raw policy exactly when H5 injected nothing
            assert (sp == policy) == (H5_BLOCK not in sp), arm
        else:        # the <instructions>/<policy> wrapper is present, so never equal to the raw policy
            assert sp != policy, arm
        C.CURRENT_AUDIT.update(arm=arm, trial=CURRENT_TRIAL, n_units_on=len(RG.on_units(arm)),
                               off_units=sorted(RG.off_units(arm)), toolkit_class=type(tk).__name__,
                               n_h2_rules=len(rules), n_h3_hints=len(tk._h3_hints), n_h4_annotators=len(anns),
                               stuck_loop_armed=not stuck_off, agent_class=type(a).__name__,
                               h5_skill_ids=injected, system_prompt_equals_policy=(sp == policy))
    return a


build.build_agent = audited_build_agent


# --------------------------------------------------------------------------- entry point
def run(base, served, task_id, label, transport_factory=None, out_root=None, enforce_contract=True):
    bare, trial = CO.split_label(label)
    assert bare in ARMS and 0 <= trial < cohort()["k_trials"], label
    global CURRENT_TRIAL
    CURRENT_TRIAL = trial                          # so the cohort read inside run_cell carries this trial's seed
    return C.run_cell(base, served, DOMAIN, task_id, label, transport_factory=transport_factory,
                      out_root=Path(out_root or OUT_ROOT), enforce_contract=enforce_contract)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True); p.add_argument("--served", default="Qwen3.5-122B-A10B")
    p.add_argument("--task", required=True); p.add_argument("--arm", required=True)
    a = p.parse_args()
    assert a.arm in cell_labels(), a.arm
    r = run(a.base, a.served, a.task, a.arm)
    raise SystemExit(0 if r["status"] == "valid" else 75 if r["status"] in {"infra_invalid", "judge_infra_invalid"} else 3)
