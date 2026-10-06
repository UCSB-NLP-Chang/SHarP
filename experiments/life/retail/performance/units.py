"""The frozen 48-unit component universe of the Life-Harness RETAIL harness.

Fixed BEFORE any inference, from the harness SOURCE STRUCTURE only -- no run data enters the partition.
Partition modules as in the Airline screen: every unit is one
authored element with its own boundary in the code, enumerated from the upstream data structures:

  H2  one unit per Rule CLASS in HarnessedRetailTools.harness_rules (_H2_RULES)          (19)
      NOTE: retail registers some rule classes under several tools (NonEmptyItemIdsRule,
      NonEmptyNewItemIdsRule, ItemIdsNewItemIdsParityRule, NoDuplicateItemIdsRule,
      ValidateNewItemIdExistsRule appear in 2-3 tool lists; 26 instances of 19 classes).  The authored
      element is the class; switching it off removes it from every tool's list.  A unit's `tool` is the
      first tool it is registered under, `tools` lists all of them.
  H3  one unit per tool key in the EFFECTIVE _RETAIL_H3_HINTS (after upstream's read-tool filter)   (8)
  H4  one unit per Annotator class registered in H4RetailAnnotationMixin.harness_annotators    (10)
      + stuck_loop_alert (HarnessedToolKitMixin._track_failure, threshold 3)                    (1)
      (annotator classes defined in retail.py but not registered in the mixin are not harness elements)
  H5  one unit per Skill in RETAIL_SKILLS                                                       (9)
  AI  AGENT_INSTRUCTION + the SYSTEM_PROMPT <instructions>/<policy> wrapper, as ONE unit          (1)

Held fixed in every arm (not units): the domain policy text, the native toolkit and its schemas apart from
H3's description text, the upstream termination protocol / user simulator / evaluator (incl. the NL judge),
the H5 prefill path (harness_h5 on in every arm; retail retrieval: top_k capped to 2, min_score 1.5), and
harness_h5_rag = False (policy RAG is off in upstream's `full` and stays off).

Arm "off_<uid>" = full minus exactly that one unit.  Switching:
  H2   drop every instance of the Rule class from `harness_rules`
  H3   drop the tool's key from `_h3_hints` (the tool keeps its native description)
  H4   drop the Annotator instance from its tool's list in `harness_annotators`
  H4 stuck_loop_alert   `_STUCK_LOOP_THRESHOLD` -> never reached (failures still propagate)
  H5   gate the skill out at the retail context-gating step; the BM25 index is untouched
  AI   system_prompt -> domain_policy (the H5 block, when retrieved, is part of domain_policy and stays)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from tau2.harness import retail as HR
from tau2.harness.h3_tools import _RETAIL_H3_HINTS
from tau2.harness.skills import RETAIL_SKILLS


@dataclass(frozen=True)
class Unit:
    uid: str        # stable id, used in arm names: off_<uid>
    layer: str      # H2 | H3 | H4 | H5 | AI
    target: str     # class name / tool name / skill id
    tool: str       # first tool the unit attaches to ("" when not tool-bound)
    tools: list = field(default_factory=list)   # every tool the unit is registered under (H2 may be several).
    # A list, not a tuple: cell.freeze() compares the recomputed contract with the JSON-loaded one by ==,
    # and tuples differ from lists after a JSON round trip.


def _units() -> list[Unit]:
    out: list[Unit] = []
    seen: dict[str, list[str]] = {}
    for tool, rules in HR.HarnessedRetailTools.harness_rules.items():
        for r in rules:
            seen.setdefault(type(r).__name__, []).append(tool)
    for n, tools in seen.items():
        out.append(Unit(f"h2_{n}", "H2", n, tools[0], list(tools)))
    for tool in _RETAIL_H3_HINTS:
        out.append(Unit(f"h3_{tool}", "H3", tool, tool, [tool]))
    for tool, anns in HR.H4RetailAnnotationMixin.harness_annotators.items():
        for a in anns:
            n = type(a).__name__
            out.append(Unit(f"h4_{n}", "H4", n, tool, [tool]))
    out.append(Unit("h4_stuck_loop_alert", "H4", "stuck_loop_alert", "", []))
    for s in RETAIL_SKILLS:
        out.append(Unit(f"h5_{s.id}", "H5", s.id, "", []))
    out.append(Unit("ai_agent_instruction", "AI", "AGENT_INSTRUCTION", "", []))
    return out


UNITS: list[Unit] = _units()
BY_UID: dict[str, Unit] = {u.uid: u for u in UNITS}
EXPECTED_COUNTS = {"H2": 19, "H3": 8, "H4": 11, "H5": 9, "AI": 1}
N_UNITS = sum(EXPECTED_COUNTS.values())        # 48

_counts = {k: sum(u.layer == k for u in UNITS) for k in EXPECTED_COUNTS}
assert _counts == EXPECTED_COUNTS, f"upstream structure moved: {_counts} != {EXPECTED_COUNTS}"
assert len(BY_UID) == len(UNITS) == N_UNITS, "unit ids must be unique"
assert sum(len(rs) for rs in HR.HarnessedRetailTools.harness_rules.values()) == 26, "H2 instance count moved"

ARMS: list[str] = ["full"] + [f"off_{u.uid}" for u in UNITS]


def unit_of(arm: str) -> Unit | None:
    if arm == "full":
        return None
    assert arm.startswith("off_") and arm[4:] in BY_UID, arm
    return BY_UID[arm[4:]]
