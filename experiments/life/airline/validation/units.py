"""The frozen 44-unit component universe of the Life-Harness airline harness.

Fixed BEFORE any inference, from the harness SOURCE STRUCTURE only -- no run data (no activation or
firing counts) enters the partition.  Every unit is one authored element with its own boundary in the
code, enumerated here from the upstream data structures rather than transcribed by hand:

  H2  one unit per Rule class in HarnessedAirlineTools.harness_rules        (19)
  H3  one unit per tool key in _AIRLINE_H3_HINTS                             (9)
  H4  one unit per Annotator class in H4AirlineAnnotationMixin.harness_annotators (7)
      + stuck_loop_alert: implemented in HarnessedToolKitMixin._track_failure, NOT in the annotator
        mixin, so it is its own element (upstream flags cannot toggle it separately)  (1)
  H5  one unit per Skill in AIRLINE_SKILLS                                   (7)
  AI  AGENT_INSTRUCTION + the SYSTEM_PROMPT <instructions>/<policy> wrapper, as ONE unit (1)

Held fixed in every arm (not units): the domain policy text, the native toolkit and its tool
schemas apart from H3's description text, the upstream termination protocol / user simulator /
evaluator, and the H5 prefill path (skill retrieval needs the pre-generated first user utterance as
its query, so the path cannot be switched off on its own; every arm keeps harness_h5 on).

Arm "off_<uid>" = full minus exactly that one unit.  How each is switched off:
  H2   drop the Rule instance from its tool's list in `harness_rules`
  H3   drop the tool's key from `_h3_hints` (the tool keeps its native description)
  H4   drop the Annotator instance from its tool's list in `harness_annotators`
  H4 stuck_loop_alert   `_STUCK_LOOP_THRESHOLD` -> never reached (failures still propagate)
  H5   gate the skill out at the airline context-gating step; the BM25 index (IDF, doc lengths) is
       untouched, so the arm differs from `full` only where this skill would have been a candidate.
       At top_k=1 the runner-up may then be injected -- the harness's own behaviour without it.
  AI   system_prompt -> domain_policy (the H5 block, when retrieved, is part of domain_policy and stays)
"""
from __future__ import annotations

from dataclasses import dataclass

from tau2.harness import airline as HA
from tau2.harness.h3_tools import _AIRLINE_H3_HINTS
from tau2.harness.skills import AIRLINE_SKILLS


@dataclass(frozen=True)
class Unit:
    uid: str        # stable id, used in arm names: off_<uid>
    layer: str      # H2 | H3 | H4 | H5 | AI
    target: str     # class name / tool name / skill id
    tool: str       # tool the unit attaches to ("" when not tool-bound)


def _units() -> list[Unit]:
    out: list[Unit] = []
    for tool, rules in HA.HarnessedAirlineTools.harness_rules.items():
        for r in rules:
            n = type(r).__name__
            out.append(Unit(f"h2_{n}", "H2", n, tool))
    for tool in _AIRLINE_H3_HINTS:
        out.append(Unit(f"h3_{tool}", "H3", tool, tool))
    for tool, anns in HA.H4AirlineAnnotationMixin.harness_annotators.items():
        for a in anns:
            n = type(a).__name__
            out.append(Unit(f"h4_{n}", "H4", n, tool))
    out.append(Unit("h4_stuck_loop_alert", "H4", "stuck_loop_alert", ""))
    for s in AIRLINE_SKILLS:
        out.append(Unit(f"h5_{s.id}", "H5", s.id, ""))
    out.append(Unit("ai_agent_instruction", "AI", "AGENT_INSTRUCTION", ""))
    return out


UNITS: list[Unit] = _units()
BY_UID: dict[str, Unit] = {u.uid: u for u in UNITS}
EXPECTED_COUNTS = {"H2": 19, "H3": 9, "H4": 8, "H5": 7, "AI": 1}

_counts = {k: sum(u.layer == k for u in UNITS) for k in EXPECTED_COUNTS}
assert _counts == EXPECTED_COUNTS, f"upstream structure moved: {_counts} != {EXPECTED_COUNTS}"
assert len(BY_UID) == len(UNITS) == 44, "unit ids must be unique"

ARMS: list[str] = ["full"] + [f"off_{u.uid}" for u in UNITS]


def unit_of(arm: str) -> Unit | None:
    if arm == "full":
        return None
    assert arm.startswith("off_") and arm[4:] in BY_UID, arm
    return BY_UID[arm[4:]]
