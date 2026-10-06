"""Full and all-pruned reference configurations for retail validation.

ARMS contains A0 (all 48 modules disabled) and full (all modules enabled).
The component table also defines the efficiency-oriented module order.
"""
from __future__ import annotations
import json
from pathlib import Path
import units as U

HERE = Path(__file__).resolve().parent
TABLES = {"B": HERE / "component_table_fine.tokens.median.designB.json"}
DESIGN, STEP, N_UNITS = "B", 3, 48

_TARGET_TO_UID = {u.target: u.uid for u in U.UNITS}
assert len(_TARGET_TO_UID) == N_UNITS, "unit targets must be unique"

_t = json.loads(TABLES["B"].read_text())
assert _t["method"].get("design") == "B" and "median" in _t["method"]["baseline"], _t["method"]
assert _t["n_components"] == N_UNITS
ORDER = {"B": [_TARGET_TO_UID[c] for c in _t["bottom_up_order"]]}
assert len(ORDER["B"]) == N_UNITS and len(set(ORDER["B"])) == N_UNITS
GATE = {"B": _t["n_search_space"]}
assert GATE["B"] == 13, GATE


def rung_sizes(design: str) -> list[int]:
    g = GATE[design]
    sizes = [g] if g else []
    n = g + STEP
    while n < N_UNITS:
        sizes.append(n); n += STEP
    return sizes


RUNG_SIZES = {"B": rung_sizes("B")}
assert RUNG_SIZES["B"] == list(range(13, 48, 3))
ALL_UIDS = [u.uid for u in U.UNITS]


def on_units(arm: str) -> list[str]:
    if arm == "full":
        return list(ALL_UIDS)
    if arm == "A0":                       # every unit off (the all-pruned harness)
        return []
    design, n = arm[0], int(arm[1:])
    assert design in ORDER and n in RUNG_SIZES[design], arm
    return ORDER[design][:n]


def off_units(arm: str) -> frozenset[str]:
    return frozenset(ALL_UIDS) - frozenset(on_units(arm))


# Evaluate the all-pruned configuration and full harness with matched tasks and seeds.
RUNGS = {"A0": ["A0", "full"]}
ARMS: list[str] = ["A0", "full"]


def describe() -> dict:
    return {"tables": {k: str(v.name) for k, v in TABLES.items()}, "gate": GATE, "step": STEP,
            "rung_sizes": RUNG_SIZES, "arms": ARMS, "rungs": RUNGS,
            "order": ORDER, "on_units": {a: on_units(a) for a in ARMS}}
