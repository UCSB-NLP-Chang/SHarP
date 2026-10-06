"""Performance-oriented retail validation ladder.

The screening baseline is the per-task median over the 48 single-off configurations
on the screening task set (50 tasks). Gate on p_eff <= .05 and order by (p_pass, p_eff).
The seven gated modules form the base rung, each next rung adds three modules,
and the final rung is the full harness. Arm A<n> enables the first n modules
in this order: A7, A10, ..., A46, full.
"""
from __future__ import annotations
import json
from pathlib import Path
import units as U

HERE = Path(__file__).resolve().parent
TABLES = {"A": HERE / "component_table_fine.tokens.median.json"}     # Performance order: gate p_eff, then sort by p_pass.
DESIGN, STEP, N_UNITS = "A", 3, 48

_TARGET_TO_UID = {u.target: u.uid for u in U.UNITS}
assert len(_TARGET_TO_UID) == N_UNITS, "unit targets must be unique"

_t = json.loads(TABLES["A"].read_text())
assert _t["method"].get("design") == "A" and "median" in _t["method"]["baseline"], _t["method"]
assert _t["n_components"] == N_UNITS
ORDER = {"A": [_TARGET_TO_UID[c] for c in _t["bottom_up_order"]]}
assert len(ORDER["A"]) == N_UNITS and len(set(ORDER["A"])) == N_UNITS
GATE = {"A": _t["n_search_space"]}
assert GATE["A"] == 7, GATE


def rung_sizes(design: str) -> list[int]:
    g = GATE[design]
    sizes = [g] if g else []
    n = g + STEP
    while n < N_UNITS:
        sizes.append(n); n += STEP
    return sizes


RUNG_SIZES = {"A": rung_sizes("A")}
assert RUNG_SIZES["A"] == list(range(7, 48, 3))
ALL_UIDS = [u.uid for u in U.UNITS]


def on_units(arm: str) -> list[str]:
    if arm == "full":
        return list(ALL_UIDS)
    design, n = arm[0], int(arm[1:])
    assert design in ORDER and n in RUNG_SIZES[design], arm
    return ORDER[design][:n]


def off_units(arm: str) -> frozenset[str]:
    return frozenset(ALL_UIDS) - frozenset(on_units(arm))


RUNGS = {"A": [f"A{n}" for n in RUNG_SIZES["A"]] + ["full"]}
ARMS: list[str] = list(RUNGS["A"])
assert len(ARMS) == 15


def describe() -> dict:
    return {"tables": {k: str(v.name) for k, v in TABLES.items()}, "gate": GATE, "step": STEP,
            "rung_sizes": RUNG_SIZES, "arms": ARMS, "rungs": RUNGS,
            "order": ORDER, "on_units": {a: on_units(a) for a in ARMS}}
