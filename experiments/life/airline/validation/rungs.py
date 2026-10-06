"""Performance-oriented and efficiency-oriented airline validation ladders.

The screening baseline is the per-task median over the 44 single-off configurations
on the screening task set (30 tasks). The performance objective gates on p_eff <= .05
and orders by (p_pass, p_eff); the efficiency objective gates on p_pass <= .05 and
orders by (p_eff, p_pass).

The gated set is the base rung, each next rung adds three modules, and the final
rung is the full harness. An empty gated set starts with three modules.
Performance arms are A4, A7, ..., A43; efficiency arms are B3, B6, ..., B42.
Both ladders share full. Arm A<n> or B<n> enables the first n modules in its order.
"""
from __future__ import annotations
import json
from pathlib import Path
import units as U

HERE = Path(__file__).resolve().parent
TABLES = {"A": HERE / "component_table_fine.tokens.median.json",
          "B": HERE / "component_table_fine.tokens.median.designB.json"}
STEP, N_UNITS = 3, 44

_TARGET_TO_UID = {u.target: u.uid for u in U.UNITS}
assert len(_TARGET_TO_UID) == N_UNITS, "unit targets must be unique"


def _order(design: str) -> list[str]:
    t = json.loads(TABLES[design].read_text())
    assert t["method"].get("design", "A") == design and "median" in t["method"]["baseline"], (design, t["method"])
    order = [_TARGET_TO_UID[c] for c in t["bottom_up_order"]]
    assert len(order) == N_UNITS and len(set(order)) == N_UNITS
    return order


ORDER = {"A": _order("A"), "B": _order("B")}
GATE = {d: json.loads(TABLES[d].read_text())["n_search_space"] for d in ORDER}
assert GATE == {"A": 4, "B": 0}, GATE


def rung_sizes(design: str) -> list[int]:
    """Gated set (if any) as the base, then +3 per rung while below 44."""
    g = GATE[design]
    sizes = [g] if g else []
    n = g + STEP
    while n < N_UNITS:
        sizes.append(n); n += STEP
    return sizes


RUNG_SIZES = {d: rung_sizes(d) for d in ORDER}
assert RUNG_SIZES["A"] == list(range(4, 44, 3)) and RUNG_SIZES["B"] == list(range(3, 44, 3))
ALL_UIDS = [u.uid for u in U.UNITS]


def on_units(arm: str) -> list[str]:
    if arm == "full":
        return list(ALL_UIDS)
    design, n = arm[0], int(arm[1:])
    assert design in ORDER and n in RUNG_SIZES[design], arm
    return ORDER[design][:n]


def off_units(arm: str) -> frozenset[str]:
    return frozenset(ALL_UIDS) - frozenset(on_units(arm))


RUNGS = {d: [f"{d}{n}" for n in RUNG_SIZES[d]] + ["full"] for d in ORDER}
ARMS: list[str] = [a for a in RUNGS["A"] if a != "full"] + [a for a in RUNGS["B"] if a != "full"] + ["full"]
assert len(ARMS) == 29


def describe() -> dict:
    return {"tables": {k: str(v.name) for k, v in TABLES.items()}, "gate": GATE, "step": STEP,
            "rung_sizes": RUNG_SIZES, "arms": ARMS, "rungs": RUNGS,
            "order": ORDER, "on_units": {a: on_units(a) for a in ARMS}}
