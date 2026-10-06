"""Task-paired saliency tests against the median single-module-ablation field.

This portable interface extracts the calculation used by the LIFE and shopping
component-table scripts. It requires a complete single-trial task-by-module grid.
Full-harness rows are not part of the reference field.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def one_sided_p(values, alternative):
    if alternative not in {"less", "greater"}:
        raise ValueError("alternative must be less or greater")
    if len(values) < 2 or not all(math.isfinite(x) for x in values):
        raise ValueError("At least two finite task-paired differences are required")
    mean = statistics.fmean(values)
    if all(value == values[0] for value in values):
        if mean == 0:
            return 1.0
        return float(not ((alternative == "less" and mean < 0) or
                          (alternative == "greater" and mean > 0)))
    from scipy.stats import ttest_1samp
    return float(ttest_1samp(values, 0.0, alternative=alternative).pvalue)


def saliency(records, components):
    """Rows contain task_id, component, score, tokens; component is the removed unit."""
    components = list(components)
    if len(components) < 2 or len(set(components)) != len(components):
        raise ValueError("Supply at least two unique component names")
    cells = {}
    for row in records:
        task, component = str(row["task_id"]), str(row["component"])
        if not task or component not in components:
            raise ValueError(f"Invalid task or component: {task!r}, {component!r}")
        key = task, component
        if key in cells:
            raise ValueError(f"Duplicate cell: {key}; provide one observation per task and module")
        score, tokens = float(row["score"]), float(row["tokens"])
        if not (math.isfinite(score) and 0 <= score <= 1 and math.isfinite(tokens) and tokens >= 0):
            raise ValueError(f"Invalid score or token count: {key}")
        cells[key] = score, tokens
    tasks = sorted({task for task, _ in cells})
    if len(tasks) < 2:
        raise ValueError("At least two tasks are required")
    missing = [(task, c) for task in tasks for c in components if (task, c) not in cells]
    if missing:
        raise ValueError(f"Incomplete ablation field: {len(missing)} missing cells; first: {missing[0]}")
    reference = {task: tuple(statistics.median(cells[task, c][j] for c in components)
                             for j in (0, 1)) for task in tasks}
    rows = []
    for c in components:
        dp = [cells[t, c][0] - reference[t][0] for t in tasks]
        dt = [cells[t, c][1] - reference[t][1] for t in tasks]
        rows.append({"component": c, "delta_pass_mean": statistics.fmean(dp),
                     "delta_tokens_mean": statistics.fmean(dt),
                     "p_pass": one_sided_p(dp, "less"), "p_eff": one_sided_p(dt, "greater")})
    return {"reference": "per-task median of all single-module ablations",
            "n_tasks": len(tasks), "n_components": len(components), "rows": rows}


def pruning_ladder(rows, objective="efficiency", alpha=0.05, step=3):
    if objective not in {"efficiency", "performance"}:
        raise ValueError("Unknown pruning objective")
    if not (0 < alpha < 1) or not isinstance(step, int) or step < 1:
        raise ValueError("alpha must be between 0 and 1; step must be a positive integer")
    gate_key, order_key = ("p_pass", "p_eff") if objective == "efficiency" else ("p_eff", "p_pass")
    key = lambda r: (r[order_key], r[gate_key], r["component"])
    protected = sorted((r for r in rows if r[gate_key] <= alpha), key=key)
    others = sorted((r for r in rows if r[gate_key] > alpha), key=key)
    order = [r["component"] for r in protected + others]
    if len(set(order)) != len(order) or not order:
        raise ValueError("Component rows must be nonempty and unique")
    sizes = sorted({0, len(order), *range(len(protected), len(order), step)})
    return {"objective": objective, "alpha": alpha, "step": step,
            "protected": [r["component"] for r in protected], "retention_order": order,
            "configurations": [{"name": f"S{k}", "retained": order[:k],
                                "removed": order[k:]} for k in reversed(sizes)]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="CSV: task_id,component,score,tokens")
    p.add_argument("--components", type=Path, required=True, help="JSON list of all candidate module names")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--objective", choices=["efficiency", "performance"], default="efficiency")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--step", type=int, default=3)
    a = p.parse_args(argv)
    with a.input.open(newline="") as f:
        result = saliency(list(csv.DictReader(f)), json.loads(a.components.read_text()))
    result["ladder"] = pruning_ladder(result["rows"], a.objective, a.alpha, a.step)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"Screened {result['n_components']} modules on {result['n_tasks']} tasks -> {a.output}")


if __name__ == "__main__":
    main()
