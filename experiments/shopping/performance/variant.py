"""Build the harness variant for a set of switched-off modules (units.py) WITHOUT touching upstream.

Three surfaces, each cut only at boundaries the source itself draws:
  * prompt.yaml        -> system_prompt / step.pre_messages re-assembled from their own heading blocks
  * action.py, planning.py -> one asserted substitution per mechanism (count == 1, else hard failure)
  * task text + tool set -> DeepPlanning's agent prompt cut at its numbered items; '## Workspace' block and
                            the execute_code tool removed together
`full` (nothing off) reproduces the pristine files byte-for-byte for the .py files and text-identically for
the prompts (asserted in checks).
"""
from __future__ import annotations
import re, shutil
from pathlib import Path
import yaml

HARNESS = "agentfold"
HARNESS_FILES = ["__init__.py", "action.py", "memory.py", "planning.py", "tool_policy.py", "prompt.yaml", "description.yaml"]

# --------------------------------------------------------------------------- system prompt blocks
SYS_HEADS = [  # (exact heading line or prefix, unit uid | "FIXED" | "CONTAINER")
    ("# Planning And Path Routing", "CONTAINER"),
    ("## Planning And Execution Model", "p_planning_execution_model"),
    ("## How To Execute The Plan", "p_how_to_execute_plan"),
    ("## Action Expectations", "p_action_expectations"),
    ("# Action", "FIXED"),
    ("# Examples", "p_examples"),
    ("# Result Processing", "p_result_processing"),
    ("# Final Answer", "p_final_answer_rules"),
    ("# Tool Strategy", "p_tool_strategy"),
    ("# Available Tools", "p_available_tools_list"),     # runs through the (disabled) skills jinja block
    ("# Rules", "p_rules"),
]
CONTAINER_CHILDREN = {"p_planning_execution_model", "p_how_to_execute_plan", "p_action_expectations"}
STEP_HEADS = [
    ("# Tool List:", "FIXED"),
    ("# Your original task:", "s_task_restatement"),
    ("# Compression Guidelines:", "s_compression_guidelines"),
    ("Example output (schematic", "s_step_example"),
    ("Note that you may invoke", "FIXED"),
]


def _segments(text: str, heads, first_unit: str):
    """Split `text` into [(unit, block_text)] at the given heading lines (each must match exactly one line)."""
    lines = text.split("\n")
    idx = []
    for head, unit in heads:
        exact = head in ("# Action",)      # '# Action' is a prefix of nothing else, but '## Action Expectations' contains it
        hits = [i for i, ln in enumerate(lines) if (ln == head if exact else ln.startswith(head))]
        assert len(hits) == 1, (head, hits)
        idx.append((hits[0], unit))
    assert [i for i, _ in idx] == sorted(i for i, _ in idx), "headings out of order"
    out = [(first_unit, "\n".join(lines[:idx[0][0]]))]
    for k, (i, unit) in enumerate(idx):
        j = idx[k + 1][0] if k + 1 < len(idx) else len(lines)
        out.append((unit, "\n".join(lines[i:j])))
    assert "\n".join(b for _, b in out) == text, "segments must re-assemble the source exactly"
    return out


def cut_system_prompt(text: str, off: frozenset) -> str:
    keep = []
    for unit, block in _segments(text, SYS_HEADS, "p_role_preamble"):
        if unit == "CONTAINER":
            if CONTAINER_CHILDREN <= off:
                continue
        elif unit != "FIXED" and unit in off:
            continue
        keep.append(block)
    return "\n".join(keep)


def cut_step_prompt(text: str, off: frozenset) -> str:
    return "\n".join(block for unit, block in _segments(text, STEP_HEADS, "FIXED") if unit == "FIXED" or unit not in off)


# --------------------------------------------------------------------------- mechanisms (asserted substitutions)
CODE_PATCHES = {
    "m_initial_plan": ("action.py", [(
        "        plan = ctx.planning.init_plan(\n"
        "            task, memory_view, tool_selection.tool_schemas_json, ctx.model,\n"
        "        )\n"
        "        ctx.memory.update_plan(plan)\n"
        "        ctx.logger.log_markdown(plan.plan, title=\"Initial Plan\", level=LogLevel.INFO)\n"
        "        step_number += 1\n",
        "        plan = None  # m_initial_plan OFF: no planner call, no Plan entry; step budget unchanged\n"
        "        step_number += 1\n", 1)]),
    "m_periodic_replan": ("planning.py", [(
        "    def should_replan(self, step_number: int, step: StepRecord) -> bool:\n        return (\n",
        "    def should_replan(self, step_number: int, step: StepRecord) -> bool:\n        return False and (  # m_periodic_replan OFF\n", 1)]),
    "m_fold_compression": ("action.py", [(
        "        compress_info = _extract_compress_info(raw_content)\n",
        "        compress_info = None  # m_fold_compression OFF\n", 1)]),
    "m_forced_final_answer": ("action.py", [(
        "            final_answer = self._force_final_answer(task, ctx, trajectory, step_number)\n",
        "            final_answer = \"\"  # m_forced_final_answer OFF\n", 1)]),
    "m_lenient_json_parsing": ("action.py", [
        ("import json_repair\n", "import json  # m_lenient_json_parsing OFF\nimport json_repair\n", 1),
        ("json_repair.loads(", "json.loads(", 3)]),
}


def build_harness(off: frozenset, pristine: Path, dest_root: Path) -> dict:
    """Write dest_root/agentfold/ = pristine with `off` applied; returns an audit dict."""
    dest = Path(dest_root) / HARNESS
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    texts = {f: (Path(pristine) / f).read_text(encoding="utf-8") for f in HARNESS_FILES if (Path(pristine) / f).is_file()}
    for uid, (fname, subs) in CODE_PATCHES.items():
        if uid not in off:
            continue
        for old, new, n in subs:
            assert texts[fname].count(old) == n, (uid, fname, texts[fname].count(old), n)
            texts[fname] = texts[fname].replace(old, new)
    prompts = yaml.safe_load(texts["prompt.yaml"])
    sys0, step0 = prompts["system_prompt"], prompts["step"]["pre_messages"]
    prompts["system_prompt"] = cut_system_prompt(sys0, off)
    prompts["step"]["pre_messages"] = cut_step_prompt(step0, off)
    prompt_changed = (prompts["system_prompt"], prompts["step"]["pre_messages"]) != (sys0, step0)
    if prompt_changed:
        texts["prompt.yaml"] = yaml.safe_dump(prompts, allow_unicode=True, sort_keys=False, width=10 ** 9)
        back = yaml.safe_load(texts["prompt.yaml"])
        assert back == prompts, "prompt.yaml round trip changed the prompts"
    for f, t in texts.items():
        (dest / f).write_text(t, encoding="utf-8")
    return {"harness_dir": str(dest), "off": sorted(off), "prompt_yaml_rewritten": prompt_changed,
            "system_prompt": prompts["system_prompt"], "step_prompt": prompts["step"]["pre_messages"],
            "code_patched": sorted(u for u in CODE_PATCHES if u in off)}


# --------------------------------------------------------------------------- task text (G units, Workspace block)
_ITEM = re.compile(r"^(\*\*)?([1-4])\.\s")
G_UNITS = ["g1_determine_requirements", "g2_min_price_strategy", "g3_cart_source_of_truth", "g4_final_output_requirements"]
GUIDE_HEAD = "**Guiding Principles"
WORKSPACE_HEAD = "\n\n## Workspace\n"


def cut_level_prompt(s1: str, off: frozenset) -> str:
    lines = s1.split("\n")
    starts = [(i, int(_ITEM.match(ln).group(2))) for i, ln in enumerate(lines) if _ITEM.match(ln)]
    assert [n for _, n in starts] == [1, 2, 3, 4], starts
    heads = [i for i, ln in enumerate(lines) if ln.startswith(GUIDE_HEAD)]
    assert len(heads) == 1 and heads[0] < starts[0][0], heads
    keep = lines[:heads[0]]
    if not set(G_UNITS) <= off:
        keep += lines[heads[0]:starts[0][0]]           # the '**Guiding Principles ...**' header (+ blank line)
    for k, (i, _) in enumerate(starts):
        j = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        if G_UNITS[k] not in off:
            keep += lines[i:j]
    return "\n".join(keep)


def cut_task(task: str, s1: str, off: frozenset) -> str:
    if off & set(G_UNITS):
        assert task.count(s1) == 1, "level prompt not found exactly once in the formatted task"
        task = task.replace(s1, cut_level_prompt(s1, off))
    if "t_execute_code" in off:
        assert task.count(WORKSPACE_HEAD) == 1, "workspace block not found exactly once"
        task = task[:task.index(WORKSPACE_HEAD)]
    return task


def config_without_execute_code(src_yaml: Path, dest_yaml: Path) -> Path:
    text = Path(src_yaml).read_text(encoding="utf-8")
    assert text.count("  - execute_code\n") == 1
    Path(dest_yaml).write_text(text.replace("  - execute_code\n", ""), encoding="utf-8")
    return Path(dest_yaml)
