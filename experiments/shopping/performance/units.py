"""The frozen 23-module universe of the JIT seed harness `agentfold` on DeepPlanning-Shopping.

Fixed BEFORE any inference, from SOURCE STRUCTURE only (heading-level prompt blocks, with the 15 official shopping APIs held fixed).  "full" = upstream `harness_factory/harnesses/
agentfold` run exactly as the published seed arm (`seed_agentfold`).  Every module is one authored element
with its own boundary; switching it off never edits prose -- text is cut at the source's own headings,
code is switched by a single asserted substitution (variant.py).

  M  mechanisms in action.py / planning.py                                   (5)
  S  blocks of prompt.yaml `step.pre_messages`                               (3)
  P  blocks of prompt.yaml `system_prompt`, at its own # / ## headings       (10)
  T  the tool the harness adds on top of the benchmark API (+ its prompt)    (1)
  G  the numbered guidance items of DeepPlanning's agent prompt              (4)

Held fixed (not modules): the task objective (mission sentence + **Core Mission**) and the user query; the
15 official shopping APIs + final_answer; the text-JSON action protocol (system "# Action", step intro,
"# Tool List" schemas, the closing "Note that you may invoke up to 5 tools..." line); memory's step-id
rendering; max_steps 40, 3x model-call budget, thinking OFF, temperature 0, 262144 ctx.
"""
from __future__ import annotations

UNITS = [
    # uid, layer, off means
    ("m_initial_plan", "M", "no planner call; no Plan entry in memory (planner prompt unused)"),
    ("m_periodic_replan", "M", "should_replan -> False: no summary call every 8 steps (summary prompts unused)"),
    ("m_fold_compression", "M", "the model's top-level `compress` field is ignored; history is never folded"),
    ("m_forced_final_answer", "M", "no forced-finalisation call at max_steps (final_answer prompts unused)"),
    ("m_lenient_json_parsing", "M", "json_repair.loads -> json.loads in the action module"),
    ("s_task_restatement", "S", "drop step '# Your original task:' (the task re-pasted at every step)"),
    ("s_compression_guidelines", "S", "drop step '# Compression Guidelines:'"),
    ("s_step_example", "S", "drop step 'Example output ...' block"),
    ("p_role_preamble", "P", "drop the two opening paragraphs of the system prompt"),
    ("p_planning_execution_model", "P", "drop '## Planning And Execution Model'"),
    ("p_how_to_execute_plan", "P", "drop '## How To Execute The Plan'"),
    ("p_action_expectations", "P", "drop '## Action Expectations'"),
    ("p_examples", "P", "drop '# Examples'"),
    ("p_result_processing", "P", "drop '# Result Processing'"),
    ("p_final_answer_rules", "P", "drop '# Final Answer'"),
    ("p_tool_strategy", "P", "drop '# Tool Strategy'"),
    ("p_available_tools_list", "P", "drop '# Available Tools' (name: description list; schemas stay in the step prompt)"),
    ("p_rules", "P", "drop '# Rules' (through the closing reward sentence)"),
    ("t_execute_code", "T", "remove the execute_code tool AND the task's '## Workspace' block (linked group)"),
    ("g1_determine_requirements", "G", "drop guidance item 1 of SHOPPING_SYSTEM_PROMPT_L{level}"),
    ("g2_min_price_strategy", "G", "drop guidance item 2 (L2: budget; L3: coupon logic)"),
    ("g3_cart_source_of_truth", "G", "drop guidance item 3 (get_cart_info verification)"),
    ("g4_final_output_requirements", "G", "drop guidance item 4"),
]
UIDS = [u[0] for u in UNITS]
LAYER = {u[0]: u[1] for u in UNITS}
EXPECTED_COUNTS = {"M": 5, "S": 3, "P": 10, "T": 1, "G": 4}
assert {k: sum(1 for u in UNITS if u[1] == k) for k in EXPECTED_COUNTS} == EXPECTED_COUNTS
assert len(set(UIDS)) == len(UIDS) == 23

SCREEN_ARMS = ["full"] + [f"off_{u}" for u in UIDS]          # single-off screen (24)
PREMISE_ARMS = ["full", "S0"]                                 # S0 = every module off


def off_set(arm: str) -> frozenset:
    """Modules switched OFF in an arm label (without trial suffix)."""
    if arm == "full":
        return frozenset()
    if arm == "S0":
        return frozenset(UIDS)
    if arm.startswith("S") and arm[1:].isdigit():                # ladder rung S_k = first k modules of LADDER_ORDER on
        k = int(arm[1:]); assert k in LADDER_RUNGS, arm
        return frozenset(UIDS) - frozenset(LADDER_ORDER[:k])
    assert arm.startswith("off_") and arm[4:] in LAYER, arm
    return frozenset([arm[4:]])


def trial_label(arm: str, trial: int) -> str:
    return arm if trial == 0 else f"{arm}@t{trial}"


def split_label(label: str):
    arm, _, t = label.partition("@t")
    return arm, (int(t) if t else 0)


# Performance-oriented validation ladder. The screening baseline is the per-task
# median over single-off configurations. Gate on p_eff <= .05 (four modules),
# then order by (p_pass, p_eff). The gated set is the base rung; add three modules
# per rung and end with the full harness. S0 and full are reference configurations.
LADDER_ORDER = [
    "s_step_example",
    "p_examples",
    "p_action_expectations",
    "m_fold_compression",
    "s_task_restatement",
    "m_periodic_replan",
    "g3_cart_source_of_truth",
    "p_available_tools_list",
    "p_rules",
    "p_tool_strategy",
    "p_planning_execution_model",
    "g1_determine_requirements",
    "p_final_answer_rules",
    "g4_final_output_requirements",
    "m_forced_final_answer",
    "m_lenient_json_parsing",
    "s_compression_guidelines",
    "t_execute_code",
    "g2_min_price_strategy",
    "p_how_to_execute_plan",
    "p_role_preamble",
    "p_result_processing",
    "m_initial_plan"
]
assert sorted(LADDER_ORDER) == sorted(UIDS)
LADDER_RUNGS = [4, 7, 10, 13, 16, 19, 22]
LADDER_ARMS = [f"S{k}" for k in LADDER_RUNGS]
