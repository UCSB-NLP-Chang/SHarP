#!/usr/bin/env python3
"""Efficiency ladder definitions with a performance gate and median-token baseline."""

from __future__ import annotations

from typing import Any
import os

from components import BY_NAME, apply_single_off, component_names


# Fixed efficiency-oriented order from training-set screening.
BOTTOM_UP_ORDER: tuple[str, ...] = (
    "gen_answer_tool",
    "paper_search_tool",
    "evidence_summarization",
    "complete_tool",
    "gather_evidence_tool",
    "agent_workflow_guidance",
    "evidence_retrieval",
    "agent_system_prompt",
    "document_metadata_extraction",
    "high_quality_concurrency",
    "high_quality_chunking",
    "lazy_evidence_fallback",
    "reset_tool",
    "structured_evidence_json",
    "answer_iteration",
    "multimodal_processing",
    "citation_metadata_context",
    "high_quality_evidence_depth",
)

CONFIG_SIZES: dict[str, int] = {f"A{size}": size for size in range(0, 19)}

PRIMARY_CONFIGS: tuple[str, ...] = (
    "A3",
    "A6",
    "A9",
    "A12",
    "A15",
    "A18",
)
if os.environ.get("SHARP_OBJECTIVE", "efficiency") == "performance":
    BOTTOM_UP_ORDER = ('gen_answer_tool', 'paper_search_tool', 'evidence_summarization', 'gather_evidence_tool', 'agent_workflow_guidance', 'evidence_retrieval', 'complete_tool', 'lazy_evidence_fallback', 'citation_metadata_context', 'multimodal_processing', 'answer_iteration', 'document_metadata_extraction', 'structured_evidence_json', 'high_quality_chunking', 'high_quality_concurrency', 'high_quality_evidence_depth', 'reset_tool', 'agent_system_prompt')
    PRIMARY_CONFIGS = ('A7', 'A10', 'A13', 'A16', 'A18')

SEARCH_DETAIL_CONFIGS: tuple[str, ...] = ()
INDEX_COMPONENTS: tuple[str, ...] = tuple(
    name for name in component_names() if BY_NAME[name].affects_index
)


def kept_components(config_name: str) -> tuple[str, ...]:
    try:
        size = CONFIG_SIZES[config_name]
    except KeyError as exc:
        raise ValueError(f"Unknown ladder config: {config_name}") from exc
    return BOTTOM_UP_ORDER[:size]


def disabled_components(config_name: str) -> tuple[str, ...]:
    kept = set(kept_components(config_name))
    return tuple(name for name in component_names() if name not in kept)


def apply_ladder_config(settings: Any, config_name: str) -> None:
    for name in disabled_components(config_name):
        apply_single_off(settings, name)


def parsing_signature(config_name: str) -> tuple[str, ...]:
    kept = set(kept_components(config_name))
    return tuple(name for name in INDEX_COMPONENTS if name in kept)


def index_label(config_name: str) -> str:
    signature = parsing_signature(config_name)
    if not signature:
        return "low-parsing"
    if signature == ("multimodal_processing",):
        return "chunk5000-mm-nodoc"
    if signature == ("multimodal_processing", "document_metadata_extraction"):
        return "chunk5000-mm-doc"
    if signature == ("document_metadata_extraction",):
        return "chunk5000-nomm-doc"
    if signature == ("high_quality_chunking", "document_metadata_extraction"):
        return "chunk7000-nomm-doc"
    if set(signature) == set(INDEX_COMPONENTS):
        return "full-parsing"
    raise ValueError(
        f"No frozen validation index substrate for {config_name}: {signature}"
    )


def balanced_config_order(task_position: int, names: tuple[str, ...]) -> tuple[str, ...]:
    """Cyclic Latin order within a phase, preserving task-level endpoint blocks."""
    if not names:
        return ()
    shift = task_position % len(names)
    return names[shift:] + names[:shift]


def shard_for_task(task_position: int, shard_count: int) -> int:
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    return task_position % shard_count


def validate_definition() -> None:
    if set(BOTTOM_UP_ORDER) != set(component_names()):
        raise RuntimeError("Bottom-up order is not exactly the 18-component inventory")
    if len(BOTTOM_UP_ORDER) != len(set(BOTTOM_UP_ORDER)):
        raise RuntimeError("Bottom-up order contains duplicates")
    if os.environ.get("SHARP_OBJECTIVE", "efficiency") == "performance":
        return
    if tuple(kept_components("A3")) != BOTTOM_UP_ORDER[:3]:
        raise RuntimeError("A3 is not the fixed base configuration of the efficiency-oriented ladder")
    if set(kept_components("A3")) != {"gen_answer_tool", "paper_search_tool", "evidence_summarization"}:
        raise RuntimeError("A3 does not match the efficiency-oriented gate from the 50-task single-off screen")


validate_definition()


__all__ = [
    "BOTTOM_UP_ORDER",
    "CONFIG_SIZES",
    "INDEX_COMPONENTS",
    "PRIMARY_CONFIGS",
    "SEARCH_DETAIL_CONFIGS",
    "apply_ladder_config",
    "balanced_config_order",
    "disabled_components",
    "index_label",
    "kept_components",
    "parsing_signature",
    "shard_for_task",
]
