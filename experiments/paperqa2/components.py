#!/usr/bin/env python3
"""Mechanical PaperQA2 component inventory for single-off screening."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


DEFAULT_TOOLS = (
    "paper_search",
    "gather_evidence",
    "gen_answer",
    "reset",
    "complete",
)


@dataclass(frozen=True)
class Component:
    name: str
    kind: str
    affects_index: bool
    off_description: str
    apply_off: Callable[[Any], None]


def _drop_tool(name: str) -> Callable[[Any], None]:
    def apply(settings: Any) -> None:
        # Filter the current tool list so multiple tool ablations compose.  A
        # PaperQA represents its default tool set as None until it is resolved
        # by the agent. Materialize that default before filtering so joint and
        # single-tool ablations have the same explicit baseline.
        current_tools = (
            DEFAULT_TOOLS
            if settings.agent.tool_names is None
            else settings.agent.tool_names
        )
        settings.agent.tool_names = [
            tool for tool in current_tools if tool != name
        ]

    return apply


def _agent_system_prompt_off(settings: Any) -> None:
    settings.agent.agent_system_prompt = None


def _agent_workflow_guidance_off(settings: Any) -> None:
    settings.agent.agent_prompt = "Use the tools to answer the question: {question}"


def _high_quality_evidence_depth_off(settings: Any) -> None:
    settings.answer.evidence_k = 10


def _high_quality_concurrency_off(settings: Any) -> None:
    settings.answer.max_concurrent_requests = 4


def _high_quality_chunking_off(settings: Any) -> None:
    settings.parsing.reader_config = {"chunk_chars": 5000, "overlap": 250}


def _evidence_retrieval_off(settings: Any) -> None:
    settings.answer.evidence_retrieval = False


def _evidence_summarization_off(settings: Any) -> None:
    settings.answer.evidence_skip_summary = True


def _structured_evidence_json_off(settings: Any) -> None:
    settings.prompts.use_json = False


def _lazy_evidence_fallback_off(settings: Any) -> None:
    settings.answer.get_evidence_if_no_contexts = False


def _answer_iteration_off(settings: Any) -> None:
    settings.prompts.answer_iteration_prompt = None


def _citation_metadata_context_off(settings: Any) -> None:
    settings.prompts.context_inner = "{name}: {text}"


def _multimodal_processing_off(settings: Any) -> None:
    settings.parsing.multimodal = False


def _document_metadata_extraction_off(settings: Any) -> None:
    settings.parsing.use_doc_details = False


COMPONENTS: tuple[Component, ...] = (
    Component("paper_search_tool", "tool", False, "remove paper_search", _drop_tool("paper_search")),
    Component("gather_evidence_tool", "tool", False, "remove gather_evidence", _drop_tool("gather_evidence")),
    Component("reset_tool", "tool", False, "remove reset", _drop_tool("reset")),
    Component("agent_system_prompt", "prompt", False, "remove the optional agent system message", _agent_system_prompt_off),
    Component("agent_workflow_guidance", "prompt", False, "retain only the question-bearing objective sentence", _agent_workflow_guidance_off),
    Component("high_quality_evidence_depth", "preset", False, "revert evidence_k from 20 to the default 10", _high_quality_evidence_depth_off),
    Component("high_quality_concurrency", "preset", False, "revert evidence concurrency from 10 to the default 4", _high_quality_concurrency_off),
    Component("high_quality_chunking", "preset", True, "revert chunk_chars from 7000 to the default 5000", _high_quality_chunking_off),
    Component("evidence_retrieval", "subsystem", False, "process available texts without embedding retrieval", _evidence_retrieval_off),
    Component("evidence_summarization", "subsystem", False, "pass retrieved text without LLM evidence summaries", _evidence_summarization_off),
    Component("structured_evidence_json", "subsystem", False, "use the supported plain-text evidence summary path", _structured_evidence_json_off),
    Component("lazy_evidence_fallback", "subsystem", False, "do not gather evidence implicitly inside gen_answer", _lazy_evidence_fallback_off),
    Component("answer_iteration", "subsystem", False, "do not inject a prior generated answer into later synthesis", _answer_iteration_off),
    Component("citation_metadata_context", "prompt", False, "omit formatted citation metadata from each context", _citation_metadata_context_off),
    Component("multimodal_processing", "subsystem", True, "parse the frozen PDFs as text only", _multimodal_processing_off),
    Component("document_metadata_extraction", "subsystem", True, "disable local document-detail extraction", _document_metadata_extraction_off),
    Component("gen_answer_tool", "tool", False, "remove gen_answer", _drop_tool("gen_answer")),
    Component("complete_tool", "tool", False, "remove complete", _drop_tool("complete")),
)

BY_NAME = {component.name: component for component in COMPONENTS}


def component_names() -> list[str]:
    return [component.name for component in COMPONENTS]


def apply_single_off(settings: Any, component_name: str) -> Component:
    try:
        component = BY_NAME[component_name]
    except KeyError as exc:
        raise ValueError(f"Unknown PaperQA2 component: {component_name}") from exc
    component.apply_off(settings)
    return component


def apply_all_off(settings: Any) -> tuple[Component, ...]:
    """Apply every pruning-candidate ablation to one settings object."""
    for component in COMPONENTS:
        component.apply_off(settings)
    return COMPONENTS


def inventory() -> list[dict[str, Any]]:
    return [
        {
            "name": component.name,
            "kind": component.kind,
            "affects_index": component.affects_index,
            "off_description": component.off_description,
        }
        for component in COMPONENTS
    ]


if __name__ == "__main__":
    import json

    print(json.dumps(inventory(), indent=2, sort_keys=True))
