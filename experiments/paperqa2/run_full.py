#!/usr/bin/env python3
"""Run pinned PaperQA2 high_quality with a local OpenAI-compatible model."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from runner_common import (
    append_jsonl,
    format_agent_prompt,
    make_result_row,
    pending_tasks,
    read_jsonl,
)
from components import apply_single_off, component_names


def local_llm_config(
    alias: str,
    served_model: str,
    api_base: str,
    api_key: str,
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "name": alias,
        "model_list": [
            {
                "model_name": alias,
                "litellm_params": {
                    "model": f"openai/{served_model}",
                    "api_base": api_base,
                    "api_key": api_key,
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "timeout": 600,
                    "extra_body": {
                        "chat_template_kwargs": {"enable_thinking": False}
                    },
                },
            }
        ],
    }


def make_settings(args: argparse.Namespace) -> Any:
    from paperqa import Settings

    settings = Settings.from_name("high_quality")
    alias = "audit-qwen35-122b"
    config = local_llm_config(alias, args.model, args.api_base, args.api_key, args.max_output_tokens)
    settings.llm = alias
    settings.llm_config = config
    settings.summary_llm = alias
    settings.summary_llm_config = config
    settings.agent.agent_llm = alias
    settings.agent.agent_llm_config = config
    settings.parsing.enrichment_llm = alias
    settings.parsing.enrichment_llm_config = config
    settings.embedding = args.embedding
    # The public runner selects device visibility before importing dependencies.
    settings.embedding_config = {"device": "cuda" if os.environ.get("SHARP_GPU_EMBEDDINGS") == "1" else "cpu"}
    settings.agent.index.paper_directory = args.corpus_dir.resolve()
    settings.agent.index.index_directory = args.index_dir.resolve()
    settings.agent.index.name = args.index_name
    settings.agent.index.use_absolute_paper_directory = True
    settings.agent.rebuild_index = False
    if args.off_component:
        apply_single_off(settings, args.off_component)
    return settings


def enforce_frozen_corpus_boundary() -> None:
    """Prevent PaperQA parsing from consulting live scholarly metadata APIs.

    Use local citation and structured-metadata extraction without querying
    Crossref or Semantic Scholar, so indexing depends only on the fixed PDFs.
    """
    import paperqa.docs as paperqa_docs
    import pypdf.filters as pypdf_filters
    from paperqa.clients.client_models import MetadataProvider

    class FrozenCorpusMetadataProvider(MetadataProvider[dict[str, Any]]):
        """Retain locally extracted fields without consulting a remote service."""

        def query_factory(self, query: dict[str, Any]) -> dict[str, Any]:
            return query

        async def _query(self, query: dict[str, Any]) -> None:
            return None

    # DocMetadataClient requires at least one provider. This provider deliberately
    # returns no enrichment, after which PaperQA keeps the citation/title/DOI it
    # extracted locally from the frozen PDF.
    paperqa_docs.DEFAULT_CLIENTS = (FrozenCorpusMetadataProvider,)
    # pypdf documents this value as configurable. One hash-validated frozen
    # decoy contains a legitimate image stream just above the 75 MB default;
    # retain a finite safety bound while allowing that PDF to be parsed.
    pypdf_filters.ZLIB_MAX_OUTPUT_LENGTH = 150_000_000


def token_telemetry(session: Any) -> dict[str, Any]:
    input_tokens = sum(pair[0] for pair in session.token_counts.values())
    output_tokens = sum(pair[1] for pair in session.token_counts.values())
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": None,
        "llm_calls": None,
        "token_counts_by_model": session.token_counts,
        "tool_calls": session.tool_history,
        "paperqa_cost": session.cost,
        "n_contexts": len(session.contexts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--index-name", default="smoke-realized-v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preprocess-output", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="audit-no-key")
    parser.add_argument("--model", default="Qwen3.5-122B-A10B")
    parser.add_argument("--embedding", default="st-multi-qa-MiniLM-L6-cos-v1")
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--off-component", choices=component_names())
    return parser.parse_args()


async def amain(args: argparse.Namespace) -> None:
    from paperqa import agent_query
    from paperqa.agents.search import get_directory_index

    enforce_frozen_corpus_boundary()
    settings = make_settings(args)
    harness_name = (
        "paperqa2-high-quality-v2026.03.18"
        if not args.off_component
        else f"paperqa2-high-quality-v2026.03.18-off-{args.off_component}"
    )
    args.index_dir.mkdir(parents=True, exist_ok=True)
    preprocess_started = time.monotonic()
    try:
        index = await get_directory_index(settings=settings, build=True)
        preprocess = {
            "schema_version": 1,
            "status": "ok",
            "wall_seconds": time.monotonic() - preprocess_started,
            "index_name": index.index_name,
            "settings_md5": settings.md5,
            "external_metadata_clients": "disabled_by_frozen_corpus_policy",
            "pypdf_zlib_max_output_length": 150_000_000,
            "note": "Offline index construction; token usage is not attributed by upstream PaperQA.",
        }
    except Exception as exc:
        preprocess = {
            "schema_version": 1,
            "status": "error",
            "wall_seconds": time.monotonic() - preprocess_started,
            "error": f"{type(exc).__name__}: {exc}",
        }
        args.preprocess_output.parent.mkdir(parents=True, exist_ok=True)
        args.preprocess_output.write_text(json.dumps(preprocess, indent=2, sort_keys=True) + "\n")
        raise
    args.preprocess_output.parent.mkdir(parents=True, exist_ok=True)
    args.preprocess_output.write_text(json.dumps(preprocess, indent=2, sort_keys=True) + "\n")

    tasks = pending_tasks(read_jsonl(args.tasks), args.output)
    for task in tasks:
        started = time.monotonic()
        try:
            response = await agent_query(
                query=format_agent_prompt(task),
                settings=settings,
                agent_type=settings.agent.agent_type,
            )
            answer = response.session.answer
            row = make_result_row(
                task=task,
                harness=harness_name,
                answer=answer,
                wall_seconds=time.monotonic() - started,
                telemetry={
                    **token_telemetry(response.session),
                    "agent_status": str(response.status),
                    "settings_md5": settings.md5,
                },
            )
        except Exception as exc:
            row = make_result_row(
                task=task,
                harness=harness_name,
                answer=None,
                wall_seconds=time.monotonic() - started,
                telemetry={"settings_md5": settings.md5},
                error=f"{type(exc).__name__}: {exc}",
            )
        append_jsonl(args.output, row)
        print(
            json.dumps(
                {
                    "question_id": task["question_id"],
                    "status": row["run_status"],
                    "parsed_choice": row["parsed_choice"],
                }
            )
        )


if __name__ == "__main__":
    asyncio.run(amain(parse_args()))
