#!/usr/bin/env python3
"""Run one task-sticky shard of the frozen PaperQA2 validation ladder."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from components import apply_single_off
from paperqa_infrastructure_guard import install_infrastructure_exception_guard
from run_full import enforce_frozen_corpus_boundary, make_settings, token_telemetry
from run_single_off import (
    install_api_host_alias,
    run_items_bounded,
    validate_index_artifacts,
    write_json_atomic,
)
from runner_common import (
    append_jsonl,
    exclusive_runner_lock,
    format_agent_prompt,
    make_result_row,
    read_jsonl,
)
from validation_ladder import (
    CONFIG_SIZES,
    INDEX_COMPONENTS,
    PRIMARY_CONFIGS,
    SEARCH_DETAIL_CONFIGS,
    apply_ladder_config,
    balanced_config_order,
    disabled_components,
    index_label,
    kept_components,
    parsing_signature,
    shard_for_task,
)


HARNESS = "paperqa2-high-quality-median-ladder"
INDEX_LABELS = {"low-parsing", "chunk5000-mm-nodoc", "chunk5000-mm-doc", "full-parsing", "chunk5000-nomm-doc", "chunk7000-nomm-doc"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preprocess-dir", type=Path, required=True)
    parser.add_argument("--schedule-output", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--phase", choices=("primary", "search_detail"), required=True)
    parser.add_argument("--config", action="append", choices=tuple(CONFIG_SIZES))
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-connect-host", required=True)
    parser.add_argument("--api-connect-port", type=int, required=True)
    parser.add_argument("--endpoint-job-id", required=True)
    parser.add_argument(
        "--allowed-prior-endpoint-job-id",
        action="append",
        type=int,
        default=[],
        help=(
            "Accept canonical rows produced by this prior endpoint when resuming "
            "under an explicitly audited within-environment comparison policy."
        ),
    )
    parser.add_argument("--api-key", default="audit-no-key")
    parser.add_argument("--model", default="Qwen3.5-122B-A10B")
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument(
        "--checkpoint-after-new-canonical-cells",
        type=int,
        help=(
            "Exit cleanly without a completion marker after this many new "
            "canonical cells. Intended for bounded infrastructure canaries."
        ),
    )
    parser.add_argument(
        "--completion-marker",
        type=Path,
        help="Optional subset/repair marker; defaults to the phase shard marker.",
    )
    return parser.parse_args()


def config_names(args: argparse.Namespace) -> tuple[str, ...]:
    defaults = PRIMARY_CONFIGS if args.phase == "primary" else SEARCH_DETAIL_CONFIGS
    names = tuple(args.config or defaults)
    if len(names) != len(set(names)):
        raise ValueError("Duplicate ladder configs requested")
    disallowed = {"A1", "A2"} & set(names)
    if disallowed:
        raise ValueError(f"Excluded early ladder configs requested: {sorted(disallowed)}")
    return names


def substrate_index_name(run_tag: str, label: str) -> str:
    if label not in INDEX_LABELS:
        raise ValueError(f"Unknown index label: {label}")
    return f"{run_tag}-{label}-paperqa2"


def base_settings_args(args: argparse.Namespace, index_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        max_output_tokens=args.max_output_tokens,
        embedding=args.embedding,
        corpus_dir=args.corpus_dir,
        index_dir=args.index_dir,
        index_name=index_name,
        off_component=None,
    )


def make_config_settings(args: argparse.Namespace, config_name: str) -> Any:
    label = index_label(config_name)
    settings = make_settings(
        base_settings_args(args, substrate_index_name(args.run_tag, label))
    )
    apply_ladder_config(settings, config_name)
    return settings


def make_substrate_settings(args: argparse.Namespace, label: str) -> Any:
    settings = make_settings(
        base_settings_args(args, substrate_index_name(args.run_tag, label))
    )
    if label == "low-parsing":
        for name in INDEX_COMPONENTS:
            apply_single_off(settings, name)
    elif label == "chunk5000-mm-nodoc":
        apply_single_off(settings, "high_quality_chunking")
        apply_single_off(settings, "document_metadata_extraction")
    elif label == "chunk5000-mm-doc":
        apply_single_off(settings, "high_quality_chunking")
    elif label == "chunk5000-nomm-doc":
        apply_single_off(settings, "high_quality_chunking")
        apply_single_off(settings, "multimodal_processing")
    elif label == "chunk7000-nomm-doc":
        apply_single_off(settings, "multimodal_processing")
    elif label != "full-parsing":
        raise ValueError(f"Unknown index label: {label}")
    return settings


def output_path(args: argparse.Namespace, config_name: str) -> Path:
    return args.output_dir / f"{config_name}.shard{args.shard_index}.jsonl"


def attempts_path(args: argparse.Namespace, config_name: str) -> Path:
    return (
        args.output_dir
        / "attempts"
        / f"{config_name}.shard{args.shard_index}.jsonl"
    )


def load_and_reconcile(
    args: argparse.Namespace,
    names: tuple[str, ...],
    settings_by_config: dict[str, Any],
    frozen_ids: set[str],
) -> dict[str, set[str]]:
    completed: dict[str, set[str]] = {}
    allowed_endpoint_job_ids = {
        int(args.endpoint_job_id),
        *(int(value) for value in args.allowed_prior_endpoint_job_id),
    }
    recovery_dir = args.output_dir / ".jsonl_recovery"
    for name in names:
        canonical_path = output_path(args, name)
        canonical = (
            read_jsonl(
                canonical_path,
                repair_torn_tail=True,
                recovery_dir=recovery_dir,
            )
            if canonical_path.exists()
            else []
        )
        ids = [str(row["question_id"]) for row in canonical]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"Duplicate canonical IDs for {name}/shard{args.shard_index}")
        if not set(ids) <= frozen_ids:
            raise RuntimeError(f"Non-frozen canonical ID for {name}/shard{args.shard_index}")
        if any(row.get("run_status") != "ok" for row in canonical):
            raise RuntimeError(f"Non-success canonical row for {name}/shard{args.shard_index}")
        expected_hash = settings_by_config[name].md5
        for row in canonical:
            telemetry = row.get("telemetry") or {}
            if telemetry.get("settings_md5") != expected_hash:
                raise RuntimeError(f"Settings hash mismatch for {name}/shard{args.shard_index}")
            if telemetry.get("ladder_config") != name:
                raise RuntimeError(f"Config identity mismatch for {name}/shard{args.shard_index}")
            if int(telemetry.get("endpoint_job_id")) not in allowed_endpoint_job_ids:
                raise RuntimeError(
                    f"Unapproved endpoint for {name}/shard{args.shard_index}"
                )
        completed[name] = set(ids)

        attempt_file = attempts_path(args, name)
        if not attempt_file.exists():
            continue
        attempts = read_jsonl(
            attempt_file,
            repair_torn_tail=True,
            recovery_dir=recovery_dir,
        )
        for row in attempts:
            question_id = str(row.get("question_id"))
            telemetry = row.get("telemetry") or {}
            if (
                question_id in frozen_ids
                and question_id not in completed[name]
                and row.get("run_status") == "ok"
                and telemetry.get("settings_md5") == expected_hash
                and telemetry.get("ladder_config") == name
                and int(telemetry.get("endpoint_job_id")) in allowed_endpoint_job_ids
            ):
                append_jsonl(canonical_path, row)
                completed[name].add(question_id)
    return completed


async def prepare_indexes(args: argparse.Namespace, names: tuple[str, ...]) -> None:
    from paperqa.agents.search import get_directory_index

    expected_documents = sum(path.is_file() for path in args.corpus_dir.rglob("*"))
    if expected_documents < 1:
        raise RuntimeError("Frozen validation corpus is empty")
    labels = tuple(dict.fromkeys(index_label(name) for name in names))
    for label in labels:
        settings = make_substrate_settings(args, label)
        metadata_path = args.preprocess_dir / f"{label}.json"
        if metadata_path.exists():
            prior = json.loads(metadata_path.read_text())
            if prior.get("status") != "ok" or prior.get("settings_md5") != settings.md5:
                raise RuntimeError(f"Incompatible preprocess metadata for {label}")
            await get_directory_index(settings=settings, build=False)
            artifacts = validate_index_artifacts(settings, expected_documents)
            print(json.dumps({"preprocess": label, "status": "reused", **artifacts}), flush=True)
            continue
        started = time.monotonic()
        # A fresh checkout builds every required parsing substrate. Resume reuses
        # only artifacts whose metadata and settings hashes were validated above.
        await get_directory_index(settings=settings, build=True)
        artifacts = validate_index_artifacts(settings, expected_documents)
        payload = {
            "schema_version": 1,
            "status": "ok",
            "label": label,
            "wall_seconds": time.monotonic() - started,
            "index_name": settings.agent.index.name,
            "settings_md5": settings.md5,
            "parsing_signature": list(parsing_signature(next(name for name in names if index_label(name) == label))),
            "external_metadata_clients": "disabled_by_frozen_corpus_policy",
            **artifacts,
        }
        write_json_atomic(metadata_path, payload)
        print(json.dumps({"preprocess": label, "status": "ok", **artifacts}), flush=True)


def validate_preprocessed_indexes(
    args: argparse.Namespace,
    names: tuple[str, ...],
    expected_documents: int,
) -> None:
    for label in tuple(dict.fromkeys(index_label(name) for name in names)):
        metadata_path = args.preprocess_dir / f"{label}.json"
        if not metadata_path.exists():
            raise RuntimeError(f"Missing preprocess metadata for {label}")
        settings = make_substrate_settings(args, label)
        prior = json.loads(metadata_path.read_text())
        if prior.get("status") != "ok" or prior.get("settings_md5") != settings.md5:
            raise RuntimeError(f"Invalid preprocess metadata for {label}")
        validate_index_artifacts(settings, expected_documents)


async def amain(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if (
        args.checkpoint_after_new_canonical_cells is not None
        and args.checkpoint_after_new_canonical_cells < 1
    ):
        raise SystemExit("--checkpoint-after-new-canonical-cells must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("--shard-index must be within --shard-count")
    names = config_names(args)
    install_api_host_alias(
        args.api_base,
        args.api_connect_host,
        args.api_connect_port,
    )
    enforce_frozen_corpus_boundary()
    tasks = read_jsonl(args.tasks)
    if len(tasks) != len({str(task["question_id"]) for task in tasks}):
        raise RuntimeError("Validation tasks contain duplicate question IDs")
    shard_tasks = [
        (position, task)
        for position, task in enumerate(tasks)
        if shard_for_task(position, args.shard_count) == args.shard_index
    ]
    frozen_ids = {str(task["question_id"]) for _, task in shard_tasks}
    expected_documents = sum(path.is_file() for path in args.corpus_dir.rglob("*"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.preprocess_dir.mkdir(parents=True, exist_ok=True)
    if args.preprocess_only:
        await prepare_indexes(args, names)
        preprocess_marker = args.completion_marker or (
            args.preprocess_dir / "preprocess.complete.json"
        )
        write_json_atomic(
            preprocess_marker,
            {
                "schema_version": 1,
                "status": "complete",
                "run_tag": args.run_tag,
                "configs": list(names),
                "index_labels": list(
                    dict.fromkeys(index_label(name) for name in names)
                ),
                "expected_documents": expected_documents,
                "logical_api_base": args.api_base,
            },
        )
        return
    validate_preprocessed_indexes(args, names, expected_documents)

    settings_by_config = {name: make_config_settings(args, name) for name in names}
    settings_by_worker = [settings_by_config]
    for _ in range(1, args.workers):
        candidate = {name: make_config_settings(args, name) for name in names}
        if any(candidate[name].md5 != settings_by_config[name].md5 for name in names):
            raise RuntimeError("Per-worker ladder settings hashes differ")
        settings_by_worker.append(candidate)

    schedule = {
        "schema_version": 1,
        "run_tag": args.run_tag,
        "phase": args.phase,
        "configs": list(names),
        "config_definitions": {
            name: {
                "kept_components": list(kept_components(name)),
                "disabled_components": list(disabled_components(name)),
                "index_label": index_label(name),
                "settings_md5": settings_by_config[name].md5,
            }
            for name in names
        },
        "schedule": "task-major cyclic Latin config order within phase",
        "endpoint_blocking": (
            "within-environment reuse of explicitly audited prior endpoints"
            if args.allowed_prior_endpoint_job_id
            else "every config for a task uses one frozen serve job/host"
        ),
        "endpoint_job_id": int(args.endpoint_job_id),
        "endpoint_host": args.api_connect_host,
        "allowed_prior_endpoint_job_ids": sorted(
            set(int(value) for value in args.allowed_prior_endpoint_job_id)
        ),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "task_positions": [position for position, _ in shard_tasks],
        "n_tasks": len(shard_tasks),
        "workers": args.workers,
        "checkpoint_after_new_canonical_cells": (
            args.checkpoint_after_new_canonical_cells
        ),
    }
    write_json_atomic(args.schedule_output, schedule)

    completed = load_and_reconcile(
        args,
        names,
        settings_by_config,
        frozen_ids,
    )
    install_infrastructure_exception_guard()
    from paperqa import agent_query

    consecutive_hard_errors = 0
    new_canonical_cells = 0
    checkpoint_reached = asyncio.Event()
    for attempt_round in range(1, 4):
        pending_tasks = [
            (position, task)
            for position, task in shard_tasks
            if any(str(task["question_id"]) not in completed[name] for name in names)
        ]
        if not pending_tasks:
            break

        async def run_task(worker_index: int, item: tuple[int, dict[str, Any]]) -> None:
            nonlocal consecutive_hard_errors, new_canonical_cells
            task_position, task = item
            question_id = str(task["question_id"])
            order = balanced_config_order(task_position, names)
            for within_phase_position, name in enumerate(order):
                if question_id in completed[name]:
                    continue
                settings = settings_by_worker[worker_index][name]
                started = time.monotonic()
                base_telemetry = {
                    "settings_md5": settings.md5,
                    "ladder_config": name,
                    "kept_components": list(kept_components(name)),
                    "disabled_components": list(disabled_components(name)),
                    "index_label": index_label(name),
                    "task_position": task_position,
                    "within_phase_position": within_phase_position,
                    "phase": args.phase,
                    "attempt_round": attempt_round,
                    "runner_worker": worker_index,
                    "shard_count": args.shard_count,
                    "shard_index": args.shard_index,
                    "endpoint_job_id": int(args.endpoint_job_id),
                    "endpoint_host": args.api_connect_host,
                }
                try:
                    response = await agent_query(
                        query=format_agent_prompt(task),
                        settings=settings,
                        agent_type=settings.agent.agent_type,
                    )
                    row = make_result_row(
                        task=task,
                        harness=f"{HARNESS}-{name}",
                        answer=response.session.answer,
                        wall_seconds=time.monotonic() - started,
                        telemetry={
                            **token_telemetry(response.session),
                            "agent_status": str(response.status),
                            **base_telemetry,
                        },
                    )
                except Exception as exc:
                    row = make_result_row(
                        task=task,
                        harness=f"{HARNESS}-{name}",
                        answer=None,
                        wall_seconds=time.monotonic() - started,
                        telemetry=base_telemetry,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                append_jsonl(attempts_path(args, name), row)
                if row["run_status"] == "ok":
                    append_jsonl(output_path(args, name), row)
                    completed[name].add(question_id)
                    new_canonical_cells += 1
                    if (
                        args.checkpoint_after_new_canonical_cells is not None
                        and new_canonical_cells
                        >= args.checkpoint_after_new_canonical_cells
                    ):
                        checkpoint_reached.set()
                    consecutive_hard_errors = 0
                else:
                    consecutive_hard_errors += 1
                print(
                    json.dumps(
                        {
                            "phase": args.phase,
                            "config": name,
                            "task_position": task_position,
                            "shard_index": args.shard_index,
                            "endpoint_job_id": int(args.endpoint_job_id),
                            "status": row["run_status"],
                            "consecutive_hard_errors": consecutive_hard_errors,
                            "canonical_cells": sum(len(ids) for ids in completed.values()),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if consecutive_hard_errors >= 3:
                    raise SystemExit("Stopped after three consecutive hard runner errors")

        await run_items_bounded(
            pending_tasks,
            args.workers,
            run_task,
            stop_requested=checkpoint_reached.is_set,
        )
        if checkpoint_reached.is_set():
            print(
                json.dumps(
                    {
                        "status": "checkpoint",
                        "new_canonical_cells": new_canonical_cells,
                        "canonical_cells": sum(
                            len(ids) for ids in completed.values()
                        ),
                        "completion_marker_written": False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return

    missing = sum(len(shard_tasks) - len(completed[name]) for name in names)
    if missing:
        raise SystemExit(f"Hard-error cells remain after bounded retries: {missing}")
    canonical_by_config = {
        name: read_jsonl(output_path(args, name)) for name in names
    }
    all_fail = [
        name
        for name, rows in canonical_by_config.items()
        if rows
        and all(
            (row.get("telemetry") or {}).get("agent_status") == "fail"
            for row in rows
        )
    ]
    if all_fail:
        raise SystemExit(
            "Result-validity gate refused all-FAIL config(s): "
            + ",".join(all_fail)
        )
    marker_path = args.completion_marker or (
        args.output_dir / f"{args.phase}.shard{args.shard_index}.complete.json"
    )
    write_json_atomic(
        marker_path,
        {
            "schema_version": 1,
            "status": "complete",
            "run_tag": args.run_tag,
            "phase": args.phase,
            "configs": list(names),
            "n_tasks": len(shard_tasks),
            "canonical_cells": len(shard_tasks) * len(names),
            "agent_status_counts": {
                name: {
                    status: sum(
                        (row.get("telemetry") or {}).get("agent_status") == status
                        for row in rows
                    )
                    for status in sorted(
                        {
                            str((row.get("telemetry") or {}).get("agent_status"))
                            for row in rows
                        }
                    )
                }
                for name, rows in canonical_by_config.items()
            },
            "result_validity_gate": "passed",
            "endpoint_job_id": int(args.endpoint_job_id),
            "endpoint_host": args.api_connect_host,
            "allowed_prior_endpoint_job_ids": sorted(
                set(int(value) for value in args.allowed_prior_endpoint_job_id)
            ),
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
        },
    )


if __name__ == "__main__":
    parsed = parse_args()
    lock = parsed.output_dir / f".{parsed.phase}.shard{parsed.shard_index}.runner.lock"
    with exclusive_runner_lock(
        lock,
        {
            "hostname": socket.gethostname(),
            "process_id": os.getpid(),
            "run_tag": parsed.run_tag,
            "phase": parsed.phase,
            "shard_count": parsed.shard_count,
            "shard_index": parsed.shard_index,
            "endpoint_job_id": parsed.endpoint_job_id,
            "endpoint_host": parsed.api_connect_host,
            "workers": parsed.workers,
        },
    ):
        asyncio.run(amain(parsed))
