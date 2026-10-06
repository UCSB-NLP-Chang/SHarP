#!/usr/bin/env python3
"""Run every PaperQA2 single-off arm in a task-major balanced schedule."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import time
from collections.abc import Awaitable, Callable, Sequence
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar
from urllib.parse import urlparse

from components import BY_NAME, component_names, inventory
from run_full import enforce_frozen_corpus_boundary, make_settings, token_telemetry
from runner_common import (
    append_jsonl,
    exclusive_runner_lock,
    format_agent_prompt,
    make_result_row,
    read_jsonl,
)


T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preprocess-dir", type=Path, required=True)
    parser.add_argument("--schedule-output", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument(
        "--api-connect-host",
        help=(
            "Route the hostname in --api-base to this infrastructure host while "
            "preserving the logical API URL inside Settings and its md5."
        ),
    )
    parser.add_argument("--api-key", default="audit-no-key")
    parser.add_argument("--model", default="Qwen3.5-122B-A10B")
    parser.add_argument("--embedding", default="st-multi-qa-MiniLM-L6-cos-v1")
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume-drain-workers", type=int, default=1)
    parser.add_argument(
        "--pause-file",
        type=Path,
        help=(
            "When this sentinel exists, workers finish their current whole task "
            "and exit before acquiring another task."
        ),
    )
    parser.add_argument(
        "--preprocess-label",
        action="append",
        choices=(
            "shared-full-parsing",
            "high_quality_chunking",
            "multimodal_processing",
            "document_metadata_extraction",
        ),
        help="Build only this preprocessing index; repeat for multiple labels.",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Exit after preprocessing without creating schedule or rollout rows.",
    )
    return parser.parse_args()


def install_api_host_alias(
    logical_api_base: str,
    connect_host: str | None,
    connect_port: int | None = None,
) -> None:
    """Route a stable logical API hostname to a replacement infrastructure host.

    The logical URL is intentionally retained in PaperQA Settings so endpoint
    replacement cannot invalidate frozen settings hashes or preprocessing reuse.
    """
    if not connect_host:
        return
    logical_url = urlparse(logical_api_base)
    logical_host = logical_url.hostname
    if not logical_host:
        raise ValueError(f"Cannot determine hostname from API base: {logical_api_base}")
    logical_port = logical_url.port or (443 if logical_url.scheme == "https" else 80)
    original_getaddrinfo = socket.getaddrinfo

    def aliased_getaddrinfo(
        host: str,
        port: int | str | None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        is_logical_endpoint = host == logical_host and (
            port is None or int(port) == logical_port
        )
        routed_host = connect_host if is_logical_endpoint else host
        routed_port = connect_port if is_logical_endpoint and connect_port else port
        return original_getaddrinfo(routed_host, routed_port, *args, **kwargs)

    socket.getaddrinfo = aliased_getaddrinfo


def arm_index_name(run_tag: str, component_name: str) -> str:
    component = BY_NAME[component_name]
    suffix = component_name if component.affects_index else "shared-full-parsing"
    return f"{run_tag}-{suffix}-paperqa2"


def balanced_component_order(task_position: int, names: list[str]) -> list[str]:
    """Cyclic Latin order: each arm occupies each temporal position evenly."""
    shift = task_position % len(names)
    return names[shift:] + names[:shift]


async def run_items_bounded(
    items: Sequence[T],
    workers: int,
    run_item: Callable[[int, T], Awaitable[None]],
    *,
    stop_requested: Callable[[], bool] | None = None,
) -> None:
    """Run whole items concurrently, assigning each item to exactly one worker."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if not items:
        return
    queue: asyncio.Queue[T] = asyncio.Queue()
    for item in items:
        queue.put_nowait(item)

    async def worker(worker_index: int) -> None:
        while True:
            if stop_requested is not None and stop_requested():
                return
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await run_item(worker_index, item)
            finally:
                queue.task_done()

    tasks = [
        asyncio.create_task(worker(worker_index))
        for worker_index in range(min(workers, len(items)))
    ]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def partition_pending_tasks(
    tasks: list[dict[str, Any]],
    names: list[str],
    completed: dict[str, set[str]],
) -> tuple[list[tuple[int, dict[str, Any]]], list[tuple[int, dict[str, Any]]]]:
    """Separate partially completed tasks from untouched pending tasks."""
    partial: list[tuple[int, dict[str, Any]]] = []
    fresh: list[tuple[int, dict[str, Any]]] = []
    for task_position, task in enumerate(tasks):
        question_id = str(task["question_id"])
        completed_arms = sum(question_id in completed[name] for name in names)
        if 0 < completed_arms < len(names):
            partial.append((task_position, task))
        elif completed_arms == 0:
            fresh.append((task_position, task))
    return partial, fresh


def settings_args(args: argparse.Namespace, component_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        max_output_tokens=args.max_output_tokens,
        embedding=args.embedding,
        corpus_dir=args.corpus_dir,
        index_dir=args.index_dir,
        index_name=arm_index_name(args.run_tag, component_name),
        off_component=component_name,
    )


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_index_artifacts(settings: Any, expected_documents: int) -> dict[str, Any]:
    """Require the claimed reusable index to contain every frozen document."""
    index_root = (
        Path(settings.agent.index.index_directory) / settings.agent.index.name
    )
    files_archives = list(index_root.rglob("files.zip"))
    document_archives = [
        path
        for path in index_root.rglob("*.zip")
        if path.name != "files.zip" and ".jsonl_recovery" not in path.parts
    ]
    if len(files_archives) != 1 or len(document_archives) != expected_documents:
        raise RuntimeError(
            "Frozen index artifact validation failed for "
            f"{settings.agent.index.name}: files.zip={len(files_archives)}, "
            f"document_archives={len(document_archives)}, "
            f"expected_documents={expected_documents}"
        )
    return {
        "index_root": str(index_root),
        "files_zip": len(files_archives),
        "document_archives": len(document_archives),
    }


def reconcile_successful_attempts(
    *,
    output_dir: Path,
    names: list[str],
    settings_by_arm: dict[str, Any],
    frozen_question_ids: set[str],
) -> tuple[dict[str, set[str]], dict[str, int]]:
    """Promote fsynced successful attempts missing from canonical after a crash."""
    attempts_dir = output_dir / "attempts"
    completed: dict[str, set[str]] = {}
    promoted: dict[str, int] = {}
    recovery_dir = output_dir / ".jsonl_recovery"
    for name in names:
        canonical_path = output_dir / f"{name}.jsonl"
        canonical_rows = (
            read_jsonl(
                canonical_path,
                repair_torn_tail=True,
                recovery_dir=recovery_dir,
            )
            if canonical_path.exists()
            else []
        )
        canonical_ids = [str(row["question_id"]) for row in canonical_rows]
        if len(canonical_ids) != len(set(canonical_ids)):
            raise RuntimeError(f"Duplicate canonical question IDs in {canonical_path}")
        if not set(canonical_ids) <= frozen_question_ids:
            raise RuntimeError(f"Non-frozen canonical question ID in {canonical_path}")
        if any(row.get("run_status") != "ok" for row in canonical_rows):
            raise RuntimeError(f"Non-success row found in canonical file {canonical_path}")
        completed[name] = set(canonical_ids)
        promoted[name] = 0
        attempts_path = attempts_dir / f"{name}.jsonl"
        if not attempts_path.exists():
            continue
        attempts = read_jsonl(
            attempts_path,
            repair_torn_tail=True,
            recovery_dir=recovery_dir,
        )
        for row in attempts:
            question_id = str(row.get("question_id"))
            telemetry = row.get("telemetry") or {}
            if (
                question_id in frozen_question_ids
                and question_id not in completed[name]
                and row.get("run_status") == "ok"
                and telemetry.get("settings_md5") == settings_by_arm[name].md5
                and telemetry.get("off_component") == name
            ):
                append_jsonl(canonical_path, row)
                completed[name].add(question_id)
                promoted[name] += 1
    return completed, promoted


def write_pause_marker(
    args: argparse.Namespace,
    *,
    missing_canonical_cells: int,
    completed: dict[str, set[str]],
) -> None:
    write_json_atomic(
        args.output_dir / "graceful_pause.complete.json",
        {
            "schema_version": 1,
            "paused_at": datetime.now(timezone.utc).isoformat(),
            "run_tag": args.run_tag,
            "workers": args.workers,
            "missing_canonical_cells": missing_canonical_cells,
            "canonical_cells": sum(len(ids) for ids in completed.values()),
            "pause_file": str(args.pause_file),
            "boundary": "all active workers finished their current whole task",
        },
    )


async def preprocess_indexes(
    args: argparse.Namespace,
    settings_by_arm: dict[str, Any],
    selected_labels: set[str] | None = None,
) -> None:
    from paperqa.agents.search import get_directory_index

    expected_documents = sum(
        path.is_file() for path in args.corpus_dir.rglob("*")
    )
    if expected_documents < 1:
        raise RuntimeError(f"Frozen corpus is empty: {args.corpus_dir}")

    # All non-index ablations share the untouched full parsing configuration.
    # Build it with an unablated setting so the shared index is definitionally
    # independent of whichever non-index arm happens to run first.
    full_args = settings_args(args, component_names()[0])
    full_args.off_component = None
    full_args.index_name = arm_index_name(args.run_tag, component_names()[0])
    build_settings: dict[str, Any] = {"shared-full-parsing": make_settings(full_args)}
    for name, settings in settings_by_arm.items():
        if BY_NAME[name].affects_index:
            build_settings[name] = settings

    if selected_labels is not None:
        build_settings = {
            label: settings
            for label, settings in build_settings.items()
            if label in selected_labels
        }
        missing = selected_labels - set(build_settings)
        if missing:
            raise ValueError(f"Unknown preprocessing labels: {sorted(missing)}")

    for label, settings in build_settings.items():
        output = args.preprocess_dir / f"{label}.json"
        if output.exists():
            prior = json.loads(output.read_text())
            if prior.get("status") == "ok" and prior.get("settings_md5") == settings.md5:
                await get_directory_index(settings=settings, build=False)
                artifacts = validate_index_artifacts(settings, expected_documents)
                print(
                    json.dumps(
                        {"preprocess": label, "status": "reused", **artifacts}
                    ),
                    flush=True,
                )
                continue
        started = time.monotonic()
        try:
            index = await get_directory_index(settings=settings, build=True)
            artifacts = validate_index_artifacts(settings, expected_documents)
            payload = {
                "schema_version": 1,
                "status": "ok",
                "label": label,
                "wall_seconds": time.monotonic() - started,
                "index_name": index.index_name,
                "settings_md5": settings.md5,
                "external_metadata_clients": "disabled_by_frozen_corpus_policy",
                "pypdf_zlib_max_output_length": 150_000_000,
                **artifacts,
            }
        except Exception as exc:
            payload = {
                "schema_version": 1,
                "status": "error",
                "label": label,
                "wall_seconds": time.monotonic() - started,
                "settings_md5": settings.md5,
                "error": f"{type(exc).__name__}: {exc}",
            }
            write_json_atomic(output, payload)
            raise
        write_json_atomic(output, payload)
        print(json.dumps({"preprocess": label, "status": "ok"}), flush=True)


async def _amain_under_lock(args: argparse.Namespace) -> None:
    install_api_host_alias(args.api_base, args.api_connect_host)
    from paperqa import agent_query

    enforce_frozen_corpus_boundary()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if not 1 <= args.resume_drain_workers <= args.workers:
        raise SystemExit("--resume-drain-workers must be between 1 and --workers")
    names = component_names()
    tasks = read_jsonl(args.tasks)
    if len(tasks) != len({str(task["question_id"]) for task in tasks}):
        raise SystemExit("Tasks contain duplicate question IDs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.preprocess_dir.mkdir(parents=True, exist_ok=True)
    if args.pause_file is None:
        args.pause_file = args.output_dir / ".pause_requested"
    settings_by_arm = {
        name: make_settings(settings_args(args, name)) for name in names
    }
    selected_labels = set(args.preprocess_label) if args.preprocess_label else None
    await preprocess_indexes(args, settings_by_arm, selected_labels)
    if args.preprocess_only:
        return
    # PaperQA Settings may contain runtime state. Give each concurrent task worker
    # its own settings objects while keeping every settings hash identical.
    settings_by_worker = [settings_by_arm]
    for _ in range(1, args.workers):
        worker_settings = {
            name: make_settings(settings_args(args, name)) for name in names
        }
        if any(
            worker_settings[name].md5 != settings_by_arm[name].md5 for name in names
        ):
            raise SystemExit("Per-worker PaperQA settings hashes differ")
        settings_by_worker.append(worker_settings)

    schedule = {
        "schema_version": 1,
        "run_tag": args.run_tag,
        "schedule": "task-major cyclic Latin component order",
        "workers": args.workers,
        "resume_drain_workers": args.resume_drain_workers,
        "concurrency_unit": "whole task",
        "within_task_execution": "sequential cyclic Latin component order",
        "partial_task_resume": "drain serially before bounded task concurrency",
        "n_tasks": len(tasks),
        "n_components": len(names),
        "components": inventory(),
        "task_positions": [
            {
                "question_id": str(task["question_id"]),
                "component_order": balanced_component_order(position, names),
            }
            for position, task in enumerate(tasks)
        ],
    }
    write_json_atomic(args.schedule_output, schedule)

    attempts_dir = args.output_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    completed, promoted = reconcile_successful_attempts(
        output_dir=args.output_dir,
        names=names,
        settings_by_arm=settings_by_arm,
        frozen_question_ids={str(task["question_id"]) for task in tasks},
    )
    if any(promoted.values()):
        print(
            json.dumps(
                {
                    "resume_reconciliation": "promoted_successful_attempts",
                    "promoted_cells": sum(promoted.values()),
                    "promoted_by_component": promoted,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    consecutive_hard_errors = 0
    for attempt_round in range(1, 4):
        attempted_this_round = 0
        partial_tasks, _ = partition_pending_tasks(tasks, names, completed)

        async def run_task(
            worker_index: int, task_item: tuple[int, dict[str, Any]]
        ) -> None:
            nonlocal attempted_this_round, consecutive_hard_errors
            task_position, task = task_item
            for name in balanced_component_order(task_position, names):
                question_id = str(task["question_id"])
                if question_id in completed[name]:
                    continue
                attempted_this_round += 1
                settings = settings_by_worker[worker_index][name]
                started = time.monotonic()
                try:
                    response = await agent_query(
                        query=format_agent_prompt(task),
                        settings=settings,
                        agent_type=settings.agent.agent_type,
                    )
                    row = make_result_row(
                        task=task,
                        harness=f"paperqa2-high-quality-v2026.03.18-off-{name}",
                        answer=response.session.answer,
                        wall_seconds=time.monotonic() - started,
                        telemetry={
                            **token_telemetry(response.session),
                            "agent_status": str(response.status),
                            "settings_md5": settings.md5,
                            "off_component": name,
                            "task_position": task_position,
                            "within_task_arm_position": balanced_component_order(
                                task_position, names
                            ).index(name),
                            "attempt_round": attempt_round,
                            "runner_worker": worker_index,
                        },
                    )
                except Exception as exc:
                    row = make_result_row(
                        task=task,
                        harness=f"paperqa2-high-quality-v2026.03.18-off-{name}",
                        answer=None,
                        wall_seconds=time.monotonic() - started,
                        telemetry={
                            "settings_md5": settings.md5,
                            "off_component": name,
                            "task_position": task_position,
                            "within_task_arm_position": balanced_component_order(
                                task_position, names
                            ).index(name),
                            "attempt_round": attempt_round,
                            "runner_worker": worker_index,
                        },
                        error=f"{type(exc).__name__}: {exc}",
                    )
                # Preserve every attempt. Only a valid replacement enters the
                # canonical 50-task paired table; behavioral FAIL/TRUNCATED
                # responses are run_status=ok and therefore remain scored.
                append_jsonl(attempts_dir / f"{name}.jsonl", row)
                if row["run_status"] == "ok":
                    append_jsonl(args.output_dir / f"{name}.jsonl", row)
                    completed[name].add(question_id)
                    consecutive_hard_errors = 0
                else:
                    consecutive_hard_errors += 1
                print(
                    json.dumps(
                        {
                            "attempt_round": attempt_round,
                            "runner_worker": worker_index,
                            "task_position": task_position,
                            "component": name,
                            "status": row["run_status"],
                            "consecutive_hard_errors": consecutive_hard_errors,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if consecutive_hard_errors >= 3:
                    raise SystemExit(
                        "Stopped after three consecutive hard runner/infrastructure errors"
                    )
        # A killed predecessor can leave one task partially canonical. Drain all
        # such tasks serially so a worker-count transition begins only after a
        # complete task boundary across all configurations.
        await run_items_bounded(
            partial_tasks,
            args.resume_drain_workers,
            run_task,
            stop_requested=args.pause_file.exists,
        )
        _, pending_tasks = partition_pending_tasks(tasks, names, completed)
        await run_items_bounded(
            pending_tasks,
            args.workers,
            run_task,
            stop_requested=args.pause_file.exists,
        )
        missing = sum(len(tasks) - len(completed[name]) for name in names)
        print(
            json.dumps(
                {
                    "attempt_round": attempt_round,
                    "attempted_cells": attempted_this_round,
                    "missing_canonical_cells": missing,
                    "workers": args.workers,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if missing == 0:
            return
        if args.pause_file.exists():
            write_pause_marker(
                args,
                missing_canonical_cells=missing,
                completed=completed,
            )
            print(
                json.dumps(
                    {
                        "status": "gracefully_paused",
                        "canonical_cells": sum(len(ids) for ids in completed.values()),
                        "missing_canonical_cells": missing,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            raise SystemExit(75)
    raise SystemExit("Hard-error cells remain after three bounded attempt rounds")


async def amain(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_context = (
        nullcontext()
        if args.preprocess_only
        else exclusive_runner_lock(
            args.output_dir / ".runner.lock",
            {
                "hostname": socket.gethostname(),
                "process_id": os.getpid(),
                "run_tag": args.run_tag,
                "workers": args.workers,
            },
        )
    )
    with lock_context:
        await _amain_under_lock(args)


if __name__ == "__main__":
    asyncio.run(amain(parse_args()))
