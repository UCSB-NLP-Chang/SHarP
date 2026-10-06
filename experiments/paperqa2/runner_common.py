"""Shared, target-blind I/O helpers for PaperQA2 baseline runners."""

from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

import fcntl

from protocol import format_agent_prompt, parse_final_choice


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _append_recovery_record(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(
    path: Path,
    *,
    repair_torn_tail: bool = False,
    recovery_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Read JSONL, optionally repairing only a torn final non-empty record.

    Every byte of the original is archived before repair. Corruption anywhere
    except the final non-empty line is never modified automatically.
    """
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    nonempty = [index for index, line in enumerate(lines) if line.strip()]
    last_nonempty = nonempty[-1] if nonempty else None
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if not repair_torn_tail or index != last_nonempty:
                raise
            recovery_root = recovery_dir or path.parent / ".jsonl_recovery"
            recovery_root.mkdir(parents=True, exist_ok=True)
            stamp = _utc_stamp()
            archive = recovery_root / f"{path.name}.{stamp}.torn-original"
            shutil.copy2(path, archive)
            temporary = path.with_name(f".{path.name}.{stamp}.repair-part")
            with temporary.open("w") as handle:
                handle.write("".join(lines[:index]))
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
            _append_recovery_record(
                recovery_root / "repairs.jsonl",
                {
                    "schema_version": 1,
                    "repaired_at": datetime.now(timezone.utc).isoformat(),
                    "path": str(path),
                    "archive": str(archive),
                    "repair": "removed_torn_final_nonempty_jsonl_record",
                    "valid_rows_preserved": len(rows),
                },
            )
            break
    return rows


def completed_question_ids(path: Path, *, repair_torn_tail: bool = False) -> set[str]:
    """Return completed IDs so interrupted runs can safely resume."""
    if not path.exists():
        return set()
    return {
        str(row["question_id"])
        for row in read_jsonl(path, repair_torn_tail=repair_torn_tail)
        if row.get("run_status") in {"ok", "error"}
    }


@contextmanager
def exclusive_runner_lock(path: Path, metadata: dict[str, Any]) -> Iterator[TextIO]:
    """Hold a non-blocking process lock for the complete rollout lifetime."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or "unknown owner"
            raise RuntimeError(f"Another rollout runner holds {path}: {owner}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "schema_version": 1,
                    "acquired_at": datetime.now(timezone.utc).isoformat(),
                    "pid": os.getpid(),
                    **metadata,
                },
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
        yield handle
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append and fsync one result; raw runs live outside the Git checkout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def make_result_row(
    *,
    task: dict[str, Any],
    harness: str,
    answer: str | None,
    wall_seconds: float,
    telemetry: dict[str, Any],
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "question_id": str(task["question_id"]),
        "harness": harness,
        "run_status": "error" if error else "ok",
        "answer": answer,
        "parsed_choice": parse_final_choice(answer),
        "wall_seconds": wall_seconds,
        "telemetry": telemetry,
        "error": error,
    }


def pending_tasks(tasks: Iterable[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    done = completed_question_ids(output)
    return [task for task in tasks if str(task["question_id"]) not in done]


__all__ = [
    "append_jsonl",
    "completed_question_ids",
    "exclusive_runner_lock",
    "format_agent_prompt",
    "make_result_row",
    "pending_tasks",
    "read_jsonl",
]
