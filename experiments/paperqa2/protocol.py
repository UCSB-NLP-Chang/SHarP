"""Shared deterministic task and scoring semantics for the PaperQA2 evaluation."""

from __future__ import annotations

import hashlib
import random
import re
import string
from dataclasses import dataclass
from typing import Any, Iterable

CHOICE_PROTOCOL_ID = "paperqa2-audit-v1-choice-order"
REFUSE_CHOICE = "Insufficient information to answer the question"
ALPHABET = string.ascii_uppercase


@dataclass(frozen=True)
class MaterializedQuestion:
    question_id: str
    question: str
    choices: tuple[str, ...]
    target_choice: str
    unsure_choice: str

    def agent_input(self) -> dict[str, Any]:
        """Return the only benchmark fields an agent is allowed to receive."""
        return {
            "question_id": self.question_id,
            "question": self.question,
            "choices": list(self.choices),
        }

    def protected_target(self) -> dict[str, str]:
        return {
            "question_id": self.question_id,
            "target_choice": self.target_choice,
            "unsure_choice": self.unsure_choice,
        }


def _choice_seed(question_id: str) -> int:
    digest = hashlib.sha256(
        f"{CHOICE_PROTOCOL_ID}:{question_id}".encode()
    ).digest()
    return int.from_bytes(digest[:16], byteorder="big", signed=False)


def materialize_question(record: dict[str, Any]) -> MaterializedQuestion:
    """Reproduce LAB-Bench MCQ semantics with per-question deterministic order."""
    question_id = str(record["id"])
    raw_choices = [record["ideal"], REFUSE_CHOICE, *record["distractors"]]
    if len(raw_choices) > len(ALPHABET):
        raise ValueError(f"Too many choices for {question_id}: {len(raw_choices)}")

    permutation = list(range(len(raw_choices)))
    random.Random(_choice_seed(question_id)).shuffle(permutation)
    choices = tuple(
        f"({letter}) {raw_choices[source_index]}"
        for letter, source_index in zip(ALPHABET, permutation)
    )
    return MaterializedQuestion(
        question_id=question_id,
        question=record["question"],
        choices=choices,
        target_choice=ALPHABET[permutation.index(0)],
        unsure_choice=ALPHABET[permutation.index(1)],
    )


def format_agent_prompt(task: dict[str, Any]) -> str:
    choices = "\n".join(task["choices"])
    return (
        f"Question: {task['question']}\n\nChoices:\n{choices}\n\n"
        "Return the final choice as `FINAL_ANSWER: X`, where X is one letter."
    )


def parse_final_choice(answer: str | None) -> str | None:
    """Parse only an explicit final-answer contract, never guess from prose."""
    if answer is None:
        return None
    stripped = answer.strip()
    if re.fullmatch(r"[A-Za-z]", stripped):
        return stripped.upper()
    matches = re.findall(r"(?im)^\s*FINAL_ANSWER\s*:\s*([A-Za-z])\s*$", answer)
    return matches[-1].upper() if matches else None


def score_answer(answer: str | None, target: dict[str, str]) -> dict[str, Any]:
    choice = parse_final_choice(answer)
    return {
        "question_id": target["question_id"],
        "parsed_choice": choice,
        "correct": choice == target["target_choice"],
        "sure": choice is not None and choice != target["unsure_choice"],
        "parse_failure": choice is None,
    }


def aggregate_scores(rows: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    materialized = list(rows)
    n_total = len(materialized)
    n_correct = sum(bool(row["correct"]) for row in materialized)
    n_sure = sum(bool(row["sure"]) for row in materialized)
    n_parse_failure = sum(bool(row["parse_failure"]) for row in materialized)
    return {
        "accuracy": n_correct / n_total if n_total else 0.0,
        "precision": n_correct / n_sure if n_sure else 0.0,
        "coverage": n_sure / n_total if n_total else 0.0,
        "n_total": n_total,
        "n_correct": n_correct,
        "n_sure": n_sure,
        "n_parse_failure": n_parse_failure,
    }
