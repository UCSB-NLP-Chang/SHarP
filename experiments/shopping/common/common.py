"""Portable paths, frozen execution settings, and JSON helpers for Shopping."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

REAL_ROOT = Path(__file__).resolve().parents[3] / "outputs/shopping"
ROOT = Path(os.environ.get("JIT_ROOT", str(REAL_ROOT)))
UPSTREAM = Path(os.environ.get("JIT_UPSTREAM", str(Path(__file__).resolve().parents[3] / "external/shopping")))
DEPS = Path(os.environ.get("JIT_DEPS", str(Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages")))
CODE = Path(__file__).resolve().parent
CHECKPOINT = os.environ.get("JIT_CHECKPOINT", "Qwen/Qwen3.5-122B-A10B-FP8")
DATASET = Path(os.environ.get("JIT_DATASET", str(UPSTREAM / "dataset" / "deepplanning_shopping")))
PYTHON = os.environ.get("JIT_PYTHON", sys.executable)
FAKE_MODEL = os.environ.get("JIT_FAKE_MODEL") == "1"   # scripted CPU integration checks only

MODEL_NAME = os.environ.get("OPENAI_MODEL", "Qwen3.5-122B-A10B")
UPSTREAM_COMMIT = "ababa06c2f54d799fd9fbc356e5368f61a452260"

SEED_BASE = 20260908
MAX_MODEL_LEN = 262144
MAX_STEPS = 40                # benchmark/config/deepplanning_shopping.yaml execution.max_steps
EXEC_MAX_TOKENS = 32000       # benchmark/config/deepplanning_shopping.yaml model.max_tokens (upstream sets its own)
EXEC_TEMPERATURE = 0.0
CODE_TIMEOUT_S = 120
THINKING_OFF = {"chat_template_kwargs": {"enable_thinking": False}}





# Cases whose published targets cannot be reached with the official tools.
# Keep them in every configuration so comparisons use the same task set.
GOLD_UNATTAINABLE = {
    "level3_case12": {
        "reason": "ground_truth_coupons requires 1x 'Cross-store: \u00a560 off every \u00a5500' "
                  "but user_info.json grants the user 0 of it; add_coupon_to_cart refuses",
        "match_rate_ceiling": 0.75,
        "case_score_ceiling": 0.0,
    },
}

# Substrings marking an *infrastructure* failure worth a missing-only re-run.
# Deliberately narrower than scripts/eval/metrics.INFRA_MARKERS: context
# exhaustion and the runtime's model-call budget are behavioural outcomes here.
INFRA_MARKERS = (
    "no space left on device", "errno 28", "connection error", "network error",
    "apiconnection", "apitimeout", "service unavailable", "bad gateway", "502", "503", "504",
    "engine is dead", "engine loop has died", "enginedeaderror", "enginecore",
    "remoteprotocol", "econnrefused", "timed out", "read timeout", "max retries",
    "code_sandbox_failure", "missing_token_telemetry", "internal server error",
    "systemexit", "endpoint_", "harnesses_from has no cases",
)
BEHAVIOURAL_MARKERS = ("maximum context length", "context length", "budget exceeded", "run timed out",
                       "generation_incomplete", "harnesses_from is missing")


class InfraInvalid(BaseException):
    """An endpoint/sandbox/filesystem failure; the cell is re-run, never scored."""


class ContractViolation(BaseException):
    """The frozen protocol was not honoured (e.g. thinking leaked); never scored."""


def is_context_error(text: str) -> bool:
    lowered = str(text or "").lower()
    return "maximum context length" in lowered or ("context length" in lowered and "token" in lowered)


def digest(path: os.PathLike) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save(path: os.PathLike, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def load_json(path: os.PathLike):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: os.PathLike, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: os.PathLike) -> list:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def ensure_upstream_on_path() -> None:
    for p in (str(DEPS), str(UPSTREAM), str(CODE)):
        if p not in sys.path:
            sys.path.insert(0, p)


def load_items(dataset: os.PathLike | None = None):
    """All shopping items via the upstream adapter (no workspace side effects)."""
    ensure_upstream_on_path()
    from benchmark.adapter.deepplanning import DeepPlanningShoppingAdapter
    return DeepPlanningShoppingAdapter().load_dataset(str(dataset or DATASET))








def load_cohort() -> dict:
    return load_json(ROOT / "cohort.json")


def cohort_sha256() -> str:
    return digest(ROOT / "cohort.json")






def endpoint_healthy(base_url: str, timeout: float = 5.0) -> bool:
    if FAKE_MODEL:
        return True
    import urllib.request
    try:
        health = base_url.rsplit("/v1", 1)[0] + "/health"
        with urllib.request.urlopen(health, timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False




def classify_error(text: str) -> str:
    """'ok' | 'behavioural' | 'infra' for an error string from a run."""
    lowered = str(text or "").lower().strip()
    if not lowered:
        return "ok"
    if any(m in lowered for m in BEHAVIOURAL_MARKERS):
        return "behavioural"
    if any(m in lowered for m in INFRA_MARKERS):
        return "infra"
    return "behavioural"


def child_env(attempt: Path, seed: int) -> dict:
    """Environment for a cell subprocess (kernel arms) or the in-process reference."""
    env = dict(os.environ)
    env.update({
        "JIT_ROOT": str(ROOT), "JIT_UPSTREAM": str(UPSTREAM), "JIT_DEPS": str(DEPS),
        "JIT_DATASET": str(DATASET), "JIT_CHECKPOINT": CHECKPOINT,
        "JIT_ATTEMPT_DIR": str(attempt),
        "JIT_TOOL_STATE_ROOT": str(attempt / "tool_state"),
        "JIT_TELEMETRY_FILE": str(attempt / "telemetry.jsonl"),
        "JIT_TOOL_LOG": str(attempt / "tools.jsonl"),
        "JIT_TASK_SEED": str(seed),
        "PYTHONPATH": os.pathsep.join([str(DEPS), str(UPSTREAM), str(CODE)]),
        "PYTHONPYCACHEPREFIX": str(ROOT / "cache" / "pycache"),
        "TIKTOKEN_CACHE_DIR": os.environ.get("TIKTOKEN_CACHE_DIR", str(ROOT / "cache" / "tiktoken")),
        "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
    })
    if FAKE_MODEL:
        env["JIT_FAKE_MODEL"] = "1"
    return env


# --------------------------------------------------------------------------- #
# CLI: freeze / verify the cohort without touching an endpoint
# --------------------------------------------------------------------------- #
