"""Process-wide patches installed before any upstream JIT entry point runs.

Importing this module (see run_upstream.py, preflight.py, cpu_checks.py) applies:

1. OpenAIServerModel: every chat request carries
   extra_body.chat_template_kwargs.enable_thinking=false (thinking OFF for meta,
   exec and judge alike); exec/judge requests carry the task seed (meta requests
   do not: best-of-N at T=1 needs distinct samples); the SDK timeout is raised for
   64k-token harness generations; every call is appended to $JIT_TELEMETRY_FILE
   with role/usage/finish_reason/reasoning presence. A context-length 400 raises
   ContextExhausted (a behavioural stop, see 6) instead of feeding the upstream
   "5 consecutive failures -> SystemExit" counter.
2. DeepPlanningShoppingAdapter: get_task_tools returns the official Qwen-Agent
   shopping tools bound to a fresh isolated state dir under
   $JIT_TOOL_STATE_ROOT; evaluate scores that state's cart.json.
3. ExecuteCodeTool.forward: bwrap-isolated execution (sandbox.py).
4. Tool.__call__: every tool execution is appended to $JIT_TOOL_LOG.
5. load_harness: planning/memory/action hooks are wrapped to emit activation
   events (initial_plan, replan, fold, memory_summary_attempt, forced_finalization).
6. MetaReActAgent._validate_harness: context exhaustion, the 3*max_steps
   model-call budget and the 1h run timeout are behavioural terminations: the
   cart is scored as-is and no repair/regeneration is triggered.
7. scripts.eval.metrics.is_infra_failure: uses common.classify_error, so the
   upstream resume/retry logic re-runs only infrastructure failures.

Upstream files are never edited; only the running process is patched.
With JIT_FAKE_MODEL=1 (CPU dry run only) the model call is replaced by
fake_model.fake_call; everything else stays identical.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import common

common.ensure_upstream_on_path()

_LOCK = threading.Lock()
_THINKING_OFF = {"chat_template_kwargs": {"enable_thinking": False}}


class ContextExhausted(BaseException):
    """The endpoint rejected the prompt as longer than max_model_len."""


def _append(env_name: str, record: dict) -> None:
    path = os.environ.get(env_name)
    if not path:
        return
    record = dict(record)
    record.setdefault("ts", time.time())
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    with _LOCK:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)


def telemetry(record: dict) -> None:
    _append("JIT_TELEMETRY_FILE", record)


def _role_from_stack() -> str:
    """Which model role is being constructed: meta (jit/meta_agent.py __init__),
    judge (selector.select_by_judge) or exec (kernel/runtime.py __init__)."""
    explicit = os.environ.get("JIT_ROLE_OVERRIDE")
    if explicit:
        return explicit
    frame = sys._getframe(1)
    while frame is not None:
        name, filename = frame.f_code.co_name, frame.f_code.co_filename
        if name == "__init__" and filename.endswith(os.path.join("jit", "meta_agent.py")):
            return "meta"
        if name == "select_by_judge":
            return "judge"
        if name == "__init__" and filename.endswith(os.path.join("kernel", "runtime.py")):
            return "exec"
        frame = frame.f_back
    return "unknown"


# `scripts.models` and `scripts.kernel` import each other; the package that must
# be entered FIRST is scripts.kernel (that is the order every upstream entry point
# happens to produce). Importing scripts.models first raises
# "cannot import name OpenAIServerModel from partially initialized module".
from scripts.kernel import runtime as _runtime  # noqa: E402

# --------------------------------------------------------------------------- #
# 1. OpenAIServerModel: thinking OFF, seed, timeout, telemetry, context stop
# --------------------------------------------------------------------------- #
from scripts.models import openai_server as _openai_server  # noqa: E402

_ORIG_INIT = _openai_server.OpenAIServerModel.__init__
if common.FAKE_MODEL:
    import fake_model as _fake_model  # noqa: E402
    _ORIG_CALL = _fake_model.fake_call
else:
    _ORIG_CALL = _openai_server.OpenAIServerModel.__call__
_ORIG_FAILURE = _openai_server.OpenAIServerModel._record_api_failure_and_maybe_exit


def _patched_init(self, model_id, api_base=None, api_key=None, organization=None, project=None,
                  temperature=None, custom_role_conversions=None, **kwargs):
    role = _role_from_stack()
    extra = dict(kwargs.get("extra_body") or {})
    ctk = dict(extra.get("chat_template_kwargs") or {})
    ctk["enable_thinking"] = False
    extra["chat_template_kwargs"] = ctk
    kwargs["extra_body"] = extra
    if role != "meta":
        kwargs.setdefault("seed", int(os.environ.get("JIT_TASK_SEED", common.SEED_BASE)))
    _ORIG_INIT(self, model_id, api_base=api_base, api_key=api_key, organization=organization,
               project=project, temperature=temperature, custom_role_conversions=custom_role_conversions, **kwargs)
    self._jit_role = role
    timeout = 3600.0 if role == "meta" else 1800.0
    try:
        self.client = self.client.with_options(timeout=timeout)
    except Exception:  # noqa: BLE001 - older SDKs
        pass
    if api_base:
        telemetry({"event": "model_init", "role": role, "model_id": model_id, "api_base": api_base,
                   "kwargs": {k: v for k, v in kwargs.items() if k != "extra_body"},
                   "extra_body": kwargs["extra_body"], "timeout": timeout})


def _patched_call(self, messages, stop_sequences=None, grammar=None, tools_to_call_from=None, **kwargs):
    started = time.time()
    role = getattr(self, "_jit_role", "unknown")
    extra = (self.kwargs or {}).get("extra_body") or {}
    thinking_off = bool((extra.get("chat_template_kwargs") or {}).get("enable_thinking") is False)
    try:
        message = _ORIG_CALL(self, messages, stop_sequences=stop_sequences, grammar=grammar,
                             tools_to_call_from=tools_to_call_from, **kwargs)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {str(exc)[:2000]}"
        telemetry({"event": "model_error", "role": role, "seconds": time.time() - started,
                   "error_type": type(exc).__name__, "error": error, "thinking_off": thinking_off})
        if type(exc).__name__ == "BadRequestError" and common.is_context_error(str(exc)):
            attempt_dir = os.environ.get("JIT_ATTEMPT_DIR")
            if attempt_dir:
                try:
                    common.save(Path(attempt_dir) / f"context_exhausted_{time.time_ns()}.json",
                                {"role": role, "error": error, "n_messages": len(messages or []),
                                 "messages": messages})
                except Exception:  # noqa: BLE001
                    pass
            telemetry({"event": "context_exhausted", "role": role, "error": error})
            raise ContextExhausted(error) from exc
        raise
    raw = getattr(message, "raw", None)
    finish = None
    usage = None
    try:
        finish = raw.choices[0].finish_reason
        usage = raw.usage.model_dump() if raw.usage is not None else None
    except Exception:  # noqa: BLE001
        pass
    reasoning = getattr(message, "reasoning_content", None) or ""
    telemetry({
        "event": "model_call", "role": role, "seconds": time.time() - started,
        "input_tokens": self.last_input_token_count, "output_tokens": self.last_output_token_count,
        "usage": usage, "finish_reason": finish,
        "reasoning_chars": len(reasoning), "content_chars": len(message.content or ""),
        "n_tool_calls": len(message.tool_calls or []) if getattr(message, "tool_calls", None) else 0,
        "thinking_off": thinking_off, "n_messages": len(messages or []),
    })
    if reasoning:
        # Thinking is off; a non-empty reasoning field is either a leak or the
        # server's --reasoning-parser qwen3 splitting the answer at a literal
        # </think> inside generated code. Recorded for EVERY role: on a meta call
        # such a split silently drops the leading harness files, which shows up
        # downstream as generation_incomplete. Counted by run_cell
        # (reasoning_present_calls); the arithmetic canary is the primary
        # thinking-off proof.
        telemetry({"event": "reasoning_present", "role": role, "reasoning_chars": len(reasoning),
                   "reasoning_head": reasoning[:200]})
    return message


@classmethod
def _patched_failure(cls, error):
    if type(error).__name__ == "BadRequestError" and common.is_context_error(str(error)):
        # A too-long prompt is a property of this run, not of the endpoint: never
        # let it accumulate towards the process-killing consecutive-failure limit.
        with cls._failure_lock:
            cls._consecutive_api_failures = 0
        return None
    return _ORIG_FAILURE.__func__(cls, error)


_openai_server.OpenAIServerModel.__init__ = _patched_init
_openai_server.OpenAIServerModel.__call__ = _patched_call
_openai_server.OpenAIServerModel._record_api_failure_and_maybe_exit = _patched_failure


# --------------------------------------------------------------------------- #
# 2. Shopping adapter: official tools + isolated state + scorer wiring
# --------------------------------------------------------------------------- #
from benchmark.adapter import deepplanning as _deepplanning  # noqa: E402
from benchmark.adapter.deepplanning_shopping_eval import evaluate_shopping_prediction  # noqa: E402
from official_tools_adapter import create_official_tools  # noqa: E402

_STATE_COUNTER = [0]


def _state_root() -> Path:
    root = os.environ.get("JIT_TOOL_STATE_ROOT")
    if not root:
        raise RuntimeError("JIT_TOOL_STATE_ROOT must point at the attempt's tool-state directory")
    return Path(root)


def _qid(item) -> str:
    return str(item.get("question_id") or f"case_{item.get('_case_id', 'unknown')}")


def _patched_get_task_tools(self, item):
    qid = _qid(item)
    with _LOCK:
        _STATE_COUNTER[0] += 1
        n = _STATE_COUNTER[0]
    state = _state_root() / f"{qid}_{n:03d}_{time.time_ns()}"
    tools = create_official_tools(item["_db_dir"], state)
    states = getattr(self, "_official_state", None)
    if states is None:
        states = self._official_state = {}
    states[qid] = str(state)
    common.save(_state_root() / f"latest_{qid}.json", {"state": str(state), "created": time.time()})
    telemetry({"event": "tool_state", "question_id": qid, "state": str(state)})
    return tools


def _patched_evaluate(self, prediction, ground_truth, **kwargs):
    item = kwargs.get("item", {}) or {}
    qid = _qid(item)
    state = (getattr(self, "_official_state", {}) or {}).get(qid)
    if not state:
        pointer = _state_root() / f"latest_{qid}.json"
        state = common.load_json(pointer)["state"] if pointer.exists() else None
    if not state:
        return {"score": 0.0, "case_score": 0.0, "match_rate": 0.0, "error": "no official tool state for " + qid}
    result = evaluate_shopping_prediction(db_dir=item.get("_db_dir", ""), cart_path=str(Path(state) / "cart.json"))
    result["cart_path"] = str(Path(state) / "cart.json")
    telemetry({"event": "evaluate", "question_id": qid, "result": result})
    return result


_deepplanning.DeepPlanningShoppingAdapter.get_task_tools = _patched_get_task_tools
_deepplanning.DeepPlanningShoppingAdapter.evaluate = _patched_evaluate


# --------------------------------------------------------------------------- #
# 3. execute_code inside bwrap
# --------------------------------------------------------------------------- #
from scripts.tools import code_tools as _code_tools  # noqa: E402
import sandbox as _sandbox  # noqa: E402


def _patched_forward(self, code, code_type=None):
    mode = (code_type or "python").strip().lower()
    if mode not in ("python", "bash"):
        return f"Error: unsupported code_type '{code_type}'. Use 'python' or 'bash'."
    safety_error = _code_tools._check_safety(code)
    if safety_error:
        return f"Error: {safety_error}"
    workspace = Path(self._workspace)
    result = _sandbox.run_in_sandbox(code, workspace, mode, self._timeout)
    if result["sandbox_error"]:
        _sandbox.record_sandbox_error(workspace, result)
        telemetry({"event": "sandbox_error", "stderr": result["sandbox_error"][:1000]})
        return "Error executing code: code_sandbox_failure"
    return _sandbox.format_like_upstream(result, workspace, self._timeout)


_code_tools.ExecuteCodeTool.forward = _patched_forward


# --------------------------------------------------------------------------- #
# 4. Tool execution log
# --------------------------------------------------------------------------- #
from scripts.tools import base as _tool_base  # noqa: E402

_ORIG_TOOL_CALL = _tool_base.Tool.__call__


def _patched_tool_call(self, *args, sanitize_inputs_outputs: bool = False, **kwargs):
    started = time.time()
    error = ""
    try:
        return _ORIG_TOOL_CALL(self, *args, sanitize_inputs_outputs=sanitize_inputs_outputs, **kwargs)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            preview = json.dumps(kwargs, ensure_ascii=False, default=str)[:600] if kwargs else json.dumps(args, default=str)[:600]
        except Exception:  # noqa: BLE001
            preview = ""
        _append("JIT_TOOL_LOG", {"name": getattr(self, "name", type(self).__name__), "seconds": time.time() - started,
                                 "arguments_preview": preview, "error": error})


_tool_base.Tool.__call__ = _patched_tool_call


# --------------------------------------------------------------------------- #
# 5. Harness activation counters
# --------------------------------------------------------------------------- #
from scripts.kernel import loader as _loader  # noqa: E402

_ORIG_LOAD_HARNESS = _loader.load_harness
_HOOKS = [
    ("planning", "init_plan", "initial_plan"),
    ("planning", "update_plan", "replan"),
    ("memory", "_do_summarize", "memory_summary_attempt"),
    ("memory", "apply_compression", "fold"),
    ("action", "_force_final_answer", "forced_finalization"),
]


def _patched_load_harness(harness_name):
    modules = _ORIG_LOAD_HARNESS(harness_name)
    for component, method, key in _HOOKS:
        obj = modules.get(component)
        original = getattr(obj, method, None)
        if obj is None or not callable(original):
            continue

        def counted(*args, _original=original, _key=key, _harness=harness_name, **kwargs):
            telemetry({"event": "activation", "key": _key, "harness": _harness})
            return _original(*args, **kwargs)

        setattr(obj, method, counted)
    telemetry({"event": "harness_loaded", "harness": harness_name,
               "classes": {k: type(v).__name__ for k, v in modules.items() if k != "prompts"}})
    return modules


_loader.load_harness = _patched_load_harness
_runtime.load_harness = _patched_load_harness   # AgentRuntime imported the name directly


# --------------------------------------------------------------------------- #
# 6. Behavioural terminations inside the meta-agent validation
# --------------------------------------------------------------------------- #
from jit import meta_agent as _meta_agent  # noqa: E402

_ORIG_VALIDATE = _meta_agent.MetaReActAgent._validate_harness


def _behavioural_result(adapter, item, reason: str, detail: str) -> dict:
    score = adapter.evaluate("", item.get("answer", ""), item=item)
    telemetry({"event": "termination", "reason": reason, "detail": detail[:500], "evaluation": score})
    return {
        "passed": float(score.get("score", 0.0) or 0.0) > 0.0,
        "steps": 0, "api_latency": 0.0, "tool_latency": _tool_base.Tool.get_total_latency(),
        "input_token_count": 0, "output_token_count": 0, "total_token_count": 0,
        "evaluation": score, "trajectory": [], "error": "",
    }


def _patched_validate(self, benchmark_adapter, item, max_steps):
    telemetry({"event": "validation_start", "question_id": _qid(item), "harness": self.workspace_name})
    try:
        result = _ORIG_VALIDATE(self, benchmark_adapter, item, max_steps)
    except ContextExhausted as exc:
        result = _behavioural_result(benchmark_adapter, item, "context_exhausted", str(exc))
    else:
        error = str(result.get("error") or "")
        if error.strip() == "Budget exceeded":
            result = _behavioural_result(benchmark_adapter, item, "model_call_budget", error)
        elif "Run timed out" in error:
            result = _behavioural_result(benchmark_adapter, item, "run_timeout", error)
    telemetry({"event": "validation_end", "question_id": _qid(item), "passed": result.get("passed"),
               "steps": result.get("steps"), "error": str(result.get("error") or "")[:500],
               "evaluation": result.get("evaluation")})
    return result


_meta_agent.MetaReActAgent._validate_harness = _patched_validate


# --------------------------------------------------------------------------- #
# 7. Infra classification used by upstream resume/retry
# --------------------------------------------------------------------------- #
from scripts.eval import metrics as _metrics  # noqa: E402


def _patched_is_infra_failure(text: str) -> bool:
    return common.classify_error(text) == "infra"


_metrics.is_infra_failure = _patched_is_infra_failure

PATCHES_APPLIED = ("thinking_off_seed_timeout_telemetry_context_stop", "official_tools_adapter", "bwrap_execute_code",
                   "tool_log", "activation_counters", "behavioural_validation_terminations", "infra_classification")
