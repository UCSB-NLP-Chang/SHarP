"""Shared recorded provider transport.

Every request: served model name from ready.json, temperature 0, seed per task,
max_tokens 16384, extra_body chat_template_kwargs.enable_thinking=false.
Thinking OFF is verified per response: any reasoning_content/reasoning is a
protocol violation and marks the cell infra-invalid (never scored).
Context guard: local Qwen chat-template token count + 16384 must fit 262144.
Upstream tau2/Life source is never modified; we replace tau2.utils.llm_utils.completion
at runtime (both LLMAgent/UserSimulator/NL judge via generate() and the reference agent
call it).
"""
from __future__ import annotations
import copy, hashlib, json, os, time
from pathlib import Path
import openai
from litellm import ModelResponse

ROOT = Path(os.environ.get("LIFE_ROOT", "outputs/life")).resolve()
CODE = Path(__file__).resolve().parent
MODEL_DIR = os.environ.get("LIFE_MODEL_DIR", "Qwen/Qwen3.5-122B-A10B-FP8")
MAX_MODEL_LEN = 262144
MAX_TOKENS = 16384
HTTP_TIMEOUT = 1800
ROLES = {"agent122b": "agent", "user122b": "user", "judge122b": "judge"}
TOKENIZER = None


class InfraInvalid(BaseException):
    """Endpoint/protocol failure: the attempt is never scored; the cell stays missing."""


class BehavioralStop(BaseException):
    """First-attempt behavioral outcome under the frozen contract (scored 0)."""


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n"); tmp.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def digest(x):
    return hashlib.sha256(x if isinstance(x, bytes) else json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()


def filehash(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def role_of(model: str) -> str:
    for k, v in ROLES.items():
        if k in model:
            return v
    raise AssertionError(f"unknown model alias {model!r}")


def get_tokenizer():
    global TOKENIZER
    if TOKENIZER is None:
        from transformers import AutoTokenizer
        TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    return TOKENIZER


def template_token_count(messages, tools):
    """Local prompt-token count using the checkpoint's chat template (thinking off).
    vLLM decodes OpenAI function.arguments JSON strings before templating; mirror that."""
    tm = copy.deepcopy(messages)
    for m in tm:
        for tc in m.get("tool_calls") or []:
            args = tc.get("function", {}).get("arguments")
            if isinstance(args, str):
                try:
                    tc["function"]["arguments"] = json.loads(args)
                except json.JSONDecodeError:
                    pass
    enc = get_tokenizer().apply_chat_template(tm, tools=tools or None, tokenize=True,
                                              add_generation_prompt=True, enable_thinking=False,
                                              return_dict=False)
    if isinstance(enc, dict):
        enc = enc["input_ids"]
    return len(enc)


class RecordedCompletion:
    """Drop-in for litellm.completion as used by tau2.utils.llm_utils.generate and reference.py."""

    def __init__(self, base_url: str, served_model: str, directory, seed: int, count_tokens: bool = True):
        self.base_url = base_url; self.served = served_model
        self.directory = Path(directory); self.directory.mkdir(parents=True, exist_ok=True)
        self.client = openai.OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"), max_retries=0, timeout=HTTP_TIMEOUT)
        self.seed = seed; self.calls = []; self.started = time.monotonic(); self.last_role = None
        self.count_tokens = count_tokens

    def __call__(self, model, messages, tools=None, tool_choice=None, **kwargs):
        role = role_of(model); self.last_role = role
        assert kwargs.get("temperature", 0) == 0, kwargs
        assert kwargs.get("seed", self.seed) == self.seed, (kwargs.get("seed"), self.seed)
        assert kwargs.get("max_tokens", MAX_TOKENS) == MAX_TOKENS, kwargs
        # Provider wire normalization only (drop null fields); no prompt rewriting.
        wire = [{k: v for k, v in m.items() if v is not None} for m in messages]
        req = dict(model=self.served, messages=wire, temperature=0, max_tokens=MAX_TOKENS, seed=self.seed,
                   extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        if tools:
            req.update(tools=tools, tool_choice=tool_choice or "auto")
        if kwargs.get("response_format") is not None:  # upstream NL-judge args
            req["response_format"] = kwargs["response_format"]
        n = len(self.calls); p = self.directory / f"call_{n:03d}"
        save(str(p) + ".request.json", req)
        count = template_token_count(wire, tools) if self.count_tokens else None
        if count is not None and count + MAX_TOKENS > MAX_MODEL_LEN:
            save(str(p) + ".behavior.json", {"status": "context_exhaustion", "input_tokens": count, "role": role})
            raise BehavioralStop("context_exhaustion")
        start = time.monotonic()
        try:
            response = self.client.chat.completions.create(**req)
        except (openai.APIConnectionError, openai.APIStatusError, openai.APITimeoutError) as e:
            save(str(p) + ".error.json", {"type": type(e).__name__, "status_code": getattr(e, "status_code", None),
                                          "message": str(e)[:2000]})
            raise InfraInvalid(type(e).__name__) from None
        raw = response.model_dump(); save(str(p) + ".response.json", raw)
        usage = raw.get("usage") or {}; msg = raw["choices"][0]["message"]
        if not all(isinstance(usage.get(k), int) for k in ["prompt_tokens", "completion_tokens", "total_tokens"]):
            raise InfraInvalid("missing_usage")
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        c = {"index": n, "role": role, "input_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"],
             "tokens": usage["total_tokens"], "estimated_input_tokens": count,
             "input_count_match": (count == usage["prompt_tokens"]) if count is not None else None,
             "finish_reason": raw["choices"][0]["finish_reason"], "reasoning_chars": len(reasoning),
             "native_tool_calls": len(msg.get("tool_calls") or []), "content_chars": len(msg.get("content") or ""),
             "seconds": time.monotonic() - start,
             "request_sha256": filehash(str(p) + ".request.json"), "response_sha256": filehash(str(p) + ".response.json")}
        self.calls.append(c); save(self.directory / "usage.json", self.calls)
        if reasoning:
            raise InfraInvalid("thinking_leak")  # protocol violation: enable_thinking=false not honored
        if not (msg.get("content") or msg.get("tool_calls")):
            raise BehavioralStop("empty_model_output")
        return ModelResponse(**raw)

    def summary(self):
        cs = self.calls
        def tok(role, key="tokens"):
            return sum(c[key] for c in cs if c["role"] == role)
        return {"llm_calls": len(cs),
                "tokens": sum(c["tokens"] for c in cs if c["role"] != "judge"),
                "input_tokens": sum(c["input_tokens"] for c in cs if c["role"] != "judge"),
                "output_tokens": sum(c["output_tokens"] for c in cs if c["role"] != "judge"),
                "agent_tokens": tok("agent"), "user_tokens": tok("user"), "judge_tokens": tok("judge"),
                "agent_input_tokens": tok("agent", "input_tokens"), "agent_output_tokens": tok("agent", "output_tokens"),
                "user_input_tokens": tok("user", "input_tokens"), "user_output_tokens": tok("user", "output_tokens"),
                "agent_llm_calls": sum(c["role"] == "agent" for c in cs), "user_llm_calls": sum(c["role"] == "user" for c in cs),
                "judge_llm_calls": sum(c["role"] == "judge" for c in cs),
                "output_truncated_calls": sum(c["finish_reason"] == "length" for c in cs),
                "reasoning_calls": sum(c["reasoning_chars"] > 0 for c in cs),
                "token_count_mismatches": sum(c["input_count_match"] is False for c in cs),
                "max_input_tokens": max((c["input_tokens"] for c in cs), default=0)}
