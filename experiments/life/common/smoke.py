"""Stubbed-LLM transport for CPU smoke tests (no model endpoint).

Scripted behaviour per role:
  user  : call 1 -> a generic opening request; after the agent's final summary -> "###STOP###"; else "please go ahead".
  agent : emits the task's gold assistant actions one per call as NATIVE tool calls (proves loop + tool
          dispatch through the real orchestrator/toolkit), then a final text summary containing communicate_info.
  judge : JSON verdict marking every NL assertion met (proves retail evaluator/judge wiring and JSON parsing).
Expectation: every arm scores 1.0 on every cohort task with zero endpoint calls (gold-action replay through
the live loop, in both harness modes).  A task where the full arm scores < 1.0 while native scores 1.0 flags
a Life H2 rule that blocks a gold action (harness/evaluator incompatibility) and is reported, not hidden.
"""
from __future__ import annotations
import json, time, uuid
from pathlib import Path
from litellm import ModelResponse
from runtime import role_of, save, MAX_TOKENS


class StubCompletion:
    def __init__(self, directory, seed, task):
        self.directory = Path(directory); self.directory.mkdir(parents=True, exist_ok=True)
        self.seed = seed; self.calls = []; self.last_role = None
        ec = task.evaluation_criteria
        self.actions = [a for a in (ec.actions or []) if a.requestor == "assistant"]
        self.user_actions = [a for a in (ec.actions or []) if a.requestor == "user"]
        self.info = list(ec.communicate_info or [])
        self.assertions = list(ec.nl_assertions or [])
        self.agent_idx = 0; self.user_calls = 0; self.agent_done = False; self.tools_seen = set()

    def _response(self, role, content=None, tool_call=None, prompt_chars=0):
        message = {"role": "assistant", "content": content, "tool_calls": None}
        finish = "stop"
        if tool_call is not None:
            name, args = tool_call
            message["tool_calls"] = [{"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                                      "function": {"name": name, "arguments": json.dumps(args)}}]
            finish = "tool_calls"
        pt = max(1, prompt_chars // 4); ct = max(1, len(content or "") // 4 + (32 if tool_call else 0))
        raw = {"id": f"stub-{len(self.calls)}", "object": "chat.completion", "created": int(time.time()), "model": "stub",
               "choices": [{"index": 0, "finish_reason": finish, "message": message}],
               "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}}
        c = {"index": len(self.calls), "role": role, "input_tokens": pt, "output_tokens": ct, "tokens": pt + ct,
             "estimated_input_tokens": None, "input_count_match": None, "finish_reason": finish, "reasoning_chars": 0,
             "native_tool_calls": 1 if tool_call else 0, "content_chars": len(content or ""), "seconds": 0.0}
        self.calls.append(c); save(self.directory / "usage.json", self.calls)
        return ModelResponse(**raw)

    def __call__(self, model, messages, tools=None, tool_choice=None, **kwargs):
        role = role_of(model); self.last_role = role
        assert kwargs.get("temperature", 0) == 0 and kwargs.get("seed", self.seed) == self.seed
        assert kwargs.get("max_tokens", MAX_TOKENS) == MAX_TOKENS
        prompt_chars = len(json.dumps(messages, default=str))
        if role == "user":
            self.user_calls += 1
            if self.user_calls == 1:
                content = "Hello, I need some help with my account today."
            elif self.agent_done:
                content = "Thank you, that is everything. ###STOP###"
            else:
                content = "Okay, please go ahead."
            return self._response(role, content=content, prompt_chars=prompt_chars)
        if role == "agent":
            assert tools, "agent call without tool schemas"
            self.tools_seen |= {t["function"]["name"] for t in tools}
            if self.agent_idx < len(self.actions):
                a = self.actions[self.agent_idx]; self.agent_idx += 1
                assert a.name in self.tools_seen, f"gold action {a.name} not in agent tool schema"
                return self._response(role, tool_call=(a.name, a.arguments), prompt_chars=prompt_chars)
            self.agent_done = True
            parts = [x for x in self.info] + [x.replace(",", "") for x in self.info]
            content = "Everything is completed. " + (" ".join(parts) if parts else "Is there anything else I can help with?")
            return self._response(role, content=content, prompt_chars=prompt_chars)
        if role == "judge":
            assert kwargs.get("response_format") == {"type": "json_object"}, kwargs
            content = json.dumps({"results": [{"expectedOutcome": x, "reasoning": "stub", "metExpectation": True}
                                              for x in self.assertions]})
            return self._response(role, content=content, prompt_chars=prompt_chars)
        raise AssertionError(role)

    def summary(self):
        cs = self.calls
        def tok(role, key="tokens"):
            return sum(c[key] for c in cs if c["role"] == role)
        return {"llm_calls": len(cs), "tokens": sum(c["tokens"] for c in cs if c["role"] != "judge"),
                "input_tokens": sum(c["input_tokens"] for c in cs if c["role"] != "judge"),
                "output_tokens": sum(c["output_tokens"] for c in cs if c["role"] != "judge"),
                "agent_tokens": tok("agent"), "user_tokens": tok("user"), "judge_tokens": tok("judge"),
                "agent_input_tokens": tok("agent", "input_tokens"), "agent_output_tokens": tok("agent", "output_tokens"),
                "user_input_tokens": tok("user", "input_tokens"), "user_output_tokens": tok("user", "output_tokens"),
                "agent_llm_calls": sum(c["role"] == "agent" for c in cs), "user_llm_calls": sum(c["role"] == "user" for c in cs),
                "judge_llm_calls": sum(c["role"] == "judge" for c in cs),
                "output_truncated_calls": 0, "reasoning_calls": 0, "token_count_mismatches": 0,
                "max_input_tokens": max((c["input_tokens"] for c in cs), default=0), "stub": True}
