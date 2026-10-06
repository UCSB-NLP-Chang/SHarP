#!/usr/bin/env python3
"""Keep endpoint/inference failures out of behavioral PaperQA results.

PaperQA intentionally converts arbitrary rollout exceptions to ``AgentStatus.FAIL``.
That also hides dead serving endpoints from experiment runners. This narrow guard
re-raises only unambiguous inference/infrastructure failures so the outer runner
records a hard error instead of canonicalizing a false behavioral failure.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, Iterator


INFRASTRUCTURE_EXCEPTION_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "ClientConnectorError",
        "ConnectError",
        "ConnectionError",
        "ConnectionRefusedError",
        "ConnectTimeout",
        "EngineDeadError",
        "InternalServerError",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "ServerDisconnectedError",
    }
)


def exception_chain(error: BaseException) -> Iterator[BaseException]:
    """Yield one exception and its causes without looping."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_infrastructure_exception(error: BaseException) -> bool:
    """Return true only for concrete serving/network failure signatures."""
    for candidate in exception_chain(error):
        if type(candidate).__name__ in INFRASTRUCTURE_EXCEPTION_NAMES:
            return True
        status_code = getattr(candidate, "status_code", None)
        if isinstance(status_code, int) and status_code >= 500:
            return True
    return False


def is_context_limit_exception(error: BaseException) -> bool:
    """Return true only for an HTTP 400 maximum-context-length failure.

    Context exhaustion is a behavioral agent outcome, not an endpoint failure.  It
    is terminal for the current trajectory, though: retrying the same oversized
    state through the generate-answer fallback cannot succeed and can wedge the
    serving engine under repeated load.
    """
    for candidate in exception_chain(error):
        if getattr(candidate, "status_code", None) != 400:
            continue
        if "maximum context length" in str(candidate).lower():
            return True
    return False


def load_agent_main_module() -> ModuleType:
    module = importlib.import_module("paperqa.agents.main")
    if not isinstance(module, ModuleType):
        raise TypeError("paperqa.agents.main did not resolve to a module")
    return module


def install_infrastructure_exception_guard() -> None:
    """Patch PaperQA's rollout boundary without changing behavioral semantics."""
    import asyncio
    import logging

    from aviary.core import ToolCall, ToolRequestMessage
    from paperqa.agents.models import AgentStatus
    from paperqa.agents.tools import GenerateAnswer

    agent_main = load_agent_main_module()
    if getattr(agent_main, "_pruning_infrastructure_exception_guard", False):
        return
    logger = logging.getLogger(agent_main.__name__)

    async def run_with_infrastructure_guard(
        rollout: Any,
        settings: Any,
        env: Any,
    ) -> tuple[Any, AgentStatus]:
        terminal_context_failure = False
        try:
            async with asyncio.timeout(settings.agent.timeout):
                status = await rollout()
        except TimeoutError:
            # The configured whole-agent timeout remains a scored outcome.
            logger.warning(
                "Agent timeout after %s-sec, just answering.",
                settings.agent.timeout,
            )
            status = AgentStatus.TRUNCATED
        except Exception as exc:
            if is_infrastructure_exception(exc):
                raise
            terminal_context_failure = is_context_limit_exception(exc)
            if terminal_context_failure:
                logger.warning(
                    "Trajectory exceeded the model maximum context length; "
                    "scoring a terminal behavioral failure without an impossible "
                    "generate-answer fallback."
                )
            else:
                logger.exception("Trajectory failed.")
            status = AgentStatus.FAIL

        if not terminal_context_failure and (
            status == AgentStatus.TRUNCATED
            or not env.state.query_tool_history(GenerateAnswer.TOOL_FN_NAME)
        ):
            generate_answer_tool = next(
                filter(
                    lambda tool: tool.info.name == GenerateAnswer.TOOL_FN_NAME,
                    env.tools,
                )
            )
            action = ToolRequestMessage(
                tool_calls=[ToolCall.from_tool(generate_answer_tool)]
            )
            try:
                await env.exec_tool_calls(
                    message=action,
                    state=env.state,
                    handle_tool_exc=False,
                )
            except Exception as exc:
                if is_infrastructure_exception(exc):
                    raise
                logger.exception("Final answer fallback failed behaviorally.")
                status = AgentStatus.FAIL
            env.state.record_action(action)
        return env.state.session, status

    agent_main._run_with_timeout_failure = run_with_infrastructure_guard
    agent_main._pruning_infrastructure_exception_guard = True
    for runner_name in ("run_aviary_agent", "run_ldp_agent"):
        runner = getattr(agent_main, runner_name)
        if runner.__globals__["_run_with_timeout_failure"] is not run_with_infrastructure_guard:
            raise RuntimeError(f"Infrastructure guard did not reach {runner_name} globals")


__all__ = [
    "INFRASTRUCTURE_EXCEPTION_NAMES",
    "exception_chain",
    "install_infrastructure_exception_guard",
    "is_context_limit_exception",
    "is_infrastructure_exception",
    "load_agent_main_module",
]
