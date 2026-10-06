#!/usr/bin/env python3
"""Compatibility for mechanically ablating PaperQA's answer-generation tool.

Upstream PaperQA normally forces one final ``gen_answer`` call after truncation
or whenever a trajectory did not call that tool.  That post-processing assumes
the tool is always present and raises ``StopIteration`` when ``gen_answer`` is
itself the ablated component.  For that one structural case, retain the session
without inventing an answer and let the normal behavioral status/score path
record the failure.  All ordinary trajectories use the upstream behavior.
"""

from __future__ import annotations

from typing import Any


def find_named_tool(tools: list[Any], name: str) -> Any | None:
    """Return a named Aviary tool, or ``None`` when it was deliberately removed."""
    return next((tool for tool in tools if tool.info.name == name), None)


def install_no_gen_answer_fallback_compat() -> None:
    """Patch only PaperQA's unconditional missing-tool fallback."""
    import asyncio
    import logging

    from aviary.core import ToolCall, ToolRequestMessage
    import paperqa.agents.main as agent_main
    from paperqa.agents.models import AgentStatus
    from paperqa.agents.tools import GenerateAnswer

    if getattr(agent_main, "_train18_no_gen_answer_compat", False):
        return

    logger = logging.getLogger(agent_main.__name__)

    async def run_with_optional_answer_fallback(
        rollout: Any,
        settings: Any,
        env: Any,
    ) -> tuple[Any, AgentStatus]:
        try:
            async with asyncio.timeout(settings.agent.timeout):
                status = await rollout()
        except TimeoutError:
            logger.warning(
                "Agent timeout after %s-sec, just answering.",
                settings.agent.timeout,
            )
            status = AgentStatus.TRUNCATED
        except Exception:
            logger.exception("Trajectory failed.")
            status = AgentStatus.FAIL

        if status == AgentStatus.TRUNCATED or not env.state.query_tool_history(
            GenerateAnswer.TOOL_FN_NAME
        ):
            generate_answer_tool = find_named_tool(
                env.tools, GenerateAnswer.TOOL_FN_NAME
            )
            if generate_answer_tool is None:
                # The missing answer is the intended behavioral consequence of
                # this arm.  Do not turn it into a retryable infrastructure error
                # and, crucially, do not reintroduce the ablated capability.
                env.state.session.has_successful_answer = False
                return env.state.session, status
            action = ToolRequestMessage(
                tool_calls=[ToolCall.from_tool(generate_answer_tool)]
            )
            await env.exec_tool_calls(
                message=action,
                state=env.state,
                handle_tool_exc=True,
            )
            env.state.record_action(action)
        return env.state.session, status

    agent_main._run_with_timeout_failure = run_with_optional_answer_fallback
    agent_main._train18_no_gen_answer_compat = True


__all__ = ["find_named_tool", "install_no_gen_answer_fallback_compat"]
