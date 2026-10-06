"""Constitutive reference agent.

Task-agnostic model/action/observation loop: append-only history, native tool-call
decoding and dispatch, the benchmark's policy text supplied unchanged as the task rules.
No H2-H5, no customer-service prompting, no THINK tools, no inline tool rescue.
"""
import json, time
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall
from tau2.utils import llm_utils


class ReferenceAgent(LLMAgent):
    @property
    def system_prompt(self):
        return ('Interact with the user and environment to complete the task. '
                'Use the provided native tool interface for actions and respond to observations. '
                'The following document defines the task rules.\n<task_rules>\n' + self.domain_policy + '\n</task_rules>')

    def _generate_next_message(self, message, state):
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        messages = state.system_messages + state.messages
        start = time.monotonic()
        r = llm_utils.completion(model=self.llm, messages=llm_utils.to_litellm_messages(messages),
                                 tools=[t.openai_schema for t in self.tools], tool_choice='auto', **self.llm_args)
        msg = r.choices[0].message
        calls = [ToolCall(id=t.id, name=t.function.name, arguments=json.loads(t.function.arguments))
                 for t in (msg.tool_calls or [])]
        return AssistantMessage(role='assistant', content=msg.content, tool_calls=calls or None,
                                cost=0.0, usage=llm_utils.get_response_usage(r), raw_data=r.to_dict(),
                                generation_time_seconds=time.monotonic() - start)


def factory(tools, domain_policy, **kwargs):
    # THINK tools (none exist in airline/retail registries) are filtered in cell.audited_build_agent.
    return ReferenceAgent(tools=tools, domain_policy=domain_policy, llm=kwargs['llm'], llm_args=kwargs['llm_args'])
