# Model serving

Run the model server in a separate environment from the harness environments.
The server must expose an OpenAI-compatible chat API with native tool calls,
accept `chat_template_kwargs.enable_thinking=false`, and return token usage.
PaperQA's full configuration additionally needs image input support.

## Small-model pipeline checks

Example for Qwen3-8B with vLLM 0.30.0 (text-only checks):

```bash
vllm serve Qwen/Qwen3-8B --served-model-name Qwen3-8B \
  --host 127.0.0.1 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768},"max_position_embeddings":131072}' \
  --max-model-len 131072 --max-num-seqs 1
```

In the harness shell:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_MODEL=Qwen3-8B
export OPENAI_API_KEY=EMPTY
```

For small-model checks across LIFE and Shopping, use at least 131072 context
capacity as the practical starting point, with sufficient GPU memory for its KV
cache. This is not a guarantee that every agent trajectory fits: tool responses
and accumulated history vary. Shopping reserves 32000 output tokens per request;
a 40960-token server leaves only 8960 tokens for input and can exhaust context
after a few tool calls. The request must fit both input and output into the
server's context window. Keep the paper's output budgets unchanged.

The paper setup uses a 262144-token context window; other settings are recorded
in the experiment protocols. A small-model check validates the pipeline, not
paper results. Qwen3-8B does not support PaperQA image enrichment: use a
vision-capable endpoint for the full configuration. A text-only PDF fixture
can check a limited text path but does not validate image handling.

Choose the tool-call parser for the model being served. The Qwen3-8B example uses
[Qwen's tool-calling setup](https://github.com/QwenLM/Qwen3/blob/main/docs/source/deployment/vllm.md)
and [documented YaRN scaling](https://huggingface.co/Qwen/Qwen3-8B#processing-long-texts).
Other models may require different parsers and context settings; see
[vLLM tool calling](https://docs.vllm.ai/en/stable/features/tool_calling/).
Use the matching tokenizer for LIFE token accounting.
