# LIFE-harness / tau2 retail: validation

The frozen cohort records task identifiers, task ordering, seeds, and module configurations. Source and benchmark-data hashes are recorded in each new run contract.

Model: Qwen3.5-122B-A10B in non-thinking mode, temperature 0, maximum output 16384 tokens, context 262144 tokens, 200 steps and 10 errors. All arms retain the domain policy, native task APIs, user simulator, and H5 prefill path.

Performance is task success. Token cost includes agent and user simulator calls; assertion-judge calls are recorded separately. Inference failures leave cells incomplete. Completed cells are preserved and only missing cells run on resume. Stub runs are integration tests and are kept separate from model evaluations.
