#!/usr/bin/env python3
"""Portable entry point for the final PaperQA2 single-off and median ladders."""
import argparse
import asyncio
import json
import hashlib
import os
from pathlib import Path
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/paperqa2"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["screen", "validation", "all-pruned"], required=True)
    p.add_argument("--objective", choices=["efficiency", "performance"], default="efficiency")
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True, help="Frozen local source-PDF directory")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "Qwen3.5-122B-A10B"))
    p.add_argument("--embedding", default="st-multi-qa-MiniLM-L6-cos-v1")
    p.add_argument("--gpu-embeddings", action="store_true", help="Opt in to GPU embeddings using the inherited CUDA device selection; default: CPU")
    p.add_argument("--allow-custom-corpus", action="store_true", help="Allow a different PDF corpus for pipeline debugging; this is not paper reproduction")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--config", action="append", help="Validation configuration, e.g. A18; repeat for more")
    p.add_argument("--preprocess-only", action="store_true")
    p.add_argument("--max-new-cells", type=int, help="Stop validation after this many new canonical cells")
    p.add_argument("--dry-run", action="store_true", help="Print the resolved run plan without importing PaperQA or making model calls")
    a = p.parse_args()
    os.environ["SHARP_GPU_EMBEDDINGS"] = "1" if a.gpu_embeddings else "0"
    # Set device visibility before importing runners or embedding dependencies.
    if not a.gpu_embeddings:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if not a.tasks.is_file() or not a.corpus.is_dir():
        p.error("--tasks must be a JSONL file and --corpus must be a local directory")
    if not a.allow_custom_corpus:
        split = "train" if a.phase == "screen" else "validation"
        expected = set(json.loads((ROOT / f"configs/splits/litqa2_{split}.json").read_text())["corpus_document_sha256"])
        actual = {hashlib.sha256(f.read_bytes()).hexdigest() for f in a.corpus.rglob("*") if f.is_file()}
        if actual != expected:
            p.error("Corpus differs from the frozen split. Use prepare_corpus.py, or --allow-custom-corpus for a separate debug run.")
    if a.workers < 1:
        p.error("--workers must be positive")
    if a.phase == "screen" and (a.config or a.max_new_cells):
        p.error("--config and --max-new-cells apply to validation")
    if a.max_new_cells is not None and a.max_new_cells < 1:
        p.error("--max-new-cells must be positive")
    os.environ["SHARP_OBJECTIVE"] = a.objective
    if a.phase == "all-pruned":
        a.config = ["A0"]
    out = a.output.resolve()
    parsed = urlparse(a.base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        p.error("--base-url must be an HTTP(S) URL")
    argv = ["paperqa-run", "--tasks", str(a.tasks.resolve()), "--corpus-dir", str(a.corpus.resolve()),
            "--index-dir", str(out / "indexes"), "--output-dir", str(out / "results"),
            "--preprocess-dir", str(out / "preprocess"), "--schedule-output", str(out / "schedule.json"),
            "--run-tag", out.name, "--api-base", a.base_url, "--model", a.model,
            "--embedding", a.embedding, "--workers", str(a.workers)]
    if a.phase != "screen":
        argv += ["--phase", "primary", "--shard-count", "1", "--shard-index", "0",
                 "--api-connect-host", parsed.hostname, "--api-connect-port", str(parsed.port or (443 if parsed.scheme == "https" else 80)),
                 "--endpoint-job-id", "0"]
        for config in a.config or []:
            argv += ["--config", config]
        if a.max_new_cells:
            argv += ["--checkpoint-after-new-canonical-cells", str(a.max_new_cells)]
        import run_validation_ladder as runner
    else:
        import run_single_off as runner
    if a.preprocess_only:
        argv.append("--preprocess-only")
    sys.argv = argv
    args = runner.parse_args()
    args.api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    if a.dry_run:
        print(json.dumps({"phase": a.phase, "tasks": str(a.tasks.resolve()), "corpus": str(a.corpus.resolve()),
                          "output": str(out), "model": a.model, "configs": a.config,
                          "preprocess_only": a.preprocess_only,
                          "embedding_device": "auto (GPU enabled)" if a.gpu_embeddings else "cpu"}, indent=2))
        return
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    from paperqa_tool_ablation_compat import install_no_gen_answer_fallback_compat
    install_no_gen_answer_fallback_compat()
    from runner_common import exclusive_runner_lock
    with exclusive_runner_lock(out / ".runner.lock", {"phase": a.phase, "objective": a.objective}):
        identity = {"phase": a.phase, "objective": a.objective, "model": a.model,
                    "base_url": a.base_url, "embedding": a.embedding,
                    "tasks_sha256": hashlib.sha256(a.tasks.read_bytes()).hexdigest(),
                    "corpus_sha256": sorted(hashlib.sha256(f.read_bytes()).hexdigest() for f in a.corpus.rglob("*") if f.is_file()),
                    "source_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT / "experiments/paperqa2").glob("*.py")}}
        if a.gpu_embeddings:
            identity["embedding_device"] = "cuda"
        marker = out / "run_identity.json"
        if marker.exists() and json.loads(marker.read_text()) != identity:
            p.error("Output belongs to a different source/model/data configuration; use a new directory")
        marker.write_text(json.dumps(identity, indent=2) + "\n")
        async def run_all():
            if a.phase != "screen" and not a.preprocess_only:
                args.preprocess_only = True
                await runner.amain(args)
                args.preprocess_only = False
            await runner.amain(args)
        asyncio.run(run_all())


if __name__ == "__main__":
    main()
