# SHarP: Saliency-based Pruning of Agent Harnesses

This repository contains the official implementation of the paper:

**[SHarP: Saliency-based Pruning of Agent Harnesses](https://arxiv.org/abs/2610.04178)**

SHarP (**S**aliency-based **Har**ness **P**runing) prunes tools, instructions, and
supporting mechanisms based on their contribution to task performance and token
cost. The code covers OpenHands/GAIA, LIFE/tau2 Airline and Retail,
JIT Agentfold/DeepPlanning Shopping, and PaperQA2/LitQA2.

## Code Architecture

```text
src/sharp/              Saliency estimation and pruning ladder construction
experiments/
  life/                 LIFE/tau2 Airline and Retail adapters
  shopping/             Agentfold variants, tool adapter, and code sandbox
  paperqa2/             PaperQA2 component switches, indexing, and scoring
  gaia/                 OpenHands module inventory and validation specs
configs/                Upstream revisions, task splits, and corpus hashes
patches/                GAIA and PaperQA upstream patches
scripts/                Setup, data preparation, and experiment entry points
environment.yml         Python 3.12 Conda environment
tests/                  Regression tests
```

## Setup

Use a separate environment for each harness to avoid dependency conflicts.
Upstream code is installed under `external/`; experiment outputs go to `outputs/`.

```bash
git clone https://github.com/UCSB-NLP-Chang/SHarP.git
cd SHarP
conda env create -f environment.yml
conda activate sharp-core
bash scripts/setup.sh core --conda
PYTHON=python bash scripts/smoke.sh
```

The smoke test runs without a model endpoint. Each experiment below includes its
own environment and data setup; run commands from the repository root.

## Model Endpoint

Start an OpenAI-compatible model server and set its connection details. The paper
uses Qwen3.5-122B-A10B. The server must support native tool calls and the runners'
non-thinking requests; see the [serving example and context requirements](docs/serving.md).

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_MODEL=Qwen3.5-122B-A10B
export OPENAI_API_KEY=EMPTY
```

## Experiments

The commands below run the stored screening and pruning configurations. Dataset
versions and task splits are pinned in `configs/`; additional acquisition details
are in [docs/datasets.md](docs/datasets.md). Use `--help` for runner options.

### LIFE / tau2 Airline and Retail

Task records, databases, and policies are included in the LIFE checkout fetched
by setup. Use the tokenizer matching the served model; replace the Qwen3.5
checkpoint below when testing another model.

```bash
conda env create -f environment.yml -n sharp-life
conda activate sharp-life
bash scripts/setup.sh life --conda
python scripts/prepare_data.py life --check-only

# Download the tokenizer for local token accounting.
python - <<'PYTOKENIZER'
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("Qwen/Qwen3.5-122B-A10B-FP8").save_pretrained("outputs/tokenizer")
PYTOKENIZER
export LIFE_MODEL_DIR="$PWD/outputs/tokenizer"

for DOMAIN in airline retail; do
  python scripts/run_life.py --domain "$DOMAIN" --phase screen
  python scripts/run_life.py --domain "$DOMAIN" --phase validation
  python scripts/run_life.py --domain "$DOMAIN" --phase all-pruned
done
python scripts/run_life.py --domain retail --phase performance
```

For Airline, `validation` runs both ladders. For Retail, `validation` runs the
efficiency ladder and `performance` runs the performance ladder. Results are in
`outputs/life/<domain>/<phase>/model/`. For one-task checks:

```bash
python scripts/run_life.py --domain airline --arm full --limit-tasks 1 --output outputs/life-airline-check
python scripts/run_life.py --domain retail --arm full --limit-tasks 1 --output outputs/life-retail-check
```

### JIT Agentfold / DeepPlanning Shopping

Requires Linux and `bubblewrap` with user namespaces enabled. DeepPlanning task
records and tool data are included in the JIT checkout fetched by setup.
Check sandbox support before setup (Ubuntu/Debian installation, if needed:
`sudo apt-get install bubblewrap`):

```bash
command -v bwrap
bwrap --ro-bind / / --unshare-user --unshare-pid --proc /proc --dev /dev -- /bin/true
```

```bash
conda env create -f environment.yml -n sharp-shopping
conda activate sharp-shopping
bash scripts/setup.sh shopping --conda
python scripts/prepare_data.py shopping --check-only

python scripts/check_shopping.py
python scripts/run_shopping.py --phase screen
python scripts/run_shopping.py --phase validation
python scripts/run_shopping.py --phase performance
```

Results are saved under `outputs/shopping/<phase>/model/`. For a one-task check:

```bash
python scripts/run_shopping.py --arm full --limit-tasks 1 --output outputs/shopping-check
```

### PaperQA2 / LitQA2

Requires a vision-capable model endpoint. Embeddings run on CPU by default;
`--gpu-embeddings` opts into the visible GPU devices.
The commands below download the LAB-Bench records and attempt to acquire 126 of
the 160 corpus PDFs automatically,
including 27 from the [licensed supplement](https://github.com/UCSB-NLP-Chang/SHarP/releases/tag/litqa2-v2-data).

```bash
conda env create -f environment.yml -n sharp-paperqa2
conda activate sharp-paperqa2
bash scripts/setup.sh paperqa2 --conda
python scripts/prepare_data.py litqa2
python scripts/download_litqa_corpus.py
```

Download the remaining 34 PDFs using the links and versions in
[docs/litqa2-pdfs.md](docs/litqa2-pdfs.md#remaining-source-downloads), then import
them from your download directory:

```bash
python scripts/download_litqa_corpus.py \
  --import-dir /path/to/browser-downloads --check-only
python scripts/download_litqa_corpus.py --check-only
```

Once all **160 PDFs** pass verification, prepare the 50/38 task splits and the
127/133-document corpora, then run screening and validation:

```bash
for SPLIT in train validation; do
  python scripts/prepare_litqa.py \
    --records outputs/data/litqa2/litqa-v2-public.jsonl --split "$SPLIT" \
    --output "outputs/litqa/$SPLIT"
  python scripts/prepare_corpus.py \
    --pdf-pool outputs/data/litqa2/pdfs --split "$SPLIT" \
    --output "outputs/litqa/$SPLIT/corpus"
done

python scripts/run_paperqa.py --phase screen \
  --tasks outputs/litqa/train/agent_inputs.jsonl \
  --corpus outputs/litqa/train/corpus --output outputs/paperqa2/screen

python scripts/run_paperqa.py --phase validation \
  --tasks outputs/litqa/validation/agent_inputs.jsonl \
  --corpus outputs/litqa/validation/corpus --output outputs/paperqa2/validation
```

Results are saved in the specified output directories. For a small validation
run, add `--config A18 --max-new-cells 1` and use a separate `--output` directory.
For a pipeline check with a partial PDF directory, also pass `--allow-custom-corpus`
and point `--corpus` to that directory; this does not reproduce the paper corpus.
Score only complete runs.
Use `--objective performance` for the performance ladder or `--phase all-pruned`
for the all-pruned configuration.

Score answers and export the complete screening grid for saliency estimation:

```bash
python experiments/paperqa2/score_runs.py \
  --results outputs/paperqa2/screen/results/gen_answer_tool.jsonl \
  --protected-targets outputs/litqa/train/protected_targets.jsonl \
  --output outputs/paperqa2/screen/gen_answer_score.json

python scripts/collect_paperqa_screen.py \
  --results outputs/paperqa2/screen/results \
  --protected-targets outputs/litqa/train/protected_targets.jsonl \
  --output outputs/paperqa2/screen/export
```

### OpenHands / GAIA

Requires Linux, access to [GAIA on Hugging Face](https://huggingface.co/datasets/gaia-benchmark/GAIA),
and a running [SearXNG service](docs/datasets.md#gaia-web-search-service).
Data preparation downloads the questions and attachments for the 110 selected
tasks from the official 2023 validation split. Upstream Modal startup messages
(`applied runtime debug patch` and `run_instance_modal` collision) are harmless
for this local-workspace runner. OpenHands executes tools locally;
use a dedicated evaluation environment.

```bash
conda env create -f environment.yml -n sharp-gaia
conda activate sharp-gaia
conda install -y -c conda-forge nodejs=22
bash scripts/setup.sh gaia --conda
uvx playwright install chromium
hf auth login
python scripts/prepare_data.py gaia
python scripts/prepare_data.py gaia --check-only
```

Set `SEARXNG_URL` to your search service, then run the experiments:

```bash
export SEARXNG_URL=http://127.0.0.1:8888
python scripts/run_gaia.py --phase screen
python scripts/run_gaia.py --phase validation
```

Scores and token usage are saved in `outputs/gaia/<phase>/summary.json`.
For a small model run, use `--arm full --limit-tasks 1 --output outputs/gaia-check`.

## Saliency Estimation

LIFE and Shopping screens produce `single_off.csv` and `components.json` under
`outputs/life/<domain>/screen/model/` and `outputs/shopping/screen/model/`,
respectively. For PaperQA, use the export above.

```bash
conda activate sharp-core
python -m sharp.screen \
  --input outputs/life/airline/screen/model/single_off.csv \
  --components outputs/life/airline/screen/model/components.json \
  --objective efficiency --output outputs/airline-saliency.json
```

Use `--objective performance` for performance saliency. Custom inputs use CSV
columns `task_id,component,score,tokens`, with one observation per task and removed
module. This CLI expects a complete single-trial ablation grid; GAIA's multi-trial
aggregation is separate. The experiment runners use the stored paper
configurations rather than automatically adopting a newly estimated ladder.

## Upstream Projects

[LIFE-harness](https://github.com/Tianshi-Xu/Life-Harness),
[JIT](https://github.com/bingreeky/JIT),
[PaperQA](https://github.com/Future-House/paper-qa), and
[OpenHands benchmarks](https://github.com/All-Hands-AI/benchmarks) are pinned in
[configs/upstreams.json](configs/upstreams.json). See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for license notices.

## Citation

If you use SHarP in your research, please cite our paper:

```bibtex
@misc{gao2026sharpsaliencybasedpruningagent,
      title={SHarP: Saliency-based Pruning of Agent Harnesses},
      author={Xinyi Gao and Qiucheng Wu and Kaizhi Qian and Handong Zhao and Shiyu Chang and Yang Zhang},
      year={2026},
      eprint={2610.04178},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2610.04178},
}
```
