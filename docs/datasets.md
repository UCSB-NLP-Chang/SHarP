# Benchmark data

Run preparation before starting a model server. All commands below run from the
repository root, in the corresponding Conda environment from the README.
Downloaded data stays in ignored `external/` or `outputs/` directories.

| Setting | SHarP screen / validation tasks | Acquisition |
| --- | --- | --- |
| LIFE / tau2 Airline | 30 / 20 | Included in the pinned LIFE checkout |
| LIFE / tau2 Retail | 50 / 40 | Included in the pinned LIFE checkout |
| Agentfold / DeepPlanning Shopping | 50 / 30 | Included in the pinned JIT checkout |
| PaperQA2 / LitQA2 | 50 / 38 | Pinned LAB-Bench records plus separate source PDFs |
| OpenHands / GAIA | 50 / 60 | Approved Hugging Face account; questions and attachments |

The SHarP splits are defined in `configs/splits/`. In particular, both SHarP GAIA
subsets come from the official **2023 validation** split; no hidden test answers
are required. LitQA2 uses 127 training-corpus PDFs and 133 validation-corpus PDFs,
including distractor documents. Downloading only each question's cited paper does
not reconstruct these corpora.

## LIFE and Shopping

Setup already retrieves the data together with the pinned upstream source. These
commands can also acquire a missing checkout and validate it without model calls:

```bash
python scripts/prepare_data.py life
python scripts/prepare_data.py shopping
```

The checks cover the selected task IDs and the tracked databases, policies, and
Shopping cart/scoring inputs. Use `--check-only` for a local check without network
access. A modified or incomplete checkout fails rather than silently reducing the
task set. Preserve any modified checkout elsewhere, then rerun the relevant setup
command to obtain a clean copy.

Sources: [LIFE-harness](https://github.com/Tianshi-Xu/Life-Harness),
[JIT](https://github.com/bingreeky/JIT), and the original
[DeepPlanning dataset](https://huggingface.co/datasets/Qwen/DeepPlanning).
The SHarP commands use the data version bundled in the pinned JIT commit, so no
additional Hugging Face download is needed for Shopping.

## LitQA2 question records

```bash
conda activate sharp-paperqa2
python scripts/prepare_data.py litqa2
for SPLIT in train validation; do
  python scripts/prepare_litqa.py \
    --records outputs/data/litqa2/litqa-v2-public.jsonl \
    --split "$SPLIT" --output "outputs/litqa/$SPLIT"
done
```

The download is pinned to a LAB-Bench commit and verified by SHA-256 in
`configs/data_sources.json`; all 88 selected question IDs must exist. Task
preparation writes `agent_inputs.jsonl` and `protected_targets.jsonl` separately.
Only the former is passed to the agent. The question records contain source
references, not the PDF bytes. Prepare the PDF corpus below before inference.

Source: [LAB-Bench](https://github.com/Future-House/LAB-Bench).

## LitQA2 PDF corpora

The release uses corpus version `litqa2-v2`: 160 distinct PDFs, shared between
the 127-document training and 133-document validation corpora. Allow about 4 GB
for the PDF pool and the two prepared copies, plus space for generated indexes.
The [source manifest](../configs/litqa2_corpus.json) records each paper's DOI,
title, version, download URL, split membership, and SHA-256. The
[PDF source list](litqa2-pdfs.md) provides clickable links for browser downloads.

```bash
python scripts/download_litqa_corpus.py
```

The script attempts 99 direct-source PDF downloads and 27 PDFs from a
[CC-licensed supplement](https://github.com/UCSB-NLP-Chang/SHarP/releases/tag/litqa2-v2-data).
It downloads the supplement once, checks the archive and each PDF, and keeps its
attribution file with the corpus. The supplement contains unchanged PDFs with
individual licenses listed in [its manifest](../configs/litqa2_supplement.json).
The remaining 34 entries require source downloads. The script resumes verified
files on later runs. An incomplete run exits nonzero
and writes `outputs/data/litqa2/pdfs/acquisition-all.json`, listing exactly which
files remain, their source pages, and expected hashes. Complete those downloads
before continuing with corpus preparation:

1. Open each remaining paper's source page from the PDF source list. For bioRxiv,
   use the specified version and posting date, then select **Download PDF**.
   For PMC/Europe PMC, select the article's PDF, using the author manuscript
   where specified. Download the main paper, not supplementary material or
   a browser-generated printout.
2. Save the PDFs together in a local directory. Publisher filenames can remain
   unchanged. Import and verify them:

```bash
python scripts/download_litqa_corpus.py \
  --import-dir /path/to/browser-downloads --check-only
python scripts/download_litqa_corpus.py --check-only
```

If the supplement was downloaded separately, import it without contacting GitHub:

```bash
python scripts/download_litqa_corpus.py \
  --supplement /path/to/litqa2-v2-cc-supplement.tar.gz --check-only
```

The import copies files whose content hashes match the manifest and leaves the
original downloads untouched. `import-report.json` lists unmatched files and
their hashes; these usually indicate a different paper version or a saved HTML
page. Use the recorded article version and PDF link, not **Print to PDF**.
If a publisher has replaced the listed PDF bytes, retain the mismatch report
and report the DOI and hash so the manifest can be updated explicitly.

The final check must report **160/160**. To prepare just one split, pass
`--split train` or `--split validation` to both download and check commands;
the corresponding counts are 127 and 133. Assemble the selected corpora:

```bash
for SPLIT in train validation; do
  python scripts/prepare_corpus.py \
    --pdf-pool outputs/data/litqa2/pdfs --split "$SPLIT" \
    --output "outputs/litqa/$SPLIT/corpus"
done
```

Use a new output directory for each prepared corpus. The script checks every
required hash before creating it. PDF licenses remain those of the respective
publishers. The optional Release supplement redistributes only the listed
CC-licensed PDFs with their attribution and original notices; other PDFs are
retrieved from their sources.
PMC bulk downloads use its [public cloud service](https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/).
PDFs available only through article pages are listed for browser acquisition.

## GAIA questions and attachments

Requires access to the [official GAIA dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA)
and Hugging Face authentication on the evaluation host.

```bash
conda activate sharp-gaia
hf auth login
python scripts/prepare_data.py gaia
python scripts/prepare_data.py gaia --check-only
```

The script downloads the pinned official validation snapshot, checks all 110
selected task IDs, and verifies that every selected attachment exists and is
nonempty. It records content hashes in `sharp-data.json` next to the snapshot.
The runner verifies these hashes before creating inference attempts and reads
this local Parquet metadata and these attachments. It does not redownload a
moving `main` revision during evaluation.

By default, the snapshot is placed at
`external/gaia/benchmarks/gaia/data`, matching the evaluator. For separate storage:

```bash
python scripts/prepare_data.py gaia --output /path/to/gaia
python scripts/run_gaia.py --data-dir /path/to/gaia --arm full --limit-tasks 1
```

Keep the entire prepared directory, including `sharp-data.json`, when moving it
to another evaluation machine. `--check-only` works without a network connection;
web-enabled agent tasks still require internet access.

## GAIA web-search service

Full and intermediate configurations need a reachable SearXNG instance with JSON
responses enabled. This is an external service, separate from the dataset.
For a local Docker deployment, using the
[official container](https://docs.searxng.org/admin/installation-docker.html):

```bash
mkdir -p outputs/searxng/config outputs/searxng/data
python - <<'PY'
from pathlib import Path
import secrets
p = Path('outputs/searxng/config/settings.yml')
if not p.exists():
    p.write_text('use_default_settings: true\nserver:\n  secret_key: "' + secrets.token_hex(32) + '"\n  limiter: false\nsearch:\n  formats: [html, json]\n')
PY
docker run -d --name sharp-searxng \
  -p 127.0.0.1:8888:8080 \
  -v "$PWD/outputs/searxng/config:/etc/searxng" \
  -v "$PWD/outputs/searxng/data:/var/cache/searxng" \
  docker.io/searxng/searxng:latest
export SEARXNG_URL=http://127.0.0.1:8888
curl --fail --get "$SEARXNG_URL/search" \
  --data-urlencode 'q=GAIA benchmark' --data-urlencode 'format=json' \
  -o outputs/searxng/check.json
python - <<'PY'
import json
from pathlib import Path
response = json.loads(Path('outputs/searxng/check.json').read_text())
assert response.get('results'), 'No search results; check SearXNG engines/network before inference'
print('Search endpoint returned', len(response['results']), 'results')
PY
```

Wait for container startup before the check. Record the deployed image digest
with `docker inspect sharp-searxng --format '{{.Image}}'`. Live search results can
change even with fixed code and data. Stop the service when finished with
`docker stop sharp-searxng`; on a later run use `docker start sharp-searxng`.
On hosts without Docker, supply a reachable SearXNG instance hosted
elsewhere and set `SEARXNG_URL` to its URL. Run the JSON search check above from
the evaluation host.
