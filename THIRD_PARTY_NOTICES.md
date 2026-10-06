# Upstream notices

SHarP integrates the following projects at the revisions in `configs/upstreams.json`:

| Project | Usage | License copy |
| --- | --- | --- |
| LIFE-harness | tau2 agent harness and benchmark | `third_party/life-LICENSE.txt` |
| JIT | Agentfold harness and DeepPlanning shopping tools/data adapter | `third_party/jit-LICENSE.txt` |
| PaperQA | Retrieval agent, plus `patches/paperqa2.patch` | `third_party/paperqa-LICENSE.txt` |
| OpenHands benchmarks | GAIA evaluator, plus `patches/gaia.patch` | `third_party/openhands-benchmarks-LICENSE.txt` |

Setup retrieves these projects into `external/` and preserves their original license
and notice files. The OpenHands SDK is retrieved at the benchmark's pinned submodule
revision and retains its own license. Dataset and model terms remain those of their
respective upstream releases. The optional LitQA2 Release supplement contains
unchanged Creative Commons PDFs with individual licenses and source attribution
in its `ATTRIBUTION.md` and `configs/litqa2_supplement.json`. Those documents retain
their original licenses; the SHarP code license does not apply to them. Other
source PDFs and gated GAIA data are not redistributed.
