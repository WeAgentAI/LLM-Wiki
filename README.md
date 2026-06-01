# Retrieval as Reasoning: Self-Evolving Agent-Native Retrieval via LLM-Wiki

**Haoliang Ming, Feifei Li, Xiaoqing Wu, Wenhui Que**

This repository
This repository contains the official code release for **LLM-Wiki**, an
agent-native retrieval system that operationalizes the
*Retrieval-as-Reasoning* paradigm. LLM-Wiki compiles documents into
structured Wiki pages with bidirectional links, exposes `wiki_search`,
`wiki_read`, and link-following operations through standard tool-calling
interfaces, and introduces an Error Book for persistent structural and
semantic self-correction. It supports both offline Wiki compilation and
online retrieval / question answering / evaluation.

> **Paper (arXiv):** https://arxiv.org/abs/2605.25480

---

## Repository layout

```
release/
├── README.md
├── LICENSE
├── requirements.txt
├── arxiv.txt
├── configs/
│   ├── page_types.yaml       # default page-type catalog
│   ├── purpose_bench.md      # default purpose template
│   └── wiki-schema.md        # wiki schema specification
├── llm_wiki_bench/
│   ├── __init__.py
│   ├── bench_config.py       # dataset paths, LLM config, wiki directory management
│   ├── llm_client.py         # OpenAI-compatible API client (chat + tool calls)
│   ├── download_datasets.py  # download public dev sets
│   ├── preprocess_bench.py   # raw paragraphs → Markdown articles
│   ├── bench_ingest.py       # two-step LLM ingestion engine
│   ├── bench_error_book.py   # error book for self-correction
│   ├── run.py                # offline wiki construction runner
│   ├── wiki_retriever.py     # wiki_search + wiki_read tools
│   ├── wiki_agent.py         # Retrieval-as-Reasoning tool-calling agent
│   ├── run_qa.py             # end-to-end retrieval + answer runner
│   └── evaluate.py           # EM / F1 evaluation
└── examples/
    └── run_hotpotqa.sh       # minimal reproduction example
```

## Requirements

```bash
pip install -r requirements.txt
```

Python ≥ 3.10. The code calls any **OpenAI-compatible** chat-completion API
(OpenAI, Azure OpenAI, vLLM, Ollama, etc.) over HTTP.

## Configure the LLM backend

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"   # or your local server

# Models used by the pipeline (override as needed):
export LLM_PREMIUM_MODEL="gpt-4o"       # strong model — synthesis steps
export LLM_FAST_MODEL="gpt-4o-mini"     # fast model  — analysis steps
```

## Quickstart: build a wiki on HotpotQA

```bash
# Full pipeline (download → preprocess → ingest)
python -m llm_wiki_bench.run --dataset hotpotqa --limit 500

# Or run each stage separately:
python -m llm_wiki_bench.run --dataset hotpotqa --only-download
python -m llm_wiki_bench.run --dataset hotpotqa --only-preprocess --limit 500
python -m llm_wiki_bench.run --dataset hotpotqa --only-ingest
```

The compiled wiki is written to `wiki_output/<dataset>/wiki/`.

## Run retrieval & answer evaluation

Once a wiki has been compiled, the agent can traverse it to answer questions.
The agent composes `wiki_search` and `wiki_read` calls, follows wikilinks,
and checks evidence sufficiency before producing a final answer.

Paper-faithful defaults: tool-call budget `T_max = 15`, patience
`P = 3` consecutive empty searches, and at most `k = 5` pages selected per
search.

```bash
# 1. Generate predictions (one JSONL line per question).
python -m llm_wiki_bench.run_qa --dataset hotpotqa --limit 500

# 2. Evaluate EM / F1 (with hop-wise and type-wise breakdowns).
python -m llm_wiki_bench.evaluate \
    --dataset hotpotqa \
    --predictions results/hotpotqa/predictions.jsonl

# Or do both in a single pass:
python -m llm_wiki_bench.run_qa --dataset hotpotqa --limit 500 --evaluate
```

Results (predictions, summary, per-question details) are written under
`results/<dataset>/`.

## Build a wiki on your own corpus

Place one Markdown file per article under `raw/<corpus_name>/articles/`, then:

```python
import sys
sys.path.insert(0, "llm_wiki_bench")

import bench_config as config
import bench_ingest

config.set_dataset("my_corpus")
config.ensure_wiki_dirs()
article_paths = sorted(config.RAW_DIR.glob("*.md"))
bench_ingest.ingest_batch(article_paths, batch_size=3)
```

## Citation

If you find this code useful, please cite:

```bibtex
@misc{ming2026retrievalreasoningselfevolvingagentnative,
      title={Retrieval as Reasoning: Self-Evolving Agent-Native Retrieval via LLM-Wiki},
      author={Haoliang Ming and Feifei Li and Xiaoqing Wu and Wenhui Que},
      year={2026},
      eprint={2605.25480},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.25480},
}
```

## License

Released under the MIT License. See [`LICENSE`](./LICENSE).
