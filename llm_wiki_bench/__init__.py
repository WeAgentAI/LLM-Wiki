"""LLM-Wiki benchmark package.

Offline wiki construction pipeline + Retrieval-as-Reasoning agent +
answer-quality evaluation for multi-hop QA datasets
(HotpotQA / MuSiQue / 2WikiMultiHopQA).
"""

import sys as _sys
from pathlib import Path as _Path

# Ensure the package directory is on sys.path so that sibling modules
# (bench_config, llm_client, bench_ingest, etc.) can import each other
# with plain `import <name>` statements.
_PKG_DIR = _Path(__file__).parent
if str(_PKG_DIR) not in _sys.path:
    _sys.path.insert(0, str(_PKG_DIR))

__all__ = [
    # Compilation pipeline
    "bench_config",
    "llm_client",
    "bench_ingest",
    "bench_error_book",
    "preprocess_bench",
    "download_datasets",
    "run",
    # Retrieval & evaluation
    "wiki_retriever",
    "wiki_agent",
    "run_qa",
    "evaluate",
]
