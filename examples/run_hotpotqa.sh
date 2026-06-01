#!/usr/bin/env bash
# Minimal end-to-end example for LLM-Wiki on HotpotQA:
#   (1) compile the wiki, (2) run the retrieval agent + answer LLM,
#   (3) evaluate EM / F1.
# Run from the `release/` directory.

set -euo pipefail

# 1. Configure the LLM backend (any OpenAI-compatible server works).
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-replace-me}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
export LLM_PREMIUM_MODEL="${LLM_PREMIUM_MODEL:-gpt-4o}"
export LLM_FAST_MODEL="${LLM_FAST_MODEL:-gpt-4o-mini}"

cd "$(dirname "$0")/.."

# 2. Compile the Wiki (download -> preprocess -> ingest).
python3 -m llm_wiki_bench.run --dataset hotpotqa --limit 500

# 3. Run retrieval + answer + evaluation.
python3 -m llm_wiki_bench.run_qa --dataset hotpotqa --limit 500 --evaluate

echo
echo "Done."
echo "  Wiki        : wiki_output/hotpotqa/wiki/"
echo "  Predictions : results/hotpotqa/predictions.jsonl"
echo "  Summary     : results/hotpotqa/hotpotqa_summary.json"
