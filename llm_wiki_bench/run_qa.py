"""End-to-end QA runner over a compiled LLM-Wiki.

For each QA pair this script

  1. invokes the Wiki agent (Retrieval-as-Reasoning) to gather evidence,
  2. asks an answer LLM to produce a short final answer from that evidence,
  3. writes a JSONL prediction file, and (optionally) runs evaluation.

Usage:

    python -m llm_wiki_bench.run_qa --dataset hotpotqa --limit 5
    python -m llm_wiki_bench.run_qa --dataset hotpotqa --limit 500 --evaluate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make sibling modules importable when launched as `python release/.../run_qa.py`
_BENCH_DIR = Path(__file__).parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.append(str(_BENCH_DIR))

import bench_config as config                        # noqa: E402
import evaluate as _evaluate                         # noqa: E402
from llm_client import call_llm, call_llm_with_tools  # noqa: E402
from wiki_agent import WikiAgent                     # noqa: E402
from wiki_retriever import WikiRetriever             # noqa: E402


# ─── Answer prompt (dataset-agnostic) ─────────────────────────────────────

_ANSWER_SYSTEM_PROMPT = """You are a question-answering assistant. Use the retrieved Wiki context as your primary source of evidence.

Output rules:
- Answer with the shortest span that fully addresses the question.
- Output ONLY the answer text — no preamble, no explanation, no quotes.
- If the context does not contain the answer, output exactly: unknown
"""

_ANSWER_USER_TEMPLATE = """## Question
{question}

## Retrieved Wiki Context
{context}

## Answer (concise span only):"""


# ─── Helpers ──────────────────────────────────────────────────────────────

def _format_context(pages: list[tuple[str, str]], pages_text: dict[str, str]) -> str:
    if not pages:
        return "(no pages retrieved)"
    chunks = []
    for rel_path, name in pages:
        body = pages_text.get(rel_path, "")
        chunks.append(f"### {name}\n**Path**: {rel_path}\n\n{body}")
    return "\n\n".join(chunks)


_PREFIXES = (
    "based on", "according to", "the answer is",
    "looking at", "from the context",
)


def _strip_prefix(answer: str) -> str:
    s = answer.strip()
    low = s.lower()
    for p in _PREFIXES:
        if low.startswith(p):
            tail = s.split(",", 1)[-1] if "," in s else s.split(":", 1)[-1]
            return tail.strip().strip(".")
    return s


def _answer(question: str, context: str, model: str | None = None) -> str:
    raw = call_llm(
        system_prompt=_ANSWER_SYSTEM_PROMPT,
        user_prompt=_ANSWER_USER_TEMPLATE.format(question=question, context=context),
        temperature=0.0,
        max_tokens=128,
        model=model,
    )
    return _strip_prefix(raw)


# ─── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Retrieval-as-Reasoning QA over a compiled LLM-Wiki."
    )
    parser.add_argument("--dataset", "-d", required=True,
                        choices=["hotpotqa", "musique", "2wikimhqa"])
    parser.add_argument("--limit", "-n", type=int, default=None,
                        help="Process only the first N QA pairs.")
    parser.add_argument("--t-max", type=int, default=15,
                        help="Maximum tool-call budget per question (default 15).")
    parser.add_argument("--patience", type=int, default=3,
                        help="Stop after this many consecutive empty searches (default 3).")
    parser.add_argument("--select-pages", type=int, default=5,
                        help="Maximum pages selected per wiki_search (default 5).")
    parser.add_argument("--retrieval-model", default=None,
                        help="Model used for the retrieval agent (default: LLM_PREMIUM_MODEL).")
    parser.add_argument("--answer-model", default=None,
                        help="Model used to write the final answer (default: LLM_MODEL).")
    parser.add_argument("--output", "-o", default=None,
                        help="Predictions output path (default: results/<dataset>/predictions.jsonl).")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation immediately after prediction.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # 1. Locate the compiled Wiki for this dataset.
    config.set_dataset(args.dataset)
    config.ensure_wiki_dirs()
    wiki_dir = Path(config.WIKI_DIR)
    if not wiki_dir.exists():
        sys.exit(f"❌ Compiled wiki not found: {wiki_dir}\n"
                 f"   Build it first: python -m llm_wiki_bench.run --dataset {args.dataset}")

    # 2. Load QA pairs.
    qa_path = Path(config.BASE_DIR) / "data" / args.dataset / "qa_pairs.jsonl"
    if not qa_path.exists():
        sys.exit(f"❌ QA pairs not found: {qa_path}\n"
                 f"   Run: python -m llm_wiki_bench.run --dataset {args.dataset} --only-preprocess")
    with open(qa_path, encoding="utf-8") as f:
        qa_pairs = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        qa_pairs = qa_pairs[: args.limit]

    print(f"Wiki dir : {wiki_dir}")
    print(f"QA pairs : {len(qa_pairs)}")
    print(f"T_max={args.t_max}  P={args.patience}  k={args.select_pages}")

    # 3. Initialize retriever + agent.
    retriever = WikiRetriever(wiki_dir)
    retriever.load()
    print(f"Loaded   : {len(retriever.pages)} pages, {len(retriever.dir_indexes)} directory indices")

    retrieval_model = args.retrieval_model or getattr(config, "LLM_PREMIUM_MODEL", None) or config.LLM_MODEL
    agent = WikiAgent(
        retriever,
        call_llm_with_tools=call_llm_with_tools,
        model=retrieval_model,
        t_max=args.t_max,
        patience=args.patience,
        select_pages=args.select_pages,
        verbose=args.verbose,
    )

    # 4. Run.
    out_path = Path(args.output) if args.output else (
        Path(config.BASE_DIR) / "results" / args.dataset / "predictions.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as fout:
        for i, qa in enumerate(qa_pairs, 1):
            question = qa["question"]
            try:
                rr = agent.retrieve(question)
                context = _format_context(rr.pages, rr.pages_text)
                prediction = _answer(question, context, model=args.answer_model)
            except Exception as e:  # noqa: BLE001
                print(f"  [{i}/{len(qa_pairs)}] ❌ {qa['id']}: {e}")
                prediction = "unknown"
                rr = None

            record = {
                "id": qa["id"],
                "question": question,
                "prediction": prediction,
                "gold_answer": qa.get("answer", ""),
                "retrieved_titles": [name for _, name in (rr.pages if rr else [])],
                "retrieval_trace": rr.trace if rr else [],
                "retrieval_steps": rr.total_calls if rr else 0,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            if i % 10 == 0 or i == len(qa_pairs) or args.verbose:
                elapsed = time.time() - t0
                print(f"  [{i}/{len(qa_pairs)}] {qa['id']}  steps={record['retrieval_steps']}  "
                      f"pages={len(record['retrieved_titles'])}  ({elapsed:.1f}s)")

    print(f"\nPredictions saved: {out_path}")
    print(f"Total time       : {time.time() - t0:.1f}s")

    # 5. Optional evaluation.
    if args.evaluate:
        predictions = _evaluate._load_predictions(out_path)
        summary, details = _evaluate.evaluate(qa_pairs, predictions)
        if summary:
            _evaluate._print_summary(summary, args.dataset)
            results_dir = Path(config.BASE_DIR) / "results" / args.dataset
            results_dir.mkdir(parents=True, exist_ok=True)
            with open(results_dir / f"{args.dataset}_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            with open(results_dir / f"{args.dataset}_details.jsonl", "w", encoding="utf-8") as f:
                for d in details:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
