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
import re
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
#
# Matches the answer-generation protocol used to produce the paper's reported
# numbers (and the other baselines in this benchmark, e.g. Dense/BM25 RAG,
# GraphRAG, LightRAG, HippoRAG2): the retrieved context is the primary
# evidence source, but the model MAY fall back on its own parametric
# knowledge when the context is incomplete. This keeps LLM-Wiki's answer
# policy consistent with every other system under comparison — only the
# retrieval side (Wiki structure + agentic search/read) differs.

_ANSWER_SYSTEM_PROMPT = """You are a multi-hop question answering assistant. Answer the question using the provided context as your primary reference. If the context is incomplete, you may supplement with your own knowledge.

## Instructions
1. The question may require combining facts from MULTIPLE pages to derive the answer.
2. First identify which pages contain relevant facts, then chain the facts together step by step.
3. For comparison questions ("which is older/larger/..."), extract the specific values from each entity and compare.
4. For bridge questions ("who is the director of the film starring X"), follow the chain: find X's film -> find that film's director.
5. For yes/no questions ("Are A and B both X?"), check each entity separately, then combine.
6. Use the Retrieval Path (if provided) as a hint for how the information connects.
7. **NATIONALITY COMPARISON RULES** (critical for "same country" questions):
   - A person's nationality is determined by their COUNTRY OF ORIGIN (birth country), not where they later moved.
   - "French-American" means the person is originally FROM FRANCE (born in France, later moved to America). Their nationality is FRENCH.
   - Similarly: "Italian-American" = Italian, "German-British" = German, "Irish-American" = Irish, etc.
   - If person A is "French" and person B is "French-American" (born in France), they ARE from the same country (France).
   - When comparing nationalities, focus on the ROOT nationality (the first/origin part of hyphenated descriptions).
   - "American film" does NOT mean the director is American -- check the director's actual nationality/birthplace.
8. **IMPORTANT**: Try your BEST to answer even with partial information. Make reasonable inferences from available context and your own knowledge.
9. If the context does not fully cover the answer, use your own knowledge to fill in the gaps.
10. Only say "unknown" if you truly cannot determine the answer from either the context or your own knowledge.

## OUTPUT FORMAT (CRITICAL -- you MUST follow this exactly)
- Output ONLY the final answer, nothing else.
- Do NOT output any reasoning, explanation, or thought process.
- Do NOT start with "Based on...", "According to...", "The answer is...", "Looking at...", or any prefix.
- Do NOT output sentences -- just the answer itself (a name, date, number, yes/no, or short phrase).
- For yes/no questions: output ONLY "yes" or "no" (lowercase).
- **Give the SHORTEST possible answer**:
  - For locations: give ONLY the city/region name, do NOT include country or administrative divisions (e.g., "Springfield" not "Springfield, Illinois, USA").
  - For dates: match the granularity of the question. If the question asks "what year", answer with just the year (e.g., "1990" not "June 15, 1990").
  - For people: use the shortest commonly recognized name (e.g., "Tom" if unambiguous, not "Thomas James Wilson III").
  - For entities: use the most concise identifying name without unnecessary qualifiers.
- Examples of CORRECT output: "yes", "no", "John Smith", "1990", "Springfield", "Portland"
- Examples of WRONG output: "Based on the context, the answer is John Smith.", "Springfield, Illinois, United States", "June 15, 1990" (when only year is asked)"""

_ANSWER_USER_TEMPLATE = """## Context (from Wiki knowledge base)
{context}

## Question
{question}

## Final Answer (ONLY the answer, no explanation):"""


# ─── Helpers ──────────────────────────────────────────────────────────────

def _format_context(pages: list[tuple[str, str]], pages_text: dict[str, str]) -> str:
    if not pages:
        return "(no pages retrieved)"
    chunks = []
    for rel_path, name in pages:
        body = pages_text.get(rel_path, "")
        chunks.append(f"### {name}\n**Path**: {rel_path}\n\n{body}")
    return "\n\n".join(chunks)


_THINKING_STEP_RE = re.compile(r'^\d+\.\s*\*\*.*\*\*', re.MULTILINE)

_UNKNOWN_INDICATORS = (
    "i cannot find any information",
    "there is no information in the context",
    "the pages do not contain any information",
    "none of the retrieved pages",
    "the context does not provide",
    "no relevant information was found",
)

_PREFIXES_TO_REMOVE = (
    "based on the provided context,", "based on the context,",
    "based on the retrieval path,", "based on the retrieved pages,",
    "according to the context,", "according to the pages,",
    "the answer is:", "the answer is", "answer:", "final answer:",
    "looking at this question step by step:", "looking at the context,",
    "the question asks",
)


def _extract_clean_answer(raw: str) -> str:
    """Extract a clean final answer from raw LLM output.

    Handles common formatting issues seen with `enable_thinking=True`:
    1. strips a leading numbered "thinking steps" block if the model leaked one,
    2. strips filler prefixes ("Based on...", "The answer is...", ...),
    3. maps clearly "no evidence found" phrasing to "unknown",
    4. for long outputs, falls back to the shortest non-bullet line as the answer,
    5. trims a trailing period on short phrase-style answers.
    """
    answer = raw.strip()

    if _THINKING_STEP_RE.match(answer):
        lines = [l.strip() for l in answer.split("\n") if l.strip()]
        non_thinking_lines = [
            l for l in lines
            if not re.match(r'^\d+\.\s*\*\*', l) and not l.startswith(("-", "*", "#"))
            and len(l) < 150
        ]
        if non_thinking_lines:
            answer = non_thinking_lines[-1]
        else:
            return "unknown"

    for prefix in _PREFIXES_TO_REMOVE:
        if answer.lower().startswith(prefix):
            answer = answer[len(prefix):].strip().lstrip(",: ").strip()
            break

    answer_lower = answer.lower()
    for indicator in _UNKNOWN_INDICATORS:
        if indicator in answer_lower and len(answer) > 100:
            return "unknown"

    if len(answer) > 200:
        lines = [l.strip() for l in answer.split("\n") if l.strip()]
        short_lines = [l for l in lines if len(l) < 100 and not l.startswith(("-", "*", "#"))]
        answer = short_lines[-1] if short_lines else (lines[0] if lines else answer)

    if answer.endswith(".") and len(answer) < 80:
        answer = answer[:-1].strip()

    return answer.strip()


def _answer(question: str, context: str, model: str | None = None) -> str:
    raw = call_llm(
        system_prompt=_ANSWER_SYSTEM_PROMPT,
        user_prompt=_ANSWER_USER_TEMPLATE.format(question=question, context=context),
        temperature=0.0,
        max_tokens=2048,
        model=model,
        enable_thinking=True,
    )
    return _extract_clean_answer(raw)


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
