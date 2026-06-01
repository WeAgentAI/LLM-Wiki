"""Answer evaluation — Exact Match (EM) and token-level F1.

Implements the same normalization as the HotpotQA official evaluation script
(lowercase, strip punctuation/articles, collapse whitespace). Reports overall
EM/F1 and, when applicable, hop-wise / type-wise F1 breakdowns.

Usage:

    python -m llm_wiki_bench.evaluate \\
        --dataset hotpotqa \\
        --predictions results/hotpotqa/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"


# ─── HotpotQA-style normalization ─────────────────────────────────────────

def _normalize_answer(text: str) -> str:
    """Lower-case, strip punctuation, articles, and extra whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = " ".join(text.split())
    return text


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_toks = _normalize_answer(prediction).split()
    gold_toks = _normalize_answer(ground_truth).split()
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pred_toks)
    r = num_same / len(gold_toks)
    return 2 * p * r / (p + r)


def score_against_aliases(prediction: str, gold_answers: list[str]) -> dict:
    """Take the maximum EM/F1 over all gold answer aliases."""
    best_em, best_f1 = 0.0, 0.0
    for gt in gold_answers:
        best_em = max(best_em, exact_match(prediction, gt))
        best_f1 = max(best_f1, token_f1(prediction, gt))
    return {"em": best_em, "f1": best_f1}


# ─── I/O helpers ──────────────────────────────────────────────────────────

def _load_qa_pairs(dataset: str) -> list[dict]:
    path = DATA_DIR / dataset / "qa_pairs.jsonl"
    if not path.exists():
        sys.exit(f"❌ QA pairs not found: {path}\n   "
                 f"Run: python -m llm_wiki_bench.run --dataset {dataset} --only-preprocess")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_predictions(path: Path) -> dict:
    preds: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                preds[item["id"]] = item
    return preds


# ─── Aggregation ──────────────────────────────────────────────────────────

def evaluate(qa_pairs: list[dict], predictions: dict) -> tuple[dict, list[dict]]:
    em_sum = f1_sum = 0.0
    total = 0
    missing = 0
    steps_sum = 0
    pages_sum = 0
    hop_f1: dict[int, list[float]] = {}
    type_f1: dict[str, list[float]] = {}
    details: list[dict] = []

    for qa in qa_pairs:
        qid = qa["id"]
        if qid not in predictions:
            missing += 1
            continue
        pred = predictions[qid]
        prediction = pred.get("prediction", "") or ""

        gold = [qa["answer"]]
        if qa.get("answer_aliases"):
            gold.extend(qa["answer_aliases"])

        scores = score_against_aliases(prediction, gold)
        em_sum += scores["em"]
        f1_sum += scores["f1"]
        total += 1

        steps = pred.get("retrieval_steps")
        if steps is None:
            steps = len(pred.get("retrieval_trace", []) or [])
        pages = len(pred.get("retrieved_titles", []) or [])
        steps_sum += steps
        pages_sum += pages

        hop = len(qa.get("supporting_titles", [])) or 1
        hop_f1.setdefault(hop, []).append(scores["f1"])

        qtype = qa.get("type", "unknown")
        type_f1.setdefault(qtype, []).append(scores["f1"])

        details.append({
            "id": qid,
            "question": qa["question"],
            "gold_answer": qa["answer"],
            "prediction": prediction,
            "em": scores["em"],
            "f1": scores["f1"],
            "retrieval_steps": steps,
            "pages_read": pages,
            "hop_count": hop,
            "type": qtype,
        })

    if total == 0:
        return {}, details

    summary = {
        "total": total,
        "missing": missing,
        "em": em_sum / total,
        "f1": f1_sum / total,
        "avg_retrieval_steps": steps_sum / total,
        "avg_pages_read": pages_sum / total,
        "hop_wise_f1": {
            f"{h}-hop": {"f1": sum(v) / len(v), "count": len(v)}
            for h, v in sorted(hop_f1.items())
        },
        "type_wise_f1": {
            t: {"f1": sum(v) / len(v), "count": len(v)}
            for t, v in sorted(type_f1.items())
        },
    }
    return summary, details


# ─── CLI ──────────────────────────────────────────────────────────────────

def _print_summary(summary: dict, dataset: str) -> None:
    print(f"\n{'='*60}\n  Evaluation — {dataset}\n{'='*60}")
    print(f"  Total evaluated: {summary['total']}  (missing: {summary['missing']})")
    print(f"  Answer F1: {summary['f1']:.4f}  ({summary['f1']*100:.1f}%)")
    print(f"  Answer EM: {summary['em']:.4f}  ({summary['em']*100:.1f}%)")

    if summary["hop_wise_f1"]:
        print("  Hop-wise F1:")
        for label, info in summary["hop_wise_f1"].items():
            print(f"    {label}: F1={info['f1']:.4f}  n={info['count']}")
    type_wise = summary["type_wise_f1"]
    if type_wise and not (len(type_wise) == 1 and "unknown" in type_wise):
        print("  Type-wise F1:")
        for label, info in type_wise.items():
            print(f"    {label}: F1={info['f1']:.4f}  n={info['count']}")

    print(f"  Avg retrieval steps: {summary['avg_retrieval_steps']:.2f}")
    print(f"  Avg pages read:      {summary['avg_pages_read']:.2f}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLM-Wiki answer predictions.")
    parser.add_argument("--dataset", "-d", required=True,
                        choices=["hotpotqa", "musique", "2wikimhqa"])
    parser.add_argument("--predictions", "-p", required=True,
                        help="Path to predictions JSONL file.")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory (default: results/<dataset>/).")
    parser.add_argument("--limit", "-n", type=int, default=None,
                        help="Evaluate only the first N QA pairs.")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        sys.exit(f"❌ Predictions file not found: {pred_path}")

    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    qa_pairs = _load_qa_pairs(args.dataset)
    if args.limit:
        qa_pairs = qa_pairs[: args.limit]
    predictions = _load_predictions(pred_path)

    summary, details = evaluate(qa_pairs, predictions)
    if not summary:
        sys.exit("❌ No predictions matched any QA pair.")

    _print_summary(summary, args.dataset)

    summary_path = output_dir / f"{args.dataset}_summary.json"
    details_path = output_dir / f"{args.dataset}_details.jsonl"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(details_path, "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"  Saved summary: {summary_path}")
    print(f"  Saved details: {details_path}")


if __name__ == "__main__":
    main()
