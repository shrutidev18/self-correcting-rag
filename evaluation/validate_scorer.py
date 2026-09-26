import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.retriever import Retriever
from app.core.scorer import Scorer
from app.utils.logger import logger

RESULTS_PATH = Path("./evaluation/results/baseline_scores.json")
VALIDATION_PATH = Path("./evaluation/results/scorer_validation.json")

THRESHOLDS = [2.5, 3.0, 3.5]


def run_validation():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    failures = [q for q in baseline["individual"] if q["context_recall"] == 0.0]
    logger.info(f"found {len(failures)} retrieval failure cases")

    retriever = Retriever()
    scorer = Scorer(threshold=3.0)  # threshold doesn't matter here, we test all three

    scored_cases = []

    for i, case in enumerate(failures):
        question = case["question"]
        logger.info(f"[{i+1}/{len(failures)}] {question[:60]}")

        chunks = retriever.retrieve(question)
        scoring = scorer.score(question, chunks)

        scored_cases.append({
            "question":       question,
            "context_recall": case["context_recall"],
            "scorer_mean":    scoring["mean"],
            "scorer_scores":  scoring["scores"],
            "token_count":    scoring["token_count"],
        })

        logger.info(f"  mean={scoring['mean']} | scores={scoring['scores']}")
        time.sleep(1)

    print("\n" + "─" * 55)
    print("THRESHOLD ANALYSIS")
    print("─" * 55)

    threshold_results = {}

    for threshold in THRESHOLDS:
        correctly_flagged = sum(
            1 for c in scored_cases if c["scorer_mean"] < threshold
        )
        accuracy = correctly_flagged / len(scored_cases)

        threshold_results[threshold] = {
            "correctly_flagged": correctly_flagged,
            "total":             len(scored_cases),
            "accuracy":          round(accuracy, 4),
        }

        print(f"threshold {threshold} → {correctly_flagged}/{len(scored_cases)} flagged correctly ({accuracy*100:.1f}%)")

    best_threshold = max(threshold_results, key=lambda t: threshold_results[t]["accuracy"])
    print(f"\nbest threshold: {best_threshold} ({threshold_results[best_threshold]['accuracy']*100:.1f}% accuracy)")
    print("─" * 55)

    total_tokens = sum(c["token_count"] for c in scored_cases)
    avg_tokens = total_tokens / len(scored_cases)
    tokens_per_100 = avg_tokens * 100

    print(f"\nCOST ANALYSIS")
    print("─" * 55)
    print(f"avg tokens per query (5 chunks): {avg_tokens:.0f}")
    print(f"tokens per 100 queries         : {tokens_per_100:.0f}")
    print(f"groq free tier                 : $0 cost")
    print("─" * 55)

    output = {
        "num_failure_cases":      len(failures),
        "threshold_results":      threshold_results,
        "best_threshold":         best_threshold,
        "avg_tokens_per_query":   round(avg_tokens, 1),
        "tokens_per_100_queries": round(tokens_per_100, 1),
        "scored_cases":           scored_cases,
    }

    with open(VALIDATION_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"saved → {VALIDATION_PATH}")


if __name__ == "__main__":
    run_validation()