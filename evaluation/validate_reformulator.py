import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.retriever    import Retriever
from app.core.scorer       import Scorer
from app.core.reformulator import Reformulator
from app.utils.logger      import logger

RESULTS_PATH    = Path("./evaluation/results/baseline_scores.json")
VALIDATION_PATH = Path("./evaluation/results/reformulator_validation.json")

GOOD_THRESHOLD        = 3.5   # matched to our tuned scorer threshold
INSUFFICIENT_THRESHOLD = 2.5  # below this = give up


def run_validation():
    #Load failure cases 
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    failures = [q for q in baseline["individual"] if q["context_recall"] == 0.0]
    logger.info(f"Testing reformulator on {len(failures)} failure cases")

    retriever    = Retriever()
    scorer       = Scorer(threshold=GOOD_THRESHOLD)
    reformulator = Reformulator()

    results = []
    fixed   = 0
    gave_up = 0

    for i, case in enumerate(failures):
        question = case["question"]
        logger.info(f"\n[{i+1}/{len(failures)}] {question}")

        # Original retrieval + score
        orig_chunks  = retriever.retrieve(question)
        orig_scoring = scorer.score(question, orig_chunks)
        logger.info(f"  Original score: {orig_scoring['mean']}")

        #  Reformulate + re-retrieve
        rewrites     = reformulator.reformulate(question)
        best_score   = orig_scoring["mean"]
        best_rewrite = None
        rewrite_scores = []

        for rewrite in rewrites:
            r_chunks  = retriever.retrieve(rewrite)
            r_scoring = scorer.score(rewrite, r_chunks)
            rewrite_scores.append({
                "rewrite": rewrite,
                "score":   r_scoring["mean"],
            })
            logger.info(f"  Rewrite: '{rewrite[:50]}' | score={r_scoring['mean']}")

            if r_scoring["mean"] > best_score:
                best_score   = r_scoring["mean"]
                best_rewrite = rewrite

        #  Outcome 
        if best_score >= GOOD_THRESHOLD:
            outcome = "fixed"
            fixed  += 1
        elif best_score < INSUFFICIENT_THRESHOLD:
            outcome  = "gave_up"
            gave_up += 1
        else:
            outcome = "improved"  # better but still not great

        logger.info(
            f"  Best score: {best_score} | "
            f"Best rewrite: {best_rewrite} | Outcome: {outcome}"
        )

        results.append({
            "question":        question,
            "original_score":  orig_scoring["mean"],
            "best_score":      best_score,
            "best_rewrite":    best_rewrite,
            "rewrite_scores":  rewrite_scores,
            "outcome":         outcome,
        })

        # Delay to avoid Groq rate limits
        time.sleep(2)

    #  Summary
    improved_any = sum(1 for r in results if r["best_score"] > r["original_score"])
    avg_original = sum(r["original_score"] for r in results) / len(results)
    avg_best     = sum(r["best_score"]     for r in results) / len(results)

    print("\n" + "─" * 55)
    print("REFORMULATOR VALIDATION RESULTS")
    print("─" * 55)
    print(f"Total failure cases tested : {len(failures)}")
    print(f"Fixed (score >= {GOOD_THRESHOLD})         : {fixed}")
    print(f"Improved (score rose)      : {improved_any}")
    print(f"Gave up (score < {INSUFFICIENT_THRESHOLD})       : {gave_up}")
    print(f"Avg original score         : {avg_original:.3f}")
    print(f"Avg best score after reform: {avg_best:.3f}")
    print(f"Score improvement          : +{avg_best - avg_original:.3f}")
    print("─" * 55)

    #Save
    output = {
        "total_failures":      len(failures),
        "fixed":               fixed,
        "improved_any":        improved_any,
        "gave_up":             gave_up,
        "avg_original_score":  round(avg_original, 3),
        "avg_best_score":      round(avg_best, 3),
        "score_improvement":   round(avg_best - avg_original, 3),
        "good_threshold":      GOOD_THRESHOLD,
        "insufficient_threshold": INSUFFICIENT_THRESHOLD,
        "individual_results":  results,
    }

    with open(VALIDATION_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved → {VALIDATION_PATH}")


if __name__ == "__main__":
    run_validation()