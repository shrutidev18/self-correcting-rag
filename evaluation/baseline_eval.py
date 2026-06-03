import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from groq import Groq
from app.core.pipeline import BasicRAGPipeline
from app.utils.config import config
from app.utils.logger import logger

TEST_QUESTIONS_PATH = Path("./data/test_questions.json")
RESULTS_PATH        = Path("./evaluation/results/baseline_scores.json")
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

client = Groq(api_key=config.GROQ_API_KEY)


def score_faithfulness(answer: str, contexts: list) -> float:
    context_str = "\n".join(contexts)
    prompt = f"""Given the context and answer below, rate how faithful the answer is to the context.
Score 1 if the answer is fully supported by context.
Score 0 if the answer contains information not in the context.
Score 0.5 if partially supported.

Context: {context_str[:1000]}
Answer: {answer}

Reply with ONLY a number: 0, 0.5, or 1"""

    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )
        score = float(response.choices[0].message.content.strip())
        return min(max(score, 0), 1)
    except:
        return 0.5


def score_answer_relevancy(question: str, answer: str) -> float:
    prompt = f"""Does the answer actually address the question asked?
Score 1 if fully relevant.
Score 0.5 if partially relevant.
Score 0 if not relevant or says it doesn't know.

Question: {question}
Answer: {answer}

Reply with ONLY a number: 0, 0.5, or 1"""

    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )
        score = float(response.choices[0].message.content.strip())
        return min(max(score, 0), 1)
    except:
        return 0.5


def score_context_recall(contexts: list, ground_truth: str) -> float:
    context_str = "\n".join(contexts)
    prompt = f"""Does the context contain enough information to answer the ground truth?
Score 1 if context clearly contains the answer.
Score 0.5 if context is partially relevant.
Score 0 if context is irrelevant.

Context: {context_str[:1000]}
Ground truth answer: {ground_truth[:300]}

Reply with ONLY a number: 0, 0.5, or 1"""

    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )
        score = float(response.choices[0].message.content.strip())
        return min(max(score, 0), 1)
    except:
        return 0.5


def run_baseline_evaluation():
    logger.info("Loading test questions...")
    with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        test_questions = json.load(f)[:100]

    logger.info(f"Running evaluation on {len(test_questions)} questions...")
    pipeline = BasicRAGPipeline()

    faithfulness_scores   = []
    relevancy_scores      = []
    context_recall_scores = []
    individual_results    = []

    for i, q in enumerate(test_questions):
        logger.info(f"[{i+1}/{len(test_questions)}] {q['question'][:60]}...")

        result   = pipeline.run(q["question"])
        contexts = [c["text"] for c in result["chunks"]]
        answer   = result["answer"]
        question = q["question"]
        truth    = q["answer"]

        time.sleep(1)
        f_score = score_faithfulness(answer, contexts)
        time.sleep(1)
        r_score = score_answer_relevancy(question, answer)
        time.sleep(1)
        c_score = score_context_recall(contexts, truth)
        time.sleep(2)

        faithfulness_scores.append(f_score)
        relevancy_scores.append(r_score)
        context_recall_scores.append(c_score)

        individual_results.append({
            "question":         question,
            "answer":           answer,
            "faithfulness":     f_score,
            "answer_relevancy": r_score,
            "context_recall":   c_score,
        })

        logger.info(f"  F={f_score} | R={r_score} | C={c_score}")

    scores = {
        "faithfulness":     round(sum(faithfulness_scores)   / len(faithfulness_scores),   4),
        "answer_relevancy": round(sum(relevancy_scores)      / len(relevancy_scores),      4),
        "context_recall":   round(sum(context_recall_scores) / len(context_recall_scores), 4),
        "num_questions":    len(test_questions),
        "individual":       individual_results,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)

    print("\n" + "─" * 50)
    print("BASELINE EVALUATION RESULTS")
    print("─" * 50)
    print(f"Faithfulness     : {scores['faithfulness']}")
    print(f"Answer Relevancy : {scores['answer_relevancy']}")
    print(f"Context Recall   : {scores['context_recall']}")
    print(f"Questions scored : {scores['num_questions']}")
    print("─" * 50)
    print(f"Saved → {RESULTS_PATH}")


if __name__ == "__main__":
    run_baseline_evaluation()