import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset
from app.core.retriever import Retriever
from app.utils.logger import logger

MAX_DOCS = 20000
MAX_TEST_Q = 100


def download_and_index():
    logger.info("loading NQ dataset from hugging face...")
    dataset = load_dataset("sentence-transformers/natural-questions", split="train")
    logger.info(f"loaded {len(dataset):,} examples")

    logger.info(f"indexing first {MAX_DOCS} documents...")
    retriever = Retriever(reset_db=True)

    docs_to_index = []
    seen_ids = set()

    for row in dataset:
        if len(docs_to_index) >= MAX_DOCS:
            break
        doc_id = str(row.get("id", len(docs_to_index)))
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)
        docs_to_index.append({
            "id":    doc_id,
            "text":  row["answer"],
            "title": row["query"],
        })

    retriever.index_documents(docs_to_index)

    logger.info(f"saving {MAX_TEST_Q} test questions...")
    test_questions = []
    for i, row in enumerate(dataset):
        if len(test_questions) >= MAX_TEST_Q:
            break
        test_questions.append({
            "id":       str(i),
            "question": row["query"],
            "answer":   row["answer"],
        })

    out_path = Path("./data/test_questions.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(test_questions, f, indent=2, ensure_ascii=False)

    logger.info(f"saved {len(test_questions)} test questions → {out_path}")

    # quick sanity check
    for q in test_questions[:3]:
        results = retriever.retrieve(q["question"], k=3)
        print(f"\nQ: {q['question']}")
        for i, chunk in enumerate(results):
            print(f"  [{i+1}] score={chunk['score']:.3f} | {chunk['text'][:100]}...")

    print("\ndone!")
    print(f"  chunks indexed : {retriever.count():,}")
    print(f"  test questions : {len(test_questions)}")
    print(f"  next step      : python cli.py")


if __name__ == "__main__":
    download_and_index()