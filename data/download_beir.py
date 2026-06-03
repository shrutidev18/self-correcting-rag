import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset
from app.core.retriever import Retriever
from app.utils.logger import logger

MAX_DOCS   = 20000
MAX_TEST_Q = 100

def download_and_index():

    logger.info("Loading NQ dataset from Hugging Face...")
    dataset = load_dataset("sentence-transformers/natural-questions", split="train")
    logger.info(f"Loaded {len(dataset):,} examples")

    logger.info(f"Indexing first {MAX_DOCS} documents...")
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

    logger.info(f"Saving {MAX_TEST_Q} test questions...")
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

    logger.info(f"Saved {len(test_questions)} test questions → {out_path}")

    logger.info("\n── Sanity check: 3 sample queries ──")
    for q in test_questions[:3]:
        results = retriever.retrieve(q["question"], k=3)
        print(f"\nQ: {q['question']}")
        for i, chunk in enumerate(results):
            print(f"  [{i+1}] score={chunk['score']:.3f} | {chunk['text'][:100]}...")

    print("\n✅ Setup complete!")
    print(f"   Chunks indexed  : {retriever.count():,}")
    print(f"   Test questions  : {len(test_questions)}")
    print(f"   Next step       : python cli.py")

if __name__ == "__main__":
    download_and_index()


'''Step 1 — Download the dataset
Downloads the Natural Questions dataset from BEIR — real Wikipedia passages with real questions and ground-truth answers. Saves to data/beir_datasets/.
Step 2 — Load corpus and queries
Loads everything into memory. corpus = all documents. queries = all questions. qrels = which documents are relevant to which question (ground truth).
Step 3 — Index documents
Takes the first 5,000 documents, passes them to Retriever.index_documents() which chunks, embeds, and stores them in ChromaDB.
Step 4 — Save test questions
Picks 100 questions whose relevant documents are inside our indexed 5,000. Saves them to data/test_questions.json. These 100 questions are used in Week 3 to measure how good your retrieval is.'''