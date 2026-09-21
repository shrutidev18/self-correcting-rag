# Self-Correcting RAG System

A Retrieval-Augmented Generation system that detects when retrieval fails and automatically fixes itself instead of hallucinating an answer.

![UI Screenshot](assets/ui_screenshot1.png)
![UI Screenshot 2](assets/ui_screenshot2.png)

---

## The Problem

Standard RAG systems blindly trust whatever documents they retrieve. When the retrieved chunks don't actually contain the answer, the LLM still tries to answer and hallucinates. There's no mechanism to detect this failure or do anything about it.

User Query → Retrieve → Generate → Answer (even if wrong)


## The Solution

This system scores the quality of retrieved chunks before generating an answer. If the score is too low, it rewrites the query, retrieves again, and picks the best result. If nothing helps, it honestly says it doesn't have enough information.

User Query → Retrieve → Score chunks → Good? → Generate → Answer
↓
Poor? → Rewrite query → Retrieve again → Generate → Answer
↓
Still poor? → "Insufficient context"


---

## Demo

[![Demo Video](assets/demo_thumbnail.png)](assets/demo.mp4)

> Or watch the demo: [Link to video]

---

## Results

Evaluated on 100 questions from the Natural Questions dataset:

| Metric | Baseline RAG | Self-Correcting RAG | Δ |
|---|---|---|---|
| Faithfulness | 0.900 | 0.845 | -6.1% |
| Answer Relevancy | 0.715 | 0.775 | **+8.4%** |
| Context Recall | 0.875 | 0.860 | -1.7% |

**Efficiency:**
- 3% of queries triggered self-correction
- Average latency: 7,490ms per query
- Model: qwen/qwen3.8-27b (Groq free tier)

---

## Architecture

![Architecture](assets/architecture.png)

### How It Works

**1. Retrieval**
The user's question is converted to a vector embedding using `sentence-transformers/all-MiniLM-L6-v2` and searched against 60,000+ indexed chunks in ChromaDB.

**2. Quality Scoring**
Each retrieved chunk is scored 1-5 by the LLM:
- **5** — Directly answers the question
- **4** — Relevant and useful
- **3** — Somewhat relevant
- **2** — Marginally related
- **1** — Completely irrelevant

The mean score is compared against a threshold. If below → self-correction triggers.

**3. Query Reformulation**
The LLM rewrites the original query 3 ways — broader terms, synonyms, sub-questions. Each rewrite is retrieved and scored. The best one is selected.

**4. Generation**
The final answer is generated using the best available chunks.

**5. LangGraph State Machine**
The entire pipeline is implemented as a LangGraph StateGraph with 6 nodes:

retrieve_node → score_node → decide_node
↓
┌─────────────┼──────────────┐
↓ ↓ ↓
generate_node reformulate_node fallback_node
↓ ↓
END retrieve_node (loop back)


---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq API — qwen/qwen3.8-27b |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (local) |
| Vector DB | ChromaDB (persistent) |
| Orchestration | LangGraph |
| Dataset | Natural Questions (HuggingFace) — 20,000 docs |
| UI | Gradio |
| Deployment | Docker |
| Testing | pytest (31 tests, all passing) |

---

## Project Structure

self-correcting-rag/
├── app/
│ ├── core/
│ │ ├── retriever.py ← ChromaDB + Sentence-BERT search
│ │ ├── generator.py ← Groq LLM answer generation
│ │ ├── scorer.py ← LLM-based chunk quality scoring
│ │ ├── reformulator.py ← Query rewriting engine
│ │ └── pipeline.py ← LangGraph state machine
│ └── utils/
│ ├── config.py ← Environment settings
│ └── logger.py ← JSONL logging
├── evaluation/
│ ├── baseline_eval.py ← Baseline RAG evaluation
│ ├── scrag_eval.py ← Self-correcting RAG evaluation
│ └── results/
│ ├── baseline_scores.json
│ ├── scrag_scores.json
│ └── comparison.json
├── tests/
│ ├── test_retriever.py ← 11 tests
│ ├── test_scorer.py ← 11 tests
│ └── test_reformulator.py ← 9 tests
├── ui/
│ └── gradio_app.py ← Browser UI
├── data/
│ └── test_questions.json ← 100 evaluation questions
├── Dockerfile
├── docker-compose.yml
└── requirements.txt


---

## Quick Start

### Option 1 — Docker (Recommended)

```bash
# Clone the repo
git clone https://github.com/shrutidev18/self-correcting-rag.git
cd self-correcting-rag

# Add your Groq API key to .env
cp .env.example .env
# Edit .env and add GROQ_API_KEY=your_key_here

# Run
docker compose up

# Open browser
http://127.0.0.1:7860
```

### Option 2 — Local

```bash
# Clone and setup
git clone https://github.com/shrutidev18/self-correcting-rag.git
cd self-correcting-rag
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Add your Groq API key
cp .env.example .env
# Edit .env and add GROQ_API_KEY=your_key_here

# Index documents (first time only)
python data/download_beir.py

# Run UI
python ui/gradio_app.py

# Or run CLI
python cli.py "What is photosynthesis?"
```

### Run Tests

```bash
pytest tests/ -v
```

---

## Get a Free Groq API Key

1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up for free
3. Create an API key
4. Add it to your `.env` file

---

## Key Findings

- Answer Relevancy improved by 8.4% with self-correction enabled
- The scorer works well for clear cases (score 5 or 1) but struggles with ambiguous chunks
- Reformulation helped most when the original query was too colloquial — e.g. "who sang go rest high on the mountain" → "original artist of Go Rest High on That Mountain"
- The system correctly refused to answer rather than hallucinate in all fallback ca-

---

## Limitations

- Scorer adds latency (~2-5 seconds per query for 5 scoring calls)
- Daily token limit on Groq free tier limits large-scale evaluation
- The scorer sometimes gives harsh scores (all 1s except one 5) which can incorrectly flag good retrievals as poor, a larger model would judge more consistently
- Self-correction only retries once — multiple retries could improve recall further

---

## Built By

**Shruti Dev**  
[LinkedIn](www.linkedin.com/in/shrutiidev) · [GitHub](https://github.com/shrutidev18)

---

## License

MIT