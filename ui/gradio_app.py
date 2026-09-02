import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr
from app.core.pipeline import BasicRAGPipeline
from app.utils.logger import logger

pipeline = None


def load_pipeline():
    global pipeline
    if pipeline is None:
        logger.info("Loading pipeline...")
        pipeline = BasicRAGPipeline()
        logger.info("Pipeline ready!")


def ask_question(question: str):
    if not question.strip():
        return "Please enter a question.", "", "", ""

    load_pipeline()
    result = pipeline.run(question)

    answer      = result["answer"]
    chunks      = result["chunks"]
    scoring     = result.get("scoring", {})
    attempts    = result.get("attempts", 1)
    latency     = result.get("latency_ms", 0)
    reformulated = result.get("reformulated_query", "")

    quality    = scoring.get("quality", "unknown")
    mean_score = scoring.get("mean", 0.0)
    scores     = scoring.get("scores", [])

    # ── Answer ────────────────────────────────────────────────────────────────
    answer_md = answer

    # ── How it worked ─────────────────────────────────────────────────────────
    if attempts == 1 and quality == "good":
        how_md = f"""### What happened

The system retrieved relevant documents on the first try and generated an answer directly.

| | |
|---|---|
| Retrieval quality | Good ({mean_score:.1f} / 5.0) |
| Attempts needed | 1 |
| Time taken | {latency:.0f}ms |
"""
    elif attempts > 1 and reformulated:
        how_md = f"""### What happened

The initial retrieval wasn't useful enough, so the system automatically rewrote the question and tried again.

**Original question**
> {question}

**Rewritten as**
> {reformulated}

| | |
|---|---|
| Initial quality | Poor ({scoring.get('mean', 0):.1f} / 5.0) |
| After rewrite | Better ({mean_score:.1f} / 5.0) |
| Attempts needed | {attempts} |
| Time taken | {latency:.0f}ms |
"""
    else:
        how_md = f"""### What happened

The system tried rewriting the question but still couldn't find enough relevant context.

| | |
|---|---|
| Attempts | {attempts} |
| Time taken | {latency:.0f}ms |
"""

    # ── Sources ───────────────────────────────────────────────────────────────
    sources_md = "### Sources used\n\n"
    for i, chunk in enumerate(chunks, 1):
        relevance = scores[i-1] if i-1 < len(scores) else "—"
        sources_md += f"**{i}.** `{chunk['id']}` — relevance {relevance}/5\n\n"
        sources_md += f"{chunk['text'][:250]}...\n\n"
        if i < len(chunks):
            sources_md += "---\n\n"

    # ── Stats ─────────────────────────────────────────────────────────────────
    stats_md = f"""### Stats

| Metric | Value |
|---|---|
| Latency | {latency:.0f}ms |
| Attempts | {attempts} |
| Chunks scored | {len(chunks)} |
| Threshold | {scoring.get('threshold', 2.0)} |
| Chunk scores | {scores} |
"""

    return answer_md, how_md, sources_md, stats_md


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Raleway:wght@400;600;700&display=swap');

* { font-family: 'Raleway', sans-serif !important; }

body, .gradio-container {
    background-color: #2d3250 !important;
    color: #ffffff !important;
}

/* Header */
.app-header {
    background: linear-gradient(135deg, #2d3250 0%, #424769 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 8px;
    border: 1px solid #424769;
}

.app-header h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #f9b17a !important;
    margin: 0 0 8px 0;
}

.app-header p {
    color: #b0b8d4 !important;
    font-size: 1rem;
    margin: 0;
    line-height: 1.6;
}

/* Input area */
.input-area {
    background: #424769 !important;
    border-radius: 12px !important;
    border: 1px solid #676f9d !important;
    padding: 4px !important;
}

.input-area textarea {
    background: #424769 !important;
    color: #ffffff !important;
    font-size: 1rem !important;
    border: none !important;
}

.input-area textarea::placeholder { color: #676f9d !important; }

/* Ask button */
.ask-btn {
    background: #f9b17a !important;
    color: #2d3250 !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border-radius: 12px !important;
    border: none !important;
    height: 100% !important;
    transition: opacity 0.2s !important;
}

.ask-btn:hover { opacity: 0.85 !important; }

/* Answer box */
.answer-panel {
    background: #424769 !important;
    border-radius: 12px !important;
    border-left: 4px solid #f9b17a !important;
    padding: 20px 24px !important;
    font-size: 1.05rem !important;
    line-height: 1.8 !important;
    color: #ffffff !important;
    min-height: 80px !important;
}

/* Info panels */
.info-panel {
    background: #363b5e !important;
    border-radius: 12px !important;
    border: 1px solid #424769 !important;
    padding: 20px !important;
    color: #ffffff !important;
}

.info-panel h3 {
    color: #f9b17a !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin-bottom: 12px !important;
}

.info-panel table {
    width: 100% !important;
    border-collapse: collapse !important;
}

.info-panel td {
    padding: 6px 8px !important;
    color: #b0b8d4 !important;
    font-size: 0.9rem !important;
    border-bottom: 1px solid #424769 !important;
}

.info-panel td:first-child { color: #676f9d !important; }

.info-panel blockquote {
    border-left: 3px solid #f9b17a !important;
    margin: 8px 0 !important;
    padding-left: 12px !important;
    color: #b0b8d4 !important;
    font-style: italic !important;
}

/* Examples */
.examples-section {
    background: #363b5e !important;
    border-radius: 12px !important;
    border: 1px solid #424769 !important;
    padding: 16px !important;
}

/* Divider */
.divider {
    border: none !important;
    border-top: 1px solid #424769 !important;
    margin: 16px 0 !important;
}

/* Hide gradio footer */
footer { display: none !important; }
.svelte-1ipelgc { display: none !important; }
"""

with gr.Blocks(title="Self-Correcting RAG", css=CSS) as demo:

    gr.HTML("""
    <div class="app-header">
        <h1>Self-Correcting RAG</h1>
        <p>
            A retrieval system that knows when it's wrong — and fixes itself.<br>
            When a question doesn't retrieve useful context, the system rewrites 
            the query and tries again, instead of hallucinating an answer.
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=5):
            question_box = gr.Textbox(
                label="",
                placeholder="Ask anything — e.g. Who sang Go Rest High on the Mountain?",
                lines=2,
                elem_classes=["input-area"],
            )
        with gr.Column(scale=1, min_width=120):
            ask_btn = gr.Button(
                "Ask →",
                variant="primary",
                size="lg",
                elem_classes=["ask-btn"],
            )

    answer_box = gr.Markdown(
        label="",
        elem_classes=["answer-panel"],
        value="*Your answer will appear here...*"
    )

    gr.HTML('<hr class="divider">')

    with gr.Row():
        with gr.Column(scale=2):
            how_box = gr.Markdown(
                label="",
                elem_classes=["info-panel"],
                value="### What happened\n\n*Ask a question to see the self-correction trace.*"
            )
        with gr.Column(scale=3):
            sources_box = gr.Markdown(
                label="",
                elem_classes=["info-panel"],
                value="### Sources used\n\n*Retrieved chunks will appear here.*"
            )
        with gr.Column(scale=1):
            stats_box = gr.Markdown(
                label="",
                elem_classes=["info-panel"],
                value="### Stats\n\n*Metrics will appear here.*"
            )

    gr.HTML('<hr class="divider">')

    gr.Examples(
        examples=[
            ["Who sang Go Rest High on the Mountain?"],
            ["What is photosynthesis?"],
            ["Who was the first president of the United States?"],
            ["Where was the movie Titanic filmed?"],
            ["What causes thunder?"],
            ["Who wrote the book The Jungle?"],
        ],
        inputs=question_box,
        label="Try these examples",
        elem_id="examples-section",
    )

    ask_btn.click(
        fn=ask_question,
        inputs=[question_box],
        outputs=[answer_box, how_box, sources_box, stats_box],
    )

    question_box.submit(
        fn=ask_question,
        inputs=[question_box],
        outputs=[answer_box, how_box, sources_box, stats_box],
    )

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
    )