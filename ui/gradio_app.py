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
        return "Please enter a question.", "", ""

    load_pipeline()

    result = pipeline.run(question)

    # Panel 1 — Answer
    answer = result["answer"]

    # Panel 2 — Retrieved chunks
    chunks_text = ""
    for i, chunk in enumerate(result["chunks"], 1):
        chunks_text += f"**Chunk {i} — Score: {chunk['score']:.3f}**\n"
        chunks_text += f"ID: `{chunk['id']}`\n\n"
        chunks_text += f"{chunk['text'][:300]}...\n\n"
        chunks_text += "---\n\n"

    # Panel 3 — Metadata
    metadata = f"""**Latency:** {result['latency_ms']:.0f}ms  
**Attempts:** {result['attempts']}  
**Chunks retrieved:** {len(result['chunks'])}  
**Sources:** {', '.join(result['sources'][:3])}"""

    return answer, chunks_text, metadata


with gr.Blocks(title="Self-Correcting RAG") as demo:

    gr.Markdown("""
    # 🔍 Self-Correcting RAG
    Ask a question and get a grounded answer from indexed documents.
    """)

    with gr.Row():
        question_box = gr.Textbox(
            label="Your Question",
            placeholder="e.g. What is photosynthesis?",
            lines=2,
            scale=4,
        )
        ask_btn = gr.Button("Ask", variant="primary", scale=1)

    with gr.Row():
        answer_box = gr.Textbox(
            label="Answer",
            lines=5,
            interactive=False,
        )

    with gr.Row():
        with gr.Column():
            chunks_box = gr.Markdown(label="Retrieved Chunks")
        with gr.Column():
            metadata_box = gr.Markdown(label="Metadata")

    ask_btn.click(
        fn=ask_question,
        inputs=[question_box],
        outputs=[answer_box, chunks_box, metadata_box],
    )

    question_box.submit(
        fn=ask_question,
        inputs=[question_box],
        outputs=[answer_box, chunks_box, metadata_box],
    )

    gr.Examples(
        examples=[
            ["What is photosynthesis?"],
            ["Who was the first president of the United States?"],
            ["What is DNA?"],
            ["What causes thunder?"],
        ],
        inputs=question_box,
    )

if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft())