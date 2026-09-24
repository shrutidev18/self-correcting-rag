import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr
from app.core.pipeline import BasicRAGPipeline
from app.core.document_processor import process_file
from app.core.indexer import Indexer
from app.utils.logger import logger

# ── Global pipeline cache ─────────────────────────────────────────────────────
_pipelines = {}


def get_pipeline(collection_name: str) -> BasicRAGPipeline:
    if collection_name not in _pipelines:
        logger.info(f"Loading pipeline for collection: {collection_name}")
        _pipelines[collection_name] = BasicRAGPipeline(collection_name=collection_name)
    return _pipelines[collection_name]


# ── Ask tab functions ─────────────────────────────────────────────────────────

def ask_question(question: str, collection_name: str):
    if not question.strip():
        return "Please enter a question.", "", "", ""

    if not collection_name or collection_name.strip() == "":
        collection_name = "sc_rag_docs"

    try:
        pipeline = get_pipeline(collection_name.strip())
        result   = pipeline.run(question)
    except RuntimeError as e:
        return str(e), "", "", ""

    answer       = result["answer"]
    scoring      = result.get("scoring", {})
    attempts     = result.get("attempts", 1)
    latency      = result.get("latency_ms", 0)
    reformulated = result.get("reformulated_query", "")
    quality      = scoring.get("quality", "unknown")
    mean_score   = scoring.get("mean", 0.0)
    scores       = scoring.get("scores", [])
    chunks       = result["chunks"]

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

    sources_md = "### Sources used\n\n"
    for i, chunk in enumerate(chunks, 1):
        relevance   = scores[i-1] if i-1 < len(scores) else "—"
        sources_md += f"**{i}.** `{chunk['id']}` — relevance {relevance}/5\n\n"
        sources_md += f"{chunk['text'][:250]}...\n\n"
        if i < len(chunks):
            sources_md += "---\n\n"

    stats_md = f"""### Stats

| Metric | Value |
|---|---|
| Latency | {latency:.0f}ms |
| Attempts | {attempts} |
| Collection | {collection_name} |
| Chunks scored | {len(chunks)} |
| Threshold | {scoring.get('threshold', 2.0)} |
| Chunk scores | {scores} |
"""

    return answer, how_md, sources_md, stats_md


# ── Upload tab functions ──────────────────────────────────────────────────────

def upload_document(file_path: str, collection_name: str):
    print(f"DEBUG: upload called | file_path={file_path} | collection={collection_name}")

    if file_path is None or file_path == "":
        return "Please upload a file first.", ""

    if not collection_name or collection_name.strip() == "":
        return "Please enter a collection name.", ""

    collection_name = collection_name.strip().lower().replace(" ", "_")

    try:
        file_result = process_file(file_path)
        text        = file_result["text"]
        filename    = file_result["filename"]
        chars       = file_result["chars"]

        indexer = Indexer(collection_name=collection_name)
        result  = indexer.index_text(text, filename=filename)

        status = f"""✅ **{filename}** indexed successfully

| | |
|---|---|
| Collection | `{collection_name}` |
| Chunks created | {result['chunks']} |
| Characters processed | {chars:,} |
| Document ID | `{result['doc_id']}` |

You can now ask questions in the **Ask** tab using collection `{collection_name}`.
"""
        doc_list = list_documents(collection_name)
        return status, doc_list

    except ValueError as e:
        return f"❌ Error: {str(e)}", ""
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return f"❌ Upload failed: {str(e)}", ""


def list_documents(collection_name: str) -> str:
    if not collection_name or collection_name.strip() == "":
        return ""

    collection_name = collection_name.strip().lower().replace(" ", "_")

    try:
        indexer = Indexer(collection_name=collection_name)
        docs    = indexer.list_documents()

        if not docs:
            return f"No documents indexed in collection `{collection_name}` yet."

        result = f"### Documents in `{collection_name}`\n\n"
        for doc in docs:
            result += f"- **{doc['filename']}** — {doc['chunks']} chunks (`{doc['doc_id']}`)\n"

        return result

    except Exception as e:
        return f"Error listing documents: {str(e)}"


def delete_document(doc_id: str, collection_name: str) -> str:
    if not doc_id.strip() or not collection_name.strip():
        return "Please provide both document ID and collection name."

    collection_name = collection_name.strip().lower().replace(" ", "_")

    try:
        indexer = Indexer(collection_name=collection_name)
        deleted = indexer.delete_document(doc_id.strip())

        if deleted == 0:
            return f"❌ No document found with ID `{doc_id}`"

        return f"✅ Deleted {deleted} chunks for document `{doc_id}`"

    except Exception as e:
        return f"❌ Delete failed: {str(e)}"


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Raleway:wght@400;600;700&display=swap');

* { font-family: 'Raleway', sans-serif !important; }

body, .gradio-container {
    background-color: #2d3250 !important;
    color: #ffffff !important;
}

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

.input-area textarea {
    background: #424769 !important;
    color: #ffffff !important;
    font-size: 1rem !important;
}

.ask-btn {
    background: #f9b17a !important;
    color: #2d3250 !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    border: none !important;
}

.answer-panel {
    background: #424769 !important;
    border-radius: 12px !important;
    border-left: 4px solid #f9b17a !important;
    padding: 20px 24px !important;
    color: #ffffff !important;
    min-height: 80px !important;
}

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
}

footer { display: none !important; }
"""

# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Self-Correcting RAG") as demo:

    gr.HTML(f"""<style>{CSS}</style>
    <div class="app-header">
        <h1>Self-Correcting RAG</h1>
        <p>
            A retrieval system that knows when it's wrong — and fixes itself.<br>
            Upload your own documents or search the built-in dataset.
        </p>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Ask ────────────────────────────────────────────────────────
        with gr.Tab("Ask"):

            with gr.Row():
                with gr.Column(scale=5):
                    question_box = gr.Textbox(
                        label="Your question",
                        placeholder="Ask anything about your documents...",
                        lines=2,
                        elem_classes=["input-area"],
                    )
                with gr.Column(scale=2):
                    collection_selector = gr.Textbox(
                        label="Collection",
                        placeholder="sc_rag_docs (default) or your collection name",
                        value="sc_rag_docs",
                    )
                with gr.Column(scale=1, min_width=100):
                    ask_btn = gr.Button("Ask →", variant="primary", elem_classes=["ask-btn"])

            answer_box = gr.Markdown(
                value="*Your answer will appear here...*",
                elem_classes=["answer-panel"],
            )

            gr.HTML('<hr style="border-color: #424769; margin: 16px 0;">')

            with gr.Row():
                with gr.Column(scale=2):
                    how_box = gr.Markdown(
                        value="### What happened\n\n*Ask a question to see the self-correction trace.*",
                        elem_classes=["info-panel"],
                    )
                with gr.Column(scale=3):
                    sources_box = gr.Markdown(
                        value="### Sources used\n\n*Retrieved chunks will appear here.*",
                        elem_classes=["info-panel"],
                    )
                with gr.Column(scale=1):
                    stats_box = gr.Markdown(
                        value="### Stats\n\n*Metrics will appear here.*",
                        elem_classes=["info-panel"],
                    )

            gr.Examples(
                examples=[
                    ["What is photosynthesis?",                 "sc_rag_docs"],
                    ["Who was the first US president?",         "sc_rag_docs"],
                    ["Who sang Go Rest High on the Mountain?",  "sc_rag_docs"],
                    ["What causes thunder?",                    "sc_rag_docs"],
                ],
                inputs=[question_box, collection_selector],
                label="Try these examples",
            )

            ask_btn.click(
                fn=ask_question,
                inputs=[question_box, collection_selector],
                outputs=[answer_box, how_box, sources_box, stats_box],
            )

            question_box.submit(
                fn=ask_question,
                inputs=[question_box, collection_selector],
                outputs=[answer_box, how_box, sources_box, stats_box],
            )

        # ── Tab 2: Upload Documents ───────────────────────────────────────────
        with gr.Tab("Upload Documents"):

            gr.Markdown("""
### Upload your own documents

Upload a PDF, Word (.docx), or TXT file to create your own searchable collection.
After uploading, go to the **Ask** tab and enter your collection name to search it.
""")

            with gr.Row():
                with gr.Column(scale=3):
                    file_upload = gr.File(
                        label="Upload document (PDF, DOCX, TXT)",
                        file_types=[".pdf", ".docx", ".doc", ".txt"],
                        type="filepath",
                    )
                with gr.Column(scale=2):
                    upload_collection = gr.Textbox(
                        label="Collection name",
                        placeholder="e.g. my_documents",
                        info="Give your collection a name. Use letters, numbers, underscores only.",
                    )
                    upload_btn = gr.Button("Upload & Index", variant="primary")

            upload_status = gr.Markdown(value="")
            doc_list_box  = gr.Markdown(value="")

            gr.HTML('<hr style="border-color: #424769; margin: 16px 0;">')

            gr.Markdown("### Manage documents")

            with gr.Row():
                with gr.Column(scale=3):
                    list_collection = gr.Textbox(
                        label="Collection name",
                        placeholder="Enter collection name to see its documents",
                    )
                with gr.Column(scale=1):
                    list_btn = gr.Button("List Documents")

            list_output = gr.Markdown(value="")

            with gr.Row():
                with gr.Column(scale=2):
                    delete_doc_id     = gr.Textbox(label="Document ID to delete", placeholder="e.g. contract_abc12345")
                with gr.Column(scale=2):
                    delete_collection = gr.Textbox(label="Collection name", placeholder="my_documents")
                with gr.Column(scale=1):
                    delete_btn = gr.Button("Delete Document", variant="stop")

            delete_output = gr.Markdown(value="")

            upload_btn.click(
                fn=upload_document,
                inputs=[file_upload, upload_collection],
                outputs=[upload_status, doc_list_box],
            )

            list_btn.click(
                fn=list_documents,
                inputs=[list_collection],
                outputs=[list_output],
            )

            delete_btn.click(
                fn=delete_document,
                inputs=[delete_doc_id, delete_collection],
                outputs=[delete_output],
            )

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
    )