import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
from chromadb.config import Settings
import gradio as gr
from app.core.pipeline import BasicRAGPipeline
from app.core.document_processor import process_file
from app.core.indexer import Indexer
from app.utils.config import config
from app.utils.logger import logger

_pipelines = {}


def _load_existing_collections() -> list:
    try:
        client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False),
        )
        return [c.name for c in client.list_collections()]
    except Exception:
        return ["sc_rag_docs"]


_known_collections = _load_existing_collections()


def get_pipeline(collection_name: str) -> BasicRAGPipeline:
    if collection_name not in _pipelines:
        logger.info(f"loading pipeline for collection: {collection_name}")
        _pipelines[collection_name] = BasicRAGPipeline(collection_name=collection_name)
    return _pipelines[collection_name]


def ask_question(question: str, collection_name: str):
    if not question.strip():
        return "Please enter a question.", "", "", ""

    if not collection_name or collection_name.strip() == "":
        collection_name = "sc_rag_docs"

    try:
        pipeline = get_pipeline(collection_name.strip())
        result = pipeline.run(question)
    except RuntimeError as e:
        return str(e), "", "", ""

    answer = result["answer"]
    scoring = result.get("scoring", {})
    attempts = result.get("attempts", 1)
    latency = result.get("latency_ms", 0)
    reformulated = result.get("reformulated_query", "")
    quality = scoring.get("quality", "unknown")
    mean_score = scoring.get("mean", 0.0)
    scores = scoring.get("scores", [])
    chunks = result["chunks"]

    if attempts == 1 and quality == "good":
        how_md = f"""#### 💡 What Happened

The system retrieved relevant documents on the **first try** and generated an answer directly.

| | |
|---|---|
| Retrieval quality | ✅ Good ({mean_score:.1f} / 5.0) |
| Attempts needed | 1 |
| Time taken | {latency:.0f}ms |
"""
    elif attempts > 1 and reformulated:
        how_md = f"""#### 💡 What Happened

Initial retrieval wasn't useful enough — the system **automatically rewrote the question** and tried again.

**Original question**
> {question}

**Rewritten as**
> {reformulated}

| | |
|---|---|
| After rewrite | 🔄 Better ({mean_score:.1f} / 5.0) |
| Attempts needed | {attempts} |
| Time taken | {latency:.0f}ms |
"""
    else:
        how_md = f"""#### 💡 What Happened

The system tried rewriting the question but couldn't find enough relevant context.

| | |
|---|---|
| Attempts | {attempts} |
| Time taken | {latency:.0f}ms |
| Status | ⚠️ Insufficient context |
"""

    sources_md = "#### 📄 Sources Used\n\n"
    for i, chunk in enumerate(chunks, 1):
        relevance = scores[i-1] if i-1 < len(scores) else "—"
        sources_md += f"**{i}.** `{chunk['id']}` — relevance **{relevance}/5**\n\n"
        sources_md += f"{chunk['text'][:250]}...\n\n"
        if i < len(chunks):
            sources_md += "---\n\n"

    stats_md = f"""#### 📊 Stats

| Metric | Value |
|---|---|
| Latency | {latency:.0f}ms |
| Attempts | {attempts} |
| Collection | `{collection_name}` |
| Chunks scored | {len(chunks)} |
| Threshold | {scoring.get('threshold', 2.0)} |
| Chunk scores | {scores} |
"""

    return answer, how_md, sources_md, stats_md


def upload_document(file_path: str, collection_name: str):
    if file_path is None or file_path == "":
        return "Please upload a file first.", ""

    if not collection_name or collection_name.strip() == "":
        return "Please enter a collection name.", ""

    collection_name = collection_name.strip().lower().replace(" ", "_")

    try:
        file_result = process_file(file_path)
        text = file_result["text"]
        filename = file_result["filename"]
        chars = file_result["chars"]

        indexer = Indexer(collection_name=collection_name)
        result = indexer.index_text(text, filename=filename)

        if collection_name not in _known_collections:
            _known_collections.append(collection_name)

        status = f"""✅ **{filename}** indexed successfully

| | |
|---|---|
| Collection | `{collection_name}` |
| Chunks created | {result['chunks']} |
| Characters processed | {chars:,} |
| Document ID | `{result['doc_id']}` |

Go to the **Ask** tab and select `{collection_name}` from the collection dropdown.
"""
        doc_list = list_documents(collection_name)
        return status, doc_list

    except ValueError as e:
        return f"❌ Error: {str(e)}", ""
    except Exception as e:
        logger.error(f"upload failed: {e}")
        return f"❌ Upload failed: {str(e)}", ""


def list_documents(collection_name: str) -> str:
    if not collection_name or collection_name.strip() == "":
        return ""

    collection_name = collection_name.strip().lower().replace(" ", "_")

    try:
        indexer = Indexer(collection_name=collection_name)
        docs = indexer.list_documents()

        if not docs:
            return f"no documents indexed in `{collection_name}` yet."

        result = f"#### 📁 Documents in `{collection_name}`\n\n"
        for doc in docs:
            result += f"- **{doc['filename']}** — {doc['chunks']} chunks · ID: `{doc['doc_id']}`\n"

        return result

    except Exception as e:
        return f"error listing documents: {str(e)}"


def delete_document(doc_id: str, collection_name: str) -> str:
    if not doc_id.strip() or not collection_name.strip():
        return "please provide both document ID and collection name."

    collection_name = collection_name.strip().lower().replace(" ", "_")

    try:
        indexer = Indexer(collection_name=collection_name)
        deleted = indexer.delete_document(doc_id.strip())

        if deleted == 0:
            return f"❌ no document found with ID `{doc_id}`"

        return f"✅ deleted {deleted} chunks for document `{doc_id}`"

    except Exception as e:
        return f"❌ delete failed: {str(e)}"


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
    padding: 28px 36px;
    margin-bottom: 16px;
    border: 1px solid #424769;
    display: flex;
    align-items: center;
    gap: 20px;
}
.app-header-text h1 {
    font-size: 1.8rem;
    font-weight: 700;
    color: #f9b17a !important;
    margin: 0 0 6px 0;
}
.app-header-text p {
    color: #b0b8d4 !important;
    font-size: 0.95rem;
    margin: 0;
    line-height: 1.6;
}

.question-card {
    background: #363b5e !important;
    border-radius: 14px !important;
    border: 1px solid #424769 !important;
    padding: 20px !important;
    margin-bottom: 12px !important;
}

.input-area textarea {
    background: #2d3250 !important;
    color: #ffffff !important;
    font-size: 1rem !important;
    border: 1px solid #424769 !important;
    border-radius: 10px !important;
    padding: 12px !important;
}

.input-area textarea::placeholder { color: #676f9d !important; }

.input-area textarea:focus {
    border-color: #f9b17a !important;
    outline: none !important;
}

.collection-card {
    background: #363b5e !important;
    border-radius: 14px !important;
    border: 1px solid #424769 !important;
    padding: 20px !important;
}

select {
    background: #2d3250 !important;
    color: #ffffff !important;
    border: 1px solid #424769 !important;
    border-radius: 10px !important;
    padding: 10px !important;
    width: 100% !important;
}

.ask-btn button {
    background: #f9b17a !important;
    color: #2d3250 !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border-radius: 10px !important;
    border: none !important;
    width: 100% !important;
    padding: 14px !important;
    margin-top: 12px !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
}
.ask-btn button:hover { opacity: 0.88 !important; }

.answer-card {
    background: #363b5e !important;
    border-radius: 14px !important;
    border-left: 4px solid #f9b17a !important;
    padding: 24px !important;
    margin: 12px 0 !important;
    min-height: 90px !important;
}

.answer-card p, .answer-card li {
    color: #e8eaf6 !important;
    font-size: 1.02rem !important;
    line-height: 1.75 !important;
}

.info-card {
    background: #2d3250 !important;
    border-radius: 14px !important;
    border: 1px solid #424769 !important;
    padding: 20px !important;
    height: 100% !important;
}

.info-card h4 {
    color: #f9b17a !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    margin-bottom: 14px !important;
    padding-bottom: 10px !important;
    border-bottom: 1px solid #424769 !important;
}

.info-card table { width: 100% !important; }

.info-card td {
    color: #b0b8d4 !important;
    font-size: 0.9rem !important;
    padding: 5px 4px !important;
}

.info-card td:first-child {
    color: #676f9d !important;
    width: 45% !important;
}

.info-card p, .info-card li {
    color: #b0b8d4 !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
}

.info-card blockquote {
    border-left: 3px solid #f9b17a !important;
    padding-left: 12px !important;
    color: #b0b8d4 !important;
    font-style: italic !important;
    margin: 8px 0 !important;
}

.info-card code {
    background: #424769 !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-size: 0.85rem !important;
    color: #f9b17a !important;
}

.upload-card {
    background: #363b5e !important;
    border-radius: 14px !important;
    border: 1px solid #424769 !important;
    padding: 24px !important;
    margin-bottom: 16px !important;
}

.upload-card h3 {
    color: #f9b17a !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    margin-bottom: 4px !important;
}

.upload-card p { color: #b0b8d4 !important; font-size: 0.9rem !important; }

.divider {
    border: none !important;
    border-top: 1px solid #424769 !important;
    margin: 20px 0 !important;
}

.tab-nav button {
    color: #b0b8d4 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 20px !important;
}

.tab-nav button.selected {
    color: #f9b17a !important;
    border-bottom: 2px solid #f9b17a !important;
}

input[type="text"],
input[type="search"],
textarea,
.gr-text-input input,
.gr-textbox textarea,
[data-testid="textbox"] textarea,
[data-testid="textbox"] input {
    background: #363b5e !important;
    color: #ffffff !important;
    border: 1px solid #424769 !important;
    border-radius: 10px !important;
}

.gr-dropdown,
[data-testid="dropdown"],
.wrap select,
select,
[role="listbox"],
[role="combobox"] {
    background: #363b5e !important;
    color: #ffffff !important;
    border: 1px solid #424769 !important;
    border-radius: 10px !important;
}

.gr-file,
[data-testid="file"],
.file-preview,
.upload-container,
.svelte-file-upload,
.gr-file-drop-zone {
    background: #363b5e !important;
    border: 2px dashed #424769 !important;
    border-radius: 14px !important;
    color: #b0b8d4 !important;
}

.gr-box, .gr-form, .gr-panel, .gr-block, .wrap, .container {
    background: transparent !important;
}

label span, .gr-label, [data-testid="label"] {
    color: #b0b8d4 !important;
    font-size: 0.9rem !important;
}

button.secondary, button[variant="secondary"], .gr-button-secondary {
    background: #424769 !important;
    color: #f9b17a !important;
    border: 1px solid #f9b17a !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

button.secondary:hover, button[variant="secondary"]:hover {
    background: #4e5580 !important;
}

button.stop, button[variant="stop"], .gr-button-stop {
    background: #363b5e !important;
    color: #ff6b6b !important;
    border: 1px solid #ff6b6b !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

button.stop:hover, button[variant="stop"]:hover {
    background: #ff6b6b !important;
    color: #ffffff !important;
}

button.primary {
    background: #f9b17a !important;
    color: #2d3250 !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    border: none !important;
}

:root,
.gradio-container,
.gradio-container > .main,
.gradio-container > .main > .wrap,
#component-0,
.app {
    background-color: #2d3250 !important;
    background: #2d3250 !important;
}

.gradio-container .wrap,
.gradio-container .gap,
.gradio-container .form {
    background: transparent !important;
    border: none !important;
}

.examples-section { margin-top: 16px !important; }

footer { display: none !important; }
"""

with gr.Blocks(title="RETRACE : Self-Correcting RAG") as demo:

    gr.HTML(f"""<style>{CSS}</style>
    <div class="app-header">
        <div class="app-header-text">
            <h1>RETRACE : Self-Correcting RAG</h1>
            <p>A retrieval system that knows when it's wrong and fixes itself.<br>
            Upload your own documents or search the built-in dataset.</p>
        </div>
    </div>
    """)

    with gr.Tabs(elem_classes=["tab-nav"]):

        with gr.Tab("🔍  Ask"):

            with gr.Row():
                with gr.Column(scale=4):
                    gr.HTML('<div class="question-card">')
                    question_box = gr.Textbox(
                        label="Your question",
                        placeholder="Ask anything about your documents...",
                        lines=3,
                        elem_classes=["input-area"],
                    )
                    gr.HTML('</div>')

                with gr.Column(scale=2):
                    gr.HTML('<div class="collection-card">')
                    collection_selector = gr.Dropdown(
                        label="Collection",
                        choices=_known_collections,
                        value="sc_rag_docs",
                        allow_custom_value=True,
                        info="Select or type a collection name",
                    )
                    ask_btn = gr.Button(
                        "Ask →",
                        variant="primary",
                        elem_classes=["ask-btn"],
                    )
                    gr.HTML('</div>')

            gr.HTML('<div class="answer-card">')
            answer_box = gr.Markdown(
                value="*Your answer will appear here...*",
                elem_classes=["answer-card"],
            )
            gr.HTML('</div>')

            gr.HTML('<hr class="divider">')

            with gr.Row():
                with gr.Column(scale=2):
                    how_box = gr.Markdown(
                        value="#### 💡 What Happened\n\n*Ask a question to see the self-correction trace.*",
                        elem_classes=["info-card"],
                    )
                with gr.Column(scale=3):
                    sources_box = gr.Markdown(
                        value="#### 📄 Sources Used\n\n*Retrieved chunks will appear here.*",
                        elem_classes=["info-card"],
                    )
                with gr.Column(scale=1):
                    stats_box = gr.Markdown(
                        value="#### 📊 Stats\n\n*Metrics will appear here.*",
                        elem_classes=["info-card"],
                    )

            gr.HTML('<div class="examples-section">')
            gr.Examples(
                examples=[
                    ["What is photosynthesis?",                "sc_rag_docs"],
                    ["Who was the first US president?",        "sc_rag_docs"],
                    ["Who sang Go Rest High on the Mountain?", "sc_rag_docs"],
                    ["What causes thunder?",                   "sc_rag_docs"],
                ],
                inputs=[question_box, collection_selector],
                label="Try these examples",
            )
            gr.HTML('</div>')

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

        with gr.Tab("📁  Upload Documents"):

            gr.HTML("""
            <div class="upload-card">
                <h3>Upload your own documents</h3>
                <p>Upload a PDF, Word (.docx), or TXT file to create a private searchable collection.
                After uploading, select your collection name in the Ask tab.</p>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=3):
                    file_upload = gr.File(
                        label="Document (PDF, DOCX, TXT)",
                        file_types=[".pdf", ".docx", ".doc", ".txt"],
                        type="filepath",
                    )
                with gr.Column(scale=2):
                    upload_collection = gr.Textbox(
                        label="Collection name",
                        placeholder="e.g. my_documents",
                        info="Letters, numbers, underscores only.",
                    )
                    upload_btn = gr.Button("⬆️  Upload & Index", variant="primary")

            upload_status = gr.Markdown(value="", elem_classes=["info-card"])
            doc_list_box = gr.Markdown(value="", elem_classes=["info-card"])

            gr.HTML('<hr class="divider">')

            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("#### Manage Documents")
                    list_collection = gr.Textbox(
                        label="Collection name",
                        placeholder="Enter collection name",
                    )
                with gr.Column(scale=1, min_width=160):
                    gr.HTML("<br>")
                    list_btn = gr.Button("📋  List Documents")

            list_output = gr.Markdown(value="", elem_classes=["info-card"])

            gr.HTML('<hr class="divider">')

            with gr.Row():
                with gr.Column(scale=2):
                    delete_doc_id = gr.Textbox(
                        label="Document ID",
                        placeholder="e.g. contract_abc12345",
                    )
                with gr.Column(scale=2):
                    delete_collection = gr.Textbox(
                        label="Collection name",
                        placeholder="my_documents",
                    )
                with gr.Column(scale=1, min_width=160):
                    gr.HTML("<br>")
                    delete_btn = gr.Button("🗑️  Delete", variant="stop")

            delete_output = gr.Markdown(value="", elem_classes=["info-card"])

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