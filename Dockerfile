FROM python:3.11-slim-bullseye

WORKDIR /app

COPY requirements-app.txt .

RUN pip install --upgrade pip \
 && pip install --only-binary numpy "numpy<2" \
 && pip install --no-cache-dir -r requirements-app.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    EMBEDDING_MODEL=all-MiniLM-L6-v2 \
    CHROMA_DB_PATH=/app/chroma_db \
    CHROMA_COLLECTION_NAME=sc_rag_docs \
    LLM_MODEL=qwen/qwen3.8-27b \
    TOP_K=5 \
    CHUNK_SIZE=256 \
    CHUNK_OVERLAP=32 \
    QUALITY_THRESHOLD=1.5 \
    MAX_RETRIES=2

EXPOSE 7860

CMD ["python", "ui/gradio_app.py"]