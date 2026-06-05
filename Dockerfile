FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --only-binary numpy "numpy<2" \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir gradio

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    EMBEDDING_MODEL=all-MiniLM-L6-v2 \
    CHROMA_DB_PATH=/app/chroma_db \
    CHROMA_COLLECTION_NAME=sc_rag_docs \
    LLM_MODEL=llama-3.1-8b-instant \
    TOP_K=5 \
    CHUNK_SIZE=256 \
    CHUNK_OVERLAP=32 \
    QUALITY_THRESHOLD=3.0 \
    MAX_RETRIES=2

EXPOSE 7860

CMD ["python", "ui/gradio_app.py"]