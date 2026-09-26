import sys
import os
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.core.pipeline import BasicRAGPipeline
from app.core.document_processor import process_file
from app.core.indexer import Indexer
from app.utils.logger import logger
from api.auth import verify_api_key


app = FastAPI(
    title="RETRACE — Self-Correcting RAG API",
    description="A RAG system that detects poor retrieval and automatically fixes itself.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# pipeline cache — one instance per collection
_pipelines = {}


def get_pipeline(collection_name: str) -> BasicRAGPipeline:
    if collection_name not in _pipelines:
        _pipelines[collection_name] = BasicRAGPipeline(collection_name=collection_name)
    return _pipelines[collection_name]


class AskRequest(BaseModel):
    question: str
    collection_name: str = "sc_rag_docs"


class AskResponse(BaseModel):
    question: str
    answer: str
    collection: str
    attempts: int
    reformulated_query: str
    latency_ms: float
    retrieval_quality: str
    mean_score: float
    sources: list


@app.get("/")
async def root():
    return {
        "name": "RETRACE — Self-Correcting RAG API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(verify_api_key)])
async def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty.")

    try:
        pipeline = get_pipeline(request.collection_name)
        result = pipeline.run(request.question)
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))

    scoring = result.get("scoring", {})

    return AskResponse(
        question=request.question,
        answer=result["answer"],
        collection=request.collection_name,
        attempts=result["attempts"],
        reformulated_query=result.get("reformulated_query", ""),
        latency_ms=result["latency_ms"],
        retrieval_quality=scoring.get("quality", "unknown"),
        mean_score=scoring.get("mean", 0.0),
        sources=result["sources"],
    )


@app.post("/upload", dependencies=[Depends(verify_api_key)])
async def upload(
    file: UploadFile = File(...),
    collection_name: str = Form(...),
):
    if not collection_name.strip():
        raise HTTPException(status_code=400, detail="collection name cannot be empty.")

    collection_name = collection_name.strip().lower().replace(" ", "_")

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        file_result = process_file(tmp_path)
        file_result["filename"] = file.filename

        indexer = Indexer(collection_name=collection_name)
        result = indexer.index_text(file_result["text"], filename=file.filename)

        return {
            "message":    f"{file.filename} indexed successfully.",
            "collection": collection_name,
            "doc_id":     result["doc_id"],
            "chunks":     result["chunks"],
            "chars":      file_result["chars"],
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"upload failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.get("/collections/{collection_name}/documents", dependencies=[Depends(verify_api_key)])
async def list_documents(collection_name: str):
    try:
        indexer = Indexer(collection_name=collection_name)
        docs = indexer.list_documents()
        return {
            "collection": collection_name,
            "documents":  docs,
            "total":      len(docs),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete(
    "/collections/{collection_name}/documents/{doc_id}",
    dependencies=[Depends(verify_api_key)],
)
async def delete_document(collection_name: str, doc_id: str):
    try:
        indexer = Indexer(collection_name=collection_name)
        deleted = indexer.delete_document(doc_id)

        if deleted == 0:
            raise HTTPException(
                status_code=404,
                detail=f"no document found with id {doc_id}",
            )

        return {
            "message":    f"deleted {deleted} chunks for document {doc_id}.",
            "collection": collection_name,
            "doc_id":     doc_id,
            "deleted":    deleted,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))