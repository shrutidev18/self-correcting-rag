import uuid
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from app.utils.config import config
from app.utils.logger import logger


class Indexer:

    def __init__(self, collection_name: str):
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(
            path=config.CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(f"indexer ready | collection={collection_name} | chunks={self.collection.count()}")

        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)

    def _chunk_text(self, text: str) -> list:
        words = text.split()
        chunks = []
        step = config.CHUNK_SIZE - config.CHUNK_OVERLAP

        for i in range(0, len(words), step):
            chunk = " ".join(words[i: i + config.CHUNK_SIZE])
            if len(chunk.strip()) > 20:
                chunks.append(chunk)

        return chunks

    def index_text(self, text: str, filename: str) -> dict:
        if not text.strip():
            raise ValueError("cannot index empty text.")

        doc_id = str(uuid.uuid4())[:8]
        safe_name = Path(filename).stem[:20].replace(" ", "_")
        full_doc_id = f"{safe_name}_{doc_id}"

        logger.info(f"indexing '{filename}' as doc_id='{full_doc_id}'")

        chunks = self._chunk_text(text)
        logger.info(f"created {len(chunks)} chunks from {len(text)} chars")

        if not chunks:
            raise ValueError(f"no chunks created from {filename} — file may be too short.")

        all_ids = []
        all_texts = []
        all_metadatas = []

        for i, chunk in enumerate(chunks):
            all_ids.append(f"{full_doc_id}_chunk_{i}")
            all_texts.append(chunk)
            all_metadatas.append({
                "doc_id":      full_doc_id,
                "filename":    filename,
                "chunk_index": i,
                "title":       Path(filename).stem,
            })

        logger.info(f"embedding {len(all_texts)} chunks...")
        all_embeddings = self.embedder.encode(
            all_texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).tolist()

        # upsert in batches
        batch = 512
        for start in range(0, len(all_ids), batch):
            end = start + batch
            self.collection.upsert(
                ids=all_ids[start:end],
                documents=all_texts[start:end],
                embeddings=all_embeddings[start:end],
                metadatas=all_metadatas[start:end],
            )

        total = self.collection.count()
        logger.info(f"indexed '{filename}' → {len(chunks)} chunks | total in collection: {total}")

        return {
            "collection": self.collection_name,
            "doc_id":     full_doc_id,
            "filename":   filename,
            "chunks":     len(chunks),
            "chars":      len(text),
        }

    def list_documents(self) -> list:
        if self.collection.count() == 0:
            return []

        results = self.collection.get(include=["metadatas"])

        docs = {}
        for meta in results["metadatas"]:
            doc_id = meta.get("doc_id", "unknown")
            filename = meta.get("filename", "unknown")
            if doc_id not in docs:
                docs[doc_id] = {"doc_id": doc_id, "filename": filename, "chunks": 0}
            docs[doc_id]["chunks"] += 1

        return list(docs.values())

    def delete_document(self, doc_id: str) -> int:
        results = self.collection.get(
            where={"doc_id": doc_id},
            include=["metadatas"],
        )

        ids_to_delete = results["ids"]

        if not ids_to_delete:
            logger.warning(f"no chunks found for doc_id={doc_id}")
            return 0

        self.collection.delete(ids=ids_to_delete)
        logger.info(f"deleted {len(ids_to_delete)} chunks for doc_id={doc_id}")
        return len(ids_to_delete)

    def count(self) -> int:
        return self.collection.count()