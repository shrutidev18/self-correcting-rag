import uuid
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from app.utils.config import config
from app.utils.logger import logger


class Indexer:
    """
    Indexes user-uploaded documents into a private ChromaDB collection.

    Each user gets their own collection — completely isolated from the
    default NQ dataset and from other users' documents.

    Usage:
        indexer = Indexer(collection_name="user_alice")
        result  = indexer.index_text(text, filename="contract.pdf")
        # result = {
        #     "collection":  "user_alice",
        #     "doc_id":      "abc123",
        #     "filename":    "contract.pdf",
        #     "chunks":      42,
        #     "chars":       15000,
        # }
    """

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

        logger.info(
            f"Indexer ready | collection={collection_name} "
            f"| chunks already indexed={self.collection.count()}"
        )

        # Load embedding model
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)

    def _chunk_text(self, text: str) -> list:
        """Split text into overlapping chunks."""
        words  = text.split()
        chunks = []
        step   = config.CHUNK_SIZE - config.CHUNK_OVERLAP

        for i in range(0, len(words), step):
            chunk = " ".join(words[i: i + config.CHUNK_SIZE])
            if len(chunk.strip()) > 20:
                chunks.append(chunk)

        return chunks

    def index_text(self, text: str, filename: str) -> dict:
        """
        Chunk, embed, and store text in this collection.

        Args:
            text:     Plain text extracted from the document.
            filename: Original filename — stored as metadata.

        Returns:
            Summary dict with doc_id, chunk count, and collection name.
        """
        if not text.strip():
            raise ValueError("Cannot index empty text.")

        # Generate a unique doc ID based on filename
        doc_id = str(uuid.uuid4())[:8]
        safe_name = Path(filename).stem[:20].replace(" ", "_")
        full_doc_id = f"{safe_name}_{doc_id}"

        logger.info(f"Indexing '{filename}' as doc_id='{full_doc_id}'")

        # Chunk
        chunks = self._chunk_text(text)
        logger.info(f"Created {len(chunks)} chunks from {len(text)} chars")

        if not chunks:
            raise ValueError(f"No chunks created from {filename} — file may be too short.")

        # Prepare IDs, texts, metadata
        all_ids       = []
        all_texts     = []
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

        # Embed
        logger.info(f"Embedding {len(all_texts)} chunks...")
        all_embeddings = self.embedder.encode(
            all_texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).tolist()

        # Store in batches
        BATCH = 512
        for start in range(0, len(all_ids), BATCH):
            end = start + BATCH
            self.collection.upsert(
                ids=all_ids[start:end],
                documents=all_texts[start:end],
                embeddings=all_embeddings[start:end],
                metadatas=all_metadatas[start:end],
            )

        total = self.collection.count()
        logger.info(
            f"Indexed '{filename}' → {len(chunks)} chunks | "
            f"Total in collection: {total}"
        )

        return {
            "collection": self.collection_name,
            "doc_id":     full_doc_id,
            "filename":   filename,
            "chunks":     len(chunks),
            "chars":      len(text),
        }

    def list_documents(self) -> list:
        """
        List all unique documents indexed in this collection.

        Returns:
            List of dicts with filename, doc_id, and chunk count.
        """
        if self.collection.count() == 0:
            return []

        # Get all metadatas
        results = self.collection.get(include=["metadatas"])

        # Group by doc_id
        docs = {}
        for meta in results["metadatas"]:
            doc_id   = meta.get("doc_id", "unknown")
            filename = meta.get("filename", "unknown")
            if doc_id not in docs:
                docs[doc_id] = {"doc_id": doc_id, "filename": filename, "chunks": 0}
            docs[doc_id]["chunks"] += 1

        return list(docs.values())

    def delete_document(self, doc_id: str) -> int:
        """
        Delete all chunks belonging to a document.

        Returns:
            Number of chunks deleted.
        """
        results = self.collection.get(
            where={"doc_id": doc_id},
            include=["metadatas"],
        )

        ids_to_delete = results["ids"]

        if not ids_to_delete:
            logger.warning(f"No chunks found for doc_id={doc_id}")
            return 0

        self.collection.delete(ids=ids_to_delete)
        logger.info(f"Deleted {len(ids_to_delete)} chunks for doc_id={doc_id}")
        return len(ids_to_delete)

    def count(self) -> int:
        return self.collection.count()