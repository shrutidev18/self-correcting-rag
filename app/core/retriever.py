import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from app.utils.config import config
from app.utils.logger import logger


class Retriever:

    def __init__(self, reset_db: bool = False, collection_name: str = None):
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)

        self.client = chromadb.PersistentClient(
            path=config.CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection_name = collection_name or config.CHROMA_COLLECTION_NAME

        if reset_db:
            try:
                self.client.delete_collection(self.collection_name)
                logger.info("existing collection deleted")
            except Exception:
                pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"collection '{self.collection_name}' ready — {self.collection.count()} chunks indexed")

    def _chunk_text(self, text: str) -> list[str]:
        words = text.split()
        chunks = []
        step = config.CHUNK_SIZE - config.CHUNK_OVERLAP

        for i in range(0, len(words), step):
            chunk = " ".join(words[i: i + config.CHUNK_SIZE])
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    def index_documents(self, documents: list[dict]) -> int:
        all_ids = []
        all_texts = []
        all_metadatas = []

        for doc in documents:
            chunks = self._chunk_text(doc["text"])
            for j, chunk in enumerate(chunks):
                all_ids.append(f"{doc['id']}_chunk_{j}")
                all_texts.append(chunk)
                all_metadatas.append({
                    "doc_id": doc["id"],
                    "title":  doc.get("title", ""),
                })

        if not all_ids:
            logger.warning("no chunks to index")
            return 0

        logger.info(f"embedding {len(all_ids)} chunks...")
        embeddings = self.embedder.encode(
            all_texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).tolist()

        # upsert in batches to avoid memory issues
        batch_size = 512
        for start in range(0, len(all_ids), batch_size):
            end = start + batch_size
            self.collection.upsert(
                ids=all_ids[start:end],
                documents=all_texts[start:end],
                embeddings=embeddings[start:end],
                metadatas=all_metadatas[start:end],
            )

        logger.info(f"indexed {len(all_ids)} chunks | total in collection: {self.collection.count()}")
        return len(all_ids)

    def retrieve(self, query: str, k: int = None) -> list[dict]:
        k = k or config.TOP_K

        if self.collection.count() == 0:
            raise RuntimeError(
                f"Collection '{self.collection_name}' is empty. "
                "Run the indexing script first."
            )

        query_embedding = self.embedder.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for i, doc in enumerate(results["documents"][0]):
            distance = results["distances"][0][i]
            meta = results["metadatas"][0][i]
            chunks.append({
                "text":   doc,
                "id":     results["ids"][0][i],
                "score":  round(1 - distance, 4),
                "doc_id": meta.get("doc_id", ""),
                "title":  meta.get("title", ""),
            })

        return chunks