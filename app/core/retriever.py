import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from app.utils.config import config
from app.utils.logger import logger


class Retriever:

    def __init__(self, reset_db: bool = False):
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)

        self.client = chromadb.PersistentClient(
            path=config.CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )

        if reset_db:
            try:
                self.client.delete_collection(config.CHROMA_COLLECTION_NAME)
                logger.info("Existing collection deleted")
            except Exception:
                pass

        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Collection ready — {self.collection.count()} chunks indexed")

    def _chunk_text(self, text: str) -> list:
        words = text.split()
        chunks = []
        step = config.CHUNK_SIZE - config.CHUNK_OVERLAP

        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + config.CHUNK_SIZE])
            if len(chunk.strip()) > 20:
                chunks.append(chunk)

        return chunks

    def index_documents(self, documents: list, batch_size: int = 64):
        if not documents:
            logger.warning("No documents to index.")
            return

        logger.info(f"Indexing {len(documents)} documents...")
        all_ids, all_texts, all_metadatas = [], [], []

        for doc in tqdm(documents, desc="Chunking"):
            chunks = self._chunk_text(doc["text"])
            for i, chunk in enumerate(chunks):
                all_ids.append(f"{doc['id']}_chunk_{i}")
                all_texts.append(chunk)
                all_metadatas.append({
                    "doc_id": doc["id"],
                    "chunk_index": i,
                    "title": doc.get("title", ""),
                })

        logger.info(f"Embedding {len(all_texts)} chunks...")
        all_embeddings = self.embedder.encode(
            all_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).tolist()

        BATCH = 512
        for start in range(0, len(all_ids), BATCH):
            end = start + BATCH
            self.collection.upsert(
                ids=all_ids[start:end],
                documents=all_texts[start:end],
                embeddings=all_embeddings[start:end],
                metadatas=all_metadatas[start:end],
            )

        logger.info(f"Done. Total chunks in DB: {self.collection.count()}")

    def retrieve(self, query: str, k: int = None) -> list:
        k = k or config.TOP_K

        if self.collection.count() == 0:
            raise RuntimeError("No documents indexed! Run data/download_beir.py first.")

        query_embedding = self.embedder.encode(
            [query],
            normalize_embeddings=True,
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for i in range(len(results["ids"][0])):
            score = round(1 - results["distances"][0][i], 4)
            chunks.append({
                "id":     results["ids"][0][i],
                "text":   results["documents"][0][i],
                "score":  score,
                "doc_id": results["metadatas"][0][i].get("doc_id", ""),
                "title":  results["metadatas"][0][i].get("title", ""),
            })

        return chunks

    def count(self) -> int:
        return self.collection.count()
    

'''__init__ — runs when you create Retriever(). Loads the embedding model and opens the database. Like hiring a librarian and opening the library doors.
_chunk_text — takes one long document and cuts it into 256-word pieces with 32-word overlaps. Why overlap? So a sentence that falls at the boundary of two chunks still appears fully in one of them. Like cutting a book into pages but letting each page share 2 lines with the next.
index_documents — the setup step. Takes all documents → chunks them → converts to embeddings → stores in ChromaDB. Run once before the project can answer anything.
retrieve — the live search. Takes a question → converts to embedding → asks ChromaDB "what chunks are closest to this?" → returns top 5 with relevance scores.'''