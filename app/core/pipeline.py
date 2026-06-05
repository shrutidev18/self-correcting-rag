#This is the coordinator. Call the retriever, then call the generator, then return the result.
import time

from app.core.retriever import Retriever
from app.core.generator import Generator
from app.core.scorer    import Scorer
from app.utils.config   import config
from app.utils.logger   import logger, log_query


class BasicRAGPipeline:

    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()
        self.scorer    = Scorer()
        logger.info("BasicRAGPipeline ready")

    def run(self, query: str, k: int = None) -> dict:
        t0 = time.time()
        k  = k or config.TOP_K

        logger.info(f"Query: '{query[:60]}'")

        # Step 1 — Retrieve
        chunks = self.retriever.retrieve(query, k=k)

        # Step 2 — Score retrieval quality
        scoring = self.scorer.score(query, chunks)
        logger.info(
            f"Retrieval quality: {scoring['quality']} "
            f"| mean={scoring['mean']} | scores={scoring['scores']}"
        )

        # Step 3 — Generate answer
        result = self.generator.generate(query, chunks)

        total_latency = (time.time() - t0) * 1000

        log_query(
            query=query,
            retrieved_chunks=chunks,
            answer=result["answer"],
            latency_ms=total_latency,
            attempts=1,
            quality_score=scoring["mean"],
        )

        return {
            "query":      query,
            "answer":     result["answer"],
            "sources":    result["sources"],
            "chunks":     chunks,
            "latency_ms": round(total_latency, 2),
            "attempts":   1,
            # New scoring fields
            "scoring": {
                "scores":    scoring["scores"],
                "mean":      scoring["mean"],
                "quality":   scoring["quality"],
                "threshold": scoring["threshold"],
                "tokens":    scoring["token_count"],
            },
        }