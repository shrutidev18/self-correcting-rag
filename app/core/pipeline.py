import time

from app.core.retriever    import Retriever
from app.core.generator    import Generator
from app.core.scorer       import Scorer
from app.core.reformulator import Reformulator
from app.utils.config      import config
from app.utils.logger      import logger, log_query

INSUFFICIENT_CONTEXT_MSG = (
    "I was unable to find relevant information for this question "
    "even after reformulating the query. Please try rephrasing your question."
)


class BasicRAGPipeline:

    def __init__(self):
        self.retriever    = Retriever()
        self.generator    = Generator()
        self.scorer       = Scorer()
        self.reformulator = Reformulator()
        logger.info("BasicRAGPipeline ready")

    def run(self, query: str, k: int = None) -> dict:
        t0 = time.time()
        k  = k or config.TOP_K

        logger.info(f"Query: '{query[:60]}'")

        # ── Step 1: Retrieve ──────────────────────────────────────────────────
        chunks  = self.retriever.retrieve(query, k=k)

        # ── Step 2: Score retrieval quality ──────────────────────────────────
        scoring = self.scorer.score(query, chunks)
        logger.info(
            f"Retrieval quality: {scoring['quality']} "
            f"| mean={scoring['mean']} | scores={scoring['scores']}"
        )

        # ── Step 3: Self-correction if quality is poor ────────────────────────
        final_query        = query
        reformulated_query = None
        attempts           = 1

        if scoring["quality"] == "poor":
            logger.info("Poor retrieval detected — attempting reformulation...")
            rewrites = self.reformulator.reformulate(query)

            if rewrites:
                best_chunks  = chunks       # fallback to original
                best_score   = scoring["mean"]
                best_rewrite = None

                for rewrite in rewrites:
                    attempts += 1
                    r_chunks  = self.retriever.retrieve(rewrite, k=k)
                    r_scoring = self.scorer.score(rewrite, r_chunks)

                    logger.info(
                        f"Rewrite: '{rewrite[:50]}' "
                        f"| mean={r_scoring['mean']} | quality={r_scoring['quality']}"
                    )

                    if r_scoring["mean"] > best_score:
                        best_score   = r_scoring["mean"]
                        best_chunks  = r_chunks
                        best_rewrite = rewrite

                if best_rewrite:
                    logger.info(
                        f"Best rewrite: '{best_rewrite[:60]}' "
                        f"| score={best_score}"
                    )
                    final_query        = best_rewrite
                    reformulated_query = best_rewrite
                    chunks             = best_chunks

                # If even the best rewrite is still too poor → insufficient context
                if best_score < 2.5:
                    logger.info(
                        f"Best score {best_score} still below 2.5 "
                        f"— returning insufficient context"
                    )
                    total_latency = (time.time() - t0) * 1000
                    log_query(
                        query=query,
                        retrieved_chunks=chunks,
                        answer=INSUFFICIENT_CONTEXT_MSG,
                        latency_ms=total_latency,
                        attempts=attempts,
                        quality_score=best_score,
                        reformulated_query=reformulated_query,
                    )
                    return {
                        "query":               query,
                        "answer":              INSUFFICIENT_CONTEXT_MSG,
                        "sources":             [],
                        "chunks":              chunks,
                        "latency_ms":          round(total_latency, 2),
                        "attempts":            attempts,
                        "reformulated_query":  reformulated_query,
                        "scoring": {
                            "scores":    scoring["scores"],
                            "mean":      best_score,
                            "quality":   "poor",
                            "threshold": scoring["threshold"],
                            "tokens":    scoring["token_count"],
                        },
                    }

        # ── Step 4: Generate answer ───────────────────────────────────────────
        result        = self.generator.generate(final_query, chunks)
        total_latency = (time.time() - t0) * 1000

        log_query(
            query=query,
            retrieved_chunks=chunks,
            answer=result["answer"],
            latency_ms=total_latency,
            attempts=attempts,
            quality_score=scoring["mean"],
            reformulated_query=reformulated_query,
        )

        return {
            "query":              query,
            "answer":             result["answer"],
            "sources":            result["sources"],
            "chunks":             chunks,
            "latency_ms":         round(total_latency, 2),
            "attempts":           attempts,
            "reformulated_query": reformulated_query,
            "scoring": {
                "scores":    scoring["scores"],
                "mean":      scoring["mean"],
                "quality":   scoring["quality"],
                "threshold": scoring["threshold"],
                "tokens":    scoring["token_count"],
            },
        }