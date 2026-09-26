import time
from typing import TypedDict

from langgraph.graph import StateGraph, END

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

MAX_ATTEMPTS = 2


# State that flows through every node in the graph.
# Each node receives the full state and returns only the fields it updates.
class RAGState(TypedDict):
    query:              str    # original user question, never changes
    current_query:      str    # may be rewritten after reformulation
    chunks:             list   # retrieved chunks from ChromaDB
    scores:             list   # per-chunk scores from the scorer
    mean_score:         float  # average of scores
    quality:            str    # "good" or "poor"
    reformulations:     list   # rewrites tried so far
    attempts:           int    # how many retrieval attempts made
    answer:             str    # final answer
    sources:            list   # chunk IDs used in the answer
    latency_ms:         float  # total time taken
    reformulated_query: str    # best rewrite used (for logging/UI)


def make_retrieve_node(retriever: Retriever):
    def retrieve_node(state: RAGState) -> dict:
        logger.info(f"[retrieve] '{state['current_query'][:60]}'")
        chunks = retriever.retrieve(state["current_query"], k=config.TOP_K)
        return {"chunks": chunks}
    return retrieve_node


def make_score_node(scorer: Scorer):
    def score_node(state: RAGState) -> dict:
        result = scorer.score(state["current_query"], state["chunks"])
        logger.info(
            f"[score] mean={result['mean']} | quality={result['quality']} | scores={result['scores']}"
        )
        return {
            "scores":     result["scores"],
            "mean_score": result["mean"],
            "quality":    result["quality"],
        }
    return score_node


def decide_node(state: RAGState) -> str:
    # Router — returns the name of the next node
    if state["quality"] == "good":
        logger.info("[decide] good retrieval → generate")
        return "generate"

    if state["attempts"] >= MAX_ATTEMPTS:
        logger.info(f"[decide] max attempts reached → fallback")
        return "fallback"

    logger.info(f"[decide] poor retrieval, attempt {state['attempts']} → reformulate")
    return "reformulate"


def make_reformulate_node(retriever: Retriever, scorer: Scorer, reformulator: Reformulator):
    def reformulate_node(state: RAGState) -> dict:
        logger.info(f"[reformulate] rewriting: '{state['current_query'][:60]}'")
        rewrites = reformulator.reformulate(state["current_query"])

        if not rewrites:
            logger.warning("[reformulate] no rewrites generated, keeping original")
            return {
                "attempts":       state["attempts"] + 1,
                "reformulations": state.get("reformulations", []),
            }

        # Try each rewrite, keep the one that scores highest
        best_query  = None
        best_score  = state["mean_score"]
        best_chunks = state["chunks"]

        for rewrite in rewrites:
            chunks  = retriever.retrieve(rewrite, k=config.TOP_K)
            scoring = scorer.score(rewrite, chunks)
            logger.info(f"[reformulate] '{rewrite[:50]}' → score={scoring['mean']}")

            if scoring["mean"] > best_score:
                best_score  = scoring["mean"]
                best_query  = rewrite
                best_chunks = chunks

        if best_query is None:
            logger.info("[reformulate] no rewrite beat original, keeping original chunks")
            best_query  = state["current_query"]
            best_chunks = state["chunks"]

        logger.info(f"[reformulate] best: '{best_query[:60]}' score={best_score}")

        return {
            "current_query":      best_query,
            "chunks":             best_chunks,
            "reformulations":     state.get("reformulations", []) + rewrites,
            "attempts":           state["attempts"] + 1,
            "reformulated_query": best_query,
        }
    return reformulate_node


def make_generate_node(generator: Generator):
    def generate_node(state: RAGState) -> dict:
        logger.info("[generate] producing answer...")
        result = generator.generate(state["current_query"], state["chunks"])
        return {
            "answer":  result["answer"],
            "sources": result["sources"],
        }
    return generate_node


def fallback_node(state: RAGState) -> dict:
    logger.info("[fallback] returning insufficient context message")
    return {
        "answer":  INSUFFICIENT_CONTEXT_MSG,
        "sources": [],
    }


class BasicRAGPipeline:

    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or config.CHROMA_COLLECTION_NAME
        self.retriever    = Retriever(collection_name=self.collection_name)
        self.generator    = Generator()
        self.scorer       = Scorer()
        self.reformulator = Reformulator()
        self.graph        = self._build_graph()
        logger.info(f"Pipeline ready | collection={self.collection_name}")

    def _build_graph(self):
        graph = StateGraph(RAGState)

        graph.add_node("retrieve",    make_retrieve_node(self.retriever))
        graph.add_node("score",       make_score_node(self.scorer))
        graph.add_node("reformulate", make_reformulate_node(self.retriever, self.scorer, self.reformulator))
        graph.add_node("generate",    make_generate_node(self.generator))
        graph.add_node("fallback",    fallback_node)

        graph.add_edge("retrieve", "score")
        graph.add_conditional_edges("score", decide_node, {
            "generate":    "generate",
            "reformulate": "reformulate",
            "fallback":    "fallback",
        })
        graph.add_edge("reformulate", "retrieve")
        graph.add_edge("generate",    END)
        graph.add_edge("fallback",    END)

        graph.set_entry_point("retrieve")
        return graph.compile()

    def run(self, query: str) -> dict:
        t0 = time.time()

        initial_state: RAGState = {
            "query":              query,
            "current_query":      query,
            "chunks":             [],
            "scores":             [],
            "mean_score":         0.0,
            "quality":            "poor",
            "reformulations":     [],
            "attempts":           1,
            "answer":             "",
            "sources":            [],
            "latency_ms":         0.0,
            "reformulated_query": "",
        }

        final_state   = self.graph.invoke(initial_state)
        total_latency = (time.time() - t0) * 1000

        log_query(
            query=query,
            retrieved_chunks=final_state["chunks"],
            answer=final_state["answer"],
            latency_ms=total_latency,
            attempts=final_state["attempts"],
            quality_score=final_state["mean_score"],
            reformulated_query=final_state.get("reformulated_query", ""),
        )

        return {
            "query":              query,
            "answer":             final_state["answer"],
            "sources":            final_state["sources"],
            "chunks":             final_state["chunks"],
            "latency_ms":         round(total_latency, 2),
            "attempts":           final_state["attempts"],
            "reformulated_query": final_state.get("reformulated_query", ""),
            "scoring": {
                "scores":    final_state["scores"],
                "mean":      final_state["mean_score"],
                "quality":   final_state["quality"],
                "threshold": self.scorer.threshold,
                "tokens":    0,
            },
        }