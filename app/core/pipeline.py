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


# ── State Definition ──────────────────────────────────────────────────────────
# This is the shared state that flows through every node.
# Each node receives the full state and returns only the fields it updates.

class RAGState(TypedDict):
    query:              str           # original user question
    current_query:      str           # may change after reformulation
    chunks:             list          # retrieved chunks
    scores:             list          # per-chunk scores from scorer
    mean_score:         float         # average score
    quality:            str           # "good" or "poor"
    reformulations:     list          # list of rewritten queries tried
    attempts:           int           # number of retrieval attempts
    answer:             str           # final answer
    sources:            list          # source chunk IDs
    latency_ms:         float         # total latency
    reformulated_query: str           # best rewrite used (for logging)


# ── Node Functions ────────────────────────────────────────────────────────────
# Each node takes the full state and returns a dict of updated fields only.

def make_retrieve_node(retriever: Retriever):
    def retrieve_node(state: RAGState) -> dict:
        logger.info(f"[retrieve_node] query='{state['current_query'][:60]}'")
        chunks = retriever.retrieve(state["current_query"], k=config.TOP_K)
        return {"chunks": chunks}
    return retrieve_node


def make_score_node(scorer: Scorer):
    def score_node(state: RAGState) -> dict:
        scoring = scorer.score(state["current_query"], state["chunks"])
        logger.info(
            f"[score_node] mean={scoring['mean']} "
            f"| quality={scoring['quality']} "
            f"| scores={scoring['scores']}"
        )
        return {
            "scores":     scoring["scores"],
            "mean_score": scoring["mean"],
            "quality":    scoring["quality"],
        }
    return score_node


def decide_node(state: RAGState) -> str:
    """
    Router — decides which node to go to next.
    Returns the name of the next node as a string.
    This is used as a conditional edge in LangGraph.
    """
    if state["quality"] == "good":
        logger.info("[decide_node] Quality good → generate")
        return "generate"

    if state["attempts"] >= MAX_ATTEMPTS:
        logger.info(f"[decide_node] Max attempts ({MAX_ATTEMPTS}) reached → fallback")
        return "fallback"

    logger.info(f"[decide_node] Quality poor, attempts={state['attempts']} → reformulate")
    return "reformulate"


def make_reformulate_node(reformulator: Reformulator):
    def reformulate_node(state: RAGState) -> dict:
        logger.info(f"[reformulate_node] Reformulating: '{state['current_query'][:60]}'")
        rewrites = reformulator.reformulate(state["current_query"])

        # Pick the first valid rewrite (best-pick logic runs via re-scoring in score_node)
        new_query = rewrites[0] if rewrites else state["current_query"]

        updated_reformulations = state.get("reformulations", []) + rewrites
        logger.info(f"[reformulate_node] New query: '{new_query}'")

        return {
            "current_query":      new_query,
            "reformulations":     updated_reformulations,
            "attempts":           state["attempts"] + 1,
            "reformulated_query": new_query,
        }
    return reformulate_node


def make_generate_node(generator: Generator):
    def generate_node(state: RAGState) -> dict:
        logger.info(f"[generate_node] Generating answer...")
        result = generator.generate(state["current_query"], state["chunks"])
        return {
            "answer":  result["answer"],
            "sources": result["sources"],
        }
    return generate_node


def fallback_node(state: RAGState) -> dict:
    logger.info("[fallback_node] Returning insufficient context message")
    return {
        "answer":  INSUFFICIENT_CONTEXT_MSG,
        "sources": [],
    }


# ── Pipeline Class ────────────────────────────────────────────────────────────

class BasicRAGPipeline:

    def __init__(self):
        self.retriever    = Retriever()
        self.generator    = Generator()
        self.scorer       = Scorer()
        self.reformulator = Reformulator()
        self.graph        = self._build_graph()
        logger.info("BasicRAGPipeline (LangGraph) ready")

    def _build_graph(self) -> any:
        """
        Build the LangGraph state machine.

        Nodes:
            retrieve   → fetch top-k chunks
            score      → score chunks, set quality flag
            decide     → router: good→generate, poor→reformulate or fallback
            reformulate→ rewrite query, increment attempts
            generate   → produce final answer
            fallback   → return insufficient context

        Edges:
            retrieve → score → decide (conditional)
            decide → generate | reformulate | fallback
            reformulate → retrieve (loop back)
            generate → END
            fallback → END
        """
        graph = StateGraph(RAGState)

        # ── Add nodes ─────────────────────────────────────────────────────────
        graph.add_node("retrieve",    make_retrieve_node(self.retriever))
        graph.add_node("score",       make_score_node(self.scorer))
        graph.add_node("reformulate", make_reformulate_node(self.reformulator))
        graph.add_node("generate",    make_generate_node(self.generator))
        graph.add_node("fallback",    fallback_node)

        # ── Add edges ─────────────────────────────────────────────────────────
        graph.add_edge("retrieve", "score")
        graph.add_conditional_edges("score", decide_node, {
            "generate":    "generate",
            "reformulate": "reformulate",
            "fallback":    "fallback",
        })
        graph.add_edge("reformulate", "retrieve")   # loop back
        graph.add_edge("generate",    END)
        graph.add_edge("fallback",    END)

        # ── Entry point ───────────────────────────────────────────────────────
        graph.set_entry_point("retrieve")

        return graph.compile()

    def run(self, query: str, k: int = None) -> dict:
        t0 = time.time()

        # ── Initial state ─────────────────────────────────────────────────────
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

        # ── Run the graph ─────────────────────────────────────────────────────
        final_state = self.graph.invoke(initial_state)

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