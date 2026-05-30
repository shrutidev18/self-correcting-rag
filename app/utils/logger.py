import json
import logging
import sys
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("sc_rag")

QUERY_LOG_PATH = LOG_DIR / "query_log.jsonl"

def log_query(
    query: str,
    retrieved_chunks: list,
    answer: str,
    latency_ms: float,
    attempts: int = 1,
    quality_score: float = None,
    reformulated_query: str = None,
):
    record = {
        "timestamp":          datetime.utcnow().isoformat(),
        "query":              query,
        "answer":             answer,
        "attempts":           attempts,
        "quality_score":      quality_score,
        "reformulated_query": reformulated_query,
        "latency_ms":         round(latency_ms, 2),
        "num_chunks":         len(retrieved_chunks),
        "chunk_ids":          [c.get("id", "") for c in retrieved_chunks],
    }
    with open(QUERY_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    logger.info(f"Query logged | attempts={attempts} | latency={latency_ms:.0f}ms")