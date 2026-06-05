import re
import time

from groq import Groq

from app.utils.config import config
from app.utils.logger import logger


# ── Prompt ────────────────────────────────────────────────────────────────────
SCORER_SYSTEM = "You are a relevance judge. Respond with ONLY a single digit 1-5. No explanation."

SCORER_PROMPT = """You are a relevance judge. Given a question and a document chunk,
rate how useful this chunk is for answering the question.

Score 1: Completely irrelevant
Score 2: Marginally related, not useful
Score 3: Somewhat relevant, partially useful
Score 4: Relevant and useful
Score 5: Directly answers the question

Question: {query}
Chunk: {chunk_text}

Respond with ONLY a number 1-5."""


def _parse_score(raw: str) -> int:
    raw = raw.strip()
    if raw in {"1", "2", "3", "4", "5"}:
        return int(raw)
    matches = re.findall(r"[1-5]", raw)
    if matches:
        return int(matches[0])
    logger.warning(f"Could not parse score from: '{raw}' — defaulting to 3")
    return 3


class Scorer:

    def __init__(self, threshold: float = None):
        config.validate()
        self.client    = Groq(api_key=config.GROQ_API_KEY)
        self.model     = config.LLM_MODEL
        self.threshold = threshold if threshold is not None else config.QUALITY_THRESHOLD
        logger.info(f"Scorer ready | model: {self.model} | threshold: {self.threshold}")

    def _score_single_chunk(self, query: str, chunk_text: str) -> tuple[int, int]:
        prompt = SCORER_PROMPT.format(
            query=query,
            chunk_text=chunk_text[:500],
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SCORER_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        )

        raw    = response.choices[0].message.content.strip()
        score  = _parse_score(raw)
        tokens = response.usage.total_tokens if response.usage else 0

        logger.debug(f"Chunk scored {score} (raw='{raw}', tokens={tokens})")
        return score, tokens

    def score(self, query: str, chunks: list) -> dict:
        if not chunks:
            logger.warning("Scorer received empty chunk list")
            return {
                "scores":      [],
                "mean":        0.0,
                "quality":     "poor",
                "threshold":   self.threshold,
                "latency_ms":  0.0,
                "token_count": 0,
            }

        t0           = time.time()
        scores       = []
        total_tokens = 0

        for i, chunk in enumerate(chunks):
            logger.info(f"Scoring chunk {i+1}/{len(chunks)} | id={chunk['id']}")
            try:
                score, tokens = self._score_single_chunk(query, chunk["text"])
            except Exception as e:
                logger.warning(f"Scoring failed for chunk {chunk['id']}: {e} — defaulting to 3")
                score, tokens = 3, 0

            scores.append(score)
            total_tokens += tokens

        mean_score = round(sum(scores) / len(scores), 3)
        quality    = "good" if mean_score >= self.threshold else "poor"
        latency_ms = round((time.time() - t0) * 1000, 2)

        logger.info(
            f"Scoring complete | scores={scores} | mean={mean_score} "
            f"| quality={quality} | tokens={total_tokens}"
        )

        return {
            "scores":      scores,
            "mean":        mean_score,
            "quality":     quality,
            "threshold":   self.threshold,
            "latency_ms":  latency_ms,
            "token_count": total_tokens,
        }