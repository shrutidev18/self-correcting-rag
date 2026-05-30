import time
from groq import Groq

from app.utils.config import config
from app.utils.logger import logger


SYSTEM_PROMPT = """You are a precise question-answering assistant.
Answer questions using ONLY the provided document chunks.

Rules:
- Answer strictly from the context. Do NOT use outside knowledge.
- If the context does not contain the answer, respond exactly:
  "I don't have enough information in the provided context to answer this."
- Keep answers concise (2-4 sentences).
- At the end, list chunk IDs you used as: Sources: [id1, id2]
"""


def build_prompt(query: str, chunks: list) -> str:
    context_lines = []
    for i, chunk in enumerate(chunks):
        context_lines.append(
            f"[Chunk {i+1} | ID: {chunk['id']} | Score: {chunk['score']:.2f}]\n"
            f"{chunk['text']}"
        )
    context_str = "\n\n".join(context_lines)

    return (
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n\n"
        f"Answer (use only the context above):"
    )


class Generator:

    def __init__(self):
        config.validate()
        self.client = Groq(api_key=config.GROQ_API_KEY)
        self.model   = config.LLM_MODEL
        logger.info(f"Generator ready | model: {self.model}")

    def generate(self, query: str, chunks: list) -> dict:
        if not chunks:
            return {
                "answer":     "I don't have enough information in the provided context to answer this.",
                "sources":    [],
                "raw_output": "",
                "latency_ms": 0.0,
            }

        prompt = build_prompt(query, chunks)
        t0 = time.time()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.1,
            max_tokens=512,
        )

        latency_ms = (time.time() - t0) * 1000
        raw_output = response.choices[0].message.content.strip()
        sources    = self._parse_sources(raw_output, chunks)
        answer     = raw_output.split("Sources:")[0].strip()

        return {
            "answer":     answer,
            "sources":    sources,
            "raw_output": raw_output,
            "latency_ms": round(latency_ms, 2),
        }

    def _parse_sources(self, raw_output: str, chunks: list) -> list:
        import re
        match = re.search(r"Sources:\s*\[(.+?)\]", raw_output)
        if match:
            return [s.strip() for s in match.group(1).split(",")]
        return [c["id"] for c in chunks]