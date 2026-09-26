import json
import re

from groq import Groq

from app.utils.config import config
from app.utils.logger import logger


REFORMULATOR_SYSTEM = "You are a search query expert. Respond with ONLY a JSON array of 3 strings. No dictionaries. No explanation."

REFORMULATOR_PROMPT = """The following question did not retrieve useful information from our document collection.
Rewrite it in 3 different ways to improve retrieval. Try: broader terms, synonyms,
breaking into sub-questions.

Original question: {query}

You MUST return a JSON array of exactly 3 plain strings like this:
["rewritten question 1", "rewritten question 2", "rewritten question 3"]

Do NOT return dictionaries. Do NOT add keys like rewrite_1 or improvement."""


def _parse_rewrites(raw: str) -> list:
    raw = raw.strip()

    # remove markdown code fences if present
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw)
    raw = raw.strip()

    # try parsing the whole response as JSON first
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return _extract_strings(parsed)
    except json.JSONDecodeError:
        pass

    # try finding a JSON array anywhere in the response
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return _extract_strings(parsed)
        except json.JSONDecodeError:
            pass

    # last resort — extract any quoted strings longer than 15 chars
    quoted = re.findall(r'"([^"]{10,})"', raw)
    questions = [s for s in quoted if len(s) > 15 and not s.startswith("rewrite_")]
    if questions:
        logger.info(f"used fallback string extraction — found {len(questions)} strings")
        return questions[:3]

    logger.warning(f"could not parse rewrites from: '{raw[:100]}'")
    return []


def _extract_strings(parsed: list) -> list:
    rewrites = []
    for item in parsed[:3]:
        if isinstance(item, str) and len(item) > 5:
            rewrites.append(item.strip())
        elif isinstance(item, dict):
            # LLM sometimes returns dicts like {"rewrite_1": "...", "improvement": "..."}
            values = [v for v in item.values() if isinstance(v, str) and len(v) > 10]
            if values:
                rewrites.append(max(values, key=len).strip())
    return rewrites


class Reformulator:

    def __init__(self):
        config.validate()
        self.client = Groq(api_key=config.GROQ_API_KEY)
        self.model = config.LLM_MODEL
        logger.info(f"Reformulator ready | model={self.model}")

    def reformulate(self, query: str) -> list:
        prompt = REFORMULATOR_PROMPT.format(query=query)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": REFORMULATOR_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=400,
            )

            raw = response.choices[0].message.content.strip()
            rewrites = _parse_rewrites(raw)

            logger.info(f"reformulated '{query[:50]}' → {len(rewrites)} rewrites")
            for i, r in enumerate(rewrites):
                logger.info(f"  rewrite {i+1}: {r}")

            return rewrites

        except Exception as e:
            logger.error(f"reformulation failed: {e}")
            return []