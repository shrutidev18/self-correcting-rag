import json
import re
import time

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
    """
    Extract list of 3 strings from LLM output.
    Handles: clean array, dicts with question values, truncated JSON.
    """
    raw = raw.strip()

    # Remove markdown code fences
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*",     "", raw)
    raw = raw.strip()

    # Strategy 1: try parsing the whole thing as JSON
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return _extract_strings(parsed)
    except json.JSONDecodeError:
        pass

    # Strategy 2: find a JSON array inside the text
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return _extract_strings(parsed)
        except json.JSONDecodeError:
            pass

    # Strategy 3: extract all quoted strings (handles truncated JSON)
    quoted = re.findall(r'"([^"]{10,})"', raw)
    # Filter out keys like "rewrite_1", "improvement", "synonym"
    questions = [s for s in quoted if len(s) > 15 and not s.startswith("rewrite_")]
    if questions:
        logger.info(f"Used fallback string extraction — found {len(questions)} strings")
        return questions[:3]

    logger.warning(f"Could not parse rewrites from: '{raw[:150]}'")
    return []


def _extract_strings(parsed: list) -> list:
    """Handle both plain strings and dicts in the parsed list."""
    rewrites = []
    for item in parsed[:3]:
        if isinstance(item, str) and len(item) > 5:
            rewrites.append(item.strip())
        elif isinstance(item, dict):
            # Take the longest string value — most likely to be the question
            values = [v for v in item.values() if isinstance(v, str) and len(v) > 10]
            if values:
                rewrites.append(max(values, key=len).strip())
    return rewrites


class Reformulator:
    """
    Rewrites a failed query into 3 alternatives to improve retrieval.

    When the scorer flags retrieval as poor, this class asks the LLM
    to think of 3 different ways to search for the same information —
    broader terms, synonyms, or sub-questions.

    Example:
        Original : "who sang go rest high on the mountain"
        Rewrite 1: "Vince Gill country song mountain"
        Rewrite 2: "Christian country music 1990s funeral song"
        Rewrite 3: "go rest high mountain song artist singer"
    """

    def __init__(self):
        config.validate()
        self.client = Groq(api_key=config.GROQ_API_KEY)
        self.model  = config.LLM_MODEL
        logger.info(f"Reformulator ready | model: {self.model}")

    def reformulate(self, query: str) -> list:
        """
        Call LLM to rewrite the query 3 ways.

        Args:
            query: The original query that got poor retrieval.

        Returns:
            List of up to 3 rewritten query strings.
            Returns empty list if LLM call fails.
        """
        prompt = REFORMULATOR_PROMPT.format(query=query)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": REFORMULATOR_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.7,
                max_tokens=400,   # increased — dicts need more tokens
            )

            raw      = response.choices[0].message.content.strip()
            rewrites = _parse_rewrites(raw)

            logger.info(f"Reformulated '{query[:50]}' → {len(rewrites)} rewrites")
            for i, r in enumerate(rewrites):
                logger.info(f"  Rewrite {i+1}: {r}")

            return rewrites

        except Exception as e:
            logger.error(f"Reformulation failed: {e}")
            return []