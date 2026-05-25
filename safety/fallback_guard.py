"""
safety/fallback_guard.py
Detects out-of-corpus queries and triggers graceful fallback.
"""
import logging
from core.interfaces import BaseFallbackGuard, Chunk

logger = logging.getLogger(__name__)

# Patterns that signal out-of-corpus questions
OUT_OF_CORPUS_SIGNALS = [
    "campus visit date", "when will", "stock price", "work from home",
    "how many students from svecw", "institution-specific", "remote work",
    "wfh policy", "current price", "market cap",
]

MIN_RELEVANCE_SCORE = 0.05  # below this = likely out-of-corpus


class FallbackGuard(BaseFallbackGuard):
    """
    Two-pronged fallback detection:
    1. Keyword signals for known out-of-corpus query types
    2. Low retrieval score threshold
    """

    def is_out_of_corpus(self, query: str, chunks: list[Chunk]) -> bool:
        q = query.lower()

        # Check keyword signals
        for signal in OUT_OF_CORPUS_SIGNALS:
            if signal in q:
                logger.info(f"Fallback triggered by keyword: '{signal}'")
                return True

        # Check retrieval score threshold
        if not chunks:
            logger.info("Fallback triggered: no chunks retrieved")
            return True

        top_score = max(c.score for c in chunks)
        if top_score < MIN_RELEVANCE_SCORE:
            logger.info(f"Fallback triggered: top score {top_score:.4f} < threshold")
            return True

        return False