"""
generation/refiner.py
Context pruning — removes low-quality chunks after reranking.
"""
import logging
from core.interfaces import RetrievalResult
from safety.overshadow_limiter import OvershadowLimiter

logger = logging.getLogger(__name__)


class ContextRefiner:
    """Wraps OvershadowLimiter with additional score-based pruning."""

    def __init__(self, limiter: OvershadowLimiter):
        self.limiter = limiter

    def refine(self, result: RetrievalResult, max_tokens: int = 2000) -> RetrievalResult:
        # Score-based pruning: remove chunks below 10% of top score
        if result.chunks:
            top = result.chunks[0].score
            threshold = top * 0.10
            result.chunks = [c for c in result.chunks if c.score >= threshold]

        # AIMD-based token cap
        return self.limiter.refine(result, max_tokens)