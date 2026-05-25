"""
safety/overshadow_limiter.py
Binary-search context cap — prevents knowledge overshadowing.

Analogy: like camera sensor resolution vs display resolution,
there's a "sweet spot" of context beyond which the LLM starts
hallucinating from context confusion rather than helping.

We find this cap using additive increase / multiplicative decrease (AIMD).
"""
import logging
from core.interfaces import BaseRefiner, RetrievalResult

logger = logging.getLogger(__name__)

# Empirically-tuned thresholds (calibrated for Gemini Flash context)
MIN_CHUNKS = 1
MAX_CHUNKS = 8
TOKEN_BUDGET = 2000  # sweet spot before overshadowing


class OvershadowLimiter(BaseRefiner):
    """
    AIMD-style context limiter:
    - Additive increase: +1 chunk per successful answer
    - Multiplicative decrease: /2 on detected hallucination signal

    Binary search analogy: probes context size like binary search
    probes sorted array — finds maximum "safe" context.
    """

    def __init__(self):
        self.current_cap = 5       # start conservatively
        self.history: list[dict] = []

    def refine(self, result: RetrievalResult, max_tokens: int = TOKEN_BUDGET) -> RetrievalResult:
        chunks = result.chunks

        # Cap by current AIMD limit
        chunks = chunks[: self.current_cap]

        # Token-budget trimming
        total_tokens = 0
        kept = []
        for chunk in chunks:
            chunk_tokens = len(chunk.text.split())
            if total_tokens + chunk_tokens > max_tokens:
                break
            kept.append(chunk)
            total_tokens += chunk_tokens

        # Always keep at least 1 chunk
        if not kept and chunks:
            kept = [chunks[0]]

        result.chunks = kept
        result.overshadow_risk = self._estimate_risk(kept, total_tokens)

        self.history.append({
            "chunks_in": len(result.chunks),
            "tokens": total_tokens,
            "overshadow_risk": result.overshadow_risk,
        })

        logger.info(
            f"Refiner: {len(chunks)} → {len(kept)} chunks | "
            f"{total_tokens} tokens | risk={result.overshadow_risk:.2f}"
        )
        return result

    def _estimate_risk(self, chunks: list, tokens: int) -> float:
        """
        Hallucination risk formula:
        - <500 tokens: very low risk
        - 500-1500 tokens: moderate (golden zone)
        - >2000 tokens: high overshadowing risk
        """
        if tokens < 500:
            return 0.1
        elif tokens < 1500:
            return 0.3
        elif tokens < 2500:
            return 0.6
        else:
            return 0.9

    def feedback_good(self):
        """Additive increase on confirmed good answer."""
        self.current_cap = min(MAX_CHUNKS, self.current_cap + 1)
        logger.debug(f"AIMD increase: cap → {self.current_cap}")

    def feedback_bad(self):
        """Multiplicative decrease on hallucination signal."""
        self.current_cap = max(MIN_CHUNKS, self.current_cap // 2)
        logger.debug(f"AIMD decrease: cap → {self.current_cap}")

    def get_diagnostics(self) -> dict:
        """Returns retrieval quality diagnostics — the 'sensor spec sheet'."""
        return {
            "current_cap": self.current_cap,
            "token_budget": TOKEN_BUDGET,
            "sweet_spot_range": "500-1500 tokens",
            "overshadow_threshold": "2000 tokens",
            "history": self.history[-5:],
        }