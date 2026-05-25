"""
retrieval/reranker.py
CrossEncoder reranking for precision after broad retrieval.
"""
import logging
from sentence_transformers import CrossEncoder
from core.interfaces import BaseReranker, RetrievalResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """
    Uses a CrossEncoder model (ms-marco) to rerank retrieved chunks.
    Much more accurate than bi-encoder similarity alone.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"Loading CrossEncoder: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, result: RetrievalResult) -> RetrievalResult:
        if not result.chunks:
            return result

        pairs = [(query, chunk.text) for chunk in result.chunks]
        scores = self.model.predict(pairs)

        for chunk, score in zip(result.chunks, scores):
            chunk.score = float(score)

        result.chunks.sort(key=lambda c: c.score, reverse=True)
        result.rerank_scores = [c.score for c in result.chunks]

        # Update quality based on rerank scores
        if result.rerank_scores:
            top_score = result.rerank_scores[0]
            result.retrieval_quality = min(1.0, max(0.0, top_score / 10.0))

        logger.debug(f"Reranked {len(result.chunks)} chunks | top={result.rerank_scores[0]:.3f}")
        return result