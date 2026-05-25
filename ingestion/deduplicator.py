"""
ingestion/deduplicator.py
Removes near-duplicate chunks using cosine similarity on TF-IDF vectors.
Reduces 400+ raw chunks to ~80-120 meaningful ones.
"""
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from core.interfaces import BaseDeduplicator, Chunk

logger = logging.getLogger(__name__)


class TFIDFDeduplicator(BaseDeduplicator):
    """
    Deduplication via TF-IDF cosine similarity.
    threshold=0.85 catches copy-paste repeats (the PDF repeats each
    interview 3x intentionally as a RAG challenge).
    """

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold

    def deduplicate(self, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return chunks

        # Conflict chunks — keep ALL (both official + portal)
        conflict_chunks = [c for c in chunks if c.conflict]
        regular_chunks = [c for c in chunks if not c.conflict]

        deduped = self._dedup_regular(regular_chunks)
        result = deduped + conflict_chunks

        removed = len(chunks) - len(result)
        logger.info(
            f"Dedup: {len(chunks)} → {len(result)} chunks "
            f"(removed {removed} near-duplicates, kept {len(conflict_chunks)} conflict pairs)"
        )
        return result

    def _dedup_regular(self, chunks: list[Chunk]) -> list[Chunk]:
        if len(chunks) <= 1:
            return chunks

        texts = [c.text for c in chunks]
        tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        matrix = tfidf.fit_transform(texts)
        sim = cosine_similarity(matrix)

        keep = []
        dropped = set()
        for i in range(len(chunks)):
            if i in dropped:
                continue
            keep.append(chunks[i])
            for j in range(i + 1, len(chunks)):
                if sim[i, j] >= self.threshold:
                    dropped.add(j)

        return keep