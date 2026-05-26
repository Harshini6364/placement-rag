"""
retrieval/retriever.py
Hybrid retriever: dense (FAISS) + sparse (BM25) with RRF fusion.
Includes metadata-aware filtering for temporal and aggregation queries.
"""
import re
import logging
from collections import defaultdict
from core.interfaces import BaseRetriever, RetrievalResult, Chunk
from retrieval.rewriter import QueryRewriter
from ingestion.embedder import HybridEmbedder

logger = logging.getLogger(__name__)

RRF_K = 60


class HybridRetriever(BaseRetriever):

    def __init__(self, embedder: HybridEmbedder, alpha: float = 0.6):
        self.embedder = embedder
        self.rewriter = QueryRewriter()
        self.alpha = alpha

    def retrieve(self, query: str, top_k: int) -> RetrievalResult:
        variants = self.rewriter.rewrite(query)
        all_dense, all_sparse = [], []

        for variant in variants:
            all_dense.extend(self.embedder.search_dense(variant, top_k))
            all_sparse.extend(self.embedder.search_sparse(variant, top_k))

        fused = self._rrf_fuse(all_dense, all_sparse, top_k)

        # Metadata boosting — pull in year-specific chunks for temporal queries
        year = self._detect_year(query)
        if year:
            year_chunks = self._get_year_chunks(year)
            existing_ids = {c.chunk_id for c in fused}
            for c in year_chunks:
                if c.chunk_id not in existing_ids:
                    fused.append(c)
            logger.debug(f"Temporal boost: injected {len(year_chunks)} chunks for year {year}")

        # Section boosting — inject all trend chunks for growth/increase queries
        if self._is_aggregation_query(query):
            agg_chunks = self._get_section_chunks(["trend", "hiring", "eligibility"])
            existing_ids = {c.chunk_id for c in fused}
            injected = 0
            for c in agg_chunks:
                if c.chunk_id not in existing_ids:
                    fused.append(c)
                    injected += 1
            logger.debug(f"Aggregation boost: injected {injected} chunks")

        quality = self._compute_quality(fused)
        overshadow = self._overshadow_risk(fused)

        return RetrievalResult(
            chunks=fused,
            query_used=variants[0],
            dense_scores=[c.score for c in fused],
            retrieval_quality=quality,
            overshadow_risk=overshadow,
        )

    def _rrf_fuse(self, dense, sparse, top_k):
        scores: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, Chunk] = {}

        for rank, chunk in enumerate(dense):
            cid = chunk.chunk_id
            scores[cid] += self.alpha / (RRF_K + rank + 1)
            chunk_map[cid] = chunk

        for rank, chunk in enumerate(sparse):
            cid = chunk.chunk_id
            scores[cid] += (1 - self.alpha) / (RRF_K + rank + 1)
            if cid not in chunk_map:
                chunk_map[cid] = chunk

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        result = []
        for cid, score in ranked:
            chunk = chunk_map[cid]
            chunk.score = score
            result.append(chunk)
        return result

    def _detect_year(self, query: str) -> int:
        """Extract explicit year from query for temporal metadata filtering."""
        match = re.search(r'\b(2021|2022|2023|2024)\b', query)
        if match:
            return int(match.group(1))
        q = query.lower()
        if any(w in q for w in ["this year", "current", "latest", "now"]):
            return 2024
        if "last year" in q:
            return 2023
        return 0

    def _is_aggregation_query(self, query: str) -> bool:
        """Detect queries that need data from multiple chunks to aggregate."""
        q = query.lower()
        agg_signals = [
            "most", "highest", "lowest", "best", "worst", "maximum", "minimum",
            "rank", "compare", "all companies", "list all", "which companies",
            "grew the most", "increased", "decreased", "trend", "growth",
            "average", "total", "sum", "across", "overall",
        ]
        return any(signal in q for signal in agg_signals)

    def _get_year_chunks(self, year: int) -> list[Chunk]:
        """Return all chunks with matching year metadata."""
        import copy
        result = []
        for chunk in self.embedder.chunks:
            if chunk.year == year:
                c = copy.copy(chunk)
                c.score = 0.01  # low base score — boosted by reranker if relevant
                result.append(c)
        return result

    def _get_section_chunks(self, sections: list[str]) -> list[Chunk]:
        """Return all chunks from specified sections for aggregation queries."""
        import copy
        result = []
        for chunk in self.embedder.chunks:
            if chunk.section in sections:
                c = copy.copy(chunk)
                c.score = 0.01
                result.append(c)
        return result

    def _compute_quality(self, chunks):
        if not chunks:
            return 0.0
        return min(1.0, sum(c.score for c in chunks[:5]) / 5 * 10)

    def _overshadow_risk(self, chunks):
        if not chunks:
            return 0.0
        total_tokens = sum(len(c.text.split()) for c in chunks)
        token_risk = min(1.0, total_tokens / 3000)
        score_spread = chunks[0].score - chunks[-1].score if len(chunks) > 1 else 0
        diversity_risk = 1.0 - min(1.0, score_spread * 5)
        return round(token_risk * 0.6 + diversity_risk * 0.4, 3)