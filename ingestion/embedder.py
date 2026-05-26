"""
ingestion/embedder.py
Embeds chunks using sentence-transformers.
Stores dense vectors in FAISS + sparse index in BM25.
"""
import os
import pickle
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from core.interfaces import BaseEmbedder, Chunk

logger = logging.getLogger(__name__)


class HybridEmbedder(BaseEmbedder):
    """
    Dense:  SentenceTransformer → FAISS IndexFlatIP
    Sparse: BM25Okapi on tokenized texts
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        faiss_path: str = "data/faiss_index",
        bm25_path: str = "data/bm25_store.pkl",
        chunks_path: str = "data/chunks.pkl",
    ):
        self.model_name = model_name
        self.faiss_path = faiss_path
        self.bm25_path = bm25_path
        self.chunks_path = chunks_path
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None
        self.faiss_index: faiss.IndexFlatIP | None = None
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def build_index(self, chunks: list[Chunk]) -> None:
        """Build FAISS + BM25 index from chunks and persist to disk."""
        self.chunks = chunks
        texts = [c.text for c in chunks]

        # Dense embeddings
        logger.info(f"Embedding {len(texts)} chunks...")
        model = self._get_model()
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,
            batch_size=32,
        )
        embeddings = embeddings.astype(np.float32)

        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings)
        logger.info(f"FAISS index built: {self.faiss_index.ntotal} vectors, dim={dim}")

        # Sparse BM25
        tokenized = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built")

        # Persist everything
        os.makedirs("data", exist_ok=True)
        faiss.write_index(self.faiss_index, self.faiss_path)
        with open(self.bm25_path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)
        with open(self.chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)

        logger.info(f"Saved FAISS index → {self.faiss_path}")
        logger.info(f"Saved BM25 + chunks → {self.bm25_path}")

    def load(self) -> None:
        """Load persisted index from disk (call this in app.py)."""
        if not os.path.exists(self.faiss_path):
            raise FileNotFoundError(
                f"FAISS index not found at '{self.faiss_path}'. "
                "Run 'python scripts/ingest.py' first."
            )
        self.faiss_index = faiss.read_index(self.faiss_path)
        with open(self.bm25_path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]
        logger.info(
            f"Loaded index: {len(self.chunks)} chunks, "
            f"{self.faiss_index.ntotal} FAISS vectors"
        )

    def search_dense(self, query: str, top_k: int) -> list[Chunk]:
        if self.faiss_index is None:
            raise RuntimeError("Index not loaded. Call load() or build_index() first.")
        model = self._get_model()
        q_emb = model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)
        scores, idxs = self.faiss_index.search(q_emb, min(top_k, len(self.chunks)))
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if 0 <= idx < len(self.chunks):
                import copy
                chunk = copy.copy(self.chunks[idx])
                chunk.score = float(score)
                results.append(chunk)
        return results

    def search_sparse(self, query: str, top_k: int) -> list[Chunk]:
        if self.bm25 is None:
            raise RuntimeError("BM25 not loaded. Call load() or build_index() first.")
        import copy
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_idxs = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_idxs:
            if scores[idx] > 0:
                chunk = copy.copy(self.chunks[idx])
                chunk.score = float(scores[idx])
                results.append(chunk)
        return results