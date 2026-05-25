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
    Dense: SentenceTransformer → FAISS IndexFlatIP (inner product = cosine on L2-normed vecs)
    Sparse: BM25Okapi on tokenized texts
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        faiss_path: str = "data/faiss_index",
        bm25_path: str = "data/bm25_store.pkl",
        chunks_path: str = "data/chunks.pkl",
    ):
        self.model = SentenceTransformer(model_name)
        self.faiss_path = faiss_path
        self.bm25_path = bm25_path
        self.chunks_path = chunks_path
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None
        self.index: faiss.IndexFlatIP | None = None

    def index(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]

        # Dense embeddings
        logger.info(f"Embedding {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        embeddings = embeddings.astype(np.float32)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        # Sparse BM25
        tokenized = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(tokenized)

        # Persist
        os.makedirs(os.path.dirname(self.faiss_path), exist_ok=True)
        faiss.write_index(self.index, self.faiss_path)
        with open(self.bm25_path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)
        logger.info(f"Saved FAISS index to {self.faiss_path}")

    def load(self) -> None:
        self.index = faiss.read_index(self.faiss_path)
        with open(self.bm25_path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]
        logger.info(f"Loaded {len(self.chunks)} chunks from disk")

    def search_dense(self, query: str, top_k: int) -> list[Chunk]:
        q_emb = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, idxs = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                chunk.score = float(score)
                results.append(chunk)
        return results

    def search_sparse(self, query: str, top_k: int) -> list[Chunk]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_idxs = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_idxs:
            chunk = self.chunks[idx]
            chunk.score = float(scores[idx])
            results.append(chunk)
        return results