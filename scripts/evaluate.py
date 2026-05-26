"""
scripts/evaluate.py
Run all 30 official evaluation queries and save results to CSV.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.WARNING)  # quiet during eval

from ingestion.embedder import HybridEmbedder
from retrieval.retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker
from generation.refiner import ContextRefiner
from generation.prompt_builder import GroundedPromptBuilder
from generation.generator import GroqGenerator
from safety.conflict_detector import PlacementConflictDetector
from safety.fallback_guard import FallbackGuard
from safety.overshadow_limiter import OvershadowLimiter
from core.pipeline import RAGPipeline, PipelineConfig
from evaluation.evaluator import RAGEvaluator


def main():
    print("Loading indexes...")
    embedder = HybridEmbedder(
        faiss_path=os.getenv("FAISS_INDEX_PATH", "data/faiss_index"),
        bm25_path=os.getenv("BM25_PATH", "data/bm25_store.pkl"),
    )
    embedder.load()

    retriever = HybridRetriever(embedder)
    reranker = CrossEncoderReranker()
    limiter = OvershadowLimiter()
    refiner = ContextRefiner(limiter)
    prompt_builder = GroundedPromptBuilder()
    generator = GroqGenerator()
    conflict_detector = PlacementConflictDetector()
    fallback_guard = FallbackGuard()

    pipeline = RAGPipeline(
        retriever, reranker, refiner, prompt_builder,
        generator, conflict_detector, fallback_guard,
        PipelineConfig(top_k_retrieve=20, top_k_rerank=5),
    )

    evaluator = RAGEvaluator(pipeline)
    df = evaluator.evaluate()

    out_path = "data/eval_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()