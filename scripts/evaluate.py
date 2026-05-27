"""
scripts/evaluate.py
Run the full 34-query evaluation suite (30 official + 4 multi-hop).
Results saved to data/eval_results.csv.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.WARNING)

from version import print_banner
print_banner()

from ingestion.embedder import HybridEmbedder
from retrieval.retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker
from generation.refiner import ContextRefiner
from generation.prompt_builder import GroundedPromptBuilder
from generation.generator import GroqGenerator
from safety.conflict_detector import PlacementConflictDetector
from safety.fallback_guard import FallbackGuard
from safety.overshadow_limiter import OvershadowLimiter
from feedback.loop import FeedbackLoop
from core.pipeline import RAGPipeline, PipelineConfig
from evaluation.evaluator import RAGEvaluator


def main():
    print("Loading indexes...")
    embedder = HybridEmbedder(
        faiss_path=os.getenv("FAISS_INDEX_PATH", "data/faiss_index"),
        bm25_path=os.getenv("BM25_PATH",          "data/bm25_store.pkl"),
    )
    embedder.load()

    retriever        = HybridRetriever(embedder)

    try:
        reranker = CrossEncoderReranker()
    except Exception:
        from core.interfaces import BaseReranker
        class PassthroughReranker(BaseReranker):
            def rerank(self, q, r): return r
        reranker = PassthroughReranker()
        print("Warning: CrossEncoder unavailable — using passthrough reranker.")

    limiter          = OvershadowLimiter()
    refiner          = ContextRefiner(limiter)
    prompt_builder   = GroundedPromptBuilder()
    generator        = GroqGenerator()
    conflict_detector= PlacementConflictDetector()
    fallback_guard   = FallbackGuard()

    pipeline = RAGPipeline(
        retriever, reranker, refiner, prompt_builder,
        generator, conflict_detector, fallback_guard,
        PipelineConfig(top_k_retrieve=20, top_k_rerank=5),
    )

    evaluator = RAGEvaluator(pipeline)

    # Full evaluation (30 official + 4 multi-hop)
    print("\nRunning full evaluation (34 queries)...")
    df = evaluator.evaluate(include_multihop=True)

    out_path = "data/eval_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Results saved → {out_path}")

    # Dedicated multi-hop deep dive
    print("\nRunning multi-hop deep dive...")
    mh_df = evaluator.evaluate_multihop_only()
    mh_df.to_csv("data/multihop_results.csv", index=False)
    print("Multi-hop results saved → data/multihop_results.csv")


if __name__ == "__main__":
    main()