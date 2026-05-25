"""
evaluation/evaluator.py
Automated evaluation pipeline — scores all 30 queries.
"""
import time
import logging
import pandas as pd
from tqdm import tqdm
from core.pipeline import RAGPipeline
from evaluation.queries import EVAL_QUERIES

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Runs all 30 official queries and produces a scored report."""

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline

    def evaluate(self) -> pd.DataFrame:
        results = []
        for q in tqdm(EVAL_QUERIES, desc="Evaluating"):
            start = time.time()
            try:
                response = self.pipeline.run(q["query"])
                latency = round(time.time() - start, 2)

                keyword_hit = self._keyword_score(
                    response.answer, q.get("expected_keywords", [])
                )
                fallback_correct = (
                    response.fallback_triggered == q.get("expected_fallback", False)
                )

                results.append({
                    "id": q["id"],
                    "difficulty": q["difficulty"],
                    "query": q["query"],
                    "answer": response.answer[:200],
                    "keyword_score": keyword_hit,
                    "fallback_correct": fallback_correct,
                    "retrieval_quality": round(response.retrieval_quality, 3),
                    "conflicts": len(response.conflicts_detected),
                    "context_tokens": response.context_tokens,
                    "latency_s": latency,
                    "consistency": response.self_consistency_score,
                })
            except Exception as e:
                logger.error(f"Query {q['id']} failed: {e}")
                results.append({"id": q["id"], "error": str(e)})

        df = pd.DataFrame(results)
        self._print_summary(df)
        return df

    def _keyword_score(self, answer: str, keywords: list[str]) -> float:
        if not keywords:
            return 1.0
        ans_lower = answer.lower()
        hits = sum(1 for kw in keywords if kw.lower() in ans_lower)
        return round(hits / len(keywords), 2)

    def _print_summary(self, df: pd.DataFrame):
        print("\n" + "=" * 60)
        print("RAG EVALUATION SUMMARY")
        print("=" * 60)
        for diff in ["Easy", "Medium", "Hard", "Expert"]:
            sub = df[df["difficulty"] == diff]
            if not sub.empty:
                avg_kw = sub["keyword_score"].mean()
                print(f"{diff:8s}: {len(sub)} queries | avg keyword score: {avg_kw:.2f}")
        print(f"\nOverall keyword score: {df['keyword_score'].mean():.2f}")
        print(f"Avg retrieval quality: {df['retrieval_quality'].mean():.3f}")
        print(f"Avg latency: {df['latency_s'].mean():.2f}s")
        print("=" * 60)