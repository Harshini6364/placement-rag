"""
evaluation/evaluator.py
Automated evaluation pipeline — scores all 30 official queries
plus 4 multi-hop reasoning queries.
Prints a visual bar chart summary in the terminal.
"""
import time
import logging
import pandas as pd
from tqdm import tqdm
from core.pipeline import RAGPipeline
from evaluation.queries import EVAL_QUERIES, MULTIHOP_QUERIES, ALL_QUERIES

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Runs all queries and produces a scored report with visual summary."""

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline

    def evaluate(self, include_multihop: bool = True) -> pd.DataFrame:
        queries = ALL_QUERIES if include_multihop else EVAL_QUERIES
        results = []

        for q in tqdm(queries, desc="Evaluating"):
            start = time.time()
            try:
                response = self.pipeline.run(q["query"])
                latency = round(time.time() - start, 2)

                keyword_score = self._keyword_score(
                    response.answer, q.get("expected_keywords", [])
                )
                fallback_correct = (
                    response.fallback_triggered == q.get("expected_fallback", False)
                )

                results.append({
                    "id":               q["id"],
                    "difficulty":       q["difficulty"],
                    "skill":            q.get("skill", ""),
                    "query":            q["query"],
                    "answer":           response.answer[:300],
                    "keyword_score":    keyword_score,
                    "fallback_correct": fallback_correct,
                    "retrieval_quality":round(response.retrieval_quality, 3),
                    "conflicts":        len(response.conflicts_detected),
                    "context_tokens":   response.context_tokens,
                    "latency_s":        latency,
                    "consistency":      response.self_consistency_score,
                    "tool_used":        "tool:" in str(response.sources),
                })

            except Exception as e:
                logger.error(f"Query {q['id']} failed: {e}")
                results.append({
                    "id":            q["id"],
                    "difficulty":    q.get("difficulty", "?"),
                    "skill":         q.get("skill", ""),
                    "query":         q["query"],
                    "answer":        f"ERROR: {e}",
                    "keyword_score": 0.0,
                    "fallback_correct": False,
                    "retrieval_quality": 0.0,
                    "conflicts":     0,
                    "context_tokens": 0,
                    "latency_s":     0.0,
                    "consistency":   0.0,
                    "tool_used":     False,
                })

        df = pd.DataFrame(results)
        self._print_summary(df)
        return df

    def evaluate_multihop_only(self) -> pd.DataFrame:
        """Run only the 4 multi-hop reasoning queries with hop trace."""
        results = []
        print("\n" + "=" * 60)
        print("MULTI-HOP REASONING EVALUATION")
        print("=" * 60)

        for q in MULTIHOP_QUERIES:
            print(f"\n[{q['id']}] {q['query']}")
            print(f"  Skill: {q['skill']}")
            print(f"  Reasoning chain:")
            for step in q.get("reasoning_chain", []):
                print(f"    {step}")

            start = time.time()
            response = self.pipeline.run(q["query"])
            latency = round(time.time() - start, 2)

            score = self._keyword_score(
                response.answer, q.get("expected_keywords", [])
            )
            print(f"  Answer: {response.answer[:200]}")
            print(f"  Score: {score:.0%} | Latency: {latency}s")

            results.append({
                "id":            q["id"],
                "query":         q["query"],
                "hops":          " → ".join(q.get("hops", [])),
                "answer":        response.answer[:300],
                "keyword_score": score,
                "latency_s":     latency,
            })

        df = pd.DataFrame(results)
        avg = df["keyword_score"].mean()
        print(f"\nMulti-hop average score: {avg:.0%}")
        return df

    def _keyword_score(self, answer: str, keywords: list[str]) -> float:
        if not keywords:
            return 1.0
        ans_lower = answer.lower()
        hits = sum(1 for kw in keywords if kw.lower() in ans_lower)
        return round(hits / len(keywords), 2)

    def _print_summary(self, df: pd.DataFrame):
        print("\n" + "=" * 60)
        print("  RAG EVALUATION SUMMARY")
        print("=" * 60)

        difficulties = ["Easy", "Medium", "Hard", "Expert", "Multi-hop"]
        for diff in difficulties:
            sub = df[df["difficulty"] == diff]
            if sub.empty:
                continue
            avg_kw   = sub["keyword_score"].mean()
            avg_lat  = sub["latency_s"].mean()
            n        = len(sub)
            bar      = "█" * int(avg_kw * 10) + "░" * (10 - int(avg_kw * 10))
            print(f"  {diff:10s} [{bar}] {avg_kw*100:5.1f}%  "
                  f"({n} queries, avg {avg_lat:.1f}s)")

        print("-" * 60)
        overall = df["keyword_score"].mean()
        bar = "█" * int(overall * 10) + "░" * (10 - int(overall * 10))
        print(f"  {'Overall':10s} [{bar}] {overall*100:5.1f}%")
        print(f"  Avg retrieval quality : {df['retrieval_quality'].mean():.3f}")
        print(f"  Avg latency           : {df['latency_s'].mean():.2f}s")
        tool_count = df["tool_used"].sum() if "tool_used" in df.columns else 0
        print(f"  Tool-assisted answers : {tool_count}")
        print(f"  Conflicts detected    : {df['conflicts'].sum()}")
        print("=" * 60)

        # Per-skill breakdown
        print("\n  SKILL BREAKDOWN")
        print("  " + "-" * 40)
        for skill, grp in df.groupby("skill"):
            if not skill:
                continue
            s = grp["keyword_score"].mean()
            print(f"  {skill[:35]:35s} {s*100:5.1f}%")
        print("=" * 60 + "\n")