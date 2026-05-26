"""
feedback/loop.py
AIMD feedback controller — tracks answer quality and adjusts retrieval cap.
"""
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FeedbackRecord:
    query: str
    retrieval_quality: float
    context_tokens: int
    overshadow_risk: float
    user_rating: int = 0  # 1=good, -1=bad, 0=unrated


class FeedbackLoop:
    """
    Additive Increase / Multiplicative Decrease controller.
    Tells the overshadow limiter to grow or shrink context cap.
    """

    def __init__(self, limiter):
        self.limiter = limiter
        self.records: list[FeedbackRecord] = []

    def record(self, query: str, quality: float, tokens: int, risk: float) -> FeedbackRecord:
        rec = FeedbackRecord(query, quality, tokens, risk)
        self.records.append(rec)
        return rec

    def good(self, rec: FeedbackRecord):
        rec.user_rating = 1
        self.limiter.feedback_good()
        logger.info(f"Feedback GOOD → cap now {self.limiter.current_cap}")

    def bad(self, rec: FeedbackRecord):
        rec.user_rating = -1
        self.limiter.feedback_bad()
        logger.info(f"Feedback BAD → cap now {self.limiter.current_cap}")

    def summary(self) -> dict:
        rated = [r for r in self.records if r.user_rating != 0]
        good = sum(1 for r in rated if r.user_rating == 1)
        bad = sum(1 for r in rated if r.user_rating == -1)
        return {
            "total_queries": len(self.records),
            "rated": len(rated),
            "good": good,
            "bad": bad,
            "accuracy": round(good / len(rated), 2) if rated else 0.0,
            "current_cap": self.limiter.current_cap,
        }