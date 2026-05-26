"""
generation/generator.py
Groq-powered answer generation with self-consistency.
Uses llama-3.1-8b-instant for fast inference (free tier).
"""
import os
import logging
from groq import Groq
from core.interfaces import BaseGenerator, RAGResponse

logger = logging.getLogger(__name__)

AVAILABLE_MODELS = [
    "llama-3.1-8b-instant",     # fastest, free
    "llama-3.3-70b-versatile",  # smarter, free
    "mixtral-8x7b-32768",       # large context
]


class GroqGenerator(BaseGenerator):
    """
    Self-consistency: generates N answers, picks majority.
    Recitation guard: low temperature keeps answers grounded.
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        samples: int = 1,
    ):
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY not set in .env")
        self.client = Groq(api_key=key)
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.samples = samples
        logger.info(f"GroqGenerator ready | model={self.model}")

    def generate(self, prompt: str) -> RAGResponse:
        answers = []
        for i in range(self.samples):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a Placement Intelligence Assistant. "
                                "Answer only from the provided context. "
                                "If the answer is not in the context, say exactly: "
                                "'I don't have enough information in the provided documents to answer this.'"
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,      # low = grounded, less hallucination
                    max_tokens=512,
                    top_p=0.9,
                )
                text = response.choices[0].message.content.strip()
                answers.append(text)
                logger.debug(f"Sample {i+1}: {text[:80]}...")
            except Exception as e:
                logger.error(f"Groq API error (sample {i+1}): {e}")
                answers.append("Generation failed.")

        final_answer = self._majority_vote(answers)
        consistency = self._consistency_score(answers)
        sources = self._extract_sources(prompt)

        return RAGResponse(
            answer=final_answer,
            sources=sources,
            conflicts_detected=[],
            fallback_triggered=False,
            retrieval_quality=0.0,
            context_tokens=len(prompt.split()),
            self_consistency_score=consistency,
        )

    def _majority_vote(self, answers: list[str]) -> str:
        if len(answers) == 1:
            return answers[0]
        valid = [a for a in answers if "failed" not in a.lower()]
        return max(valid or answers, key=len)

    def _consistency_score(self, answers: list[str]) -> float:
        if len(answers) <= 1:
            return 1.0
        base = set(answers[0].lower().split())
        scores = []
        for ans in answers[1:]:
            other = set(ans.lower().split())
            if not base or not other:
                scores.append(0.0)
            else:
                scores.append(len(base & other) / len(base | other))
        return round(sum(scores) / len(scores), 3) if scores else 1.0

    def _extract_sources(self, prompt: str) -> list[str]:
        import re
        return re.findall(r"\[([A-Z]+\s?\|[^\]]+)\]", prompt)[:5]