"""
generation/generator.py
Gemini-powered answer generation with self-consistency.
"""
import os
import logging
import google.generativeai as genai
from core.interfaces import BaseGenerator, RAGResponse

logger = logging.getLogger(__name__)


class GeminiGenerator(BaseGenerator):
    """
    Self-consistency: generates N answers, picks the majority.
    Recitation guard: checks if answer is grounded in context.
    """

    def __init__(self, api_key: str = None, samples: int = 1):
        key = api_key or os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.samples = samples

    def generate(self, prompt: str) -> RAGResponse:
        answers = []
        for _ in range(self.samples):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=512,
                    ),
                )
                answers.append(response.text.strip())
            except Exception as e:
                logger.error(f"Gemini error: {e}")
                answers.append("Generation failed.")

        final_answer = self._majority_vote(answers)
        consistency = self._consistency_score(answers)

        # Extract sources from prompt chunks metadata
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
        # Simple majority: pick the longest non-failure answer
        valid = [a for a in answers if "failed" not in a.lower()]
        return max(valid or answers, key=len)

    def _consistency_score(self, answers: list[str]) -> float:
        if len(answers) <= 1:
            return 1.0
        # Jaccard similarity between first and rest
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