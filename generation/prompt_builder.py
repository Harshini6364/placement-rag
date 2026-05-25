"""
generation/prompt_builder.py
Builds grounded prompts with conflict warnings and fallback signals.
"""
from core.interfaces import BasePromptBuilder, RetrievalResult

SYSTEM_PROMPT = """You are a Placement Intelligence Assistant for SVECW students.
Answer ONLY based on the provided context chunks.
If the context does not contain the answer, say exactly:
"I don't have enough information in the provided documents to answer this."

Rules:
- Never invent numbers, company names, or eligibility criteria.
- If you see [OFFICIAL SOURCE] and [PORTAL SOURCE] for the same company, flag the conflict.
- For temporal questions, use the year metadata in the chunks.
- For multi-hop questions, reason step by step before giving a final answer.
- Be concise and direct.
"""


class GroundedPromptBuilder(BasePromptBuilder):
    """Builds a grounded, conflict-aware prompt."""

    def build(self, query: str, result: RetrievalResult) -> str:
        context_parts = []
        for i, chunk in enumerate(result.chunks, 1):
            meta = f"[{chunk.section.upper()} | {chunk.company or 'General'}"
            if chunk.year:
                meta += f" | {chunk.year}"
            if chunk.conflict:
                meta += " | CONFLICT"
            meta += "]"
            context_parts.append(f"Chunk {i} {meta}:\n{chunk.text}")

        context = "\n\n".join(context_parts)

        prompt = f"""{SYSTEM_PROMPT}

=== RETRIEVED CONTEXT ===
{context}

=== QUESTION ===
{query}

=== ANSWER ===
Think step by step, then give your final answer:"""

        return prompt