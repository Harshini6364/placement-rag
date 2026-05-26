"""
generation/prompt_builder.py
Builds grounded prompts with aggregation instructions,
conflict warnings, temporal reasoning guidance, and fallback signals.
"""
from core.interfaces import BasePromptBuilder, RetrievalResult

SYSTEM_PROMPT = """You are a Placement Intelligence Assistant for SVECW students.

STRICT RULES:
1. Answer ONLY from the provided context chunks. Never invent data.
2. If context lacks the answer, say exactly:
   "I don't have enough information in the provided documents to answer this."
3. CONFLICT: If you see [OFFICIAL SOURCE] and [PORTAL SOURCE] for the same company
   with different values, say: "Conflicting data found — official: X, portal: Y. Verify with placement cell."
4. TEMPORAL: For trend/growth questions, compare values across years using year metadata.
   Show the calculation: final_value - start_value = growth.
5. AGGREGATION: For "which company has highest/lowest/most" questions:
   - Scan ALL relevant chunks provided
   - Compare the values explicitly
   - Show your reasoning before the final answer
6. MULTI-HOP: For complex eligibility queries (CGPA + backlog + package):
   - Step 1: Filter by CGPA condition
   - Step 2: Filter by backlog condition
   - Step 3: Sort by package
   - State each step clearly
7. EDGE CASES: If query asks about CGPA < 6.3 (below all thresholds), say no company qualifies.
"""


class GroundedPromptBuilder(BasePromptBuilder):

    def build(self, query: str, result: RetrievalResult) -> str:
        context_parts = []
        for i, chunk in enumerate(result.chunks, 1):
            meta = f"[{chunk.section.upper()}"
            if chunk.company:
                meta += f" | {chunk.company}"
            if chunk.year:
                meta += f" | {chunk.year}"
            if chunk.conflict:
                meta += " | CONFLICT"
            if chunk.source == "vision":
                meta += " | VISION_CHART"
            meta += "]"
            context_parts.append(f"Chunk {i} {meta}:\n{chunk.text}")

        context = "\n\n".join(context_parts)

        # Detect query type for targeted instruction
        query_instruction = self._query_type_instruction(query)

        prompt = f"""{SYSTEM_PROMPT}

=== RETRIEVED CONTEXT ({len(result.chunks)} chunks) ===
{context}

=== QUESTION ===
{query}

=== INSTRUCTION ===
{query_instruction}

Think step by step, then give your final answer:"""

        return prompt

    def _query_type_instruction(self, query: str) -> str:
        q = query.lower()

        if any(w in q for w in ["grew", "growth", "increase", "trend", "2021", "2022", "2023", "2024"]):
            return (
                "This is a TEMPORAL query. "
                "Find year-tagged chunks and compute: growth = value_2024 - value_2021. "
                "Show the arithmetic clearly."
            )
        if any(w in q for w in ["highest", "most", "best", "rank", "maximum", "compare all"]):
            return (
                "This is an AGGREGATION query. "
                "Scan all chunks, list all relevant values, then identify the maximum/minimum. "
                "Show the comparison table before your final answer."
            )
        if any(w in q for w in ["cgpa", "backlog", "qualify", "eligible", "apply"]):
            return (
                "This is a MULTI-HOP ELIGIBILITY query. "
                "Step 1: filter companies where min_cgpa <= student_cgpa. "
                "Step 2: filter where max_backlogs >= student_backlogs. "
                "Step 3: sort remaining by package descending. "
                "Show each step."
            )
        if any(w in q for w in ["conflict", "discrepancy", "6.4 or 7.0", "which is correct"]):
            return (
                "This is a CONFLICT RESOLUTION query. "
                "Find both [OFFICIAL SOURCE] and [PORTAL SOURCE] chunks. "
                "Present both values and recommend verifying with the placement cell."
            )
        return "Answer directly and concisely from the context provided."