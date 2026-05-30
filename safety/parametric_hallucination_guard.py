"""
safety/parametric_hallucination_guard.py

Mitigates three types of parametric hallucination:

1. SELF-CONSISTENCY
   Run the same prompt N times, collect answers, pick majority.
   If answers disagree → low confidence → warn user.

2. CHAIN OF ACTIONS
   Decompose complex queries into sub-steps.
   Execute each sub-step, verify output before proceeding.
   Never jump to conclusion without completing each step.

3. RECITATION
   Force LLM to cite exact text from chunks.
   Compare cited text against actual retrieved chunks.
   If cited text not found in any chunk → hallucination detected.
"""
import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class HallucinationReport:
    """Report of hallucination checks for one response."""
    self_consistency_score: float = 1.0      # 0-1, higher = more consistent
    consistency_passed:     bool  = True
    recitation_score:       float = 1.0      # 0-1, higher = better grounded
    recitation_passed:      bool  = True
    chain_verified:         bool  = True
    flagged_phrases:        list  = field(default_factory=list)
    final_verdict:          str   = "PASS"   # PASS / WARN / FAIL
    explanation:            str   = ""


class SelfConsistencyChecker:
    """
    Runs the same prompt multiple times and measures agreement.

    Why: LLMs are stochastic. Running once might give a lucky wrong answer.
    Running 3 times and picking majority is much more reliable.

    Camera analogy: like taking 3 photos and using the sharpest one
    instead of trusting a single shot.
    """

    def __init__(self, generator, num_samples: int = 3):
        self.generator = generator
        self.num_samples = num_samples

    def check(self, prompt: str) -> tuple[str, float]:
        """
        Returns (best_answer, consistency_score).
        consistency_score = 1.0 means all samples agreed perfectly.
        """
        if self.num_samples <= 1:
            result = self.generator.generate(prompt)
            return result.answer, 1.0

        answers = []
        for i in range(self.num_samples):
            try:
                result = self.generator.generate(prompt)
                answers.append(result.answer.strip())
                logger.debug(f"Sample {i+1}: {result.answer[:80]}")
            except Exception as e:
                logger.warning(f"Self-consistency sample {i+1} failed: {e}")

        if not answers:
            return "Generation failed.", 0.0

        # Pick majority answer using pairwise similarity
        best = self._majority_vote(answers)
        score = self._consistency_score(answers)

        logger.info(
            f"Self-consistency: {self.num_samples} samples, "
            f"score={score:.2f}, best='{best[:60]}'"
        )
        return best, score

    def _majority_vote(self, answers: list[str]) -> str:
        """Pick the answer most similar to all others."""
        if len(answers) == 1:
            return answers[0]

        best_answer = answers[0]
        best_score  = -1.0

        for candidate in answers:
            total = 0.0
            for other in answers:
                if candidate != other:
                    total += self._jaccard(candidate, other)
            avg = total / (len(answers) - 1)
            if avg > best_score:
                best_score  = avg
                best_answer = candidate

        return best_answer

    def _consistency_score(self, answers: list[str]) -> float:
        """Average pairwise Jaccard similarity between all answers."""
        if len(answers) <= 1:
            return 1.0
        pairs  = 0
        total  = 0.0
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                total += self._jaccard(answers[i], answers[j])
                pairs += 1
        return round(total / pairs, 3) if pairs > 0 else 1.0

    def _jaccard(self, a: str, b: str) -> float:
        set_a = set(a.lower().split())
        set_b = set(b.lower().split())
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)


class RecitationChecker:
    """
    Forces the LLM to cite source text, then verifies the citation
    exists in the actual retrieved chunks.

    Why: if the LLM says a number that appears in no chunk,
    it came from parametric memory (training data) = hallucination.

    Example:
    LLM says: "Amazon CGPA cutoff is 7.0"
    Chunks contain: "Amazon. Minimum CGPA required: 6.4"
    7.0 not found in any chunk → FLAGGED as parametric hallucination.
    """

    # Numbers and named entities are high-risk for hallucination
    HIGH_RISK_PATTERNS = [
        r'\b\d+\.?\d*\s*(?:lpa|cgpa|%|years?|backlogs?)\b',  # numbers with units
        r'\b(?:tcs|infosys|amazon|google|microsoft|wipro|cognizant|'
        r'capgemini|ibm|adobe|oracle|sap|hcl|qualcomm|intel|samsung|'
        r'deloitte|accenture|flipkart)\b',                      # company names
        r'\b\d{4}\b',                                           # years
    ]

    def check(self, answer: str, chunks: list) -> tuple[float, list[str]]:
        """
        Returns (recitation_score, flagged_phrases).
        recitation_score = fraction of high-risk claims found in chunks.
        """
        chunk_texts = " ".join(c.text.lower() for c in chunks)
        claims      = self._extract_claims(answer)

        if not claims:
            return 1.0, []

        verified = 0
        flagged  = []

        for claim in claims:
            claim_lower = claim.lower().strip()
            if claim_lower in chunk_texts:
                verified += 1
            else:
                # Try partial match for numbers with units
                core = re.sub(r'[^\d\.\w]', ' ', claim_lower).strip()
                if core and core in chunk_texts:
                    verified += 1
                else:
                    flagged.append(claim)
                    logger.warning(
                        f"Recitation check FAILED: '{claim}' not found in chunks"
                    )

        score = round(verified / len(claims), 3)
        logger.info(
            f"Recitation: {verified}/{len(claims)} claims verified, "
            f"score={score:.2f}"
        )
        return score, flagged

    def _extract_claims(self, answer: str) -> list[str]:
        """Extract high-risk factual claims from answer text."""
        claims = []
        for pattern in self.HIGH_RISK_PATTERNS:
            matches = re.findall(pattern, answer.lower())
            claims.extend(matches)
        return list(set(claims))


class ChainOfActionsVerifier:
    """
    Verifies that the LLM actually executed each reasoning step
    before giving a final answer.

    Why: without verification, the LLM might skip reasoning steps
    and jump to a conclusion using parametric memory.

    Example query: "A student with CGPA 7.6 and 1 backlog wants highest pay"
    Expected steps:
      Step 1: filter by CGPA ✓
      Step 2: filter by backlogs ✓
      Step 3: sort by package ✓
    If Step 1 output not found → chain broken → flag.
    """

    STEP_INDICATORS = [
        r'step\s*\d+',
        r'first[,\s]', r'second[,\s]', r'third[,\s]',
        r'filtering', r'filter by',
        r'sorted by', r'ranking by',
        r'eligible companies',
        r'therefore', r'thus', r'hence',
        r'final answer',
    ]

    def verify(self, answer: str, query: str) -> tuple[bool, str]:
        """
        Returns (chain_verified, explanation).
        For multi-hop queries, checks that reasoning steps appear in answer.
        """
        if not self._is_multihop(query):
            return True, "Single-hop query — chain verification not required."

        found_steps = []
        for indicator in self.STEP_INDICATORS:
            if re.search(indicator, answer.lower()):
                found_steps.append(indicator.replace(r'\s*', ' ').replace(r'[,\s]', ''))

        if len(found_steps) >= 2:
            return True, f"Chain verified — found indicators: {found_steps[:3]}"
        else:
            return False, (
                "Multi-hop query but answer lacks step-by-step reasoning. "
                "Answer may have skipped eligibility filtering steps."
            )

    def _is_multihop(self, query: str) -> bool:
        """
        Only flag as multi-hop if the query genuinely requires
        combining multiple pieces of information.
        Simple lookups like 'What is Google's CGPA?' are NOT multi-hop.
        """
        q = query.lower()
        # Simple direct lookup patterns — NOT multi-hop
        simple_patterns = [
            r'^what is .+\'?s? (cgpa|package|bond|backlog)',
            r'^what (is|are) the .+ (for|of) \w+',
            r'^(does|do) \w+ allow',
            r'^which (language|technology|tech)',
            r'^how many backlogs',
        ]
        import re
        for pat in simple_patterns:
            if re.search(pat, q):
                return False

    # Genuine multi-hop signals — requires combining 2+ conditions
        multihop_signals = [
        # Eligibility + package combination
        (r'cgpa', r'backlog'),
        (r'cgpa', r'highest|maximum|best'),
        (r'eligible', r'highest|maximum|best|most'),
        # Tech focus + package combination
        (r'python|java|c\+\+', r'highest|package|pay'),
        # Hiring + eligibility combination
        (r'analyst|intern|sde', r'cgpa|eligible|qualify'),
        # Bond + package combination
        (r'bond.free|no bond|zero bond', r'lpa|package|pay'),
        ]

        for sig1, sig2 in multihop_signals:
            if re.search(sig1, q) and re.search(sig2, q):
                return True

        return False


class System2AttentionFilter:
    """
    System 2 Attention (S2A) — regenerates context to remove noise
    before final answer generation.

    Why: irrelevant chunks in context distract the LLM.
    For example, if asked about Google's CGPA and context contains
    10 chunks about other companies, the LLM may confuse them.

    S2A process:
    1. Ask LLM: "Which of these chunks are relevant to the question?"
    2. Keep only chunks the LLM identifies as relevant
    3. Generate final answer with cleaned context

    Like highlighting only the relevant sentences in a textbook
    before answering an exam question.
    """

    def __init__(self, generator):
        self.generator = generator

    def filter_context(self, query: str, chunks: list) -> list:
        """
        Returns subset of chunks that are relevant to the query.
        Falls back to original chunks if filtering fails.
        """
        if len(chunks) <= 2:
            return chunks  # too few to filter

        # Build attention prompt
        chunk_summaries = "\n".join(
            f"Chunk {i+1} [{c.section}|{c.company}]: {c.text[:120]}..."
            for i, c in enumerate(chunks)
        )

        attention_prompt = f"""You are a relevance filter.

Question: {query}

Available chunks:
{chunk_summaries}

List ONLY the chunk numbers that directly help answer this question.
Output format: just comma-separated numbers like: 1, 3, 5
Do not explain. Just the numbers."""

        try:
            result = self.generator.generate(attention_prompt)
            raw    = result.answer.strip()

            # Parse chunk numbers from response
            numbers = [
                int(n.strip())
                for n in re.findall(r'\d+', raw)
                if 1 <= int(n.strip()) <= len(chunks)
            ]

            if not numbers:
                logger.warning("S2A returned no chunks — using original")
                return chunks

            filtered = [chunks[n - 1] for n in sorted(set(numbers))]
            logger.info(
                f"S2A: {len(chunks)} → {len(filtered)} chunks "
                f"(kept: {numbers})"
            )
            return filtered

        except Exception as e:
            logger.warning(f"S2A filter failed: {e} — using original chunks")
            return chunks


class ParametricHallucinationGuard:
    """
    Master guard that orchestrates all four hallucination prevention techniques.

    Usage:
        guard  = ParametricHallucinationGuard(generator)
        answer, report = guard.run(query, prompt, chunks)

    The report tells you exactly which checks passed and failed.
    """

    def __init__(
        self,
        generator,
        enable_self_consistency: bool = True,
        enable_recitation:       bool = True,
        enable_chain_verify:     bool = True,
        enable_s2a:              bool = True,
        consistency_samples:     int  = 2,
        consistency_threshold:   float = 0.4,
        recitation_threshold:    float = 0.5,
    ):
        self.generator  = generator
        self.s2a        = System2AttentionFilter(generator)
        self.sc         = SelfConsistencyChecker(generator, consistency_samples)
        self.recitation = RecitationChecker()
        self.chain      = ChainOfActionsVerifier()

        self.enable_sc         = enable_self_consistency
        self.enable_recitation = enable_recitation
        self.enable_chain      = enable_chain_verify
        self.enable_s2a        = enable_s2a

        self.consistency_threshold = consistency_threshold
        self.recitation_threshold  = recitation_threshold

    def run(
        self,
        query:  str,
        prompt: str,
        chunks: list,
    ) -> tuple[str, HallucinationReport]:
        """
        Full hallucination prevention pipeline.
        Returns (final_answer, report).
        """
        report = HallucinationReport()

        # ── Step 1: System 2 Attention ─────────────────────────────────
        if self.enable_s2a and len(chunks) > 3:
            filtered_chunks = self.s2a.filter_context(query, chunks)
            # Rebuild prompt with filtered chunks if count reduced
            if len(filtered_chunks) < len(chunks):
                from generation.prompt_builder import GroundedPromptBuilder
                from core.interfaces import RetrievalResult
                filtered_result = RetrievalResult(
                    chunks=filtered_chunks,
                    query_used=query,
                )
                prompt = GroundedPromptBuilder().build(query, filtered_result)
                chunks = filtered_chunks
                logger.info(f"S2A reduced context: {len(chunks)} chunks used")

        # ── Step 2: Self-consistency ───────────────────────────────────
        if self.enable_sc:
            answer, sc_score = self.sc.check(prompt)
            report.self_consistency_score = sc_score
            report.consistency_passed     = sc_score >= self.consistency_threshold
            if not report.consistency_passed:
                logger.warning(
                    f"Self-consistency LOW: {sc_score:.2f} "
                    f"(threshold {self.consistency_threshold})"
                )
        else:
            result = self.generator.generate(prompt)
            answer = result.answer

        # ── Step 3: Chain of actions verification ─────────────────────
        if self.enable_chain:
            chain_ok, chain_msg = self.chain.verify(answer, query)
            report.chain_verified = chain_ok
            if not chain_ok:
                logger.warning(f"Chain verification FAILED: {chain_msg}")

        # ── Step 4: Recitation check ───────────────────────────────────
        if self.enable_recitation and chunks:
            rec_score, flagged = self.recitation.check(answer, chunks)
            report.recitation_score  = rec_score
            report.recitation_passed = rec_score >= self.recitation_threshold
            report.flagged_phrases   = flagged

            if not report.recitation_passed:
                logger.warning(
                    f"Recitation FAILED: score={rec_score:.2f}, "
                    f"flagged={flagged}"
                )

        # ── Step 5: Final verdict ──────────────────────────────────────
        failures = []
        if not report.consistency_passed:
            failures.append(
                f"low consistency ({report.self_consistency_score:.2f})"
            )
        if not report.recitation_passed:
            failures.append(
                f"ungrounded claims: {report.flagged_phrases[:2]}"
            )
        if not report.chain_verified:
            failures.append("incomplete reasoning chain")

        if not failures:
            report.final_verdict = "PASS"
            report.explanation   = "All hallucination checks passed."
        elif len(failures) == 1:
            report.final_verdict = "WARN"
            report.explanation   = (
                f"Minor concern: {failures[0]}. "
                "Answer may be partially ungrounded."
            )
        else:
            report.final_verdict = "FAIL"
            report.explanation   = (
                f"Multiple hallucination signals: {'; '.join(failures)}. "
                "Treat this answer with caution."
            )

        return answer, report