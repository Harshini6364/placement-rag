"""
core/pipeline.py
RAG pipeline with tool augmentation and full hallucination prevention:
  - System 2 Attention (S2A)
  - Self-consistency
  - Chain of actions verification
  - Recitation checking
"""
import logging
from dataclasses import dataclass, field
from core.interfaces import (
    BaseRetriever, BaseReranker, BaseRefiner,
    BasePromptBuilder, BaseGenerator,
    BaseConflictDetector, BaseFallbackGuard,
    RAGResponse, RetrievalResult,
)
from tools.tool_router import ToolRouter, RouterDecision
from safety.parametric_hallucination_guard import (
    ParametricHallucinationGuard,
    HallucinationReport,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    top_k_retrieve:          int   = 20
    top_k_rerank:            int   = 5
    max_context_tokens:      int   = 2048
    enable_conflict_detection: bool = True
    enable_fallback_guard:   bool  = True
    enable_tools:            bool  = True
    # Hallucination prevention
    enable_self_consistency: bool  = True
    enable_recitation:       bool  = True
    enable_chain_verify:     bool  = True
    enable_s2a:              bool  = True
    consistency_samples:     int   = 2      # keep low to avoid rate limits
    consistency_threshold:   float = 0.4
    recitation_threshold:    float = 0.5


@dataclass
class EnrichedRAGResponse:
    """RAGResponse extended with hallucination report."""
    answer:               str
    sources:              list
    conflicts_detected:   list
    fallback_triggered:   bool
    retrieval_quality:    float
    context_tokens:       int
    self_consistency_score: float = 1.0
    chain_of_thought:     str   = ""
    hallucination_report: HallucinationReport = field(
        default_factory=HallucinationReport
    )


class RAGPipeline:

    def __init__(
        self,
        retriever:        BaseRetriever,
        reranker:         BaseReranker,
        refiner:          BaseRefiner,
        prompt_builder:   BasePromptBuilder,
        generator:        BaseGenerator,
        conflict_detector: BaseConflictDetector,
        fallback_guard:   BaseFallbackGuard,
        config:           PipelineConfig = None,
    ):
        self.retriever         = retriever
        self.reranker          = reranker
        self.refiner           = refiner
        self.prompt_builder    = prompt_builder
        self.generator         = generator
        self.conflict_detector = conflict_detector
        self.fallback_guard    = fallback_guard
        self.config            = config or PipelineConfig()
        self.tool_router       = ToolRouter()

        # Hallucination guard — wraps generator
        self.hallucination_guard = ParametricHallucinationGuard(
            generator=generator,
            enable_self_consistency=self.config.enable_self_consistency,
            enable_recitation=      self.config.enable_recitation,
            enable_chain_verify=    self.config.enable_chain_verify,
            enable_s2a=             self.config.enable_s2a,
            consistency_samples=    self.config.consistency_samples,
            consistency_threshold=  self.config.consistency_threshold,
            recitation_threshold=   self.config.recitation_threshold,
        )

    def run(self, query: str) -> RAGResponse:
        logger.info(f"Pipeline start | query='{query}'")

        # ── Tool routing ───────────────────────────────────────────────
        if self.config.enable_tools:
            decision: RouterDecision = self.tool_router.route(query)

            if decision.needs_tool and decision.result:
                result = decision.result
                if result.success:
                    logger.info(
                        f"Tool '{decision.tool_name}' answered | "
                        f"reason={decision.reason}"
                    )

                    if decision.tool_name in ("calculator", "opinion_guard",
                                              "database"):
                        rag_response = self._run_rag(query)
                        combined = (
                            f"{result.output}\n\n"
                            f"Additional context from documents:\n"
                            f"{rag_response.answer}"
                        )
                        return RAGResponse(
                            answer=combined,
                            sources=[f"tool:{decision.tool_name}"]
                                    + rag_response.sources,
                            conflicts_detected=rag_response.conflicts_detected,
                            fallback_triggered=False,
                            retrieval_quality=rag_response.retrieval_quality,
                            context_tokens=rag_response.context_tokens,
                            self_consistency_score=1.0,
                            chain_of_thought=(
                                f"Tool: {decision.tool_name} | "
                                f"{decision.reason}"
                            ),
                        )
                    else:
                        return RAGResponse(
                            answer=result.output,
                            sources=[result.source_url]
                                    if result.source_url
                                    else [f"tool:{decision.tool_name}"],
                            conflicts_detected=[],
                            fallback_triggered=False,
                            retrieval_quality=1.0,
                            context_tokens=len(result.output.split()),
                            self_consistency_score=1.0,
                            chain_of_thought=(
                                f"Tool: {decision.tool_name} | "
                                f"{decision.reason}"
                            ),
                        )

        return self._run_rag(query)

    def _run_rag(self, query: str) -> RAGResponse:
        # Stage 2: Retrieve
        result: RetrievalResult = self.retriever.retrieve(
            query, self.config.top_k_retrieve
        )

        # Stage 3: Rerank
        result = self.reranker.rerank(query, result)
        result.chunks = result.chunks[: self.config.top_k_rerank]

        # Safety checks
        conflicts = []
        if self.config.enable_conflict_detection:
            conflicts = self.conflict_detector.detect(result.chunks)

        fallback = False
        if self.config.enable_fallback_guard:
            fallback = self.fallback_guard.is_out_of_corpus(
                query, result.chunks
            )

        # Stage 4: Refine
        result = self.refiner.refine(result, self.config.max_context_tokens)

        # Stage 5: Build prompt
        prompt = self.prompt_builder.build(query, result)

        # Stage 6: Generate WITH hallucination prevention
        answer, h_report = self.hallucination_guard.run(
            query=query,
            prompt=prompt,
            chunks=result.chunks,
        )

        # Append hallucination warning to answer if needed
        if h_report.final_verdict == "WARN":
            answer += (
                f"\n\n⚠️ Note: {h_report.explanation}"
            )
        elif h_report.final_verdict == "FAIL":
            answer += (
                f"\n\n🔴 Reliability warning: {h_report.explanation} "
                "Please verify this answer against the source document."
            )

        if fallback:
            answer = (
                "I don't have enough information in the provided "
                "documents to answer this. " + answer
            )

        return RAGResponse(
            answer=answer,
            sources=self._extract_sources(prompt),
            conflicts_detected=conflicts,
            fallback_triggered=fallback,
            retrieval_quality=result.retrieval_quality,
            context_tokens=result.overshadow_risk,
            self_consistency_score=h_report.self_consistency_score,
            chain_of_thought=(
                f"Verdict: {h_report.final_verdict} | "
                f"SC: {h_report.self_consistency_score:.2f} | "
                f"Recitation: {h_report.recitation_score:.2f} | "
                f"Chain: {h_report.chain_verified}"
            ),
        )

    def _extract_sources(self, prompt: str) -> list[str]:
        import re
        return re.findall(r"\[([A-Z]+\s?\|[^\]]+)\]", prompt)[:5]