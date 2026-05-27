"""
core/pipeline.py
RAG pipeline with tool augmentation.
Tool router runs first — if a tool handles the query, RAG is skipped or supplemented.
"""
import logging
from dataclasses import dataclass
from core.interfaces import (
    BaseRetriever, BaseReranker, BaseRefiner,
    BasePromptBuilder, BaseGenerator,
    BaseConflictDetector, BaseFallbackGuard,
    RAGResponse, RetrievalResult,
)
from tools.tool_router import ToolRouter, RouterDecision

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    top_k_retrieve: int = 20
    top_k_rerank: int = 5
    max_context_tokens: int = 2048
    enable_conflict_detection: bool = True
    enable_fallback_guard: bool = True
    self_consistency_samples: int = 1
    enable_tools: bool = True


class RAGPipeline:

    def __init__(
        self,
        retriever: BaseRetriever,
        reranker: BaseReranker,
        refiner: BaseRefiner,
        prompt_builder: BasePromptBuilder,
        generator: BaseGenerator,
        conflict_detector: BaseConflictDetector,
        fallback_guard: BaseFallbackGuard,
        config: PipelineConfig = None,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.refiner = refiner
        self.prompt_builder = prompt_builder
        self.generator = generator
        self.conflict_detector = conflict_detector
        self.fallback_guard = fallback_guard
        self.config = config or PipelineConfig()
        self.tool_router = ToolRouter()

    def run(self, query: str) -> RAGResponse:
        logger.info(f"Pipeline start | query='{query}'")

        # ── Tool routing (agent step) ──────────────────────────────────────
        if self.config.enable_tools:
            decision: RouterDecision = self.tool_router.route(query)

            if decision.needs_tool and decision.result:
                result = decision.result
                if result.success:
                    # Tool answered — skip RAG or supplement
                    logger.info(f"Tool '{decision.tool_name}' answered | reason={decision.reason}")

                    # For calculator/opinion, RAG adds context if available
                    if decision.tool_name in ("calculator", "opinion_guard"):
                        rag_response = self._run_rag(query)
                        combined_answer = (
                            f"{result.output}\n\n"
                            f"Additional context from placement documents:\n{rag_response.answer}"
                        )
                        return RAGResponse(
                            answer=combined_answer,
                            sources=[f"tool:{decision.tool_name}"] + rag_response.sources,
                            conflicts_detected=rag_response.conflicts_detected,
                            fallback_triggered=False,
                            retrieval_quality=rag_response.retrieval_quality,
                            context_tokens=rag_response.context_tokens,
                            self_consistency_score=1.0,
                            chain_of_thought=f"Tool used: {decision.tool_name} | {decision.reason}",
                        )
                    else:
                        # Pure tool answer (web search, date)
                        return RAGResponse(
                            answer=result.output,
                            sources=[result.source_url] if result.source_url else [f"tool:{decision.tool_name}"],
                            conflicts_detected=[],
                            fallback_triggered=False,
                            retrieval_quality=1.0,
                            context_tokens=len(result.output.split()),
                            self_consistency_score=1.0,
                            chain_of_thought=f"Tool used: {decision.tool_name} | {decision.reason}",
                        )

        # ── RAG pipeline (in-corpus queries) ──────────────────────────────
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
            fallback = self.fallback_guard.is_out_of_corpus(query, result.chunks)

        # Stage 4: Refine
        result = self.refiner.refine(result, self.config.max_context_tokens)

        # Stage 5: Build prompt
        prompt = self.prompt_builder.build(query, result)

        # Stage 6: Generate
        response = self.generator.generate(prompt)
        response.conflicts_detected = conflicts
        response.fallback_triggered = fallback
        response.retrieval_quality = result.retrieval_quality

        if fallback:
            response.answer = (
                "I don't have enough information in the provided documents to answer this. "
                + response.answer
            )

        return response