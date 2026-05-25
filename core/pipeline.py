"""
core/pipeline.py
The common RAG pipeline — software module that wires all 6 stages.
Open/Closed Principle: extend by subclassing stages, not editing this file.
"""
import logging
from dataclasses import dataclass
from core.interfaces import (
    BaseRetriever, BaseReranker, BaseRefiner,
    BasePromptBuilder, BaseGenerator,
    BaseConflictDetector, BaseFallbackGuard,
    RAGResponse, RetrievalResult
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    top_k_retrieve: int = 20
    top_k_rerank: int = 5
    max_context_tokens: int = 2048
    enable_conflict_detection: bool = True
    enable_fallback_guard: bool = True
    self_consistency_samples: int = 3


class RAGPipeline:
    """
    The 6-stage RAG pipeline:
    Rewrite → Retrieve → Rerank → Refine → Insert → Generate

    Dependency Injection: every stage is injected via constructor.
    Low coupling: stages know nothing about each other.
    High cohesion: each stage does exactly one job.
    """

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

    def run(self, query: str) -> RAGResponse:
        logger.info(f"Pipeline start | query='{query}'")

        # Stage 1: Rewrite (handled inside retriever)
        # Stage 2: Retrieve
        result: RetrievalResult = self.retriever.retrieve(
            query, self.config.top_k_retrieve
        )
        logger.info(f"Retrieved {len(result.chunks)} chunks | quality={result.retrieval_quality:.2f}")

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

        # Stage 4: Refine — overshadow limiter
        result = self.refiner.refine(result, self.config.max_context_tokens)

        # Stage 5: Insert — build prompt
        prompt = self.prompt_builder.build(query, result)

        # Stage 6: Generate
        response = self.generator.generate(prompt)
        response.conflicts_detected = conflicts
        response.fallback_triggered = fallback
        response.retrieval_quality = result.retrieval_quality

        if fallback:
            response.answer = (
                "I don't have enough information in the provided documents to answer this query. "
                + response.answer
            )

        logger.info(f"Pipeline done | fallback={fallback} | conflicts={len(conflicts)}")
        return response