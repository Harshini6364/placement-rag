"""
core/interfaces.py
Abstract base classes for every RAG stage.
Single Responsibility + Dependency Inversion (SOLID).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    company: str = ""
    section: str = ""
    year: int = 0
    conflict: bool = False
    metadata: dict = field(default_factory=dict)
    score: float = 0.0


@dataclass
class RetrievalResult:
    chunks: list[Chunk]
    query_used: str
    dense_scores: list[float] = field(default_factory=list)
    sparse_scores: list[float] = field(default_factory=list)
    rerank_scores: list[float] = field(default_factory=list)
    retrieval_quality: float = 0.0
    overshadow_risk: float = 0.0


@dataclass
class RAGResponse:
    answer: str
    sources: list[str]
    conflicts_detected: list[str]
    fallback_triggered: bool
    retrieval_quality: float
    context_tokens: int
    self_consistency_score: float = 0.0
    chain_of_thought: str = ""


class BaseParser(ABC):
    @abstractmethod
    def parse(self, path: str) -> list[dict]:
        ...


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, parsed_sections: list[dict]) -> list[Chunk]:
        ...


class BaseDeduplicator(ABC):
    @abstractmethod
    def deduplicate(self, chunks: list[Chunk]) -> list[Chunk]:
        ...


class BaseEmbedder(ABC):
    @abstractmethod
    def build_index(self, chunks: list[Chunk]) -> None:   # renamed from index
        ...

    @abstractmethod
    def search_dense(self, query: str, top_k: int) -> list[Chunk]:
        ...


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> RetrievalResult:
        ...


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, result: RetrievalResult) -> RetrievalResult:
        ...


class BaseRefiner(ABC):
    @abstractmethod
    def refine(self, result: RetrievalResult, max_tokens: int) -> RetrievalResult:
        ...


class BasePromptBuilder(ABC):
    @abstractmethod
    def build(self, query: str, result: RetrievalResult) -> str:
        ...


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> RAGResponse:
        ...


class BaseConflictDetector(ABC):
    @abstractmethod
    def detect(self, chunks: list[Chunk]) -> list[str]:
        ...


class BaseFallbackGuard(ABC):
    @abstractmethod
    def is_out_of_corpus(self, query: str, chunks: list[Chunk]) -> bool:
        ...


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, queries: list[dict]) -> dict:
        ...