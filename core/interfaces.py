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
    """Atomic unit of retrieved knowledge."""
    chunk_id: str
    text: str
    source: str          # 'official' | 'portal' | section name
    company: str = ""
    section: str = ""
    year: int = 0
    conflict: bool = False
    metadata: dict = field(default_factory=dict)
    score: float = 0.0


@dataclass
class RetrievalResult:
    """Output of retrieval stage with quality metrics."""
    chunks: list[Chunk]
    query_used: str
    dense_scores: list[float] = field(default_factory=list)
    sparse_scores: list[float] = field(default_factory=list)
    rerank_scores: list[float] = field(default_factory=list)
    retrieval_quality: float = 0.0   # 0-1 score
    overshadow_risk: float = 0.0     # 0-1 risk of context overflow


@dataclass
class RAGResponse:
    """Final answer with full provenance."""
    answer: str
    sources: list[str]
    conflicts_detected: list[str]
    fallback_triggered: bool
    retrieval_quality: float
    context_tokens: int
    self_consistency_score: float = 0.0
    chain_of_thought: str = ""


class BaseParser(ABC):
    """Parses raw documents into structured content."""
    @abstractmethod
    def parse(self, path: str) -> list[dict]:
        ...


class BaseChunker(ABC):
    """Splits parsed content into Chunk objects."""
    @abstractmethod
    def chunk(self, parsed_sections: list[dict]) -> list[Chunk]:
        ...


class BaseDeduplicator(ABC):
    """Removes near-duplicate chunks before indexing."""
    @abstractmethod
    def deduplicate(self, chunks: list[Chunk]) -> list[Chunk]:
        ...


class BaseEmbedder(ABC):
    """Embeds and stores chunks for retrieval."""
    @abstractmethod
    def index(self, chunks: list[Chunk]) -> None:
        ...

    @abstractmethod
    def search_dense(self, query: str, top_k: int) -> list[Chunk]:
        ...


class BaseRetriever(ABC):
    """Hybrid retriever — dense + sparse."""
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> RetrievalResult:
        ...


class BaseReranker(ABC):
    """Reranks retrieved chunks with a cross-encoder."""
    @abstractmethod
    def rerank(self, query: str, result: RetrievalResult) -> RetrievalResult:
        ...


class BaseRefiner(ABC):
    """Prunes context to prevent overshadowing."""
    @abstractmethod
    def refine(self, result: RetrievalResult, max_tokens: int) -> RetrievalResult:
        ...


class BasePromptBuilder(ABC):
    """Builds the final grounded prompt."""
    @abstractmethod
    def build(self, query: str, result: RetrievalResult) -> str:
        ...


class BaseGenerator(ABC):
    """Generates the grounded answer."""
    @abstractmethod
    def generate(self, prompt: str) -> RAGResponse:
        ...


class BaseConflictDetector(ABC):
    """Detects conflicting information across chunks."""
    @abstractmethod
    def detect(self, chunks: list[Chunk]) -> list[str]:
        ...


class BaseFallbackGuard(ABC):
    """Determines if query is out-of-corpus."""
    @abstractmethod
    def is_out_of_corpus(self, query: str, chunks: list[Chunk]) -> bool:
        ...


class BaseEvaluator(ABC):
    """Evaluates RAG system on benchmark queries."""
    @abstractmethod
    def evaluate(self, queries: list[dict]) -> dict:
        ...