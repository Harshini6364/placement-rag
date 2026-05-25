"""
app.py
Streamlit UI — the face of your RAG system.
"""
import os
import sys
import logging
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from core.pipeline import RAGPipeline, PipelineConfig
from ingestion.embedder import HybridEmbedder
from retrieval.retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker
from generation.refiner import ContextRefiner
from generation.prompt_builder import GroundedPromptBuilder
from generation.generator import GeminiGenerator
from safety.conflict_detector import PlacementConflictDetector
from safety.fallback_guard import FallbackGuard
from safety.overshadow_limiter import OvershadowLimiter

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Placement RAG", page_icon="🎓", layout="wide")
st.title("🎓 Placement Intelligence Assistant")
st.caption("SVECW · RAG-ATHON 24 | Powered by Gemini + Hybrid RAG")


@st.cache_resource
def load_pipeline():
    embedder = HybridEmbedder(
        faiss_path=os.getenv("FAISS_INDEX_PATH", "data/faiss_index"),
        bm25_path=os.getenv("BM25_PATH", "data/bm25_store.pkl"),
    )
    embedder.load()
    retriever = HybridRetriever(embedder)
    reranker = CrossEncoderReranker()
    limiter = OvershadowLimiter()
    refiner = ContextRefiner(limiter)
    prompt_builder = GroundedPromptBuilder()
    generator = GeminiGenerator()
    conflict_detector = PlacementConflictDetector()
    fallback_guard = FallbackGuard()
    pipeline = RAGPipeline(
        retriever, reranker, refiner, prompt_builder,
        generator, conflict_detector, fallback_guard,
        PipelineConfig(top_k_retrieve=20, top_k_rerank=5),
    )
    return pipeline, limiter


pipeline, limiter = load_pipeline()

query = st.text_input("Ask a placement question:", placeholder="e.g. Which company pays the most for Python developers?")

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Ask", type="primary") and query:
        with st.spinner("Thinking..."):
            response = pipeline.run(query)

        st.markdown("### Answer")
        if response.fallback_triggered:
            st.warning(response.answer)
        elif response.conflicts_detected:
            st.error("\n".join(response.conflicts_detected))
            st.markdown(response.answer)
        else:
            st.success(response.answer)

        with st.expander("📊 Retrieval Diagnostics"):
            diag = limiter.get_diagnostics()
            st.json({
                "retrieval_quality": response.retrieval_quality,
                "context_tokens": response.context_tokens,
                "consistency_score": response.self_consistency_score,
                "overshadow_risk": diag.get("history", [{}])[-1].get("overshadow_risk", "N/A"),
                "context_cap": diag["current_cap"],
                "sweet_spot": diag["sweet_spot_range"],
            })

        if response.sources:
            with st.expander("📚 Sources used"):
                for s in response.sources:
                    st.markdown(f"- `{s}`")

with col2:
    st.markdown("### Quick test queries")
    quick = [
        "What is the CGPA requirement for Google?",
        "Which companies are bond-free with package > 40 LPA?",
        "Is the Amazon CGPA cutoff 6.4 or 7.0?",
        "A student with CGPA 7.6, 1 backlog wants the highest paying job",
        "What is TCS's campus visit date?",
    ]
    for q_ex in quick:
        if st.button(q_ex, key=q_ex):
            st.session_state["_q"] = q_ex