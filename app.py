"""
app.py
Streamlit UI with feedback loop integration.
"""
import os
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
from generation.generator import GroqGenerator
from safety.conflict_detector import PlacementConflictDetector
from safety.fallback_guard import FallbackGuard
from safety.overshadow_limiter import OvershadowLimiter
from feedback.loop import FeedbackLoop

logging.basicConfig(level=logging.WARNING)

st.set_page_config(
    page_title="Placement RAG — SVECW",
    page_icon="🎓",
    layout="wide",
)


@st.cache_resource
def load_pipeline():
    embedder = HybridEmbedder(
        faiss_path=os.getenv("FAISS_INDEX_PATH", "data/faiss_index"),
        bm25_path=os.getenv("BM25_PATH", "data/bm25_store.pkl"),
    )
    embedder.load()

    retriever = HybridRetriever(embedder)

    try:
        reranker = CrossEncoderReranker()
    except Exception:
        from core.interfaces import BaseReranker, RetrievalResult
        class PassthroughReranker(BaseReranker):
            def rerank(self, query, result):
                return result
        reranker = PassthroughReranker()

    limiter = OvershadowLimiter()
    refiner = ContextRefiner(limiter)
    prompt_builder = GroundedPromptBuilder()
    generator = GroqGenerator()
    conflict_detector = PlacementConflictDetector()
    fallback_guard = FallbackGuard()
    feedback = FeedbackLoop(limiter)

    pipeline = RAGPipeline(
        retriever, reranker, refiner, prompt_builder,
        generator, conflict_detector, fallback_guard,
        PipelineConfig(top_k_retrieve=20, top_k_rerank=5),
    )
    return pipeline, limiter, feedback


pipeline, limiter, feedback = load_pipeline()

# ── Session state init ────────────────────────────────────
for key, default in {
    "history": [],
    "last_response": None,
    "last_record": None,
    "last_query": "",
    "input_query": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pipeline Config")
    top_k = st.slider("Top-K retrieve", 5, 30, 20)
    top_rerank = st.slider("Top-K after rerank", 1, 10, 5)
    pipeline.config.top_k_retrieve = top_k
    pipeline.config.top_k_rerank = top_rerank

    st.divider()
    st.header("📊 Overshadow Diagnostics")
    diag = limiter.get_diagnostics()
    st.metric("Context cap (chunks)", diag["current_cap"])
    st.metric("Sweet spot", diag["sweet_spot_range"])
    st.metric("Overshadow threshold", f"{diag['overshadow_threshold']} tokens")

    if diag["history"]:
        last = diag["history"][-1]
        risk = last.get("overshadow_risk", 0)
        color = "🟢" if risk < 0.3 else "🟡" if risk < 0.6 else "🔴"
        st.metric("Last risk", f"{color} {risk:.2f}")

    st.divider()
    fb_summary = feedback.summary()
    st.header("🔄 Feedback Loop")
    st.metric("Queries rated", fb_summary["rated"])
    st.metric("Accuracy", f"{fb_summary['accuracy']*100:.0f}%")

    st.divider()
    st.header("💡 Quick Test Queries")
    quick_queries = [
        "What is the CGPA requirement for Google?",
        "Which companies are bond-free with package > 40 LPA?",
        "Is the Amazon CGPA cutoff 6.4 or 7.0?",
        "A student with CGPA 7.6, 1 backlog — highest paying job?",
        "Which company hires the most Interns?",
        "Which company had the most package growth from 2021 to 2024?",
        "What is TCS's campus visit date at SVECW?",
        "I have CGPA 5.0. Where can I apply?",
    ]
    for qq in quick_queries:
        if st.button(qq, use_container_width=True):
            st.session_state.input_query = qq
            st.rerun()

# ── Main area ─────────────────────────────────────────────
st.title("🎓 Placement Intelligence Assistant")
st.caption("SVECW · RAG-ATHON 24 | Hybrid RAG + Groq + AIMD Overshadow Control")

# Text input — value driven by session state so sidebar buttons populate it
query = st.text_input(
    "Ask a placement question:",
    value=st.session_state.input_query,
    placeholder="e.g. Which Python-focused company hires the most interns?",
    key="query_input",
)

ask_clicked = st.button("🔍 Ask", type="primary")

# Run pipeline when Ask is clicked OR when a sidebar button populated the query
if ask_clicked and query.strip():
    st.session_state.input_query = ""   # clear prefill after use

    # Show the question the user asked
    st.markdown(f"**Your question:** {query.strip()}")
    st.markdown("---")

    with st.spinner("Retrieving → Reranking → Generating..."):
        response = pipeline.run(query.strip())

    rec = feedback.record(
        query,
        response.retrieval_quality,
        response.context_tokens,
        limiter.get_diagnostics()["history"][-1].get("overshadow_risk", 0)
        if limiter.get_diagnostics()["history"] else 0,
    )
    st.session_state.last_response = response
    st.session_state.last_record = rec
    st.session_state.last_query = query.strip()
    st.session_state.history.append((query.strip(), response))

# ── Display answer ────────────────────────────────────────
if st.session_state.last_response:
    response = st.session_state.last_response

    # Always show what was asked
    if st.session_state.last_query:
        st.markdown(f"**Q: {st.session_state.last_query}**")

    st.markdown("### Answer")

    if response.fallback_triggered:
        st.warning(
            "⚠️ **Out-of-corpus query** — this information is not available "
            "in the provided placement documents.\n\n" + response.answer
        )
    elif response.conflicts_detected:
        for conflict in response.conflicts_detected:
            st.error(conflict)
        st.markdown(response.answer)
    else:
        st.success(response.answer)

    # Feedback buttons
    col_g, col_b, _ = st.columns([1, 1, 6])
    with col_g:
        if st.button("👍 Good answer"):
            feedback.good(st.session_state.last_record)
            st.toast("Thanks! Context cap increased.", icon="✅")
    with col_b:
        if st.button("👎 Wrong answer"):
            feedback.bad(st.session_state.last_record)
            st.toast("Got it. Context cap reduced.", icon="⚠️")

    # Diagnostics
    with st.expander("📊 Retrieval Diagnostics"):
        d = limiter.get_diagnostics()
        last_hist = d["history"][-1] if d["history"] else {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retrieval quality", f"{response.retrieval_quality:.2f}")
        c2.metric("Context tokens", response.context_tokens)
        c3.metric("Overshadow risk", f"{last_hist.get('overshadow_risk', 0):.2f}")
        c4.metric("Consistency", f"{response.self_consistency_score:.2f}")

        risk_val = last_hist.get("overshadow_risk", 0)
        if risk_val < 0.3:
            st.success("🟢 Context in sweet spot — low hallucination risk.")
        elif risk_val < 0.6:
            st.warning("🟡 Moderate risk — answer may miss nuance.")
        else:
            st.error("🔴 High overshadow risk — consider rating 👎.")

    if response.sources:
        with st.expander("📚 Sources used"):
            for s in response.sources:
                st.code(s)

# ── History ───────────────────────────────────────────────
if len(st.session_state.history) > 1:
    with st.expander(f"🕓 Query history ({len(st.session_state.history)} queries)"):
        for i, (q, r) in enumerate(reversed(st.session_state.history[-10:]), 1):
            st.markdown(f"**Q{i}:** {q}")
            st.markdown(f"↳ {r.answer[:150]}...")
            st.divider()