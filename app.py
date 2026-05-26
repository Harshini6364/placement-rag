"""
app.py
Streamlit UI with caching. Feedback loop and overshadow limiter
run silently in the background (removed from UI per requirements).
"""
import os
import hashlib
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


@st.cache_data(ttl=3600, show_spinner=False)
def cached_pipeline_run(query: str, top_k: int, top_rerank: int):
    """
    Cache answers by (query, top_k, top_rerank).
    Same question = instant reply, no API call. TTL = 1 hour.
    """
    pipeline = st.session_state._pipeline
    pipeline.config.top_k_retrieve = top_k
    pipeline.config.top_k_rerank = top_rerank
    response = pipeline.run(query)
    return {
        "answer": response.answer,
        "sources": response.sources,
        "conflicts_detected": response.conflicts_detected,
        "fallback_triggered": response.fallback_triggered,
        "retrieval_quality": response.retrieval_quality,
        "context_tokens": response.context_tokens,
        "self_consistency_score": response.self_consistency_score,
    }


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
        from core.interfaces import BaseReranker
        class PassthroughReranker(BaseReranker):
            def rerank(self, query, result):
                return result
        reranker = PassthroughReranker()

    limiter = OvershadowLimiter()        # active in background
    refiner = ContextRefiner(limiter)
    prompt_builder = GroundedPromptBuilder()
    generator = GroqGenerator()
    conflict_detector = PlacementConflictDetector()
    fallback_guard = FallbackGuard()
    feedback = FeedbackLoop(limiter)     # active in background

    pipeline = RAGPipeline(
        retriever, reranker, refiner, prompt_builder,
        generator, conflict_detector, fallback_guard,
        PipelineConfig(top_k_retrieve=20, top_k_rerank=5),
    )
    return pipeline, limiter, feedback


pipeline, limiter, feedback = load_pipeline()
st.session_state._pipeline = pipeline

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "history": [],
    "last_response": None,
    "last_record": None,
    "last_query": "",
    "text_input_value": "",
    "trigger_query": "",
    "cache_hits": 0,
    "total_queries": 0,
    "seen_keys": set(),
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pipeline Config")
    top_k = st.slider("Top-K retrieve", 5, 30, 20)
    top_rerank = st.slider("Top-K after rerank", 1, 10, 5)

    st.divider()
    st.header("⚡ Cache Stats")
    total = st.session_state.total_queries
    hits = st.session_state.cache_hits
    misses = total - hits
    st.metric("Total queries", total)
    st.metric("Cache hits ⚡", hits)
    st.metric("API calls made", misses)
    if total > 0:
        st.progress(hits / total, text=f"Hit rate: {hits/total*100:.0f}%")
    if st.button("🗑️ Clear cache", use_container_width=True):
        cached_pipeline_run.clear()
        st.session_state.cache_hits = 0
        st.session_state.total_queries = 0
        st.session_state.seen_keys = set()
        st.toast("Cache cleared!", icon="🗑️")

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
        "Which Python-focused company hires the most Interns?",
        "Compare Google and Amazon on all dimensions.",
    ]
    for qq in quick_queries:
        if st.button(qq, use_container_width=True, key=f"quick_{qq[:30]}"):
            st.session_state.text_input_value = qq
            st.session_state.trigger_query = qq
            st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🎓 Placement Intelligence Assistant")
st.caption("SVECW · RAG-ATHON 24 | Hybrid RAG + Groq + Vision Chart Support")

query = st.text_input(
    "Ask a placement question:",
    value=st.session_state.text_input_value,
    placeholder="e.g. Which Python-focused company hires the most interns?",
)
st.session_state.text_input_value = query

col_ask, col_clear = st.columns([1, 5])
with col_ask:
    ask_clicked = st.button("🔍 Ask", type="primary", use_container_width=True)
with col_clear:
    if st.button("✖ Clear"):
        st.session_state.text_input_value = ""
        st.session_state.last_response = None
        st.session_state.last_query = ""
        st.rerun()

# ── Determine query to run ────────────────────────────────────────────────────
run_query = None
if ask_clicked and query.strip():
    run_query = query.strip()
if st.session_state.trigger_query:
    run_query = st.session_state.trigger_query
    st.session_state.trigger_query = ""

# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_query:
    st.session_state.total_queries += 1

    cache_key = hashlib.md5(
        f"{run_query}_{top_k}_{top_rerank}".encode()
    ).hexdigest()
    already_cached = cache_key in st.session_state.seen_keys
    if already_cached:
        st.session_state.cache_hits += 1
    st.session_state.seen_keys.add(cache_key)

    spinner_msg = (
        "⚡ Returning cached answer..."
        if already_cached
        else "Retrieving → Reranking → Generating..."
    )
    with st.spinner(spinner_msg):
        cached_result = cached_pipeline_run(run_query, top_k, top_rerank)

    # Feedback loop records silently (no UI shown)
    rec = feedback.record(
        run_query,
        cached_result["retrieval_quality"],
        cached_result["context_tokens"],
        limiter.get_diagnostics()["history"][-1].get("overshadow_risk", 0)
        if limiter.get_diagnostics()["history"] else 0,
    )

    st.session_state.last_response = cached_result
    st.session_state.last_record = rec
    st.session_state.last_query = run_query
    st.session_state.history.append((run_query, cached_result))

# ── Display answer ────────────────────────────────────────────────────────────
if st.session_state.last_response:
    response = st.session_state.last_response

    st.markdown("---")
    st.markdown("### 🙋 Your question")
    st.info(st.session_state.last_query)

    st.markdown("### 💬 Answer")

    if response["fallback_triggered"]:
        st.warning(
            "⚠️ **Out-of-corpus** — this information is not in the placement documents.\n\n"
            + response["answer"]
        )
    elif response["conflicts_detected"]:
        for conflict in response["conflicts_detected"]:
            st.error(conflict)
        st.markdown(response["answer"])
    else:
        st.success(response["answer"])

    # Retrieval diagnostics (kept, shown on demand)
    with st.expander("📊 Retrieval Diagnostics"):
        d = limiter.get_diagnostics()          # overshadow limiter runs silently
        last_hist = d["history"][-1] if d["history"] else {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retrieval quality", f"{response['retrieval_quality']:.2f}")
        c2.metric("Context tokens", response["context_tokens"])
        c3.metric("Overshadow risk", f"{last_hist.get('overshadow_risk', 0):.2f}")
        c4.metric("Consistency", f"{response['self_consistency_score']:.2f}")

        risk_val = last_hist.get("overshadow_risk", 0)
        if risk_val < 0.3:
            st.success("🟢 Sweet spot — low hallucination risk.")
        elif risk_val < 0.6:
            st.warning("🟡 Moderate risk.")
        else:
            st.error("🔴 High overshadow risk.")

    if response["sources"]:
        with st.expander("📚 Sources used"):
            for s in response["sources"]:
                st.code(s)

# ── History ───────────────────────────────────────────────────────────────────
if len(st.session_state.history) > 1:
    with st.expander(f"🕓 History ({len(st.session_state.history)} queries)"):
        for i, (q, r) in enumerate(reversed(st.session_state.history[-10:]), 1):
            cached_tag = "⚡" if i > 1 else ""
            st.markdown(f"**Q{i} {cached_tag}:** {q}")
            ans = r["answer"] if isinstance(r, dict) else r.answer
            st.markdown(f"↳ {ans[:150]}...")
            st.divider()