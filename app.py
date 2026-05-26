"""
app.py
Streamlit UI — persistent sidebar history (ChatGPT style),
response time shown instead of cache labels,
history preserved across reruns.
"""
import os
import time
import hashlib
import logging
import json
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

# ── Persistent storage path ───────────────────────────────────────────────────
HISTORY_FILE = "data/chat_history.json"


def load_history() -> list[dict]:
    """Load chat history from disk — persists across page reloads."""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def save_history(history: list[dict]):
    """Save chat history to disk."""
    try:
        os.makedirs("data", exist_ok=True)
        with open(HISTORY_FILE, "w") as f:
            json.dump(history[-50:], f, indent=2)  # keep last 50
    except Exception:
        pass


# ── Pipeline loader ───────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def cached_pipeline_run(query: str, top_k: int, top_rerank: int):
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
            def rerank(self, q, r): return r
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
st.session_state._pipeline = pipeline

# ── Session state ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_history()   # load from disk on first run
if "text_input_value" not in st.session_state:
    st.session_state.text_input_value = ""
if "trigger_query" not in st.session_state:
    st.session_state.trigger_query = ""
if "seen_keys" not in st.session_state:
    st.session_state.seen_keys = set()
if "active_index" not in st.session_state:
    st.session_state.active_index = None   # which history item is shown in main area

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Placement RAG")
    st.caption("SVECW · RAG-ATHON 24")

    st.divider()
    st.markdown("### ⚙️ Config")
    top_k = st.slider("Retrieve top-K", 5, 30, 20)
    top_rerank = st.slider("Rerank top-K", 1, 10, 5)

    st.divider()
    st.markdown("### 💡 Quick Queries")
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
        "Which company grew the most from 2021 to 2024?",
        "List all companies that allow at least 2 backlogs.",
    ]
    for qq in quick_queries:
        if st.button(qq, use_container_width=True, key=f"q_{qq[:25]}"):
            st.session_state.text_input_value = qq
            st.session_state.trigger_query = qq
            st.rerun()

    st.divider()

    # ── Chat history (ChatGPT style) ──────────────────────────────────────────
    history = st.session_state.chat_history
    if history:
        st.markdown("### 🕓 History")

        col_clear, col_export = st.columns(2)
        with col_clear:
            if st.button("🗑 Clear all", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.active_index = None
                save_history([])
                st.rerun()
        with col_export:
            export_text = "\n\n".join(
                f"Q: {h['query']}\nA: {h['answer']}\nTime: {h['response_time']}s"
                for h in history
            )
            st.download_button(
                "⬇ Export",
                export_text,
                file_name="placement_rag_history.txt",
                use_container_width=True,
            )

        st.markdown("")

        # Show each history item as a clickable button
        for i, item in enumerate(reversed(history)):
            idx = len(history) - 1 - i   # actual index in list
            is_active = (st.session_state.active_index == idx)

            # Truncate query for display
            display_q = item["query"][:42] + "..." if len(item["query"]) > 42 else item["query"]
            response_time = item.get("response_time", "?")

            # Highlight active item
            label = f"{'▶ ' if is_active else ''}{display_q}"
            if st.button(
                label,
                key=f"hist_{idx}",
                use_container_width=True,
                help=f"Response time: {response_time}s | Click to view",
            ):
                st.session_state.active_index = idx
                st.rerun()

            # Show response time below each entry
            st.caption(f"⏱ {response_time}s")
    else:
        st.caption("No history yet. Ask a question to get started.")

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🎓 Placement Intelligence Assistant")
st.caption("SVECW · RAG-ATHON 24 | Hybrid RAG + Groq + Vision + Temporal Reasoning")

# Text input
query = st.text_input(
    "Ask a placement question:",
    value=st.session_state.text_input_value,
    placeholder="e.g. Which company grew the most from 2021 to 2024?",
)
st.session_state.text_input_value = query

col_ask, col_clear = st.columns([1, 5])
with col_ask:
    ask_clicked = st.button("🔍 Ask", type="primary", use_container_width=True)
with col_clear:
    if st.button("✖ Clear"):
        st.session_state.text_input_value = ""
        st.session_state.active_index = None
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
    cache_key = hashlib.md5(
        f"{run_query}_{top_k}_{top_rerank}".encode()
    ).hexdigest()
    is_cached = cache_key in st.session_state.seen_keys
    st.session_state.seen_keys.add(cache_key)

    start_time = time.time()
    with st.spinner("Thinking..."):
        result = cached_pipeline_run(run_query, top_k, top_rerank)
    elapsed = round(time.time() - start_time, 2)

    # Save to history
    history_item = {
        "query": run_query,
        "answer": result["answer"],
        "sources": result["sources"],
        "conflicts_detected": result["conflicts_detected"],
        "fallback_triggered": result["fallback_triggered"],
        "retrieval_quality": result["retrieval_quality"],
        "context_tokens": result["context_tokens"],
        "self_consistency_score": result["self_consistency_score"],
        "response_time": elapsed,   # actual wall time — cached hits show ~0.0s
    }
    st.session_state.chat_history.append(history_item)
    save_history(st.session_state.chat_history)

    # Point active view to this new item
    st.session_state.active_index = len(st.session_state.chat_history) - 1
    st.session_state.text_input_value = ""
    st.rerun()

# ── Display: all history in main area (newest at top) ────────────────────────
history = st.session_state.chat_history

if not history:
    st.info("Ask a question above or click a quick query from the sidebar.")
else:
    # If a sidebar history item was clicked, show only that one highlighted
    active_idx = st.session_state.active_index

    # Always show ALL questions — newest first
    for i, item in enumerate(reversed(history)):
        idx = len(history) - 1 - i
        is_active = (active_idx == idx)

        # Container with border highlight for active item
        with st.container(border=True):
            # Header row: question + response time
            col_q, col_t = st.columns([5, 1])
            with col_q:
                st.markdown(f"**🙋 Q: {item['query']}**")
            with col_t:
                rt = item.get("response_time", "?")
                # Response time naturally shows cache effect:
                # first time = 1-3s, repeated = ~0.0s
                if isinstance(rt, float) and rt < 0.1:
                    st.markdown(f"**⚡ {rt}s**")
                else:
                    st.markdown(f"⏱ {rt}s")

            # Answer
            if item["fallback_triggered"]:
                st.warning(
                    "⚠️ Out-of-corpus — not in placement documents.\n\n"
                    + item["answer"]
                )
            elif item["conflicts_detected"]:
                for conflict in item["conflicts_detected"]:
                    st.error(conflict)
                st.markdown(item["answer"])
            else:
                st.success(item["answer"])

            # Diagnostics (collapsed by default)
            with st.expander("📊 Diagnostics", expanded=False):
                c1, c2, c3 = st.columns(3)
                c1.metric("Retrieval quality", f"{item['retrieval_quality']:.2f}")
                c2.metric("Context tokens", item["context_tokens"])
                c3.metric("Consistency", f"{item['self_consistency_score']:.2f}")

            if item["sources"]:
                with st.expander("📚 Sources", expanded=False):
                    for s in item["sources"]:
                        st.code(s)

        # Divider between items
        if i < len(history) - 1:
            st.markdown("")