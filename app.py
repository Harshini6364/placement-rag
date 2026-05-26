"""
app.py
Streamlit UI — ChatGPT-style layout:
  • Sidebar: New Chat + session history only
  • Main: scrollable chat (oldest top → newest bottom), sticky input bar at bottom
  • No background color on answer bubbles — plain text like ChatGPT
  • Input clears automatically after answer
  • Response time shown after the answer (under assistant bubble)
  • Sources shown in small monospace font
  • Enter key submits the query
"""
import os
import time
import hashlib
import logging
import json
import uuid
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

import re

def clean_answer(text: str) -> str:
    """
    Strip chain-of-thought reasoning (Step N:, numbered headers, MULTI-HOP preamble)
    and return only the final user-facing answer.
    """
    # Remove "To answer this question, I will follow..." preamble lines
    text = re.sub(r"(?im)^.*?(follow|using|apply).{0,60}(step|query|multi.hop).*$\n?", "", text)

    # Split on common final-answer signals and take everything after
    final_markers = [
        r"(?im)^#+\s*(final answer|answer|result|conclusion)[:\s]*$",
        r"(?im)\*\*(final answer|answer|result)[:\*\s]+\*\*",
        r"(?im)^(final answer|in summary|in conclusion|therefore|so,)[:\s]",
    ]
    for marker in final_markers:
        parts = re.split(marker, text, maxsplit=1)
        if len(parts) > 1:
            text = parts[-1].strip()
            break

    # If still has Step N: blocks, remove all of them and keep everything after the last one
    step_pattern = re.compile(r"(?im)^(step\s*\d+[:\.\)].*?)(?=step\s*\d+[:\.\)]|\Z)", re.DOTALL)
    steps = list(step_pattern.finditer(text))
    if steps:
        # Everything after the last step block
        last_end = steps[-1].end()
        remainder = text[last_end:].strip()
        if remainder:
            text = remainder
        else:
            # No trailing content — just remove step headers, keep bullet content
            text = re.sub(r"(?im)^step\s*\d+[:\.\)][^\n]*\n?", "", text)

    # Remove bold markdown step headers like **Step 1: ...**
    text = re.sub(r"\*\*step\s*\d+[:\.\)][^\*]*\*\*\n?", "", text, flags=re.IGNORECASE)

    # Remove "Assuming the student..." lines that are reasoning artifacts
    text = re.sub(r"(?im)^assuming\b.*$\n?", "", text)

    return text.strip()

st.set_page_config(
    page_title="Placement RAG — SVECW",
    page_icon="🎓",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer    {visibility: hidden;}

/* Main content: leave room for sticky input */
.block-container {
    padding-bottom: 110px !important;
    padding-top: 1.2rem !important;
    max-width: 820px !important;
    margin: 0 auto !important;
}

/* Sidebar */
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem;
}

/* ── User bubble (right) ── */
.chat-q {
    display: flex;
    justify-content: flex-end;
    margin: 10px 0 2px 0;
}
.chat-q-bubble {
    background: #2e5ff6;
    color: #fff;
    border-radius: 18px 18px 4px 18px;
    padding: 10px 15px;
    max-width: 75%;
    font-size: 0.95rem;
    line-height: 1.5;
    word-wrap: break-word;
}

/* ── Assistant answer — green box ── */
.chat-a {
    display: flex;
    justify-content: flex-start;
    margin: 4px 0 0 0;
}
.chat-a-text {
    max-width: 92%;
    font-size: 0.95rem;
    line-height: 1.7;
    word-wrap: break-word;
    white-space: pre-wrap;
    color: #1a3a1a;
    background: #eafbea;
    border: 1px solid #b2e0b2;
    border-radius: 10px;
    padding: 12px 16px;
}
.chat-a-text.warning-text {
    background: #fffbe6;
    border-color: #f0a500;
    color: #5a3e00;
}
.chat-a-text.error-text {
    background: #fff0f0;
    border-color: #e05252;
    color: #5a0000;
}

/* ── Response time chip — normal readable size ── */
.rt-chip {
    font-size: 0.88rem;
    color: #888;
    margin: 2px 0 10px 4px;
    display: block;
}

/* ── Sources expander: smaller font ── */
.sources-block {
    font-size: 0.75rem !important;
    font-family: monospace;
    color: #aaa;
    white-space: pre-wrap;
    word-break: break-all;
    line-height: 1.4;
}

/* Scroll anchor */
#chat-bottom { height: 1px; }
</style>
""", unsafe_allow_html=True)

# ── Persistent storage ────────────────────────────────────────────────────────
HISTORY_FILE = "data/chat_history.json"


def load_all_sessions() -> list[dict]:
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                data = json.load(f)
            if not isinstance(data, list) or len(data) == 0:
                return []
            if isinstance(data[0], dict) and "query" in data[0] and "messages" not in data[0]:
                title = data[0].get("query", "Imported chat")
                title = (title[:45] + "...") if len(title) > 45 else title
                return [{
                    "id": str(uuid.uuid4()),
                    "title": f"[Imported] {title}",
                    "messages": data,
                }]
            return data
    except Exception:
        pass
    return []


def save_all_sessions(sessions: list[dict]):
    try:
        os.makedirs("data", exist_ok=True)
        with open(HISTORY_FILE, "w") as f:
            json.dump(sessions[-30:], f, indent=2)
    except Exception:
        pass


# ── Pipeline ──────────────────────────────────────────────────────────────────
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
if "all_sessions" not in st.session_state:
    st.session_state.all_sessions = load_all_sessions()

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())

if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""

if "input_key" not in st.session_state:
    st.session_state.input_key = 0

TOP_K      = 20
TOP_RERANK = 5


def get_current_session() -> dict | None:
    for s in st.session_state.all_sessions:
        if s["id"] == st.session_state.current_session_id:
            return s
    return None


def start_new_chat():
    st.session_state.current_session_id = str(uuid.uuid4())
    st.session_state.pending_query = ""
    st.session_state.input_key += 1


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Placement RAG")
    st.caption("SVECW · RAG-ATHON 24")

    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        start_new_chat()
        st.rerun()

    st.divider()

    all_sessions = st.session_state.all_sessions
    if all_sessions:
        st.markdown("### 🕓 Chats")

        col_clear, col_export = st.columns(2)
        with col_clear:
            if st.button("🗑 Clear all", use_container_width=True):
                st.session_state.all_sessions = []
                save_all_sessions([])
                start_new_chat()
                st.rerun()
        with col_export:
            export_lines = []
            for sess in all_sessions:
                export_lines.append(f"=== {sess.get('title', 'Chat')} ===")
                for msg in sess.get("messages", []):
                    export_lines.append(f"Q: {msg.get('query', '')}")
                    export_lines.append(f"A: {msg.get('answer', '')}")
                    export_lines.append(f"Time: {msg.get('response_time', '?')}s")
                    export_lines.append("")
            st.download_button(
                "⬇ Export",
                "\n".join(export_lines),
                file_name="placement_rag_history.txt",
                use_container_width=True,
            )

        st.markdown("")

        for sess in reversed(all_sessions):
            is_active = (sess["id"] == st.session_state.current_session_id)
            label = f"{'▶ ' if is_active else ''}{sess.get('title', 'Chat')}"
            if st.button(
                label,
                key=f"sess_{sess['id']}",
                use_container_width=True,
                help=f"{len(sess.get('messages', []))} question(s)",
            ):
                st.session_state.current_session_id = sess["id"]
                st.session_state.pending_query = ""
                st.rerun()
    else:
        st.caption("No previous chats yet.")

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style='text-align:center;margin-bottom:2px;'>🎓 Placement Intelligence Assistant</h2>"
    "<p style='text-align:center;color:#888;font-size:0.83rem;margin-bottom:1rem;'>"
    "SVECW · RAG-ATHON 24 &nbsp;|&nbsp; Hybrid RAG + Groq + Vision + Temporal Reasoning"
    "</p>",
    unsafe_allow_html=True,
)

# ── Chat messages ─────────────────────────────────────────────────────────────
current_session = get_current_session()

if not current_session or not current_session.get("messages"):
    st.markdown(
        "<div style='text-align:center;color:#555;margin-top:80px;font-size:1rem;'>"
        "Ask anything about placements at SVECW ↓"
        "</div>",
        unsafe_allow_html=True,
    )
else:
    for item in current_session["messages"]:

        q_html = item["query"].replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(
            f'<div class="chat-q"><div class="chat-q-bubble">{q_html}</div></div>',
            unsafe_allow_html=True,
        )

        answer_text = clean_answer(item["answer"]).replace("<", "&lt;").replace(">", "&gt;")

        if item.get("fallback_triggered"):
            css_class = "chat-a-text warning-text"
            answer_text = "⚠️ Out-of-corpus — not in placement documents.\n\n" + answer_text
        elif item.get("conflicts_detected"):
            css_class = "chat-a-text error-text"
            conflicts_html = "\n".join(item["conflicts_detected"]).replace("<","&lt;").replace(">","&gt;")
            answer_text = f"⚠️ Conflicts:\n{conflicts_html}\n\n{answer_text}"
        else:
            css_class = "chat-a-text"

        st.markdown(
            f'<div class="chat-a"><div class="{css_class}">{answer_text}</div></div>',
            unsafe_allow_html=True,
        )

        rt = item.get("response_time", "?")
        if isinstance(rt, float) and rt < 0.1:
            rt_label = f"⚡ {rt}s (cached)"
        else:
            rt_label = f"⏱ {rt}s"
        st.markdown(
            f'<span class="rt-chip">{rt_label}</span>',
            unsafe_allow_html=True,
        )

        if item.get("sources"):
            with st.expander("📚 Sources", expanded=False):
                sources_text = "\n".join(item["sources"])
                st.markdown(
                    f'<div class="sources-block">{sources_text}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='margin-bottom:6px'></div>", unsafe_allow_html=True)

st.markdown('<div id="chat-bottom"></div>', unsafe_allow_html=True)
st.markdown("""
<script>
(function() {
    var el = document.getElementById('chat-bottom');
    if (el) el.scrollIntoView({ behavior: 'smooth' });
})();
</script>
""", unsafe_allow_html=True)

# ── Input bar (bottom) ────────────────────────────────────────────────────────
st.markdown("---")
col_input, col_send = st.columns([6, 1])

# Use a dynamic key so we can reset the widget by incrementing input_key
with col_input:
    query = st.text_input(
        label="input",
        label_visibility="collapsed",
        placeholder="Ask a placement question…",
        key=f"chat_input_{st.session_state.input_key}",
    )
with col_send:
    ask_clicked = st.button("Send ➤", type="primary", use_container_width=True)

# ── Determine what to run ─────────────────────────────────────────────────────
# Enter submits: text_input returns the current value on every keystroke;
# Streamlit re-runs on Enter automatically, so if query is non-empty and
# the Send button was NOT the trigger, we treat a non-empty query as submitted
# when the user presses Enter (Streamlit fires a rerun on Enter in text_input).
# We gate on pending_query being empty so we don't double-run.

run_query = None

if ask_clicked and query.strip():
    run_query = query.strip()
elif not ask_clicked and query.strip() and not st.session_state.pending_query:
    # Enter was pressed — Streamlit re-ran with the filled value
    run_query = query.strip()

# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_query:
    # Mark as in-flight so a stale rerun doesn't double-fire
    st.session_state.pending_query = run_query

    start_time = time.time()
    with st.spinner("Thinking…"):
        result = cached_pipeline_run(run_query, TOP_K, TOP_RERANK)
    elapsed = round(time.time() - start_time, 2)

    new_message = {
        "query":                run_query,
        "answer":               result["answer"],
        "sources":              result["sources"],
        "conflicts_detected":   result["conflicts_detected"],
        "fallback_triggered":   result["fallback_triggered"],
        "retrieval_quality":    result["retrieval_quality"],
        "context_tokens":       result["context_tokens"],
        "self_consistency_score": result["self_consistency_score"],
        "response_time":        elapsed,
    }

    sess = get_current_session()
    if sess is None:
        title = run_query[:45] + ("..." if len(run_query) > 45 else "")
        st.session_state.all_sessions.append({
            "id":       st.session_state.current_session_id,
            "title":    title,
            "messages": [new_message],
        })
    else:
        sess["messages"].append(new_message)

    save_all_sessions(st.session_state.all_sessions)

    # Clear input by bumping the key (forces Streamlit to recreate the widget empty)
    st.session_state.input_key += 1
    st.session_state.pending_query = ""
    st.rerun()