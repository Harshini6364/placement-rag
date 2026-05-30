"""
app.py
Streamlit UI — ChatGPT-style layout.
Fixes: no gaps, markdown tables rendered, hallucination inside sources expander.
"""
import os
import time
import logging
import json
import uuid
import re
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

# ── Answer cleaning ───────────────────────────────────────────────────────────

def clean_answer(text: str) -> str:
    """
    1. Remove chain-of-thought preamble lines
    2. Extract only final answer section
    3. Collapse multiple blank lines into one
    4. Strip leading/trailing whitespace
    """
    # Remove "To answer this / I will follow / using AGGREGATION rules" preamble
    text = re.sub(
        r"(?im)^.*?(follow|using|apply).{0,60}(step|query|multi.hop|aggregation).*$\n?",
        "", text
    )

    # Extract content after final answer markers
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

    # Remove Step N: blocks, keep trailing content
    step_pattern = re.compile(
        r"(?im)^(step\s*\d+[:\.\)].*?)(?=step\s*\d+[:\.\)]|\Z)", re.DOTALL
    )
    steps = list(step_pattern.finditer(text))
    if steps:
        last_end  = steps[-1].end()
        remainder = text[last_end:].strip()
        text = remainder if remainder else re.sub(
            r"(?im)^step\s*\d+[:\.\)][^\n]*\n?", "", text
        )

    # Remove bold step headers
    text = re.sub(r"\*\*step\s*\d+[:\.\)][^\*]*\*\*\n?", "", text, flags=re.IGNORECASE)

    # Remove "Assuming the student..." reasoning artifacts
    text = re.sub(r"(?im)^assuming\b.*$\n?", "", text)

    # ── KEY FIX: collapse 2+ consecutive blank lines into exactly one ─────────
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove blank lines that are just whitespace
    lines = text.split("\n")
    cleaned = []
    prev_blank = False
    for line in lines:
        is_blank = line.strip() == ""
        if is_blank and prev_blank:
            continue          # skip consecutive blanks
        cleaned.append(line)
        prev_blank = is_blank

    return "\n".join(cleaned).strip()


def answer_to_html(text: str) -> str:
    """
    Convert cleaned answer text to HTML.
    Renders markdown tables, bold, and bullet lists properly.
    Replaces newlines with <br> so no gaps appear.
    """
    import html as html_mod

    lines  = text.split("\n")
    output = []
    in_table = False
    table_rows = []

    def flush_table():
        nonlocal table_rows
        if not table_rows:
            return ""
        html_table = '<table style="border-collapse:collapse;width:100%;margin:6px 0;font-size:0.88rem;">'
        for ri, row in enumerate(table_rows):
            # Skip pure separator rows like |---|---|
            if all(re.match(r'^[-: ]+$', c.strip()) for c in row if c.strip()):
                continue
            html_table += "<tr>"
            tag = "th" if ri == 0 else "td"
            style = (
                'style="border:1px solid #ccc;padding:5px 10px;'
                + ('background:#f0f0f0;font-weight:600;"' if ri == 0 else '"')
            )
            for cell in row:
                html_table += f"<{tag} {style}>{html_mod.escape(cell.strip())}</{tag}>"
            html_table += "</tr>"
        html_table += "</table>"
        table_rows.clear()
        return html_table

    for line in lines:
        # Detect markdown table rows  |col|col|
        if re.match(r"^\s*\|", line):
            in_table = True
            cells = [c for c in line.split("|")]
            # Remove first and last empty cells from split
            if cells and cells[0].strip() == "":
                cells = cells[1:]
            if cells and cells[-1].strip() == "":
                cells = cells[:-1]
            table_rows.append(cells)
            continue
        else:
            if in_table:
                output.append(flush_table())
                in_table = False

        # Empty line → single line break (not double)
        if line.strip() == "":
            output.append("<br>")
            continue

        # Escape HTML
        safe = html_mod.escape(line)

        # Bold **text**
        safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)

        # Bullet points
        if re.match(r"^\s*[-*•]\s+", safe):
            safe = re.sub(r"^\s*[-*•]\s+", "", safe)
            safe = f"• {safe}<br>"
        # Numbered list
        elif re.match(r"^\s*\d+\.\s+", safe):
            safe = f"{safe}<br>"
        else:
            safe = f"{safe}<br>"

        output.append(safe)

    # Flush any remaining table
    if in_table:
        output.append(flush_table())

    return "".join(output)


# ── Page config ───────────────────────────────────────────────────────────────
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

.block-container {
    padding-bottom: 110px !important;
    padding-top: 1.2rem !important;
    max-width: 820px !important;
    margin: 0 auto !important;
}

section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem;
}

/* User bubble */
.chat-q { display:flex; justify-content:flex-end; margin:10px 0 2px 0; }
.chat-q-bubble {
    background: #2e5ff6; color: #fff;
    border-radius: 18px 18px 4px 18px;
    padding: 10px 15px; max-width: 75%;
    font-size: 0.95rem; line-height: 1.5; word-wrap: break-word;
}

/* Assistant bubble */
.chat-a { display:flex; justify-content:flex-start; margin:4px 0 0 0; }
.chat-a-text {
    max-width: 92%;
    font-size: 0.93rem;
    line-height: 1.6;
    word-wrap: break-word;
    color: #1a3a1a;
    background: #eafbea;
    border: 1px solid #b2e0b2;
    border-radius: 10px;
    padding: 10px 14px;
}
.chat-a-text.warning-text {
    background: #fffbe6; border-color: #f0a500; color: #5a3e00;
}
.chat-a-text.error-text {
    background: #fff0f0; border-color: #e05252; color: #5a0000;
}

/* Response time */
.rt-chip {
    font-size: 0.82rem; color: #999;
    margin: 2px 0 8px 4px; display: block;
}

/* Sources */
.sources-block {
    font-size: 0.74rem; font-family: monospace;
    color: #aaa; white-space: pre-wrap;
    word-break: break-all; line-height: 1.4;
}

/* Hallucination badge */
.h-pass  { color:#1a7a1a; font-weight:600; }
.h-warn  { color:#8a5a00; font-weight:600; }
.h-fail  { color:#8a0000; font-weight:600; }
.h-label { font-size:0.78rem; color:#666; margin-right:6px; }
.h-row   { font-size:0.79rem; margin:2px 0; font-family:monospace; }

#chat-bottom { height:1px; }
</style>
""", unsafe_allow_html=True)

# ── Persistent storage ────────────────────────────────────────────────────────
HISTORY_FILE = "data/chat_history.json"


def load_all_sessions() -> list[dict]:
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                data = json.load(f)
            if not isinstance(data, list) or not data:
                return []
            if isinstance(data[0], dict) and "query" in data[0] and "messages" not in data[0]:
                title = data[0].get("query", "Imported")
                title = (title[:45] + "...") if len(title) > 45 else title
                return [{"id": str(uuid.uuid4()),
                         "title": f"[Imported] {title}",
                         "messages": data}]
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
    pipeline.config.top_k_rerank   = top_rerank
    response = pipeline.run(query)
    return {
        "answer":                 response.answer,
        "sources":                response.sources,
        "conflicts_detected":     response.conflicts_detected,
        "fallback_triggered":     response.fallback_triggered,
        "retrieval_quality":      response.retrieval_quality,
        "context_tokens":         response.context_tokens,
        "self_consistency_score": response.self_consistency_score,
        "chain_of_thought":       getattr(response, "chain_of_thought", ""),
    }


@st.cache_resource
def load_pipeline():
    embedder = HybridEmbedder(
        faiss_path=os.getenv("FAISS_INDEX_PATH", "data/faiss_index"),
        bm25_path=os.getenv("BM25_PATH",          "data/bm25_store.pkl"),
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
    limiter           = OvershadowLimiter()
    refiner           = ContextRefiner(limiter)
    prompt_builder    = GroundedPromptBuilder()
    generator         = GroqGenerator()
    conflict_detector = PlacementConflictDetector()
    fallback_guard    = FallbackGuard()
    feedback          = FeedbackLoop(limiter)
    pipeline          = RAGPipeline(
        retriever, reranker, refiner, prompt_builder,
        generator, conflict_detector, fallback_guard,
        PipelineConfig(top_k_retrieve=20, top_k_rerank=5),
    )
    return pipeline, limiter, feedback


pipeline, limiter, feedback = load_pipeline()
st.session_state._pipeline = pipeline

# ── Session state ─────────────────────────────────────────────────────────────
if "all_sessions"       not in st.session_state:
    st.session_state.all_sessions       = load_all_sessions()
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())
if "pending_query"      not in st.session_state:
    st.session_state.pending_query      = ""
if "input_key"          not in st.session_state:
    st.session_state.input_key          = 0

TOP_K      = 20
TOP_RERANK = 5


def get_current_session() -> dict | None:
    for s in st.session_state.all_sessions:
        if s["id"] == st.session_state.current_session_id:
            return s
    return None


def start_new_chat():
    st.session_state.current_session_id = str(uuid.uuid4())
    st.session_state.pending_query       = ""
    st.session_state.input_key          += 1


# ── Hallucination helpers ─────────────────────────────────────────────────────
def parse_hallucination(cot: str) -> dict | None:
    if not cot or "Verdict:" not in cot:
        return None
    v = re.search(r"Verdict:\s*(\w+)", cot)
    s = re.search(r"SC:\s*([\d\.]+)", cot)
    r = re.search(r"Recitation:\s*([\d\.]+)", cot)
    c = re.search(r"Chain:\s*(\w+)", cot)
    return {
        "verdict":    v.group(1) if v else "?",
        "sc":         float(s.group(1)) if s else None,
        "recitation": float(r.group(1)) if r else None,
        "chain":      c.group(1) if c else "?",
    }


def render_hallucination_badge(h: dict):
    verdict = h.get("verdict", "?")
    css     = {"PASS": "h-pass", "WARN": "h-warn", "FAIL": "h-fail"}.get(verdict, "h-label")
    emoji   = {"PASS": "✅", "WARN": "⚠️", "FAIL": "🔴"}.get(verdict, "❓")
    sc_val  = h.get("sc")
    rec_val = h.get("recitation")
    sc_bar  = ("█" * int((sc_val or 0) * 5) + "░" * (5 - int((sc_val or 0) * 5))) if sc_val is not None else "?"
    rec_bar = ("█" * int((rec_val or 0) * 5) + "░" * (5 - int((rec_val or 0) * 5))) if rec_val is not None else "?"
    chain_ok = str(h.get("chain", "?"))
    st.markdown(
        f"""
<div style="margin-top:8px;border-top:1px solid #ddd;padding-top:6px;">
  <span class="h-label">🧠 Hallucination guard</span>
  <span class="{css}">{emoji} {verdict}</span><br>
  <span class="h-row">Self-consistency &nbsp;{sc_bar}&nbsp; {sc_val}</span><br>
  <span class="h-row">Recitation check &nbsp;{rec_bar}&nbsp; {rec_val}</span><br>
  <span class="h-row">Chain verified &nbsp;&nbsp;&nbsp;{'✅' if chain_ok=='True' else '⚠️'}&nbsp; {chain_ok}</span>
</div>""",
        unsafe_allow_html=True,
    )


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
                export_lines.append(f"=== {sess.get('title','Chat')} ===")
                for msg in sess.get("messages", []):
                    export_lines.append(f"Q: {msg.get('query','')}")
                    export_lines.append(f"A: {msg.get('answer','')}")
                    export_lines.append(f"Time: {msg.get('response_time','?')}s\n")
            st.download_button(
                "⬇ Export", "\n".join(export_lines),
                file_name="placement_rag_history.txt",
                use_container_width=True,
            )
        st.markdown("")
        for sess in reversed(all_sessions):
            is_active = (sess["id"] == st.session_state.current_session_id)
            label = f"{'▶ ' if is_active else ''}{sess.get('title','Chat')}"
            if st.button(label, key=f"sess_{sess['id']}",
                         use_container_width=True,
                         help=f"{len(sess.get('messages',[]))} question(s)"):
                st.session_state.current_session_id = sess["id"]
                st.session_state.pending_query       = ""
                st.rerun()
    else:
        st.caption("No previous chats yet.")

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style='text-align:center;margin-bottom:2px;'>🎓 Placement Intelligence Assistant</h2>"
    "<p style='text-align:center;color:#888;font-size:0.83rem;margin-bottom:1rem;'>"
    "SVECW · RAG-ATHON 24 &nbsp;|&nbsp; Hybrid RAG + Groq + Vision + Temporal Reasoning</p>",
    unsafe_allow_html=True,
)

# ── Chat messages ─────────────────────────────────────────────────────────────
current_session = get_current_session()

if not current_session or not current_session.get("messages"):
    st.markdown(
        "<div style='text-align:center;color:#555;margin-top:80px;font-size:1rem;'>"
        "Ask anything about placements at SVECW ↓</div>",
        unsafe_allow_html=True,
    )
else:
    for item in current_session["messages"]:

        # User bubble
        q_html = item["query"].replace("<","&lt;").replace(">","&gt;")
        st.markdown(
            f'<div class="chat-q"><div class="chat-q-bubble">{q_html}</div></div>',
            unsafe_allow_html=True,
        )

        # Build answer HTML (renders tables, no gaps)
        cleaned     = clean_answer(item["answer"])
        answer_html = answer_to_html(cleaned)

        if item.get("fallback_triggered"):
            css_class   = "chat-a-text warning-text"
            prefix      = "<strong>⚠️ Out-of-corpus — not in placement documents.</strong><br><br>"
            answer_html = prefix + answer_html
        elif item.get("conflicts_detected"):
            css_class      = "chat-a-text error-text"
            conflicts_html = "<br>".join(
                c.replace("<","&lt;").replace(">","&gt;")
                for c in item["conflicts_detected"]
            )
            answer_html = f"<strong>⚠️ Conflicts:</strong><br>{conflicts_html}<br><br>" + answer_html
        else:
            css_class = "chat-a-text"

        st.markdown(
            f'<div class="chat-a"><div class="{css_class}">{answer_html}</div></div>',
            unsafe_allow_html=True,
        )

        # Response time chip
        rt = item.get("response_time", "?")
        rt_label = (f"⚡ {rt}s (cached)" if isinstance(rt, float) and rt < 0.1
                    else f"⏱ {rt}s")
        st.markdown(f'<span class="rt-chip">{rt_label}</span>', unsafe_allow_html=True)

        # Sources + hallucination report (same expander)
        has_sources = bool(item.get("sources"))
        h_data      = parse_hallucination(item.get("chain_of_thought", ""))

        if has_sources or h_data:
            with st.expander("📚 Sources", expanded=False):
                if has_sources:
                    st.markdown(
                        f'<div class="sources-block">'
                        f'{chr(10).join(item["sources"])}</div>',
                        unsafe_allow_html=True,
                    )
                if h_data:
                    render_hallucination_badge(h_data)

        st.markdown("<div style='margin-bottom:4px'></div>", unsafe_allow_html=True)


st.markdown('<div id="chat-bottom"></div>', unsafe_allow_html=True)
st.markdown("""
<script>
(function(){
  var el=document.getElementById('chat-bottom');
  if(el) el.scrollIntoView({behavior:'smooth'});
})();
</script>""", unsafe_allow_html=True)

# ── Input bar ─────────────────────────────────────────────────────────────────
st.markdown("---")
col_input, col_send = st.columns([6, 1])

with col_input:
    query = st.text_input(
        label="input", label_visibility="collapsed",
        placeholder="Ask a placement question…",
        key=f"chat_input_{st.session_state.input_key}",
    )
with col_send:
    ask_clicked = st.button("Send ➤", type="primary", use_container_width=True)

# ── Run ───────────────────────────────────────────────────────────────────────
run_query = None
if ask_clicked and query.strip():
    run_query = query.strip()
elif not ask_clicked and query.strip() and not st.session_state.pending_query:
    run_query = query.strip()

if run_query:
    st.session_state.pending_query = run_query

    start_time = time.time()
    with st.spinner("Thinking…"):
        result = cached_pipeline_run(run_query, TOP_K, TOP_RERANK)
    elapsed = round(time.time() - start_time, 2)

    new_message = {
        "query":                  run_query,
        "answer":                 result["answer"],
        "sources":                result["sources"],
        "conflicts_detected":     result["conflicts_detected"],
        "fallback_triggered":     result["fallback_triggered"],
        "retrieval_quality":      result["retrieval_quality"],
        "context_tokens":         result["context_tokens"],
        "self_consistency_score": result["self_consistency_score"],
        "chain_of_thought":       result.get("chain_of_thought", ""),
        "response_time":          elapsed,
    }

    sess = get_current_session()
    if sess is None:
        title = run_query[:45] + ("..." if len(run_query) > 45 else "")
        st.session_state.all_sessions.append({
            "id": st.session_state.current_session_id,
            "title": title,
            "messages": [new_message],
        })
    else:
        sess["messages"].append(new_message)

    save_all_sessions(st.session_state.all_sessions)
    st.session_state.input_key    += 1
    st.session_state.pending_query = ""
    st.rerun()