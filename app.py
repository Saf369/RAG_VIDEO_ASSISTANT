try:
    import torch
    torch.classes.__path__ = []
except ImportError:
    pass

import streamlit as st
import os
import time
from dotenv import load_dotenv

load_dotenv()

# Bridge Streamlit Cloud secrets → os.environ (so downstream modules work unchanged)
try:
    for key in st.secrets:
        if key not in os.environ:
            os.environ[key] = str(st.secrets[key])
except Exception:
    pass  # Not on Streamlit Cloud, .env already loaded

from utils.audio_processor import download_audio_from_youtube, convert_to_wav, chunk_audio
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_actionable_items, extract_questions, extract_key_discussion_points
from core.rag_engine import build_vector_store, build_rag_chain, get_retriever, ask_question


# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Root variables */
    :root {
        --accent-primary: #7c3aed;
        --accent-secondary: #a78bfa;
        --accent-gradient: linear-gradient(135deg, #7c3aed 0%, #2563eb 50%, #06b6d4 100%);
        --surface-glass: rgba(255, 255, 255, 0.03);
        --border-subtle: rgba(255, 255, 255, 0.06);
        --text-primary: #f1f5f9;
        --text-muted: #94a3b8;
        --radius-lg: 16px;
        --radius-md: 12px;
    }

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* Hero header */
    .hero-header {
        text-align: center;
        padding: 2.5rem 1rem 2rem;
        margin-bottom: 1.5rem;
    }
    .hero-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .hero-header p {
        color: var(--text-muted);
        font-size: 1.1rem;
        font-weight: 400;
    }

    /* Metric cards */
    .metric-card {
        background: var(--surface-glass);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        backdrop-filter: blur(12px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .metric-card:hover {
        border-color: var(--accent-secondary);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(124, 58, 237, 0.15);
    }
    .metric-card .label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--accent-secondary);
        margin-bottom: 0.4rem;
    }
    .metric-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text-primary);
    }

    /* Insight card */
    .insight-card {
        background: var(--surface-glass);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(12px);
    }
    .insight-card h3 {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .insight-card ul {
        margin: 0;
        padding-left: 1.2rem;
    }
    .insight-card li {
        color: var(--text-muted);
        font-size: 0.9rem;
        line-height: 1.7;
        margin-bottom: 0.3rem;
    }

    /* Chat messages */
    .chat-msg {
        padding: 1rem 1.25rem;
        border-radius: var(--radius-md);
        margin-bottom: 0.75rem;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .chat-user {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.15), rgba(37, 99, 235, 0.1));
        border: 1px solid rgba(124, 58, 237, 0.2);
        margin-left: 3rem;
    }
    .chat-bot {
        background: var(--surface-glass);
        border: 1px solid var(--border-subtle);
        margin-right: 3rem;
    }

    /* Progress bar styling */
    .stProgress > div > div > div {
        background: var(--accent-gradient);
    }

    /* Status badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 500;
    }
    .status-ready {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-processing {
        background: rgba(251, 191, 36, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.5);
        backdrop-filter: blur(16px);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdown"] {
        padding: 0.5rem 0;
    }

    /* Divider */
    .subtle-divider {
        height: 1px;
        background: var(--border-subtle);
        margin: 1.5rem 0;
    }

    /* Expander text */
    .streamlit-expanderContent {
        font-size: 0.9rem;
        line-height: 1.7;
        color: var(--text-muted);
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State Init ─────────────────────────────────────────────────────────
defaults = {
    "pipeline_done": False,
    "transcription": None,
    "title": None,
    "summary": None,
    "actionable_items": [],
    "discussion_points": [],
    "questions": [],
    "rag_chain": None,
    "chat_history": [],
    "processing": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Helper: parse list items ───────────────────────────────────────────────────
def parse_items(items: list) -> list:
    """Clean up extracted list items, removing empty lines and numbering artifacts."""
    cleaned = []
    for item in items:
        item = item.strip()
        if not item:
            continue
        # Skip "no ... found" fallback messages
        if item.lower().startswith("no ") and "found" in item.lower():
            continue
        cleaned.append(item)
    return cleaned


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎬 Video Assistant")
    st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)

    url = st.text_input(
        "YouTube URL",
        placeholder="https://youtu.be/...",
        help="Paste any YouTube video URL to analyze"
    )

    process_btn = st.button(
        "🚀 Process Video",
        use_container_width=True,
        disabled=st.session_state.processing or not url,
        type="primary",
    )

    if st.session_state.pipeline_done:
        st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<span class="status-badge status-ready">● Ready for Q&A</span>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)
    st.caption("Built with Sarvam STT · Mistral LLM · ChromaDB")


# ─── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>AI Video Assistant</h1>
    <p>Transcribe, summarize, and query any YouTube video with AI</p>
</div>
""", unsafe_allow_html=True)


# ─── Pipeline Processing ────────────────────────────────────────────────────────
if process_btn and url:
    st.session_state.processing = True
    st.session_state.pipeline_done = False
    st.session_state.chat_history = []

    progress = st.progress(0, text="Starting pipeline...")

    try:
        # Step 1: Download
        progress.progress(5, text="⬇️  Downloading audio...")
        audio_path = download_audio_from_youtube(url)
        if not audio_path or not os.path.exists(audio_path):
            st.error("Failed to download audio. Please check the URL.")
            st.session_state.processing = False
            st.stop()

        # Step 2: Convert
        progress.progress(15, text="🔄  Converting to WAV...")
        wav_path = convert_to_wav(audio_path)

        # Step 3: Chunk
        progress.progress(25, text="✂️  Chunking audio...")
        chunks = chunk_audio(wav_path)

        # Step 4: Transcribe
        progress.progress(35, text="🗣️  Transcribing audio...")
        transcription = transcribe_all(chunks, translate=True)
        st.session_state.transcription = transcription

        # Step 5: Title & Summary
        progress.progress(55, text="📝  Generating title & summary...")
        st.session_state.title = generate_title(transcription)
        st.session_state.summary = summarize(transcription)

        # Step 6: Extract insights
        progress.progress(70, text="🔍  Extracting insights...")
        st.session_state.actionable_items = extract_actionable_items(transcription)
        st.session_state.discussion_points = extract_key_discussion_points(transcription)
        st.session_state.questions = extract_questions(transcription)

        # Step 7: Build RAG
        progress.progress(88, text="🧠  Building vector store for Q&A...")
        vector_store = build_vector_store(transcription)
        chain = build_rag_chain(get_retriever(vector_store))
        st.session_state.rag_chain = chain

        progress.progress(100, text="✅  Pipeline complete!")
        time.sleep(0.5)
        progress.empty()

        st.session_state.pipeline_done = True

    except Exception as e:
        st.error(f"Pipeline error: {e}")

    finally:
        st.session_state.processing = False

    st.rerun()


# ─── Results Display ────────────────────────────────────────────────────────────
if st.session_state.pipeline_done:

    # ── Title ────────────────────────────────────────────────────────────────
    title_text = (st.session_state.title or "").strip().strip("*").strip('"')
    st.markdown(f"## 🎥 {title_text}")

    # ── Metrics row ──────────────────────────────────────────────────────────
    transcript = st.session_state.transcription or ""
    word_count = len(transcript.split())
    dp = parse_items(st.session_state.discussion_points)
    ai = parse_items(st.session_state.actionable_items)
    oq = parse_items(st.session_state.questions)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Words Transcribed</div>
            <div class="value">{word_count:,}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Discussion Points</div>
            <div class="value">{len(dp)}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Actionable Items</div>
            <div class="value">{len(ai)}</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Open Questions</div>
            <div class="value">{len(oq)}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)

    # ── Summary ──────────────────────────────────────────────────────────────
    st.markdown("### 📋 Summary")
    st.markdown(st.session_state.summary or "_No summary generated._")

    st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)

    # ── Insights (3-column layout) ───────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        items_html = "".join(f"<li>{item}</li>" for item in dp) if dp else "<li style='color:#64748b'>None found</li>"
        st.markdown(f"""
        <div class="insight-card">
            <h3>💬 Discussion Points</h3>
            <ul>{items_html}</ul>
        </div>""", unsafe_allow_html=True)

    with col2:
        items_html = "".join(f"<li>{item}</li>" for item in ai) if ai else "<li style='color:#64748b'>None found</li>"
        st.markdown(f"""
        <div class="insight-card">
            <h3>✅ Actionable Items</h3>
            <ul>{items_html}</ul>
        </div>""", unsafe_allow_html=True)

    with col3:
        items_html = "".join(f"<li>{item}</li>" for item in oq) if oq else "<li style='color:#64748b'>None found</li>"
        st.markdown(f"""
        <div class="insight-card">
            <h3>❓ Open Questions</h3>
            <ul>{items_html}</ul>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)

    # ── Full Transcript (expandable) ─────────────────────────────────────────
    with st.expander("📜 Full Transcript"):
        st.text(transcript)

    st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)

    # ── RAG Q&A Chat ─────────────────────────────────────────────────────────
    st.markdown("### 💬 Ask Questions About the Video")
    st.caption("Powered by RAG — your questions are answered using the actual transcript.")

    # Display chat history
    for msg in st.session_state.chat_history:
        role_class = "chat-user" if msg["role"] == "user" else "chat-bot"
        icon = "🧑" if msg["role"] == "user" else "🤖"
        st.markdown(
            f'<div class="chat-msg {role_class}">{icon} &nbsp; {msg["content"]}</div>',
            unsafe_allow_html=True,
        )

    # Chat input
    user_query = st.chat_input("Ask anything about this video...")

    if user_query and st.session_state.rag_chain:
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        with st.spinner("Thinking..."):
            answer = ask_question(user_query, st.session_state.rag_chain)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()


# ─── Empty State ─────────────────────────────────────────────────────────────────
if not st.session_state.pipeline_done and not st.session_state.processing:
    st.markdown("""
    <div style="text-align:center; padding:4rem 1rem; color:var(--text-muted);">
        <p style="font-size:3rem; margin-bottom:0.5rem;">🎬</p>
        <p style="font-size:1.1rem; font-weight:500;">Paste a YouTube URL in the sidebar to get started</p>
        <p style="font-size:0.85rem; margin-top:0.5rem;">
            The pipeline will transcribe, summarize, extract insights, and let you chat with the video.
        </p>
    </div>
    """, unsafe_allow_html=True)
