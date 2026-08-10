"""Streamlit front-end for the Groq-powered RAG assistant."""
import os

# Silence Streamlit's module watcher before it imports. Without this it tries to
# introspect every transformers submodule and floods the console with harmless
# "No module named 'torchvision'" tracebacks. Must be set before importing st.
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import tempfile

import streamlit as st
from dotenv import load_dotenv

import rag_core

load_dotenv()

st.set_page_config(
    page_title="DocuMind · Ask your PDFs",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    :root { --brand:#5b5bd6; --brand-2:#8b5cf6; --ink:#1f2233; }
    .stApp { background: #f6f7fb !important; }
    #MainMenu, footer {visibility: hidden;}

    /* Force readable dark text everywhere in the main area */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp li,
    section[data-testid="stSidebar"] * { color: var(--ink); }

    .hero {
        background: linear-gradient(120deg, #5b5bd6 0%, #8b5cf6 55%, #6366f1 100%);
        padding: 28px 34px; border-radius: 18px;
        box-shadow: 0 10px 30px rgba(91,91,214,.25); margin-bottom: 22px;
    }
    .hero h1, .hero p, .hero span { color: #ffffff !important; }
    .hero h1 { margin: 0; font-size: 30px; font-weight: 800; letter-spacing:-.5px;}
    .hero p  { margin: 6px 0 0; font-size: 15px; }

    .pill {
        display:inline-block; background: rgba(255,255,255,.22);
        padding: 4px 12px; border-radius: 999px; font-size: 12px;
        margin-top: 12px; margin-right:6px;
    }

    .chat-user, .chat-bot {
        padding: 14px 18px; border-radius: 16px; margin: 8px 0;
        line-height: 1.55; font-size: 15px; color: var(--ink) !important;
    }
    .chat-user, .chat-user * {
        background: #ffffff; border-color:#e7e8f0; color: var(--ink) !important;
    }
    .chat-user { border: 1px solid #e7e8f0; box-shadow: 0 2px 6px rgba(20,20,50,.05); }
    .chat-bot { background: #efeafd; border: 1px solid #ded4fb; }
    .chat-bot, .chat-bot * { color:#241b47 !important; }

    .role { font-size: 12px; font-weight: 700; text-transform: uppercase;
            letter-spacing:.6px; opacity:.6; margin-bottom: 4px; }

    .source-tag {
        display:inline-block; background:#ffffff; border:1px solid #ded4fb;
        color:#5b3fd6 !important; padding:3px 10px; border-radius:999px;
        font-size:12px; margin:4px 6px 0 0;
    }
    .empty {
        text-align:center; padding:60px 20px; color:#6b7080 !important;
        border:2px dashed #cfd2e0; border-radius:18px; background:#ffffff;
    }
    section[data-testid="stSidebar"] { background:#ffffff; border-right:1px solid #ececf3;}
    div.stButton > button {
        background: var(--brand) !important; color:#ffffff !important; border:none;
        border-radius:10px; padding:8px 16px; font-weight:600; width:100%;
    }
    div.stButton > button:hover { background:#4a4ac4 !important; color:#ffffff !important; }
    div.stButton > button * { color:#ffffff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested" not in st.session_state:
    st.session_state.ingested = []  # list of (name, chunk_count)


@st.cache_resource(show_spinner="Warming up the embedding model…")
def warmup():
    # Trigger model + vector store load once.
    rag_core.get_storage()
    return True


# --------------------------------------------------------------------------- #
# Sidebar – documents & settings
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### 📚 Your documents")
    uploaded = st.file_uploader(
        "Upload a PDF", type=["pdf"], accept_multiple_files=False
    )

    if uploaded is not None:
        if st.button("➕ Add to knowledge base"):
            with st.spinner(f"Reading and indexing “{uploaded.name}”…"):
                tmp_dir = tempfile.mkdtemp()
                tmp_path = os.path.join(tmp_dir, uploaded.name)
                with open(tmp_path, "wb") as f:
                    f.write(uploaded.getbuffer())
                try:
                    result = rag_core.ingest_pdf(tmp_path, source_id=uploaded.name)
                    st.session_state.ingested.append((uploaded.name, result.ingested))
                    st.success(f"Indexed {result.ingested} chunks from {uploaded.name}")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Could not ingest: {e}")

    if st.session_state.ingested:
        st.markdown("**Indexed so far**")
        for name, n in st.session_state.ingested:
            st.markdown(f"- {name}  ·  `{n}` chunks")
    else:
        st.caption("No documents yet — upload a PDF to begin.")

    st.divider()
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Context passages (top-k)", 1, 10, 5)
    st.caption(f"Model: `{rag_core.GROQ_MODEL}`")
    st.caption(f"Embeddings: `{os.getenv('EMBED_MODEL', 'BAAI/bge-small-en-v1.5')}`")

    if st.button("🧹 Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="hero">
        <h1>📄 DocuMind</h1>
        <p>Chat with your PDFs — retrieval-augmented answers powered by Groq.</p>
        <span class="pill">⚡ Groq LLM</span>
        <span class="pill">🔎 Qdrant vector search</span>
        <span class="pill">🔒 Local embeddings</span>
    </div>
    """,
    unsafe_allow_html=True,
)

warmup()

# --------------------------------------------------------------------------- #
# Conversation
# --------------------------------------------------------------------------- #
if not st.session_state.messages:
    st.markdown(
        '<div class="empty">👋 Upload a PDF from the sidebar, then ask a question '
        "below to get grounded answers with sources.</div>",
        unsafe_allow_html=True,
    )

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="chat-user"><div class="role">You</div>{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        sources_html = ""
        if msg.get("sources"):
            tags = "".join(f'<span class="source-tag">📎 {s}</span>' for s in msg["sources"])
            sources_html = f'<div style="margin-top:10px">{tags}</div>'
        st.markdown(
            f'<div class="chat-bot"><div class="role">DocuMind</div>'
            f'{msg["content"]}{sources_html}</div>',
            unsafe_allow_html=True,
        )

# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
question = st.chat_input("Ask something about your documents…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Thinking…"):
        try:
            result = rag_core.answer_question(question, top_k=top_k)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result.answer,
                    "sources": result.sources,
                }
            )
        except Exception as e:  # noqa: BLE001
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"⚠️ Something went wrong: {e}",
                    "sources": [],
                }
            )
    st.rerun()
