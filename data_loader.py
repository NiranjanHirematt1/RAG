"""PDF loading, chunking and (local) embeddings.

Groq does not provide an embeddings endpoint, so embeddings are computed
locally with a small sentence-transformers model. No API key required.
"""
import os
import tempfile
import threading

# The default HuggingFace cache (C:\Users\<you>\.cache\huggingface) is often
# locked or permission-denied on Windows (WinError 5), especially under OneDrive.
# Redirect it to a writable, non-synced folder before importing the model libs.
if not os.environ.get("HF_HOME"):
    _base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    _cache = os.path.join(_base, "documind_hf_cache")
    os.makedirs(_cache, exist_ok=True)
    os.environ["HF_HOME"] = _cache

from dotenv import load_dotenv
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

# Load the embedding model once, lazily and thread-safely.
_model = None
_model_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(EMBED_MODEL)
    return _model


def load_and_chunk_pdf(path: str) -> list[str]:
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks: list[str] = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = _get_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.tolist()
