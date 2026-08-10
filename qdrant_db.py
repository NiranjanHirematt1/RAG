import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

load_dotenv()


class QdrantStorage:
    """Vector store wrapper around Qdrant.

    If QDRANT_URL is set (e.g. http://localhost:6333) it connects to a running
    Qdrant server. Otherwise it uses a local on-disk store at QDRANT_PATH, so
    nothing extra needs to be running.
    """

    def __init__(
        self,
        url: str | None = None,
        path: str | None = None,
        collection_name: str | None = None,
        dim: int | None = None,
    ):
        url = url if url is not None else os.getenv("QDRANT_URL", "").strip()
        path = path or os.getenv("QDRANT_PATH", "qdrant_local")
        self.collection = collection_name or os.getenv("QDRANT_COLLECTION", "docs")
        dim = dim or int(os.getenv("EMBED_DIM", "384"))

        if url:
            self.qdrant_client = QdrantClient(url=url, timeout=30)
        else:
            self.qdrant_client = QdrantClient(path=path)

        if not self.qdrant_client.collection_exists(self.collection):
            self.qdrant_client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE,
                ),
            )

    def upsert(self, ids, vectors, payloads):
        points = [
            PointStruct(
                id=ids[i],
                vector=vectors[i],
                payload=payloads[i],
            )
            for i in range(len(ids))
        ]

        self.qdrant_client.upsert(
            collection_name=self.collection,
            points=points,
        )

    def search(self, query_vector, top_k: int = 5):
        results = self.qdrant_client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k,
        ).points

        contexts = []
        sources = set()

        for r in results:
            payload = getattr(r, "payload", None) or {}

            text = payload.get("text")
            source = payload.get("source")

            if text:
                contexts.append(text)

            if source:
                sources.add(source)

        return {
            "contexts": contexts,
            "sources": list(sources),
        }
