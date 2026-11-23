import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

load_dotenv()

UPSERT_BATCH = 64


class QdrantStorage:
    """Vector store wrapper around Qdrant.

    - QDRANT_URL + QDRANT_API_KEY set -> Qdrant Cloud cluster (recommended for deploy)
    - QDRANT_URL set, no key        -> self-hosted Qdrant server (e.g. Docker)
    - nothing set                   -> local on-disk store at QDRANT_PATH (dev only)
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        path: str | None = None,
        collection_name: str | None = None,
        dim: int | None = None,
    ):
        url = url if url is not None else os.getenv("QDRANT_URL", "").strip()
        api_key = api_key if api_key is not None else os.getenv("QDRANT_API_KEY", "").strip()
        path = path or os.getenv("QDRANT_PATH", "qdrant_local")
        self.collection = collection_name or os.getenv("QDRANT_COLLECTION", "docs")
        dim = dim or int(os.getenv("EMBED_DIM", "384"))

        if url:
            self.qdrant_client = QdrantClient(url=url, api_key=api_key or None, timeout=60)
            self.mode = "cloud" if api_key else "server"
        else:
            self.qdrant_client = QdrantClient(path=path)
            self.mode = "local"

        if not self.qdrant_client.collection_exists(self.collection):
            self.qdrant_client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        else:
            existing = self.qdrant_client.get_collection(self.collection).config.params.vectors
            size = getattr(existing, "size", None)
            if size is not None and size != dim:
                raise RuntimeError(
                    f"Collection '{self.collection}' has vector size {size}, but EMBED_DIM={dim}. "
                    "Use a new QDRANT_COLLECTION name or delete the old collection."
                )

        # Qdrant Cloud requires a payload index to filter/facet on a field.
        # Local mode ignores indexes, so only do this against a server.
        if self.mode != "local":
            self.qdrant_client.create_payload_index(
                collection_name=self.collection,
                field_name="source",
                field_schema=PayloadSchemaType.KEYWORD,
            )

    def upsert(self, ids, vectors, payloads):
        # Batch so large PDFs don't exceed the cloud request-size limit.
        for start in range(0, len(ids), UPSERT_BATCH):
            end = start + UPSERT_BATCH
            points = [
                PointStruct(id=i, vector=v, payload=p)
                for i, v, p in zip(ids[start:end], vectors[start:end], payloads[start:end])
            ]
            self.qdrant_client.upsert(collection_name=self.collection, points=points, wait=True)

    def delete_source(self, source: str):
        """Remove every chunk of a document (used before re-ingesting it)."""
        self.qdrant_client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])
            ),
            wait=True,
        )

    def list_sources(self, limit: int = 100) -> list[tuple[str, int]]:
        """(document name, chunk count) for every document in the collection."""
        if self.mode != "local":
            res = self.qdrant_client.facet(collection_name=self.collection, key="source", limit=limit)
            return [(str(h.value), h.count) for h in res.hits]
        counts: dict[str, int] = {}
        offset = None
        while True:
            points, offset = self.qdrant_client.scroll(
                collection_name=self.collection, limit=256, offset=offset,
                with_payload=["source"], with_vectors=False,
            )
            for p in points:
                s = (p.payload or {}).get("source")
                if s:
                    counts[s] = counts.get(s, 0) + 1
            if offset is None:
                break
        return sorted(counts.items())

    def search(self, query_vector, top_k: int = 5):
        results = self.qdrant_client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k,
        ).points

        contexts = []
        sources = []
        for r in results:
            payload = r.payload or {}
            text = payload.get("text")
            source = payload.get("source")
            if text:
                contexts.append(text)
            if source and source not in sources:  # keep rank order, dedupe
                sources.append(source)

        return {"contexts": contexts, "sources": sources}
