"""
Local Search Tool - RAG search against local FAISS vector store.

Uses Gemini embeddings to encode queries and search a FAISS index
built from knowledge base documents.
"""

import logging
import pickle
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from google import genai
from google.genai import types

from src.config import get_settings
from src.tools.base import BaseTool, ToolConfig, ToolResult

logger = logging.getLogger(__name__)

# Embedding config
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768  # Reduced from 3072 for speed/storage


class VectorStore:
    """FAISS vector store with Gemini embeddings."""

    def __init__(self, store_path: Path, embedding_dim: int = EMBEDDING_DIM) -> None:
        self._store_path = store_path
        self._embedding_dim = embedding_dim
        self._index: faiss.IndexFlatIP | None = None
        self._documents: list[dict[str, Any]] = []
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            settings = get_settings()
            api_key = settings.models.gemini_api_key
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set. Required for embeddings.")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def _embed(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        """Generate embeddings for a list of texts using Gemini."""
        client = self._get_client()
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self._embedding_dim,
            ),
        )
        embeddings = response.embeddings or []
        vectors = [e.values for e in embeddings]
        arr = np.array(vectors, dtype=np.float32)
        # L2-normalize for cosine similarity via inner product
        faiss.normalize_L2(arr)
        return arr

    def build(self, documents: list[dict[str, Any]]) -> None:
        """
        Build the FAISS index from a list of documents.

        Args:
            documents: List of dicts with at least 'content' key.
                       Optional keys: 'id', 'source', 'title', 'metadata'.
        """
        if not documents:
            raise ValueError("No documents to index.")

        texts = [doc["content"] for doc in documents]
        logger.info("Embedding %d documents...", len(texts))

        # Batch embeddings (Gemini supports up to 100 per call)
        all_vectors: list[np.ndarray] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors = self._embed(batch, task_type="RETRIEVAL_DOCUMENT")
            all_vectors.append(vectors)

        matrix = np.vstack(all_vectors)

        # Build FAISS index (Inner Product for normalized vectors = cosine similarity)
        self._index = faiss.IndexFlatIP(self._embedding_dim)
        self._index.add(matrix)
        self._documents = documents

        logger.info("FAISS index built: %d vectors, dim=%d", self._index.ntotal, self._embedding_dim)

    def save(self) -> None:
        """Persist index and documents to disk."""
        if self._index is None:
            raise ValueError("No index to save. Call build() first.")

        self._store_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._store_path / "index.faiss"))
        with open(self._store_path / "documents.pkl", "wb") as f:
            pickle.dump(self._documents, f)

        logger.info("Saved FAISS index to %s", self._store_path)

    def load(self) -> bool:
        """Load index and documents from disk. Returns True if successful."""
        index_path = self._store_path / "index.faiss"
        docs_path = self._store_path / "documents.pkl"

        if not index_path.exists() or not docs_path.exists():
            return False

        self._index = faiss.read_index(str(index_path))
        with open(docs_path, "rb") as f:
            self._documents = pickle.load(f)  # noqa: S301

        logger.info(
            "Loaded FAISS index: %d vectors from %s",
            self._index.ntotal,
            self._store_path,
        )
        return True

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """
        Search the index for documents similar to query.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of dicts with 'content', 'score', 'source', etc.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        query_vec = self._embed([query], task_type="RETRIEVAL_QUERY")
        scores, indices = self._index.search(query_vec, min(top_k, self._index.ntotal))

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0:
                continue
            doc = self._documents[idx]
            results.append({
                "id": doc.get("id", f"doc_{idx}"),
                "content": doc["content"],
                "score": float(score),
                "source": doc.get("source", "unknown"),
                "title": doc.get("title", ""),
            })

        return results

    @property
    def is_loaded(self) -> bool:
        return self._index is not None and self._index.ntotal > 0


class LocalSearchTool(BaseTool):
    """
    RAG Search tool backed by a FAISS vector store with Gemini embeddings.

    Capabilities:
    - Search local vector store for relevant documents
    - Privacy-safe (embeddings generated at index time; queries use Gemini API)
    - Returns ranked results with similarity scores

    This is the privacy-friendly search option.
    """

    def __init__(self, config: ToolConfig | None = None) -> None:
        super().__init__(config)
        settings = get_settings()
        store_path = Path(settings.vector_store.path)
        self._store = VectorStore(store_path)
        # Try to load existing index
        self._store.load()

    def default_config(self) -> ToolConfig:
        return ToolConfig(
            name="local_search",
            description=(
                "Searches the local knowledge base (RAG) for information. "
                "Use for company documents, internal knowledge, or when privacy is required. "
                "Always available regardless of privacy settings."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant documents",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 3)",
                    },
                },
                "required": ["query"],
            },
            estimated_cost=0.0,
            estimated_latency_ms=200.0,
            is_local=True,
        )

    def run(
        self,
        query: str,
        top_k: int = 3,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Execute search against local FAISS vector store.

        Args:
            query: The search query
            top_k: Number of results to return

        Returns:
            ToolResult with matching documents
        """
        start_time = time.perf_counter()

        if not self._store.is_loaded:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                output={
                    "query": query,
                    "results": [],
                    "total_results": 0,
                },
                cost=0.0,
                latency_ms=latency_ms,
                error="Vector store not initialized. Run: python -m scripts.ingest_knowledge",
            )

        try:
            results = self._store.search(query, top_k)
            latency_ms = (time.perf_counter() - start_time) * 1000

            return ToolResult(
                output={
                    "query": query,
                    "results": results,
                    "total_results": len(results),
                },
                cost=0.0,
                latency_ms=latency_ms,
                metadata={"top_k": top_k, "source": "faiss"},
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error("Local search failed: %s", e)
            return ToolResult(
                output=None,
                cost=0.0,
                latency_ms=latency_ms,
                error=f"Local search error: {e}",
            )
