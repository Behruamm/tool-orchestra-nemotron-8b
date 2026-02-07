"""Tests for the FAISS-backed local search tool."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.tools.local_search import LocalSearchTool, VectorStore


@pytest.fixture
def mock_settings():
    """Mock settings for local search."""
    with patch("src.tools.local_search.get_settings") as mock:
        settings = MagicMock()
        settings.vector_store.path = Path("/tmp/test_vectorstore")
        settings.models.gemini_api_key = "test-gemini-key"
        mock.return_value = settings
        yield settings


@pytest.fixture
def mock_embeddings():
    """Patch the Gemini embedding call to return deterministic vectors."""
    dim = 768

    def fake_embed_content(model, contents, config=None):
        response = MagicMock()
        embeddings = []
        for i, _ in enumerate(contents):
            # Create deterministic but distinct vectors per text
            vec = np.zeros(dim, dtype=np.float32)
            # Use hash of content to create a semi-unique vector
            idx = hash(contents[i]) % dim
            vec[idx] = 1.0
            emb = MagicMock()
            emb.values = vec.tolist()
            embeddings.append(emb)
        response.embeddings = embeddings
        return response

    with patch("src.tools.local_search.genai") as mock_genai:
        mock_client = MagicMock()
        mock_client.models.embed_content.side_effect = fake_embed_content
        mock_genai.Client.return_value = mock_client
        yield mock_client


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "id": "doc_0",
            "content": "Tool Orchestra is a cost-aware multi-model orchestration framework.",
            "source": "overview.md",
            "title": "Overview",
        },
        {
            "id": "doc_1",
            "content": "Phi-4 is a local model used for cost-efficient text processing.",
            "source": "models.md",
            "title": "Models",
        },
        {
            "id": "doc_2",
            "content": "FAISS is used for vector similarity search in the local knowledge base.",
            "source": "tools.md",
            "title": "Tools",
        },
    ]


class TestVectorStore:
    def test_build_and_search(self, mock_settings, mock_embeddings, sample_documents):
        store = VectorStore(Path("/tmp/test_vs"), embedding_dim=768)
        store.build(sample_documents)

        assert store.is_loaded
        results = store.search("orchestration framework", top_k=2)
        assert len(results) <= 2
        for r in results:
            assert "content" in r
            assert "score" in r
            assert "id" in r

    def test_build_empty_raises(self, mock_settings, mock_embeddings):
        store = VectorStore(Path("/tmp/test_vs"))
        with pytest.raises(ValueError, match="No documents"):
            store.build([])

    def test_search_empty_index(self, mock_settings):
        store = VectorStore(Path("/tmp/test_vs"))
        results = store.search("anything")
        assert results == []

    def test_is_loaded_false_initially(self, mock_settings):
        store = VectorStore(Path("/tmp/test_vs"))
        assert store.is_loaded is False

    def test_save_and_load(self, mock_settings, mock_embeddings, sample_documents, tmp_path):
        store_path = tmp_path / "vectorstore"
        store = VectorStore(store_path, embedding_dim=768)
        store.build(sample_documents)
        store.save()

        # Load into a new instance
        store2 = VectorStore(store_path, embedding_dim=768)
        loaded = store2.load()
        assert loaded is True
        assert store2.is_loaded
        results = store2.search("test", top_k=1)
        assert len(results) >= 1

    def test_load_nonexistent(self, mock_settings, tmp_path):
        store = VectorStore(tmp_path / "nonexistent")
        assert store.load() is False

    def test_save_without_build_raises(self, mock_settings):
        store = VectorStore(Path("/tmp/test_vs"))
        with pytest.raises(ValueError, match="No index to save"):
            store.save()


class TestLocalSearchTool:
    def test_config(self, mock_settings):
        with patch.object(VectorStore, "load", return_value=False):
            tool = LocalSearchTool()
        assert tool.name == "local_search"
        assert tool.is_local is True
        assert tool.estimated_cost == 0.0

    def test_search_no_index(self, mock_settings):
        with patch.object(VectorStore, "load", return_value=False):
            tool = LocalSearchTool()
        result = tool.run(query="anything")
        assert result.error is not None
        assert "not initialized" in result.error
        assert result.output["results"] == []

    def test_search_with_loaded_index(self, mock_settings, mock_embeddings, sample_documents):
        with patch.object(VectorStore, "load", return_value=False):
            tool = LocalSearchTool()

        # Build index directly on the tool's store
        tool._store.build(sample_documents)

        result = tool.run(query="orchestration", top_k=2)
        assert result.success
        assert result.output["query"] == "orchestration"
        assert len(result.output["results"]) <= 2
        assert result.metadata.get("source") == "faiss"
