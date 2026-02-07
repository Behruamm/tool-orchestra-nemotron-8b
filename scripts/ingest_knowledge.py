"""
Knowledge Base Ingestion Script.

Reads documents from data/knowledge/, chunks them, embeds with Gemini,
and builds a FAISS index for the local_search tool.

Usage:
    python -m scripts.ingest_knowledge
    python -m scripts.ingest_knowledge --knowledge-dir ./data/knowledge
"""

import argparse
import logging
from pathlib import Path

from src.config import get_settings
from src.tools.local_search import VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".py", ".yaml", ".yml"}


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - chunk_overlap

    return chunks


def load_documents(knowledge_dir: Path, chunk_size: int, chunk_overlap: int) -> list[dict[str, str]]:
    """Load and chunk all documents from the knowledge directory."""
    documents: list[dict[str, str]] = []

    if not knowledge_dir.exists():
        logger.warning("Knowledge directory does not exist: %s", knowledge_dir)
        return documents

    files = [f for f in knowledge_dir.rglob("*") if f.is_file() and f.suffix in SUPPORTED_EXTENSIONS]

    if not files:
        logger.warning("No supported files found in %s", knowledge_dir)
        return documents

    for file_path in sorted(files):
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("Skipping non-UTF-8 file: %s", file_path)
            continue

        if not text.strip():
            continue

        chunks = chunk_text(text, chunk_size, chunk_overlap)
        for i, chunk in enumerate(chunks):
            documents.append({
                "id": f"{file_path.stem}_{i}",
                "content": chunk,
                "source": str(file_path.relative_to(knowledge_dir)),
                "title": file_path.stem.replace("_", " ").title(),
            })

    logger.info("Loaded %d chunks from %d files", len(documents), len(files))
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest knowledge base into FAISS vector store")
    parser.add_argument(
        "--knowledge-dir",
        type=Path,
        default=None,
        help="Path to knowledge base directory (default: data/knowledge/)",
    )
    args = parser.parse_args()

    settings = get_settings()
    knowledge_dir = args.knowledge_dir or settings.knowledge_dir
    store_path = Path(settings.vector_store.path)
    chunk_size = settings.vector_store.chunk_size
    chunk_overlap = settings.vector_store.chunk_overlap

    logger.info("Knowledge dir: %s", knowledge_dir)
    logger.info("Vector store path: %s", store_path)
    logger.info("Chunk size: %d, overlap: %d", chunk_size, chunk_overlap)

    # Load and chunk documents
    documents = load_documents(knowledge_dir, chunk_size, chunk_overlap)
    if not documents:
        logger.error("No documents found. Add files to %s", knowledge_dir)
        return

    # Build and save FAISS index
    store = VectorStore(store_path)
    store.build(documents)
    store.save()

    logger.info("Done! Indexed %d chunks into %s", len(documents), store_path)


if __name__ == "__main__":
    main()
