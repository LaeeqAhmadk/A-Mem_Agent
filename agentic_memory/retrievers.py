
from typing import Dict, Any, List
import json
import logging

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)


def _parse_list(value: Any) -> List[str]:
    """Convert a list or JSON-encoded list into a list of strings."""
    if isinstance(value, list):
        return [str(item) for item in value]

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except (json.JSONDecodeError, TypeError):
            pass

    return []


def _serialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Convert metadata into values supported by ChromaDB."""
    processed = {}

    for key, value in metadata.items():
        if value is None:
            processed[key] = ""
        elif isinstance(value, (list, dict)):
            processed[key] = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, (str, int, float, bool)):
            processed[key] = value
        else:
            processed[key] = str(value)

    return processed


def _restore_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Restore JSON-encoded collections and numeric strings where possible."""
    restored = {}

    for key, value in metadata.items():
        if not isinstance(value, str):
            restored[key] = value
            continue

        text = value.strip()

        if text.startswith("[") or text.startswith("{"):
            try:
                restored[key] = json.loads(text)
                continue
            except json.JSONDecodeError:
                pass

        # Restore numeric values only for fields normally used as numbers.
        if key in {"retrieval_count", "distance", "score"}:
            try:
                restored[key] = (
                    float(text) if "." in text else int(text)
                )
                continue
            except ValueError:
                pass

        restored[key] = value

    return restored


class ChromaRetriever:
    """Local vector retrieval using ChromaDB and sentence-transformers."""

    def __init__(
        self,
        collection_name: str = "memories",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """Initialize ChromaDB with a local embedding model."""
        self.client = chromadb.Client(Settings(allow_reset=True))

        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=model_name,
            device="cpu",
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
        )

    def add_document(
        self,
        document: str,
        metadata: Dict[str, Any],
        doc_id: str,
    ) -> None:
        """Add a memory or update an existing memory with the same ID."""
        if not isinstance(document, str) or not document.strip():
            raise ValueError("Document must be a non-empty string.")

        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError("Document ID must be a non-empty string.")

        metadata = metadata or {}
        enhanced_document = document

        context = metadata.get("context")
        if context and context != "General":
            enhanced_document += f" context: {context}"

        keywords = _parse_list(metadata.get("keywords", []))
        if keywords:
            enhanced_document += f" keywords: {', '.join(keywords)}"

        tags = _parse_list(metadata.get("tags", []))
        if tags:
            enhanced_document += f" tags: {', '.join(tags)}"

        processed_metadata = _serialize_metadata(metadata)
        processed_metadata["enhanced_content"] = enhanced_document

        # Upsert inserts a new document or updates an existing ID.
        self.collection.upsert(
            ids=[doc_id],
            documents=[enhanced_document],
            metadatas=[processed_metadata],
        )

    def delete_document(self, doc_id: str) -> None:
        """Delete a memory by its ID."""
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError("Document ID must be a non-empty string.")

        self.collection.delete(ids=[doc_id])

    def search(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Return ChromaDB search results in its standard format."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query must be a non-empty string.")

        if k <= 0 or self.collection.count() == 0:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        result_count = min(k, self.collection.count())

        results = self.collection.query(
            query_texts=[query],
            n_results=result_count,
        )

        # Restore list/dictionary metadata that was JSON-encoded.
        if results.get("metadatas"):
            for query_metadata in results["metadatas"]:
                if query_metadata:
                    for index, metadata in enumerate(query_metadata):
                        if metadata is not None:
                            query_metadata[index] = _restore_metadata(
                                metadata
                            )

        return results
