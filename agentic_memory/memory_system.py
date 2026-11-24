
import keyword
from typing import List, Dict, Optional, Any, Tuple
import uuid
from datetime import datetime
from .llm_controller import LLMController
from .retrievers import ChromaRetriever
import json
import logging
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import word_tokenize
import pickle
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class MemoryNote:
    """A memory note representing a single unit of information."""

    def __init__(
        self,
        content: str,
        id: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        links: Optional[Dict] = None,
        retrieval_count: Optional[int] = None,
        timestamp: Optional[str] = None,
        last_accessed: Optional[str] = None,
        context: Optional[str] = None,
        evolution_history: Optional[List] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        self.content = content
        self.id = id or str(uuid.uuid4())

        self.keywords = keywords or []
        self.links = links or []
        self.context = context or "General"
        self.category = category or "Uncategorized"
        self.tags = tags or []

        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time

        self.retrieval_count = (
            retrieval_count if retrieval_count is not None else 0
        )
        self.evolution_history = evolution_history or []


class AgenticMemorySystem:
    """Core system for creating, retrieving, updating, and evolving memories."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        llm_backend: str = "groq",
        llm_model: str = "openai/gpt-oss-20b",
        evo_threshold: int = 100,
        api_key: Optional[str] = None,
        sglang_host: str = "http://localhost",
        sglang_port: int = 30000,
        llm_temperature: float = 0.3,
    ):
        """Initialize the memory system."""
        load_dotenv()
        self.memories = {}
        self.model_name = model_name
        self.llm_temperature = llm_temperature

        # Create the retriever without resetting the ChromaDB client.
        self.retriever = ChromaRetriever(
            collection_name="memories",
            model_name=self.model_name,
        )
        # Restore previously saved memories from persistent ChromaDB.
        self._load_existing_memories()

        # Initialize the selected LLM backend.
        self.llm_controller = LLMController(
            llm_backend,
            llm_model,
            api_key,
            sglang_host,
            sglang_port,
        )

        self.evo_cnt = 0
        self.evo_threshold = max(1, evo_threshold)

        self._evolution_system_prompt = """
You are an AI memory evolution agent responsible for managing and evolving
a knowledge base. Analyze the new memory note and its nearest neighbors.

New memory context:
{context}

New memory content:
{content}

New memory keywords:
{keywords}

Nearest-neighbor memories:
{nearest_neighbors_memories}

Decide whether the new memory should evolve and whether to update connections
or neighboring memories.

Return a JSON object with this structure:
{{
    "should_evolve": true,
    "actions": ["strengthen", "update_neighbor"],
    "suggested_connections": ["memory_id_1"],
    "tags_to_update": ["tag_1"],
    "new_context_neighborhood": ["context for neighbor 1"],
    "new_tags_neighborhood": [["tag for neighbor 1"]]
}}

The number of neighbors is {neighbor_number}.
Use only the supplied memory IDs. Keep the neighborhood arrays aligned
with the order of the supplied neighbors. Do not invent memory IDs.
"""

    
    def _load_existing_memories(self) -> None:
        """Restore MemoryNote objects from the persistent ChromaDB collection."""
        try:
            results = self.retriever.get_all_documents()

            ids = results.get("ids", [])
            metadatas = results.get("metadatas", [])

            for memory_id, metadata in zip(ids, metadatas):
                metadata = metadata or {}

                content = metadata.get("content", "")

                if not content:
                    logger.warning(
                        "Skipping memory %s because its original content is missing.",
                        memory_id,
                    )
                    continue

                note = MemoryNote(
                    content=content,
                    id=memory_id,
                    keywords=metadata.get("keywords", []),
                    links=metadata.get("links", []),
                    retrieval_count=metadata.get("retrieval_count", 0),
                    timestamp=metadata.get("timestamp"),
                    last_accessed=metadata.get("last_accessed"),
                    context=metadata.get("context", "General"),
                    evolution_history=metadata.get("evolution_history", []),
                    category=metadata.get("category", "Uncategorized"),
                    tags=metadata.get("tags", []),
                )

                self.memories[memory_id] = note

            logger.info(
                "Restored %d memories from persistent ChromaDB.",
                len(self.memories),
            )

            print(f"Restored {len(self.memories)} existing memories.")

        except Exception:
            logger.exception("Failed to restore memories from ChromaDB.")
            raise
    @staticmethod
    def _to_bool(value: Any) -> bool:
        """Convert a Boolean-like value returned by an LLM to bool."""
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return bool(value)

    def analyze_content(self, content: str) -> Dict:
        """Use the configured LLM to extract keywords, context, and tags."""

        prompt = f"""
Analyze the following content and return a JSON object containing:
- keywords: at least three specific, distinct keywords
- context: one sentence describing the main topic and purpose
- tags: at least three broad, useful classification tags

Content:
{content}
"""

        try:
            response = self.llm_controller.llm.get_completion(
                prompt,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "keywords": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "context": {"type": "string"},
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["keywords", "context", "tags"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                },
                temperature=self.llm_temperature,
            )

            result = json.loads(response)
            if not isinstance(result, dict):
                raise ValueError("LLM analysis did not return a JSON object.")

            return {
                "keywords": result.get("keywords", []),
                "context": result.get("context", "General"),
                "tags": result.get("tags", []),
            }

        except Exception as exc:
            logger.warning("Content analysis failed: %s", exc)
            return {"keywords": [], "context": "General", "tags": []}

    def _metadata_for(self, note: MemoryNote) -> Dict[str, Any]:
        """Build ChromaDB-compatible metadata for a memory."""
        return {
            "id": note.id,
            "content": note.content,
            "keywords": note.keywords,
            "links": note.links,
            "retrieval_count": note.retrieval_count,
            "timestamp": note.timestamp,
            "last_accessed": note.last_accessed,
            "context": note.context,
            "evolution_history": note.evolution_history,
            "category": note.category,
            "tags": note.tags,
        }

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        """Create, analyze, and store a new memory."""

        if time is not None:
            kwargs["timestamp"] = time

        note = MemoryNote(content=content, **kwargs)

        needs_analysis = (
            not note.keywords
            or note.context == "General"
            or not note.tags
        )

        if needs_analysis:
            analysis = self.analyze_content(content)

            if not note.keywords:
                note.keywords = analysis.get("keywords", [])

            if note.context == "General":
                note.context = analysis.get("context", "General")

            if not note.tags:
                note.tags = analysis.get("tags", [])

        evo_label, note = self.process_memory(note)

        # Store the memory in the in-process memory dictionary.
        self.memories[note.id] = note

        # Add or update the vector record.
        self.retriever.add_document(
            note.content,
            self._metadata_for(note),
            note.id,
        )

        if evo_label:
            self.evo_cnt += 1

            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()

        return note.id

    def consolidate_memories(self):
        """Rebuild the collection from the current in-process memories."""

        # Clear only the memories collection, not the entire ChromaDB client.
        existing_ids = self.retriever.collection.get().get("ids", [])

        if existing_ids:
            self.retriever.collection.delete(ids=existing_ids)

        # Reinsert every known memory with its current metadata.
        for memory in self.memories.values():
            self.retriever.add_document(
                memory.content,
                self._metadata_for(memory),
                memory.id,
            )

    def find_related_memories(
        self,
        query: str,
        k: int = 5,
    ) -> Tuple[str, List[str]]:
        """Return formatted nearest memories and their IDs."""

        if not self.memories:
            return "", []

        try:
            results = self.retriever.search(query, k)
            memory_lines = []
            memory_ids = []

            result_ids = results.get("ids", [[]])
            result_metadata = results.get("metadatas", [[]])

            if not result_ids or not result_ids[0]:
                return "", []

            metadata_rows = result_metadata[0] if result_metadata else []

            for index, memory_id in enumerate(result_ids[0]):
                metadata = (
                    metadata_rows[index]
                    if index < len(metadata_rows) and metadata_rows[index]
                    else {}
                )

                memory_lines.append(
                    f"memory_id:{memory_id}\t"
                    f"talk start time:{metadata.get('timestamp', '')}\t"
                    f"memory content:{metadata.get('content', '')}\t"
                    f"memory context:{metadata.get('context', '')}\t"
                    f"memory keywords:{metadata.get('keywords', [])}\t"
                    f"memory tags:{metadata.get('tags', [])}"
                )
                memory_ids.append(memory_id)

            return "\n".join(memory_lines), memory_ids

        except Exception:
            logger.exception("Error finding related memories")
            return "", []

    def find_related_memories_raw(self, query: str, k: int = 5) -> str:
        """Return related memories and any linked memories as text."""

        memory_text, memory_ids = self.find_related_memories(query, k)
        if not memory_ids:
            return ""

        lines = [memory_text]

        for memory_id in memory_ids:
            memory = self.memories.get(memory_id)
            if not memory:
                continue

            for link_id in memory.links:
                neighbor = self.memories.get(link_id)
                if neighbor:
                    lines.append(
                        f"talk start time:{neighbor.timestamp}\t"
                        f"memory content:{neighbor.content}\t"
                        f"memory context:{neighbor.context}\t"
                        f"memory keywords:{neighbor.keywords}\t"
                        f"memory tags:{neighbor.tags}"
                    )

        return "\n".join(lines)
    
    def _update_memory_access(self, memory: MemoryNote) -> None:
        """Update and persist memory access statistics."""

        memory.retrieval_count += 1
        memory.last_accessed = datetime.now().strftime("%Y%m%d%H%M")

        self.retriever.add_document(
        memory.content,
        self._metadata_for(memory),
        memory.id,
    )

    def read(self, memory_id: str) -> Optional[MemoryNote]:
        """Retrieve a memory by its ID and update access statistics."""

        memory = self.memories.get(memory_id)

        if memory is not None:
            self._update_memory_access(memory)

        return memory

    def update(self, memory_id: str, **kwargs) -> bool:
        """Update an existing memory and its vector record."""

        note = self.memories.get(memory_id)
        if note is None:
            return False

        for key, value in kwargs.items():
            if hasattr(note, key):
                setattr(note, key, value)

        self.retriever.add_document(
            note.content,
            self._metadata_for(note),
            note.id,
        )
        return True

    def delete(self, memory_id: str) -> bool:
        """Delete a memory from both the dictionary and vector store."""

        if memory_id not in self.memories:
            return False

        self.retriever.delete_document(memory_id)
        del self.memories[memory_id]
        return True

    def _search_raw(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Return IDs and vector distances from ChromaDB."""

        results = self.retriever.search(query, k)
        ids = results.get("ids", [[]])
        distances = results.get("distances", [[]])

        if not ids or not ids[0]:
            return []

        distance_values = distances[0] if distances else []

        return [
            {
                "id": memory_id,
                "score": (
                    distance_values[index]
                    if index < len(distance_values)
                    else None
                ),
            }
            for index, memory_id in enumerate(ids[0])
        ]

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search memories by semantic similarity and update access statistics."""

        if k <= 0:
            return []

        results = self.retriever.search(query, k)
        result_ids = results.get("ids", [[]])
        distances = results.get("distances", [[]])

        if not result_ids or not result_ids[0]:
            return []

        distance_values = distances[0] if distances else []
        found = []

        for index, memory_id in enumerate(result_ids[0]):
            memory = self.memories.get(memory_id)

            if memory is None:
                continue

        self._update_memory_access(memory)

        found.append(
            {
                "id": memory_id,
                "content": memory.content,
                "context": memory.context,
                "keywords": memory.keywords,
                "tags": memory.tags,
                "score": (
                    distance_values[index]
                    if index < len(distance_values)
                    else None
                ),
            }
        )

        return found[:k]

    def _search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Compatibility wrapper for the current semantic search method."""
        return self.search(query, k)

    
    def search_agentic(
        self,
        query: str,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search memories and prioritize their linked neighbors."""

        if not self.memories or k <= 0:
            return []

        try:
            results = self.retriever.search(query, k)

            result_ids = results.get("ids", [[]])
            distances = results.get("distances", [[]])

            if not result_ids or not result_ids[0]:
                return []

            distance_values = distances[0] if distances else []

            memories = []
            seen_ids = set()

            for index, memory_id in enumerate(result_ids[0]):

                if len(memories) >= k:
                    break

                memory = self.memories.get(memory_id)

                if memory is None or memory_id in seen_ids:
                    continue

                # Add the primary semantic search result.
                self._update_memory_access(memory)

                memories.append(
                    {
                        "id": memory_id,
                        "content": memory.content,
                        "context": memory.context,
                        "keywords": memory.keywords,
                        "tags": memory.tags,
                        "timestamp": memory.timestamp,
                        "category": memory.category,
                        "is_neighbor": False,
                        "score": (
                            distance_values[index]
                            if index < len(distance_values)
                            else None
                        ),
                    }
                )

                seen_ids.add(memory_id)

                # Immediately add linked neighbors.
                for link_id in memory.links:

                    if len(memories) >= k:
                        break

                    if link_id in seen_ids:
                        continue

                    neighbor = self.memories.get(link_id)

                    if neighbor is None:
                        continue

                    self._update_memory_access(neighbor)

                    memories.append(
                        {
                            "id": link_id,
                            "content": neighbor.content,
                            "context": neighbor.context,
                            "keywords": neighbor.keywords,
                            "tags": neighbor.tags,
                            "timestamp": neighbor.timestamp,
                            "category": neighbor.category,
                            "is_neighbor": True,
                        }
                    )

                    seen_ids.add(link_id)

            return memories[:k]

        except Exception:
            logger.exception("Error in search_agentic")
            return []


    def process_memory(self, note: MemoryNote) -> Tuple[bool, MemoryNote]:
        """Evaluate a memory and optionally update links or neighbors."""

        if not self.memories:
            return False, note

        try:
            neighbors_text, memory_ids = self.find_related_memories(
                note.content,
                k=5,
            )

            if not neighbors_text or not memory_ids:
                return False, note

            prompt = self._evolution_system_prompt.format(
                content=note.content,
                context=note.context,
                keywords=note.keywords,
                nearest_neighbors_memories=neighbors_text,
                neighbor_number=len(memory_ids),
            )

            response = self.llm_controller.llm.get_completion(
                prompt,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "should_evolve": {"type": "boolean"},
                                "actions": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "suggested_connections": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "tags_to_update": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "new_context_neighborhood": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "new_tags_neighborhood": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                            },
                            "required": [
                                "should_evolve",
                                "actions",
                                "suggested_connections",
                                "tags_to_update",
                                "new_context_neighborhood",
                                "new_tags_neighborhood",
                            ],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                },
                temperature=self.llm_temperature,
            )

            data = json.loads(response)
            should_evolve = self._to_bool(
                data.get("should_evolve", False)
            )

            if not should_evolve:
                return False, note

            for action in data.get("actions", []):
                if action == "strengthen":
                    for candidate in data.get("suggested_connections", []):
                        candidate = str(candidate).replace(
                            "memory_id:", ""
                        ).strip()

                        if (
                            candidate in memory_ids
                            and candidate not in note.links
                        ):
                            note.links.append(candidate)

                    new_tags = data.get("tags_to_update", [])
                    if isinstance(new_tags, list) and new_tags:
                        note.tags = [str(tag) for tag in new_tags]

                elif action == "update_neighbor":
                    new_contexts = data.get(
                        "new_context_neighborhood", []
                    )
                    new_tags_list = data.get(
                        "new_tags_neighborhood", []
                    )

                    for index, memory_id in enumerate(memory_ids):
                        neighbor = self.memories.get(memory_id)
                        if neighbor is None:
                            continue

                        if index < len(new_contexts):
                            new_context = new_contexts[index]
                            if (
                                isinstance(new_context, str)
                                and new_context.strip()
                            ):
                                neighbor.context = new_context

                        if index < len(new_tags_list):
                            new_tags = new_tags_list[index]
                            if isinstance(new_tags, list) and new_tags:
                                neighbor.tags = [
                                    str(tag) for tag in new_tags
                                ]

                        self.retriever.add_document(
                            neighbor.content,
                            self._metadata_for(neighbor),
                            neighbor.id,
                        )

            return True, note

        except Exception:
            logger.exception("Error processing memory evolution")
            return False, note
