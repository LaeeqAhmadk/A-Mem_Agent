
import json
import unittest
from unittest.mock import MagicMock, patch

from agentic_memory.memory_system import AgenticMemorySystem, MemoryNote


class TestAgenticMemorySystem(unittest.TestCase):

    def setUp(self):
        """Create a fresh memory system with a mocked LLM."""
        self.llm_controller = MagicMock()
        self.llm_controller.llm.get_completion.side_effect = (
            self.fake_llm_completion
        )

        # Replace the real LLM controller before creating the system.
        self.llm_patcher = patch(
            "agentic_memory.memory_system.LLMController",
            return_value=self.llm_controller,
        )
        self.llm_patcher.start()
        self.addCleanup(self.llm_patcher.stop)

        self.memory_system = AgenticMemorySystem(
            model_name="all-MiniLM-L6-v2",
            llm_backend="groq",
            llm_model="llama-3.1-8b-instant",
        )

    def fake_llm_completion(self, prompt, *args, **kwargs):
        """
        Return predictable JSON instead of calling Groq.
        The same mock handles metadata analysis and evolution decisions.
        """
        prompt_lower = str(prompt).lower()

        if "should_evolve" in prompt_lower or "evolve" in prompt_lower:
            response = {
                "should_evolve": False,
                "reason": "No evolution needed for this test.",
                "updated_tags": [],
                "new_links": [],
            }
        else:
            response = {
                "keywords": ["AI", "memory", "testing"],
                "context": "Software Development",
                "tags": ["AI", "Testing"],
            }

        return json.dumps(response)

    def add_test_note(self, content, **kwargs):
        """Add a note and retrieve the stored MemoryNote."""
        self.memory_system.add_note(content, **kwargs)

        for note in self.memory_system.memories.values():
            if note.content == content:
                return note

        self.fail(f"Memory was not stored: {content}")

    def test_create_memory_with_metadata(self):
        """A memory should be stored with the supplied metadata."""
        note = self.add_test_note(
            "Artificial intelligence can solve complex problems.",
            keywords=["AI", "problem solving"],
            context="Research",
            tags=["AI", "Research"],
            category="Technology",
        )

        self.assertIsInstance(note, MemoryNote)
        self.assertEqual(
            note.content,
            "Artificial intelligence can solve complex problems.",
        )
        self.assertEqual(note.keywords, ["AI", "problem solving"])
        self.assertEqual(note.context, "Research")
        self.assertEqual(note.tags, ["AI", "Research"])
        self.assertEqual(note.category, "Technology")
        self.assertIsNotNone(note.id)

    def test_missing_metadata_is_generated_without_groq(self):
        """Missing metadata should be supplied by the mocked LLM."""
        note = self.add_test_note(
            "Machine learning is a branch of artificial intelligence."
        )

        self.assertTrue(note.keywords)
        self.assertEqual(note.context, "Software Development")
        self.assertTrue(note.tags)

        # Confirm the mock, rather than a live Groq request, was used.
        self.llm_controller.llm.get_completion.assert_called()

    def test_memory_is_saved_in_chromadb(self):
        """The stored memory should be retrievable through semantic search."""
        content = "ChromaDB stores vector embeddings for semantic search."
        note = self.add_test_note(
            content,
            keywords=["ChromaDB", "vectors"],
            context="Database",
            tags=["Vector Search"],
        )

        result = self.memory_system.find_related_memories(
            "ChromaDB vector embeddings",
            k=5,
        )

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        _, memory_ids = result
        self.assertIn(note.id, memory_ids)

    def test_update_memory(self):
        """Updating a memory should change its stored attributes."""
        note = self.add_test_note(
            "This is an original memory.",
            keywords=["original"],
            context="General",
            tags=["old"],
        )

        success = self.memory_system.update(
            note.id,
            content="This is an updated memory.",
            context="Updated Context",
            keywords=["updated"],
            tags=["new"],
        )

        self.assertTrue(success)

        updated_note = self.memory_system.read(note.id)
        self.assertIsNotNone(updated_note)
        self.assertEqual(updated_note.content, "This is an updated memory.")
        self.assertEqual(updated_note.context, "Updated Context")
        self.assertEqual(updated_note.keywords, ["updated"])
        self.assertEqual(updated_note.tags, ["new"])

    def test_read_existing_memory(self):
        """Reading an existing ID should return its MemoryNote."""
        note = self.add_test_note(
            "A memory that can be read.",
            keywords=["read"],
            context="Testing",
            tags=["read-test"],
        )

        result = self.memory_system.read(note.id)

        self.assertIsInstance(result, MemoryNote)
        self.assertEqual(result.id, note.id)
        self.assertEqual(result.content, note.content)

    def test_read_missing_memory(self):
        """Reading an unknown ID should return None."""
        result = self.memory_system.read("nonexistent-memory-id")
        self.assertIsNone(result)

    def test_delete_memory(self):
        """Deleting a memory should remove it from the system."""
        note = self.add_test_note(
            "This memory will be deleted.",
            keywords=["delete"],
            context="Testing",
            tags=["delete-test"],
        )

        success = self.memory_system.delete(note.id)

        self.assertTrue(success)
        self.assertIsNone(self.memory_system.read(note.id))
        self.assertNotIn(note.id, self.memory_system.memories)

    def test_delete_missing_memory(self):
        """Deleting an unknown ID should fail gracefully."""
        success = self.memory_system.delete("nonexistent-memory-id")
        self.assertFalse(success)

    def test_find_related_memories(self):
        """Related-memory search should return text and IDs."""
        note = self.add_test_note(
            "Deep learning uses neural networks to learn patterns.",
            keywords=["deep learning", "neural networks"],
            context="Artificial Intelligence",
            tags=["Deep Learning"],
        )

        result = self.memory_system.find_related_memories(
            "neural networks and deep learning",
            k=5,
        )

        self.assertIsInstance(result, tuple)
        memory_text, memory_ids = result

        self.assertIsInstance(memory_text, str)
        self.assertIsInstance(memory_ids, list)
        self.assertIn(note.id, memory_ids)

    def test_find_related_memories_raw(self):
        """The raw retrieval method should return formatted text."""
        self.add_test_note(
            "Natural language processing helps computers understand text.",
            keywords=["NLP", "text"],
            context="AI",
            tags=["NLP"],
        )

        result = self.memory_system.find_related_memories_raw(
            "natural language processing",
            k=5,
        )

        self.assertIsInstance(result, str)
        self.assertTrue(result.strip())

    def test_process_first_memory(self):
        """The first memory should not require evolution."""
        note = MemoryNote(
            content="This is the first memory.",
            keywords=["first"],
            context="Testing",
            tags=["first-memory"],
        )

        evolved, processed_note = self.memory_system.process_memory(note)

        self.assertFalse(evolved)
        self.assertIsInstance(processed_note, MemoryNote)

    def test_process_memory_with_existing_memories(self):
        """Processing a related memory should return a valid result."""
        self.add_test_note(
            "Neural networks are used in deep learning.",
            keywords=["neural networks"],
            context="AI",
            tags=["Deep Learning"],
        )

        second_note = MemoryNote(
            content="Deep learning models are trained using data.",
            keywords=["deep learning", "training"],
            context="AI",
            tags=["Deep Learning"],
        )

        evolved, processed_note = self.memory_system.process_memory(
            second_note
        )

        self.assertIsInstance(evolved, bool)
        self.assertIsInstance(processed_note, MemoryNote)

    def test_memory_relationships(self):
        """Memory links should be updateable and remain accessible."""
        first = self.add_test_note(
            "Python is used for machine learning.",
            keywords=["Python", "ML"],
            context="Programming",
            tags=["Python"],
        )

        second = self.add_test_note(
            "Scikit-learn provides machine learning algorithms.",
            keywords=["Scikit-learn", "ML"],
            context="Programming",
            tags=["Machine Learning"],
        )

        first.links = [second.id]
        success = self.memory_system.update(first.id, links=first.links)

        self.assertTrue(success)

        updated_first = self.memory_system.read(first.id)
        self.assertIn(second.id, updated_first.links)

    def test_consolidate_memories(self):
        """Consolidation should preserve the notes in system memory."""
        first = self.add_test_note(
            "Computer vision analyzes images.",
            keywords=["computer vision", "images"],
            context="AI",
            tags=["CV"],
        )

        second = self.add_test_note(
            "Image classification is a computer vision task.",
            keywords=["classification", "images"],
            context="AI",
            tags=["CV"],
        )

        self.memory_system.consolidate_memories()

        self.assertIn(first.id, self.memory_system.memories)
        self.assertIn(second.id, self.memory_system.memories)

    def test_llm_controller_is_mocked(self):
        """The tests must not depend on a live LLM API."""
        self.add_test_note("Verify that the LLM mock is active.")

        self.llm_controller.llm.get_completion.assert_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
