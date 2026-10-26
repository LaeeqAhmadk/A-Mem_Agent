
"""Test utilities for the A-Mem memory system."""

import json
from typing import Any, Dict, List, Optional

from agentic_memory.llm_controller import BaseLLMController


class MockLLMController(BaseLLMController):
    """Mock LLM controller for testing without an external API."""

    def __init__(
        self,
        mock_response: Optional[str] = None,
    ):
        self.mock_response = (
            mock_response
            if mock_response is not None
            else json.dumps(
                {
                    "keywords": ["AI", "testing"],
                    "context": "General",
                    "tags": ["AI", "Testing"],
                    "should_evolve": False,
                    "reason": "Mock response for testing.",
                    "updated_tags": [],
                    "new_links": [],
                }
            )
        )

        self.completion_calls: List[Dict[str, Any]] = []

    def get_completion(
        self,
        prompt: str,
        response_format: dict = None,
        temperature: float = 0.7,
    ) -> str:
        """Return a predefined response without calling a real LLM."""

        self.completion_calls.append(
            {
                "prompt": prompt,
                "response_format": response_format,
                "temperature": temperature,
            }
        )

        return self.mock_response

    def get_embedding(self, text: str) -> List[float]:
        """Return a deterministic 384-dimensional mock vector."""

        return [0.0] * 384

    def set_response(self, response: str) -> None:
        """Set the response returned by future completion calls."""

        self.mock_response = response

    def set_json_response(self, response: Dict[str, Any]) -> None:
        """Set a response using a Python dictionary."""

        self.mock_response = json.dumps(response)

    def reset_calls(self) -> None:
        """Clear the recorded completion calls."""

        self.completion_calls.clear()
