"""
LLM controllers for A-Mem.

Primary backend:
    Groq API with Llama 3.1 8B Instant

Other optional backends:
    OpenAI, Ollama, SGLang, and OpenRouter

For the current implementation, set:
    GROQ_API_KEY=your_groq_api_key

Install the main dependencies:
    pip install openai litellm requests python-dotenv
"""

from typing import Dict, Optional, Literal, Any
import os
import re
import json
import time
import requests
from abc import ABC, abstractmethod


# ============================================================
# Base Controller
# ============================================================

class BaseLLMController(ABC):
    """Base class for all LLM controllers."""

    @abstractmethod
    def get_completion(
        self,
        prompt: str,
        response_format: dict = None,
        temperature: float = 1.0,
        max_tokens: int = None,
    ) -> str:
        """Generate a completion from the selected LLM."""
        pass

    def _generate_empty_value(
        self,
        schema_type: str,
        schema_items: dict = None,
    ) -> Any:
        """Generate an empty value matching a JSON schema type."""

        if schema_type == "array":
            return []

        if schema_type == "string":
            return ""

        if schema_type == "object":
            return {}

        if schema_type in ("number", "integer"):
            return 0

        if schema_type == "boolean":
            return False

        return None

    def _generate_empty_response(self, response_format: dict) -> dict:
        """Generate an empty response matching the expected JSON schema."""

        if not response_format:
            return {}

        schema = (
            response_format.get("json_schema", {}).get("schema", {})
        )

        # Also support a schema supplied directly.
        if not schema and "properties" in response_format:
            schema = response_format

        result = {}

        for prop_name, prop_schema in schema.get("properties", {}).items():
            result[prop_name] = self._generate_empty_value(
                prop_schema.get("type", "string"),
                prop_schema.get("items"),
            )

        return result


# ============================================================
# Groq Controller
# ============================================================

class GroqController(BaseLLMController):
    """
    LLM controller using the Groq API.

    The default model is Llama 3.1 8B Instant.

    API key:
        Read from GROQ_API_KEY environment variable.

    Groq uses an OpenAI-compatible API, so the OpenAI Python
    package can be used with Groq's base URL.
    """

    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        api_key: Optional[str] = None,
        max_retries: int = 4,
        min_interval: float = 2.0,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The OpenAI Python package is required for Groq's "
                "OpenAI-compatible API. Install it using:\n"
                "pip install openai"
            ) from exc

        if api_key is None:
            api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "Groq API key not found. Set the GROQ_API_KEY "
                "environment variable in your .env file."
            )

        self.model = model
        self.max_retries = max(1, max_retries)
        self.min_interval = max(0.0, min_interval)
        self._last_call = 0.0

        # Disable automatic SDK retries because we handle retries
        # ourselves, including rate-limit delays.
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.BASE_URL,
            max_retries=0,
        )

    def _throttle(self):
        """Add a small delay between requests."""

        elapsed = time.time() - self._last_call
        wait = self.min_interval - elapsed

        if wait > 0:
            time.sleep(wait)

        self._last_call = time.time()

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """
        Extract a JSON object from model output.

        Handles:
        - Normal JSON responses
        - JSON enclosed in Markdown code fences
        - Extra text before or after the JSON object
        """

        if not text:
            return None

        text = text.strip()

        # First, try parsing the complete response.
        try:
            parsed = json.loads(text)

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            pass

        # Remove Markdown code fences if present.
        text = re.sub(
            r"^\s*```(?:json)?\s*|\s*```\s*$",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

        try:
            parsed = json.loads(text)

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            pass

        # Try extracting the outermost JSON object.
        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end > start:
            try:
                parsed = json.loads(text[start:end + 1])

                if isinstance(parsed, dict):
                    return parsed

            except json.JSONDecodeError:
                pass

        return None

    def get_completion(
        self,
        prompt: str,
        response_format: dict = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """
        Generate a JSON completion using Groq.

        A-Mem uses structured information for memory creation,
        memory linking, and memory evolution. Therefore, this
        controller asks the model to return valid JSON.
        """

        from openai import (
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            AuthenticationError,
            BadRequestError,
        )

        schema = (
            (response_format or {})
            .get("json_schema", {})
            .get("schema")
        )

        system_message = (
            "You are an assistant helping an agentic memory system. "
            "Follow the user's instructions carefully. "
            "Return exactly one valid JSON object. "
            "Do not include Markdown code fences or explanations "
            "outside the JSON object."
        )

        if schema:
            system_message += (
                "\nThe JSON object must follow this schema:\n"
                + json.dumps(schema, ensure_ascii=False)
            )

        # Ask Groq for JSON mode. The schema is also included in
        # the system message because JSON mode does not guarantee
        # that every schema rule will be followed.
        kwargs = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_error = None

        for attempt in range(1, self.max_retries + 1):
            self._throttle()

            try:
                response = self.client.chat.completions.create(
                    **kwargs
                )

                content = response.choices[0].message.content or ""
                parsed = self._extract_json(content)

                if parsed is not None:
                    return json.dumps(
                        parsed,
                        ensure_ascii=False,
                    )

                last_error = ValueError(
                    "The model response was not a valid JSON object."
                )

                print(
                    f"[Groq] Invalid JSON response "
                    f"({attempt}/{self.max_retries})."
                )

            except RateLimitError as exc:
                last_error = exc

                retry_after = None

                try:
                    retry_after = float(
                        exc.response.headers.get("retry-after")
                    )
                except (AttributeError, TypeError, ValueError):
                    pass

                delay = (
                    retry_after
                    if retry_after is not None
                    else min(2 ** attempt, 30)
                )

                print(
                    f"[Groq] Rate limit reached. "
                    f"Waiting {delay:.1f} seconds before retry."
                )

                if attempt < self.max_retries:
                    time.sleep(delay)

            except (
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
            ) as exc:
                last_error = exc
                delay = min(2 ** attempt, 20)

                print(
                    f"[Groq] Temporary connection/server error: {exc}. "
                    f"Retrying in {delay} seconds."
                )

                if attempt < self.max_retries:
                    time.sleep(delay)

            except (AuthenticationError, BadRequestError) as exc:
                # These usually indicate an invalid API key,
                # unsupported model, or invalid request.
                raise RuntimeError(
                    f"Groq rejected the request: {exc}. "
                    "Check your API key, model name, and request."
                ) from exc

            except Exception as exc:
                last_error = exc

                print(
                    f"[Groq] Unexpected error "
                    f"({attempt}/{self.max_retries}): {exc}"
                )

                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 10))

        # Do not silently pretend that the LLM succeeded.
        # Raise a clear error so we can diagnose the problem.
        raise RuntimeError(
            "Groq failed to generate a valid JSON response after "
            f"{self.max_retries} attempts. Last error: {last_error}"
        )


# ============================================================
# OpenAI Controller (Optional)
# ============================================================

class OpenAIController(BaseLLMController):
    """Optional controller for users who have an OpenAI API key."""

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "Install the OpenAI package using: pip install openai"
            ) from exc

        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY."
            )

        self.model = model
        self.client = OpenAI(api_key=api_key)

    def get_completion(
        self,
        prompt: str,
        response_format: dict = None,
        temperature: float = 1.0,
        max_tokens: int = None,
    ) -> str:

        kwargs = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return a valid JSON object.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": temperature,
        }

        if response_format:
            kwargs["response_format"] = response_format

        if max_tokens is not None:
            if (
                "gpt-5" in self.model.lower()
                or "o1" in self.model.lower()
                or "o3" in self.model.lower()
            ):
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["max_tokens"] = max_tokens

        response = self.client.chat.completions.create(**kwargs)

        return response.choices[0].message.content or ""


# ============================================================
# Ollama Controller (Local Models)
# ============================================================

class OllamaController(BaseLLMController):
    """
    Controller for locally running models through Ollama.

    This will be useful when we migrate away from Groq later.
    It does not require a paid API key.
    """

    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model

        try:
            import ollama
        except ImportError as exc:
            raise ImportError(
                "Install the Ollama Python package using: "
                "pip install ollama"
            ) from exc

        self.ollama = ollama

    def get_completion(
        self,
        prompt: str,
        response_format: dict = None,
        temperature: float = 0.2,
        max_tokens: int = None,
    ) -> str:

        options = {"temperature": temperature}

        if max_tokens is not None:
            options["num_predict"] = max_tokens

        try:
            kwargs = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Return a valid JSON object only.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "options": options,
                "format": "json",
            }

            response = self.ollama.chat(**kwargs)

            content = response["message"]["content"]

            parsed = GroqController._extract_json(content)

            if parsed is None:
                raise ValueError(
                    "Ollama returned invalid JSON."
                )

            return json.dumps(parsed, ensure_ascii=False)

        except Exception as exc:
            raise RuntimeError(
                f"Ollama completion failed: {exc}"
            ) from exc


# ============================================================
# SGLang Controller (Optional Local Server)
# ============================================================

class SGLangController(BaseLLMController):
    """Controller for an SGLang server."""

    def __init__(
        self,
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        sglang_host: str = "http://localhost",
        sglang_port: int = 30000,
    ):
        self.model = model
        self.base_url = f"{sglang_host}:{sglang_port}"

    def get_completion(
        self,
        prompt: str,
        response_format: dict = None,
        temperature: float = 0.2,
        max_tokens: int = 1000,
    ) -> str:

        payload = {
            "text": (
                "Return a valid JSON object only.\n\n"
                + prompt
            ),
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
            },
        }

        try:
            response = requests.post(
                f"{self.base_url}/generate",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )

            response.raise_for_status()

            result = response.json()
            generated_text = result.get("text", "")

            parsed = GroqController._extract_json(generated_text)

            if parsed is None:
                raise ValueError(
                    "SGLang returned invalid JSON."
                )

            return json.dumps(parsed, ensure_ascii=False)

        except Exception as exc:
            raise RuntimeError(
                f"SGLang completion failed: {exc}"
            ) from exc


# ============================================================
# OpenRouter Controller (Optional)
# ============================================================

class OpenRouterController(BaseLLMController):
    """Optional controller for OpenRouter through LiteLLM."""

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        api_key: Optional[str] = None,
    ):
        if not model.startswith("openrouter/"):
            self.model = f"openrouter/{model}"
        else:
            self.model = model

        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError(
                "OpenRouter API key not found. "
                "Set OPENROUTER_API_KEY."
            )

        os.environ["OPENROUTER_API_KEY"] = api_key

    def get_completion(
        self,
        prompt: str,
        response_format: dict = None,
        temperature: float = 0.2,
        max_tokens: int = None,
    ) -> str:

        try:
            from litellm import completion
        except ImportError as exc:
            raise ImportError(
                "Install LiteLLM using: pip install litellm"
            ) from exc

        kwargs = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return a valid JSON object only.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": temperature,
        }

        if response_format:
            kwargs["response_format"] = response_format

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = completion(**kwargs)

        return response.choices[0].message.content or ""


# ============================================================
# Main LLM Controller
# ============================================================

class LLMController:
    """
    Main LLM controller for A-Mem.

    The default backend is Groq, not OpenAI.

    Example:
        controller = LLMController()

    Or explicitly:
        controller = LLMController(
            backend="groq",
            model="llama-3.1-8b-instant"
        )
    """

    def __init__(
        self,
        backend: Literal[
            "groq",
            "openai",
            "ollama",
            "sglang",
            "openrouter",
        ] = "groq",
        model: str = "llama-3.1-8b-instant",
        api_key: Optional[str] = None,
        sglang_host: str = "http://localhost",
        sglang_port: int = 30000,
    ):

        if backend == "groq":
            self.llm = GroqController(
                model=model,
                api_key=api_key,
            )

        elif backend == "openai":
            self.llm = OpenAIController(
                model=model,
                api_key=api_key,
            )

        elif backend == "ollama":
            self.llm = OllamaController(model=model)

        elif backend == "sglang":
            self.llm = SGLangController(
                model=model,
                sglang_host=sglang_host,
                sglang_port=sglang_port,
            )

        elif backend == "openrouter":
            self.llm = OpenRouterController(
                model=model,
                api_key=api_key,
            )

        else:
            raise ValueError(
                "Unsupported backend. Choose groq, openai, ollama, "
                "sglang, or openrouter."
            )

        self.backend = backend
        self.model = model

    def get_completion(
        self,
        prompt: str,
        response_format: dict = None,
        temperature: float = 0.2,
        max_tokens: int = None,
    ) -> str:
        """Generate a completion using the selected backend."""

        return self.llm.get_completion(
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
        )