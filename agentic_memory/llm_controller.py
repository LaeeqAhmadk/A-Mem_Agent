"""
LLM controllers for A-Mem.

Primary backend:
    Groq API using direct HTTP requests.

Other optional backends:
    OpenAI, Ollama, SGLang, and OpenRouter

For the current implementation, set:
    GROQ_API_KEY=your_groq_api_key

Main dependencies:
    pip install requests python-dotenv

Optional dependencies:
    pip install openai ollama litellm
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
    LLM controller using the Groq API through direct HTTP requests.

    This implementation does not depend on the OpenAI Python SDK
    or its compiled jiter extension.

    API key:
        Read from GROQ_API_KEY environment variable.
    """

    BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        model: str = "openai/gpt-oss-20b",
        api_key: Optional[str] = None,
        max_retries: int = 4,
        min_interval: float = 2.0,
    ):
        if api_key is None:
            api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "Groq API key not found. Set the GROQ_API_KEY "
                "environment variable in your .env file."
            )

        self.model = model
        self.api_key = api_key
        self.max_retries = max(1, max_retries)
        self.min_interval = max(0.0, min_interval)
        self._last_call = 0.0

        self.session = requests.Session()

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _throttle(self):
        """Add a small delay between API requests."""

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

        try:
            parsed = json.loads(text)

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            pass

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
        Generate a JSON completion using the Groq REST API.

        A-Mem uses structured information for memory creation,
        memory linking, and memory evolution.
        """

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

        # LLMController may explicitly pass None.
        # Use a safe default instead of sending null to Groq.
        if max_tokens is None:
            max_tokens = 2048

        payload = {
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
            "response_format": {
                "type": "json_object"
            },
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_error = None

        for attempt in range(1, self.max_retries + 1):

            self._throttle()

            try:
                response = self.session.post(
                    self.BASE_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=(15, 120),
                )

                # Handle rate limits.
                if response.status_code == 429:

                    last_error = RuntimeError(
                        f"Groq rate limit: {response.text[:500]}"
                    )

                    retry_after = response.headers.get("retry-after")

                    try:
                        delay = float(retry_after)
                    except (TypeError, ValueError):
                        delay = min(2 ** attempt, 30)

                    print(
                        f"[Groq] Rate limit reached. "
                        f"Waiting {delay:.1f} seconds."
                    )

                    if attempt < self.max_retries:
                        time.sleep(delay)

                    continue

                # Handle invalid requests and authentication errors.
                if response.status_code in (400, 401, 403, 404):

                    raise RuntimeError(
                        f"Groq rejected the request "
                        f"(HTTP {response.status_code}): "
                        f"{response.text[:1000]}. "
                        "Check your API key, model name, and request."
                    )

                # Handle temporary Groq server errors.
                if response.status_code >= 500:

                    last_error = RuntimeError(
                        f"Groq server error: HTTP "
                        f"{response.status_code}: "
                        f"{response.text[:500]}"
                    )

                    delay = min(2 ** attempt, 20)

                    print(
                        f"[Groq] Server error. "
                        f"Retrying in {delay} seconds."
                    )

                    if attempt < self.max_retries:
                        time.sleep(delay)

                    continue

                response.raise_for_status()

                result = response.json()

                content = (
                    result.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content") or ""
                )

                parsed = self._extract_json(content)

                if parsed is not None:

                    return json.dumps(
                        parsed,
                        ensure_ascii=False,
                    )

                last_error = ValueError(
                    "The model response was not a valid JSON object. "
                    f"Response: {str(result)[:1000]}"
                )

                print(
                    f"[Groq] Invalid JSON response "
                    f"({attempt}/{self.max_retries})."
                )

            except requests.exceptions.RequestException as exc:

                last_error = exc
                delay = min(2 ** attempt, 20)

                print(
                    f"[Groq] Connection error: {exc}. "
                    f"Retrying in {delay} seconds."
                )

                if attempt < self.max_retries:
                    time.sleep(delay)

            except RuntimeError:
                raise

            except Exception as exc:

                last_error = exc

                print(
                    f"[Groq] Unexpected error "
                    f"({attempt}/{self.max_retries}): {exc}"
                )

                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 10))

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
            model="openai/gpt-oss-20b"
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
        model: str = "openai/gpt-oss-20b",
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