from typing import Dict, Optional, Literal, Any
import os
import re
import json
import time
from abc import ABC, abstractmethod
from litellm import completion
import requests

class BaseLLMController(ABC):
    @abstractmethod
    def get_completion(self, prompt: str) -> str:
        """Get completion from LLM"""
        pass

    def _generate_empty_value(self, schema_type: str, schema_items: dict = None) -> Any:
        """Generate empty value based on JSON schema type."""
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type == "number" or schema_type == "integer":
            return 0
        elif schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: dict) -> dict:
        """Generate empty response matching the expected schema."""
        if not response_format or "json_schema" not in response_format:
            return {}

        schema = response_format["json_schema"]["schema"]
        result = {}

        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = self._generate_empty_value(
                    prop_schema["type"],
                    prop_schema.get("items")
                )

        return result

class OpenAIController(BaseLLMController):
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
            self.model = model
            if api_key is None:
                api_key = os.getenv('OPENAI_API_KEY')
            if api_key is None:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 1.0, max_tokens: int = None) -> str:
        # Build kwargs dynamically based on model type
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt}
            ],
            "response_format": response_format,
            "temperature": temperature,
        }

        # GPT-5 and newer reasoning models use max_completion_tokens
        if max_tokens is not None:
            if "gpt-5" in self.model.lower() or "o1" in self.model.lower() or "o3" in self.model.lower():
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["max_tokens"] = max_tokens

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


class GroqController(BaseLLMController):
    """LLM controller for the Groq API (free tier), via its OpenAI-compatible endpoint.

    Why this differs from OpenAIController:
      * Smaller Groq models (e.g. llama-3.1-8b-instant) may not support strict
        `json_schema` structured outputs, so we use plain JSON mode
        (`{"type": "json_object"}`) and put the schema into the prompt instead.
      * The free tier is rate limited, so calls are throttled and 429 errors
        are retried with backoff.
      * Small models sometimes return malformed JSON, so the reply is parsed
        tolerantly and retried if it still can't be parsed.

    Args:
        model: Groq model id, e.g. "llama-3.1-8b-instant".
        api_key: Groq key. If None, reads GROQ_API_KEY from the environment.
        max_retries: Attempts per call (rate limits, network errors, bad JSON).
        min_interval: Minimum seconds between requests. 2.0 matches ~30 requests/min;
            raise it if you still get 429 errors, lower it if your limits are higher.
    """
    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self, model: str = "llama-3.1-8b-instant", api_key: Optional[str] = None,
                 max_retries: int = 5, min_interval: float = 2.0):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")

        if api_key is None:
            api_key = os.getenv("GROQ_API_KEY")
        if api_key is None:
            raise ValueError("Groq API key not found. Set GROQ_API_KEY environment variable.")

        self.model = model
        self.max_retries = max_retries
        self.min_interval = min_interval
        self._last_call = 0.0
        # max_retries=0: we handle retries ourselves so we can honour Groq's retry-after header
        self.client = OpenAI(api_key=api_key, base_url=self.BASE_URL, max_retries=0)

    def _throttle(self):
        wait = self.min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """Parse a JSON object from model output, tolerating code fences and extra text."""
        if not text:
            return None
        text = text.strip()
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
        # Strip ```json fences, then fall back to the outermost {...}
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                parsed = json.loads(text[start:end + 1])
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None
        return None

    def get_completion(self, prompt: str, response_format: dict = None,
                       temperature: float = 1.0, max_tokens: int = None) -> str:
        from openai import RateLimitError, APIConnectionError, APITimeoutError, InternalServerError

        system_msg = "You must respond with a JSON object."
        schema = (response_format or {}).get("json_schema", {}).get("schema")
        if schema:
            system_msg += (" Return only the JSON object, with no extra text, "
                           "matching this JSON schema exactly:\n" + json.dumps(schema))

        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                parsed = self._extract_json(content)
                if parsed is not None:
                    return json.dumps(parsed)
                print(f"[Groq] Invalid JSON (attempt {attempt}/{self.max_retries}), retrying...")
            except RateLimitError as e:
                retry_after = None
                try:
                    retry_after = float(e.response.headers.get("retry-after"))
                except (AttributeError, TypeError, ValueError):
                    pass
                delay = retry_after if retry_after else min(2 ** attempt, 60)
                print(f"[Groq] Rate limited (attempt {attempt}/{self.max_retries}), waiting {delay:.0f}s...")
                time.sleep(delay)
            except (APIConnectionError, APITimeoutError, InternalServerError) as e:
                delay = min(2 ** attempt, 30)
                print(f"[Groq] Transient error '{e}' (attempt {attempt}/{self.max_retries}), waiting {delay}s...")
                time.sleep(delay)
            except Exception as e:
                # Non-retryable (bad key, invalid model name, etc.): fail fast with a clear message
                print(f"[Groq] Non-retryable error: {e}")
                break

        print("[Groq] All attempts failed; returning empty response.")
        return json.dumps(self._generate_empty_response(response_format))


class OllamaController(BaseLLMController):
    def __init__(self, model: str = "llama2"):
        from ollama import chat
        self.model = model

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 1.0) -> str:
        try:
            response = completion(
                model="ollama_chat/{}".format(self.model),
                messages=[
                    {"role": "system", "content": "You must respond with a JSON object."},
                    {"role": "user", "content": prompt}
                ],
                response_format=response_format,
            )
            return response.choices[0].message.content
        except Exception as e:
            empty_response = self._generate_empty_response(response_format)
            return json.dumps(empty_response)

class SGLangController(BaseLLMController):
    """LLM controller for SGLang server using HTTP requests.

    SGLang provides fast local inference with RadixAttention for efficient KV cache reuse.
    This controller communicates with a SGLang server via HTTP.

    Args:
        model: Model identifier (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        sglang_host: SGLang server host URL (default: "http://localhost")
        sglang_port: SGLang server port (default: 30000)
    """
    def __init__(self, model: str = "llama2", sglang_host: str = "http://localhost", sglang_port: int = 30000):
        self.model = model
        self.sglang_host = sglang_host
        self.sglang_port = sglang_port
        self.base_url = f"{sglang_host}:{sglang_port}"

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 1.0) -> str:
        try:
            json_schema = response_format.get("json_schema", {}).get("schema", {})
            json_schema_str = json.dumps(json_schema)

            payload = {
                "text": prompt,
                "sampling_params": {
                    "temperature": temperature,
                    "max_new_tokens": 1000,
                    "json_schema": json_schema_str
                }
            }

            response = requests.post(
                f"{self.base_url}/generate",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("text", "")
                return generated_text
            else:
                print(f"SGLang server returned status {response.status_code}: {response.text}")
                raise Exception(f"SGLang server error: {response.status_code}")

        except Exception as e:
            print(f"SGLang completion error: {e}")
            empty_response = self._generate_empty_response(response_format)
            return json.dumps(empty_response)

class OpenRouterController(BaseLLMController):
    """LLM controller for OpenRouter API using litellm.

    OpenRouter provides access to multiple LLM providers through a unified API.
    This controller uses litellm to interface with OpenRouter, supporting any model
    available on the OpenRouter platform.

    Args:
        model: Model identifier (e.g., "openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet").
               The "openrouter/" prefix is automatically added if not present.
        api_key: OpenRouter API key. If None, reads from OPENROUTER_API_KEY env variable.

    Raises:
        ValueError: If API key is not provided and not found in environment.

    Examples:
        >>> controller = OpenRouterController("openai/gpt-4o-mini", api_key="your-key")
        >>> controller = OpenRouterController("google/gemini-2.0-flash-001:free")
    """

    def __init__(self, model: str = "openai/gpt-4o-mini", api_key: Optional[str] = None):
        # For litellm, prepend "openrouter/" if not already present
        if not model.startswith("openrouter/"):
            self.model = f"openrouter/{model}"
        else:
            self.model = model

        if api_key is None:
            api_key = os.getenv('OPENROUTER_API_KEY')
        if api_key is None:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")

        # Set the environment variable for litellm to use
        os.environ['OPENROUTER_API_KEY'] = api_key
        self.api_key = api_key

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 1.0) -> str:
        """Get completion from OpenRouter API.

        Args:
            prompt: The prompt to send to the LLM.
            response_format: JSON schema specifying the expected response format.
            temperature: Sampling temperature (0.0 to 1.0).

        Returns:
            JSON string containing the LLM response.
        """
        try:
            response = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You must respond with a JSON object."},
                    {"role": "user", "content": prompt}
                ],
                response_format=response_format,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            # Silently fall back to empty response on error
            empty_response = self._generate_empty_response(response_format)
            return json.dumps(empty_response)

class LLMController:
    """LLM-based controller for memory metadata generation.

    Supports multiple backends: OpenAI, Ollama, SGLang, OpenRouter, and Groq.
    """
    def __init__(self,
                 backend: Literal["openai", "ollama", "sglang", "openrouter", "groq"] = "openai",
                 model: str = "gpt-4",
                 api_key: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000):
        if backend == "openai":
            self.llm = OpenAIController(model, api_key)
        elif backend == "ollama":
            self.llm = OllamaController(model)
        elif backend == "sglang":
            self.llm = SGLangController(model, sglang_host, sglang_port)
        elif backend == "openrouter":
            self.llm = OpenRouterController(model, api_key)
        elif backend == "groq":
            self.llm = GroqController(model, api_key)
        else:
            raise ValueError("Backend must be one of: 'openai', 'ollama', 'sglang', 'openrouter', 'groq'")

    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 1.0) -> str:
        return self.llm.get_completion(prompt, response_format, temperature)