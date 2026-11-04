from abc import ABC, abstractmethod
from typing import List
import asyncio
import httpx
from server.app.config import Settings


class ProviderUnavailableError(RuntimeError):
    """Transient provider failure (rate limit, 5xx, or transport error) after retries."""


_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
_EMBEDDING_RETRY_ATTEMPTS = 4
_EMBEDDING_RETRY_DELAYS = (4.0, 12.0, 25.0)


class EmbeddingProvider(ABC):
    @abstractmethod
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a list of texts."""
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, settings: Settings):
        self.base_url = settings.embedding_base_url.rstrip("/")
        self.api_key = settings.embedding_api_key
        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions
        self.timeout = settings.embedding_timeout_seconds

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not self.api_key:
            # Deterministic pseudo-embedding generator for test/dev mode when no key is set
            import hashlib
            import numpy as np

            vectors = []
            for text in texts:
                # Generate deterministic 1536-dim vector based on sha256 hash
                seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
                rng = np.random.RandomState(seed)
                vec = rng.randn(self.dimensions)
                norm = np.linalg.norm(vec)
                normalized = (vec / (norm if norm > 0 else 1.0)).tolist()
                vectors.append(normalized)
            return vectors

        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "input": texts,
            "model": self.model,
            "dimensions": self.dimensions,
        }

        last_error = "unknown error"
        for attempt in range(_EMBEDDING_RETRY_ATTEMPTS):
            retry_after = None
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(url, json=payload, headers=headers)
            except httpx.TransportError as exc:
                last_error = f"transport error: {exc!s}"
            else:
                if resp.status_code == 200:
                    data = resp.json()
                    items = data.get("data") or []
                    if len(items) != len(texts):
                        raise RuntimeError(
                            f"Embedding provider returned {len(items)} vectors for {len(texts)} inputs"
                        )
                    # Providers may scramble order via `index`; Gemini's OpenAI-compatible
                    # endpoint omits `index` entirely for the first item, so fall back to
                    # the response position when it is absent.
                    ordered = sorted(
                        enumerate(items), key=lambda pair: pair[1].get("index", pair[0])
                    )
                    return [item["embedding"] for _, item in ordered]

                if resp.status_code not in _TRANSIENT_STATUS_CODES:
                    raise RuntimeError(
                        f"Embedding provider error (status {resp.status_code}): {resp.text}"
                    )
                last_error = f"status {resp.status_code}: {resp.text[:300]}"
                retry_after = resp.headers.get("retry-after")

            if attempt < _EMBEDDING_RETRY_ATTEMPTS - 1:
                delay = _EMBEDDING_RETRY_DELAYS[min(attempt, len(_EMBEDDING_RETRY_DELAYS) - 1)]
                if retry_after:
                    try:
                        delay = min(max(float(retry_after), delay), 30.0)
                    except ValueError:
                        pass
                await asyncio.sleep(delay)

        raise ProviderUnavailableError(
            f"Embedding provider unavailable after {_EMBEDDING_RETRY_ATTEMPTS} attempts: {last_error}"
        )


class ChatProvider(ABC):
    @abstractmethod
    async def generate_response(
        self,
        messages: List[dict],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """Generate chat response for conversation messages."""
        pass


class OpenAIChatProvider(ChatProvider):
    def __init__(self, settings: Settings):
        self.base_url = settings.chat_base_url.rstrip("/")
        self.api_key = settings.chat_api_key
        self.model = settings.chat_model
        self.timeout = settings.chat_timeout_seconds

    async def generate_response(
        self,
        messages: List[dict],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        if not self.api_key:
            # In test/dev mode with no API key, synthesize grounded answer from context if present
            context_snippet = ""
            for msg in reversed(messages):
                if msg.get("role") == "user" and "Context information is below" in msg.get(
                    "content", ""
                ):
                    context_snippet = msg["content"]
                    break

            if context_snippet:
                return f"Based on the provided documentation: {context_snippet[:150]}... (Generated in local dev mode)"
            return "This is a simulated assistant response generated without an external API key."

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                raise RuntimeError(f"Chat provider error (status {resp.status_code}): {resp.text}")
            data = resp.json()
            return data["choices"][0]["message"]["content"]
