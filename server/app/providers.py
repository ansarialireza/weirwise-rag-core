from abc import ABC, abstractmethod
from typing import List
import asyncio
import hashlib
import time
from collections import OrderedDict

import httpx
from server.app.config import Settings


class ProviderUnavailableError(RuntimeError):
    """Transient provider failure (rate limit, 5xx, or transport error) after retries."""


_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
_EMBEDDING_RETRY_ATTEMPTS = 6
_EMBEDDING_RETRY_DELAYS = (2.0, 5.0, 10.0, 20.0, 30.0)
_CHAT_RETRY_ATTEMPTS = 3
_CHAT_RETRY_DELAYS = (1.0, 3.0)

# One keep-alive HTTP client per event loop. A fresh TLS handshake per provider
# call cost ~600ms against Gemini embeddings in the measured baseline.
_clients: dict[int, httpx.AsyncClient] = {}


def _pooled_client(max_connections: int, timeout: float) -> httpx.AsyncClient:
    loop_id = id(asyncio.get_running_loop())
    client = _clients.get(loop_id)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections // 2 or 1,
                keepalive_expiry=120.0,
            ),
        )
        _clients[loop_id] = client
    return client


async def close_provider_clients() -> None:
    """Close pooled clients on application shutdown."""
    clients = list(_clients.values())
    _clients.clear()
    for client in clients:
        try:
            await client.aclose()
        except Exception:
            pass


class _EmbeddingCache:
    """Bounded TTL LRU for query embeddings. Text -> vector is deterministic per model."""

    def __init__(self, max_size: int, ttl_seconds: int):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._items: OrderedDict[str, tuple[float, List[float]]] = OrderedDict()

    @staticmethod
    def _key(base_url: str, model: str, dimensions: int, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{base_url}|{model}|{dimensions}|{digest}"

    def get(self, key: str) -> List[float] | None:
        entry = self._items.get(key)
        if not entry:
            return None
        expires_at, vector = entry
        if expires_at < time.monotonic():
            self._items.pop(key, None)
            return None
        self._items.move_to_end(key)
        return vector

    def put(self, key: str, vector: List[float]) -> None:
        if self.max_size <= 0 or self.ttl <= 0:
            return
        self._items[key] = (time.monotonic() + self.ttl, vector)
        self._items.move_to_end(key)
        while len(self._items) > self.max_size:
            self._items.popitem(last=False)


_shared_embedding_cache: _EmbeddingCache | None = None


def get_shared_embedding_cache(max_size: int, ttl_seconds: int) -> _EmbeddingCache:
    """Singleton query embedding cache so repeated searches reuse vectors across requests."""
    global _shared_embedding_cache
    if _shared_embedding_cache is None:
        _shared_embedding_cache = _EmbeddingCache(max_size, ttl_seconds)
    return _shared_embedding_cache


def clear_shared_embedding_cache() -> None:
    """Clear shared query embedding cache (useful for testing or cache invalidation)."""
    global _shared_embedding_cache
    if _shared_embedding_cache is not None:
        _shared_embedding_cache._items.clear()


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
        self.max_connections = settings.embedding_pool_connections
        self._cache = get_shared_embedding_cache(
            settings.embedding_cache_size, settings.embedding_cache_ttl_seconds
        )

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not self.api_key:
            # Deterministic pseudo-embedding generator for test/dev mode when no key is set
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

        cache_keys = [
            self._cache._key(self.base_url, self.model, self.dimensions, text) for text in texts
        ]
        results: dict[int, List[float]] = {}
        missing: list[tuple[int, str]] = []
        for index, key in enumerate(cache_keys):
            cached = self._cache.get(key)
            if cached is not None:
                results[index] = cached
            else:
                missing.append((index, texts[index]))

        if missing:
            vectors = await self._request_embeddings([text for _, text in missing])
            for (index, _), vector in zip(missing, vectors):
                results[index] = vector
                self._cache.put(cache_keys[index], vector)

        return [results[index] for index in range(len(texts))]

    async def _request_embeddings(self, texts: List[str]) -> List[List[float]]:
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

        client = _pooled_client(self.max_connections, self.timeout)
        last_error = "unknown error"
        for attempt in range(_EMBEDDING_RETRY_ATTEMPTS):
            retry_after = None
            try:
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

        client = _pooled_client(10, self.timeout)
        last_error = "unknown error"
        for attempt in range(_CHAT_RETRY_ATTEMPTS):
            retry_after = None
            try:
                resp = await client.post(url, json=payload, headers=headers)
            except httpx.TransportError as exc:
                last_error = f"transport error: {exc!s}"
            else:
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
                if resp.status_code not in _TRANSIENT_STATUS_CODES:
                    raise RuntimeError(
                        f"Chat provider error (status {resp.status_code}): {resp.text}"
                    )
                last_error = f"status {resp.status_code}: {resp.text[:300]}"
                retry_after = resp.headers.get("retry-after")

            if attempt < _CHAT_RETRY_ATTEMPTS - 1:
                delay = _CHAT_RETRY_DELAYS[min(attempt, len(_CHAT_RETRY_DELAYS) - 1)]
                if retry_after:
                    try:
                        delay = min(max(float(retry_after), delay), 30.0)
                    except ValueError:
                        pass
                await asyncio.sleep(delay)

        raise ProviderUnavailableError(
            f"Chat provider unavailable after {_CHAT_RETRY_ATTEMPTS} attempts: {last_error}"
        )
