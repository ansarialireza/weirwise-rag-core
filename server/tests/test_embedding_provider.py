import httpx
import pytest

from server.app.config import Settings
from server.app.providers import OpenAIEmbeddingProvider


class _FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _FakeAsyncClient:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json=None, headers=None):
        self.requests.append({"url": url, "json": json, "headers": headers})
        return _FakeResponse(self.payload)


def _provider() -> OpenAIEmbeddingProvider:
    return OpenAIEmbeddingProvider(
        Settings(
            embedding_api_key="test-key",
            embedding_base_url="https://embedding.example/v1",
            embedding_model="gemini-embedding-001",
            embedding_dimensions=3,
        )
    )


@pytest.mark.asyncio
async def test_embedding_provider_tolerates_missing_index(monkeypatch):
    payload = {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": [1.0, 0.0, 0.0]},
            {"object": "embedding", "index": 1, "embedding": [0.0, 1.0, 0.0]},
            {"object": "embedding", "index": 2, "embedding": [0.0, 0.0, 1.0]},
        ],
    }
    fake = _FakeAsyncClient(payload)
    monkeypatch.setattr(httpx, "AsyncClient", fake)

    vectors = await _provider().get_embeddings(["a", "b", "c"])

    assert vectors == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert fake.requests[0]["json"]["input"] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_embedding_provider_rejects_vector_count_mismatch(monkeypatch):
    payload = {
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": [1.0, 0.0, 0.0]}],
    }
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient(payload))

    with pytest.raises(RuntimeError, match="returned 1 vectors for 2 inputs"):
        await _provider().get_embeddings(["a", "b"])
