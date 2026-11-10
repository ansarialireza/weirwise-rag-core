"""Adversarial and extreme edge case testing for Weirwise Core.

Tests:
1. RTL, Persian/Arabic Unicode text, emoji, complex scripts chunking and retrieval
2. Extreme whitespace, empty text, single character text chunking
3. Vector search deterministic tie-breaking on identical distance vectors
4. Non-existent and malformed payload injection (boundary verification)
5. Null vs missing field handling in search request schemas
"""

import pytest
import uuid
from httpx import AsyncClient, ASGITransport
from server.app.main import app
from server.app.ingestion import chunk_text
from server.app.retrieval import retrieve_relevant_chunks
from server.app.models import Tenant, APIKey, KnowledgeBase, Document, DocumentChunk
from server.app.database import AsyncSessionLocal
from server.app.auth import generate_api_key


def test_adversarial_unicode_and_rtl_chunking():
    # Persian text with ZWNJ (Zero Width Non-Joiner) and emojis
    persian_text = (
        "هوش مصنوعی و یادگیری ماشین یکی از مهم‌ترین فناوری‌های عصر حاضر هستند. 🚀\n\n"
        "سیستم‌های بازیابی اطلاعات (RAG) به مدل‌های زبانی کمک می‌کنند تا پاسخ‌های دقیق‌تری ارائه دهند.\n\n"
        "این یک متن آزمایشی است که شامل کاراکترهای نیم‌فاصله (‌) و اعراب و شکلک‌های متنوع می‌باشد: 🔍 ✨ 📊"
    )
    chunks = chunk_text(persian_text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 3
    # Ensure Unicode characters and emojis are preserved intact
    assert any("هوش مصنوعی" in c for c in chunks)
    assert any("🚀" in c for c in chunks)
    assert any("نیم‌فاصله" in c for c in chunks)


def test_adversarial_boundary_text_chunking():
    # 1. Empty string
    assert chunk_text("", chunk_size=500, chunk_overlap=50) == []
    # 2. Only whitespace
    assert chunk_text("   \n\n   \t  \n  ", chunk_size=500, chunk_overlap=50) == []
    # 3. Single character
    assert chunk_text("A", chunk_size=500, chunk_overlap=50) == ["A"]
    # 4. Paragraph exactly equal to chunk_size
    exact_text = "x" * 100
    chunks = chunk_text(exact_text, chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert len(chunks[0]) == 100


@pytest.mark.asyncio
@pytest.mark.postgres
async def test_deterministic_tie_breaking_in_vector_retrieval():
    """Verify that identical distance vectors break ties deterministically by chunk_index and id."""
    tenant_name = f"TieBreak-{uuid.uuid4().hex[:6]}"
    raw_key, key_hash = generate_api_key(prefix="ww_tie_")

    async with AsyncSessionLocal() as session:
        t = Tenant(name=tenant_name)
        session.add(t)
        await session.flush()
        k = APIKey(tenant_id=t.id, key_hash=key_hash, name="tie-key")
        session.add(k)
        kb = KnowledgeBase(tenant_id=t.id, name="Tie KB")
        session.add(kb)
        await session.flush()

        doc = Document(
            tenant_id=t.id,
            knowledge_base_id=kb.id,
            filename="tie.txt",
            storage_path="mock/tie.txt",
            mime_type="text/plain",
            file_size=100,
            content_hash="tiehash",
            status="completed",
        )
        session.add(doc)
        await session.flush()

        # Insert 3 chunks with identical embeddings but different chunk_index
        fixed_vec = [1.0] + [0.0] * 1535
        c0 = DocumentChunk(
            id=uuid.uuid4(),
            tenant_id=t.id,
            knowledge_base_id=kb.id,
            document_id=doc.id,
            chunk_index=0,
            content="Chunk Zero",
            embedding=fixed_vec,
        )
        c1 = DocumentChunk(
            id=uuid.uuid4(),
            tenant_id=t.id,
            knowledge_base_id=kb.id,
            document_id=doc.id,
            chunk_index=1,
            content="Chunk One",
            embedding=fixed_vec,
        )
        c2 = DocumentChunk(
            id=uuid.uuid4(),
            tenant_id=t.id,
            knowledge_base_id=kb.id,
            document_id=doc.id,
            chunk_index=2,
            content="Chunk Two",
            embedding=fixed_vec,
        )
        session.add_all([c0, c1, c2])
        await session.commit()
        tenant_id = t.id
        kb_id = kb.id

    class MockProvider:
        async def get_embeddings(self, texts):
            return [[1.0] + [0.0] * 1535 for _ in texts]

    async with AsyncSessionLocal() as session:
        res = await retrieve_relevant_chunks(
            db=session,
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            query="test query",
            embedding_provider=MockProvider(),
            top_k=3,
        )
        assert len(res) == 3
        # Must be ordered deterministically by chunk_index asc when similarity scores tie (1.0)
        assert [r["content"] for r in res] == ["Chunk Zero", "Chunk One", "Chunk Two"]
        assert all(r["similarity_score"] == 1.0 for r in res)


@pytest.mark.asyncio
@pytest.mark.postgres
async def test_search_endpoint_with_null_and_boundary_score_thresholds():
    """Verify search endpoint handles score_threshold: 0.0, 1.0, null, and rejects out-of-range floats."""
    transport = ASGITransport(app=app)
    raw_key, key_hash = generate_api_key(prefix="ww_bound_")

    async with AsyncSessionLocal() as session:
        t = Tenant(name="Boundaries Tenant")
        session.add(t)
        await session.flush()
        k = APIKey(tenant_id=t.id, key_hash=key_hash, name="bound-key")
        session.add(k)
        kb = KnowledgeBase(tenant_id=t.id, name="Bound KB")
        session.add(kb)
        await session.commit()
        kb_id = str(kb.id)

    headers = {"X-API-Key": raw_key}

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Null score_threshold
        r_null = await client.post(
            "/api/v1/search",
            json={"knowledge_base_id": kb_id, "query": "hello", "score_threshold": None},
            headers=headers,
        )
        assert r_null.status_code == 200

        # 2. Threshold = 0.0
        r_zero = await client.post(
            "/api/v1/search",
            json={"knowledge_base_id": kb_id, "query": "hello", "score_threshold": 0.0},
            headers=headers,
        )
        assert r_zero.status_code == 200

        # 3. Threshold = 1.0
        r_one = await client.post(
            "/api/v1/search",
            json={"knowledge_base_id": kb_id, "query": "hello", "score_threshold": 1.0},
            headers=headers,
        )
        assert r_one.status_code == 200

        # 4. Out-of-bounds negative -> 422
        r_neg = await client.post(
            "/api/v1/search",
            json={"knowledge_base_id": kb_id, "query": "hello", "score_threshold": -0.01},
            headers=headers,
        )
        assert r_neg.status_code == 422

        # 5. Out-of-bounds > 1.0 -> 422
        r_over = await client.post(
            "/api/v1/search",
            json={"knowledge_base_id": kb_id, "query": "hello", "score_threshold": 1.01},
            headers=headers,
        )
        assert r_over.status_code == 422

        # 6. Correlation ID Propagation and Generation
        r_corr = await client.get("/health/live", headers={"X-Request-ID": "test-trace-12345"})
        assert r_corr.headers.get("X-Request-ID") == "test-trace-12345"
        assert "X-Response-Time-Ms" in r_corr.headers

        # Auto-generated Correlation ID
        r_auto_corr = await client.get("/health/live")
        assert r_auto_corr.headers.get("X-Request-ID") is not None
        assert len(r_auto_corr.headers.get("X-Request-ID")) > 10


@pytest.mark.asyncio
@pytest.mark.postgres
async def test_search_endpoint_with_metadata_filters():
    """Verify search endpoint filters results by metadata key-value pairs."""
    transport = ASGITransport(app=app)
    raw_key, key_hash = generate_api_key(prefix="ww_meta_")

    async with AsyncSessionLocal() as session:
        t = Tenant(name="Meta Filter Tenant")
        session.add(t)
        await session.flush()
        k = APIKey(tenant_id=t.id, key_hash=key_hash, name="meta-key")
        session.add(k)
        kb = KnowledgeBase(tenant_id=t.id, name="Meta KB")
        session.add(kb)
        await session.flush()

        doc = Document(
            tenant_id=t.id,
            knowledge_base_id=kb.id,
            filename="meta.txt",
            storage_path="mock/meta.txt",
            mime_type="text/plain",
            file_size=100,
            content_hash="metahash",
            status="completed",
        )
        session.add(doc)
        await session.flush()

        fixed_vec = [1.0] + [0.0] * 1535
        c1 = DocumentChunk(
            id=uuid.uuid4(),
            tenant_id=t.id,
            knowledge_base_id=kb.id,
            document_id=doc.id,
            chunk_index=0,
            content="Alpha section",
            embedding=fixed_vec,
            chunk_metadata={"category": "finance", "confidential": True},
        )
        c2 = DocumentChunk(
            id=uuid.uuid4(),
            tenant_id=t.id,
            knowledge_base_id=kb.id,
            document_id=doc.id,
            chunk_index=1,
            content="Beta section",
            embedding=fixed_vec,
            chunk_metadata={"category": "engineering", "confidential": False},
        )
        session.add_all([c1, c2])
        await session.commit()
        kb_id = str(kb.id)

    headers = {"X-API-Key": raw_key}

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Search with metadata filter category=finance
        r_finance = await client.post(
            "/api/v1/search",
            json={
                "knowledge_base_id": kb_id,
                "query": "section",
                "metadata_filters": {"category": "finance"},
            },
            headers=headers,
        )
        assert r_finance.status_code == 200
        res_data = r_finance.json()["results"]
        assert len(res_data) == 1
        assert res_data[0]["content"] == "Alpha section"

        # Search with non-matching metadata filter
        r_none = await client.post(
            "/api/v1/search",
            json={
                "knowledge_base_id": kb_id,
                "query": "section",
                "metadata_filters": {"category": "hr"},
            },
            headers=headers,
        )
        assert r_none.status_code == 200
        assert len(r_none.json()["results"]) == 0


@pytest.mark.asyncio
async def test_middleware_request_id_sanitization():
    """Verify that malformed or malicious X-Request-ID headers are sanitized to a valid UUID."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Valid custom request ID should be echoed
        resp_valid = await client.get(
            "/health/live", headers={"X-Request-ID": "valid-req-id_123.test"}
        )
        assert resp_valid.status_code == 200
        assert resp_valid.headers["X-Request-ID"] == "valid-req-id_123.test"
        assert "X-Response-Time-Ms" in resp_valid.headers

        # 2. Malicious header injection / CRLF should be replaced with UUID
        resp_crlf = await client.get(
            "/health/live", headers={"X-Request-ID": "evil\r\nInjected: True"}
        )
        assert resp_crlf.status_code == 200
        assert resp_crlf.headers["X-Request-ID"] != "evil\r\nInjected: True"
        uuid.UUID(resp_crlf.headers["X-Request-ID"])  # Must be valid UUID

        # 3. Oversized header (>128 chars) should be replaced with UUID
        oversized_id = "a" * 200
        resp_long = await client.get("/health/live", headers={"X-Request-ID": oversized_id})
        assert resp_long.status_code == 200
        assert resp_long.headers["X-Request-ID"] != oversized_id
        uuid.UUID(resp_long.headers["X-Request-ID"])
