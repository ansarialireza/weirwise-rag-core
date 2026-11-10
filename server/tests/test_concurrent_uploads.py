import asyncio
import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy import text, select
from server.app.main import app
from server.app.database import AsyncSessionLocal, engine, Base
from server.app.models import Tenant, APIKey, KnowledgeBase, Document, IngestionJob
from server.app.auth import generate_api_key

pytestmark = [pytest.mark.asyncio, pytest.mark.postgres]


@pytest.fixture(autouse=True)
async def setup_db():
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    yield


async def test_concurrent_duplicate_uploads():
    """Simulate 5 simultaneous uploads of the exact same document bytes to the same knowledge base."""
    raw_key, key_hash = generate_api_key(prefix="ww_conc_")
    async with AsyncSessionLocal() as session:
        t = Tenant(name="Concurrent Tenant")
        session.add(t)
        await session.flush()
        k = APIKey(tenant_id=t.id, key_hash=key_hash, name="conc_key")
        session.add(k)
        kb = KnowledgeBase(tenant_id=t.id, name="KB Concurrent")
        session.add(kb)
        await session.commit()
        kb_id = kb.id

    content = b"Identical file content for concurrent upload test.\nSecond line of content."

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:

        async def upload_one():
            files = {"file": ("identical.txt", content, "text/plain")}
            return await client.post(
                "/api/v1/documents/upload",
                data={"knowledge_base_id": str(kb_id)},
                files=files,
                headers={"X-API-Key": raw_key},
            )

        responses = await asyncio.gather(*[upload_one() for _ in range(5)])

        # All requests must succeed (HTTP 200 or 202)
        status_codes = [r.status_code for r in responses]
        for sc in status_codes:
            assert sc in (200, 202)

        # Inspect database: exactly 1 document and at most 1 active job must exist
        async with AsyncSessionLocal() as session:
            docs_res = await session.execute(
                select(Document).where(Document.knowledge_base_id == kb_id)
            )
            docs = docs_res.scalars().all()
            assert len(docs) == 1

            doc = docs[0]
            jobs_res = await session.execute(
                select(IngestionJob).where(IngestionJob.document_id == doc.id)
            )
            jobs = jobs_res.scalars().all()
            assert len(jobs) == 1


async def test_concurrent_retry_of_failed_document():
    """Simulate concurrent retries of a failed document, ensuring only 1 active job is enqueued."""
    raw_key, key_hash = generate_api_key(prefix="ww_retry_")
    async with AsyncSessionLocal() as session:
        t = Tenant(name="Retry Tenant")
        session.add(t)
        await session.flush()
        k = APIKey(tenant_id=t.id, key_hash=key_hash, name="retry_key")
        session.add(k)
        kb = KnowledgeBase(tenant_id=t.id, name="KB Retry")
        session.add(kb)
        await session.flush()

        doc = Document(
            tenant_id=t.id,
            knowledge_base_id=kb.id,
            filename="failed_doc.txt",
            storage_path="/fake/failed.txt",
            mime_type="text/plain",
            file_size=30,
            content_hash="hash_failed_simulated",
            status="failed",
            error_message="Previous failure",
        )
        session.add(doc)
        await session.commit()
        kb_id = kb.id

    content = b"Content matching failed document simulation."
    # We will upload with content_hash matching
    import hashlib

    content_hash = hashlib.sha256(content).hexdigest()

    async with AsyncSessionLocal() as session:
        d = (
            (await session.execute(select(Document).where(Document.knowledge_base_id == kb_id)))
            .scalars()
            .first()
        )
        d.content_hash = content_hash
        await session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:

        async def upload_retry():
            files = {"file": ("failed_doc.txt", content, "text/plain")}
            return await client.post(
                "/api/v1/documents/upload",
                data={"knowledge_base_id": str(kb_id)},
                files=files,
                headers={"X-API-Key": raw_key},
            )

        # Fire 3 concurrent uploads for this failed document
        responses = await asyncio.gather(*[upload_retry() for _ in range(3)])
        for r in responses:
            assert r.status_code in (200, 202)

        # Verify exactly 1 job was created for this document
        async with AsyncSessionLocal() as session:
            jobs_res = await session.execute(
                select(IngestionJob).where(
                    IngestionJob.document_id == d.id,
                )
            )
            all_jobs = jobs_res.scalars().all()
            assert len(all_jobs) == 1
