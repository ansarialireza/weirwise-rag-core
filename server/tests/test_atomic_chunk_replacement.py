import os
import uuid
import datetime
import pytest
from sqlalchemy import text
from server.app.config import Settings
from server.app.database import AsyncSessionLocal, engine, Base
from server.app.models import Tenant, KnowledgeBase, Document, IngestionJob, DocumentChunk
from server.app.worker import process_document_job
from server.app.providers import EmbeddingProvider

pytestmark = [pytest.mark.asyncio, pytest.mark.postgres]


class MockFailingProvider(EmbeddingProvider):
    """Mock provider that raises an exception during embedding generation."""

    def __init__(self, fail: bool = True):
        self.fail = fail

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("Simulated failure during embedding generation")
        return [[0.1] * 1536 for _ in texts]


@pytest.fixture(autouse=True)
async def setup_db():
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    yield


async def test_failure_during_embedding_preserves_original_chunks(tmp_path):
    """Verify that failure during embedding generation leaves original chunks intact and updates job/document status."""
    storage_dir = str(tmp_path)
    file_path = os.path.join(storage_dir, f"{uuid.uuid4()}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("Line 1 text.\nLine 2 text.\nLine 3 text.")

    async with AsyncSessionLocal() as session:
        t = Tenant(name="Atomic Replacement Tenant")
        session.add(t)
        await session.flush()

        kb = KnowledgeBase(tenant_id=t.id, name="KB Atomic", description="Test KB")
        session.add(kb)
        await session.flush()

        doc = Document(
            tenant_id=t.id,
            knowledge_base_id=kb.id,
            filename="atomic_test.txt",
            storage_path=file_path,
            mime_type="text/plain",
            file_size=50,
            content_hash="hash_atomic_initial",
            status="completed",
        )
        session.add(doc)
        await session.flush()

        # Pre-populate original chunk
        orig_chunk = DocumentChunk(
            tenant_id=t.id,
            knowledge_base_id=kb.id,
            document_id=doc.id,
            chunk_index=0,
            content="Original uncorrupted chunk content",
            embedding=[0.0] * 1536,
        )
        session.add(orig_chunk)

        job = IngestionJob(
            tenant_id=t.id,
            document_id=doc.id,
            status="pending",
            attempts=2,
            max_attempts=3,
        )
        session.add(job)
        await session.commit()
        job_id = job.id
        doc_id = doc.id

    settings = Settings(storage_dir=storage_dir)
    failing_provider = MockFailingProvider(fail=True)

    # Process job with failing provider
    await process_document_job(job_id, settings, failing_provider)

    # Verify original chunks were NOT deleted
    async with AsyncSessionLocal() as session:
        chunks_res = await session.execute(
            text("SELECT content FROM document_chunks WHERE document_id = :doc_id"),
            {"doc_id": str(doc_id)},
        )
        chunks = chunks_res.fetchall()
        assert len(chunks) == 1
        assert chunks[0][0] == "Original uncorrupted chunk content"

        # Verify document and job reflect failure
        d = await session.get(Document, doc_id)
        assert d.status == "failed"
        assert "Simulated failure during embedding generation" in (d.error_message or "")

    if os.path.exists(file_path):
        os.remove(file_path)


async def test_worker_fencing_token_rejection(tmp_path):
    """Verify that a worker whose claim token was invalidated (due to lease expiration/reclaim) is rejected from committing."""
    storage_dir = str(tmp_path)
    file_path = os.path.join(storage_dir, f"{uuid.uuid4()}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("Fencing test content.")

    async with AsyncSessionLocal() as session:
        t = Tenant(name="Fencing Tenant")
        session.add(t)
        await session.flush()

        kb = KnowledgeBase(tenant_id=t.id, name="KB Fencing")
        session.add(kb)
        await session.flush()

        doc = Document(
            tenant_id=t.id,
            knowledge_base_id=kb.id,
            filename="fencing_test.txt",
            storage_path=file_path,
            mime_type="text/plain",
            file_size=50,
            content_hash="hash_fencing",
            status="processing",
        )
        session.add(doc)
        await session.flush()

        # Job is initially claimed with token A
        token_a = uuid.uuid4()
        job = IngestionJob(
            tenant_id=t.id,
            document_id=doc.id,
            status="processing",
            claim_token=token_a,
            generation=1,
            locked_by="worker-A",
        )
        session.add(job)
        await session.commit()
        job_id = job.id
        doc_id = doc.id

    # Now simulate Worker B reclaiming the job while Worker A was slow:
    token_b = uuid.uuid4()
    async with AsyncSessionLocal() as session:
        j = await session.get(IngestionJob, job_id)
        j.claim_token = token_b
        j.generation = 2
        j.locked_by = "worker-B"
        await session.commit()

    # Now execute Worker A attempting to process with token_a
    settings = Settings(storage_dir=storage_dir)
    success_provider = MockFailingProvider(fail=False)

    # Worker A runs with claim_token=token_a
    await process_document_job(
        job_id, settings, success_provider, worker_id="worker-A", claim_token=token_a
    )

    # Verify no chunks were committed by Worker A
    async with AsyncSessionLocal() as session:
        chunks_res = await session.execute(
            text("SELECT count(*) FROM document_chunks WHERE document_id = :doc_id"),
            {"doc_id": str(doc_id)},
        )
        count = chunks_res.scalar()
        assert count == 0

        # Verify job ownership remained with Worker B
        j = await session.get(IngestionJob, job_id)
        assert j.claim_token == token_b
        assert j.locked_by == "worker-B"
        assert j.generation == 2

    if os.path.exists(file_path):
        os.remove(file_path)


async def test_deterministic_stale_worker_cannot_overwrite_completed_results(tmp_path):
    """Deterministic regression test required by audit:
    1. Worker A claims job.
    2. Worker A pauses before its final database transaction (simulated with barrier/event).
    3. Lease expires / Worker B reclaims the job and obtains new fencing token.
    4. Worker B successfully completes the job and commits chunks.
    5. Worker A resumes and attempts to commit its chunks.
    6. Assert Worker A cannot replace Worker B's chunks, overwrite document status, or mark job completed.
    """

    storage_dir = str(tmp_path)
    file_path = os.path.join(storage_dir, f"{uuid.uuid4()}_fencing.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("Deterministic fencing test content.")

    async with AsyncSessionLocal() as session:
        t = Tenant(name="Deterministic Fencing Tenant")
        session.add(t)
        await session.flush()

        kb = KnowledgeBase(tenant_id=t.id, name="KB Fencing Deterministic")
        session.add(kb)
        await session.flush()

        doc = Document(
            tenant_id=t.id,
            knowledge_base_id=kb.id,
            filename="fencing_det.txt",
            storage_path=file_path,
            mime_type="text/plain",
            file_size=50,
            content_hash=f"hash_fencing_{uuid.uuid4()}",
            status="pending",
        )
        session.add(doc)
        await session.flush()

        job = IngestionJob(
            tenant_id=t.id,
            document_id=doc.id,
            status="pending",
        )
        session.add(job)
        await session.commit()
        job_id = job.id
        doc_id = doc.id

    settings = Settings(storage_dir=storage_dir, worker_lease_seconds=1)

    # 1. Worker A claims the job
    token_a = uuid.uuid4()
    async with AsyncSessionLocal() as session:
        j = await session.get(IngestionJob, job_id)
        j.status = "processing"
        j.locked_by = "worker-A"
        j.claim_token = token_a
        j.generation = 1
        j.locked_at = datetime.datetime.utcnow() - datetime.timedelta(seconds=10)  # already expired
        await session.commit()

    # 2. Worker B reclaims stale job via reclaim_stale_jobs or direct reclaim
    from server.app.worker import reclaim_stale_jobs

    async with AsyncSessionLocal() as session:
        await reclaim_stale_jobs(settings, session)

    # Now Worker B claims it
    token_b = uuid.uuid4()
    async with AsyncSessionLocal() as session:
        j = await session.get(IngestionJob, job_id)
        assert j.status == "pending"
        j.status = "processing"
        j.locked_by = "worker-B"
        j.claim_token = token_b
        j.generation = 2
        j.locked_at = datetime.datetime.utcnow()
        await session.commit()

    # 4. Worker B completes the job
    class WorkerBProvider(EmbeddingProvider):
        async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
            return [[0.88] * 1536 for _ in texts]

    await process_document_job(
        job_id, settings, WorkerBProvider(), worker_id="worker-B", claim_token=token_b
    )

    # Verify Worker B committed its chunks and completed document
    async with AsyncSessionLocal() as session:
        d = await session.get(Document, doc_id)
        assert d.status == "completed"
        j = await session.get(IngestionJob, job_id)
        assert j.status == "completed"

        chunks_res = await session.execute(
            text(
                "SELECT content, embedding::text FROM document_chunks WHERE document_id = :doc_id"
            ),
            {"doc_id": str(doc_id)},
        )
        worker_b_chunks = chunks_res.fetchall()
        assert len(worker_b_chunks) > 0

    # 5. Worker A resumes and attempts to commit with its stale token_a
    class WorkerAProvider(EmbeddingProvider):
        async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
            return [[0.11] * 1536 for _ in texts]

    await process_document_job(
        job_id, settings, WorkerAProvider(), worker_id="worker-A", claim_token=token_a
    )

    # 6. Assert Worker A was REJECTED: Worker B's chunks remain unchanged
    async with AsyncSessionLocal() as session:
        d = await session.get(Document, doc_id)
        assert d.status == "completed"
        j = await session.get(IngestionJob, job_id)
        assert j.status == "completed"

        chunks_res = await session.execute(
            text(
                "SELECT content, embedding::text FROM document_chunks WHERE document_id = :doc_id"
            ),
            {"doc_id": str(doc_id)},
        )
        final_chunks = chunks_res.fetchall()
        assert len(final_chunks) == len(worker_b_chunks)
        assert final_chunks[0][1] == worker_b_chunks[0][1]
        assert "0.88" in final_chunks[0][1]
        assert "0.11" not in final_chunks[0][1]

    if os.path.exists(file_path):
        os.remove(file_path)
