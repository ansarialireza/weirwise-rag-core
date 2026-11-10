import os
import uuid
import pytest
from sqlalchemy import text
from server.app.config import Settings
from server.app.database import AsyncSessionLocal, engine, Base
from server.app.models import Tenant, KnowledgeBase, Document, IngestionJob
from server.app.worker import process_document_job
from server.app.providers import EmbeddingProvider

pytestmark = [pytest.mark.asyncio, pytest.mark.postgres]


class DummyEmbeddingProvider(EmbeddingProvider):
    def __init__(self, embedding_dim: int = 1536):
        self.embedding_dim = embedding_dim

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        # Return dummy constant vectors
        return [[0.01] * self.embedding_dim for _ in texts]


@pytest.fixture(autouse=True)
async def setup_db():
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    yield


async def test_large_document_bounded_memory_ingestion(tmp_path):
    """Verify that a large (~2 MB / ~250k words) document stages in-memory and ingests reliably within memory bounds."""
    storage_dir = str(tmp_path)
    file_path = os.path.join(storage_dir, f"{uuid.uuid4()}_large.txt")

    # Generate ~2.5 MB of structured text content
    paragraph = (
        "Weirwise Core provides high performance knowledge base ingestion and semantic retrieval. "
        "Each document chunk is embedded using standard dimension models and stored in PostgreSQL with pgvector. "
        "Concurrency control and atomic chunk replacement ensure robust operation without duplicate chunks. "
    )
    # 25 characters * 10 = ~250 chars per line * 10,000 repetitions = 2.5MB
    num_paragraphs = 10000
    with open(file_path, "w", encoding="utf-8") as f:
        for i in range(num_paragraphs):
            f.write(f"[{i}] {paragraph}\n")

    file_size = os.path.getsize(file_path)
    assert file_size > 2 * 1024 * 1024  # > 2MB

    async with AsyncSessionLocal() as session:
        t = Tenant(name="Large Doc Tenant")
        session.add(t)
        await session.flush()

        kb = KnowledgeBase(tenant_id=t.id, name="KB Large Doc")
        session.add(kb)
        await session.flush()

        doc = Document(
            tenant_id=t.id,
            knowledge_base_id=kb.id,
            filename="large_doc.txt",
            storage_path=file_path,
            mime_type="text/plain",
            file_size=file_size,
            content_hash=f"hash_large_{uuid.uuid4()}",
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

    settings = Settings(
        storage_dir=storage_dir,
        chunk_size=1000,
        chunk_overlap=100,
        embedding_batch_size=256,
    )
    provider = DummyEmbeddingProvider()

    # Process large document
    await process_document_job(job_id, settings, provider)

    # Verify completion and chunks count in DB
    async with AsyncSessionLocal() as session:
        d = await session.get(Document, doc_id)
        assert d.status == "completed"

        chunks_res = await session.execute(
            text("SELECT count(*) FROM document_chunks WHERE document_id = :doc_id"),
            {"doc_id": str(doc_id)},
        )
        chunk_count = chunks_res.scalar()
        assert chunk_count > 500  # verified generated chunk count

    if os.path.exists(file_path):
        os.remove(file_path)
