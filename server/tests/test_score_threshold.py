import pytest
import uuid
from unittest.mock import AsyncMock
from server.app.retrieval import retrieve_relevant_chunks
from server.app.models import DocumentChunk
from server.app.schemas import SearchRequest
from pydantic import ValidationError


class DummyEmbeddingProvider:
    async def get_embeddings(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


def test_search_request_score_threshold_validation():
    kb_id = uuid.uuid4()
    # Valid thresholds
    req1 = SearchRequest(knowledge_base_id=kb_id, query="test", score_threshold=0.5)
    assert req1.score_threshold == 0.5

    req2 = SearchRequest(knowledge_base_id=kb_id, query="test", score_threshold=0.0)
    assert req2.score_threshold == 0.0

    req3 = SearchRequest(knowledge_base_id=kb_id, query="test", score_threshold=1.0)
    assert req3.score_threshold == 1.0

    req4 = SearchRequest(knowledge_base_id=kb_id, query="test", score_threshold=None)
    assert req4.score_threshold is None

    # Invalid thresholds < 0.0 or > 1.0
    with pytest.raises(ValidationError):
        SearchRequest(knowledge_base_id=kb_id, query="test", score_threshold=-0.1)

    with pytest.raises(ValidationError):
        SearchRequest(knowledge_base_id=kb_id, query="test", score_threshold=1.1)


@pytest.mark.asyncio
async def test_retrieval_score_threshold_filtering():
    tenant_id = uuid.uuid4()
    kb_id = uuid.uuid4()

    # Mock DB session in fallback (in-memory) mode
    chunk_high = DocumentChunk(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        document_id=uuid.uuid4(),
        content="High similarity match",
        embedding=[1.0, 0.0, 0.0],  # cosine sim = 1.0
        chunk_metadata={},
    )
    chunk_low = DocumentChunk(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        document_id=uuid.uuid4(),
        content="Low similarity match",
        embedding=[0.0, 1.0, 0.0],  # orthogonal, cosine sim = 0.0
        chunk_metadata={},
    )

    class MockResult:
        def scalars(self):
            class MockScalars:
                def all(self):
                    return [chunk_high, chunk_low]

            return MockScalars()

    mock_db = AsyncMock()
    mock_db.bind.dialect.name = "sqlite"
    mock_db.execute.return_value = MockResult()

    # 1. No threshold -> returns both
    all_res = await retrieve_relevant_chunks(
        db=mock_db,
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        query="test",
        embedding_provider=DummyEmbeddingProvider(),
        top_k=5,
        score_threshold=None,
    )
    assert len(all_res) == 2
    assert all_res[0]["similarity_score"] == 1.0
    assert all_res[1]["similarity_score"] == 0.0

    # 2. Threshold = 0.5 -> filters out chunk_low
    filtered_res = await retrieve_relevant_chunks(
        db=mock_db,
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        query="test",
        embedding_provider=DummyEmbeddingProvider(),
        top_k=5,
        score_threshold=0.5,
    )
    assert len(filtered_res) == 1
    assert filtered_res[0]["content"] == "High similarity match"
    assert filtered_res[0]["similarity_score"] == 1.0

    # 3. Very high threshold -> empty results
    empty_res = await retrieve_relevant_chunks(
        db=mock_db,
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        query="test",
        embedding_provider=DummyEmbeddingProvider(),
        top_k=5,
        score_threshold=1.0001,  # even if passed directly to function
    )
    assert len(empty_res) == 0


@pytest.mark.asyncio
@pytest.mark.postgres
async def test_native_pgvector_score_threshold_real_db():
    """Verify native PostgreSQL pgvector cosine distance filtering against live PostgreSQL."""
    from server.app.database import AsyncSessionLocal, engine, Base
    from server.app.models import Tenant, KnowledgeBase, Document
    from sqlalchemy import text

    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    # 1536-dimensional vectors
    vec_perfect = [1.0] + [0.0] * 1535  # sim = 1.0 with query [1.0, 0.0, ...]
    vec_partial = [0.8, 0.6] + [0.0] * 1534  # norm = 1.0, dot = 0.8 -> sim = 0.8
    vec_orthogonal = [0.0, 1.0] + [0.0] * 1534  # dot = 0.0 -> sim = 0.0

    async with AsyncSessionLocal() as session:
        tenant = Tenant(name="PgVector Threshold Test Tenant")
        session.add(tenant)
        await session.flush()

        kb = KnowledgeBase(tenant_id=tenant.id, name="Threshold KB")
        session.add(kb)
        await session.flush()

        doc = Document(
            tenant_id=tenant.id,
            knowledge_base_id=kb.id,
            filename="threshold_test.txt",
            storage_path="mock/path.txt",
            mime_type="text/plain",
            file_size=100,
            content_hash="mockhash",
            status="completed",
        )
        session.add(doc)
        await session.flush()

        tenant_id = tenant.id
        kb_id = kb.id
        doc_id = doc.id

        c1 = DocumentChunk(
            id=uuid.uuid4(),
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            document_id=doc_id,
            chunk_index=0,
            content="Perfect Match",
            embedding=vec_perfect,
            chunk_metadata={"tag": "perfect"},
        )
        c2 = DocumentChunk(
            id=uuid.uuid4(),
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            document_id=doc_id,
            chunk_index=1,
            content="Partial Match",
            embedding=vec_partial,
            chunk_metadata={"tag": "partial"},
        )
        c3 = DocumentChunk(
            id=uuid.uuid4(),
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            document_id=doc_id,
            chunk_index=2,
            content="Orthogonal Match",
            embedding=vec_orthogonal,
            chunk_metadata={"tag": "orthogonal"},
        )
        session.add_all([c1, c2, c3])
        await session.commit()

    class Provider1536:
        async def get_embeddings(self, texts):
            return [[1.0] + [0.0] * 1535 for _ in texts]

    async with AsyncSessionLocal() as session:
        # 1. No threshold -> returns top 3 in order
        res_all = await retrieve_relevant_chunks(
            db=session,
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            query="test",
            embedding_provider=Provider1536(),
            top_k=5,
            score_threshold=None,
        )
        assert len(res_all) == 3
        assert res_all[0]["content"] == "Perfect Match"
        assert res_all[0]["similarity_score"] == 1.0
        assert res_all[1]["content"] == "Partial Match"
        assert round(res_all[1]["similarity_score"], 2) == 0.80
        assert res_all[2]["content"] == "Orthogonal Match"
        assert round(res_all[2]["similarity_score"], 2) == 0.0

        # 2. Threshold = 0.75 -> filters out Orthogonal Match, keeps Perfect and Partial
        res_thresh = await retrieve_relevant_chunks(
            db=session,
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            query="test",
            embedding_provider=Provider1536(),
            top_k=5,
            score_threshold=0.75,
        )
        assert len(res_thresh) == 2
        assert [r["content"] for r in res_thresh] == ["Perfect Match", "Partial Match"]

        # 3. Threshold = 0.95 -> only Perfect Match
        res_high = await retrieve_relevant_chunks(
            db=session,
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            query="test",
            embedding_provider=Provider1536(),
            top_k=5,
            score_threshold=0.95,
        )
        assert len(res_high) == 1
        assert res_high[0]["content"] == "Perfect Match"


@pytest.mark.asyncio
@pytest.mark.postgres
async def test_vector_similarity_semantics_parity():
    """Verify exact parity between PostgreSQL/pgvector and SQLite fallback across identical,

    orthogonal, opposing, zero, and tied vectors.
    """
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from server.app.database import Base, AsyncSessionLocal, engine
    from server.app.models import Tenant, KnowledgeBase, Document, DocumentChunk
    from sqlalchemy import text

    # Known test vectors
    # Query: [1, 0, 0, ...]
    v_ident = [1.0] + [0.0] * 1535  # sim = 1.0
    v_ortho = [0.0, 1.0] + [0.0] * 1534  # sim = 0.0
    v_oppose = [-1.0] + [0.0] * 1535  # raw cos = -1.0 -> clamped to 0.0
    v_zero = [0.0] * 1536  # norm 0.0 -> clamped to 0.0
    v_tie1 = [0.5, 0.5] + [0.0] * 1534  # tied vector 1
    v_tie2 = [0.5, 0.5] + [0.0] * 1534  # tied vector 2

    class ParityProvider:
        async def get_embeddings(self, texts):
            return [[1.0] + [0.0] * 1535 for _ in texts]

    # 1. Test in SQLite fallback engine
    sqlite_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with sqlite_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sqlite_maker = async_sessionmaker(sqlite_engine, expire_on_commit=False)

    async with sqlite_maker() as session:
        t = Tenant(name="SQLiteParity")
        session.add(t)
        await session.flush()
        kb = KnowledgeBase(tenant_id=t.id, name="KB")
        session.add(kb)
        await session.flush()
        doc = Document(
            tenant_id=t.id,
            knowledge_base_id=kb.id,
            filename="d.txt",
            storage_path="d.txt",
            mime_type="text/plain",
            file_size=1,
            content_hash="h",
            status="completed",
        )
        session.add(doc)
        await session.flush()

        chunks = [
            DocumentChunk(
                tenant_id=t.id,
                knowledge_base_id=kb.id,
                document_id=doc.id,
                chunk_index=0,
                content="Identical",
                embedding=v_ident,
            ),
            DocumentChunk(
                tenant_id=t.id,
                knowledge_base_id=kb.id,
                document_id=doc.id,
                chunk_index=1,
                content="Orthogonal",
                embedding=v_ortho,
            ),
            DocumentChunk(
                tenant_id=t.id,
                knowledge_base_id=kb.id,
                document_id=doc.id,
                chunk_index=2,
                content="Opposing",
                embedding=v_oppose,
            ),
            DocumentChunk(
                tenant_id=t.id,
                knowledge_base_id=kb.id,
                document_id=doc.id,
                chunk_index=3,
                content="Zero",
                embedding=v_zero,
            ),
            DocumentChunk(
                tenant_id=t.id,
                knowledge_base_id=kb.id,
                document_id=doc.id,
                chunk_index=4,
                content="Tie1",
                embedding=v_tie1,
            ),
            DocumentChunk(
                tenant_id=t.id,
                knowledge_base_id=kb.id,
                document_id=doc.id,
                chunk_index=5,
                content="Tie2",
                embedding=v_tie2,
            ),
        ]
        session.add_all(chunks)
        await session.commit()

        # Test A: No threshold
        res_no_thresh = await retrieve_relevant_chunks(
            session, t.id, kb.id, "q", ParityProvider(), top_k=10
        )
        scores = {r["content"]: r["similarity_score"] for r in res_no_thresh}
        assert scores["Identical"] == 1.0
        assert scores["Orthogonal"] == 0.0
        # Opposing must be clamped to 0.0 (not -1.0)
        assert scores["Opposing"] == 0.0
        assert scores["Zero"] == 0.0

        # Deterministic tie-breaking on tied chunks Tie1 and Tie2 (chunk_index 4 vs 5)
        tie_ranks = [r["content"] for r in res_no_thresh if r["content"] in ("Tie1", "Tie2")]
        assert tie_ranks == ["Tie1", "Tie2"]

        # Test B: score_threshold = 0.0 (exact zero threshold boundary)
        # Raw similarities: Identical (1.0), Tie1/Tie2 (~0.71), Orthogonal (0.0), Opposing (-1.0).
        # raw_sim >= 0.0 includes Identical, Tie1, Tie2, and Orthogonal; excludes Opposing (-1.0).
        res_thresh_zero = await retrieve_relevant_chunks(
            session, t.id, kb.id, "q", ParityProvider(), top_k=10, score_threshold=0.0
        )
        zero_contents = [r["content"] for r in res_thresh_zero]
        assert "Identical" in zero_contents
        assert "Orthogonal" in zero_contents
        assert "Tie1" in zero_contents
        assert "Tie2" in zero_contents
        assert "Opposing" not in zero_contents

        # Test C: score_threshold = 0.5 (positive threshold)
        res_thresh_pos = await retrieve_relevant_chunks(
            session, t.id, kb.id, "q", ParityProvider(), top_k=10, score_threshold=0.5
        )
        pos_contents = [r["content"] for r in res_thresh_pos]
        assert "Identical" in pos_contents
        assert "Tie1" in pos_contents
        assert "Tie2" in pos_contents
        assert "Orthogonal" not in pos_contents
        assert "Opposing" not in pos_contents

    # 2. Test in live PostgreSQL / pgvector
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as session:
        t_pg = Tenant(name="PgParity")
        session.add(t_pg)
        await session.flush()
        kb_pg = KnowledgeBase(tenant_id=t_pg.id, name="PgKB")
        session.add(kb_pg)
        await session.flush()
        doc_pg = Document(
            tenant_id=t_pg.id,
            knowledge_base_id=kb_pg.id,
            filename="d.txt",
            storage_path="d.txt",
            mime_type="text/plain",
            file_size=1,
            content_hash="h_pg",
            status="completed",
        )
        session.add(doc_pg)
        await session.flush()

        pg_chunks = [
            DocumentChunk(
                tenant_id=t_pg.id,
                knowledge_base_id=kb_pg.id,
                document_id=doc_pg.id,
                chunk_index=0,
                content="Identical",
                embedding=v_ident,
            ),
            DocumentChunk(
                tenant_id=t_pg.id,
                knowledge_base_id=kb_pg.id,
                document_id=doc_pg.id,
                chunk_index=1,
                content="Orthogonal",
                embedding=v_ortho,
            ),
            DocumentChunk(
                tenant_id=t_pg.id,
                knowledge_base_id=kb_pg.id,
                document_id=doc_pg.id,
                chunk_index=2,
                content="Opposing",
                embedding=v_oppose,
            ),
            DocumentChunk(
                tenant_id=t_pg.id,
                knowledge_base_id=kb_pg.id,
                document_id=doc_pg.id,
                chunk_index=3,
                content="Tie1",
                embedding=v_tie1,
            ),
            DocumentChunk(
                tenant_id=t_pg.id,
                knowledge_base_id=kb_pg.id,
                document_id=doc_pg.id,
                chunk_index=4,
                content="Tie2",
                embedding=v_tie2,
            ),
        ]
        session.add_all(pg_chunks)
        await session.commit()

        # PG Query with no threshold
        pg_res = await retrieve_relevant_chunks(
            session, t_pg.id, kb_pg.id, "q", ParityProvider(), top_k=10
        )
        pg_scores = {r["content"]: r["similarity_score"] for r in pg_res}
        assert pg_scores["Identical"] == 1.0
        assert pg_scores["Orthogonal"] == 0.0
        assert pg_scores["Opposing"] == 0.0

        # PG Query with positive threshold 0.5
        pg_res_pos = await retrieve_relevant_chunks(
            session, t_pg.id, kb_pg.id, "q", ParityProvider(), top_k=10, score_threshold=0.5
        )
        pg_pos_contents = [r["content"] for r in pg_res_pos]
        assert "Identical" in pg_pos_contents
        assert "Tie1" in pg_pos_contents
        assert "Tie2" in pg_pos_contents
        assert "Orthogonal" not in pg_pos_contents
        assert "Opposing" not in pg_pos_contents

        # PG Query with exact zero threshold 0.0 (cosine_distance <= 1.0)
        # Cosine distance: Identical (0.0 <= 1.0), Orthogonal (1.0 <= 1.0), Opposing (2.0 > 1.0).
        # Opposing is excluded; Identical, Tie1, Tie2, and Orthogonal are included.
        pg_res_zero = await retrieve_relevant_chunks(
            session, t_pg.id, kb_pg.id, "q", ParityProvider(), top_k=10, score_threshold=0.0
        )
        pg_zero_contents = [r["content"] for r in pg_res_zero]
        assert "Identical" in pg_zero_contents
        assert "Orthogonal" in pg_zero_contents
        assert "Tie1" in pg_zero_contents
        assert "Tie2" in pg_zero_contents
        assert "Opposing" not in pg_zero_contents

        # Verify exact set parity between SQLite fallback and PostgreSQL at threshold 0.0
        assert set(zero_contents) - {"Zero"} == set(pg_zero_contents)
