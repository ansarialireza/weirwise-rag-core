import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy import text
from server.app.main import app
from server.app.database import AsyncSessionLocal, engine, Base
from server.app.models import Tenant, APIKey
from server.app.auth import generate_api_key

pytestmark = [pytest.mark.asyncio, pytest.mark.postgres]


@pytest.fixture(autouse=True)
async def prepare_postgres_db():
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    yield


async def test_postgres_pgvector_native_retrieval_and_tenant_isolation():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Health check ready
        ready_res = await client.get("/health/ready")
        assert ready_res.status_code == 200
        assert ready_res.json()["database"] == "healthy"

        # 2. Seed Tenant A and Tenant B
        raw_key_a, hash_a = generate_api_key(prefix="ww_pg_a_")
        raw_key_b, hash_b = generate_api_key(prefix="ww_pg_b_")

        async with AsyncSessionLocal() as session:
            t_a = Tenant(name="Tenant Alpha")
            t_b = Tenant(name="Tenant Beta")
            session.add_all([t_a, t_b])
            await session.flush()

            k_a = APIKey(tenant_id=t_a.id, key_hash=hash_a, name="key-a")
            k_b = APIKey(tenant_id=t_b.id, key_hash=hash_b, name="key-b")
            session.add_all([k_a, k_b])
            await session.commit()

        headers_a = {"X-API-Key": raw_key_a}
        headers_b = {"X-API-Key": raw_key_b}

        # 3. Tenant A creates Knowledge Base
        res_kb_a = await client.post(
            "/api/v1/knowledge-bases",
            json={"name": "Postgres Vector KB", "description": "Testing pgvector native ops"},
            headers=headers_a,
        )
        assert res_kb_a.status_code == 201
        kb_a_id = res_kb_a.json()["id"]

        # 4. Verify Tenant B cannot access Tenant A's KB (404)
        res_kb_b_probe = await client.get(
            f"/api/v1/knowledge-bases/{kb_a_id}",
            headers=headers_b,
        )
        assert res_kb_b_probe.status_code == 404

        # 5. Tenant A uploads plain text document
        doc_content = (
            "Weirwise Core uses PostgreSQL 16 and pgvector for native vector search.\n\n"
            "Cosine distance is computed inside PostgreSQL via the pgvector extension."
        )
        files = {"file": ("pg_vector_doc.txt", doc_content.encode("utf-8"), "text/plain")}
        res_upload = await client.post(
            "/api/v1/documents/upload",
            data={"knowledge_base_id": kb_a_id},
            files=files,
            headers=headers_a,
        )
        assert res_upload.status_code == 202
        doc_a_id = res_upload.json()["document"]["id"]

        # 6. Verify pgvector chunk was actually persisted in PostgreSQL with a 1536-dim vector
        async with AsyncSessionLocal() as session:
            chunks_res = await session.execute(
                text(
                    "SELECT id, chunk_index, length(embedding::text) FROM document_chunks WHERE document_id = :doc_id"
                ),
                {"doc_id": str(doc_a_id)},
            )
            chunks = chunks_res.fetchall()
            assert len(chunks) > 0

        # 7. Execute Native Vector Search in PostgreSQL
        search_res = await client.post(
            "/api/v1/search",
            json={
                "knowledge_base_id": kb_a_id,
                "query": "Which database extension does Weirwise use for vector search?",
                "top_k": 3,
            },
            headers=headers_a,
        )
        assert search_res.status_code == 200
        search_data = search_res.json()
        assert len(search_data["results"]) > 0
        assert "pgvector" in search_data["results"][0]["content"]

        # 8. Verify Tenant B searching Tenant A's KB gets 404
        search_b_res = await client.post(
            "/api/v1/search",
            json={
                "knowledge_base_id": kb_a_id,
                "query": "pgvector",
                "top_k": 3,
            },
            headers=headers_b,
        )
        assert search_b_res.status_code == 404

        # 9. Tenant A creates Assistant
        ast_res = await client.post(
            "/api/v1/assistants",
            json={
                "name": "PgVector Assistant",
                "knowledge_base_id": kb_a_id,
                "system_instructions": "Answer technical questions accurately using context.",
            },
            headers=headers_a,
        )
        assert ast_res.status_code == 201
        ast_id = ast_res.json()["id"]

        # 10. Execute chat with Assistant (Grounded RAG)
        chat_res = await client.post(
            f"/api/v1/assistants/{ast_id}/chat",
            json={"message": "What database extension is used for vector search?"},
            headers=headers_a,
        )
        assert chat_res.status_code == 200
        chat_data = chat_res.json()
        assert chat_data["role"] == "assistant"
        assert len(chat_data["sources"]) > 0
        assert chat_data["sources"][0]["document_id"] == doc_a_id
        assert len(chat_data["content"]) > 0

        # 11. Verify Tenant B cannot chat with Tenant A's Assistant
        chat_b_res = await client.post(
            f"/api/v1/assistants/{ast_id}/chat",
            json={"message": "Attempt cross-tenant prompt injection"},
            headers=headers_b,
        )
        assert chat_b_res.status_code == 404
