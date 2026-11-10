import asyncio
import tempfile
from pathlib import Path

import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from server.app.main import app
from server.app.database import get_db, Base
from server.app.models import Tenant, APIKey
from server.app.auth import generate_api_key

_test_database_directory = tempfile.TemporaryDirectory(prefix="weirwise-core-e2e-")
TEST_DB_FILE = str(Path(_test_database_directory.name) / "test_e2e.db")
TEST_DATABASE_URL = f"sqlite+aiosqlite:///{TEST_DB_FILE}"

engine = create_async_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


async def override_get_db():
    async with TestingSessionLocal() as session:
        session._session_factory = TestingSessionLocal
        yield session


@pytest.fixture(scope="function", autouse=True)
def setup_test_db():
    app.dependency_overrides[get_db] = override_get_db

    async def init_models():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    try:
        asyncio.run(init_models())
        yield
    finally:
        app.dependency_overrides.pop(get_db, None)

        async def cleanup():
            await engine.dispose()

        asyncio.run(cleanup())


@pytest.mark.asyncio
async def test_full_e2e_tenant_isolation_and_rag():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Health check
        res_live = await client.get("/health/live")
        assert res_live.status_code == 200
        assert res_live.json()["status"] == "alive"

        # 2. Seed Tenant 1 and Tenant 2 directly in DB
        raw_key_1, hash_1 = generate_api_key(prefix="ww_t1_")
        raw_key_2, hash_2 = generate_api_key(prefix="ww_t2_")

        async with TestingSessionLocal() as db:
            t1 = Tenant(name="Tenant One")
            t2 = Tenant(name="Tenant Two")
            db.add_all([t1, t2])
            await db.flush()

            k1 = APIKey(tenant_id=t1.id, key_hash=hash_1, name="key1")
            k2 = APIKey(tenant_id=t2.id, key_hash=hash_2, name="key2")
            db.add_all([k1, k2])
            await db.commit()

        # 3. Test unauthenticated request fails
        res_no_auth = await client.post("/api/v1/knowledge-bases", json={"name": "Test KB"})
        assert res_no_auth.status_code == 401

        # 4. Tenant 1 creates Knowledge Base
        res_kb = await client.post(
            "/api/v1/knowledge-bases",
            json={"name": "Engineering Docs", "description": "Internal technical specs"},
            headers={"X-API-Key": raw_key_1},
        )
        assert res_kb.status_code == 201
        kb1_data = res_kb.json()
        kb1_id = kb1_data["id"]

        # 5. Tenant 2 CANNOT access Tenant 1 Knowledge Base (Tenant Isolation)
        res_t2_access = await client.get(
            f"/api/v1/knowledge-bases/{kb1_id}",
            headers={"X-API-Key": raw_key_2},
        )
        assert res_t2_access.status_code == 404

        # 6. Tenant 1 uploads document to Knowledge Base
        sample_doc_content = (
            "Weirwise Architecture Specification v0.1.0.\n\n"
            "The default vector embedding dimension is 1536.\n\n"
            "Weirwise employs PostgreSQL with pgvector for efficient retrieval."
        )
        files = {"file": ("arch_spec.txt", sample_doc_content.encode("utf-8"), "text/plain")}
        res_upload = await client.post(
            "/api/v1/documents/upload",
            data={"knowledge_base_id": kb1_id},
            files=files,
            headers={"X-API-Key": raw_key_1},
        )
        assert res_upload.status_code == 202
        doc_data = res_upload.json()["document"]
        doc_id = doc_data["id"]

        # 7. List documents for Tenant 1
        res_docs = await client.get(
            f"/api/v1/knowledge-bases/{kb1_id}/documents",
            headers={"X-API-Key": raw_key_1},
        )
        assert res_docs.status_code == 200
        assert len(res_docs.json()) == 1
        assert res_docs.json()[0]["filename"] == "arch_spec.txt"

        # 8. Search Knowledge Base
        res_search = await client.post(
            "/api/v1/search",
            json={
                "knowledge_base_id": kb1_id,
                "query": "What is the vector dimension?",
                "top_k": 2,
            },
            headers={"X-API-Key": raw_key_1},
        )
        assert res_search.status_code == 200
        search_data = res_search.json()
        assert len(search_data["results"]) > 0
        assert "1536" in search_data["results"][0]["content"]

        # 9. Tenant 2 cannot search Tenant 1 Knowledge Base
        res_t2_search = await client.post(
            "/api/v1/search",
            json={"knowledge_base_id": kb1_id, "query": "1536", "top_k": 2},
            headers={"X-API-Key": raw_key_2},
        )
        assert res_t2_search.status_code == 404

        # 10. Tenant 1 creates Assistant
        res_ast = await client.post(
            "/api/v1/assistants",
            json={
                "name": "Docs Assistant",
                "knowledge_base_id": kb1_id,
                "system_instructions": "Answer truthfully based on provided context.",
            },
            headers={"X-API-Key": raw_key_1},
        )
        assert res_ast.status_code == 201
        ast_id = res_ast.json()["id"]

        # 11. Chat with Assistant (RAG grounded execution)
        res_chat = await client.post(
            f"/api/v1/assistants/{ast_id}/chat",
            json={"message": "What is the vector embedding dimension in Weirwise?"},
            headers={"X-API-Key": raw_key_1},
        )
        assert res_chat.status_code == 200
        chat_data = res_chat.json()
        assert "role" in chat_data and chat_data["role"] == "assistant"
        assert len(chat_data["sources"]) > 0
        assert chat_data["sources"][0]["document_id"] == doc_id
        assert "content" in chat_data and len(chat_data["content"]) > 0
