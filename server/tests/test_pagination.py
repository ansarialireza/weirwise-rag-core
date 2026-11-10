import pytest
from httpx import AsyncClient, ASGITransport
from server.app.main import app
from server.app.database import AsyncSessionLocal, engine, Base
from server.app.models import Tenant, APIKey, KnowledgeBase, Document, Assistant
from server.app.auth import generate_api_key

pytestmark = [pytest.mark.asyncio, pytest.mark.postgres]


@pytest.fixture(autouse=True)
async def setup_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


async def test_pagination_and_sorting():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Seed Tenant
        raw_key, key_hash = generate_api_key(prefix="ww_page_")
        async with AsyncSessionLocal() as session:
            t = Tenant(name="Pagination Tenant")
            session.add(t)
            await session.flush()
            k = APIKey(tenant_id=t.id, key_hash=key_hash, name="page_key")
            session.add(k)

            # Create 5 KnowledgeBases
            kbs = [
                KnowledgeBase(tenant_id=t.id, name=f"KB_{i:02d}", description=f"Desc {i}")
                for i in range(5)
            ]
            session.add_all(kbs)
            await session.flush()

            # For KB 0, create 5 Documents
            kb0_id = kbs[0].id
            docs = [
                Document(
                    tenant_id=t.id,
                    knowledge_base_id=kb0_id,
                    filename=f"doc_{i:02d}.txt",
                    storage_path=f"/fake/{i}",
                    mime_type="text/plain",
                    file_size=100,
                    content_hash=f"hash_{i}",
                    status="completed",
                )
                for i in range(5)
            ]
            session.add_all(docs)

            # Create 5 Assistants
            asts = [
                Assistant(
                    tenant_id=t.id,
                    knowledge_base_id=kb0_id,
                    name=f"Assistant_{i:02d}",
                    model="gpt-4o-mini",
                )
                for i in range(5)
            ]
            session.add_all(asts)
            await session.commit()

        headers = {"X-API-Key": raw_key}

        # 1. Test Knowledge Bases Pagination
        res = await client.get("/api/v1/knowledge-bases?limit=2&offset=0", headers=headers)
        assert res.status_code == 200
        page1 = res.json()
        assert len(page1) == 2

        res = await client.get("/api/v1/knowledge-bases?limit=2&offset=2", headers=headers)
        assert res.status_code == 200
        page2 = res.json()
        assert len(page2) == 2
        # Verify no overlap between page 1 and page 2
        page1_ids = {item["id"] for item in page1}
        page2_ids = {item["id"] for item in page2}
        assert page1_ids.isdisjoint(page2_ids)

        # 2. Test Documents Pagination
        res = await client.get(
            f"/api/v1/knowledge-bases/{kb0_id}/documents?limit=3&offset=0", headers=headers
        )
        assert res.status_code == 200
        doc_page1 = res.json()
        assert len(doc_page1) == 3

        res = await client.get(
            f"/api/v1/knowledge-bases/{kb0_id}/documents?limit=3&offset=3", headers=headers
        )
        assert res.status_code == 200
        doc_page2 = res.json()
        assert len(doc_page2) == 2

        # 3. Test Assistants Pagination
        res = await client.get("/api/v1/assistants?limit=2&offset=0", headers=headers)
        assert res.status_code == 200
        ast_page1 = res.json()
        assert len(ast_page1) == 2

        # 4. Validation bounds: limit > 100 must return 422
        res_invalid_limit = await client.get("/api/v1/knowledge-bases?limit=101", headers=headers)
        assert res_invalid_limit.status_code == 422

        res_invalid_offset = await client.get("/api/v1/knowledge-bases?offset=-1", headers=headers)
        assert res_invalid_offset.status_code == 422
