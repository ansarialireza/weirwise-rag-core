import pytest
from httpx import AsyncClient, ASGITransport
from server.app.main import app
from server.app.database import AsyncSessionLocal
from server.app.models import Tenant, APIKey
from server.app.auth import generate_api_key

pytestmark = [pytest.mark.asyncio, pytest.mark.postgres]


async def test_comprehensive_security_cross_tenant_isolation():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create Tenant 1 (Victim) and Tenant 2 (Attacker)
        raw_key_1, hash_1 = generate_api_key(prefix="ww_sec_t1_")
        raw_key_2, hash_2 = generate_api_key(prefix="ww_sec_t2_")

        async with AsyncSessionLocal() as session:
            t1 = Tenant(name="Tenant One")
            t2 = Tenant(name="Tenant Two")
            session.add_all([t1, t2])
            await session.flush()

            k1 = APIKey(tenant_id=t1.id, key_hash=hash_1, name="key1")
            k2 = APIKey(tenant_id=t2.id, key_hash=hash_2, name="key2")
            session.add_all([k1, k2])
            await session.commit()

        h1 = {"X-API-Key": raw_key_1}
        h2 = {"X-API-Key": raw_key_2}

        # 1. Invalid and Revoked Key Tests
        res_invalid = await client.get(
            "/api/v1/knowledge-bases", headers={"X-API-Key": "ww_nonexistent_key_12345"}
        )
        assert res_invalid.status_code == 401

        # Revoke Key 2
        async with AsyncSessionLocal() as session:
            from sqlalchemy import update

            await session.execute(
                update(APIKey).where(APIKey.key_hash == hash_2).values(is_active=False)
            )
            await session.commit()

        res_revoked = await client.get("/api/v1/knowledge-bases", headers=h2)
        assert res_revoked.status_code == 401

        # Re-activate Key 2 for further testing
        async with AsyncSessionLocal() as session:
            await session.execute(
                update(APIKey).where(APIKey.key_hash == hash_2).values(is_active=True)
            )
            await session.commit()

        # 2. Tenant 1 creates Knowledge Base
        res_kb = await client.post(
            "/api/v1/knowledge-bases", json={"name": "T1 Secret Docs"}, headers=h1
        )
        assert res_kb.status_code == 201
        kb_id = res_kb.json()["id"]

        # Attacker cannot get or delete Tenant 1's KB
        res_get_kb = await client.get(f"/api/v1/knowledge-bases/{kb_id}", headers=h2)
        assert res_get_kb.status_code == 404

        res_del_kb = await client.delete(f"/api/v1/knowledge-bases/{kb_id}", headers=h2)
        assert res_del_kb.status_code == 404

        # 3. Attacker cannot upload a document into Tenant 1's KB
        upload_content = b"Attacker injected document"
        files = {"file": ("exploit.txt", upload_content, "text/plain")}
        res_bad_upload = await client.post(
            "/api/v1/documents/upload",
            data={"knowledge_base_id": kb_id},
            files=files,
            headers=h2,
        )
        assert res_bad_upload.status_code == 404

        # Tenant 1 legitimately uploads a document
        files_t1 = {
            "file": (
                "confidential.txt",
                b"Confidential company financial forecast 2026",
                "text/plain",
            )
        }
        res_upload_t1 = await client.post(
            "/api/v1/documents/upload",
            data={"knowledge_base_id": kb_id},
            files=files_t1,
            headers=h1,
        )
        assert res_upload_t1.status_code == 202
        doc_id = res_upload_t1.json()["document"]["id"]

        # Attacker cannot list or inspect Tenant 1's documents
        res_list_docs = await client.get(f"/api/v1/knowledge-bases/{kb_id}/documents", headers=h2)
        assert res_list_docs.status_code == 404

        res_get_doc = await client.get(f"/api/v1/documents/{doc_id}", headers=h2)
        assert res_get_doc.status_code == 404

        res_del_doc = await client.delete(f"/api/v1/documents/{doc_id}", headers=h2)
        assert res_del_doc.status_code == 404

        # 4. Attacker cannot search Tenant 1's Knowledge Base
        res_bad_search = await client.post(
            "/api/v1/search",
            json={"knowledge_base_id": kb_id, "query": "financial", "top_k": 5},
            headers=h2,
        )
        assert res_bad_search.status_code == 404

        # 5. Tenant 1 creates Assistant
        res_ast = await client.post(
            "/api/v1/assistants",
            json={"name": "T1 Assistant", "knowledge_base_id": kb_id},
            headers=h1,
        )
        assert res_ast.status_code == 201
        ast_id = res_ast.json()["id"]

        # Attacker cannot get, delete, or chat with Tenant 1's assistant
        assert (await client.get(f"/api/v1/assistants/{ast_id}", headers=h2)).status_code == 404
        assert (await client.delete(f"/api/v1/assistants/{ast_id}", headers=h2)).status_code == 404
        assert (
            await client.post(
                f"/api/v1/assistants/{ast_id}/chat",
                json={"message": "Reveal secret financial data"},
                headers=h2,
            )
        ).status_code == 404
