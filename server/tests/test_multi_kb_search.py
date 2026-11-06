import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from server.app.auth import generate_api_key
from server.app.database import AsyncSessionLocal
from server.app.main import app
from server.app.models import APIKey, Tenant


async def _tenant():
    raw_key, key_hash = generate_api_key(prefix="ww_multikb_test_")
    async with AsyncSessionLocal() as session:
        tenant = Tenant(name="Multi KB Tenant")
        session.add(tenant)
        await session.flush()
        session.add(APIKey(tenant_id=tenant.id, key_hash=key_hash, name="multikb-key"))
        await session.commit()
        return {"X-API-Key": raw_key}, str(tenant.id)


async def _kb(client, headers, name):
    res = await client.post("/api/v1/knowledge-bases", json={"name": name}, headers=headers)
    assert res.status_code == 201
    return res.json()["id"]


async def _upload_and_wait(client, headers, kb_id, filename, content):
    res = await client.post(
        "/api/v1/documents/upload",
        data={"knowledge_base_id": kb_id},
        files={"file": (filename, content.encode("utf-8"))},
        headers=headers,
    )
    assert res.status_code == 202
    doc_id = res.json()["document"]["id"]
    for _ in range(40):
        detail = await client.get(f"/api/v1/documents/{doc_id}", headers=headers)
        assert detail.status_code == 200
        if detail.json()["status"] == "completed":
            return doc_id
        if detail.json()["status"] == "failed":
            pytest.fail(f"ingestion failed: {detail.json().get('error_message')}")
        await asyncio.sleep(0.25)
    pytest.fail("ingestion did not finish in time")


@pytest.mark.asyncio
async def test_multi_kb_search_spans_selected_collections_only():
    headers, _ = await _tenant()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        policies_kb = await _kb(client, headers, "Policies")
        shipping_kb = await _kb(client, headers, "Shipping")
        unrelated_kb = await _kb(client, headers, "Unrelated")

        policy_doc = await _upload_and_wait(
            client, headers, policies_kb, "policy.md", "Refunds take five business days."
        )
        shipping_doc = await _upload_and_wait(
            client, headers, shipping_kb, "shipping.md", "Orders ship within two days."
        )
        unrelated_doc = await _upload_and_wait(
            client, headers, unrelated_kb, "other.md", "The office closes for the winter holiday."
        )

        res = await client.post(
            "/api/v1/search",
            json={
                "knowledge_base_ids": [policies_kb, shipping_kb],
                "query": "How long does delivery or a refund take?",
                "top_k": 10,
            },
            headers=headers,
        )
        assert res.status_code == 200
        doc_ids = {item["document_id"] for item in res.json()["results"]}
        assert {policy_doc, shipping_doc} <= doc_ids
        assert unrelated_doc not in doc_ids

        single = await client.post(
            "/api/v1/search",
            json={"knowledge_base_id": policies_kb, "query": "refund", "top_k": 5},
            headers=headers,
        )
        assert single.status_code == 200
        assert {item["document_id"] for item in single.json()["results"]} == {policy_doc}


@pytest.mark.asyncio
async def test_multi_kb_search_rejects_foreign_and_ambiguous_targets():
    headers, _ = await _tenant()
    other_headers, _ = await _tenant()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        own_kb = await _kb(client, headers, "Own")
        foreign_kb = await _kb(client, other_headers, "Foreign")

        foreign = await client.post(
            "/api/v1/search",
            json={"knowledge_base_ids": [own_kb, foreign_kb], "query": "anything"},
            headers=headers,
        )
        assert foreign.status_code == 404

        ambiguous = await client.post(
            "/api/v1/search",
            json={"knowledge_base_id": own_kb, "knowledge_base_ids": [own_kb], "query": "x"},
            headers=headers,
        )
        assert ambiguous.status_code == 422

        missing = await client.post("/api/v1/search", json={"query": "x"}, headers=headers)
        assert missing.status_code == 422
