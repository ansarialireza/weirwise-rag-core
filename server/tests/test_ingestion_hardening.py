import asyncio
import io
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.datastructures import UploadFile

from server.app.api import _spool_upload
from server.app.auth import generate_api_key
from server.app.config import Settings, get_settings
from server.app.database import AsyncSessionLocal
from server.app.ingestion import extract_text_from_file
from server.app.main import app
from server.app.models import APIKey, Tenant


def test_extract_docx_includes_paragraphs_and_table_cells(tmp_path):
    import docx

    document = docx.Document()
    document.add_paragraph("Return policy for damaged goods.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Window"
    table.cell(0, 1).text = "30 days"
    table.cell(1, 0).text = "Shipping"
    table.cell(1, 1).text = "Free"
    path = tmp_path / "policy.docx"
    document.save(path)

    text = extract_text_from_file(
        str(path), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

    assert "Return policy for damaged goods." in text
    assert "Window | 30 days" in text
    assert "Shipping | Free" in text


def test_extract_rejects_pdf_without_signature(tmp_path):
    path = tmp_path / "fake.pdf"
    path.write_bytes(b"this is not a pdf")

    with pytest.raises(ValueError, match="PDF"):
        extract_text_from_file(str(path), "application/pdf")


def test_extract_rejects_undecodable_text(tmp_path):
    path = tmp_path / "binary.txt"
    path.write_bytes(b"\xff\xfe\x00bad bytes")

    with pytest.raises(UnicodeDecodeError):
        extract_text_from_file(str(path), "text/plain")


async def _tenant_headers():
    raw_key, key_hash = generate_api_key(prefix="ww_ingest_test_")
    async with AsyncSessionLocal() as session:
        tenant = Tenant(name="Ingestion Hardening Tenant")
        session.add(tenant)
        await session.flush()
        session.add(APIKey(tenant_id=tenant.id, key_hash=key_hash, name="ingest-key"))
        await session.commit()
        return {"X-API-Key": raw_key}, tenant.id


async def _create_kb(client, headers):
    res = await client.post(
        "/api/v1/knowledge-bases", json={"name": "Ingestion KB"}, headers=headers
    )
    assert res.status_code == 201
    return res.json()["id"]


@pytest.mark.asyncio
@pytest.mark.postgres
async def test_upload_rejects_unsafe_payloads():
    headers, _ = await _tenant_headers()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        kb_id = await _create_kb(client, headers)

        async def upload(name: str, content: bytes):
            return await client.post(
                "/api/v1/documents/upload",
                data={"knowledge_base_id": kb_id},
                files={"file": (name, content)},
                headers=headers,
            )

        assert (await upload("notes.bin", b"whatever")).status_code == 415
        assert (await upload("empty.txt", b"")).status_code == 400
        assert (await upload("binary.txt", b"text\x00tail")).status_code == 400
        assert (await upload("invalid.pdf", b"not-a-pdf")).status_code == 400
        assert (await upload("invalid.docx", b"not-a-zip")).status_code == 400


@pytest.mark.asyncio
async def test_spool_upload_enforces_size_cap_and_cleans_partial(tmp_path):
    settings = Settings(
        storage_dir=str(tmp_path),
        max_upload_size_bytes=16,
        upload_read_chunk_bytes=8,
    )
    upload = UploadFile(filename="big.txt", file=io.BytesIO(b"x" * 64))

    with pytest.raises(Exception) as excinfo:
        await _spool_upload(upload, settings)

    assert getattr(excinfo.value, "status_code", None) == 413
    assert list(Path(tmp_path).glob(".upload-*.part")) == []


@pytest.mark.asyncio
@pytest.mark.postgres
async def test_inline_ingestion_leaves_no_embedding_staging_files(tmp_path, monkeypatch):
    monkeypatch.setenv("WEIRWISE_STORAGE_DIR", str(tmp_path))
    get_settings.cache_clear()

    headers, _ = await _tenant_headers()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        kb_id = await _create_kb(client, headers)
        res = await client.post(
            "/api/v1/documents/upload",
            data={"knowledge_base_id": kb_id},
            files={
                "file": (
                    "staging.txt",
                    b"Refunds are processed within five business days.\n\n"
                    b"Shipping inside the EU takes two to three days.",
                )
            },
            headers=headers,
        )
        assert res.status_code == 202
        doc_id = res.json()["document"]["id"]

        status = None
        for _ in range(40):
            detail = await client.get(f"/api/v1/documents/{doc_id}", headers=headers)
            assert detail.status_code == 200
            status = detail.json()["status"]
            if status not in ("pending", "processing"):
                break
            await asyncio.sleep(0.25)

    assert status == "completed"
    assert list(Path(tmp_path).glob(".embeddings-*.npy")) == []
    get_settings.cache_clear()
