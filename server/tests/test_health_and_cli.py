import pytest
from httpx import AsyncClient, ASGITransport
from server.app.main import app
from server.app.config import get_settings
from server.app.cli import check_db_cmd


@pytest.mark.asyncio
async def test_health_live_endpoint_version():
    settings = get_settings()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        assert data["version"] == settings.app_version
        assert data["version"] == "1.0.0"


@pytest.mark.asyncio
@pytest.mark.postgres
async def test_health_ready_endpoint_version():
    settings = get_settings()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["version"] == settings.app_version
        assert data["version"] == "1.0.0"


@pytest.mark.asyncio
@pytest.mark.postgres
async def test_cli_check_db_command():
    exit_code = await check_db_cmd()
    assert exit_code == 0


@pytest.mark.asyncio
async def test_cli_check_db_command_failure(monkeypatch):
    class FailingEngine:
        def connect(self):
            raise ConnectionRefusedError("Simulated connection failure")

    monkeypatch.setattr("server.app.cli.engine", FailingEngine())
    exit_code = await check_db_cmd()
    assert exit_code == 1
